#include "../headers/canny_scalar.h"

#include <cstring>   // memset, memcpy
#include <cmath>     // sqrtf
#include <cstdio>    // fopen, fread, fclose

// ============================================================================
// IMAGE I/O
// ============================================================================

uint8_t* load_raw_image(const char* path, int width, int height) {
    int size = width * height;
    uint8_t* buf = (uint8_t*)aligned_alloc(64, align64(size));
    if (!buf) return nullptr;

    FILE* f = fopen(path, "rb");
    if (!f) {
        free(buf);
        return nullptr;
    }

    fread(buf, 1, size, f);
    fclose(f);
    return buf;
}

bool save_raw_image(const char* path, const uint8_t* data, int width, int height) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    fwrite(data, 1, width * height, f);
    fclose(f);
    return true;
}

// ============================================================================
// GENERIC 2D CONVOLUTION (template)
// ============================================================================

template<typename PixelT, typename AccumT, typename KernelT>
void convolve2D(const PixelT* input, PixelT* output,
                int width, int height,
                const KernelT* kernel, int ksize,
                AccumT divisor) {
    int radius = ksize / 2;   // 5x5 kernel -> radius 2, 3x3 kernel -> radius 1

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            AccumT sum = 0;

            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    int py = y + ky;
                    int px = x + kx;

                    // Zero-padding: out-of-bounds pixels contribute 0
                    if (py >= 0 && py < height && px >= 0 && px < width) {
                        // kernel is stored row-major: index = (ky+radius)*ksize + (kx+radius)
                        KernelT coeff = kernel[(ky + radius) * ksize + (kx + radius)];
                        sum += (AccumT)input[py * width + px] * (AccumT)coeff;
                    }
                }
            }

            AccumT result = sum / divisor;
            if (result < 0)   result = 0;     // clamp low
            if (result > 255) result = 255;   // clamp high
            output[y * width + x] = (PixelT)result;
        }
    }
}

// Generate the body for the standard grayscale case so it can be linked elsewhere.
template void convolve2D<uint8_t, int32_t, int16_t>(
    const uint8_t*, uint8_t*, int, int, const int16_t*, int, int32_t);

// ============================================================================
// STAGE 1 - GAUSSIAN BLUR
// ============================================================================

void gaussian_blur_scalar(const uint8_t* input, uint8_t* output,
                          int width, int height) {
    // 5x5 integer Gaussian kernel, coefficients sum to 273 
    static const int16_t GAUSSIAN_KERNEL[25] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
    static const int32_t GAUSSIAN_SUM = 273;   // divisor that normalizes brightness

    convolve2D<uint8_t, int32_t, int16_t>(
        input, output, width, height, GAUSSIAN_KERNEL, 5, GAUSSIAN_SUM);
}

// ============================================================================
// STAGE 2 - SOBEL GRADIENTS
// ============================================================================

void sobel_gradients_scalar(const uint8_t* input,
                            int16_t* gx, int16_t* gy,
                            int width, int height) {
    // Sobel-X: responds to vertical edges (horizontal intensity change)
    static const int KX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    // Sobel-Y: responds to horizontal edges (vertical intensity change).
    // Positive on the top row per the project specification.
    static const int KY[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int32_t sum_x = 0;   // 32-bit accumulator avoids overflow during accumulation
            int32_t sum_y = 0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int py = y + ky;
                    int px = x + kx;

                    // Zero-padding boundary handling
                    if (py >= 0 && py < height && px >= 0 && px < width) {
                        int p = input[py * width + px];
                        sum_x += p * KX[ky + 1][kx + 1];
                        sum_y += p * KY[ky + 1][kx + 1];
                    }
                }
            }

            int idx = y * width + x;
            gx[idx] = (int16_t)sum_x;   // safe: |sum| <= 1020 fits in int16_t
            gy[idx] = (int16_t)sum_y;
        }
    }
}

// ============================================================================
// STAGE 3a - GRADIENT MAGNITUDE, L1 NORM
// ============================================================================

void compute_magnitude_l1_scalar(const int16_t* gx, const int16_t* gy,
                                 uint8_t* magnitude,
                                 int width, int height) {
    int total = width * height;

    // Temporary 32-bit buffer holds raw (un-normalized) magnitudes between passes
    int32_t* raw = (int32_t*)aligned_alloc(64, align64(total * sizeof(int32_t)));

    // Pass 1: compute |gx| + |gy| and track the global maximum
    int32_t max_mag = 0;
    for (int i = 0; i < total; ++i) {
        int32_t ax = gx[i] < 0 ? -gx[i] : gx[i];   // |gx|
        int32_t ay = gy[i] < 0 ? -gy[i] : gy[i];   // |gy|
        int32_t mag = ax + ay;
        raw[i] = mag;
        if (mag > max_mag) max_mag = mag;
    }

    // Pass 2: scale every pixel to [0, 255] using the global maximum
    for (int i = 0; i < total; ++i) {
        magnitude[i] = (max_mag > 0) ? (uint8_t)(raw[i] * 255 / max_mag) : 0;
    }

    free(raw);
}

// ============================================================================
// STAGE 3a - GRADIENT MAGNITUDE, L2 NORM
// ============================================================================

void compute_magnitude_l2_scalar(const int16_t* gx, const int16_t* gy,
                                 uint8_t* magnitude,
                                 int width, int height) {
    int total = width * height;

    // Temporary float buffer holds raw magnitudes between passes
    float* raw = (float*)aligned_alloc(64, align64(total * sizeof(float)));

    // Pass 1: compute sqrt(gx^2 + gy^2) and track the global maximum
    float max_mag = 0.0f;
    for (int i = 0; i < total; ++i) {
        float fx = (float)gx[i];
        float fy = (float)gy[i];
        float mag = sqrtf(fx * fx + fy * fy);
        raw[i] = mag;
        if (mag > max_mag) max_mag = mag;
    }

    // Pass 2: scale every pixel to [0, 255] using the global maximum
    for (int i = 0; i < total; ++i) {
        magnitude[i] = (max_mag > 0.0f) ? (uint8_t)(raw[i] * 255.0f / max_mag) : 0;
    }

    free(raw);
}

// ============================================================================
// STAGE 3b - GRADIENT DIRECTION
// ============================================================================

void compute_direction_scalar(const int16_t* gx, const int16_t* gy,
                              uint8_t* direction,
                              int width, int height) {
    int total = width * height;

    for (int i = 0; i < total; ++i) {
        int32_t ax = gx[i] < 0 ? -gx[i] : gx[i];   // |gx|
        int32_t ay = gy[i] < 0 ? -gy[i] : gy[i];   // |gy|

        uint8_t angle;
        if      (ay * 5 < ax * 2)             angle = 0;    // tan < tan(22.5) -> horizontal
        else if (ax * 5 < ay * 2)             angle = 90;   // tan > tan(67.5) -> vertical
        else if ((int32_t)gx[i] * gy[i] > 0) angle = 45;   // same-sign diagonal
        else                                   angle = 135;  // opposite-sign diagonal

        direction[i] = angle;
    }
}

// ============================================================================
// STAGE 4 - NON-MAXIMUM SUPPRESSION  [bonus]
// ============================================================================

void non_maximum_suppression_scalar(const uint8_t* magnitude,
                                    const uint8_t* direction,
                                    uint8_t* output,
                                    int width, int height) {
    std::memset(output, 0, width * height);   // clears borders and all suppressed pixels

    // Skip the 1-pixel border: those pixels have out-of-bounds neighbors
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            int mag = magnitude[idx];

            int n1 = 0, n2 = 0;   // the two neighbors along the gradient direction
            switch (direction[idx]) {
                case 0:                                 // horizontal gradient
                    n1 = magnitude[idx - 1];            // left
                    n2 = magnitude[idx + 1];            // right
                    break;
                case 90:                                // vertical gradient
                    n1 = magnitude[idx - width];        // top
                    n2 = magnitude[idx + width];        // bottom
                    break;
                case 45:                                // diagonal
                    n1 = magnitude[idx - width + 1];    // top-right
                    n2 = magnitude[idx + width - 1];    // bottom-left
                    break;
                case 135:                               // anti-diagonal
                    n1 = magnitude[idx - width - 1];    // top-left
                    n2 = magnitude[idx + width + 1];    // bottom-right
                    break;
            }

            // Keep the pixel only if it is at least as large as both neighbors
            output[idx] = (mag >= n1 && mag >= n2) ? (uint8_t)mag : 0;
        }
    }
}

// ============================================================================
// STAGE 5a - DOUBLE THRESHOLDING  [bonus]
// ============================================================================

void double_threshold_scalar(const uint8_t* nms_input,
                             uint8_t* labels,
                             int width, int height,
                             uint8_t low_threshold,
                             uint8_t high_threshold) {
    int total = width * height;
    for (int i = 0; i < total; ++i) {
        if      (nms_input[i] >= high_threshold) labels[i] = EDGE_STRONG;
        else if (nms_input[i] >= low_threshold)  labels[i] = EDGE_WEAK;
        else                                      labels[i] = EDGE_NONE;
    }
}

// ============================================================================
// STAGE 5b - HYSTERESIS EDGE TRACING  [bonus]
// ============================================================================

void hysteresis_scalar(const uint8_t* labels,
                       uint8_t* output,
                       int width, int height) {
    int total = width * height;

    // Mutable working copy so weak pixels can be promoted to strong in place
    uint8_t* work = (uint8_t*)aligned_alloc(64, align64(total));
    std::memcpy(work, labels, total);

    // Iterative flood-fill: each pass promotes weak pixels adjacent to a strong
    // pixel. Repeat until a full pass makes no change, meaning every reachable
    // chain has been traced. This is what prevents broken edge lines.
    bool changed = true;
    while (changed) {
        changed = false;

        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;
                if (work[idx] != EDGE_WEAK) continue;   // only weak pixels can be promoted

                // Scan the 8-connected neighborhood for any strong pixel
                bool touches_strong = false;
                for (int ny = -1; ny <= 1 && !touches_strong; ++ny) {
                    for (int nx = -1; nx <= 1; ++nx) {
                        if (work[(y + ny) * width + (x + nx)] == EDGE_STRONG) {
                            touches_strong = true;
                            break;
                        }
                    }
                }

                if (touches_strong) {
                    work[idx] = EDGE_STRONG;   // promote
                    changed = true;            // a change means another pass is needed
                }
            }
        }
    }

    // Final output: strong pixels become edges (255), everything else is background (0)
    for (int i = 0; i < total; ++i) {
        output[i] = (work[i] == EDGE_STRONG) ? 255 : 0;
    }

    free(work);
}