#include "../headers/canny_scalar.h"

#include <cstdio>    // fopen, fread, fwrite, fclose

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
