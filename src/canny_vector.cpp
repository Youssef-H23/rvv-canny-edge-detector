#include "../headers/canny_vector.h"
#include "../headers/canny_scalar.h"   // align64
#include <riscv_vector.h>
#include <cstdint>
#include <cstdlib>   // aligned_alloc, free
#include <cstring>   // memset

// ============================================================================
// STAGE 1 - GAUSSIAN BLUR (RISC-V VECTORIZED)
//
// Self-contained 5x5 convolution (no separate convolve2D_rvv helper):
// multiply-accumulate a 5x5 neighborhood per pixel, then normalize. The outer
// loops (kernel row ky, kernel column kx) stay scalar -- ksize*ksize is only
// 25, so there's nothing to gain vectorizing them. What repeats over every
// output pixel is the inner load/widen/multiply-accumulate, so that's the
// part that's vectorized.
//
// Normalization is a fixed-point multiply-shift, (sum * 240) >> 16, instead
// of a real divide (65536/273 ~= 240, approximating sum / 273): the divisor
// is a compile-time constant, but a vector integer divide is a real per-lane
// instruction with no constant-divisor strength reduction (unlike scalar
// sum/divisor, which GCC turns into multiply+shift automatically), and it is
// slow under QEMU's emulation.
//
// Border handling: not implemented yet. The output is pre-zeroed and only
// pixels at least `radius` away from every edge are computed, matching the
// project guide's "interior first" hint for Phase 6.
// ============================================================================
void gaussian_blur_rvv(const uint8_t* input, uint8_t* output, int width, int height) {
    static const int16_t GAUSSIAN_KERNEL[25] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
    const int ksize       = 5;
    const int radius      = ksize / 2;
    const uint32_t fixed_mul   = 240;  // fixed-point numerator
    const uint32_t fixed_shift = 16;   // right-shift amount

    std::memset(output, 0, (size_t)width * height);

    for (int y = radius; y < height - radius; ++y) {
        int x = radius;
        int pixels_left = width - 2 * radius;

        while (pixels_left > 0) {
            size_t vl = __riscv_vsetvl_e8m2(pixels_left);
            vuint32m8_t vec_sum = __riscv_vmv_v_x_u32m8(0, vl);

            for (int ky = -radius; ky <= radius; ++ky) {
                const uint8_t* row = input + (size_t)(y + ky) * width + x; // points to the first pixel of the current kernel row
                for (int kx = -radius; kx <= radius; ++kx) {
                    int16_t weight = GAUSSIAN_KERNEL[(ky + radius) * ksize + (kx + radius)];
                    if (weight == 0) continue;   // no need to load/multiply/accumulate if the weight is 0

                    vuint8m2_t pix    = __riscv_vle8_v_u8m2(row + kx, vl);
                    vuint16m4_t pix16 = __riscv_vwcvtu_x_x_v_u16m4(pix, vl);
                    vec_sum = __riscv_vwmaccu_vx_u32m8(vec_sum, (uint16_t)weight, pix16, vl);
                }
            }

            vuint32m8_t normalized = __riscv_vsrl_vx_u32m8(
                __riscv_vmul_vx_u32m8(vec_sum, fixed_mul, vl), fixed_shift, vl);
            vuint16m4_t n16 = __riscv_vncvt_x_x_w_u16m4(normalized, vl);
            vuint8m2_t  n8  = __riscv_vncvt_x_x_w_u8m2(n16, vl);
            __riscv_vse8_v_u8m2(output + (size_t)y * width + x, n8, vl);

            x += (int)vl;
            pixels_left -= (int)vl;
        }
    }
}



// ============================================================================
// STAGE 1 - GAUSSIAN BLUR (SEPARABLE FILTER, RISC-V VECTORIZED)
// ============================================================================
// The 5x5 integer Gaussian kernel [1,4,7,4,1]^T * [1,4,7,4,1] is separable.
// Horizontal pass: accumulate 5 horizontal neighbors into int32 temp buffer.
// Vertical pass:   accumulate 5 vertical neighbors from temp, normalize to u8.
// This reduces MACs from 25 to 10 and eliminates LMUL=4 register pressure.
void gaussian_blur_rvv_separable(const uint8_t *input, uint8_t *output, int w, int h) {
    static const int16_t K[5] = {1, 4, 7, 4, 1};
    const int ksum = 273;

    int total = w * h;
    int32_t *temp = (int32_t *)aligned_alloc(64, align64(total * sizeof(int32_t)));
    if (!temp) return;

    for (int y = 0; y < h; ++y) {
        const uint8_t *row = input + y * w;
        int32_t *dst = temp + y * w;

        for (int x = 2; x < w - 2; ) {
            int n = w - 2 - x;
            size_t vl = __riscv_vsetvl_e8mf2(n);
            const uint8_t *src = row + x - 2;

            vuint32m2_t sum = __riscv_vmv_v_x_u32m2(0, vl);
            vuint8mf2_t col;

            col = __riscv_vle8_v_u8mf2(src + 0, vl);
            sum = __riscv_vwmaccu_vx_u32m2(sum, K[0], __riscv_vwcvtu_x_x_v_u16m1(col, vl), vl);
            col = __riscv_vle8_v_u8mf2(src + 1, vl);
            sum = __riscv_vwmaccu_vx_u32m2(sum, K[1], __riscv_vwcvtu_x_x_v_u16m1(col, vl), vl);
            col = __riscv_vle8_v_u8mf2(src + 2, vl);
            sum = __riscv_vwmaccu_vx_u32m2(sum, K[2], __riscv_vwcvtu_x_x_v_u16m1(col, vl), vl);
            col = __riscv_vle8_v_u8mf2(src + 3, vl);
            sum = __riscv_vwmaccu_vx_u32m2(sum, K[3], __riscv_vwcvtu_x_x_v_u16m1(col, vl), vl);
            col = __riscv_vle8_v_u8mf2(src + 4, vl);
            sum = __riscv_vwmaccu_vx_u32m2(sum, K[4], __riscv_vwcvtu_x_x_v_u16m1(col, vl), vl);

            __riscv_vse32_v_u32m2((uint32_t *)(dst + x), sum, vl);
            x += vl;
        }
    }

    std::memset(output, 0, w * h);

    for (int y = 2; y < h - 2; ++y) {
        int32_t *r0 = temp + (y - 2) * w;
        int32_t *r1 = temp + (y - 1) * w;
        int32_t *r2 = temp + y * w;
        int32_t *r3 = temp + (y + 1) * w;
        int32_t *r4 = temp + (y + 2) * w;
        uint8_t *out = output + y * w + 2;

        int n = w - 4;
        int32_t *c0 = r0 + 2, *c1 = r1 + 2, *c2 = r2 + 2, *c3 = r3 + 2, *c4 = r4 + 2;

        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m2(n);

            vuint32m2_t s0 = __riscv_vle32_v_u32m2((const uint32_t *)c0, vl);
            vuint32m2_t s1 = __riscv_vle32_v_u32m2((const uint32_t *)c1, vl);
            vuint32m2_t s2 = __riscv_vle32_v_u32m2((const uint32_t *)c2, vl);
            vuint32m2_t s3 = __riscv_vle32_v_u32m2((const uint32_t *)c3, vl);
            vuint32m2_t s4 = __riscv_vle32_v_u32m2((const uint32_t *)c4, vl);

            vuint32m2_t sum = __riscv_vmv_v_x_u32m2(0, vl);
            sum = __riscv_vmacc_vx_u32m2(sum, K[0], s0, vl);
            sum = __riscv_vmacc_vx_u32m2(sum, K[1], s1, vl);
            sum = __riscv_vmacc_vx_u32m2(sum, K[2], s2, vl);
            sum = __riscv_vmacc_vx_u32m2(sum, K[3], s3, vl);
            sum = __riscv_vmacc_vx_u32m2(sum, K[4], s4, vl);

            vuint32m2_t norm32 = __riscv_vdivu_vx_u32m2(sum, ksum, vl);
            vuint16m1_t n16    = __riscv_vncvt_x_x_w_u16m1(norm32, vl);
            vuint8mf2_t n8     = __riscv_vncvt_x_x_w_u8mf2(n16, vl);
            __riscv_vse8_v_u8mf2(out, n8, vl);

            c0 += vl; c1 += vl; c2 += vl; c3 += vl; c4 += vl;
            out += vl; n -= vl;
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (y >= 2 && y < h - 2 && x >= 2 && x < w - 2) continue;
            int32_t s = 0;
            for (int ky = -2; ky <= 2; ++ky)
                for (int kx = -2; kx <= 2; ++kx) {
                    int ix = x + kx, iy = y + ky;
                    if (ix >= 0 && ix < w && iy >= 0 && iy < h)
                        s += static_cast<int32_t>(input[iy * w + ix]) * K[kx + 2] * K[ky + 2];
                }
            s /= ksum;
            if (s < 0) s = 0;
            if (s > 255) s = 255;
            output[y * w + x] = static_cast<uint8_t>(s);
        }
    }

    free(temp);
}


// ============================================================================
// STAGE 2 - SOBEL GRADIENTS (RISC-V VECTORIZED)
// ============================================================================
static const int KX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
static const int KY[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

void sobel_gradients_rvv(const uint8_t *input, int16_t *gx, int16_t *gy, int w, int h) {

    std::memset(gx, 0, w * h * sizeof(int16_t));
    std::memset(gy, 0, w * h * sizeof(int16_t));

    for (int y = 1; y < h - 1; ++y) {
        const uint8_t *r0 = input + (y - 1) * w;
        const uint8_t *r1 = input + (y - 0) * w;
        const uint8_t *r2 = input + (y + 1) * w;
        int16_t *gx_out = gx + y * w + 1;
        int16_t *gy_out = gy + y * w + 1;

        int n = w - 2;
        const uint8_t *s0 = r0 + 1, *s1 = r1 + 1, *s2 = r2 + 1;

        while (n > 0) {
            size_t vl = __riscv_vsetvl_e8m1(n);

            vuint8m1_t r0_m1 = __riscv_vle8_v_u8m1(s0 - 1, vl);
            vuint8m1_t r0_0  = __riscv_vle8_v_u8m1(s0 - 0, vl);
            vuint8m1_t r0_p1 = __riscv_vle8_v_u8m1(s0 + 1, vl);
            vuint8m1_t r1_m1 = __riscv_vle8_v_u8m1(s1 - 1, vl);
            vuint8m1_t r1_p1 = __riscv_vle8_v_u8m1(s1 + 1, vl);
            vuint8m1_t r2_m1 = __riscv_vle8_v_u8m1(s2 - 1, vl);
            vuint8m1_t r2_0  = __riscv_vle8_v_u8m1(s2 - 0, vl);
            vuint8m1_t r2_p1 = __riscv_vle8_v_u8m1(s2 + 1, vl);

            auto widen = [](vuint8m1_t v, size_t L) {
                return __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwcvtu_x_x_v_u16m2(v, L));
            };

            vint16m2_t tl  = widen(r0_m1, vl);
            vint16m2_t tr  = widen(r0_0,  vl);
            vint16m2_t tr2 = widen(r0_p1, vl);
            vint16m2_t ml  = widen(r1_m1, vl);
            vint16m2_t mr  = widen(r1_p1, vl);
            vint16m2_t bl  = widen(r2_m1, vl);
            vint16m2_t bc  = widen(r2_0,  vl);
            vint16m2_t br  = widen(r2_p1, vl);

            vint16m2_t gx_top = __riscv_vsub_vv_i16m2(tr2, tl, vl);
            vint16m2_t gx_mid = __riscv_vsll_vx_i16m2(__riscv_vsub_vv_i16m2(mr, ml, vl), 1, vl);
            vint16m2_t gx_bot = __riscv_vsub_vv_i16m2(br, bl, vl);
            vint16m2_t vgx = __riscv_vadd_vv_i16m2(
                __riscv_vadd_vv_i16m2(gx_top, gx_mid, vl), gx_bot, vl);

            vint16m2_t gy_left = __riscv_vsub_vv_i16m2(bl, tl, vl);
            vint16m2_t gy_mid  = __riscv_vsll_vx_i16m2(__riscv_vsub_vv_i16m2(bc, tr, vl), 1, vl);
            vint16m2_t gy_rght = __riscv_vsub_vv_i16m2(br, tr2, vl);
            vint16m2_t vgy = __riscv_vadd_vv_i16m2(
                __riscv_vadd_vv_i16m2(gy_left, gy_mid, vl), gy_rght, vl);

            __riscv_vse16_v_i16m2(gx_out, vgx, vl);
            __riscv_vse16_v_i16m2(gy_out, vgy, vl);

            s0 += vl; s1 += vl; s2 += vl;
            gx_out += vl; gy_out += vl;
            n -= vl;
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (y >= 1 && y < h - 1 && x >= 1 && x < w - 1) continue;
            int32_t sgx = 0, sgy = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    int ix = x + kx, iy = y + ky;
                    if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
                        int p = input[iy * w + ix];
                        sgx += p * KX[ky + 1][kx + 1];
                        sgy += p * KY[ky + 1][kx + 1];
                    }
                }
            gx[y * w + x] = (int16_t)sgx;
            gy[y * w + x] = (int16_t)sgy;
        }
    }
}

// ============================================================================
// STAGE 3a - GRADIENT MAGNITUDE, L1 NORM (RISC-V VECTORIZED)
//
// New RVV concept: vector reduction (project guide section 6.5). Pass 1
// computes |gx|+|gy| per element and folds a running global maximum into a
// scalar via __riscv_vredmaxu_vs -- a reduction collapses a vector down to
// one value, written into element 0 of an m1 destination register, and
// extracted with __riscv_vmv_x_s. The seed passed into each chunk's
// reduction is the running max so far, so the maximum survives across
// strip-mined chunks. Pass 2 normalizes every element by that global max,
// exactly like compute_magnitude_l1_scalar's two-pass approach.
// ============================================================================
void compute_magnitude_l1_rvv(const int16_t* gx, const int16_t* gy,
                              uint8_t* magnitude, int width, int height) {
    int total = width * height;
    uint16_t* raw = (uint16_t*)aligned_alloc(64, align64(total * sizeof(uint16_t)));

    // ---- Pass 1: |gx| + |gy|, tracking the global max in a vector register ----
    vuint16m1_t running_max = __riscv_vmv_s_x_u16m1(0, 1);

    int i = 0;
    while (i < total) {
        size_t vl = __riscv_vsetvl_e16m4(total - i);

        vint16m4_t vgx = __riscv_vle16_v_i16m4(gx + i, vl);
        vint16m4_t vgy = __riscv_vle16_v_i16m4(gy + i, vl);

        // |x| = max(x, -x). Sum fits comfortably in int16 (max 1020+1020=2040).
        vint16m4_t abs_gx = __riscv_vmax_vv_i16m4(vgx, __riscv_vneg_v_i16m4(vgx, vl), vl);
        vint16m4_t abs_gy = __riscv_vmax_vv_i16m4(vgy, __riscv_vneg_v_i16m4(vgy, vl), vl);
        vint16m4_t mag    = __riscv_vadd_vv_i16m4(abs_gx, abs_gy, vl);
        vuint16m4_t umag  = __riscv_vreinterpret_v_i16m4_u16m4(mag);

        __riscv_vse16_v_u16m4(raw + i, umag, vl);

        // Fold this chunk's max into the running max (the seed carries the
        // previous chunks' max forward, so it survives the whole image).
        running_max = __riscv_vredmaxu_vs_u16m4_u16m1(umag, running_max, vl);

        i += (int)vl;
    }

    uint16_t max_mag = __riscv_vmv_x_s_u16m1_u16(running_max);

    // ---- Pass 2: normalize to [0, 255] ----
    if (max_mag == 0) {
        std::memset(magnitude, 0, total);
        free(raw);
        return;
    }

    i = 0;
    while (i < total) {
        size_t vl = __riscv_vsetvl_e16m4(total - i);

        vuint16m4_t vraw   = __riscv_vle16_v_u16m4(raw + i, vl);
        vuint32m8_t vraw32 = __riscv_vwcvtu_x_x_v_u32m8(vraw, vl);
        vuint32m8_t scaled = __riscv_vmul_vx_u32m8(vraw32, 255, vl);
        vuint32m8_t norm32 = __riscv_vdivu_vx_u32m8(scaled, max_mag, vl);
        vuint16m4_t norm16 = __riscv_vncvt_x_x_w_u16m4(norm32, vl);
        vuint8m2_t  norm8  = __riscv_vncvt_x_x_w_u8m2(norm16, vl);

        __riscv_vse8_v_u8m2(magnitude + i, norm8, vl);
        i += (int)vl;
    }

    free(raw);
}
