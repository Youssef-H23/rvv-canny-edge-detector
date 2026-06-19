#include <riscv_vector.h>
#include "../headers/canny_scalar.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>

// ============================================================================
// STAGE 1 - GAUSSIAN BLUR (SEPARABLE FILTER, RISC-V VECTORIZED)
// ============================================================================
// The 5x5 integer Gaussian kernel [1,4,7,4,1]^T * [1,4,7,4,1] is separable.
// Horizontal pass: accumulate 5 horizontal neighbors into int32 temp buffer.
// Vertical pass:   accumulate 5 vertical neighbors from temp, normalize to u8.
// This reduces MACs from 25 to 10 and eliminates LMUL=4 register pressure.
void gaussian_blur_scalar(const uint8_t *input, uint8_t *output, int w, int h) {
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

void sobel_gradients_scalar(const uint8_t *input, int16_t *gx, int16_t *gy, int w, int h) {

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
