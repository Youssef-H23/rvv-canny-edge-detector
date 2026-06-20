// ============================================================================
// PRESENTATION BENCHMARK: filter/norm timing comparisons
//
// Not a correctness test (see host_tests.cpp / vector_tests.cpp for that).
// This is a standalone QEMU benchmark, same style as lmul_experiment.cpp,
// that times pairs of already-implemented kernels against each other so the
// numbers are ready ahead of the live demo:
//
//   1. Gaussian 5x5 2D convolution vs separable (1x5 + 5x1), SCALAR
//   2. Gaussian 5x5 2D convolution vs separable (1x5 + 5x1), RVV
//   3. Magnitude L1 (|gx|+|gy|) vs L2 (sqrt(gx^2+gy^2)), SCALAR
//      (no RVV L2 exists yet, so this pair is scalar-only)
// ============================================================================
#include "../headers/canny_scalar.h"
#include "../headers/canny_vector.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main() {
    const int W = 512;
    const int H = 512;
    const int TOTAL = W * H;
    const int ITERATIONS = 100;

    uint8_t* input  = (uint8_t*)aligned_alloc(64, align64(TOTAL));
    uint8_t* out_a  = (uint8_t*)aligned_alloc(64, align64(TOTAL));
    uint8_t* out_b  = (uint8_t*)aligned_alloc(64, align64(TOTAL));
    int16_t* gx     = (int16_t*)aligned_alloc(64, align64(TOTAL * sizeof(int16_t)));
    int16_t* gy     = (int16_t*)aligned_alloc(64, align64(TOTAL * sizeof(int16_t)));
    uint8_t* mag_a  = (uint8_t*)aligned_alloc(64, align64(TOTAL));
    uint8_t* mag_b  = (uint8_t*)aligned_alloc(64, align64(TOTAL));

    // Deterministic synthetic data -- timing for these kernels doesn't depend
    // on pixel content, only on width*height, so a real photo buys nothing here.
    for (int i = 0; i < TOTAL; ++i) input[i] = (uint8_t)(i % 256);
    for (int i = 0; i < TOTAL; ++i) {
        gx[i] = (int16_t)((i % 2041) - 1020);        // covers full Sobel output range
        gy[i] = (int16_t)(((i * 7) % 2041) - 1020);
    }

    printf("=========================================================\n");
    printf("   FILTER / NORM TIMING COMPARISONS (%d iterations, %dx%d)\n", ITERATIONS, W, H);
    printf("=========================================================\n\n");

    // ---- 1. Gaussian blur: 2D vs separable, SCALAR ----
    printf("--- GAUSSIAN BLUR 5x5, SCALAR ---\n");
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) gaussian_blur_scalar(input, out_a, W, H);
    auto e1 = std::chrono::high_resolution_clock::now();
    double t_2d_scalar = std::chrono::duration<double>(e1 - s1).count();
    printf("[2D conv]      Time: %6.4f s (Base)\n", t_2d_scalar);

    auto s2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) gaussian_blur_separable_scalar(input, out_b, W, H);
    auto e2 = std::chrono::high_resolution_clock::now();
    double t_sep_scalar = std::chrono::duration<double>(e2 - s2).count();
    printf("[Separable]    Time: %6.4f s (Speedup: %.2fx)\n\n", t_sep_scalar, t_2d_scalar / t_sep_scalar);

    // ---- 2. Gaussian blur: 2D vs separable, RVV ----
    printf("--- GAUSSIAN BLUR 5x5, RVV ---\n");
    auto s3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) gaussian_blur_rvv(input, out_a, W, H);
    auto e3 = std::chrono::high_resolution_clock::now();
    double t_2d_rvv = std::chrono::duration<double>(e3 - s3).count();
    printf("[2D conv]      Time: %6.4f s (Base)\n", t_2d_rvv);

    auto s4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) gaussian_blur_rvv_separable(input, out_b, W, H);
    auto e4 = std::chrono::high_resolution_clock::now();
    double t_sep_rvv = std::chrono::duration<double>(e4 - s4).count();
    printf("[Separable]    Time: %6.4f s (Speedup: %.2fx)\n\n", t_sep_rvv, t_2d_rvv / t_sep_rvv);

    printf("--- SCALAR -> RVV SPEEDUP ---\n");
    printf("[2D conv]      %.2fx\n", t_2d_scalar / t_2d_rvv);
    printf("[Separable]    %.2fx\n\n", t_sep_scalar / t_sep_rvv);

    // ---- 3. Magnitude: L1 vs L2, SCALAR (no L2 RVV kernel exists) ----
    printf("--- GRADIENT MAGNITUDE, SCALAR ---\n");
    auto s5 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) compute_magnitude_l1_scalar(gx, gy, mag_a, W, H);
    auto e5 = std::chrono::high_resolution_clock::now();
    double t_l1 = std::chrono::duration<double>(e5 - s5).count();
    printf("[L1, |gx|+|gy|]      Time: %6.4f s (Base)\n", t_l1);

    auto s6 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) compute_magnitude_l2_scalar(gx, gy, mag_b, W, H);
    auto e6 = std::chrono::high_resolution_clock::now();
    double t_l2 = std::chrono::duration<double>(e6 - s6).count();
    printf("[L2, sqrt(gx^2+gy^2)] Time: %6.4f s (%.2fx slower than L1)\n\n", t_l2, t_l2 / t_l1);

    free(input); free(out_a); free(out_b);
    free(gx); free(gy); free(mag_a); free(mag_b);
    return 0;
}
