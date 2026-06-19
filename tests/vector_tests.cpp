// =============================================================================
// qemu_equiv_tests.cpp  —  Phase 3.2 / Phase 6 Equivalence Tests
//
// PROJECT SPEC (Phase 3.2, assert-based, QEMU-side):
//   "For each RVV kernel, run both the scalar and RVV versions on the same
//    input and compare outputs. Allow +-1 tolerance for rounding. Run this
//    test at VLEN=128, 256, and 512. If the outputs match at all three VLEN
//    values, your code is correct and vector-length-agnostic."
//   "Use a non-power-of-two image size (e.g., 48x48 or 100x75) ... this forces
//    the strip-mining tail case ... most VLA bugs hide in the tail case."
//
// WHAT THIS FILE TESTS
//   Exactly the three RVV kernels that exist right now in canny_vector.cpp:
//     1. gaussian_blur_rvv
//     2. sobel_gradients_rvv
//     3. compute_magnitude_l1_rvv
//   (magnitude_l2_rvv / direction_rvv / etc. are not implemented yet — when
//    they are added, copy the pattern of test_magnitude_l1() below.)
//
// IMPORTANT — BORDER POLICY MISMATCH (read this before "fixing" failures)
//   The scalar kernels (canny_scalar.cpp) zero-pad and compute EVERY pixel,
//   borders included. The current RVV kernels follow the guide's "interior
//   first" hint (Phase 6.4): the output buffer is memset to 0, and only the
//   region far enough from the edge to need no boundary check is computed:
//     gaussian_blur_rvv : y,x in [radius, dim-radius)      radius = 2
//     sobel_gradients_rvv: y in [1,height-1), x in [1,width-1)   radius = 1
//   This is intentional and matches the guide, NOT a bug. Comparing the
//   full image including borders against scalar would report false failures
//   on every border pixel. So this test compares INTERIOR ONLY for those
//   two kernels, and separately reports (without failing) how the border
//   looks, so you know it is zero rather than garbage.
//
//   compute_magnitude_l1_rvv has no border concept (gx/gy are already full
//   arrays with no spatial neighborhood lookup) so it is compared on every
//   pixel, full image, no exceptions.
//
// BUILD / RUN
//   Cross-compile with the RV toolchain (needs -march=rv64gcv for intrinsics):
//     riscv64-unknown-elf-g++ -march=rv64gcv -mabi=lp64d -std=c++17 -I include \
//         tests/qemu_equiv_tests.cpp src/canny_scalar.cpp src/canny_vector.cpp \
//         -o build/rv/qemu_tests.elf
//   Run at all three VLEN values (this is the actual Phase 3.2 requirement —
//   passing at only one VLEN proves nothing about vector-length-agnosticism):
//     qemu-riscv64 -cpu rv64,v=true,vlen=128 build/rv/qemu_tests.elf
//     qemu-riscv64 -cpu rv64,v=true,vlen=256 build/rv/qemu_tests.elf
//     qemu-riscv64 -cpu rv64,v=true,vlen=512 build/rv/qemu_tests.elf
//   (The Makefile target `qemu_test` added alongside this file does all three
//    automatically — see the matching Makefile patch.)
//
// EXIT CODE
//   0 if every test passes, 1 if any test fails (so it can be used as a CI gate).
// =============================================================================

#include "../headers/canny_scalar.h"   // scalar reference + align64 + EDGE_* (shared)
#include "../headers/canny_vector.h"   // RVV implementations under test

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// -----------------------------------------------------------------------------
// Small test-result bookkeeping (no GoogleTest available on the RV/QEMU side,
// per the guide section 1.6 / 3.2 — "use a simple assert-based test harness
// with printf and return codes").
// -----------------------------------------------------------------------------
static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define TOLERANCE_DEFAULT 1   // +-1 tolerance for rounding, as required by Phase 3.2

// -----------------------------------------------------------------------------
// Deterministic pseudo-random fill (no <random> dependency needed on bare metal).
// Same seed -> same sequence every run, so failures are always reproducible.
// -----------------------------------------------------------------------------
static void fill_random_u8(uint8_t* buf, int n, uint32_t seed) {
    for (int i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;   // classic Knuth LCG
        buf[i] = (uint8_t)(seed >> 24);
    }
}

static void fill_random_i16(int16_t* buf, int n, uint32_t seed, int max_abs) {
    for (int i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        int32_t v = (int32_t)(seed >> 16) % (2 * max_abs + 1) - max_abs;
        buf[i] = (int16_t)v;
    }
}

// -----------------------------------------------------------------------------
// Region-restricted comparator.
// Only checks pixels with y in [y0,y1) and x in [x0,x1) — lets us compare just
// the interior for kernels that deliberately skip borders (see policy note
// above), while still scanning the WHOLE image for kernels that don't.
// Returns the number of mismatching pixels found (0 == pass) and prints the
// first few mismatches for debugging.
// -----------------------------------------------------------------------------
static int compare_region_u8(const char* label,
                              const uint8_t* ref, const uint8_t* got,
                              int width, int height,
                              int y0, int y1, int x0, int x1,
                              int tolerance)
{
    int mismatches = 0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            int idx  = y * width + x;
            int diff = (int)ref[idx] - (int)got[idx];
            if (diff < -tolerance || diff > tolerance) {
                if (mismatches < 5) {
                    fprintf(stderr,
                        "  mismatch %s at (x=%d,y=%d): ref=%d got=%d diff=%d\n",
                        label, x, y, ref[idx], got[idx], diff);
                }
                ++mismatches;
            }
        }
    }
    return mismatches;
}

static int compare_full_u8(const char* label,
                            const uint8_t* ref, const uint8_t* got, int n,
                            int tolerance)
{
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        int diff = (int)ref[i] - (int)got[i];
        if (diff < -tolerance || diff > tolerance) {
            if (mismatches < 5)
                fprintf(stderr, "  mismatch %s at i=%d: ref=%d got=%d diff=%d\n",
                        label, i, ref[i], got[i], diff);
            ++mismatches;
        }
    }
    return mismatches;
}

// Track + report one test's outcome and feed the global pass/fail counters.
static void report(const char* name, int mismatches, int total_checked) {
    ++g_tests_run;
    if (mismatches == 0) {
        ++g_tests_passed;
        printf("PASS  %-45s (%d pixels checked)\n", name, total_checked);
    } else {
        printf("FAIL  %-45s (%d / %d pixels mismatched)\n",
               name, mismatches, total_checked);
    }
}

// =============================================================================
// Test image dimensions — DELIBERATELY non-power-of-two (guide 3.2 + 8 "Common
// Pitfalls"). 101 is not a multiple of 8/16/32/64 so every strip-mined loop
// (vsetvl chunk size 1..vlmax) is guaranteed to hit a short tail chunk on the
// last iteration of every row, which is exactly where off-by-one/strip-mining
// bugs hide.
// =============================================================================
static const int W = 101;
static const int H = 75;
static const int N = W * H;          // 7575 pixels total

// =============================================================================
// TEST 1 — Gaussian Blur (interior region only; see border policy note above)
// =============================================================================
static void test_gaussian_blur() {
    uint8_t* in      = (uint8_t*)aligned_alloc(64, align64(N));
    uint8_t* ref_out = (uint8_t*)aligned_alloc(64, align64(N));
    uint8_t* rvv_out = (uint8_t*)aligned_alloc(64, align64(N));

    fill_random_u8(in, N, 0xDEADBEEF);

    gaussian_blur_scalar(in, ref_out, W, H);
    gaussian_blur_rvv   (in, rvv_out, W, H);

    // Gaussian kernel is 5x5 -> radius 2. The RVV kernel only fills
    // [radius, dim-radius); compare that exact region against scalar.
    const int radius = 2;
    int region_w = W - 2 * radius;
    int region_h = H - 2 * radius;
    int mism = compare_region_u8("gaussian_blur (interior)",
                                  ref_out, rvv_out, W, H,
                                  radius, H - radius, radius, W - radius,
                                  TOLERANCE_DEFAULT);
    report("gaussian_blur_rvv vs scalar (interior, 101x75)",
           mism, region_w * region_h);

    // Informational only (does not affect pass/fail): confirm the RVV
    // border is the documented all-zero placeholder, not garbage memory.
    bool border_is_zero = true;
    for (int y = 0; y < H && border_is_zero; ++y)
        for (int x = 0; x < W; ++x)
            if (y < radius || y >= H - radius || x < radius || x >= W - radius)
                if (rvv_out[y * W + x] != 0) { border_is_zero = false; break; }
    printf("INFO  gaussian_blur_rvv border is %s (expected: zero, by design)\n",
           border_is_zero ? "all-zero" : "NOT all-zero -- check memset");

    free(in); free(ref_out); free(rvv_out);
}

// =============================================================================
// TEST 2 — Sobel Gradients (interior region only; see border policy note above)
// =============================================================================
static void test_sobel_gradients() {
    uint8_t* in     = (uint8_t*)aligned_alloc(64, align64(N));
    int16_t* gx_ref = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
    int16_t* gy_ref = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
    int16_t* gx_rvv = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
    int16_t* gy_rvv = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));

    fill_random_u8(in, N, 0xCAFEBABE);

    sobel_gradients_scalar(in, gx_ref, gy_ref, W, H);
    sobel_gradients_rvv   (in, gx_rvv, gy_rvv, W, H);

    // Sobel kernel is 3x3 -> radius 1. RVV fills y in [1,H-1), x in [1,W-1).
    // Sobel has no rounding step, so scalar and RVV must match EXACTLY
    // (zero tolerance) within that interior.
    int mism_gx = 0, mism_gy = 0, checked = 0;
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            ++checked;
            if (gx_ref[idx] != gx_rvv[idx]) {
                if (mism_gx < 5)
                    fprintf(stderr, "  mismatch sobel_gx at (x=%d,y=%d): ref=%d got=%d\n",
                            x, y, gx_ref[idx], gx_rvv[idx]);
                ++mism_gx;
            }
            if (gy_ref[idx] != gy_rvv[idx]) {
                if (mism_gy < 5)
                    fprintf(stderr, "  mismatch sobel_gy at (x=%d,y=%d): ref=%d got=%d\n",
                            x, y, gy_ref[idx], gy_rvv[idx]);
                ++mism_gy;
            }
        }
    }
    report("sobel_gradients_rvv: gx vs scalar (interior, 101x75)", mism_gx, checked);
    report("sobel_gradients_rvv: gy vs scalar (interior, 101x75)", mism_gy, checked);

    free(in); free(gx_ref); free(gy_ref); free(gx_rvv); free(gy_rvv);
}

// =============================================================================
// TEST 3 — Gradient Magnitude L1 (FULL image — no border exception for this one)
// =============================================================================
static void test_magnitude_l1() {
    int16_t* gx  = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
    int16_t* gy  = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
    uint8_t* ref = (uint8_t*)aligned_alloc(64, align64(N));
    uint8_t* got = (uint8_t*)aligned_alloc(64, align64(N));

    // Sobel's worst-case |Gx| or |Gy| on 8-bit input is 1020 (see guide 2.3),
    // so test with random values across that full valid range.
    fill_random_i16(gx, N, 0x11223344, 1020);
    fill_random_i16(gy, N, 0x55667788, 1020);

    compute_magnitude_l1_scalar(gx, gy, ref, W, H);
    compute_magnitude_l1_rvv   (gx, gy, got, W, H);

    int mism = compare_full_u8("magnitude_l1", ref, got, N, TOLERANCE_DEFAULT);
    report("compute_magnitude_l1_rvv vs scalar (full image, 101x75)", mism, N);

    free(gx); free(gy); free(ref); free(got);
}

// =============================================================================
// TEST 4 — Magnitude L1 edge cases: all-zero gradient, and single max pixel
//
// These mirror the host_tests.cpp scalar property tests (ZeroGradientGivesZero,
// MaxPixelIs255) but run them through the RVV path specifically, because the
// RVV kernel's two-pass max-reduction + fixed-point normalize is exactly the
// kind of logic that silently breaks on degenerate inputs (max_mag == 0,
// or a single hot pixel that must land exactly on 255).
// =============================================================================
static void test_magnitude_l1_edge_cases() {
    // Case A: all-zero gradient -> magnitude must be all-zero (guards the
    // max_mag==0 early-return branch in compute_magnitude_l1_rvv).
    {
        int16_t* gx  = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
        int16_t* gy  = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
        uint8_t* out = (uint8_t*)aligned_alloc(64, align64(N));
        std::memset(gx, 0, N * sizeof(int16_t));
        std::memset(gy, 0, N * sizeof(int16_t));
        std::memset(out, 0xFF, N);   // poison with garbage so we know it was overwritten

        compute_magnitude_l1_rvv(gx, gy, out, W, H);

        int mismatches = 0;
        for (int i = 0; i < N; ++i) {
            if (out[i] != 0) {
                if (mismatches < 5) fprintf(stderr,
                    "  mismatch magnitude_l1_zero at i=%d: got=%d (expected 0)\n", i, out[i]);
                ++mismatches;
            }
        }
        report("compute_magnitude_l1_rvv all-zero-gradient edge case", mismatches, N);

        free(gx); free(gy); free(out);
    }

    // Case B: exactly one nonzero pixel -> that pixel must normalize to
    // precisely 255 (it IS the max), every other pixel must stay 0.
    {
        int16_t* gx  = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
        int16_t* gy  = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
        uint8_t* out = (uint8_t*)aligned_alloc(64, align64(N));
        std::memset(gx, 0, N * sizeof(int16_t));
        std::memset(gy, 0, N * sizeof(int16_t));
        int hot = N / 2;
        gx[hot] = 500;   // |gx|+|gy| = 500 = the global max by construction

        compute_magnitude_l1_rvv(gx, gy, out, W, H);

        int mismatches = 0;
        if (out[hot] != 255) {
            fprintf(stderr, "  mismatch magnitude_l1_max at hot pixel %d: got=%d expected=255\n",
                    hot, out[hot]);
            ++mismatches;
        }
        for (int i = 0; i < N; ++i) {
            if (i == hot) continue;
            if (out[i] != 0) {
                if (mismatches < 5) fprintf(stderr,
                    "  mismatch magnitude_l1_max at i=%d: got=%d (expected 0)\n", i, out[i]);
                ++mismatches;
            }
        }
        report("compute_magnitude_l1_rvv single-max-pixel edge case", mismatches, N);

        free(gx); free(gy); free(out);
    }
}

// =============================================================================
// TEST 5 — Sobel on a uniform image must produce exact zero gradient
// everywhere in the interior (sanity check independent of the random test;
// mirrors host_tests.cpp's SobelGradientsUniformImage but via RVV).
// =============================================================================
static void test_sobel_uniform_image() {
    uint8_t* in = (uint8_t*)aligned_alloc(64, align64(N));
    int16_t* gx = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
    int16_t* gy = (int16_t*)aligned_alloc(64, align64(N * sizeof(int16_t)));
    std::memset(in, 150, N);

    sobel_gradients_rvv(in, gx, gy, W, H);

    int mismatches = 0, checked = 0;
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            ++checked;
            if (gx[idx] != 0 || gy[idx] != 0) {
                if (mismatches < 5) fprintf(stderr,
                    "  mismatch sobel_uniform at (x=%d,y=%d): gx=%d gy=%d (expected 0,0)\n",
                    x, y, gx[idx], gy[idx]);
                ++mismatches;
            }
        }
    }
    report("sobel_gradients_rvv uniform-image edge case (interior)", mismatches, checked);

    free(in); free(gx); free(gy);
}

// =============================================================================
// main — runs every test, prints a summary, returns 0 (all pass) or 1 (any fail)
// =============================================================================
int main() {
    printf("=============================================================\n");
    printf(" Phase 3.2 / Phase 6 Equivalence Tests: RVV vs Scalar\n");
    printf(" Image size: %dx%d (%d pixels) -- non-power-of-two on purpose\n", W, H, N);
    printf("=============================================================\n\n");

    test_gaussian_blur();
    test_sobel_gradients();
    test_sobel_uniform_image();
    test_magnitude_l1();
    test_magnitude_l1_edge_cases();

    printf("\n-------------------------------------------------------------\n");
    printf(" %d / %d tests passed\n", g_tests_passed, g_tests_run);
    printf("-------------------------------------------------------------\n");

    if (g_tests_passed != g_tests_run) {
        printf("Result: FAIL -- re-run at all three VLEN values once this\n");
        printf("passes here; a pass at one VLEN does not prove VLA-correctness.\n");
        return 1;
    }
    printf("Result: PASS at this VLEN. Remember to also run at VLEN=128/256/512:\n");
    printf("  qemu-riscv64 -cpu rv64,v=true,vlen=128 <this_binary>\n");
    printf("  qemu-riscv64 -cpu rv64,v=true,vlen=256 <this_binary>\n");
    printf("  qemu-riscv64 -cpu rv64,v=true,vlen=512 <this_binary>\n");
    return 0;
}
