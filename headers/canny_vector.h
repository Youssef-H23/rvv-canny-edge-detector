#ifndef CANNY_VECTOR_H
#define CANNY_VECTOR_H

#include <cstdint>

// ============================================================================
// RVV-vectorized pipeline stages. Same signatures as the *_scalar versions
// in canny_scalar.h so they can be swapped in via the USE_RVV switch in
// main.cpp.
//
// gaussian_blur_rvv and compute_magnitude_l1_rvv are the two stages the
// project guide explicitly walks through (sections 6.4 and 6.5).
// sobel_gradients_rvv vectorizes Gx/Gy directly -- not a worked example in
// the guide, but justified by profiling data showing Sobel Gradients at
// ~21% of pipeline time.
// ============================================================================

void gaussian_blur_rvv(const uint8_t* input, uint8_t* output,
                       int width, int height);

void gaussian_blur_rvv_separable(const uint8_t *input, uint8_t *output, int w, int h);
void sobel_gradients_rvv(const uint8_t* input,
                         int16_t* gx, int16_t* gy,
                         int width, int height);

void compute_magnitude_l1_rvv(const int16_t* gx, const int16_t* gy,
                              uint8_t* magnitude,
                              int width, int height);

#endif  // CANNY_VECTOR_H
