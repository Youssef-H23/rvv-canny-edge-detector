#ifndef CANNY_SCALAR_H
#define CANNY_SCALAR_H

#include <cstdint>
#include <cstdlib>

// ============================================================================
// Memory alignment helper.
//   Input  : a size in bytes (n)
//   Op     : rounds n up to the next multiple of 64
//   Output : aligned size suitable for aligned_alloc(64, ...)
// Required because some RVV vector loads/stores assume 64-byte aligned buffers.
// ============================================================================
inline size_t align64(size_t n) {
    return (n + 63) & ~static_cast<size_t>(63);   // clear low 6 bits = round up to 64
}

// ============================================================================
// IMAGE I/O
// ============================================================================

// Loads a raw grayscale image from disk.
//   Input  : file path, image width, image height
//   Op     : reads exactly width*height bytes (1 byte = 1 pixel, no header)
//   Output : pointer to a 64-byte aligned buffer (caller must free), nullptr on error
uint8_t* load_raw_image(const char* path, int width, int height);

// Saves a raw grayscale image to disk.
//   Input  : file path, pixel buffer, width, height
//   Op     : writes exactly width*height bytes to disk (no header)
//   Output : true on success, false on file error
bool save_raw_image(const char* path, const uint8_t* data, int width, int height);

// ============================================================================
// GENERIC 2D CONVOLUTION (template)
//
//   Input  : input image, kernel coefficients, kernel size, divisor
//   Op     : for each pixel, multiply-accumulate the ksize*ksize neighborhood
//            with the kernel, divide by divisor, clamp to [0, 255]
//   Output : convolved image written to 'output'
//
// Template parameters:
//   PixelT  - pixel type              (uint8_t for 8-bit grayscale)
//   AccumT  - accumulator type        (int32_t to avoid overflow while accumulating)
//   KernelT - kernel coefficient type (int16_t for integer kernels)
//
// Boundary handling: zero-padding. Out-of-bounds pixels are treated as 0,
// which keeps the boundary condition uniform and simplifies later vectorization.
//
// ============================================================================
template<typename PixelT, typename AccumT, typename KernelT>
void convolve2D(const PixelT* input, PixelT* output,
                int width, int height,
                const KernelT* kernel, int ksize,
                AccumT divisor);

// Forces the standard grayscale instantiation to be compiled in canny_scalar.cpp
// so other translation units (main, tests) can link against it.
extern template void convolve2D<uint8_t, int32_t, int16_t>(
    const uint8_t*, uint8_t*, int, int, const int16_t*, int, int32_t);

// ============================================================================
// STAGE 1 - GAUSSIAN BLUR
//
//   Input  : grayscale image (width*height bytes)
//   Op     : 2D convolution with a 5x5 integer Gaussian kernel (coeffs sum to 273),
//            divide by 273, clamp to [0, 255]
//   Output : blurred grayscale image
//
// Output fits in uint8_t: max sum = 255 * 273 = 69615, divided by 273 = 255.
// Thin wrapper that calls convolve2D with the Gaussian kernel.
// ============================================================================
void gaussian_blur_scalar(const uint8_t* input, uint8_t* output,
                          int width, int height);

#endif  // CANNY_SCALAR_H
