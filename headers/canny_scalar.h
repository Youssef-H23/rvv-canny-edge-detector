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

#endif  // CANNY_SCALAR_H
