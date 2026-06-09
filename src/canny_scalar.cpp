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
