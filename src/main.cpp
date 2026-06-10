#include "../headers/canny_scalar.h"

#include <cstdlib>   // atoi, aligned_alloc, free
#include <cstdio>    // printf, fprintf



int main(int argc, char* argv[]) {

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.raw> <width> <height>\n", argv[0]);
        fprintf(stderr, "Example: %s images/input.raw 512 512\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    int width  = atoi(argv[2]);
    int height = atoi(argv[3]);
    int total  = width * height;   // number of pixels

    printf("=== Canny Edge Detection - Phase 2 Scalar Pipeline ===\n");
    printf("Image: %s  |  Size: %dx%d\n\n", input_path, width, height);

    uint8_t* input = load_raw_image(input_path, width, height);
    if (!input) {
        fprintf(stderr, "Error: could not open '%s'\n", input_path);
        return 1;
    }

    // All buffers are 64-byte aligned for RVV compatibility in later phases.
    uint8_t* blurred = (uint8_t*)aligned_alloc(64, align64(total));

    int cx = width / 2, cy = height / 2;   // center pixel, used for sanity prints

    // ---- Stage 1: Gaussian blur ----
    printf("[Stage 1]  Gaussian blur (5x5, zero-padding)\n");
    gaussian_blur_scalar(input, blurred, width, height);
    printf("           center pixel: input=%d  blurred=%d\n",
           input[cy * width + cx], blurred[cy * width + cx]);

    // ---- Save output ----
    save_raw_image("output_blurred.raw", blurred, width, height);
    printf("output_blurred.raw\n");

    free(input); free(blurred);

    printf("\nDone.\n");
    return 0;
}
