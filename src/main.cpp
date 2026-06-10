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
    int16_t* gx      = (int16_t*)aligned_alloc(64, align64(total * sizeof(int16_t)));
    int16_t* gy      = (int16_t*)aligned_alloc(64, align64(total * sizeof(int16_t)));
    uint8_t* mag_l1  = (uint8_t*)aligned_alloc(64, align64(total));
    uint8_t* mag_l2  = (uint8_t*)aligned_alloc(64, align64(total));
    uint8_t* dir     = (uint8_t*)aligned_alloc(64, align64(total));

    int cx = width / 2, cy = height / 2;   // center pixel, used for sanity prints

    // ---- Stage 1: Gaussian blur ----
    printf("[Stage 1]  Gaussian blur (5x5, zero-padding)\n");
    gaussian_blur_scalar(input, blurred, width, height);
    printf("           center pixel: input=%d  blurred=%d\n",
           input[cy * width + cx], blurred[cy * width + cx]);

    // ---- Save output ----
    save_raw_image("output_blurred.raw", blurred, width, height);
    printf("output_blurred.raw\n");

    // ---- Stage 2: Sobel gradients ----
    printf("[Stage 2]  Sobel gradients (3x3, SoA layout)\n");
    sobel_gradients_scalar(blurred, gx, gy, width, height);
    printf("           center gx=%d  gy=%d\n",
           gx[cy * width + cx], gy[cy * width + cx]);

    // ---- Stage 3a: Magnitude (both norms, for comparison) ----
    printf("[Stage 3a] Gradient magnitude L1 (|gx|+|gy|)\n");
    compute_magnitude_l1_scalar(gx, gy, mag_l1, width, height);

    printf("[Stage 3a] Gradient magnitude L2 (sqrt(gx^2+gy^2))\n");
    compute_magnitude_l2_scalar(gx, gy, mag_l2, width, height);

    long diff_sum = 0;
    for (int i = 0; i < total; ++i)
        diff_sum += abs((int)mag_l1[i] - (int)mag_l2[i]);
    printf("           avg L1 vs L2 difference: %.2f\n", (double)diff_sum / total);

    // ---- Stage 3b: Direction ----
    printf("[Stage 3b] Gradient direction (0/45/90/135)\n");
    compute_direction_scalar(gx, gy, dir, width, height);
    int c0 = 0, c45 = 0, c90 = 0, c135 = 0;
    for (int i = 0; i < total; ++i) {
        if      (dir[i] == 0)   c0++;
        else if (dir[i] == 45)  c45++;
        else if (dir[i] == 90)  c90++;
        else                     c135++;
    }
    printf("           distribution -> 0:%d  45:%d  90:%d  135:%d\n",
           c0, c45, c90, c135);

    // ---- Save outputs ----
    save_raw_image("output_magnitude_l1.raw", mag_l1, width, height);
    save_raw_image("output_magnitude_l2.raw", mag_l2, width, height);
    save_raw_image("output_direction.raw",    dir,    width, height);
    printf("output_magnitude_l1.raw / magnitude_l2 / direction\n");

    free(input); free(blurred); free(gx); free(gy);
    free(mag_l1); free(mag_l2); free(dir);

    printf("\nDone.\n");
    return 0;
}
