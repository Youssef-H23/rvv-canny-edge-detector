#define _POSIX_C_SOURCE 199309L  
#include "../headers/canny_scalar.h"
#include <cstdlib>   // atoi, aligned_alloc, free
#include <cstdio>    // printf, fprintf, abs
#include <time.h>    // clock_gettime, CLOCK_MONOTONIC


// BARE-METAL FILE I/O FIX FOR QEMU USER-MODE

extern "C" {
    int _openat(int dirfd, const char *pathname, int flags, int mode);
    int _open(const char *pathname, int flags, int mode) {
        return _openat(-100, pathname, flags, mode); // -100 is AT_FDCWD in Linux
    }
}

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
    uint8_t* nms     = (uint8_t*)aligned_alloc(64, align64(total));
    uint8_t* labels  = (uint8_t*)aligned_alloc(64, align64(total));
    uint8_t* edges   = (uint8_t*)aligned_alloc(64, align64(total));

    int cx = width / 2, cy = height / 2;   // center pixel, used for sanity prints

    printf("--- Running Benchmark (100 Iterations) ---\n");
    
    // START TIMING
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // BENCHMARK LOOP FOR STABLE PROFILING
    for (int iter = 0; iter < 100; ++iter) {
        
        // ---- Stage 1: Gaussian blur ----
        if (iter == 0) printf("[Stage 1]  Gaussian blur (5x5, zero-padding)\n");
        gaussian_blur_scalar(input, blurred, width, height);
        if (iter == 0) {
            printf("           center pixel: input=%d  blurred=%d\n",
                   input[cy * width + cx], blurred[cy * width + cx]);
        }

        // ---- Stage 2: Sobel gradients ----
        if (iter == 0) printf("[Stage 2]  Sobel gradients (3x3, SoA layout)\n");
        sobel_gradients_scalar(blurred, gx, gy, width, height);
        if (iter == 0) {
            printf("           center gx=%d  gy=%d\n",
                   gx[cy * width + cx], gy[cy * width + cx]);
        }

        // ---- Stage 3a: Magnitude (both norms, for comparison) ----
        if (iter == 0) printf("[Stage 3a] Gradient magnitude L1 (|gx|+|gy|)\n");
        compute_magnitude_l1_scalar(gx, gy, mag_l1, width, height);

        if (iter == 0) printf("[Stage 3a] Gradient magnitude L2 (sqrt(gx^2+gy^2))\n");
        compute_magnitude_l2_scalar(gx, gy, mag_l2, width, height);

        if (iter == 0) {
            long diff_sum = 0;
            for (int i = 0; i < total; ++i)
                diff_sum += abs((int)mag_l1[i] - (int)mag_l2[i]);
            printf("           avg L1 vs L2 difference: %.2f\n", (double)diff_sum / total);
        }

        // ---- Stage 3b: Direction ----
        if (iter == 0) printf("[Stage 3b] Gradient direction (0/45/90/135)\n");
        compute_direction_scalar(gx, gy, dir, width, height);
        if (iter == 0) {
            int c0 = 0, c45 = 0, c90 = 0, c135 = 0;
            for (int i = 0; i < total; ++i) {
                if      (dir[i] == 0)   c0++;
                else if (dir[i] == 45)  c45++;
                else if (dir[i] == 90)  c90++;
                else                     c135++;
            }
            printf("           distribution -> 0:%d  45:%d  90:%d  135:%d\n",
                   c0, c45, c90, c135);
        }
         
		// The rest of the pipeline uses the L1 magnitude (the norm targeted for Phase 6).
		 
        // ---- Stage 4: Non-maximum suppression ----
        if (iter == 0) printf("[Stage 4]  Non-maximum suppression\n");
        non_maximum_suppression_scalar(mag_l1, dir, nms, width, height);

        // ---- Stage 5a: Double thresholding ----
        if (iter == 0) printf("[Stage 5a] Double thresholding (low=20, high=80)\n");
        double_threshold_scalar(nms, labels, width, height, 20, 80);
        if (iter == 0) {
            int strong = 0, weak = 0;
            for (int i = 0; i < total; ++i) {
                if      (labels[i] == EDGE_STRONG) strong++;
                else if (labels[i] == EDGE_WEAK)   weak++;
            }
            printf("           strong=%d  weak=%d\n", strong, weak);
        }

        // ---- Stage 5b: Hysteresis ----
        if (iter == 0) printf("[Stage 5b] Hysteresis edge tracing (iterative)\n");
        hysteresis_scalar(labels, edges, width, height);
        if (iter == 0) {
            int final_edges = 0;
            for (int i = 0; i < total; ++i)
                if (edges[i] == 255) final_edges++;
            printf("           final edge pixels: %d\n", final_edges);
        }
    }

    // END TIMING
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

    printf("\nExecution Time for 100 iterations: %f seconds\n", time_taken);

    // ---- Save outputs ----
    printf("\n--- Saving outputs ---\n");
    save_raw_image("output_blurred.raw",      blurred, width, height);
    save_raw_image("output_magnitude_l1.raw", mag_l1,  width, height);
    save_raw_image("output_magnitude_l2.raw", mag_l2,  width, height);
    save_raw_image("output_direction.raw",    dir,     width, height);
    save_raw_image("output_nms.raw",          nms,     width, height);
    save_raw_image("output_edges.raw",        edges,   width, height);
    printf("output_blurred.raw / magnitude_l1 / magnitude_l2 / direction / nms / edges\n");

    free(input); free(blurred); free(gx); free(gy);
    free(mag_l1); free(mag_l2); free(dir);
    free(nms); free(labels); free(edges);

    printf("\nDone.\n");
    return 0;
}




