#include <gtest/gtest.h>
#include <vector>
#include "../headers/canny_scalar.h"

// STAGE 1: GAUSSIAN BLUR TESTS

// the same uniform image. We use 16x16 so interior pixels are far enough
// from the border that zero-padding does not affect them.
TEST(CannyPipelineTest, GaussianBlurUniformImage) {
    int width = 16, height = 16;

    std::vector<uint8_t> input_image(width * height, 128);
    std::vector<uint8_t> output_image(width * height, 0);

    gaussian_blur_scalar(input_image.data(), output_image.data(), width, height);

    // Check every interior pixel (skip 2-pixel border = kernel radius)
    for (int y = 2; y < height - 2; ++y)
        for (int x = 2; x < width - 2; ++x)
            EXPECT_NEAR(output_image[y * width + x], 128, 1)
                << "at (" << x << "," << y << ")";
}


TEST(CannyPipelineTest, GaussianBlurBlackImage) {
    int width = 16, height = 16;

    std::vector<uint8_t> input_image(width * height, 0);
    std::vector<uint8_t> output_image(width * height, 99);  // garbage fill to catch non-writes

    gaussian_blur_scalar(input_image.data(), output_image.data(), width, height);

    for (int i = 0; i < width * height; ++i)
        EXPECT_EQ(output_image[i], 0) << "at pixel " << i;
}


// symmetrically (the kernel is symmetric so the output must be too).
TEST(CannyPipelineTest, GaussianBlurImpulseSymmetric) {
    int width = 15, height = 15;
    int cx = width / 2, cy = height / 2;

    std::vector<uint8_t> input_image(width * height, 0);
    std::vector<uint8_t> output_image(width * height, 0);

    input_image[cy * width + cx] = 255;  // single bright pixel at centre

    gaussian_blur_scalar(input_image.data(), output_image.data(), width, height);

    // Check four-fold symmetry around the centre for the 5x5 spread
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            uint8_t v00 = output_image[(cy + dy) * width + (cx + dx)];
            uint8_t v01 = output_image[(cy + dy) * width + (cx - dx)];
            uint8_t v10 = output_image[(cy - dy) * width + (cx + dx)];
            uint8_t v11 = output_image[(cy - dy) * width + (cx - dx)];
            EXPECT_EQ(v00, v01) << "H-symmetry failed dy=" << dy << " dx=" << dx;
            EXPECT_EQ(v00, v10) << "V-symmetry failed dy=" << dy << " dx=" << dx;
            EXPECT_EQ(v00, v11) << "Diagonal-symmetry failed dy=" << dy << " dx=" << dx;
        }
    }
}
 
 
// STAGE 2: SOBEL GRADIENTS TESTS


TEST(CannyPipelineTest, SobelGradientsUniformImage) {
    int width = 16, height = 16;

    std::vector<uint8_t> blurred_image(width * height, 150);
    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);

    sobel_gradients_scalar(blurred_image.data(), gx.data(), gy.data(), width, height);

    // Check interior pixels only (skip 1-pixel border = Sobel kernel radius)
    for (int y = 1; y < height - 1; ++y)
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            EXPECT_EQ(gx[idx], 0) << "gx non-zero at (" << x << "," << y << ")";
            EXPECT_EQ(gy[idx], 0) << "gy non-zero at (" << x << "," << y << ")";
        }
}


TEST(CannyPipelineTest, SobelGradientsVerticalEdge) {
    int width = 20, height = 20;

    std::vector<uint8_t> input_image(width * height, 0);
    for (int y = 0; y < height; ++y)
        for (int x = width / 2; x < width; ++x)
            input_image[y * width + x] = 255;

    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);

    sobel_gradients_scalar(input_image.data(), gx.data(), gy.data(), width, height);

    // At the edge column, gx should be large and gy should be ~0
    int ex = width / 2;
    for (int y = 2; y < height - 2; ++y) {
        int idx = y * width + ex;
        EXPECT_GT(gx[idx], 500)     << "Expected large gx at vertical edge, row " << y;
        EXPECT_NEAR(gy[idx], 0, 4)  << "Expected near-zero gy at vertical edge, row " << y;
    }
}

TEST(CannyPipelineTest, SobelGradientsHorizontalEdge) {
    int width = 20, height = 20;

    std::vector<uint8_t> input_image(width * height, 0);
    for (int y = height / 2; y < height; ++y)
        for (int x = 0; x < width; ++x)
            input_image[y * width + x] = 255;

    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);

    sobel_gradients_scalar(input_image.data(), gx.data(), gy.data(), width, height);

    int ey = height / 2;
    for (int x = 2; x < width - 2; ++x) {
        int idx = ey * width + x;
        EXPECT_GT(std::abs((int)gy[idx]), 500) << "Expected large |gy| at horizontal edge, col " << x;
        EXPECT_NEAR(gx[idx], 0, 4)             << "Expected near-zero gx at horizontal edge, col " << x;
    }
}


TEST(CannyPipelineTest, SobelGradientsDiagonalEdge) {
    int width = 20, height = 20;

    std::vector<uint8_t> input_image(width * height, 0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if (x + y >= width)
                input_image[y * width + x] = 255;

    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);

    sobel_gradients_scalar(input_image.data(), gx.data(), gy.data(), width, height);

    // Along the diagonal both components should be nonzero
    int found = 0;
    for (int y = 2; y < height - 2; ++y) {
        int x = width - y;
        if (x < 2 || x >= width - 2) continue;
        int idx = y * width + x;
        if (std::abs((int)gx[idx]) > 100 && std::abs((int)gy[idx]) > 100)
            ++found;
    }
    EXPECT_GT(found, 3) << "Expected multiple diagonal pixels with both gx and gy nonzero";
}



// ============================================================================
// STAGE 3a: MAGNITUDE L1 TESTS
// ============================================================================

// Requirement: L1 gives nonzero output on a random image and does not produce all-zeros.
TEST(CannyPipelineTest, MagnitudeL1NonzeroOnRandomInput) {
    int width = 10, height = 10;

    std::vector<int16_t> gx(width * height);
    std::vector<int16_t> gy(width * height);
    std::vector<uint8_t> magnitude(width * height, 0);

    for (int i = 0; i < width * height; ++i) {
        gx[i] = (int16_t)((i * 37 + 13) % 1020 - 510);
        gy[i] = (int16_t)((i * 53 +  7) % 1020 - 510);
    }

    compute_magnitude_l1_scalar(gx.data(), gy.data(), magnitude.data(), width, height);

    bool has_nonzero = false;
    for (int i = 0; i < width * height; ++i)
        if (magnitude[i] > 0) { has_nonzero = true; break; }

    EXPECT_TRUE(has_nonzero) << "L1 magnitude produced all-zeros on a non-zero gradient image";
}

// Requirement: Output is always in [0, 255] (no overflow or underflow).
TEST(CannyPipelineTest, MagnitudeL1OutputClamped) {
    int width = 10, height = 10;

    std::vector<int16_t> gx(width * height);
    std::vector<int16_t> gy(width * height);
    std::vector<uint8_t> magnitude(width * height, 0);

    for (int i = 0; i < width * height; ++i) {
        gx[i] = (i % 2 == 0) ?  1020 : -1020;
        gy[i] = (i % 3 == 0) ?   512 :     0;
    }

    compute_magnitude_l1_scalar(gx.data(), gy.data(), magnitude.data(), width, height);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_GE(magnitude[i],   0) << "underflow at pixel " << i;
        EXPECT_LE(magnitude[i], 255) << "overflow at pixel "  << i;
    }
}

// Requirement: Zero gradient -> zero magnitude.
TEST(CannyPipelineTest, MagnitudeL1ZeroGradientGivesZero) {
    int width = 10, height = 10;

    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);
    std::vector<uint8_t> magnitude(width * height, 0);

    compute_magnitude_l1_scalar(gx.data(), gy.data(), magnitude.data(), width, height);

    for (int i = 0; i < width * height; ++i)
        EXPECT_EQ(magnitude[i], 0) << "at pixel " << i;
}

// Requirement: The pixel with the largest gradient must map to 255.
TEST(CannyPipelineTest, MagnitudeL1MaxPixelIs255) {
    int width = 10, height = 10;

    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);
    std::vector<uint8_t> magnitude(width * height, 0);

    gx[50] = 500;

    compute_magnitude_l1_scalar(gx.data(), gy.data(), magnitude.data(), width, height);

    EXPECT_EQ(magnitude[50], 255) << "pixel with max gradient must map to 255";
    for (int i = 0; i < width * height; ++i)
        if (i != 50)
            EXPECT_EQ(magnitude[i], 0) << "zero-gradient pixel " << i << " must be 0";
}

// ============================================================================
// STAGE 3a: MAGNITUDE L2 TESTS
// ============================================================================

// Requirement: L2 gives nonzero output on a random image and does not produce all-zeros.
TEST(CannyPipelineTest, MagnitudeL2NonzeroOnRandomInput) {
    int width = 10, height = 10;

    std::vector<int16_t> gx(width * height);
    std::vector<int16_t> gy(width * height);
    std::vector<uint8_t> magnitude(width * height, 0);

    for (int i = 0; i < width * height; ++i) {
        gx[i] = (int16_t)((i * 37 + 13) % 1020 - 510);
        gy[i] = (int16_t)((i * 53 +  7) % 1020 - 510);
    }

    compute_magnitude_l2_scalar(gx.data(), gy.data(), magnitude.data(), width, height);

    bool has_nonzero = false;
    for (int i = 0; i < width * height; ++i)
        if (magnitude[i] > 0) { has_nonzero = true; break; }

    EXPECT_TRUE(has_nonzero) << "L2 magnitude produced all-zeros on a non-zero gradient image";
}

// Requirement: Output is always in [0, 255].
TEST(CannyPipelineTest, MagnitudeL2OutputClamped) {
    int width = 10, height = 10;

    std::vector<int16_t> gx(width * height);
    std::vector<int16_t> gy(width * height);
    std::vector<uint8_t> magnitude(width * height, 0);

    for (int i = 0; i < width * height; ++i) {
        gx[i] = (int16_t)(i * 20 % 1020);
        gy[i] = (int16_t)(i *  5 % 1020);
    }

    compute_magnitude_l2_scalar(gx.data(), gy.data(), magnitude.data(), width, height);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_GE(magnitude[i],   0) << "underflow at pixel " << i;
        EXPECT_LE(magnitude[i], 255) << "overflow at pixel "  << i;
    }
}

// Requirement: Zero gradient -> zero magnitude.
TEST(CannyPipelineTest, MagnitudeL2ZeroGradientGivesZero) {
    int width = 10, height = 10;

    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);
    std::vector<uint8_t> magnitude(width * height, 0);

    compute_magnitude_l2_scalar(gx.data(), gy.data(), magnitude.data(), width, height);

    for (int i = 0; i < width * height; ++i)
        EXPECT_EQ(magnitude[i], 0) << "at pixel " << i;
}

// Requirement: The pixel with the largest gradient must map to 255.
TEST(CannyPipelineTest, MagnitudeL2MaxPixelIs255) {
    int width = 10, height = 10;

    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);
    std::vector<uint8_t> magnitude(width * height, 0);

    gx[50] = 300;
    gy[50] = 400;   // L2 = 500, only pixel 50 is nonzero

    compute_magnitude_l2_scalar(gx.data(), gy.data(), magnitude.data(), width, height);

    EXPECT_EQ(magnitude[50], 255) << "pixel with max gradient must map to 255";
    for (int i = 0; i < width * height; ++i)
        if (i != 50)
            EXPECT_EQ(magnitude[i], 0) << "zero-gradient pixel " << i << " must be 0";
}




// ============================================================================
// STAGE 3b: GRADIENT DIRECTION TESTS
// ============================================================================

// Requirement: Vertical edge image -> direction = 0 (horizontal gradient).
TEST(CannyPipelineTest, DirectionVerticalEdgeGivesDir0) {
    int width = 20, height = 20;

    std::vector<uint8_t> input(width * height, 0);
    std::vector<uint8_t> blurred(width * height, 0);
    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);
    std::vector<uint8_t> direction(width * height, 0);

    // Left half black, right half white -> vertical edge at x = width/2
    for (int y = 0; y < height; ++y)
        for (int x = width / 2; x < width; ++x)
            input[y * width + x] = 255;

    gaussian_blur_scalar(input.data(), blurred.data(), width, height);
    sobel_gradients_scalar(blurred.data(), gx.data(), gy.data(), width, height);
    compute_direction_scalar(gx.data(), gy.data(), direction.data(), width, height);

    // At the edge column, direction must be 0 (gradient is horizontal)
    int ex = width / 2;
    for (int y = 3; y < height - 3; ++y)
        EXPECT_EQ(direction[y * width + ex], 0)
            << "Expected direction=0 at vertical edge, row " << y;
}

// Requirement: Horizontal edge image -> direction = 90 (vertical gradient).
TEST(CannyPipelineTest, DirectionHorizontalEdgeGivesDir90) {
    int width = 20, height = 20;

    std::vector<uint8_t> input(width * height, 0);
    std::vector<uint8_t> blurred(width * height, 0);
    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);
    std::vector<uint8_t> direction(width * height, 0);

    // Top half black, bottom half white -> horizontal edge at y = height/2
    for (int y = height / 2; y < height; ++y)
        for (int x = 0; x < width; ++x)
            input[y * width + x] = 255;

    gaussian_blur_scalar(input.data(), blurred.data(), width, height);
    sobel_gradients_scalar(blurred.data(), gx.data(), gy.data(), width, height);
    compute_direction_scalar(gx.data(), gy.data(), direction.data(), width, height);

    // At the edge row, direction must be 90 (gradient is vertical)
    int ey = height / 2;
    for (int x = 3; x < width - 3; ++x)
        EXPECT_EQ(direction[ey * width + x], 90)
            << "Expected direction=90 at horizontal edge, col " << x;
}

// Requirement: Diagonal edge -> direction = 45 or 135.
TEST(CannyPipelineTest, DirectionDiagonalEdgeGivesDir45Or135) {
    int width = 20, height = 20;

    std::vector<uint8_t> input(width * height, 0);
    std::vector<uint8_t> blurred(width * height, 0);
    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);
    std::vector<uint8_t> direction(width * height, 0);

    // Top-left black, bottom-right white -> diagonal edge
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if (x + y >= width)
                input[y * width + x] = 255;

    gaussian_blur_scalar(input.data(), blurred.data(), width, height);
    sobel_gradients_scalar(blurred.data(), gx.data(), gy.data(), width, height);
    compute_direction_scalar(gx.data(), gy.data(), direction.data(), width, height);

    // Along the diagonal, direction must be 45 or 135
    int found = 0;
    for (int y = 3; y < height - 3; ++y) {
        int x = width - y;
        if (x < 3 || x >= width - 3) continue;
        int d = direction[y * width + x];
        if (d == 45 || d == 135) ++found;
    }
    EXPECT_GT(found, 2) << "Expected diagonal pixels with direction=45 or 135";
}

// GoogleTest Main Entry Point
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}














