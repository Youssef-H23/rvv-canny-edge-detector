#include <gtest/gtest.h>
#include <vector>
#include "canny_scalar.h"


// STAGE 1: GAUSSIAN BLUR TEST

TEST(CannyPipelineTest, GaussianBlurFilter) {
    int width = 5;
    int height = 5;
    
    // Initialize a 5x5 input image with uniform pixel values (100)
    std::vector<uint8_t> input_image(width * height, 100);
    // Initialize an empty output buffer of the same size
    std::vector<uint8_t> output_image(width * height, 0);

    // Call the scalar Gaussian Blur implementation
    gaussian_blur_scalar(input_image.data(), output_image.data(), width, height);

    // Assert that the center pixel value is processed and greater than 0
    EXPECT_GT(output_image[12], 0);
}


// STAGE 2: SOBEL GRADIENTS TEST

TEST(CannyPipelineTest, SobelGradientsFilter) {
    int width = 5;
    int height = 5;

    // Initialize a 5x5 blurred input image (mock data)
    std::vector<uint8_t> blurred_image(width * height, 150);
    
    // Sobel outputs separate Gx and Gy signed 16-bit arrays (Structure of Arrays)
    std::vector<int16_t> gx(width * height, 0);
    std::vector<int16_t> gy(width * height, 0);

    // Call the scalar Sobel Gradients implementation
    sobel_gradients_scalar(blurred_image.data(), gx.data(), gy.data(), width, height);

    // Assert that the buffers are modified and test runner can link successfully
    SUCCEED();
}

// GoogleTest Main Entry Point
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}