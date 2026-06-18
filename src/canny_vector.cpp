#include <riscv_vector.h>
#include <cstdint>
#include <cstring> // For std::memset

// ============================================================================
// STAGE 1 - GAUSSIAN BLUR (RISC-V VECTORIZED)
// ============================================================================
// We keep the name "gaussian_blur_scalar" so the Makefile and tests 
// can use this file without needing to change their code!
void gaussian_blur_scalar(const uint8_t* input, uint8_t* output, int width, int height) {
    
    // 1. CLEAR THE CANVAS
    // The 5x5 kernel cannot safely process the outer 2-pixel border of the image.
    // We paint the entire output black (0) first to handle the borders.
    std::memset(output, 0, width * height);

    // 2. THE Y-AXIS SLIDING WINDOW
    // Start at row 2 and stop 2 rows before the bottom to prevent vertical crashing.
    for (int y = 2; y < height - 2; ++y) {
        
        // --- SET UP THE 5 VERTICAL POINTERS ---
        const uint8_t* row0 = input + ((y - 2) * width); // 2 rows above
        const uint8_t* row1 = input + ((y - 1) * width); // 1 row above
        const uint8_t* row2 = input + ((y - 0) * width); // TARGET ROW (Center)
        const uint8_t* row3 = input + ((y + 1) * width); // 1 row below
        const uint8_t* row4 = input + ((y + 2) * width); // 2 rows below

        // Set up the pointer where we will write the finished blurred pixels
        uint8_t* out_row = output + (y * width);

        // We skip the 2 leftmost and 2 rightmost pixels of the row to prevent horizontal crashing.
        int pixels_left = width - 4; 
        
        // Advance all pointers by 2 spaces to the right to skip the left border
        row0 += 2; row1 += 2; row2 += 2; row3 += 2; row4 += 2; out_row += 2;

        // 3. THE X-AXIS STRIP-MINING LOOP (THE BULLDOZER)
        while (pixels_left > 0) {
            
            // Ask hardware: "How many 8-bit pixels can you scoop right now?"
            size_t vl = __riscv_vsetvl_e8m1(pixels_left);

            // Create a giant 32-bit Vector Accumulator filled with 0s. 
            // We use 32-bit because multiplying 255 by 41 will overflow an 8-bit box!
            vuint32m4_t vec_sum = __riscv_vmv_v_x_u32m4(0, vl);

            // ====================================================================
            // THE MATH: ROW 0 (Kernel Weights: 1, 4, 7, 4, 1)
            // ====================================================================
            // Step A: Load the overlapping 8-bit scoops
            vuint8m1_t r0_m2 = __riscv_vle8_v_u8m1(row0 - 2, vl); // Far Left
            vuint8m1_t r0_m1 = __riscv_vle8_v_u8m1(row0 - 1, vl); // Mid Left
            vuint8m1_t r0_0  = __riscv_vle8_v_u8m1(row0 - 0, vl); // Center
            vuint8m1_t r0_p1 = __riscv_vle8_v_u8m1(row0 + 1, vl); // Mid Right
            vuint8m1_t r0_p2 = __riscv_vle8_v_u8m1(row0 + 2, vl); // Far Right

            // Step B: Widen pixels to 16-bit, multiply by weight, and add to the 32-bit sum!
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 1, __riscv_vwcvtu_x_x_v_u16m2(r0_m2, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 4, __riscv_vwcvtu_x_x_v_u16m2(r0_m1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 7, __riscv_vwcvtu_x_x_v_u16m2(r0_0,  vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 4, __riscv_vwcvtu_x_x_v_u16m2(r0_p1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 1, __riscv_vwcvtu_x_x_v_u16m2(r0_p2, vl), vl);


            // ====================================================================
            // THE MATH: ROW 1 (Kernel Weights: 4, 16, 26, 16, 4)
            // ====================================================================
            vuint8m1_t r1_m2 = __riscv_vle8_v_u8m1(row1 - 2, vl);
            vuint8m1_t r1_m1 = __riscv_vle8_v_u8m1(row1 - 1, vl);
            vuint8m1_t r1_0  = __riscv_vle8_v_u8m1(row1 - 0, vl);
            vuint8m1_t r1_p1 = __riscv_vle8_v_u8m1(row1 + 1, vl);
            vuint8m1_t r1_p2 = __riscv_vle8_v_u8m1(row1 + 2, vl);

            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 4,  __riscv_vwcvtu_x_x_v_u16m2(r1_m2, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 16, __riscv_vwcvtu_x_x_v_u16m2(r1_m1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 26, __riscv_vwcvtu_x_x_v_u16m2(r1_0,  vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 16, __riscv_vwcvtu_x_x_v_u16m2(r1_p1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 4,  __riscv_vwcvtu_x_x_v_u16m2(r1_p2, vl), vl);


            // ====================================================================
            // THE MATH: ROW 2 - TARGET ROW (Kernel Weights: 7, 26, 41, 26, 7)
            // ====================================================================
            vuint8m1_t r2_m2 = __riscv_vle8_v_u8m1(row2 - 2, vl);
            vuint8m1_t r2_m1 = __riscv_vle8_v_u8m1(row2 - 1, vl);
            vuint8m1_t r2_0  = __riscv_vle8_v_u8m1(row2 - 0, vl);
            vuint8m1_t r2_p1 = __riscv_vle8_v_u8m1(row2 + 1, vl);
            vuint8m1_t r2_p2 = __riscv_vle8_v_u8m1(row2 + 2, vl);

            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 7,  __riscv_vwcvtu_x_x_v_u16m2(r2_m2, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 26, __riscv_vwcvtu_x_x_v_u16m2(r2_m1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 41, __riscv_vwcvtu_x_x_v_u16m2(r2_0,  vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 26, __riscv_vwcvtu_x_x_v_u16m2(r2_p1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 7,  __riscv_vwcvtu_x_x_v_u16m2(r2_p2, vl), vl);


            // ====================================================================
            // THE MATH: ROW 3 (Kernel Weights: 4, 16, 26, 16, 4)
            // ====================================================================
            vuint8m1_t r3_m2 = __riscv_vle8_v_u8m1(row3 - 2, vl);
            vuint8m1_t r3_m1 = __riscv_vle8_v_u8m1(row3 - 1, vl);
            vuint8m1_t r3_0  = __riscv_vle8_v_u8m1(row3 - 0, vl);
            vuint8m1_t r3_p1 = __riscv_vle8_v_u8m1(row3 + 1, vl);
            vuint8m1_t r3_p2 = __riscv_vle8_v_u8m1(row3 + 2, vl);

            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 4,  __riscv_vwcvtu_x_x_v_u16m2(r3_m2, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 16, __riscv_vwcvtu_x_x_v_u16m2(r3_m1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 26, __riscv_vwcvtu_x_x_v_u16m2(r3_0,  vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 16, __riscv_vwcvtu_x_x_v_u16m2(r3_p1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 4,  __riscv_vwcvtu_x_x_v_u16m2(r3_p2, vl), vl);


            // ====================================================================
            // THE MATH: ROW 4 (Kernel Weights: 1, 4, 7, 4, 1)
            // ====================================================================
            vuint8m1_t r4_m2 = __riscv_vle8_v_u8m1(row4 - 2, vl);
            vuint8m1_t r4_m1 = __riscv_vle8_v_u8m1(row4 - 1, vl);
            vuint8m1_t r4_0  = __riscv_vle8_v_u8m1(row4 - 0, vl);
            vuint8m1_t r4_p1 = __riscv_vle8_v_u8m1(row4 + 1, vl);
            vuint8m1_t r4_p2 = __riscv_vle8_v_u8m1(row4 + 2, vl);

            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 1, __riscv_vwcvtu_x_x_v_u16m2(r4_m2, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 4, __riscv_vwcvtu_x_x_v_u16m2(r4_m1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 7, __riscv_vwcvtu_x_x_v_u16m2(r4_0,  vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 4, __riscv_vwcvtu_x_x_v_u16m2(r4_p1, vl), vl);
            vec_sum = __riscv_vwmaccu_vx_u32m4(vec_sum, 1, __riscv_vwcvtu_x_x_v_u16m2(r4_p2, vl), vl);


            // 4. THE NORMALIZATION
            // The sum is currently huge. We must divide all pixels by 273 (the sum of all kernel weights).
            vuint32m4_t vec_final32 = __riscv_vdivu_vx_u32m4(vec_sum, 273, vl);

            // Convert the huge 32-bit numbers back down into small 8-bit pixels!
            // First, squeeze 32-bit into 16-bit:
            vuint16m2_t vec_final16 = __riscv_vncvt_x_x_w_u16m2(vec_final32, vl);
            // Then, squeeze 16-bit into 8-bit:
            vuint8m1_t  vec_final8  = __riscv_vncvt_x_x_w_u8m1(vec_final16, vl);

            // 5. STORE THE RESULT
            // Dump the finished 8-bit pixels back into the output row.
            __riscv_vse8_v_u8m1(out_row, vec_final8, vl);

            // 6. ADVANCE THE MACHINE
            // Move all reading glasses and the writing pen forward by 'vl' pixels
            row0 += vl;
            row1 += vl;
            row2 += vl;
            row3 += vl;
            row4 += vl;
            out_row += vl;
            
            // Subtract the chunk we just processed from the remaining count
            pixels_left -= vl;
        }
    }
}

// ============================================================================
// STAGE 2 - SOBEL GRADIENTS (RISC-V VECTORIZED)
// ============================================================================
void sobel_gradients_scalar(const uint8_t* input, int16_t* gx, int16_t* gy, int width, int height) {
    
    // 1. CLEAR THE CANVAS
    // Sobel 3x3 cannot safely process the 1-pixel outer border. 
    // Notice we use sizeof(int16_t) because gx and gy use 16-bit memory boxes!
    std::memset(gx, 0, width * height * sizeof(int16_t));
    std::memset(gy, 0, width * height * sizeof(int16_t));

    // 2. THE Y-AXIS SLIDING WINDOW (Only need to skip 1 row for 3x3)
    for (int y = 1; y < height - 1; ++y) {
        
        // --- SET UP THE 3 VERTICAL POINTERS ---
        const uint8_t* row0 = input + ((y - 1) * width); // 1 row above
        const uint8_t* row1 = input + ((y - 0) * width); // TARGET ROW
        const uint8_t* row2 = input + ((y + 1) * width); // 1 row below

        // We have TWO writing pens now!
        int16_t* gx_out = gx + (y * width);
        int16_t* gy_out = gy + (y * width);

        int pixels_left = width - 2; 
        
        // Advance by 1 to skip the left border
        row0 += 1; row1 += 1; row2 += 1; gx_out += 1; gy_out += 1;

        // 3. THE X-AXIS STRIP-MINING LOOP
        while (pixels_left > 0) {
            size_t vl = __riscv_vsetvl_e8m1(pixels_left);

            // --- STEP A: LOAD 8-BIT PIXELS ---
            vuint8m1_t r0_m1_8 = __riscv_vle8_v_u8m1(row0 - 1, vl); // Top-Left
            vuint8m1_t r0_0_8  = __riscv_vle8_v_u8m1(row0 - 0, vl); // Top-Center
            vuint8m1_t r0_p1_8 = __riscv_vle8_v_u8m1(row0 + 1, vl); // Top-Right

            vuint8m1_t r1_m1_8 = __riscv_vle8_v_u8m1(row1 - 1, vl); // Mid-Left
            // (We actually don't need to load the Center pixel! Both Sobel matrices multiply it by 0!)
            vuint8m1_t r1_p1_8 = __riscv_vle8_v_u8m1(row1 + 1, vl); // Mid-Right

            vuint8m1_t r2_m1_8 = __riscv_vle8_v_u8m1(row2 - 1, vl); // Bot-Left
            vuint8m1_t r2_0_8  = __riscv_vle8_v_u8m1(row2 - 0, vl); // Bot-Center
            vuint8m1_t r2_p1_8 = __riscv_vle8_v_u8m1(row2 + 1, vl); // Bot-Right

            // --- STEP B: WIDEN TO 16-BIT SIGNED INTEGERS ---
            // We widen the 8-bit pixels to 16-bit unsigned, then forcibly reinterpret them as Signed 16-bit.
            // (Using a quick C++ macro here to keep the code beautifully clean)
            #define WIDEN(vec8) __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwcvtu_x_x_v_u16m2(vec8, vl))

            vint16m2_t r0_m1 = WIDEN(r0_m1_8);
            vint16m2_t r0_0  = WIDEN(r0_0_8);
            vint16m2_t r0_p1 = WIDEN(r0_p1_8);

            vint16m2_t r1_m1 = WIDEN(r1_m1_8);
            vint16m2_t r1_p1 = WIDEN(r1_p1_8);

            vint16m2_t r2_m1 = WIDEN(r2_m1_8);
            vint16m2_t r2_0  = WIDEN(r2_0_8);
            vint16m2_t r2_p1 = WIDEN(r2_p1_8);
            #undef WIDEN

            // --- STEP C: CALCULATE GX (Horizontal Gradient) ---
            // Gx = (Right Column) - (Left Column)
            // Top row diff: r0_p1 - r0_m1
            vint16m2_t gx_top = __riscv_vsub_vv_i16m2(r0_p1, r0_m1, vl);
            
            // Mid row diff is multiplied by 2. We use 'vsll' (Shift Left Logical by 1) which is 2x faster than multiply!
            vint16m2_t gx_mid_diff = __riscv_vsub_vv_i16m2(r1_p1, r1_m1, vl);
            vint16m2_t gx_mid      = __riscv_vsll_vx_i16m2(gx_mid_diff, 1, vl); 
            
            // Bot row diff: r2_p1 - r2_m1
            vint16m2_t gx_bot = __riscv_vsub_vv_i16m2(r2_p1, r2_m1, vl);

            // Add them all together!
            vint16m2_t vec_gx = __riscv_vadd_vv_i16m2(__riscv_vadd_vv_i16m2(gx_top, gx_mid, vl), gx_bot, vl);


            // --- STEP D: CALCULATE GY (Vertical Gradient) ---
            // Gy = (Bottom Row) - (Top Row)
            vint16m2_t gy_left = __riscv_vsub_vv_i16m2(r2_m1, r0_m1, vl);
            
            vint16m2_t gy_mid_diff = __riscv_vsub_vv_i16m2(r2_0, r0_0, vl);
            vint16m2_t gy_mid      = __riscv_vsll_vx_i16m2(gy_mid_diff, 1, vl); // Multiply by 2 via shift
            
            vint16m2_t gy_right = __riscv_vsub_vv_i16m2(r2_p1, r0_p1, vl);

            // Add them all together!
            vint16m2_t vec_gy = __riscv_vadd_vv_i16m2(__riscv_vadd_vv_i16m2(gy_left, gy_mid, vl), gy_right, vl);


            // --- STEP E: STORE RESULTS ---
            // Dump the finished 16-bit answers back into the output rows
            __riscv_vse16_v_i16m2(gx_out, vec_gx, vl);
            __riscv_vse16_v_i16m2(gy_out, vec_gy, vl);

            // --- STEP F: ADVANCE THE MACHINE ---
            row0 += vl; row1 += vl; row2 += vl;
            gx_out += vl; gy_out += vl;
            pixels_left -= vl;
        }
    }
}