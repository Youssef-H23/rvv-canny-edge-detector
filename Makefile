# ==========================================
# COMPILER CONFIGURATIONS
# ==========================================
HOST_CXX = g++
RV_CXX   = riscv64-unknown-elf-g++

# Compiler Flags
HOST_FLAGS = 
RV_FLAGS   = -march=rv64gcv -mabi=lp64d

# GoogleTest Paths
GTEST_INC  = -I$(HOME)/.local/include
GTEST_LIBS = -L$(HOME)/.local/lib -lgtest -lgtest_main -pthread

# ==========================================
# SEPARATE OUTPUT DIRECTORIES
# ==========================================
HOST_BUILD_DIR = build_host
RV_BUILD_DIR   = build_rv

# Source files
SRC_FILES   = main.cpp
TEST_FILES  = host_tests.cpp


# ==========================================
# QEMU CONFIGURATIONS
# ==========================================

VLEN = 128  # Set VLEN to 128 bits for RISC-V Vector Extension

# ==========================================
.PHONY: all test canny_rv run clean

all: test canny_rv

# --- RULE 1: make test (Host-side Native Testing) ---
test: $(TEST_FILES)
	@mkdir -p $(HOST_BUILD_DIR)
	$(HOST_CXX) $(HOST_FLAGS) $(GTEST_INC) $(TEST_FILES) $(GTEST_LIBS) -o $(HOST_BUILD_DIR)/host_tests
	@echo "--- Running Host-side GoogleTest ---"
	./$(HOST_BUILD_DIR)/host_tests

# --- RULE 2: make canny_rv (RISC-V Cross-Compilation) ---
canny_rv: $(SRC_FILES)
	@mkdir -p $(RV_BUILD_DIR)
	$(RV_CXX) $(RV_FLAGS) $(SRC_FILES) -o $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- RISC-V Binary Successfully Compiled ---"

# --- RULE 3: make run (Execute on QEMU Emulator) ---
run: $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- Launching execution on QEMU (VLEN=$(VLEN)) ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) $(RV_BUILD_DIR)/canny_pipeline.elf

# --- RULE 4: make clean (Remove generated layout artifacts) ---
clean:
	rm -rf $(HOST_BUILD_DIR) $(RV_BUILD_DIR)
	@echo "--- Workspace Cleaned ---"