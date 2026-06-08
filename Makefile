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
HOST_BUILD_DIR = build/host
RV_BUILD_DIR   = build/rv

# Source files
SRC_FILES   = src/main.cpp
TEST_FILES  = tests/host_tests.cpp


# ==========================================
# QEMU CONFIGURATIONS
# ==========================================

# Default VLEN for QEMU (128, 256, or 512)
VLEN = 128

# ==========================================
.PHONY: all test canny_rv run sweep clean

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


# --- RULE 5: make sweep (Test multiple VLEN configurations) ---
sweep: $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- Sweeping VLEN across 128, 256, 512, and 1024 bits ---"
	@for v in 128 256 512; do \
		echo "========================================="; \
		echo "Testing VLEN=$$v"; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$v $(RV_BUILD_DIR)/canny_pipeline.elf; \
	done
	@echo "========================================="
	@echo "--- Sweep Complete ---"