# ==========================================
# COMPILER CONFIGURATIONS
# ==========================================
HOST_CXX = g++
RV_CXX   = riscv64-unknown-elf-g++

# Compiler Flags
HOST_FLAGS = -std=c++17 -Wall -I include
RV_FLAGS   = -march=rv64gcv -mabi=lp64d -std=c++17 -I include

# GoogleTest Paths
GTEST_INC  = -I$(HOME)/.local/include
GTEST_LIBS = -L$(HOME)/.local/lib -lgtest -lgtest_main -pthread

# ==========================================
# SEPARATE OUTPUT DIRECTORIES
# ==========================================
HOST_BUILD_DIR = build/host
RV_BUILD_DIR   = build/rv

# Source files
PIPELINE_SRC = src/canny_scalar.cpp           # shared implementation
MAIN_SRC     = src/main.cpp                    # pipeline entry point
TEST_FILES   = tests/host_tests.cpp

# ==========================================
# QEMU + IMAGE CONFIGURATIONS
# ==========================================
VLEN   = 128
IMG    = images/car.raw
WIDTH  = 512
HEIGHT = 341

# ==========================================
.PHONY: all test host_run canny_rv run sweep clean

all: test canny_rv

# --- RULE 1: make test (Host-side GoogleTest) ---
# Links the pipeline implementation but NOT main.cpp (GoogleTest provides its own main)
test: $(TEST_FILES) $(PIPELINE_SRC)
	@mkdir -p $(HOST_BUILD_DIR)
	$(HOST_CXX) $(HOST_FLAGS) $(GTEST_INC) \
		$(TEST_FILES) $(PIPELINE_SRC) \
		$(GTEST_LIBS) -o $(HOST_BUILD_DIR)/host_tests
	@echo "--- Running Host-side GoogleTest ---"
	./$(HOST_BUILD_DIR)/host_tests

# --- RULE 2: make host_run (Run pipeline natively on host for quick visual checks) ---
host_run: $(MAIN_SRC) $(PIPELINE_SRC)
	@mkdir -p $(HOST_BUILD_DIR)
	$(HOST_CXX) $(HOST_FLAGS) $(MAIN_SRC) $(PIPELINE_SRC) -o $(HOST_BUILD_DIR)/canny_host
	@echo "--- Running on Host ---"
	./$(HOST_BUILD_DIR)/canny_host $(IMG) $(WIDTH) $(HEIGHT)

# --- RULE 3: make canny_rv (RISC-V Cross-Compilation) ---
canny_rv: $(MAIN_SRC) $(PIPELINE_SRC)
	@mkdir -p $(RV_BUILD_DIR)
	$(RV_CXX) $(RV_FLAGS) $(MAIN_SRC) $(PIPELINE_SRC) -o $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- RISC-V Binary Successfully Compiled ---"

# --- RULE 4: make run (Execute on QEMU) ---
run: $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- Launching on QEMU (VLEN=$(VLEN)) ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) \
		$(RV_BUILD_DIR)/canny_pipeline.elf $(IMG) $(WIDTH) $(HEIGHT)

# --- RULE 5: make sweep (Test multiple VLEN configurations) ---
sweep: $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- Sweeping VLEN across 128, 256, 512 bits ---"
	@for v in 128 256 512; do \
		echo "========================================="; \
		echo "Testing VLEN=$$v"; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$v \
			$(RV_BUILD_DIR)/canny_pipeline.elf $(IMG) $(WIDTH) $(HEIGHT); \
	done
	@echo "========================================="
	@echo "--- Sweep Complete ---"

# --- RULE 6: make clean ---
clean:
	rm -rf $(HOST_BUILD_DIR) $(RV_BUILD_DIR)
	@echo "--- Workspace Cleaned ---"