# ==========================================
# COMPILER CONFIGURATIONS
# ==========================================
TOOLCHAIN_BIN = compiler/built_thingy/bin
HOST_CXX = g++
RV_CXX   = $(TOOLCHAIN_BIN)/riscv64-unknown-elf-g++

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
VECTOR_SRC   = src/canny_vector.cpp           # RVV intrinsics (Gaussian + Sobel)
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
.PHONY: all test host_run canny_rv canny_vector run run_vector sweep_VLEN sweep_opt show convert clean 

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
	$(RV_CXX) $(RV_FLAGS) $(MAIN_SRC) -O3 $(PIPELINE_SRC)  -o $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- RISC-V Binary Successfully Compiled ---"

# --- RULE 3b: make canny_vector (RISC-V Cross-Compilation, RVV Intrinsics) ---
canny_vector: $(MAIN_SRC) $(VECTOR_SRC) $(PIPELINE_SRC)
	@mkdir -p $(RV_BUILD_DIR)
	$(RV_CXX) $(RV_FLAGS) -DUSE_RVV -O3 $(MAIN_SRC) $(VECTOR_SRC) $(PIPELINE_SRC) -o $(RV_BUILD_DIR)/canny_vector.elf
	@echo "--- RISC-V Vector Binary Successfully Compiled ---"

# --- RULE 4: make run (Execute on QEMU) ---
run: canny_rv $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- Launching on QEMU (VLEN=$(VLEN)) ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) \
		$(RV_BUILD_DIR)/canny_pipeline.elf $(IMG) $(WIDTH) $(HEIGHT)

# --- RULE 4b: make run_vector (Execute Vector Pipeline on QEMU) ---
run_vector: canny_vector $(RV_BUILD_DIR)/canny_vector.elf
	@echo "--- Launching Vector Pipeline on QEMU (VLEN=$(VLEN)) ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) \
		$(RV_BUILD_DIR)/canny_vector.elf $(IMG) $(WIDTH) $(HEIGHT)

# --- RULE 5: make sweep (Test multiple VLEN configurations) ---
sweep_VLEN: $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- Sweeping VLEN across 128, 256, 512 bits ---"
	@for v in 128 256 512; do \
		echo "========================================="; \
		echo "Testing VLEN=$$v"; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$$v \
			$(RV_BUILD_DIR)/canny_pipeline.elf $(IMG) $(WIDTH) $(HEIGHT); \
	done
	@echo "========================================="
	@echo "--- Sweep Complete ---"


# --- RULE 6: make sweep_opt (Test multiple optimization levels) ---
sweep_opt: $(RV_BUILD_DIR)/canny_pipeline.elf
	@echo "--- Sweeping Optimization Levels O0, O1, O2, O3 Os Ofast---"
	@for opt in O0 O1 O2 O3 Os Ofast; do \
		echo "========================================="; \
		echo "Testing Optimization Level: $$opt"; \
		$(RV_CXX) $(RV_FLAGS) -$$opt $(MAIN_SRC) $(PIPELINE_SRC) -o $(RV_BUILD_DIR)/canny_pipeline_$$opt.elf; \
		echo "Binary size: $$(stat -c%s $(RV_BUILD_DIR)/canny_pipeline_$$opt.elf) bytes"; \
		qemu-riscv64 -cpu rv64,v=true,vlen=$(VLEN) \
			$(RV_BUILD_DIR)/canny_pipeline_$$opt.elf $(IMG) $(WIDTH) $(HEIGHT); \
	done
	@echo "========================================="
	@echo "--- Sweep Complete ---"



# --- RULE 7: make show (Convert output/raw/*.raw to PNGs and display) ---
show:
	@echo "--- Converting output/raw/*.raw to PNG ---"
	python3 scripts/show_output.py
# --- RULE 8: make convert (Convert output PNGs back to raw for diffing) ---
convert:
	@echo "--- Converting to raw---"
	python3 scripts/convert_to_raw.py
# --- RULE 7: make clean ---
clean:
	rm -rf $(HOST_BUILD_DIR) $(RV_BUILD_DIR)
	@echo "--- Workspace Cleaned ---"
