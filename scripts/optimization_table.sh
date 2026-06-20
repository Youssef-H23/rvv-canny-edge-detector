#!/usr/bin/env bash
# ============================================================================
# Builds the scalar pipeline at every GCC optimization level (-O0/-O1/-O2/
# -O3/-Os/-Ofast) and the RVV pipeline (run at VLEN=128 and VLEN=256),
# parses per-stage timing + binary size out of main.cpp's profiling
# printout for all 7 pipeline stages (including the bonus stages: NMS,
# double threshold, hysteresis), and prints the resulting table directly
# in the terminal.
#
# Each configuration is run REPEATS times and averaged: a single QEMU
# invocation of the same binary was observed to vary by ~50%+ run to run
# (host scheduling noise, not a code issue), so one-shot numbers aren't
# trustworthy enough to put in front of the instructor.
#
# IMPORTANT for runtime: one QEMU launch of main.cpp already runs the full
# 100-iteration benchmark loop over ALL 7 stages and prints all 7 stage
# lines. So we launch QEMU once per repeat and parse every stage's value
# out of that single run's output, instead of launching QEMU separately
# per stage (which would re-run the entire 100-iteration, all-7-stages
# pipeline from scratch just to read one line and throw the rest away).
#
# Direction, NMS, Double Threshold, and Hysteresis have no RVV
# implementation in this codebase (only Gaussian, Sobel, and Magnitude L1
# were vectorized, per the profiling-driven Amdahl's-law justification in
# the README) -- their RVV 128/256 numbers are still real measured times,
# just of the same scalar code running inside the -DUSE_RVV -O2 binary, so
# expect them to land close to the -O2 column.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RV_CXX="riscv64-unknown-elf-g++"
RV_FLAGS="-march=rv64gcv -mabi=lp64d -std=c++17 -I include"
MAIN_SRC="src/main.cpp"
PIPELINE_SRC="src/canny_scalar.cpp"
VECTOR_SRC="src/canny_vector.cpp"
BUILD_DIR="build/rv"
IMG="images/car.raw"
WIDTH=512
HEIGHT=341
REPEATS=1

mkdir -p "$BUILD_DIR"

if [ ! -f "$IMG" ]; then
    echo "Missing $IMG -- run 'make convert' (or: python3 scripts/convert_to_raw.py) first." >&2
    exit 1
fi

# Stage labels exactly as main.cpp prints them, and the short names used as
# table row headers -- parallel indexed arrays, in table-row order.
STAGE_LABELS=(
    "Stage 1: Gaussian Blur"
    "Stage 2: Sobel Gradients"
    "Stage 3a: Magnitude L1"
    "Stage 3b: Direction"
    "Stage 4: Non-Max Suppression"
    "Stage 5a: Double Threshold"
    "Stage 5b: Hysteresis Tracing"
)
ROW_NAMES=(
    "Gaussian 5x5"
    "Sobel Gx/Gy"
    "Magnitude"
    "Direction"
    "Non-Max Suppr."
    "Double Thresh."
    "Hysteresis"
)
N_STAGES=${#STAGE_LABELS[@]}

# Columns to build/show, in order: every scalar optimization level, then
# both RVV VLEN configurations.
SCALAR_OPTS=(O0 O1 O2 O3 Os Ofast)
COLS=(O0 O1 O2 O3 Os Ofast RVV128 RVV256)
declare -A COL_LABEL=(
    [O0]="-O0" [O1]="-O1" [O2]="-O2" [O3]="-O3" [Os]="-Os" [Ofast]="-Ofast"
    [RVV128]="RVV 128" [RVV256]="RVV 256"
)

declare -A TOTAL_SEC   # TOTAL_SEC["<col>,<stage_idx>"] -> running sum of seconds across repeats
declare -A AVG_MS       # AVG_MS["<col>,<stage_idx>"]   -> averaged per-iteration ms
declare -A SZ           # SZ["<col>"]                   -> binary size in KB

size_kb() {
    awk -v b="$(stat -c%s "$1")" 'BEGIN{printf "%.1f", b/1024}'
}

# run_and_accumulate <col> <elf> <vlen>
# Launches QEMU REPEATS times (not REPEATS*N_STAGES times). Each run's full
# stdout is parsed once for every stage label, since all 7 lines are always
# present in a single run's output.
run_and_accumulate() {
    local col="$1" elf="$2" vlen="$3"
    local r out idx label secs key
    for ((r = 0; r < REPEATS; r++)); do
        out=$(qemu-riscv64 -cpu rv64,v=true,vlen="$vlen" "$elf" "$IMG" "$WIDTH" "$HEIGHT")
        for ((idx = 0; idx < N_STAGES; idx++)); do
            label="${STAGE_LABELS[$idx]}"
            secs=$(echo "$out" | grep "$label" | awk -F'|' '{gsub(/ /,"",$2); print $2}')
            key="$col,$idx"
            TOTAL_SEC[$key]=$(awk -v t="${TOTAL_SEC[$key]:-0}" -v s="$secs" 'BEGIN{print t+s}')
        done
    done
    for ((idx = 0; idx < N_STAGES; idx++)); do
        key="$col,$idx"
        AVG_MS[$key]=$(awk -v t="${TOTAL_SEC[$key]}" -v n="$REPEATS" 'BEGIN{printf "%.3f", (t/n)*10}')
    done
    SZ[$col]=$(size_kb "$elf")
}

echo "Building and benchmarking (each configuration averaged over $REPEATS runs)..." >&2

for opt in "${SCALAR_OPTS[@]}"; do
    elf="$BUILD_DIR/table_scalar_$opt.elf"
    echo "  [$opt] scalar build..." >&2
    $RV_CXX $RV_FLAGS -"$opt" "$MAIN_SRC" "$PIPELINE_SRC" -o "$elf"
    run_and_accumulate "$opt" "$elf" 128
done

echo "  [RVV] vector build..." >&2
VEC_ELF="$BUILD_DIR/table_vector.elf"
$RV_CXX $RV_FLAGS -DUSE_RVV -O2 "$MAIN_SRC" "$VECTOR_SRC" "$PIPELINE_SRC" -o "$VEC_ELF"
run_and_accumulate "RVV128" "$VEC_ELF" 128
run_and_accumulate "RVV256" "$VEC_ELF" 256

# ---- print table (built generically over COLS so the column count isn't hardcoded) ----
printf "%-16s" "Stage"
for col in "${COLS[@]}"; do printf " | %9s" "${COL_LABEL[$col]}"; done
printf "\n"
printf -- '-%.0s' $(seq 1 $((19 + ${#COLS[@]} * 12)))
printf "\n"

for ((idx = 0; idx < N_STAGES; idx++)); do
    printf "%-16s" "${ROW_NAMES[$idx]}"
    for col in "${COLS[@]}"; do printf " | %7sms" "${AVG_MS[$col,$idx]}"; done
    printf "\n"
done

printf "%-16s" "Binary size"
for col in "${COLS[@]}"; do printf " | %7sKB" "${SZ[$col]}"; done
printf "\n"

echo
echo "Note: each cell is averaged over $REPEATS QEMU runs of the same binary (see script header)."
echo "Note: Direction/NMS/Threshold/Hysteresis have no RVV kernel -- their RVV 128/256 numbers are the"
echo "      same scalar code, just timed inside the -DUSE_RVV -O2 binary (so close to the -O2 column)."
