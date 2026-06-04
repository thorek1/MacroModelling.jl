#!/bin/bash
# run_all_dynare.sh — Phase 2 of Dynare comparison
#
# Iterates over model directories in /work/output/, runs Dynare on each
# .mod file, and extracts results to CSV via extract_dynare_results.m.
#
# Expected volume mount: -v host_output_dir:/work/output

set -euo pipefail

OUTPUT_DIR="/work/output"
EXTRACT_SCRIPT="/work/extract_dynare_results.m"
THREAD_COUNT="${THREAD_COUNT:-1}"

# Enforce single-thread execution in CI unless explicitly overridden.
export OMP_NUM_THREADS="$THREAD_COUNT"
export OMP_THREAD_LIMIT="$THREAD_COUNT"
export OMP_DYNAMIC="FALSE"
export MKL_NUM_THREADS="$THREAD_COUNT"
export MKL_DYNAMIC="FALSE"
export OPENBLAS_NUM_THREADS="$THREAD_COUNT"
export BLIS_NUM_THREADS="$THREAD_COUNT"
export VECLIB_MAXIMUM_THREADS="$THREAD_COUNT"
export TBB_NUM_THREADS="$THREAD_COUNT"

# Detect Dynare's Octave path
DYNARE_MATLAB=""
for p in /usr/lib/dynare/matlab /usr/share/dynare/matlab /usr/local/lib/dynare/matlab; do
    if [ -d "$p" ]; then
        DYNARE_MATLAB="$p"
        break
    fi
done

if [ -z "$DYNARE_MATLAB" ]; then
    echo "ERROR: Could not find Dynare matlab directory"
    exit 1
fi

echo "Using Dynare at: $DYNARE_MATLAB"
OCTAVE_VERSION="$(octave --version | head -1)"
DYNARE_VERSION_OUTPUT="$(octave --no-gui --eval "addpath('$DYNARE_MATLAB'); dynare_version();" 2>&1 | tr '\r\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
OCTAVE_BLAS="$(octave --no-gui --quiet --eval "try; disp(__octave_config_info__('BLAS_LIBS')); catch; disp('unknown'); end;" 2>&1 | tr '\r\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
OCTAVE_LAPACK="$(octave --no-gui --quiet --eval "try; disp(__octave_config_info__('LAPACK_LIBS')); catch; disp('unknown'); end;" 2>&1 | tr '\r\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
OCTAVE_LINKED_BLAS="$(ldd /usr/bin/octave-cli 2>/dev/null | grep -Ei 'libblas|openblas|mkl' | tr '\r\n' '|' | sed 's/|$//; s/^ *//')"
OCTAVE_LINKED_LAPACK="$(ldd /usr/bin/octave-cli 2>/dev/null | grep -Ei 'liblapack|mkl' | tr '\r\n' '|' | sed 's/|$//; s/^ *//')"
METADATA_FILE="$OUTPUT_DIR/comparison_environment_dynare.txt"
{
    echo "dynare_driver=Octave"
    echo "dynare_matlab_path=$DYNARE_MATLAB"
    echo "dynare_version=$DYNARE_VERSION_OUTPUT"
    echo "octave_version=$OCTAVE_VERSION"
    echo "blas=$OCTAVE_BLAS"
    echo "lapack=$OCTAVE_LAPACK"
    echo "linked_blas=$OCTAVE_LINKED_BLAS"
    echo "linked_lapack=$OCTAVE_LINKED_LAPACK"
    echo "hostname=$(hostname 2>/dev/null || echo unknown)"
    echo "kernel=$(uname -srmo 2>/dev/null || uname -a)"
    echo "arch=$(uname -m 2>/dev/null || echo unknown)"
    echo "cpu_threads=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo unknown)"
    echo "thread_count_requested=$THREAD_COUNT"
    echo "env_OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "env_OMP_THREAD_LIMIT=$OMP_THREAD_LIMIT"
    echo "env_OMP_DYNAMIC=$OMP_DYNAMIC"
    echo "env_MKL_NUM_THREADS=$MKL_NUM_THREADS"
    echo "env_MKL_DYNAMIC=$MKL_DYNAMIC"
    echo "env_OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    echo "env_BLIS_NUM_THREADS=$BLIS_NUM_THREADS"
    echo "env_VECLIB_MAXIMUM_THREADS=$VECLIB_MAXIMUM_THREADS"
    echo "env_TBB_NUM_THREADS=$TBB_NUM_THREADS"
} > "$METADATA_FILE"
echo "Octave version: $OCTAVE_VERSION"
echo "Requested thread count: $THREAD_COUNT"
echo "Thread env: OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "Linked BLAS: ${OCTAVE_LINKED_BLAS:-unknown}"
echo "Linked LAPACK: ${OCTAVE_LINKED_LAPACK:-unknown}"
echo "Dynare version: $DYNARE_VERSION_OUTPUT"

# Runtime CSV header
RUNTIME_CSV="$OUTPUT_DIR/runtime_dynare.csv"
echo "model,elapsed_seconds" > "$RUNTIME_CSV"
TOTAL_START=$(date +%s%N)

# Process each model
for model_dir in "$OUTPUT_DIR"/*/; do
    model_name=$(basename "$model_dir")
    mod_file="$model_dir/${model_name}.mod"
    dynare_stub="m"

    if [ ! -f "$mod_file" ]; then
        echo "SKIP: No .mod file found for $model_name"
        continue
    fi

    echo "========================================"
    echo "Running Dynare on: $model_name"
    echo "========================================"

    dynare_out_dir="$model_dir/dynare"
    mkdir -p "$dynare_out_dir"

    # Work in a temporary directory to avoid Dynare file pollution
    workdir=$(mktemp -d)
    cp "$mod_file" "$workdir/${dynare_stub}.mod"
    cp "$EXTRACT_SCRIPT" "$workdir/"

    # Add nograph to stoch_simul to avoid graphics toolkit errors in headless mode
    sed -i 's/stoch_simul(/stoch_simul(nograph, /' "$workdir/${dynare_stub}.mod"

    MODEL_START=$(date +%s%N)

    (
        cd "$workdir"
        octave --no-gui --eval "
            addpath('$DYNARE_MATLAB');
            model_name = '$model_name';
            output_dir = 'dynare_output';
            total_tic = tic;
            dynare $dynare_stub noclearall;
            extract_dynare_results;
            elapsed = toc(total_tic);
            fid = fopen(fullfile('dynare_output', 'runtime_seconds.csv'), 'w');
            fprintf(fid, '%.6f\n', elapsed);
            fclose(fid);
            fprintf('Wall-clock time for %s (post-startup): %.3f s\n', model_name, elapsed);
        "

        # Copy results to the mounted output directory
        if [ -d "dynare_output" ]; then
            cp dynare_output/* "$dynare_out_dir/"
            echo "Results copied to $dynare_out_dir"
        else
            echo "ERROR: No output produced for $model_name"
            exit 1
        fi
    )

    MODEL_END=$(date +%s%N)
    MODEL_ELAPSED=$(awk "BEGIN {printf \"%.6f\", ($MODEL_END - $MODEL_START) / 1000000000}")

    # Subtract benchmark loop time to get warmup-only elapsed
    BENCH_ELAPSED_FILE="$dynare_out_dir/bench_elapsed_seconds.csv"
    if [ -f "$BENCH_ELAPSED_FILE" ]; then
        BENCH_ELAPSED=$(cat "$BENCH_ELAPSED_FILE")
        WARMUP_ELAPSED=$(awk "BEGIN {printf \"%.6f\", $MODEL_ELAPSED - $BENCH_ELAPSED}")
    else
        WARMUP_ELAPSED="$MODEL_ELAPSED"
        BENCH_ELAPSED="0"
    fi
    echo "$model_name,$WARMUP_ELAPSED" >> "$RUNTIME_CSV"
    echo "Done: $model_name (warmup: ${WARMUP_ELAPSED} s, bench: ${BENCH_ELAPSED} s, total: ${MODEL_ELAPSED} s)"

    rm -rf "$workdir"
done

TOTAL_END=$(date +%s%N)
TOTAL_ELAPSED=$(awk "BEGIN {printf \"%.6f\", ($TOTAL_END - $TOTAL_START) / 1000000000}")
# Sum warmup times from runtime CSV for accurate total (excludes benchmarks)
WARMUP_TOTAL=$(awk -F',' 'NR>1 && $1!="TOTAL" {sum+=$2} END {printf "%.6f", sum}' "$RUNTIME_CSV")
echo "TOTAL,$WARMUP_TOTAL" >> "$RUNTIME_CSV"
echo "Total wall-clock time (warmup only): ${WARMUP_TOTAL} s"
echo "Phase 2 complete."
