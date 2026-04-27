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
METADATA_FILE="$OUTPUT_DIR/comparison_environment_dynare.txt"
{
    echo "dynare_driver=Octave"
    echo "dynare_matlab_path=$DYNARE_MATLAB"
    echo "dynare_version=$DYNARE_VERSION_OUTPUT"
    echo "octave_version=$OCTAVE_VERSION"
    echo "blas=$OCTAVE_BLAS"
    echo "lapack=$OCTAVE_LAPACK"
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
echo "Dynare version: $DYNARE_VERSION_OUTPUT"

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

    (
        cd "$workdir"
        octave --no-gui --eval "
            addpath('$DYNARE_MATLAB');
            model_name = '$model_name';
            output_dir = 'dynare_output';
            dynare $dynare_stub noclearall;
            extract_dynare_results;
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

    rm -rf "$workdir"
    echo "Done: $model_name"
done

echo "Phase 2 complete."
