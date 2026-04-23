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
echo "Octave version: $(octave --version | head -1)"
octave --no-gui --eval "addpath('$DYNARE_MATLAB'); dynare_version();"

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
