#!/usr/bin/env bash
# Example invocation:
# ./test/dynare_comparison/run_thread_sweep_macos.sh \
#   --julia-exe "$HOME/.juliaup/bin/julia" \
#   --thread-counts 1,2,4,8

set -euo pipefail

script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_root/../.." && pwd)"

thread_counts_csv="1,2,4,8"
output_root=""
julia_exe=""
generate_julia_script=""
dynare_script=""
compare_script=""
sweep_compare_script=""
extract_script=""
dynare_docker_image_tag="${DYNARE_DOCKER_IMAGE_TAG:-macromodelling-dynare-testing}"
only_models_csv=""
validate_only=0

print_usage() {
    cat <<'USAGE'
Usage: run_thread_sweep_macos.sh [options]

Options:
  --thread-counts <csv>          Comma-separated thread counts (default: 1,2,4,8)
  --output-root <path>           Final output root (default: test/dynare_comparison/output_thread_sweep)
  --julia-exe <path>             Path to Julia executable
  --generate-julia-script <path> Phase-1 Julia script (default: generate_julia_results.jl)
  --dynare-script <path>         Phase-2 Dynare shell script (default: run_all_dynare.sh)
  --extract-script <path>        Dynare extract MATLAB script (default: extract_dynare_results.m)
  --compare-script <path>        Phase-3 Julia script (default: compare_results.jl)
  --sweep-compare-script <path>  Sweep summary Julia script (default: compare_thread_sweep_results.jl)
  --dynare-docker-image-tag <s>  Docker image tag (default: macromodelling-dynare-testing)
  --only-models <csv>            Restrict models (forwarded to phase 1)
  --validate-only                Print resolved plan and exit
  -h, --help                     Show this help
USAGE
}

resolve_existing_path() {
    local description="$1"
    shift

    local candidate
    for candidate in "$@"; do
        if [[ -z "${candidate// }" ]]; then
            continue
        fi
        if [[ -e "$candidate" ]]; then
            cd -- "$(dirname -- "$candidate")"
            local resolved
            resolved="$(pwd)/$(basename -- "$candidate")"
            cd -- "$repo_root"
            printf '%s\n' "$resolved"
            return 0
        fi
    done

    printf 'Could not find %s. Checked: %s\n' "$description" "$*" >&2
    return 1
}

get_julia_executable() {
    local candidates=()
    if [[ -n "$julia_exe" ]]; then
        candidates+=("$julia_exe")
    fi
    if [[ -n "${JULIA_EXE:-}" ]]; then
        candidates+=("$JULIA_EXE")
    fi
    if [[ -x "$HOME/.juliaup/bin/julia" ]]; then
        candidates+=("$HOME/.juliaup/bin/julia")
    fi
    if [[ -x "/Applications/Julia-1.12.app/Contents/Resources/julia/bin/julia" ]]; then
        candidates+=("/Applications/Julia-1.12.app/Contents/Resources/julia/bin/julia")
    fi

    local julia_on_path
    julia_on_path="$(command -v julia || true)"
    if [[ -n "$julia_on_path" ]]; then
        candidates+=("$julia_on_path")
    fi

    resolve_existing_path "Julia executable" "${candidates[@]}"
}

make_staging_output_root() {
    local resolved_output_parent="$1"
    local output_root_leaf="$2"
    local timestamp random_id
    timestamp="$(date +%Y%m%d%H%M%S)"
    random_id="$RANDOM"
    printf '%s/%s.__staging_%s_%s\n' "$resolved_output_parent" "$output_root_leaf" "$timestamp" "$random_id"
}

publish_staged_output_root() {
    local stage_output_root="$1"
    local final_output_root="$2"
    local resolved_output_parent="$3"
    local output_root_leaf="$4"

    if [[ ! -d "$stage_output_root" ]]; then
        printf 'Staged sweep output not found: %s\n' "$stage_output_root" >&2
        return 1
    fi

    if [[ -e "$final_output_root" ]]; then
        local previous_output_root
        previous_output_root="${resolved_output_parent}/${output_root_leaf}.__previous_$(date +%Y%m%d%H%M%S)_$RANDOM"
        printf 'Moving existing output root aside: %s -> %s\n' "$final_output_root" "$previous_output_root"
        mv "$final_output_root" "$previous_output_root"
    fi

    printf 'Publishing staged sweep output: %s -> %s\n' "$stage_output_root" "$final_output_root"
    mv "$stage_output_root" "$final_output_root"
}

invoke_julia_script() {
    local executable="$1"
    local project_root="$2"
    local script_path="$3"
    local output_argument="$4"
    local description="$5"
    local requested_thread_count="$6"
    local use_thread_count="$7"
    shift 7
    local extra_script_args=("$@")

    local julia_args=("--project=${project_root}")
    if [[ "$use_thread_count" == "1" ]]; then
        julia_args+=("--threads=${requested_thread_count}")
    fi
    julia_args+=("$script_path")

    local extra
    for extra in "${extra_script_args[@]}"; do
        if [[ -n "${extra// }" ]]; then
            julia_args+=("$extra")
        fi
    done

    julia_args+=("$output_argument")

    printf 'Running Julia step: %s\n' "$description"
    "$executable" "${julia_args[@]}"
}

invoke_dynare_phase() {
    local thread_output_dir="$1"
    local requested_thread_count="$2"
    local resolved_dynare_script="$3"
    local resolved_extract_script="$4"

    if ! docker image inspect "$dynare_docker_image_tag" >/dev/null 2>&1; then
        printf 'Docker image %s not found. Build it first with: docker build -t %s test/dynare_comparison/\n' "$dynare_docker_image_tag" "$dynare_docker_image_tag" >&2
        return 1
    fi

    printf 'Running Dynare step for %s thread(s)\n' "$requested_thread_count"
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -e THREAD_COUNT="$requested_thread_count" \
        -v "$thread_output_dir:/work/output" \
        -v "$resolved_extract_script:/work/extract_dynare_results.m:ro" \
        -v "$resolved_dynare_script:/work/run_all_dynare.sh:ro" \
        --entrypoint /bin/bash \
        "$dynare_docker_image_tag" \
        /work/run_all_dynare.sh
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --thread-counts)
            thread_counts_csv="$2"
            shift 2
            ;;
        --output-root)
            output_root="$2"
            shift 2
            ;;
        --julia-exe)
            julia_exe="$2"
            shift 2
            ;;
        --generate-julia-script)
            generate_julia_script="$2"
            shift 2
            ;;
        --dynare-script)
            dynare_script="$2"
            shift 2
            ;;
        --extract-script)
            extract_script="$2"
            shift 2
            ;;
        --compare-script)
            compare_script="$2"
            shift 2
            ;;
        --sweep-compare-script)
            sweep_compare_script="$2"
            shift 2
            ;;
        --dynare-docker-image-tag)
            dynare_docker_image_tag="$2"
            shift 2
            ;;
        --only-models)
            only_models_csv="$2"
            shift 2
            ;;
        --validate-only)
            validate_only=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n\n' "$1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$output_root" ]]; then
    output_root="$script_root/output_thread_sweep"
fi
if [[ -z "$generate_julia_script" ]]; then
    generate_julia_script="$script_root/generate_julia_results.jl"
fi
if [[ -z "$dynare_script" ]]; then
    dynare_script="$script_root/run_all_dynare.sh"
fi
if [[ -z "$extract_script" ]]; then
    extract_script="$script_root/extract_dynare_results.m"
fi
if [[ -z "$compare_script" ]]; then
    compare_script="$script_root/compare_results.jl"
fi
if [[ -z "$sweep_compare_script" ]]; then
    sweep_compare_script="$script_root/compare_thread_sweep_results.jl"
fi

requested_output_root="$output_root"
output_root_leaf="$(basename -- "$requested_output_root")"
output_root_parent="$(dirname -- "$requested_output_root")"

mkdir -p "$output_root_parent"
resolved_output_parent="$(cd -- "$output_root_parent" && pwd)"
resolved_output_root="$resolved_output_parent/$output_root_leaf"
staging_output_root="$(make_staging_output_root "$resolved_output_parent" "$output_root_leaf")"

cd -- "$repo_root"

resolved_julia_exe="$(get_julia_executable)"
resolved_generate_julia_script="$(resolve_existing_path 'Julia phase-1 script' "$generate_julia_script")"
resolved_dynare_script="$(resolve_existing_path 'Dynare phase-2 script' "$dynare_script")"
resolved_extract_script="$(resolve_existing_path 'Dynare extract script' "$extract_script")"
resolved_compare_script="$(resolve_existing_path 'Julia phase-3 script' "$compare_script")"
resolved_sweep_compare_script="$(resolve_existing_path 'thread-sweep summary script' "$sweep_compare_script")"

thread_counts_raw=()
while IFS= read -r token || [[ -n "$token" ]]; do
    thread_counts_raw+=("$token")
done < <(printf '%s' "$thread_counts_csv" | tr ',' '\n' | sed 's/^ *//; s/ *$//')

if [[ "${#thread_counts_raw[@]}" -eq 0 ]]; then
    printf 'At least one thread count must be provided.\n' >&2
    exit 1
fi

thread_counts=()
for t in "${thread_counts_raw[@]}"; do
    if [[ -z "$t" ]]; then
        continue
    fi
    if [[ ! "$t" =~ ^[0-9]+$ ]] || [[ "$t" -lt 1 ]]; then
        printf 'Invalid thread count: %s\n' "$t" >&2
        exit 1
    fi
    already_seen=0
    for existing_thread in "${thread_counts[@]-}"; do
        if [[ "$existing_thread" == "$t" ]]; then
            already_seen=1
            break
        fi
    done
    if [[ "$already_seen" -eq 0 ]]; then
        thread_counts+=("$t")
    fi
done

if [[ "${#thread_counts[@]}" -eq 0 ]]; then
    printf 'At least one valid thread count must be provided.\n' >&2
    exit 1
fi

IFS=$'\n' thread_counts=( $(printf '%s\n' "${thread_counts[@]}" | sort -n) )
unset IFS

phase1_extra_args=()
if [[ -n "${only_models_csv// }" ]]; then
    phase1_extra_args+=("--only-models=${only_models_csv}")
    printf 'Restricting sweep to models: %s\n' "$only_models_csv"
fi

printf 'Repository root: %s\n' "$repo_root"
printf 'Julia executable: %s\n' "$resolved_julia_exe"
printf 'Dynare Docker image tag: %s\n' "$dynare_docker_image_tag"
printf 'Final sweep output root: %s\n' "$resolved_output_root"
printf 'Sweep staging root: %s\n' "$staging_output_root"
printf 'Thread counts: %s\n' "${thread_counts[*]}"

if [[ "$validate_only" -eq 1 ]]; then
    printf 'Validation only mode enabled.\n'
    for thread_count in "${thread_counts[@]}"; do
        thread_output_dir="$staging_output_root/threads_${thread_count}"
        printf 'Planned output directory: %s\n' "$thread_output_dir"
    done
    exit 0
fi

mkdir -p "$staging_output_root"

cleanup_on_error() {
    if [[ -d "$staging_output_root" ]]; then
        printf 'Keeping staged sweep output for inspection: %s\n' "$staging_output_root" >&2
    fi
}
trap cleanup_on_error ERR

for thread_count in "${thread_counts[@]}"; do
    thread_output_dir="$staging_output_root/threads_${thread_count}"

    printf '========================================\n'
    printf 'Running sweep for thread count: %s\n' "$thread_count"
    printf '========================================\n'

    invoke_julia_script \
        "$resolved_julia_exe" \
        "$repo_root" \
        "$resolved_generate_julia_script" \
        "$thread_output_dir" \
        "Phase 1 export for ${thread_count} thread(s)" \
        "$thread_count" \
        1 \
        "${phase1_extra_args[@]-}"

    invoke_dynare_phase \
        "$thread_output_dir" \
        "$thread_count" \
        "$resolved_dynare_script" \
        "$resolved_extract_script"

    invoke_julia_script \
        "$resolved_julia_exe" \
        "$repo_root" \
        "$resolved_compare_script" \
        "$thread_output_dir" \
        "Phase 3 compare for ${thread_count} thread(s)" \
        "$thread_count" \
        1
done

invoke_julia_script \
    "$resolved_julia_exe" \
    "$repo_root" \
    "$resolved_sweep_compare_script" \
    "$staging_output_root" \
    'Cross-thread benchmark summary' \
    1 \
    0

publish_staged_output_root \
    "$staging_output_root" \
    "$resolved_output_root" \
    "$resolved_output_parent" \
    "$output_root_leaf"

trap - ERR
printf 'Thread sweep complete.\n'