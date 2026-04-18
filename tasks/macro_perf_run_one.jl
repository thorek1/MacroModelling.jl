using Pkg; Pkg.activate(".")
using Serialization, Printf

const MODEL_NAME = ARGS[1]
const TAG = length(ARGS) >= 2 ? ARGS[2] : "untagged"
const OUTFILE = joinpath(@__DIR__, "perf_results", "$(MODEL_NAME).$(TAG).jls")

function meminfo()
    s = read("/proc/self/status", String)
    rss  = parse(Int, match(r"VmRSS:\s+(\d+)", s).captures[1]) / 1024
    peak = parse(Int, match(r"VmHWM:\s+(\d+)", s).captures[1]) / 1024
    (; rss, peak)
end

const MODEL_PATHS = Dict(
    "NAWM_EAUS_2008"   => "models/NAWM_EAUS_2008.jl",
    "GNSS_2010"        => "models/GNSS_2010.jl",
    "Smets_Wouters_2007" => "models/Smets_Wouters_2007.jl",
    "FRBUS"            => "models/FRBUS.jl",
)

ENV["MM_PROFILE"] = "1"

const t_using = @elapsed using MacroModelling
const mem_using = meminfo()
println("[$MODEL_NAME] using MM: $(round(t_using;digits=2))s peak=$(round(mem_using.peak;digits=1))MB")
flush(stdout)

const MODEL_PATH = joinpath(@__DIR__, "..", MODEL_PATHS[MODEL_NAME])

# Read the file and split into @model and @parameters blocks so we can time each independently.
const SRC = read(MODEL_PATH, String)
# Find @parameters
const PARAMS_RE = r"@parameters"m
const m_params  = match(PARAMS_RE, SRC)
@assert m_params !== nothing "no @parameters in $MODEL_PATH"
const SRC_MODEL  = SRC[1:m_params.offset-1]
const SRC_PARAMS = SRC[m_params.offset:end]

# === @model ===
GC.gc(); GC.gc()
const mem_pre_model = meminfo()
const t_model = @elapsed include_string(Main, SRC_MODEL, MODEL_PATH)
const mem_post_model = meminfo()
println("[$MODEL_NAME] @model:      $(round(t_model;digits=2))s peak=$(round(mem_post_model.peak;digits=1))MB Δrss=$(round(mem_post_model.rss-mem_pre_model.rss;digits=1))MB")
flush(stdout)

# Reset profiler so that @model's contribution doesn't bleed into @parameters timings (none currently do, but be safe).
const HAS_PROFILER = isdefined(MacroModelling, :MacroPerf)
HAS_PROFILER && MacroModelling.MacroPerf.reset!()

# === @parameters ===
GC.gc(); GC.gc()
const mem_pre_params = meminfo()
const t_params = @elapsed include_string(Main, SRC_PARAMS, MODEL_PATH)
const mem_post_params = meminfo()
println("[$MODEL_NAME] @parameters: $(round(t_params;digits=2))s peak=$(round(mem_post_params.peak;digits=1))MB Δrss=$(round(mem_post_params.rss-mem_pre_params.rss;digits=1))MB")
flush(stdout)

# Snapshot per-phase profiler
const phases = HAS_PROFILER ? MacroModelling.MacroPerf.snapshot() : NamedTuple{(:phase,:seconds,:calls,:bytes),Tuple{Symbol,Float64,Int,UInt64}}[]
println("[$MODEL_NAME] per-phase wall time:")
for r in phases
    @printf("  %-40s %8.3f s  calls=%4d  bytes=%.1f MB\n", String(r.phase), r.seconds, r.calls, r.bytes/1024^2)
end

# Get a tiny invariant snapshot of the model's NSSS and key sizes for functional-equivalence diffing.
m = getfield(Main, Symbol(MODEL_NAME))
ss_vec = try
    collect(get_steady_state(m))
catch err
    @warn "get_steady_state failed" err
    Float64[]
end

result = (
    model      = MODEL_NAME,
    tag        = TAG,
    t_using    = t_using,
    t_model    = t_model,
    t_params   = t_params,
    mem_using  = mem_using,
    mem_post_model = mem_post_model,
    mem_post_params = mem_post_params,
    peak_overall = meminfo().peak,
    phases     = phases,
    ss_hash    = isempty(ss_vec) ? UInt64(0) : hash(round.(vec(Array(ss_vec)); digits=8)),
    ss_norm    = isempty(ss_vec) ? NaN : sqrt(sum(x->x*x, filter(isfinite, ss_vec))),
    ss_n       = length(ss_vec),
    ss_nans    = count(!isfinite, ss_vec),
)

mkpath(dirname(OUTFILE))
serialize(OUTFILE, result)
println("[$MODEL_NAME] wrote $OUTFILE")
println("[$MODEL_NAME] PEAK overall: $(round(meminfo().peak;digits=1))MB ss_n=$(result.ss_n) ss_nans=$(result.ss_nans) ss_norm=$(round(result.ss_norm;digits=3))")
