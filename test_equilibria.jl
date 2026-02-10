#!/usr/bin/env julia
# Test script to compare equilibria between refactored and main branch
# Focus on the 2 models that found different equilibria: Gali_2015_chapter_3_nonlinear, QUEST3_2009

using Revise
using MacroModelling

# Helper to compare SS values
function compare_ss_with_main(model, main_ss_file)
    ss = get_steady_state(model, verbose=true)
    ss_vec = collect(ss[1:end, 1])
    names = collect(ss.axis1)
    
    println("  Model solved. $(length(ss_vec)) variables.")
    println("  First 5 values: ", ss_vec[1:min(5, length(ss_vec))])
    
    return ss_vec, names
end

# Test Gali 2015 chapter 3 nonlinear
println("=" ^ 60)
println("Testing: Gali_2015_chapter_3_nonlinear")
println("=" ^ 60)
include("models/Gali_2015_chapter_3_nonlinear.jl")
ss_gali = get_steady_state(Gali_2015_chapter_3_nonlinear, verbose=true)
ss_gali_vec = collect(ss_gali[1:end, 1])
ss_gali_names = collect(ss_gali.axis1)
println("SS values:")
for (i, (n, v)) in enumerate(zip(ss_gali_names, ss_gali_vec))
    println("  $n = $v")
end

println()

# Test QUEST3 2009
println("=" ^ 60)
println("Testing: QUEST3_2009")
println("=" ^ 60)
include("models/QUEST3_2009.jl")
ss_quest = get_steady_state(QUEST3_2009, verbose=true)
ss_quest_vec = collect(ss_quest[1:end, 1])
ss_quest_names = collect(ss_quest.axis1)
println("SS values (first 15):")
for (i, (n, v)) in enumerate(zip(ss_quest_names, ss_quest_vec))
    i > 15 && break
    println("  $n = $v")
end

println()

# Also test previously passing models to check for regressions
println("=" ^ 60)
println("Testing: RBC_baseline (regression check)")
println("=" ^ 60)
include("models/RBC_baseline.jl")
ss_rbc = get_steady_state(RBC_baseline, verbose=true)
ss_rbc_vec = collect(ss_rbc[1:end, 1])
println("First 3 SS values: ", ss_rbc_vec[1:3])
expected = [5.936252888048734, 47.39025414828825, 6.8840579710144985]
diff = maximum(abs.(ss_rbc_vec[1:3] .- expected))
println("Max diff from expected: $diff")
println(diff < 1e-10 ? "✓ PASS" : "✗ FAIL")

println()
println("=" ^ 60)
println("Testing: FS2000 (regression check)")
println("=" ^ 60)
include("models/FS2000.jl")
ss_fs = get_steady_state(FS2000, verbose=true)
ss_fs_vec = collect(ss_fs[1:end, 1])
println("First 5 SS values: ", ss_fs_vec[1:5])
