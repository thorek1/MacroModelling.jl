#!/usr/bin/env julia

using Revise
using MacroModelling
using ForwardDiff
using Zygote
using FiniteDifferences
using LinearAlgebra
using Statistics

const ALGORITHM = Symbol(get(ENV, "ALGORITHM", "third_order"))
const FDM_ORDER = parse(Int, get(ENV, "FDM_ORDER", "4"))
const FDM_MAX_RANGE = parse(Float64, get(ENV, "FDM_MAX_RANGE", "1e-3"))

println("Julia:     ", VERSION)
println("Algorithm: ", ALGORITHM)
println("FDM:       central_fdm($(FDM_ORDER), 1, max_range=$(FDM_MAX_RANGE))")
println()

include(joinpath(@__DIR__, "..", "test", "models", "Caldara_et_al_2012_estim.jl"))

model = Caldara_et_al_2012_estim
p0 = copy(model.parameter_values)

opts = MacroModelling.merge_calculation_options(verbose = false)

# warm-up compile
MacroModelling.solve!(model, algorithm = ALGORITHM, opts = opts)

function ss_norm_objective(p)
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, ALGORITHM)
    _, SS_and_pars, _, _, solved = MacroModelling.get_relevant_steady_state_and_state_update(
        Val(ALGORITHM),
        p,
        model;
        opts = opts,
        estimation = true,
    )
    solved || error("get_relevant_steady_state_and_state_update did not solve")
    return norm(SS_and_pars)
end

function S_component_norm_objective(p, component_index::Int)
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, ALGORITHM)
    _, _, 𝐒, _, solved = MacroModelling.get_relevant_steady_state_and_state_update(
        Val(ALGORITHM),
        p,
        model;
        opts = opts,
        estimation = true,
    )
    solved || error("get_relevant_steady_state_and_state_update did not solve")

    if 𝐒 isa AbstractMatrix
        component_index == 1 || error("Requested 𝐒_$component_index but solution is a single matrix")
        return norm(𝐒)
    elseif 𝐒 isa AbstractVector
        1 <= component_index <= length(𝐒) || error("Requested 𝐒_$component_index but only $(length(𝐒)) solution matrices are available")
        return norm(𝐒[component_index])
    else
        error("Unexpected type for 𝐒: $(typeof(𝐒))")
    end
end

S₁_norm_objective(p) = S_component_norm_objective(p, 1)
S₂_norm_objective(p) = S_component_norm_objective(p, 2)
S₃_norm_objective(p) = S_component_norm_objective(p, 3)

function state_norm_objective(p)
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, ALGORITHM)
    _, _, _, state, solved = MacroModelling.get_relevant_steady_state_and_state_update(
        Val(ALGORITHM),
        p,
        model;
        opts = opts,
        estimation = true,
    )
    solved || error("get_relevant_steady_state_and_state_update did not solve")

    if state isa AbstractArray{<:Real}
        return norm(state)
    elseif state isa AbstractVector
        return sum(norm, state)
    else
        error("Unexpected type for state: $(typeof(state))")
    end
end

function summarize_diff(name, g_ref, g_test)
    Δ = g_test .- g_ref
    abs_max = maximum(abs, Δ)
    abs_mean = mean(abs, Δ)
    rel_norm = norm(Δ) / max(norm(g_ref), eps(Float64))
    println("$name")
    println("  length              = ", length(g_test))
    println("  max abs diff        = ", abs_max)
    println("  mean abs diff       = ", abs_mean)
    println("  relative norm       = ", rel_norm)
    println()
end

function run_gradient_comparison_block(block_name, objective_fn, p)
    println("\n", "="^70)
    println("  $block_name")
    println("="^70, "\n")

    y0 = objective_fn(p)
    println("Objective value:  ", y0)
    println("Parameter length: ", length(p))
    println()

    g_fd = nothing
    g_fwd = nothing
    g_zyg = nothing

    println("Computing FiniteDifferences gradient...")
    fdm = FiniteDifferences.central_fdm(FDM_ORDER, 1, max_range = FDM_MAX_RANGE)
    fd_raw = FiniteDifferences.grad(fdm, objective_fn, p)
    g_fd = fd_raw isa Tuple ? fd_raw[1] : fd_raw
    println("  done - norm(g_fd) = ", norm(g_fd))
    println()

    println("Computing ForwardDiff gradient...")
    try
        g_fwd = ForwardDiff.gradient(objective_fn, p)
        println("  done - norm(g_fwd) = ", norm(g_fwd))
    catch err
        println("  failed: ", sprint(showerror, err, catch_backtrace()))
    end
    println()

    println("Computing Zygote gradient...")
    try
        g_zyg = Zygote.gradient(objective_fn, p)[1]
        println("  done - norm(g_zyg) = ", norm(g_zyg))
    catch err
        println("  failed: ", sprint(showerror, err, catch_backtrace()))
    end
    println()

    println("=== Comparisons (reference = FiniteDifferences) ===")
    if g_fwd !== nothing
        summarize_diff("ForwardDiff vs FiniteDifferences", g_fd, g_fwd)
    end
    if g_zyg !== nothing
        summarize_diff("Zygote vs FiniteDifferences", g_fd, g_zyg)
    end
    if g_fwd !== nothing && g_zyg !== nothing
        summarize_diff("ForwardDiff vs Zygote", g_fwd, g_zyg)
    end

    if g_fwd === nothing && g_zyg === nothing
        error("Both ForwardDiff and Zygote gradient computations failed in block: $block_name")
    end
end

function main()
    # run_gradient_comparison_block("Gradient of norm(SS_and_pars) w.r.t. parameter_values", ss_norm_objective, p0)
    # run_gradient_comparison_block("Gradient of norm(𝐒₁) w.r.t. parameter_values", S₁_norm_objective, p0)
    # run_gradient_comparison_block("Gradient of norm(𝐒₂) w.r.t. parameter_values", S₂_norm_objective, p0)
    run_gradient_comparison_block("Gradient of norm(𝐒₃) w.r.t. parameter_values", S₃_norm_objective, p0)
    # run_gradient_comparison_block("Gradient of norm(state) w.r.t. parameter_values", state_norm_objective, p0)
end

main()
