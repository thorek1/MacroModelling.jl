#!/usr/bin/env julia

# using Revise - commented out for automated testing
using MacroModelling
using ForwardDiff
using Zygote
using FiniteDifferences
using LinearAlgebra
using Statistics
using SparseArrays
using ChainRulesCore

const FDM_ORDER = parse(Int, get(ENV, "FDM_ORDER", "4"))
const FDM_MAX_RANGE = parse(Float64, get(ENV, "FDM_MAX_RANGE", "1e-4"))
const RUN_FD = lowercase(get(ENV, "RUN_FD", "true")) == "true"
const RUN_FWD = lowercase(get(ENV, "RUN_FWD", "false")) == "true"
const RUN_ZYG = lowercase(get(ENV, "RUN_ZYG", "true")) == "true"
const INPUT_BLOCKS = Set(strip.(split(get(ENV, "INPUT_BLOCKS", "grad1,grad2,grad3,s2,s1"), ",")))

println("Julia:      ", VERSION)
println("FDM:        central_fdm($(FDM_ORDER), 1, max_range=$(FDM_MAX_RANGE))")
println("RUN_FD:     ", RUN_FD)
println("RUN_FWD:    ", RUN_FWD)
println("RUN_ZYG:    ", RUN_ZYG)
println("INPUT_BLOCKS: ", join(sort!(collect(INPUT_BLOCKS)), ", "))
println()

include(joinpath(@__DIR__, "..", "test", "models", "Caldara_et_al_2012_estim.jl"))

const model = Caldara_et_al_2012_estim
const p0 = copy(model.parameter_values)
const opts = MacroModelling.merge_calculation_options(verbose = false)

struct ThirdOrderInputs{T<:Real}
    ∇₁::Matrix{T}
    ∇₂::SparseMatrixCSC{T,Int}
    ∇₃::SparseMatrixCSC{T,Int}
    𝐒₁::Matrix{T}
    𝐒₂::SparseMatrixCSC{T,Int}
end

function sparse_with_new_values(template::SparseMatrixCSC{<:Real,Int}, values::AbstractVector{T}) where {T<:Real}
    length(values) == nnz(template) || error("Value vector length mismatch: expected $(nnz(template)), got $(length(values))")
    return SparseMatrixCSC(size(template, 1), size(template, 2), copy(template.colptr), copy(template.rowval), collect(values))
end

function _template_nzvals_from_structure(template::SparseMatrixCSC, Δ)
    out = similar(template.nzval, promote_type(eltype(template.nzval), eltype(Δ)))
    @inbounds for col in 1:size(template, 2)
        for k in template.colptr[col]:(template.colptr[col + 1] - 1)
            out[k] = Δ[template.rowval[k], col]
        end
    end
    return out
end

function ChainRulesCore.rrule(::typeof(sparse_with_new_values), template::SparseMatrixCSC{<:Real,Int}, values::AbstractVector{T}) where {T<:Real}
    y = sparse_with_new_values(template, values)
    project_values = ChainRulesCore.ProjectTo(values)

    function sparse_with_new_values_pullback(ȳ)
        dvalues = if ȳ isa ChainRulesCore.AbstractZero
            zero(values)
        elseif ȳ isa SparseMatrixCSC || ȳ isa AbstractMatrix
            _template_nzvals_from_structure(template, ȳ)
        else
            zero(values)
        end
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), project_values(dvalues)
    end

    return y, sparse_with_new_values_pullback
end

function promote_inputs(base::ThirdOrderInputs, ::Type{T}) where {T<:Real}
    ∇₁T = Matrix{T}(base.∇₁)
    ∇₂T = SparseMatrixCSC(size(base.∇₂, 1), size(base.∇₂, 2), copy(base.∇₂.colptr), copy(base.∇₂.rowval), T.(base.∇₂.nzval))
    ∇₃T = SparseMatrixCSC(size(base.∇₃, 1), size(base.∇₃, 2), copy(base.∇₃.colptr), copy(base.∇₃.rowval), T.(base.∇₃.nzval))
    𝐒₁T = Matrix{T}(base.𝐒₁)
    𝐒₂T = SparseMatrixCSC(size(base.𝐒₂, 1), size(base.𝐒₂, 2), copy(base.𝐒₂.colptr), copy(base.𝐒₂.rowval), T.(base.𝐒₂.nzval))
    return ThirdOrderInputs(∇₁T, ∇₂T, ∇₃T, 𝐒₁T, 𝐒₂T)
end

function build_third_order_inputs(p)
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, :third_order)

    SS_and_pars, (solution_error, _) = MacroModelling.get_NSSS_and_parameters(model, p, opts = opts, estimation = true)
    abs(solution_error) < opts.tol.NSSS_acceptance_tol || error("get_NSSS_and_parameters did not converge")

    ∇₁ = Matrix(MacroModelling.calculate_jacobian(p, SS_and_pars, model.caches, model.functions.jacobian))
    𝐒₁, _, solved1 = MacroModelling.calculate_first_order_solution(
        ∇₁,
        model.constants,
        model.workspaces,
        model.caches;
        initial_guess = model.caches.qme_solution,
        opts = opts,
    )
    solved1 || error("calculate_first_order_solution did not solve")

    ∇₂ = MacroModelling.calculate_hessian(p, SS_and_pars, model.caches, model.functions.hessian)
    𝐒₂, solved2 = MacroModelling.calculate_second_order_solution(
        ∇₁,
        ∇₂,
        𝐒₁,
        model.constants,
        model.workspaces,
        model.caches;
        initial_guess = model.caches.second_order_solution,
        opts = opts,
    )
    solved2 || error("calculate_second_order_solution did not solve")

    𝐒₂ *= model.constants.second_order.𝐔₂
    if !(𝐒₂ isa AbstractSparseMatrix)
        𝐒₂ = sparse(𝐒₂)
    end

    ∇₃ = MacroModelling.calculate_third_order_derivatives(p, SS_and_pars, model.caches, model.functions.third_order_derivatives)

    return ThirdOrderInputs(
        ∇₁,
        ∇₂ isa SparseMatrixCSC ? ∇₂ : sparse(∇₂),
        ∇₃ isa SparseMatrixCSC ? ∇₃ : sparse(∇₃),
        Matrix(𝐒₁),
        𝐒₂ isa SparseMatrixCSC ? 𝐒₂ : sparse(𝐒₂),
    )
end

function third_order_norm(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂)
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, :third_order)

    𝐒₃, solved3 = MacroModelling.calculate_third_order_solution(
        ∇₁,
        ∇₂,
        ∇₃,
        𝐒₁,
        𝐒₂,
        model.constants,
        model.workspaces,
        model.caches;
        initial_guess = model.caches.third_order_solution,
        opts = opts,
    )
    solved3 || error("calculate_third_order_solution did not solve")
    return norm(𝐒₃)
end

function summarize_diff(name, g_ref, g_test)
    Δ = g_test .- g_ref
    abs_max = maximum(abs, Δ)
    abs_mean = mean(abs, Δ)
    rel_norm = norm(Δ) / max(norm(g_ref), eps(Float64))
    println(name)
    println("  length              = ", length(g_test))
    println("  max abs diff        = ", abs_max)
    println("  mean abs diff       = ", abs_mean)
    println("  relative norm       = ", rel_norm)
    println()
end

function run_gradient_comparison_block(block_name, objective_fn, x0)
    println("\n", "="^70)
    println("  ", block_name)
    println("="^70, "\n")

    y0 = objective_fn(x0)
    println("Objective value: ", y0)
    println("Input length:    ", length(x0))
    println()

    g_fd = nothing
    g_fwd = nothing
    g_zyg = nothing

    if RUN_ZYG
        println("Computing Zygote gradient...")
        try
            g_zyg = Zygote.gradient(objective_fn, x0)[1]
            println("  done - norm(g_zyg) = ", norm(g_zyg))
        catch err
            println("  failed: ", sprint(showerror, err, catch_backtrace()))
        end
        println()
    end

    if RUN_FD
        println("Computing FiniteDifferences gradient...")
        fdm = FiniteDifferences.central_fdm(FDM_ORDER, 1, max_range = FDM_MAX_RANGE)
        fd_raw = FiniteDifferences.grad(fdm, objective_fn, x0)
        g_fd = fd_raw isa Tuple ? fd_raw[1] : fd_raw
        println("  done - norm(g_fd) = ", norm(g_fd))
        println()
    end

    if RUN_FWD
        println("Computing ForwardDiff gradient...")
        try
            g_fwd = ForwardDiff.gradient(objective_fn, x0)
            println("  done - norm(g_fwd) = ", norm(g_fwd))
        catch err
            println("  failed: ", sprint(showerror, err, catch_backtrace()))
        end
        println()
    end

    if g_fd !== nothing
        println("=== Comparisons (reference = FiniteDifferences) ===")
        if g_fwd !== nothing
            summarize_diff("ForwardDiff vs FiniteDifferences", g_fd, g_fwd)
        end
        if g_zyg !== nothing
            summarize_diff("Zygote vs FiniteDifferences", g_fd, g_zyg)
        end
    end

    if g_fwd !== nothing && g_zyg !== nothing
        summarize_diff("ForwardDiff vs Zygote", g_fwd, g_zyg)
    end

    if g_fd === nothing && g_fwd === nothing && g_zyg === nothing
        error("All gradient computations failed in block: $block_name")
    end
end

function main()
    # warm-up compile path
    MacroModelling.solve!(model, algorithm = :third_order, opts = opts)

    base = build_third_order_inputs(p0)

    if "grad1" in INPUT_BLOCKS
        dims = size(base.∇₁)
        x0 = vec(copy(base.∇₁))
        objective = x -> begin
            promoted = MacroModelling.@ignore_derivatives promote_inputs(base, eltype(x))
            ∇₁x = reshape(x, dims)
            third_order_norm(∇₁x, promoted.∇₂, promoted.∇₃, promoted.𝐒₁, promoted.𝐒₂)
        end
        run_gradient_comparison_block("Gradient of norm(S₃) wrt vec(∇₁)", objective, x0)
    end

    if "grad2" in INPUT_BLOCKS
        x0 = copy(base.∇₂.nzval)
        objective = x -> begin
            promoted = MacroModelling.@ignore_derivatives promote_inputs(base, eltype(x))
            ∇₂x = sparse_with_new_values(promoted.∇₂, x)
            third_order_norm(promoted.∇₁, ∇₂x, promoted.∇₃, promoted.𝐒₁, promoted.𝐒₂)
        end
        run_gradient_comparison_block("Gradient of norm(S₃) wrt ∇₂.nzval", objective, x0)
    end

    if "grad3" in INPUT_BLOCKS
        x0 = copy(base.∇₃.nzval)
        objective = x -> begin
            promoted = MacroModelling.@ignore_derivatives promote_inputs(base, eltype(x))
            ∇₃x = sparse_with_new_values(promoted.∇₃, x)
            third_order_norm(promoted.∇₁, promoted.∇₂, ∇₃x, promoted.𝐒₁, promoted.𝐒₂)
        end
        run_gradient_comparison_block("Gradient of norm(S₃) wrt ∇₃.nzval", objective, x0)
    end

    if "s2" in INPUT_BLOCKS
        x0 = copy(base.𝐒₂.nzval)
        objective = x -> begin
            promoted = MacroModelling.@ignore_derivatives promote_inputs(base, eltype(x))
            𝐒₂x = sparse_with_new_values(promoted.𝐒₂, x)
            third_order_norm(promoted.∇₁, promoted.∇₂, promoted.∇₃, promoted.𝐒₁, 𝐒₂x)
        end
        run_gradient_comparison_block("Gradient of norm(S₃) wrt 𝐒₂.nzval", objective, x0)
    end

    if "s1" in INPUT_BLOCKS
        dims = size(base.𝐒₁)
        x0 = vec(copy(base.𝐒₁))
        objective = x -> begin
            promoted = MacroModelling.@ignore_derivatives promote_inputs(base, eltype(x))
            𝐒₁x = reshape(x, dims)
            third_order_norm(promoted.∇₁, promoted.∇₂, promoted.∇₃, 𝐒₁x, promoted.𝐒₂)
        end
        run_gradient_comparison_block("Gradient of norm(S₃) wrt vec(𝐒₁)", objective, x0)
    end
end

main()
