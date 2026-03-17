using Revise
using MacroModelling
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using TimerOutputs: TimerOutput, @timeit
using ChainRulesCore: rrule, NoTangent

const MM = MacroModelling
const LL = LinearAlgebra
const ℒ = LinearAlgebra

function _perm_source_to_target_from_columns(P)
    n = size(P, 2)
    map = zeros(Int, n)
    if P isa SparseMatrixCSC
        @inbounds for src in 1:n
            for idx in P.colptr[src]:(P.colptr[src + 1] - 1)
                if !iszero(P.nzval[idx])
                    map[src] = P.rowval[idx]
                    break
                end
            end
        end
    else
        @inbounds for src in 1:n
            col = @view P[:, src]
            dst = findfirst(!iszero, col)
            map[src] = isnothing(dst) ? 0 : dst
        end
    end
    return map
end

function _accumulate_kron_A_entry!(∂A, Bσ, row_idx::Int, col_idx::Int, val,
                                   nrows::Int, n1::Int, n2::Int, m1::Int,
                                   const_n1n2::Int, const_n1n2m1::Int)
    linear_idx = (col_idx - 1) * nrows + row_idx
    i = (linear_idx - 1) % n1 + 1
    k = ((linear_idx - 1) ÷ n1) % n2 + 1
    j = ((linear_idx - 1) ÷ const_n1n2) % m1 + 1
    l = ((linear_idx - 1) ÷ const_n1n2m1) + 1
    @inbounds ∂A[k, l] += Bσ[i, j] * val
    return nothing
end

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2003.jl"))

model = Smets_Wouters_2003

# include(joinpath(@__DIR__, "..", "models", "FS2000.jl"))

# model = FS2000

parameters = copy(model.parameter_values)
opts = MM.merge_calculation_options(verbose = false)

# Set to true to execute the pullback immediately.
# Keep false to step through the closure manually in REPL.
# run_pullback_now = false

# -----------------------------------------------------------------------------
# Step 0: Build exact inputs passed to calculate_third_order_solution
# -----------------------------------------------------------------------------
MM.clear_solution_caches!(model, :third_order)

# Initialize derivative/function caches for third-order path once.
_, _, _, _, solved_warmup = MM.get_solution(model, parameters, algorithm = :third_order, verbose = false)
@assert solved_warmup "Warmup third-order solve failed."
MM.clear_solution_caches!(model, :third_order)

SS_and_pars, (solution_error, nsss_iters) = MM.get_NSSS_and_parameters(model, parameters, opts = opts)
@assert solution_error <= opts.tol.NSSS_acceptance_tol "NSSS solve did not satisfy acceptance tolerance."

∇₁ = MM.calculate_jacobian(parameters, SS_and_pars, model.caches, model.functions.jacobian, model.workspaces)

𝐒₁, qme_sol, solved1 = MM.calculate_first_order_solution(∇₁,
                                                         model.constants,
                                                         model.workspaces,
                                                         model.caches;
                                                         opts = opts,
                                                         initial_guess = model.caches.qme_solution)
@assert solved1 "First-order solution failed."

∇₂ = MM.calculate_hessian(parameters, SS_and_pars, model.caches, model.functions.hessian, model.workspaces)

𝐒₂, solved2 = MM.calculate_second_order_solution(∇₁,
                                                 ∇₂,
                                                 𝐒₁,
                                                 model.constants,
                                                 model.workspaces,
                                                 model.caches;
                                                 initial_guess = model.caches.second_order_solution,
                                                 opts = opts)
@assert solved2 "Second-order solution failed."

∇₃ = MM.calculate_third_order_derivatives(parameters,
                                          SS_and_pars,
                                          model.caches,
                                          model.functions.third_order_derivatives,
                                          model.workspaces)

# -----------------------------------------------------------------------------
# Step 1: Primal + pullback for calculate_third_order_solution
# -----------------------------------------------------------------------------
third_out, third_pb = rrule(MM.calculate_third_order_solution,
                            ∇₁,
                            ∇₂,
                            ∇₃,
                            𝐒₁,
                            𝐒₂,
                            model.constants,
                            model.workspaces,
                            model.caches;
                            initial_guess = model.caches.third_order_solution,
                            opts = opts)

𝐒₃_raw, solved3 = third_out
@assert solved3 "Third-order primal solve in rrule forward pass failed."

# Objective from benchmark/bench.jl:
# norm(get_solution(model, x, algorithm = :third_order)[4] * model.constants.third_order.𝐔₃)
𝐒₃_full = 𝐒₃_raw * model.constants.third_order.𝐔₃
loss = LL.norm(𝐒₃_full)

# Seed cotangent for 𝐒₃_raw from f(X) = norm(X * U₃):
# ∂f/∂X = (X*U₃ / norm(X*U₃)) * U₃'
scale = max(loss, eps(eltype(loss)))
∂𝐒₃_raw_rr = (𝐒₃_full / scale) * model.constants.third_order.𝐔₃'

println("third_order_solved=", solved3,
        " size(𝐒₃_raw)=", size(𝐒₃_raw),
        " nnz(𝐒₃_raw)=", nnz(sparse(𝐒₃_raw)))
println("loss_norm_S3_full=", loss)

println("Ready to walk through the pullback closure.")
println("Manual call:")
println("  third_grads = third_pb((∂𝐒₃_raw_rr, NoTangent()))")
println("  ∂∇₁ = third_grads[2]; ∂∇₂ = third_grads[3]; ∂∇₃ = third_grads[4]; ∂𝐒₁ = third_grads[5]; ∂𝐒₂ = third_grads[6]")

# -----------------------------------------------------------------------------
# Step 2: REPL-style manual chain from ∂𝐒₃_raw_rr to parameter tangents
# Mirrors pullback_3rd in rrules.jl for get_solution(..., algorithm=:third_order)
# -----------------------------------------------------------------------------
estimation = true
nVar = length(model.constants.post_model_macro.var)

nsss_out_rr, nsss_pb = rrule(MM.get_NSSS_and_parameters,
                                                         model,
                                                         parameters;
                                                         opts = opts,
                                                         estimation = estimation)
SS_and_pars_rr = nsss_out_rr[1]

∇₁_rr, jac_pb = rrule(MM.calculate_jacobian,
                                          parameters,
                                          SS_and_pars_rr,
                                          model.caches,
                                          model.functions.jacobian,
                                          model.workspaces)

first_out_rr, first_pb = rrule(MM.calculate_first_order_solution,
                                                           ∇₁_rr,
                                                           model.constants,
                                                           model.workspaces,
                                                           model.caches;
                                                           opts = opts,
                                                           initial_guess = model.caches.qme_solution)
𝐒₁_rr = first_out_rr[1]

∇₂_rr, hess_pb = rrule(MM.calculate_hessian,
                                           parameters,
                                           SS_and_pars_rr,
                                           model.caches,
                                           model.functions.hessian,
                                           model.workspaces)

second_out_rr, second_pb = rrule(MM.calculate_second_order_solution,
                                                                 ∇₁_rr,
                                                                 ∇₂_rr,
                                                                 𝐒₁_rr,
                                                                 model.constants,
                                                                 model.workspaces,
                                                                 model.caches;
                                                                 initial_guess = model.caches.second_order_solution,
                                                                 opts = opts)
𝐒₂_raw_rr = second_out_rr[1]

∇₃_rr, third_deriv_pb = rrule(MM.calculate_third_order_derivatives,
                                                          parameters,
                                                          SS_and_pars_rr,
                                                          model.caches,
                                                          model.functions.third_order_derivatives,
                                                          model.workspaces)

# third_out_rr, third_pb_rr = rrule(MM.calculate_third_order_solution,
#                                                                   ∇₁_rr,
#                                                                   ∇₂_rr,
#                                                                   ∇₃_rr,
#                                                                   𝐒₁_rr,
#                                                                   𝐒₂_raw_rr,
#                                                                   model.constants,
#                                                                   model.workspaces,
#                                                                   model.caches;
#                                                                   initial_guess = model.caches.third_order_solution,
#                                                                   opts = opts)
# 𝐒₃_raw_rr = third_out_rr[1]
# @assert third_out_rr[2] "third_pb_rr forward pass failed."

𝐒₃_full_rr = 𝐒₃_raw * model.constants.third_order.𝐔₃
loss_rr = LL.norm(𝐒₃_full_rr)
scale_rr = max(loss_rr, eps(eltype(loss_rr)))
∂𝐒₃_raw_rr = (𝐒₃_full_rr / scale_rr) * model.constants.third_order.𝐔₃'

println("manual-chain seed ready: norm(S3*U3)=", loss_rr)

# Start here in REPL when stepping manually:
#   ∂𝐒₃_raw_rr
pb_seed_rr = (∂𝐒₃_raw_rr, NoTangent())

println("Pullback REPL entrypoint ready.")
println("Direct call:")
println("  third_grads_rr = third_pb_rr(pb_seed_rr)")

# Bindings to run copied rrule body snippets directly in this script/REPL.
# These provide the same names used inside rrules.jl.
workspaces = model.workspaces
constants = model.constants
cache = model.caches
initial_guess = model.caches.third_order_solution

S = eltype(∇₁_rr)
R = eltype(parameters)

∇₁ = ∇₁_rr
∇₂ = ∇₂_rr
∇₃ = ∇₃_rr
𝑺₁ = 𝐒₁_rr
𝐒₂ = 𝐒₂_raw_rr

Higher_order_workspace = MM.Higher_order_workspace
choose_matrix_format = MM.choose_matrix_format
ensure_higher_order_solution_buffers! = MM.ensure_higher_order_solution_buffers!
compressed_permuted_mixed_kron = MM.compressed_permuted_mixed_kron
compressed_kron³ = MM.compressed_kron³
mat_mult_kron = MM.mat_mult_kron
fill_kron_adjoint! = MM.fill_kron_adjoint!
fill_kron_adjoint_∂A! = MM.fill_kron_adjoint_∂A!
solve_sylvester_equation = MM.solve_sylvester_equation
ensure_third_order_pullback_workspaces! = MM.ensure_third_order_pullback_workspaces!
compressed_permuted_mixed_kron_pullback! = MM.compressed_permuted_mixed_kron_pullback!
compressed_kron³_pullback! = MM.compressed_kron³_pullback!

# -----------------------------------------------------------------------------
# Full third_order_solution_pullback reference from
# src/custom_autodiff_rules/rrules.jl
#
# This is the full closure body so you can follow the same logic in this file
# while stepping from pb_seed_rr = (∂𝐒₃_raw_rr, NoTangent()).
# -----------------------------------------------------------------------------

    # --- workspace / constants ---------------------------------------------------
    if !(eltype(workspaces.third_order.Ŝ) == S)
        workspaces.third_order = Higher_order_workspace(T = S)
    end
    ℂ = workspaces.third_order
    M₂ = constants.second_order
    M₃ = constants.third_order
    T = constants.post_model_macro

    # Expand compressed inputs to full space for internal computation
    ∇₂ = ∇₂ * M₂.𝐔∇₂
    𝐒₂ = sparse(𝐒₂ * M₂.𝐔₂)::SparseMatrixCSC{S, Int}

    i₊ = T.future_not_past_and_mixed_idx
    i₋ = T.past_not_future_and_mixed_idx
    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    ensure_higher_order_solution_buffers!(ℂ, n, nₑ₋)

    initial_guess_sylv = if length(initial_guess) == 0
        zeros(S, 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{S} ? initial_guess : Matrix{S}(initial_guess)
    else
        zeros(S, 0, 0)
    end

    # --- forward pass (mirrors the primal, but stores intermediates) ---------------

    # 1st-order solution with zero-column
    𝐒₁ = ℂ.𝐒₁::Matrix{S}
    copyto!(@view(𝐒₁[:,1:n₋]), @view(𝑺₁[:,1:n₋]))
    fill!(@view(𝐒₁[:,n₋+1]), zero(S))
    copyto!(@view(𝐒₁[:,n₋+2:end]), @view(𝑺₁[:,n₋+1:end]))

    𝐒₁₋╱𝟏ₑ = ℂ.𝐒₁₋╱𝟏ₑ::Matrix{S}
    copyto!(@view(𝐒₁₋╱𝟏ₑ[1:n₋,:]), @view(𝐒₁[i₋,:]))
    fill!(@view(𝐒₁₋╱𝟏ₑ[n₋+1:end,:]), zero(S))
    @inbounds 𝐒₁₋╱𝟏ₑ[n₋+1,n₋+1] = one(S)
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]]

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]; zeros(n₋ + n + nₑ, nₑ₋)]
    𝐒₁₊╱𝟎 = choose_matrix_format(𝐒₁₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * M₂.𝐈ₙ₋ - ∇₁[:,range(1,n) .+ n₊]

    ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

    if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
        return (∇₁₊𝐒₁➕∇₁₀, false), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    spinv = inv(∇₁₊𝐒₁➕∇₁₀lu)
    spinv = choose_matrix_format(spinv)

    ∇₁₊ = @views ∇₁[:,1:n₊] * M₂.𝐈ₙ₊

    A = spinv * ∇₁₊

    # --- B matrix -----------------------------------------------------------------
    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)

    B = compressed_permuted_mixed_kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔,
                                       sparse_preallocation = ℂ.tmp_sparse_prealloc7)

    B += compressed_kron³(𝐒₁₋╱𝟏ₑ, tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc1)

    # --- 𝐗₃ (C-matrix ingredients) -----------------------------------------------
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:]; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
                                          𝐒₂
                                          zeros(n₋ + nₑ, nₑ₋^2)]
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold = 0.0, min_length = 10, tol = opts.tol.droptol)

    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:]; zeros(n₋ + n + nₑ, nₑ₋^2)]

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    S1p0_kron_sigma = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔
    tmpkron22 = compressed_permuted_mixed_kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                                               S1p0_kron_sigma,
                                               sparse_preallocation = ℂ.tmp_sparse_prealloc6)

    𝐒₂₊╱𝟎 = choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ∇₁₊ = choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    𝐒₂₋╱𝟎 = [𝐒₂[i₋,:]; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    # Terms (a)+(b): ∇₂ * kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎) * [tmpkron2 + 𝐏₁ₗ * tmpkron2 * 𝐏₁ᵣ] * 𝐏𝐂₃
    tmpkron2 = ℒ.kron(M₂.𝛔, choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.droptol))
    D_ab = (tmpkron2 + M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ) * M₃.𝐏𝐂₃
    𝐗₃ = mat_mult_kron(∇₂, collect(𝐒₁₊╱𝟎), collect(𝐒₂₊╱𝟎), D_ab, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc2)

    # Term (c): ∇₂ * kron(⎸𝐒₁..⎹, ⎸𝐒₂k..⎹) * 𝐏𝐂₃
    𝐗₃ += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, M₃.𝐏𝐂₃, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc3)

    # Term (d): ∇₂ * kron(⎸𝐒₁..⎹, 𝐒₂₊╱𝟎*𝛔) * 𝐏𝐂₃
    S2p0_sigma = 𝐒₂₊╱𝟎 * M₂.𝛔
    𝐗₃ += mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(S2p0_sigma), M₃.𝐏𝐂₃, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)

    # Term (e): ∇₁₊ * 𝐒₂ * kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎) * 𝐏𝐂₃
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.droptol)
    mm_𝐒₂_kron = mat_mult_kron(𝐒₂, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)
    𝐗₃ += ∇₁₊ * mm_𝐒₂_kron * M₃.𝐏𝐂₃

    𝐗₃ += ∇₃ * tmpkron22

    # Compute compressed_kron³(aux) WITHOUT rowmask: the pullback needs ∂∇₃ at ALL
    # positions (including currently-zero columns of ∇₃) so that gradients flow
    # correctly through calculate_third_order_derivatives back to parameters.
    ck3_aux_mat = compressed_kron³(aux, rowmask = M₃.∇₃_rowmask, tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc5)
    ck3_aux = ∇₃ * ck3_aux_mat
    𝐗₃ += ck3_aux
    
    C = spinv * 𝐗₃

    # --- solve Sylvester  A·𝐒₃·B + C = 𝐒₃ ----------------------------------------
    𝐒₃, solved = solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                            initial_guess = initial_guess_sylv,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
                                            tol = opts.tol.sylvester_tol,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose)

    𝐒₃ = choose_matrix_format(𝐒₃, multithreaded = false, tol = opts.tol.droptol)
    𝐒₃_stable = copy(𝐒₃)

    if !solved
        return (𝐒₃_stable, solved), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # cache update (same as primal)
    if 𝐒₃_stable isa Matrix{S} && cache.third_order_solution isa Matrix{S} && size(cache.third_order_solution) == size(𝐒₃_stable)
        copyto!(cache.third_order_solution, 𝐒₃_stable)
    elseif 𝐒₃_stable isa SparseMatrixCSC{S, Int} && cache.third_order_solution isa SparseMatrixCSC{S, Int} &&
           size(cache.third_order_solution) == size(𝐒₃_stable) &&
           cache.third_order_solution.colptr == 𝐒₃_stable.colptr &&
           cache.third_order_solution.rowval == 𝐒₃_stable.rowval
        copyto!(cache.third_order_solution.nzval, 𝐒₃_stable.nzval)
    else
        cache.third_order_solution = 𝐒₃_stable
    end

    # --- precompute transposed constants for pullback -----------------------------
    # Use pre-cached transposes from constants (computed once at model compile time)
    𝐏𝐂₃t = M₃.𝐏𝐂₃ᵀ
    𝛔t  = M₂.𝛔ᵀ
    𝐔∇₂t = M₂.𝐔∇₂ᵀ
    𝐔₂t  = M₂.𝐔₂ᵀ

    # Use pre-cached transposes of permutation matrices (for out2 terms a,b pullback)
    M₃𝐏₁ₗt = M₃.𝐏₁ₗᵀ
    M₃𝐏₁ᵣt = M₃.𝐏₁ᵣᵀ

    # Materialized transposes of forward-pass intermediates
    ∇₂t = choose_matrix_format(∇₂')
    ∇₃t = choose_matrix_format(∇₃')
    D_ab_t = choose_matrix_format(D_ab')
    tmpkron22_t = choose_matrix_format(tmpkron22')
    ck3_aux_mat_t = choose_matrix_format(ck3_aux_mat')
    𝐒₂t = choose_matrix_format(𝐒₂', density_threshold = 1.0)
    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋')
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎')
    S2p0_sigma_t = choose_matrix_format(S2p0_sigma')

    mm_𝐒₂_kron_t = choose_matrix_format(mm_𝐒₂_kron')

    # --- ensure pullback workspace buffers ---
    ensure_third_order_pullback_workspaces!(ℂ, S, T, M₂, M₃)

    tmpkron22_ck3_aux_mat_t = choose_matrix_format(tmpkron22_t + ck3_aux_mat_t)




∂𝐒₃_solved = pb_seed_rr


# @profview begin
pullback_timer = TimerOutput()
# for i in 1:10
# function third_order_solution_pullback(∂𝐒₃_solved)
@timeit pullback_timer "total" begin
    ∂𝐒₃ = ∂𝐒₃_solved[1]

        # --- adjoint Sylvester:  Aᵀ ∂C_adj Bᵀ + ∂𝐒₃ = ∂C_adj --------------------
    @timeit pullback_timer "adjoint_sylvester" begin
        ∂C_adj, slvd = solve_sylvester_equation(A', B', Matrix{Float64}(∂𝐒₃), ℂ.sylvester_workspace,
                                                                                          sylvester_algorithm = opts.sylvester_algorithm³,
                                                                                          tol = opts.tol.sylvester_tol,
                                                                                          acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                                                                          verbose = opts.verbose)
      
        ∂C_adj = choose_matrix_format(∂C_adj)
    end

        # --- Initialize all gradient accumulators ---
    @timeit pullback_timer "initialize_accumulators" begin
        # Dense workspace temporaries (overwritten by mul! each call)
        ∂𝐗₃           = ℂ.∂𝐗₃_3rd
        ∂A             = ℂ.∂A_3rd
        ∂B_from_sylv   = ℂ.∂B_sylv_3rd
        ∂out2          = ℂ.∂out2_3rd
        ∇₂t_∂out2     = ℂ.∇₂t_∂out2_3rd
        mul_tmp        = ℂ.mul_tmp_3rd
        ∂∇₁₊𝐒₁➕∇₁₀   = ℂ.∂∇₁₊𝐒₁➕∇₁₀_3rd

        # Dense workspace accumulators (need zeroing)
        ∂spinv         = ℂ.∂spinv_3rd
        ∂∇₁            = ℂ.∂∇₁_3rd;  fill!(∂∇₁, zero(S))
        ∂𝐒₁₃           = ℂ.∂𝐒₁_3rd;  fill!(∂𝐒₁₃, zero(S))

        # Sparse-preserving gradient accumulators (reuse workspace buffers)
        ∂𝐒₂            = zero(𝐒₂)  # sparse — must stay fresh

        ∂𝐒₁₊╱𝟎_tmp    = ℂ.∂𝐒₁₊╱𝟎_tmp_3rd;  fill!(∂𝐒₁₊╱𝟎_tmp, zero(S))
        ∂𝐒₂₊╱𝟎        = ℂ.∂𝐒₂₊╱𝟎_3rd;       fill!(∂𝐒₂₊╱𝟎, zero(S))
        ∂L_c           = ℂ.∂L_c_3rd;          fill!(∂L_c, zero(S))
        ∂R_c           = ℂ.∂R_c_3rd;          fill!(∂R_c, zero(S))
        ∂L_d           = ℂ.∂L_d_3rd;          fill!(∂L_d, zero(S))
        ∂R_d           = ℂ.∂R_d_3rd;          fill!(∂R_d, zero(S))
        ∂𝐒₁₋╱𝟏ₑ_t8   = ℂ.∂𝐒₁₋╱𝟏ₑ_t8_3rd;  fill!(∂𝐒₁₋╱𝟏ₑ_t8, zero(S))
        ∂𝐒₂₋╱𝟎        = ℂ.∂𝐒₂₋╱𝟎_3rd;       fill!(∂𝐒₂₋╱𝟎, zero(S))
        ∂𝐒₁₋╱𝟏ₑ₃     = ℂ.∂𝐒₁₋╱𝟏ₑ_3rd;     fill!(∂𝐒₁₋╱𝟏ₑ₃, zero(S))
        ∂𝐒₁₊╱𝟎₃      = ℂ.∂𝐒₁₊╱𝟎_3rd;       fill!(∂𝐒₁₊╱𝟎₃, zero(S))
        ∂S1S1_stack    = ℂ.∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋_3rd; fill!(∂S1S1_stack, zero(S))
        ∂aux           = ℂ.∂aux_3rd;          fill!(∂aux, zero(S))
        ∂𝛔_discard     = ℂ.∂𝛔_discard_3rd;   fill!(∂𝛔_discard, zero(S))
        end

        # --- gradient of A, B, C from 𝐒₃ = A·𝐒₃·B + C ---------------------------
        @timeit pullback_timer "backprop_A_B_C" begin
        # ∂A = ∂C_adj * B' * 𝐒₃_stable' — use ∂𝐗₃ as temp for intermediate
        ℒ.mul!(∂𝐗₃, ∂C_adj, B')
        ℒ.mul!(∂A, ∂𝐗₃, 𝐒₃_stable')
        # ∂B_from_sylv = 𝐒₃_stable' * A' * ∂C_adj — reuse ∂𝐗₃ as temp
        ℒ.mul!(∂𝐗₃, A', ∂C_adj)
        ℒ.mul!(∂B_from_sylv, 𝐒₃_stable', ∂𝐗₃)
        # ∂𝐗₃ = spinv' * ∂C_adj
        ∂𝐗₃ = choose_matrix_format(spinv' * ∂C_adj, density_threshold = 1.0, min_length = 0)

        # C = spinv * 𝐗₃  →  ∂spinv
        # A = spinv * ∇₁₊  →  ∂spinv accumulation
        ℒ.mul!(∂spinv, ∂C_adj, 𝐗₃')
        ℒ.mul!(∂spinv, ∂A, ∇₁₊', 1, 1)
        end

        # =====================================================================
        #  ∂∇₃  (linear: ∇₃ appears in two additive terms of 𝐗₃)
        # =====================================================================
        @timeit pullback_timer "nabla3" begin
        ∂∇₃ = ∂𝐗₃ * tmpkron22_ck3_aux_mat_t
        end

        # =====================================================================
        #  ∂∇₂  (∇₂ is linear in out2 → 𝐗₃_pre → 𝐗₃)
        # =====================================================================
        @timeit pullback_timer "nabla2" begin
        ℒ.mul!(∂out2, ∂𝐗₃, 𝐏𝐂₃t)

        ∂mid_ab = ∂𝐗₃ * D_ab_t
        ∂∇₂ = mat_mult_kron(∂mid_ab, collect(𝐒₁₊╱𝟎'), collect(𝐒₂₊╱𝟎'))
        ∂∇₂ = ∂∇₂ + mat_mult_kron(∂out2, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t)
        ∂∇₂ = ∂∇₂ + mat_mult_kron(∂out2, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, S2p0_sigma_t)
        end

        # =====================================================================
        #  ∂𝐒₂  (𝐒₂ enters out2 via several stacking matrices)
        # =====================================================================
        @timeit pullback_timer "S2" begin
        ℒ.mul!(∇₂t_∂out2, ∇₂t, ∂out2)
        ∂tmpkron1 = (∇₂t * ∂mid_ab)
        fill_kron_adjoint!(∂𝐒₂₊╱𝟎, ∂𝐒₁₊╱𝟎_tmp, ∂tmpkron1, 𝐒₂₊╱𝟎, 𝐒₁₊╱𝟎)
        @views ∂𝐒₂[i₊,:] .+= ∂𝐒₂₊╱𝟎[1:length(i₊),:]

        ∂kron_c = (∇₂t_∂out2)
        fill_kron_adjoint!(∂R_c, ∂L_c, ∂kron_c, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)

        n₊_len = length(i₊)
        ∂top_block = ∂R_c[1:n₊_len, :]
        @views ∂𝐒₂[i₊,:] .+= ∂top_block * kron𝐒₁₋╱𝟏ₑ'
        ∂𝐒₂_padded = 𝐒₁' * ℒ.I(n)[:,i₊] * ∂top_block
        @views ∂𝐒₂[i₋,:] .+= ∂𝐒₂_padded[1:n₋, :]
        @views ∂𝐒₂ .+= ∂R_c[n₊_len .+ (1:n), :]

        fill_kron_adjoint!(∂R_d, ∂L_d, ∂kron_c, S2p0_sigma, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
        ∂𝐒₂₊╱𝟎_d = ∂R_d * 𝛔t
        @views ∂𝐒₂[i₊,:] .+= ∂𝐒₂₊╱𝟎_d[1:length(i₊),:]

        tmp_t8 = ∇₁₊' * ∂out2
        ∂𝐒₂ = ∂𝐒₂ + mat_mult_kron(tmp_t8, collect(𝐒₁₋╱𝟏ₑ'), collect(𝐒₂₋╱𝟎'))
        ∂kron_term8 = ((∇₁₊ * 𝐒₂)' * ∂out2)
        fill_kron_adjoint!(∂𝐒₂₋╱𝟎, ∂𝐒₁₋╱𝟏ₑ_t8, ∂kron_term8, 𝐒₂₋╱𝟎, 𝐒₁₋╱𝟏ₑ)
        @views ∂𝐒₂[i₋,:] .+= ∂𝐒₂₋╱𝟎[1:n₋,:]
        end

        # =====================================================================
        #  ∂∇₁
        # =====================================================================
        @timeit pullback_timer "nabla1" begin
        ℒ.mul!(mul_tmp, spinv', ∂spinv)
        ℒ.mul!(∂∇₁₊𝐒₁➕∇₁₀, mul_tmp, spinv')
        ℒ.rmul!(∂∇₁₊𝐒₁➕∇₁₀, -1)

        ∂∇₁[:,1:n₊] -= ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋] * 𝐒₁[i₊,1:n₋]'
        ∂∇₁[:,range(1,n) .+ n₊] -= ∂∇₁₊𝐒₁➕∇₁₀

        ∂∇₁₊ = ℂ.∂∇₁₊_3rd
        ℒ.mul!(∂∇₁₊, spinv', ∂A)
        ℒ.mul!(∂∇₁₊, ∂out2, mm_𝐒₂_kron_t, 1, 1)
        ∂∇₁[:,1:n₊] += ∂∇₁₊ * ℒ.I(n)[:,i₊]
        end

        # =====================================================================
        #  ∂𝑺₁
        # =====================================================================
        @timeit pullback_timer "S1" begin
        @timeit pullback_timer "seed_stack" begin
        ℒ.axpy!(1, ∂L_c, ∂S1S1_stack)
        ℒ.axpy!(1, ∂L_d, ∂S1S1_stack)
        end

        @timeit pullback_timer "tmpkron22_pullback" begin
        ∂tmpkron22 = (∇₃t * ∂𝐗₃)
        ∂S1S1_from_ck = ℂ.∂S1S1_from_ck_3rd
        fill!(∂S1S1_from_ck, zero(S))
        ∂S1p0_kron_sigma = ℂ.∂S1p0_kron_sigma_3rd
        fill!(∂S1p0_kron_sigma, zero(S))
        compressed_permuted_mixed_kron_pullback!(∂S1S1_from_ck,
                                                 ∂S1p0_kron_sigma,
                                                 ∂tmpkron22,
                                                 ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                                                 S1p0_kron_sigma;
                                                 tol = opts.tol.droptol)
                            end

                            @timeit pullback_timer "S1p0_kron_adjoint" begin
        ∂S1p0_kron = (∂S1p0_kron_sigma * 𝛔t)
        ∂S1p0_left = ℂ.∂S1p0_left_3rd
        fill!(∂S1p0_left, zero(S))
        ∂S1p0_right = ℂ.∂S1p0_right_3rd
        fill!(∂S1p0_right, zero(S))
        fill_kron_adjoint!(∂S1p0_left, ∂S1p0_right, ∂S1p0_kron, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)

        ℒ.axpy!(1, ∂S1S1_from_ck, ∂S1S1_stack)
        ℒ.axpy!(1, ∂S1p0_left, ∂𝐒₁₊╱𝟎₃)
        ℒ.axpy!(1, ∂S1p0_right, ∂𝐒₁₊╱𝟎₃)
        end

        @timeit pullback_timer "ck3_aux_pullback" begin
        ∂ck3_aux = collect(∇₃t * ∂𝐗₃)
        compressed_kron³_pullback!(∂aux, ∂ck3_aux, aux)
        ℒ.mul!(∂S1S1_stack, M₃.𝐒𝐏', ∂aux, 1, 1)

        ℒ.axpy!(1, ∂𝐒₁₊╱𝟎_tmp, ∂𝐒₁₊╱𝟎₃)
        end

        @timeit pullback_timer "B_pullback" begin
        compressed_permuted_mixed_kron_pullback!(∂𝐒₁₋╱𝟏ₑ₃, ∂𝛔_discard, ∂B_from_sylv, 𝐒₁₋╱𝟏ₑ, M₂.𝛔; tol = opts.tol.droptol)
        compressed_kron³_pullback!(∂𝐒₁₋╱𝟏ₑ₃, ∂B_from_sylv, 𝐒₁₋╱𝟏ₑ)
        end

        @timeit pullback_timer "nabla2_cross_term" begin
        @timeit pullback_timer "build_tmp_a" begin
        Gt = sparse(∇₂t_∂out2')
        B1 = collect(𝐒₁₊╱𝟎)
        C1 = collect(𝐒₂₊╱𝟎)

        n_rowB = size(B1, 1)
        n_colB = size(B1, 2)
        n_rowC = size(C1, 1)
        n_colC = size(C1, 2)
        nrows_tmp = n_colB * n_colC

        Bσ = collect(M₂.𝛔)
        n1, m1 = size(Bσ)
        n2 = size(∂𝐒₁₋╱𝟏ₑ₃, 1)
        const_n1n2 = n1 * n2
        const_n1n2m1 = n1 * n2 * m1

        row_map = _perm_source_to_target_from_columns(M₃𝐏₁ₗt)
        col_map = _perm_source_to_target_from_columns(M₃𝐏₁ᵣt')

        Ā = zeros(S, n_rowC, n_rowB)
        ĀB = zeros(S, n_rowC, n_colB)
        CĀB = zeros(S, n_colC, n_colB)

        rv = Gt isa SparseMatrixCSC ? Gt.rowval : Gt.A.rowval
        active_rows = unique(rv)
        for src_col in active_rows
            @views copyto!(Ā, Gt[src_col, :])
            ℒ.mul!(ĀB, Ā, B1)
            ℒ.mul!(CĀB, C1', ĀB)
            for tmp_row in eachindex(CĀB)
                val = CĀB[tmp_row]
                abs(val) > eps(S) || continue

                _accumulate_kron_A_entry!(∂𝐒₁₋╱𝟏ₑ₃, Bσ, tmp_row, src_col, val,
                                          nrows_tmp, n1, n2, m1,
                                          const_n1n2, const_n1n2m1)

                perm_row = row_map[tmp_row]
                perm_col = col_map[src_col]
                _accumulate_kron_A_entry!(∂𝐒₁₋╱𝟏ₑ₃, Bσ, perm_row, perm_col, val,
                                          nrows_tmp, n1, n2, m1,
                                          const_n1n2, const_n1n2m1)
            end
        end
        end

        @timeit pullback_timer "axpy_t8" begin
        ℒ.axpy!(1, ∂𝐒₁₋╱𝟏ₑ_t8, ∂𝐒₁₋╱𝟏ₑ₃)
        end

        @timeit pullback_timer "top_block_kron" begin
        ∂kron𝐒₁₋╱𝟏ₑ₃ = (𝐒₂t * ℒ.I(n)[:,i₊] * ∂top_block)
        fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ₃, ∂𝐒₁₋╱𝟏ₑ₃, ∂kron𝐒₁₋╱𝟏ₑ₃, 𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)
        end

        @timeit pullback_timer "final_assembly" begin
        S2_padded = [𝐒₂[i₋,:]; zeros(S, nₑ + 1, nₑ₋^2)]
        @views ∂𝐒₁₃[i₊,:] .+= ∂top_block * S2_padded'

        n₊l = length(i₊)
        ∂top_S1S1 = ∂S1S1_stack[1:n₊l, :]
        @views ∂𝐒₁₃[i₊,:] .+= ∂top_S1S1 * 𝐒₁₋╱𝟏ₑ'
        ∂𝐒₁₋╱𝟏ₑ₃ .+= 𝐒₁' * ℒ.I(n)[:,i₊] * ∂top_S1S1
        @views ∂𝐒₁₃ .+= ∂S1S1_stack[n₊l .+ (1:n), :]

        @views ∂𝐒₁₃[i₊,:] .+= ∂𝐒₁₊╱𝟎₃[1:n₊l,:]
        @views ∂𝐒₁₃[i₋,:] .+= ∂𝐒₁₋╱𝟏ₑ₃[1:length(i₋),:]
        ∂𝐒₁₃[i₊,1:n₋] -= ∇₁[:,1:n₊]' * ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋]

        ∂𝑺₁ = [∂𝐒₁₃[:,1:n₋] ∂𝐒₁₃[:,n₋+2:end]]
        end
        end
        end

        # Map ∂∇₂ and ∂𝐒₂ back to compressed space
        @timeit pullback_timer "compress_outputs" begin
        ∂∇₂ = ∂∇₂ * 𝐔∇₂t
        ∂𝐒₂ = ∂𝐒₂ * 𝐔₂t
        end

        manual_third_pullback_grads = (NoTangent(), ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent())
    end
# end
# end
pullback_timer

# Actual pullback execution for calculate_third_order_solution rrule.
# This runs the real closure code from src/custom_autodiff_rules/rrules.jl.
# Start from pb_seed_rr (which contains ∂𝐒₃_raw_rr) and inspect each object below.

third_grads_rr = third_pb_rr(pb_seed_rr)

∂∇₁_from_3rd_rr = third_grads_rr[2]
∂∇₂_from_3rd_rr = third_grads_rr[3]
∂∇₃_from_3rd_rr = third_grads_rr[4]
∂𝐒₁_from_3rd_rr = third_grads_rr[5]
∂𝐒₂_from_3rd_rr = third_grads_rr[6]

∂parameters_manual = zeros(eltype(parameters), length(parameters))
∂SS_and_pars_manual = zeros(eltype(parameters), length(SS_and_pars_rr))

third_deriv_grads_rr = third_deriv_pb(∂∇₃_from_3rd_rr)
∂parameters_manual .+= third_deriv_grads_rr[2]
∂SS_and_pars_manual .+= third_deriv_grads_rr[3]

∂𝐒₂_total_rr = Matrix(∂𝐒₂_from_3rd_rr)
second_grads_rr = second_pb((∂𝐒₂_total_rr, NoTangent()))
∂∇₁_from_2nd_rr = second_grads_rr[2]
∂∇₂_from_2nd_rr = second_grads_rr[3]
∂𝐒₁_from_2nd_rr = second_grads_rr[4]

∂∇₂_total_rr = ∂∇₂_from_3rd_rr + ∂∇₂_from_2nd_rr
hess_grads_rr = hess_pb(∂∇₂_total_rr)
∂parameters_manual .+= hess_grads_rr[2]
∂SS_and_pars_manual .+= hess_grads_rr[3]

∂𝐒₁_total_rr = ∂𝐒₁_from_3rd_rr + ∂𝐒₁_from_2nd_rr
first_grads_rr = first_pb((∂𝐒₁_total_rr, NoTangent(), NoTangent()))

∂∇₁_total_rr = ∂∇₁_from_3rd_rr + ∂∇₁_from_2nd_rr + first_grads_rr[2]
jac_grads_rr = jac_pb(∂∇₁_total_rr)
∂parameters_manual .+= jac_grads_rr[2]
∂SS_and_pars_manual .+= jac_grads_rr[3]

nsss_grads_rr = nsss_pb((∂SS_and_pars_manual, NoTangent()))
∂parameters_manual .+= nsss_grads_rr[3]

println("manual_chain parameter tangent norm=", LL.norm(∂parameters_manual))
println("\nTimerOutputs report for manual third_order_solution_pullback walkthrough:")
show(pullback_timer)
println()

# -----------------------------------------------------------------------------
# Step 3: Compare with real pullback of bench objective path
# bench objective path: norm(get_solution(model, x, algorithm=:third_order)[4] * U₃)
# -----------------------------------------------------------------------------
sol_out_rr, sol_pb_rr = rrule(MM.get_solution,
                                                          model,
                                                          parameters;
                                                          algorithm = :third_order,
                                                          verbose = false)

𝐒₃_sol_raw = sol_out_rr[4]
𝐒₃_sol_full = 𝐒₃_sol_raw * model.constants.third_order.𝐔₃
loss_sol = LL.norm(𝐒₃_sol_full)
scale_sol = max(loss_sol, eps(eltype(loss_sol)))
∂𝐒₃_sol_raw = (𝐒₃_sol_full / scale_sol) * model.constants.third_order.𝐔₃'

sol_grads_rr = sol_pb_rr((NoTangent(), NoTangent(), NoTangent(), ∂𝐒₃_sol_raw, NoTangent()))
∂parameters_real = sol_grads_rr[3]

Δp = ∂parameters_manual - ∂parameters_real
max_abs_diff_params = maximum(abs, Δp)
rel_diff_params = LL.norm(Δp) / max(LL.norm(∂parameters_real), eps(eltype(loss_sol)))

println("real_pullback parameter tangent norm=", LL.norm(∂parameters_real))
println("manual_vs_real params: max_abs_diff=", max_abs_diff_params,
                " rel_diff=", rel_diff_params)
