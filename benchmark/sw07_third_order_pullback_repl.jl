#=
  REPL-style script to step through the third-order solution pullback
  for the Smets–Wouters 2007 model.

  Objective (same as bench.jl):
      f(params) = norm( S3_raw * U3 )
  Tangent wrt S3_raw:
      ∂f/∂S3 = (S3*U3 / norm(S3*U3)) * U3'

  This script:
    1. Builds all primal inputs (∇₁, ∇₂, ∇₃, 𝑺₁, 𝑺₂)
    2. Runs the rrule forward pass → captures S3_raw + closure variables
    3. Computes the cotangent seed ∂S3_raw from norm(S3 * U3)
    4. PASTES THE PULLBACK CODE INLINE so you can step through it

  Run the whole file once, then use Debugger.jl to step through the pullback.
=#

using Revise
using MacroModelling
using LinearAlgebra
using SparseArrays
using ChainRulesCore: rrule, NoTangent

const MM = MacroModelling
const ℒ = LinearAlgebra

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))

model = Smets_Wouters_2007
parameters = copy(model.parameter_values)
opts = MM.merge_calculation_options(verbose = false)

# ==============================================================================
# STEP 1: Build primal inputs
# ==============================================================================
MM.clear_solution_caches!(model, :third_order)

# Warm-up (derivative caches)
_, _, _, _, solved_warmup = MM.get_solution(model, parameters,
                                            algorithm = :third_order, verbose = false)
@assert solved_warmup
MM.clear_solution_caches!(model, :third_order)

# Non-stochastic steady state
SS_and_pars, (solution_error, _) = MM.get_NSSS_and_parameters(model, parameters; opts = opts)
@assert solution_error <= opts.tol.NSSS_acceptance_tol

# Jacobian ∇₁
∇₁ = MM.calculate_jacobian(parameters, SS_and_pars,
                            model.caches, model.functions.jacobian, model.workspaces)

# First-order perturbation solution
𝑺₁, _, solved1 = MM.calculate_first_order_solution(
    ∇₁, model.constants, model.workspaces, model.caches;
    opts = opts, initial_guess = model.caches.qme_solution)
@assert solved1

# Hessian ∇₂ (compressed)
∇₂_input = MM.calculate_hessian(parameters, SS_and_pars,
                                  model.caches, model.functions.hessian, model.workspaces)

# Second-order perturbation solution (compressed)
𝑺₂_input, solved2 = MM.calculate_second_order_solution(
    ∇₁, ∇₂_input, 𝑺₁, model.constants, model.workspaces, model.caches;
    initial_guess = model.caches.second_order_solution, opts = opts)
@assert solved2

# Third-order derivative tensor ∇₃
∇₃ = MM.calculate_third_order_derivatives(
    parameters, SS_and_pars,
    model.caches, model.functions.third_order_derivatives, model.workspaces)

println("Step 1 done – primal inputs ready.")


# ==============================================================================
# STEP 2: rrule forward pass - captures all closure variables
# ==============================================================================

third_out, third_pb = rrule(MM.calculate_third_order_solution,
                            ∇₁, ∇₂_input, ∇₃, 𝑺₁, 𝑺₂_input,
                            model.constants, model.workspaces, model.caches;
                            initial_guess = model.caches.third_order_solution,
                            opts = opts)

𝐒₃_raw, solved3 = third_out
@assert solved3 "Third-order Sylvester solve failed."

println("Step 2 done – S3_raw: ", size(𝐒₃_raw), "  nnz = ", nnz(sparse(𝐒₃_raw)))


# ==============================================================================
# STEP 3: Compute cotangent seed from  f = norm(S3_raw * U3)
# ==============================================================================

M₃ = model.constants.third_order
𝐔₃ = M₃.𝐔₃

𝐒₃_full = 𝐒₃_raw * 𝐔₃
loss = ℒ.norm(𝐒₃_full)
scale = max(loss, eps(eltype(loss)))
∂𝐒₃_raw = (𝐒₃_full / scale) * 𝐔₃'

println("Step 3 done – loss = ", loss)


# ==============================================================================
# STEP 4: INLINE PULLBACK CODE
# ==============================================================================
# This is the exact pullback code from rrules.jl third_order_solution_pullback.
# All variables it needs are captured from the rrule closure above.

# Access closure variables (these are what the rrule captured)
# The closure contains: A, B, C, spinv, ∇₁₊, ∇₂t, ∇₃t, D_ab_t, tmpkron22, ck3_aux_mat,
#                       S2p0_sigma, mm_𝐒₂_kron, M₂, M₃, T, i₊, i₋, n₊, n₋, n, nₑ, nₑ₋,
#                       ℂ, opts, and many transposes

# We need to rebuild some intermediates that were computed in the forward pass
# but not all are captured in the closure. Let's get what we need.

S = eltype(∇₁)
ℂ = model.workspaces.third_order
M₂ = model.constants.second_order
T  = model.constants.post_model_macro

# Expand compressed inputs
∇₂ = ∇₂_input * M₂.𝐔∇₂
𝐒₂ = sparse(𝑺₂_input * M₂.𝐔₂)::SparseMatrixCSC{S, Int}

i₊ = T.future_not_past_and_mixed_idx
i₋ = T.past_not_future_and_mixed_idx
n₋ = T.nPast_not_future_and_mixed
n₊ = T.nFuture_not_past_and_mixed
nₑ = T.nExo
n   = T.nVars
nₑ₋ = n₋ + 1 + nₑ

# Build S1 embedding (same as forward pass)
𝐒₁ = ℂ.𝐒₁::Matrix{S}
copyto!(@view(𝐒₁[:,1:n₋]), @view(𝑺₁[:,1:n₋]))
fill!(@view(𝐒₁[:,n₋+1]), zero(S))
copyto!(@view(𝐒₁[:,n₋+2:end]), @view(𝑺₁[:,n₋+1:end]))

# S1_{-1e} matrix
𝐒₁₋╱𝟏ₑ = ℂ.𝐒₁₋╱𝟏ₑ::Matrix{S}
copyto!(@view(𝐒₁₋╱𝟏ₑ[1:n₋,:]), @view(𝐒₁[i₋,:]))
fill!(@view(𝐒₁₋╱𝟏ₑ[n₋+1:end,:]), zero(S))
@inbounds 𝐒₁₋╱𝟏ₑ[n₋+1,n₋+1] = one(S)
𝐒₁₋╱𝟏ₑ = MM.choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold=1.0, min_length=10, tol=opts.tol.droptol)

# S1 stacking matrix
⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [
    (𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
    𝐒₁
    ℒ.I(nₑ₋)[[range(1,n₋)..., n₋+1 .+ range(1,nₑ)...],:]
]

# S1 on future rows
𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]; zeros(S, n₋+n+nₑ, nₑ₋)]
𝐒₁₊╱𝟎 = MM.choose_matrix_format(𝐒₁₊╱𝟎, density_threshold=1.0, min_length=10, tol=opts.tol.droptol)

# ∇₁₊·S1 + ∇₁₀
∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * M₂.𝐈ₙ₋ - ∇₁[:,range(1,n) .+ n₊]
∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)
spinv = inv(∇₁₊𝐒₁➕∇₁₀lu)
spinv = MM.choose_matrix_format(spinv)

∇₁₊ = @views ∇₁[:,1:n₊] * M₂.𝐈ₙ₊

# A matrix
A = spinv * ∇₁₊

# B matrix
kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)
B = MM.compressed_permuted_mixed_kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔,
                                       sparse_preallocation = ℂ.tmp_sparse_prealloc7)
B += MM.compressed_kron³(𝐒₁₋╱𝟏ₑ, tol = opts.tol.droptol,
                         sparse_preallocation = ℂ.tmp_sparse_prealloc1)

# S2 stacking matrices
⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [
    (𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:]; zeros(S, nₑ+1, nₑ₋^2)])[i₊,:]
    𝐒₂
    zeros(S, n₋+nₑ, nₑ₋^2)
]
⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = MM.choose_matrix_format(
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold=0.0, min_length=10, tol=opts.tol.droptol)

𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:]; zeros(S, n₋+n+nₑ, nₑ₋^2)]
𝐒₂₊╱𝟎 = MM.choose_matrix_format(𝐒₂₊╱𝟎, density_threshold=1.0, min_length=10, tol=opts.tol.droptol)

aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

S1p0_kron_sigma = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔

tmpkron22 = MM.compressed_permuted_mixed_kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                                               S1p0_kron_sigma,
                                               sparse_preallocation = ℂ.tmp_sparse_prealloc6)

∇₁₊ = MM.choose_matrix_format(∇₁₊, density_threshold=1.0, min_length=10, tol=opts.tol.droptol)

S2p0_sigma = 𝐒₂₊╱𝟎 * M₂.𝛔

# Build X3 (C matrix ingredients)
tmpkron2 = ℒ.kron(M₂.𝛔, MM.choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold=0.0, tol=opts.tol.droptol))
D_ab = (tmpkron2 + M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ) * M₃.𝐏𝐂₃

𝐗₃ = MM.mat_mult_kron(∇₂, collect(𝐒₁₊╱𝟎), collect(𝐒₂₊╱𝟎), D_ab,
                       sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc2)

𝐗₃ += MM.mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                        ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, M₃.𝐏𝐂₃,
                        sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc3)

𝐗₃ += MM.mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(S2p0_sigma), M₃.𝐏𝐂₃,
                        sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)

𝐒₁₋╱𝟏ₑ = MM.choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold=0.0, tol=opts.tol.droptol)
mm_𝐒₂_kron = MM.mat_mult_kron(𝐒₂, 𝐒₁₋╱𝟏ₑ, 
    [𝐒₂[i₋,:]; zeros(S, size(𝐒₁,2)-n₋, nₑ₋^2)], sparse = true,
    sparse_preallocation = ℂ.tmp_sparse_prealloc4)
𝐗₃ += ∇₁₊ * mm_𝐒₂_kron * M₃.𝐏𝐂₃

𝐗₃ += ∇₃ * tmpkron22

ck3_aux_mat = MM.compressed_kron³(aux, rowmask = M₃.∇₃_rowmask,
                                  tol = opts.tol.droptol,
                                  sparse_preallocation = ℂ.tmp_sparse_prealloc5)
𝐗₃ += ∇₃ * ck3_aux_mat

C = spinv * 𝐗₃

# Solve Sylvester
𝐒₃, solved = MM.solve_sylvester_equation(A, B, C, ℂ.sylvester_workspace,
                                           initial_guess = zeros(S, 0, 0),
                                           sylvester_algorithm = opts.sylvester_algorithm³,
                                           tol = opts.tol.sylvester_tol,
                                           acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                           verbose = opts.verbose)
@assert solved

𝐒₃_stable = copy(𝐒₃)

# Precompute transposes
𝐏𝐂₃t = M₃.𝐏𝐂₃'
𝛔t   = M₂.𝛔'
𝐔∇₂t = M₂.𝐔∇₂'
𝐔₂t  = M₂.𝐔₂'

M₃𝐏₁ₗt = M₃.𝐏₁ₗ'
M₃𝐏₁ᵣt = M₃.𝐏₁ᵣ'

∇₂t   = MM.choose_matrix_format(∇₂')
∇₃t   = MM.choose_matrix_format(∇₃')
D_ab_t = MM.choose_matrix_format(D_ab')
tmpkron22_t = MM.choose_matrix_format(tmpkron22')
ck3_aux_mat_t = MM.choose_matrix_format(ck3_aux_mat')
𝐒₂t   = MM.choose_matrix_format(𝐒₂', density_threshold=1.0)
⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t = MM.choose_matrix_format(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋')
⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t = MM.choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎')
S2p0_sigma_t = MM.choose_matrix_format(S2p0_sigma')
mm_𝐒₂_kron_t = MM.choose_matrix_format(mm_𝐒₂_kron')

tmpkron22_ck3_aux_mat_t = MM.choose_matrix_format(tmpkron22_t + ck3_aux_mat_t)

# Ensure pullback workspaces
MM.ensure_third_order_pullback_workspaces!(ℂ, S, T, M₂, M₃)

println("Step 4 done – forward pass intermediates rebuilt.")


# ==============================================================================
# STEP 5: INLINE PULLBACK - paste the pullback code here for stepping
# ==============================================================================
# Below is the pullback code. You can use Debugger.jl to step through it:
#   using Debugger
#   @enter third_order_solution_pullback(∂𝐒₃_raw)
#
# Or copy-paste sections to run them individually.

function third_order_solution_pullback(∂𝐒₃)
    #= 
    Pullback for calculate_third_order_solution.
    This is pasted inline so you can step through it in the REPL.
    =#
    
    if ℒ.norm(∂𝐒₃) < opts.tol.sylvester_tol
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    println("  [Pullback] Step 1: adjoint Sylvester")
    # --- adjoint Sylvester:  Aᵀ ∂C_adj Bᵀ + ∂𝐒₃ = ∂C_adj --------------------
    ∂C_adj, slvd = MM.solve_sylvester_equation(A', B', Matrix{Float64}(∂𝐒₃), ℂ.sylvester_workspace,
                                              sylvester_algorithm = opts.sylvester_algorithm³,
                                              tol = opts.tol.sylvester_tol,
                                              acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                              verbose = opts.verbose)
    if !slvd
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    ∂C_adj = MM.choose_matrix_format(∂C_adj)
    println("    ||∂C_adj|| = ", ℒ.norm(Matrix(∂C_adj)))

    # --- Initialize all gradient accumulators ---
    println("  [Pullback] Step 2: initialize accumulators")
    ∂𝐗₃           = ℂ.∂𝐗₃_3rd
    ∂A             = ℂ.∂A_3rd
    ∂B_from_sylv   = ℂ.∂B_sylv_3rd
    ∂out2          = ℂ.∂out2_3rd
    ∇₂t_∂out2     = ℂ.∇₂t_∂out2_3rd
    mul_tmp        = ℂ.mul_tmp_3rd
    ∂∇₁₊𝐒₁➕∇₁₀   = ℂ.∂∇₁₊𝐒₁➕∇₁₀_3rd

    ∂spinv         = ℂ.∂spinv_3rd
    ∂∇₁            = ℂ.∂∇₁_3rd;  fill!(∂∇₁, zero(S))
    ∂𝐒₁₃           = ℂ.∂𝐒₁_3rd;  fill!(∂𝐒₁₃, zero(S))

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
    ∂𝛔_discard2    = ℂ.∂tmpkron0_σ_3rd;   fill!(∂𝛔_discard2, zero(S))

    # --- gradient of A, B, C from 𝐒₃ = A·𝐒₃·B + C ---------------------------
    println("  [Pullback] Step 3: ∂A, ∂B, ∂spinv, ∂X3")
    ℒ.mul!(∂𝐗₃, ∂C_adj, B')
    ℒ.mul!(∂A, ∂𝐗₃, 𝐒₃_stable')
    ℒ.mul!(∂𝐗₃, A', ∂C_adj)
    ℒ.mul!(∂B_from_sylv, 𝐒₃_stable', ∂𝐗₃)
    ∂𝐗₃ = MM.choose_matrix_format(spinv' * ∂C_adj, density_threshold = 1.0, min_length = 0)
    ℒ.mul!(∂spinv, ∂C_adj, 𝐗₃')
    ℒ.mul!(∂spinv, ∂A, ∇₁₊', 1, 1)

    # ∂∇₃
    println("  [Pullback] Step 4: ∂∇₃")
    ∂∇₃ = ∂𝐗₃ * tmpkron22_ck3_aux_mat_t

    # ∂∇₂
    println("  [Pullback] Step 5: ∂∇₂")
    ℒ.mul!(∂out2, ∂𝐗₃, 𝐏𝐂₃t)
    ∂mid_ab = ∂𝐗₃ * D_ab_t
    ∂∇₂ = MM.mat_mult_kron(∂mid_ab, collect(𝐒₁₊╱𝟎'), collect(𝐒₂₊╱𝟎'))
    ∂∇₂ = ∂∇₂ + MM.mat_mult_kron(∂out2, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎t)
    ∂∇₂ = ∂∇₂ + MM.mat_mult_kron(∂out2, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋t, S2p0_sigma_t)
    println("    ||∂∇₂|| = ", ℒ.norm(Matrix(∂∇₂)))

    # ∂𝐒₂
    println("  [Pullback] Step 6: ∂𝐒₂")
    ℒ.mul!(∇₂t_∂out2, ∇₂t, ∂out2)
    ∂tmpkron1 = (∇₂t * ∂mid_ab)
    MM.fill_kron_adjoint!(∂𝐒₂₊╱𝟎, ∂𝐒₁₊╱𝟎_tmp, ∂tmpkron1, 𝐒₂₊╱𝟎, 𝐒₁₊╱𝟎)
    @views ∂𝐒₂[i₊,:] .+= ∂𝐒₂₊╱𝟎[1:length(i₊),:]

    ∂kron_c = (∇₂t_∂out2)
    MM.fill_kron_adjoint!(∂R_c, ∂L_c, ∂kron_c, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
    n₊_len = length(i₊)
    ∂top_block = ∂R_c[1:n₊_len, :]
    @views ∂𝐒₂[i₊,:] .+= ∂top_block * kron𝐒₁₋╱𝟏ₑ'
    ∂𝐒₂_padded = 𝐒₁' * ℒ.I(n)[:,i₊] * ∂top_block
    @views ∂𝐒₂[i₋,:] .+= ∂𝐒₂_padded[1:n₋, :]
    @views ∂𝐒₂ .+= ∂R_c[n₊_len .+ (1:n), :]

    MM.fill_kron_adjoint!(∂R_d, ∂L_d, ∂kron_c, S2p0_sigma, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
    ∂𝐒₂₊╱𝟎_d = ∂R_d * 𝛔t
    @views ∂𝐒₂[i₊,:] .+= ∂𝐒₂₊╱𝟎_d[1:length(i₊),:]

    tmp_t8 = ∇₁₊' * ∂out2
    ∂𝐒₂ = ∂𝐒₂ + MM.mat_mult_kron(tmp_t8, collect(𝐒₁₋╱𝟏ₑ'), collect([𝐒₂[i₋,:]; zeros(S, size(𝐒₁,2)-n₋, nₑ₋^2)]'))

    ∂kron_term8 = ((∇₁₊ * 𝐒₂)' * ∂out2)
    MM.fill_kron_adjoint!(∂𝐒₂₋╱𝟎, ∂𝐒₁₋╱𝟏ₑ_t8, ∂kron_term8, [𝐒₂[i₋,:]; zeros(S, size(𝐒₁,2)-n₋, nₑ₋^2)], 𝐒₁₋╱𝟏ₑ)
    @views ∂𝐒₂[i₋,:] .+= ∂𝐒₂₋╱𝟎[1:n₋,:]
    println("    ||∂𝐒₂|| = ", ℒ.norm(Matrix(∂𝐒₂)))

    # ∂∇₁
    println("  [Pullback] Step 7: ∂∇₁")
    ℒ.mul!(mul_tmp, spinv', ∂spinv)
    ℒ.mul!(∂∇₁₊𝐒₁➕∇₁₀, mul_tmp, spinv')
    ℒ.rmul!(∂∇₁₊𝐒₁➕∇₁₀, -1)

    ∂∇₁[:,1:n₊] -= ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.I(n)[:,i₋] * 𝐒₁[i₊,1:n₋]'
    ∂∇₁[:,range(1,n) .+ n₊] -= ∂∇₁₊𝐒₁➕∇₁₀

    ∂∇₁₊ = ℂ.∂∇₁₊_3rd
    ℒ.mul!(∂∇₁₊, spinv', ∂A)
    ℒ.mul!(∂∇₁₊, ∂out2, mm_𝐒₂_kron_t, 1, 1)
    ∂∇₁[:,1:n₊] += ∂∇₁₊ * ℒ.I(n)[:,i₊]
    println("    ||∂∇₁|| = ", ℒ.norm(Matrix(∂∇₁)))

    # ∂𝑺₁
    println("  [Pullback] Step 8: ∂𝑺₁ (most complex)")
    ℒ.axpy!(1, ∂L_c, ∂S1S1_stack)
    ℒ.axpy!(1, ∂L_d, ∂S1S1_stack)

    ∂tmpkron22 = (∇₃t * ∂𝐗₃)
    ∂S1S1_from_ck = ℂ.∂S1S1_from_ck_3rd; fill!(∂S1S1_from_ck, zero(S))
    ∂S1p0_kron_sigma = ℂ.∂S1p0_kron_sigma_3rd; fill!(∂S1p0_kron_sigma, zero(S))
    MM.compressed_permuted_mixed_kron_pullback!(∂S1S1_from_ck,
                         ∂S1p0_kron_sigma,
                         ∂tmpkron22,
                         ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                         S1p0_kron_sigma;
                         tol = opts.tol.droptol)

    ∂S1p0_kron = (∂S1p0_kron_sigma * 𝛔t)
    ∂S1p0_left = ℂ.∂S1p0_left_3rd; fill!(∂S1p0_left, zero(S))
    ∂S1p0_right = ℂ.∂S1p0_right_3rd; fill!(∂S1p0_right, zero(S))
    MM.fill_kron_adjoint!(∂S1p0_left, ∂S1p0_right, ∂S1p0_kron, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)

    ℒ.axpy!(1, ∂S1S1_from_ck, ∂S1S1_stack)
    ℒ.axpy!(1, ∂S1p0_left, ∂𝐒₁₊╱𝟎₃)
    ℒ.axpy!(1, ∂S1p0_right, ∂𝐒₁₊╱𝟎₃)

    ∂ck3_aux = collect(∇₃t * ∂𝐗₃)
    MM.compressed_kron³_pullback!(∂aux, ∂ck3_aux, aux)
    ℒ.mul!(∂S1S1_stack, M₃.𝐒𝐏', ∂aux, 1, 1)

    ℒ.axpy!(1, ∂𝐒₁₊╱𝟎_tmp, ∂𝐒₁₊╱𝟎₃)

    MM.compressed_permuted_mixed_kron_pullback!(∂𝐒₁₋╱𝟏ₑ₃, ∂𝛔_discard, ∂B_from_sylv, 𝐒₁₋╱𝟏ₑ, M₂.𝛔; tol = opts.tol.droptol)
    MM.compressed_kron³_pullback!(∂𝐒₁₋╱𝟏ₑ₃, ∂B_from_sylv, 𝐒₁₋╱𝟏ₑ)

    tmp_a = collect(MM.mat_mult_kron(collect(∇₂t_∂out2'), collect(𝐒₁₊╱𝟎), collect(𝐒₂₊╱𝟎))')
    ∂tmpkron2 = (tmp_a + M₃𝐏₁ₗt * tmp_a * M₃𝐏₁ᵣt)
    MM.fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ₃, ∂𝛔_discard2, ∂tmpkron2, 𝐒₁₋╱𝟏ₑ, collect(M₂.𝛔))

    ℒ.axpy!(1, ∂𝐒₁₋╱𝟏ₑ_t8, ∂𝐒₁₋╱𝟏ₑ₃)

    ∂kron𝐒₁₋╱𝟏ₑ₃ = (𝐒₂t * ℒ.I(n)[:,i₊] * ∂top_block)
    MM.fill_kron_adjoint!(∂𝐒₁₋╱𝟏ₑ₃, ∂𝐒₁₋╱𝟏ₑ₃, ∂kron𝐒₁₋╱𝟏ₑ₃, 𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)

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
    println("    ||∂𝑺₁|| = ", ℒ.norm(Matrix(∂𝑺₁)))

    # Map back to compressed space
    println("  [Pullback] Step 9: compress gradients")
    ∂∇₂ = ∂∇₂ * 𝐔∇₂t
    ∂𝐒₂ = ∂𝐒₂ * 𝐔₂t
    println("    ||∂∇₂_compressed|| = ", ℒ.norm(Matrix(∂∇₂)))
    println("    ||∂𝐒₂_compressed|| = ", ℒ.norm(Matrix(∂𝐒₂)))

    return (NoTangent(), ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent())
end

println("\nStep 5 done – pullback function defined.")
println("Run: third_grads = third_order_solution_pullback(∂𝐒₃_raw)")
println("Or with Debugger: @enter third_order_solution_pullback(∂𝐒₃_raw)")


# ==============================================================================
# STEP 6: Run the inline pullback
# ==============================================================================
println("\nRunning inline pullback...")
@time third_grads = third_order_solution_pullback(∂𝐒₃_raw)

∂∇₁ = third_grads[2]
∂∇₂ = third_grads[3]
∂∇₃ = third_grads[4]
∂𝑺₁ = third_grads[5]
∂𝐒₂ = third_grads[6]

println("\nPullback complete. Gradient norms:")
println("  ||∂∇₁|| = ", ℒ.norm(Matrix(∂∇₁)))
println("  ||∂∇₂|| = ", ℒ.norm(Matrix(∂∇₂)))
println("  ||∂∇₃|| = ", ℒ.norm(Matrix(∂∇₃)))
println("  ||∂𝑺₁|| = ", ℒ.norm(Matrix(∂𝑺₁)))
println("  ||∂𝐒₂|| = ", ℒ.norm(Matrix(∂𝐒₂)))


# ==============================================================================
# STEP 7: Verify against rrule pullback
# ==============================================================================
println("\nVerifying against rrule pullback...")
rrule_grads = third_pb((∂𝐒₃_raw, NoTangent()))

labels = ("∂∇₁", "∂∇₂", "∂∇₃", "∂𝑺₁", "∂𝐒₂")
for (k, lab) in enumerate(labels)
    manual_k = Matrix(third_grads[k+1])
    rrule_k  = Matrix(rrule_grads[k+1])
    Δ = manual_k - rrule_k
    max_abs = maximum(abs, Δ)
    rel = ℒ.norm(Δ) / max(ℒ.norm(rrule_k), eps())
    println("  $lab:  max|Δ|=$max_abs  rel=$rel")
end
