using Revise
using MacroModelling
using BenchmarkTools
using LinearAlgebra
using SparseArrays

const MM = MacroModelling
const LL = LinearAlgebra

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))

model = Smets_Wouters_2007
parameters = copy(model.parameter_values)
opts = MM.merge_calculation_options(verbose = false)

# -----------------------------------------------------------------------------
# Step 0: Build the exact inputs passed to calculate_third_order_solution
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

∇₂_input = copy(∇₂)
𝐒₂_input = copy(𝐒₂)

# Inputs you asked for (passed to calculate_third_order_solution):
#   ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, model.constants, model.workspaces, model.caches

# -----------------------------------------------------------------------------
# Step 1: Plain code from calculate_third_order_solution (primal)
# -----------------------------------------------------------------------------
S = eltype(∇₁)
if !(eltype(model.workspaces.third_order.Ŝ) == S)
    model.workspaces.third_order = MM.Higher_order_workspace(T = S)
end

ℂ = model.workspaces.third_order
M₂ = model.constants.second_order
M₃ = model.constants.third_order
T = model.constants.post_model_macro

# Expand compressed hessian to full space
∇₂ = ∇₂ * M₂.𝐔∇₂

# Expand compressed second-order solution to full space
𝐒₂ = sparse(𝐒₂ * M₂.𝐔₂)

# Indices and dimensions
i₊ = T.future_not_past_and_mixed_idx
i₋ = T.past_not_future_and_mixed_idx

n₋ = T.nPast_not_future_and_mixed
n₊ = T.nFuture_not_past_and_mixed
nₑ = T.nExo
n = T.nVars
nₑ₋ = n₋ + 1 + nₑ

MM.ensure_higher_order_solution_buffers!(ℂ, n, nₑ₋)

initial_guess = model.caches.third_order_solution
initial_guess_sylv = if length(initial_guess) == 0
    zeros(S, 0, 0)
elseif eltype(initial_guess) <: AbstractFloat
    initial_guess isa Matrix{S} ? initial_guess : Matrix{S}(initial_guess)
else
    zeros(S, 0, 0)
end

# 1st order solution embedding
𝐒₁buf = ℂ.𝐒₁::Matrix{S}
copyto!(@view(𝐒₁buf[:, 1:n₋]), @view(𝐒₁[:, 1:n₋]))
fill!(@view(𝐒₁buf[:, n₋ + 1]), zero(S))
copyto!(@view(𝐒₁buf[:, n₋ + 2:end]), @view(𝐒₁[:, n₋ + 1:end]))

𝐒₁₋╱𝟏ₑ = ℂ.𝐒₁₋╱𝟏ₑ::Matrix{S}
copyto!(@view(𝐒₁₋╱𝟏ₑ[1:n₋, :]), @view(𝐒₁buf[i₋, :]))
fill!(@view(𝐒₁₋╱𝟏ₑ[n₋ + 1:end, :]), zero(S))
@inbounds 𝐒₁₋╱𝟏ₑ[n₋ + 1, n₋ + 1] = one(S)

𝐒₁₋╱𝟏ₑ = MM.choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [
    (𝐒₁buf * 𝐒₁₋╱𝟏ₑ)[i₊, :]
    𝐒₁buf
    LL.I(nₑ₋)[[range(1, n₋)..., n₋ + 1 .+ range(1, nₑ)...], :]
]

𝐒₁₊╱𝟎 = @views [
    𝐒₁buf[i₊, :]
    zeros(S, n₋ + n + nₑ, nₑ₋)
]
𝐒₁₊╱𝟎 = MM.choose_matrix_format(𝐒₁₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:, 1:n₊] * 𝐒₁buf[i₊, 1:n₋] * LL.I(n)[i₋, :] - ∇₁[:, range(1, n) .+ n₊]

∇₁₊𝐒₁➕∇₁₀lu = LL.lu(∇₁₊𝐒₁➕∇₁₀, check = false)
if !LL.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
    error("Third-order setup failed: LU factorization of ∇₁₊𝐒₁➕∇₁₀ was unsuccessful.")
end

∇₁₊ = @views ∇₁[:, 1:n₊] * M₂.𝐈ₙ₊
A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊

B = MM.compressed_permuted_mixed_kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔, sparse_preallocation = ℂ.tmp_sparse_prealloc7)
B += MM.compressed_kron³(𝐒₁₋╱𝟏ₑ, tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc1)

⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [
    (𝐒₂ * LL.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁buf * [𝐒₂[i₋, :] ; zeros(S, nₑ + 1, nₑ₋^2)])[i₊, :]
    𝐒₂
    zeros(S, n₋ + nₑ, nₑ₋^2)
]
⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = MM.choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎,
                                                density_threshold = 0.0,
                                                min_length = 10,
                                                tol = opts.tol.droptol)

𝐒₂₊╱𝟎 = @views [
    𝐒₂[i₊, :]
    zeros(S, n₋ + n + nₑ, nₑ₋^2)
]

aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

𝐒₂₊╱𝟎 = MM.choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)
∇₁₊ = MM.choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

𝐒₂₋╱𝟎 = [𝐒₂[i₋, :] ; zeros(S, size(𝐒₁buf, 2) - n₋, nₑ₋^2)]

# Terms (a)+(b)
tmpkron2_sp = LL.kron(M₂.𝛔, MM.choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.droptol))
D_ab = (tmpkron2_sp + M₃.𝐏₁ₗ * tmpkron2_sp * M₃.𝐏₁ᵣ) * M₃.𝐏𝐂₃

𝐗₃ = MM.mat_mult_kron(∇₂, collect(𝐒₁₊╱𝟎), collect(𝐒₂₊╱𝟎), D_ab,
                      sparse = true,
                      sparse_preallocation = ℂ.tmp_sparse_prealloc2)

# Term (c)
𝐗₃ += MM.mat_mult_kron(∇₂,
                       ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                       ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎,
                       M₃.𝐏𝐂₃,
                       sparse = true,
                       sparse_preallocation = ℂ.tmp_sparse_prealloc3)

# Term (d)
𝐗₃ += MM.mat_mult_kron(∇₂,
                       ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                       collect(𝐒₂₊╱𝟎 * M₂.𝛔),
                       M₃.𝐏𝐂₃,
                       sparse = true,
                       sparse_preallocation = ℂ.tmp_sparse_prealloc4)

# Term (e)
𝐒₁₋╱𝟏ₑ = MM.choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.droptol)
𝐗₃ += MM.mat_mult_kron(∇₁₊ * 𝐒₂,
                       𝐒₁₋╱𝟏ₑ,
                       𝐒₂₋╱𝟎,
                       M₃.𝐏𝐂₃,
                       sparse = true)

# Mixed ∇₃ term
if length(ℂ.tmpkron0) > 0 && eltype(ℂ.tmpkron0) == S
    LL.kron!(ℂ.tmpkron0, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
else
    ℂ.tmpkron0 = LL.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
end

ℂ.tmpkron0 *= M₂.𝛔

tmpkron22 = MM.compressed_permuted_mixed_kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,
                                              ℂ.tmpkron0,
                                              sparse_preallocation = ℂ.tmp_sparse_prealloc6)
𝐗₃ += ∇₃ * tmpkron22

# Cubic ∇₃ term
𝐗₃ += ∇₃ * MM.compressed_kron³( aux,
                                rowmask = M₃.∇₃_rowmask,
                                tol = opts.tol.droptol,
                                sparse_preallocation = ℂ.tmp_sparse_prealloc5)

C = ∇₁₊𝐒₁➕∇₁₀lu \ 𝐗₃

𝐒₃, solved3 = MM.solve_sylvester_equation(A,
                                           B,
                                           C,
                                           ℂ.sylvester_workspace,
                                           initial_guess = initial_guess_sylv,
                                           sylvester_algorithm = opts.sylvester_algorithm³,
                                           tol = opts.tol.sylvester_tol,
                                           acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                           verbose = opts.verbose)

𝐒₃ = MM.choose_matrix_format(𝐒₃, multithreaded = false, tol = opts.tol.droptol)

if solved3
    if 𝐒₃ isa Matrix{S} && model.caches.third_order_solution isa Matrix{S} && size(model.caches.third_order_solution) == size(𝐒₃)
        copyto!(model.caches.third_order_solution, 𝐒₃)
    elseif 𝐒₃ isa SparseMatrixCSC{S, Int} && model.caches.third_order_solution isa SparseMatrixCSC{S, Int} &&
           size(model.caches.third_order_solution) == size(𝐒₃) &&
           model.caches.third_order_solution.colptr == 𝐒₃.colptr &&
           model.caches.third_order_solution.rowval == 𝐒₃.rowval
        copyto!(model.caches.third_order_solution.nzval, 𝐒₃.nzval)
    else
        model.caches.third_order_solution = copy(𝐒₃)
    end
end

println("third_order_solved=", solved3, " size(𝐒₃)=", size(𝐒₃), " nnz(𝐒₃)=", nnz(sparse(𝐒₃)))

# -----------------------------------------------------------------------------
# Step 2: Check against calculate_third_order_solution output
# -----------------------------------------------------------------------------
𝐒₃_ref, solved3_ref = MM.calculate_third_order_solution(∇₁,
                            ∇₂_input,
                            ∇₃,
                            𝐒₁,
                            𝐒₂_input,
                            model.constants,
                            model.workspaces,
                            model.caches;
                            initial_guess = zeros(eltype(∇₁), 0, 0),
                            opts = opts)

Δ = Matrix(𝐒₃) - Matrix(𝐒₃_ref)
max_abs_diff = maximum(abs, Δ)
rel_diff = norm(Δ) / max(norm(Matrix(𝐒₃_ref)), eps())

println("third_order_ref_solved=", solved3_ref,
    " max_abs_diff=", max_abs_diff,
    " rel_diff=", rel_diff)
