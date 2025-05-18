using Revise
# ENV["JULIA_CONDAPKG_BACKEND"] = "MicroMamba"
using MacroModelling
using BenchmarkTools
import MacroModelling: clear_solution_caches!, get_NSSS_and_parameters, get_symbols, replace_symbols, match_pattern, take_nth_order_derivatives, Tolerances, block_solver, levenberg_marquardt, solve_ss, transform, choose_matrix_format
using SparseArrays
import LinearSolve as 𝒮
using Random, Zygote, FiniteDifferences

include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")
get_solution(m)
model = m
observables = [:k,:c]
get_variables(m)

include("models/SW07_nonlinear.jl")
model = SW07_nonlinear
observables = [:k,:c,:y]


Random.seed!(1)
simulated_data = simulate(model)

get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

fin_grad = FiniteDifferences.grad(FiniteDifferences.forward_fdm(3,1),#, max_range = 1e-2),
x-> begin println(x); get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = false) end, model.parameter_values)


params = [0.026024285942547642, 0.18, 1.5, 10.0, 0.95827, 0.22137, 0.97391, 0.70524, 0.11421, 0.83954, 0.9745, 0.69414, 0.93617, 5.5811, 1.4103, 0.68049, 0.80501, 2.2061, 0.56351, 0.24165, 0.49552, 1.3443, 1.931, 0.82512, 0.097844, 0.25114, 0.8731, 0.12575, 0.4419, 0.53817, 0.18003, 64.5595, 0.667]
get_loglikelihood(model, simulated_data(observables, :, :simulate), params, verbose = true)


back_grad[1] ≈ fin_grad[1]

𝓂 = SW07_nonlinear

import Symbolics
import MacroTools
using SparseArrays
import RuntimeGeneratedFunctions
import LinearAlgebra as ℒ
import DataStructures: CircularBuffer

RuntimeGeneratedFunctions.init(@__MODULE__)
cse = true
skipzeros = true
density_threshold::Float64 = .1
min_length::Int = 10000


include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")
𝓂 = m

solver_parameters = 𝓂.solver_parameters
cold_start = true
verbose = true
tol = Tolerances()
initial_parameters = 𝓂.parameter_values

clear_solution_caches!(𝓂, :first_order)

# (initial_parameters, 𝓂, tol, verbose, cold_start, solver_parameters)->begin
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3868 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3875 =#
initial_parameters = if typeof(initial_parameters) == Vector{Float64}
    initial_parameters
else
    ℱ.value.(initial_parameters)
end
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3877 =#
initial_parameters_tmp = copy(initial_parameters)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3879 =#
parameters = copy(initial_parameters)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3880 =#
params_flt = copy(initial_parameters)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3882 =#
current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - initial_parameters)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3883 =#
closest_solution_init = 𝓂.NSSS_solver_cache[end]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3885 =#
for pars = 𝓂.NSSS_solver_cache
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3886 =#
copy!(initial_parameters_tmp, pars[end])
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3888 =#
ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3890 =#
latest = sum(abs2, initial_parameters_tmp)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3891 =#
if latest <= current_best
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3892 =#
    current_best = latest
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3893 =#
    closest_solution_init = pars
end
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3895 =#
end
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3900 =#
range_iters = 0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3901 =#
solution_error = 1.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3902 =#
solved_scale = 0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3904 =#
scale = 1.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3906 =#
NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3907 =#
push!(NSSS_solver_cache_scale, closest_solution_init)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3909 =#
# while range_iters <= if cold_start
#             1
#         else
#             500
#         end && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3910 =#
range_iters += 1
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3911 =#
fail_fast_solvers_only = if range_iters > 1
        true
    else
        false
    end
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3913 =#
if abs(solved_scale - scale) < 0.0001
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3915 =#
    # break
end
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3928 =#
current_best = sum(abs2, (NSSS_solver_cache_scale[end])[end] - initial_parameters)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3929 =#
closest_solution = NSSS_solver_cache_scale[end]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3931 =#
for pars = NSSS_solver_cache_scale
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3932 =#
    copy!(initial_parameters_tmp, pars[end])
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3934 =#
    ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3936 =#
    latest = sum(abs2, initial_parameters_tmp)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3938 =#
    if latest <= current_best
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3939 =#
        current_best = latest
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3940 =#
        closest_solution = pars
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3942 =#
end
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3946 =#
if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3947 =#
    parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
else
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3949 =#
    parameters = copy(initial_parameters)
end
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3951 =#
params_flt = parameters
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3955 =#
cap_share = parameters[1]
R_ss = parameters[2]
I_K_ratio = parameters[3]
phi_pi = parameters[4]
Pi_real = parameters[7]
rhoz = parameters[8]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3956 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3957 =#
Pi_ss = R_ss - Pi_real
rho_z_delta = rhoz
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3958 =#
NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3959 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3960 =#
iters = 0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3961 =#
➕₃ = min(600, max(-1.0e12, R_ss - 1))
solution_error += +(abs(➕₃ - (R_ss - 1)))
if solution_error > tol.NSSS_acceptance_tol
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
    if verbose
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
        println("Failed for analytical variables with error $(solution_error)")
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
    scale = scale * 0.3 + solved_scale * 0.7
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
    # continue
end
R = exp(➕₃)
Pi = Pi_ss
beta = Pi / R
solution_error += abs(min(max(1.1920928955078125e-7, beta), 0.9999998807907104) - beta)
if solution_error > tol.NSSS_acceptance_tol
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
    if verbose
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
        println("Failed for bounded variables with error $(solution_error)")
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
    scale = scale * 0.3 + solved_scale * 0.7
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
    # continue
end
➕₁ = min(max(2.220446049250313e-16, (R * beta) ^ (1 / phi_pi)), 1.0e12)
solution_error += abs(➕₁ - (R * beta) ^ (1 / phi_pi))
if solution_error > tol.NSSS_acceptance_tol
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
    if verbose
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
        println("Failed for analytical aux variables with error $(solution_error)")
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
    scale = scale * 0.3 + solved_scale * 0.7
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
    # continue
end
Pibar = Pi / ➕₁
z_delta = 1
A = 1
params_and_solved_vars = [cap_share, I_K_ratio, beta]
lbs = [1.1920928955078125e-7, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, 1.1920928955078125e-7]
ubs = [0.9999998807907104, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 0.9999998807907104]
inits = [max.(lbs[1:length(closest_solution[1])], min.(ubs[1:length(closest_solution[1])], closest_solution[1])), closest_solution[2]]



parameters_and_solved_vars = params_and_solved_vars
n_block =   1
SS_solve_block  =   𝓂.ss_solve_blocks_in_place[1]
guess_and_pars_solved_vars  =   inits
lbs =   lbs
ubs =   ubs
parameters  =   solver_parameters
fail_fast_solvers_only  =   fail_fast_solvers_only
cold_start  =   cold_start
verbose =   verbose

# function block_solver(parameters_and_solved_vars::Vector{T}, 
#                         n_block::Int, 
#                         SS_solve_block::ss_solve_block,
#                         guess_and_pars_solved_vars::Vector{Vector{T}}, 
#                         lbs::Vector{T}, 
#                         ubs::Vector{T},
#                         parameters::Vector{solver_parameters},
#                         fail_fast_solvers_only::Bool,
#                         cold_start::Bool,
#                         verbose::Bool ;
#                         tol::Tolerances = Tolerances(),
#                         # rtol::AbstractFloat = sqrt(eps()),
#                         # timeout = 120,
#                         # starting_points::Vector{Float64} = [1.205996189998029, 0.7688, 0.897, 1.2],#, 0.9, 0.75, 1.5, -0.5, 2.0, .25]
#                         # verbose::Bool = false
#                         )::Tuple{Vector{T},Tuple{T, Int}} where T <: AbstractFloat

    # tol = parameters[1].ftol
    # rtol = parameters[1].rel_xtol

    solved_yet = false

    guess = guess_and_pars_solved_vars[1]

    sol_values = guess

    closest_parameters_and_solved_vars = sum(abs, guess_and_pars_solved_vars[2]) == Inf ? parameters_and_solved_vars : guess_and_pars_solved_vars[2]

    # res = ss_solve_blocks(parameters_and_solved_vars, guess)

    SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.func_buffer, guess, parameters_and_solved_vars) # TODO: make the block a struct
    # TODO: do the function creation with Symbolics as this will solve the compilation bottleneck for large functions

    res = SS_solve_block.ss_problem.func_buffer

    sol_minimum  = ℒ.norm(res)

    # if !cold_start
    #     if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol
    #         # ∇ = 𝒟.jacobian(x->(ss_solve_blocks(parameters_and_solved_vars, x)), backend, guess)

    #         # ∇̂ = ℒ.lu!(∇, check = false)

    #         SS_solve_block.ss_problem.jac(SS_solve_block.ss_problem.jac_buffer, guess, parameters_and_solved_vars)

    #         ∇ = SS_solve_block.ss_problem.jac_buffer

    #         ∇̂ = ℒ.lu(∇, check = false)
            
    #         if ℒ.issuccess(∇̂)
    #             guess_update = ∇̂ \ res

    #             new_guess = guess - guess_update

    #             rel_sol_minimum = ℒ.norm(guess_update) / max(ℒ.norm(new_guess), sol_minimum)
    #         else
    #             rel_sol_minimum = 1.0
    #         end
    #     else
    #         rel_sol_minimum = 0.0
    #     end
    # else
        rel_sol_minimum = 1.0
    # end
    
    if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
        solved_yet = true

        if verbose
            println("Block: $n_block, - Solved using previous solution; residual norm: $sol_minimum")
        end
    end

    total_iters = [0,0]

    SS_optimizer = levenberg_marquardt

    # if cold_start
        guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only



SS_optimizer    =   SS_optimizer
SS_solve_block    =   SS_solve_block
parameters_and_solved_vars    =   parameters_and_solved_vars
closest_parameters_and_solved_vars    =   closest_parameters_and_solved_vars
lbs   =   lbs
ubs   =   ubs
tol   =   tol
total_iters   =   total_iters
n_block   =   n_block
verbose   =   verbose
guess =   guesses[1]
solver_params =   parameters[4]
extended_problem  =   false
separate_starting_value   =   false


# function solve_ss(SS_optimizer::Function,
#                     SS_solve_block::ss_solve_block,
#                     parameters_and_solved_vars::Vector{T},
#                     closest_parameters_and_solved_vars::Vector{T},
#                     lbs::Vector{T},
#                     ubs::Vector{T},
#                     tol::Tolerances,
#                     total_iters::Vector{Int},
#                     n_block::Int,
#                     verbose::Bool,
#                     guess::Vector{T},
#                     solver_params::solver_parameters,
#                     extended_problem::Bool,
#                     separate_starting_value::Union{Bool,T})::Tuple{Vector{T}, Vector{Int}, T, T} where T <: AbstractFloat
    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    if separate_starting_value isa Float64
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], fill(separate_starting_value, length(guess))))
        sol_values_init[ubs[1:length(guess)] .<= 1] .= .1 # capture cases where part of values is small
    else
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], [g < 1e12 ? g : solver_params.starting_value for g in guess]))
    end





fnj =   extended_problem ? SS_solve_block.extended_ss_problem : SS_solve_block.ss_problem
initial_guess   =   extended_problem ? vcat(sol_values_init, closest_parameters_and_solved_vars) : sol_values_init
parameters_and_solved_vars  =   parameters_and_solved_vars
lower_bounds    =   extended_problem ? lbs : lbs[1:length(guess)]
upper_bounds    =   extended_problem ? ubs : ubs[1:length(guess)]
parameters  =   solver_params

# function levenberg_marquardt(
#     fnj::function_and_jacobian,
#     initial_guess::Array{T,1}, 
#     parameters_and_solved_vars::Array{T,1},
#     lower_bounds::Array{T,1}, 
#     upper_bounds::Array{T,1},
#     parameters::solver_parameters;
#     tol::Tolerances = Tolerances()
#     )::Tuple{Vector{T}, Tuple{Int, Int, T, T}} where {T <: AbstractFloat}
    # issues with optimization: https://www.gurobi.com/documentation/8.1/refman/numerics_gurobi_guidelines.html

    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    iterations = 250
    
    ϕ̄ = parameters.ϕ̄
    ϕ̂ = parameters.ϕ̂
    μ̄¹ = parameters.μ̄¹
    μ̄² = parameters.μ̄²
    p̄¹ = parameters.p̄¹
    p̄² = parameters.p̄²
    ρ = parameters.ρ
    ρ¹ = parameters.ρ¹
    ρ² = parameters.ρ²
    ρ³ = parameters.ρ³
    ν = parameters.ν
    λ¹ = parameters.λ¹
    λ² = parameters.λ²
    λ̂¹ = parameters.λ̂¹
    λ̂² = parameters.λ̂²
    λ̅¹ = parameters.λ̅¹
    λ̅² = parameters.λ̅²
    λ̂̅¹ = parameters.λ̂̅¹
    λ̂̅² = parameters.λ̂̅²
    transformation_level = parameters.transformation_level
    shift = parameters.shift
    backtracking_order = parameters.backtracking_order

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert all(lower_bounds .< upper_bounds)
    @assert backtracking_order ∈ [2,3] "Backtracking order can only be quadratic (2) or cubic (3)."

    max_linesearch_iterations = 600

    # function f̂(x) 
    #     f(undo_transform(x,transformation_level))  
    # #     # f(undo_transform(x,transformation_level,shift))  
    # end

    upper_bounds  = transform(upper_bounds,transformation_level)
    # upper_bounds  = transform(upper_bounds,transformation_level,shift)
    lower_bounds  = transform(lower_bounds,transformation_level)
    # lower_bounds  = transform(lower_bounds,transformation_level,shift)

    current_guess = copy(transform(initial_guess,transformation_level))
    current_guess_untransformed = copy(transform(initial_guess,transformation_level))
    # current_guess = copy(transform(initial_guess,transformation_level,shift))
    previous_guess = similar(current_guess)
    previous_guess_untransformed = similar(current_guess)
    guess_update = similar(current_guess)
    factor = similar(current_guess)
    # ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))
    ∇ = fnj.jac_buffer
    # ∇̂ = similar(fnj.jac_buffer)
    ∇̄ = similar(fnj.jac_buffer)

    ∇̂ = choose_matrix_format(∇' * ∇, multithreaded = false)
    
    if ∇̂ isa SparseMatrixCSC
        prob = 𝒮.LinearProblem(∇̂, guess_update, 𝒮.UMFPACKFactorization())
    else
        prob = 𝒮.LinearProblem(∇̂, guess_update)#, 𝒮.CholeskyFactorization)
    end

    sol_cache = 𝒮.init(prob)
    
    # prep = 𝒟.prepare_jacobian(f̂, backend, current_guess)

    largest_step = (1.0)
    largest_residual = (1.0)
    largest_relative_step = (1.0)

    μ¹ = μ̄¹
    μ² = μ̄²

    p¹ = p̄¹
    p² = p̄²

    grad_iter = 0
    func_iter = 0

    # for iter in 1:iterations
        # make the jacobian and f calls nonallocating
        copy!(current_guess_untransformed, current_guess)
        
        if transformation_level > 0
            factor .= 1
            for _ in 1:transformation_level
                factor .*= cosh.(current_guess_untransformed)
                current_guess_untransformed .= sinh.(current_guess_untransformed)
            end
        end

        fnj.jac(∇, current_guess_untransformed, parameters_and_solved_vars)
        # 𝒟.jacobian!(f̂, ∇, prep, backend, current_guess)

        if transformation_level > 0
            if ∇ isa SparseMatrixCSC
                # ∇̄ = ∇ .* factor'
                copy!(∇̄.nzval, ∇.nzval)
                @inbounds for j in 1:size(∇, 2)
                    col_start = ∇̄.colptr[j]
                    col_end = ∇̄.colptr[j+1] - 1
                    for k in col_start:col_end
                        ∇̄.nzval[k] *= factor[j]
                    end
                end
            else
                # ℒ.mul!(∇̄, ∇, factor')
                @. ∇̄ = ∇ * factor'
                # ∇ .*= factor'
            end
        end

        grad_iter += 1

        previous_guess .= current_guess

        # ∇̂ .= ∇' * ∇
        if ∇̄ isa SparseMatrixCSC && ∇̂ isa SparseMatrixCSC
            ∇̂ = ∇̄' * ∇̄
        else
            ℒ.mul!(∇̂, ∇̄', ∇̄)
        end
# 5×5 Matrix{Float64}:
#   323.367  -66.847    -373.332  -219.421   -66.847
#   -66.847   29.9399    152.684   123.78     -1.03584
#  -373.332  152.684     835.387   653.857     0.0
#  -219.421  123.78      653.857   559.047   -28.9688
#   -66.847   -1.03584     0.0     -28.9688   30.0046
        fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)

        μ¹s = μ¹ * sum(abs2, fnj.func_buffer)^p¹
        # μ¹s = μ¹ * sum(abs2, f̂(current_guess))^p¹
        func_iter += 1

        for i in 1:size(∇̂,1)
            ∇̂[i,i] += μ¹s
            ∇̂[i,i] += μ² * ∇̂[i,i]^p²
        end
        # ∇̂ .+= μ¹ * sum(abs2, f̂(current_guess))^p¹ * ℒ.I + μ² * ℒ.Diagonal(∇̂).^p²

        if !all(isfinite, ∇̂)
            largest_relative_step = 1.0
            largest_residual = 1.0
            # break
        end

        fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)

        ℒ.mul!(guess_update, ∇̄', fnj.func_buffer)

        sol_cache.A = ∇̂
        sol_cache.b = guess_update
        𝒮.solve!(sol_cache)
        copy!(guess_update, sol_cache.u)

        if !isfinite(sum(guess_update))
            largest_relative_step = 1.0
            largest_residual = 1.0
            # break
        end

        ℒ.axpy!(-1, guess_update, current_guess)
        # current_guess .-= ∇̄ \ ∇' * f̂(current_guess)

        minmax!(current_guess, lower_bounds, upper_bounds)

        copy!(previous_guess_untransformed, previous_guess)

        for _ in 1:transformation_level
            previous_guess_untransformed .= sinh.(previous_guess_untransformed)
        end
        
        fnj.func(fnj.func_buffer, previous_guess_untransformed, parameters_and_solved_vars)

        P = sum(abs2, fnj.func_buffer)
        # P = sum(abs2, f̂(previous_guess))
        P̃ = P
        
        fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)

        P̋ = sum(abs2, fnj.func_buffer)
        # P̋ = sum(abs2, f̂(current_guess))

        func_iter += 3

        α = 1.0
        ᾱ = 1.0

        ν̂ = ν

        guess_update .= current_guess - previous_guess

        fnj.func(fnj.func_buffer, previous_guess_untransformed, parameters_and_solved_vars)

        g = fnj.func_buffer' * ∇̄ * guess_update
        # g = f̂(previous_guess)' * ∇ * guess_update
        U = sum(abs2,guess_update)
        func_iter += 1




    sol_new_tmp, info = SS_optimizer(   extended_problem ? SS_solve_block.extended_ss_problem : SS_solve_block.ss_problem,
                                        extended_problem ? vcat(sol_values_init, closest_parameters_and_solved_vars) : sol_values_init,
                                        parameters_and_solved_vars,
                                        extended_problem ? lbs : lbs[1:length(guess)],
                                        extended_problem ? ubs : ubs[1:length(guess)],
                                        solver_params,
                                        tol = tol   )





        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                            guesses[1], 
                                                            parameters[1],
                                                            true,
                                                            false)

                                                            
        for g in guesses
            for p in parameters
                for ext in [true, false] # try first the system where values and parameters can vary, next try the system where only values can vary
                    if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol# || rel_sol_minimum > rtol
                        if solved_yet continue end

                        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                        # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                            g, 
                                                            p,
                                                            ext,
                                                            false)
                        if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                            println(i)
                            solved_yet = true
                        end
                        i+=1
                    end
                end
            end
        end


solution = block_solver(params_and_solved_vars, 1, 𝓂.ss_solve_blocks_in_place[1], inits, lbs, ubs, solver_parameters, fail_fast_solvers_only, cold_start, verbose)
iters += (solution[2])[2]
solution_error += (solution[2])[1]
if solution_error > tol.NSSS_acceptance_tol
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2986 =#
    if verbose
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2986 =#
        println("Failed after solving block with error $(solution_error)")
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2986 =#
    scale = scale * 0.3 + solved_scale * 0.7
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2986 =#
    # continue
end
sol = solution[1]
alpha = sol[1]
c = sol[2]
delta = sol[3]
k = sol[4]
y = sol[5]







SS(𝓂)

SSS(𝓂)
SSS(𝓂, algorithm = :pruned_third_order)
get_solution(𝓂)
get_std(𝓂)
get_std(𝓂, algorithm = :pruned_second_order)
get_std(𝓂, algorithm = :pruned_third_order)

𝓂.third_order_derivatives[1]

SS_and_pars = 𝓂.solution.non_stochastic_steady_state

𝓂.jacobian[2](𝓂.jacobian[1],𝓂.parameter_values,SS_and_pars)
𝓂.jacobian[1][157]
𝓂 = Caldara_et_al_2012
density_threshold::Float64 = .1 
min_length::Int = 1000
import NaNMath
(/)((*)((*)(-1, (NaNMath.pow)((/)((getindex)(𝓂.parameter_values, 4), (getindex)(SS_and_pars, 10)), (+)(-1, (getindex)(SS_and_pars, 4)))), (getindex)(SS_and_pars, 4)), (getindex)(SS_and_pars, 10))


future_varss  = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₁₎$")))
present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎$")))
past_varss    = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₋₁₎$")))
shock_varss   = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₓ₎$")))
ss_varss      = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₛₛ₎$")))

sort!(future_varss  ,by = x->replace(string(x),r"₍₁₎$"=>"")) #sort by name without time index because otherwise eps_zᴸ⁽⁻¹⁾₍₋₁₎ comes before eps_z₍₋₁₎
sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))
sort!(past_varss    ,by = x->replace(string(x),r"₍₋₁₎$"=>""))
sort!(shock_varss   ,by = x->replace(string(x),r"₍ₓ₎$"=>""))
sort!(ss_varss      ,by = x->replace(string(x),r"₍ₛₛ₎$"=>""))

dyn_future_list = collect(reduce(union, 𝓂.dyn_future_list))
dyn_present_list = collect(reduce(union, 𝓂.dyn_present_list))
dyn_past_list = collect(reduce(union, 𝓂.dyn_past_list))
dyn_exo_list = collect(reduce(union,𝓂.dyn_exo_list))
dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.dyn_ss_list))) .* "₍ₛₛ₎")

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))

vars_raw = vcat(dyn_future_list[indexin(sort(future),future)],
                dyn_present_list[indexin(sort(present),present)],
                dyn_past_list[indexin(sort(past),past)],
                dyn_exo_list[indexin(sort(exo),exo)])

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

dyn_var_idxs = vcat(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx)
                
pars_and_SS = Expr[]
for (i, p) in enumerate(vcat(𝓂.parameters, 𝓂.calibration_equations_parameters))
    push!(pars_and_SS, :($p = parameters_and_SS[$i]))
end

nn = length(pars_and_SS)

for (i, p) in enumerate(dyn_ss_list[indexin(sort(stst),stst)])
    push!(pars_and_SS, :($p = parameters_and_SS[$(i + nn)]))
end

deriv_vars = Expr[]
for (i, u) in enumerate(vars_raw)
    push!(deriv_vars, :($u = variables[$i]))
end

eeqqss = Expr[]
for (i, u) in enumerate(𝓂.dyn_equations)
    push!(eeqqss, :(ℰ[$i] = $u))
end

funcs = :(function calculate_residual_of_dynamic_equations!(ℰ, variables, parameters_and_SS)
    $(pars_and_SS...)
    $(𝓂.calibration_equations_no_var...)
    $(deriv_vars...)
    @inbounds begin
    $(eeqqss...)
    end
    return nothing
end)


pars_ext = vcat(𝓂.parameters, 𝓂.calibration_equations_parameters)
parameters_and_SS = vcat(pars_ext, dyn_ss_list[indexin(sort(stst),stst)])

np = length(parameters_and_SS)
nv = length(vars_raw)
nc = length(𝓂.calibration_equations)
nps = length(𝓂.parameters)
nxs = maximum(dyn_var_idxs) + nc

Symbolics.@variables 𝔓[1:np] 𝔙[1:nv]

parameter_dict = Dict{Symbol, Symbol}()
back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
calib_vars = Symbol[]
calib_expr = []
SS_mapping = Dict{Symbolics.Num, Symbolics.Num}()


for (i,v) in enumerate(parameters_and_SS)
    push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
    push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
    if i > nps
        if i > length(pars_ext)
            push!(SS_mapping, 𝔓[i] => 𝔙[dyn_ss_idx[i-length(pars_ext)]])
        else
            push!(SS_mapping, 𝔓[i] => 𝔙[nxs + i - nps])
        end
    end
end

for (i,v) in enumerate(vars_raw)
    push!(parameter_dict, v => :($(Symbol("𝔙_$i"))))
    push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔙_$i"))), @__MODULE__) => 𝔙[i])
    if i <= length(dyn_var_idxs)
        push!(SS_mapping, 𝔙[i] => 𝔙[dyn_var_idxs[i]])
    else
        push!(SS_mapping, 𝔙[i] => 0)
    end
end


for v in 𝓂.calibration_equations_no_var
    push!(calib_vars, v.args[1])
    push!(calib_expr, v.args[2])
end


calib_replacements = Dict{Symbol,Any}()
for (i,x) in enumerate(calib_vars)
    replacement = Dict(x => calib_expr[i])
    for ii in i+1:length(calib_vars)
        calib_expr[ii] = replace_symbols(calib_expr[ii], replacement)
    end
    push!(calib_replacements, x => calib_expr[i])
end


dyn_equations = 𝓂.dyn_equations |> 
    x -> replace_symbols.(x, Ref(calib_replacements)) |> 
    x -> replace_symbols.(x, Ref(parameter_dict)) |> 
    x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
    x -> Symbolics.substitute.(x, Ref(back_to_array_dict))

# StatsFuns
function norminvcdf(p::T)::T where T
    -erfcinv(2*p) * 1.4142135623730951
end
norminv(p) = norminvcdf(p)
qnorm(p)= norminvcdf(p)

function normlogpdf(z::T)::T where T
    -(abs2(z) + 1.8378770664093453) / 2
end
function normpdf(z::T)::T where T
    exp(-abs2(z)/2) * 0.3989422804014327
end

function normcdf(z::T)::T where T
    erfc(-z * 0.7071067811865475) / 2
end
pnorm(p) = normcdf(p)
dnorm(p) = normpdf(p)

Symbolics.@register_symbolic norminvcdf(p)
Symbolics.@register_symbolic norminv(p)
Symbolics.@register_symbolic qnorm(p)
Symbolics.@register_symbolic normlogpdf(z)
Symbolics.@register_symbolic normpdf(z)
Symbolics.@register_symbolic normcdf(z)
Symbolics.@register_symbolic pnorm(p)
Symbolics.@register_symbolic dnorm(p)



dyn_equations = 𝓂.dyn_equations |> 
    x -> replace_symbols.(x, Ref(calib_replacements)) |> 
    x -> replace_symbols.(x, Ref(parameter_dict)) |> 
    x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
    x -> Symbolics.substitute.(x, Ref(back_to_array_dict)) #|>
    # x -> Symbolics.substitute.(x, Ref(SS_mapping))
# SS_mapping
# results = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nSS)


    # Compute the 1st order derivative with respect to X (Jacobian)
    spX_order_1 = Symbolics.sparsejacobian(dyn_equations, 𝔙) # nϵ x nx


    spX_order_1_sub = copy(spX_order_1)

    # spX_order_1_sub.nzval .= Symbolics.fast_substitute(spX_order_1_sub.nzval, Dict(Symbolics.scalarize(𝔛𝔛) .=> 𝔙))
    spX_order_1_sub.nzval .= Symbolics.substitute(spX_order_1_sub.nzval, SS_mapping)

    ∇₁_dyn = spX_order_1_sub

    lennz = nnz(∇₁_dyn)

    # if (lennz / length(∇₁_dyn) > density_threshold) || (length(∇₁_dyn) < min_length)
    #     derivatives_mat = convert(Matrix, ∇₁_dyn)
    #     buffer = zeros(Float64, size(∇₁_dyn))
    # else
        derivatives_mat = ∇₁_dyn
        buffer = similar(∇₁_dyn, Float64)
        buffer.nzval .= 0
    # end
    
    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end
    
    _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔓, 𝔙, 
                                            cse = cse, 
                                            skipzeros = skipzeros, 
                                            parallel = parallel,
                                            expression_module = @__MODULE__,
                                            expression = Val(false))::Tuple{<:Function, <:Function};

    # 𝓂.jacobian = buffer, func_exprs
𝓂.jacobian[2]

    ∇₁_parameters = derivatives[1][2][:,1:nps]

    lennz = nnz(∇₁_parameters)

    if (lennz / length(∇₁_parameters) > density_threshold) || (length(∇₁_parameters) < min_length)
        ∇₁_parameters_mat = convert(Matrix, ∇₁_parameters)
        buffer_parameters = zeros(Float64, size(∇₁_parameters))
    else
        ∇₁_parameters_mat = ∇₁_parameters
        buffer_parameters = similar(∇₁_parameters, Float64)
        buffer_parameters.nzval .= 0
    end

    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, func_∇₁_parameters = Symbolics.build_function(∇₁_parameters_mat, 𝔓, 𝔙, 
                                                        cse = cse, 
                                                        skipzeros = skipzeros, 
                                                        parallel = parallel,
                                                        expression_module = @__MODULE__,
                                                        expression = Val(false))::Tuple{<:Function, <:Function}

                                            
spX_order_1 = Symbolics.sparsejacobian(dyn_equations_sub, 𝔙) # nϵ x nx

spX_order_1[:,18:21]



spX_order_1_mapped = Symbolics.substitute(spX_order_1.nzval, SS_mapping)

𝔙[1:maximum(dyn_var_idxs)]



max_perturbation_order = 1

import SparseArrays: sparse!

typeof(dyn_equations)

# function take_nth_order_derivatives(
#     dyn_equations::Vector{Symbolics.Num},
#     𝔙::Vector{Symbolics.Num},
#     𝔓::Vector{Symbolics.Num},
#     SS_mapping::Dict{Symbolics.Num, Symbolics.Num},
#     nSS::Int;
#     max_perturbation_order::Int = 1,
#     output_compressed::Bool = true # Controls compression for X derivatives (order >= 2)
# )::Vector{Tuple{SparseMatrixCSC{Symbolics.Num, Int}, SparseMatrixCSC{Symbolics.Num, Int}}}#, Tuple{Symbolics.Arr{Symbolics.Num, 1}, Symbolics.Arr{Symbolics.Num, 1}}}

nx = length(𝔙)
np = length(𝔓)
nϵ = length(dyn_equations)

if max_perturbation_order < 1
    throw(ArgumentError("max_perturbation_order must be at least 1"))
end

results = [] # To store pairs of sparse matrices (X_matrix, P_matrix) for each order

# --- Order 1 ---
# Compute the 1st order derivative with respect to X (Jacobian)
spX_order_1 = Symbolics.sparsejacobian(dyn_equations, 𝔙) # nϵ x nx


spX_order_1_sub = copy(spX_order_1)

# spX_order_1_sub.nzval .= Symbolics.fast_substitute(spX_order_1_sub.nzval, Dict(Symbolics.scalarize(𝔛𝔛) .=> 𝔙))
spX_order_1_sub.nzval .= Symbolics.substitute(spX_order_1_sub.nzval, SS_mapping)

# Compute the derivative of the non-zeros of the 1st X-derivative w.r.t. P
# This is an intermediate step. The final P matrix will be built from this.
spP_of_flatX_nzval_order_1 = Symbolics.sparsejacobian(spX_order_1_sub.nzval, 𝔓[1:end-nSS]) # nnz(spX_order_1) x np

# Determine dimensions for the Order 1 P matrix
X_nrows_1 = nϵ
X_ncols_1 = nx
P_nrows_1 = X_nrows_1 * X_ncols_1
P_ncols_1 = np

# Build the Order 1 P matrix (dimensions nϵ*nx x np)
sparse_rows_1_P = Int[] # Row index in the flattened space of spX_order_1
sparse_cols_1_P = Int[] # Column index for parameters (1 to np)
sparse_vals_1_P = Symbolics.Num[]

# Map linear index in spX_order_1.nzval to its (row, col) in spX_order_1
nz_lin_to_rc_1 = Dict{Int, Tuple{Int, Int}}()
k_lin = 1
for j = 1:size(spX_order_1, 2) # col
    for ptr = spX_order_1.colptr[j]:(spX_order_1.colptr[j+1]-1)
            r = spX_order_1.rowval[ptr] # row
            nz_lin_to_rc_1[k_lin] = (r, j)
            k_lin += 1
    end
end


# Iterate through the non-zero entries of spP_of_flatX_nzval_order_1
k_temp_P = 1 # linear index counter for nzval
for p_col = 1:size(spP_of_flatX_nzval_order_1, 2) # Parameter index
    for i_ptr_temp_P = spP_of_flatX_nzval_order_1.colptr[p_col]:(spP_of_flatX_nzval_order_1.colptr[p_col+1]-1)
        temp_row = spP_of_flatX_nzval_order_1.rowval[i_ptr_temp_P] # Row index in spP_of_flatX_nzval (corresponds to temp_row-th nzval of spX_order_1)
        p_val = spP_of_flatX_nzval_order_1.nzval[i_ptr_temp_P] # Derivative value w.r.t. parameter

        # Get the (row, col) in spX_order_1 corresponding to this derivative
        r_X1, c_X1 = nz_lin_to_rc_1[temp_row]

        # Calculate the row index in spP_order_1 (flattened index of spX_order_1)
        # P_row_idx = (r_X1 - 1) * X_ncols_1 + c_X1
        P_row_idx = (c_X1 - 1) * X_nrows_1 + r_X1
        P_col_idx = p_col # Parameter column index

        push!(sparse_rows_1_P, P_row_idx)
        push!(sparse_cols_1_P, P_col_idx)
        push!(sparse_vals_1_P, p_val)

        k_temp_P += 1
    end
end

spP_order_1 = sparse!(sparse_rows_1_P, sparse_cols_1_P, sparse_vals_1_P, P_nrows_1, P_ncols_1)


# Store the pair for order 1
push!(results, (spX_order_1_sub, spP_order_1))

if max_perturbation_order > 1
    # --- Prepare for higher orders (Order 2 to max_perturbation_order) ---
    # Initialize map for Order 1: linear index in spX_order_1.nzval -> (row, (v1,))
    # This map is needed to trace indices for Order 2
    # We already built nz_lin_to_rc_1 above, reuse it and wrap the variable index in a Tuple
    nz_to_indices_prev = Dict{Int, Tuple{Int, Tuple{Int}}}()
    k_lin = 1
    for j = 1:size(spX_order_1, 2)
        for ptr = spX_order_1.colptr[j]:(spX_order_1.colptr[j+1]-1)
            r = spX_order_1.rowval[ptr]
            nz_to_indices_prev[k_lin] = (r, (j,)) # Store (equation row, (v1,))
            k_lin += 1
        end
    end

    nzvals_prev = spX_order_1.nzval # nzvals from Order 1 X-matrix

    # --- Iterate for orders n = 2, 3, ..., max_perturbation_order ---
    for n = 2:max_perturbation_order

        # Compute the Jacobian of the previous level's nzval w.r.t. 𝔛
        # This gives a flat matrix where rows correspond to non-zeros from order n-1 X-matrix
        # and columns correspond to the n-th variable we differentiate by (x_vn).
        sp_flat_curr_X_rn = Symbolics.sparsejacobian(nzvals_prev, 𝔙) # nnz(spX_order_(n-1)) x nx

        sp_flat_curr_X = copy(sp_flat_curr_X_rn)

        sp_flat_curr_X.nzval .= Symbolics.substitute(sp_flat_curr_X.nzval, SS_mapping)

        # Build the nz_to_indices map for the *current* level (order n)
        # Map: linear index in sp_flat_curr_X.nzval -> (original_row_f, (v_1, ..., v_n))
        nz_to_indices_curr = Dict{Int, Tuple{Int, Tuple{Vararg{Int}}}}()
        k_lin_curr = 1 # linear index counter for nzval of sp_flat_curr_X
        # Iterate through the non-zeros of the current flat Jacobian
        for col_curr = 1:size(sp_flat_curr_X, 2) # Column index in sp_flat_curr_X (corresponds to v_n)
            for ptr_curr = sp_flat_curr_X.colptr[col_curr]:(sp_flat_curr_X.colptr[col_curr+1]-1)
                row_curr = sp_flat_curr_X.rowval[ptr_curr] # Row index in sp_flat_curr_X (corresponds to the row_curr-th nzval of previous level)

                # Get previous indices info from the map of order n-1
                prev_info = nz_to_indices_prev[row_curr]
                orig_row_f = prev_info[1] # Original equation row
                vars_prev = prev_info[2] # Tuple of variables from previous order (v_1, ..., v_{n-1})

                # Append the current variable index (v_n)
                vars_curr = (vars_prev..., col_curr) # Full tuple (v_1, ..., v_n)

                # Store info for the current level's non-zero
                nz_to_indices_curr[k_lin_curr] = (orig_row_f, vars_curr)
                k_lin_curr += 1
            end
        end

        # --- Construct the X-derivative sparse matrix for order n (compressed or uncompressed) ---
        local spX_order_n # Declare variable to hold the resulting X matrix
        local X_ncols_n # Number of columns in the resulting spX_order_n matrix

        if output_compressed
            # COMPRESSED output: nϵ x binomial(nx + n - 1, n)
            sparse_rows_n = Int[]
            sparse_cols_n = Int[] # This will store the compressed column index
            sparse_vals_n = Symbolics.Num[]

            # Calculate the total number of compressed columns for order n
            X_ncols_n = Int(binomial(nx + n - 1, n))

            # Iterate through the non-zero entries of the current flat Jacobian (sp_flat_curr_X)
            k_flat_curr = 1 # linear index counter for nzval of sp_flat_curr_X
            for col_flat_curr = 1:size(sp_flat_curr_X, 2) # This corresponds to the n-th variable (v_n)
                for i_ptr_flat_curr = sp_flat_curr_X.colptr[col_flat_curr]:(sp_flat_curr_X.colptr[col_flat_curr+1]-1)
                    # row_flat_curr = sp_flat_curr_X.rowval[i_ptr_flat_curr] # Row index in sp_flat_curr_X
                    val = sp_flat_curr_X.nzval[i_ptr_flat_curr] # The derivative value

                    # Get the full info for this non-zero from the map
                    # The linear index in sp_flat_curr_X.nzval is k_flat_curr
                    orig_row_f, var_indices_full = nz_to_indices_curr[k_flat_curr] # (v_1, ..., v_n)

                    # Check the compression rule: v_n <= v_{n-1} <= ... <= v_1
                    is_compressed = true
                    for k_rule = 1:(n-1)
                        # Check v_{n-k_rule+1} <= v_{n-k_rule}
                        if var_indices_full[n-k_rule+1] > var_indices_full[n-k_rule]
                            is_compressed = false
                            break
                        end
                    end

                    if is_compressed
                        # Calculate the compressed column index c_n for the tuple (v_1, ..., v_n)
                        # using the derived formula: c_n = sum_{k=1}^{n-1} binomial(v_k + n - k - 1, n - k + 1) + v_n
                        compressed_col_idx = 0
                        for k_formula = 1:(n-1)
                            term = binomial(var_indices_full[k_formula] + n - k_formula - 1, n - k_formula + 1)
                            compressed_col_idx += term
                        end
                        # Add the last term: v_n (var_indices_full[n])
                        compressed_col_idx += var_indices_full[n]

                        push!(sparse_rows_n, orig_row_f)
                        push!(sparse_cols_n, compressed_col_idx)
                        push!(sparse_vals_n, val)
                    end

                    k_flat_curr += 1 # Increment linear index counter for sp_flat_curr_X.nzval
                end
            end
            # Construct the compressed sparse matrix for order n
            spX_order_n = sparse!(sparse_rows_n, sparse_cols_n, sparse_vals_n, X_nrows_1, X_ncols_n)

        else # output_compressed == false
            # UNCOMPRESSED output: nϵ x nx^n
            sparse_rows_n_uncomp = Int[]
            sparse_cols_n_uncomp = Int[] # Uncompressed column index (1 to nx^n)
            sparse_vals_n_uncomp = Symbolics.Num[]

            # Total number of uncompressed columns
            X_ncols_n = Int(BigInt(nx)^n) # Use BigInt for the power calculation, cast to Int

            # Iterate through the non-zero entries of the current flat Jacobian (sp_flat_curr_X)
            k_flat_curr = 1 # linear index counter for nzval of sp_flat_curr_X
            for col_flat_curr = 1:size(sp_flat_curr_X, 2) # This corresponds to the n-th variable (v_n)
                for i_ptr_flat_curr = sp_flat_curr_X.colptr[col_flat_curr]:(sp_flat_curr_X.colptr[col_flat_curr+1]-1)
                    # row_flat_curr = sp_flat_curr_X.rowval[i_ptr_flat_curr] # Row index in sp_flat_curr_X
                    val = sp_flat_curr_X.nzval[i_ptr_flat_curr] # The derivative value

                    # Get the full info for this non-zero from the map
                    # The linear index in sp_flat_curr_X.nzval is k_flat_curr
                    orig_row_f, var_indices_full = nz_to_indices_curr[k_flat_curr] # (v_1, ..., v_n)

                    # Calculate the UNCOMPRESSED column index for the tuple (v_1, ..., v_n)
                    # This maps the tuple (v1, ..., vn) to a unique index from 1 to nx^n
                    # Formula: 1 + (v1-1)*nx^(n-1) + (v2-1)*nx^(n-2) + ... + (vn-1)*nx^0
                    uncompressed_col_idx = 1 # 1-based
                    power_of_nx = BigInt(nx)^(n-1) # Start with nx^(n-1) for v1 term
                    for i = 1:n
                        uncompressed_col_col_idx_term = (var_indices_full[i] - 1) * power_of_nx
                        # Check for overflow before adding
                        # if (uncompressed_col_idx > 0 && uncompressed_col_col_idx_term > 0 && uncompressed_col_idx + uncompressed_col_col_idx_term <= uncompressed_col_idx) ||
                        #    (uncompressed_col_idx < 0 && uncompressed_col_col_idx_term < 0 && uncompressed_col_idx + uncompressed_col_col_idx_term >= uncompressed_col_idx)
                        #    error("Integer overflow calculating uncompressed column index")
                        # end
                        uncompressed_col_idx += uncompressed_col_col_idx_term

                        if i < n # Avoid nx^-1
                            power_of_nx = div(power_of_nx, nx) # Integer division
                        end
                    end

                    push!(sparse_rows_n_uncomp, orig_row_f)
                    push!(sparse_cols_n_uncomp, Int(uncompressed_col_idx)) # Cast to Int
                    push!(sparse_vals_n_uncomp, val)

                    k_flat_curr += 1 # Increment linear index counter for sp_flat_curr_X.nzval
                end
            end
            # Construct the uncompressed sparse matrix for order n
            spX_order_n = sparse!(sparse_rows_n_uncomp, sparse_cols_n_uncomp, sparse_vals_n_uncomp, X_nrows_1, X_ncols_n)

        end # End of if output_compressed / else


        # --- Compute the P-derivative sparse matrix for order n ---
        # This is the Jacobian of the nzval of the intermediate flat X-Jacobian (sp_flat_curr_X) w.r.t. 𝔓.
        # sp_flat_curr_X.nzval contains expressions for d^n f_i / (dx_v1 ... dx_vn) for all
        # non-zero such values that were propagated from the previous step.
        spP_of_flatX_nzval_curr = Symbolics.sparsejacobian(sp_flat_curr_X.nzval, 𝔓[1:end-nSS]) # nnz(sp_flat_curr_X) x np
        
        # Determine the desired dimensions of spP_order_n
        # Dimensions are (rows of spX_order_n * cols of spX_order_n) x np
        P_nrows_n = nϵ * X_ncols_n
        P_ncols_n = np

        sparse_rows_n_P = Int[] # Row index in the flattened space of spX_order_n (1 to P_nrows_n)
        sparse_cols_n_P = Int[] # Column index for parameters (1 to np)
        sparse_vals_n_P = Symbolics.Num[]

        # Iterate through the non-zero entries of spP_of_flatX_nzval_curr
        # Its rows correspond to the non-zeros in sp_flat_curr_X
        k_temp_P = 1 # linear index counter for nzval of spP_of_flatX_nzval_curr
        for p_col = 1:size(spP_of_flatX_nzval_curr, 2) # Column index in spP_of_flatX_nzval_curr (corresponds to parameter index)
            for i_ptr_temp_P = spP_of_flatX_nzval_curr.colptr[p_col]:(spP_of_flatX_nzval_curr.colptr[p_col+1]-1)
                temp_row = spP_of_flatX_nzval_curr.rowval[i_ptr_temp_P] # Row index in spP_of_flatX_nzval_curr (corresponds to the temp_row-th nzval of sp_flat_curr_X)
                p_val = spP_of_flatX_nzval_curr.nzval[i_ptr_temp_P] # The derivative w.r.t. parameter value

                # Get the full info for the X-derivative term that this P-derivative is from
                # temp_row is the linear index in sp_flat_curr_X.nzval
                # This corresponds to the derivative d^n f_orig_row_f / (dx_v1 ... dx_vn)
                orig_row_f, var_indices_full = nz_to_indices_curr[temp_row] # (v_1, ..., v_n)

                # We need to find the column index (X_col_idx) this term corresponds to
                # in the final spX_order_n matrix (which might be compressed or uncompressed)
                local X_col_idx # Column index in the final spX_order_n matrix (1 to X_ncols_n)

                if output_compressed
                    # Calculate the compressed column index
                    compressed_col_idx = 0
                    for k_formula = 1:(n-1)
                        term = binomial(var_indices_full[k_formula] + n - k_formula - 1, n - k_formula + 1)
                        compressed_col_idx += term
                    end
                    compressed_col_idx += var_indices_full[n]
                    X_col_idx = compressed_col_idx # The column in spX_order_n is the compressed one

                else # output_compressed == false
                    # Calculate the uncompressed column index
                    uncompressed_col_idx = 1
                    power_of_nx = BigInt(nx)^(n-1)
                    for i = 1:n
                        uncompressed_col_idx += (var_indices_full[i] - 1) * power_of_nx
                        if i < n
                            power_of_nx = div(power_of_nx, nx)
                        end
                    end
                    X_col_idx = Int(uncompressed_col_idx) # The column in spX_order_n is the uncompressed one
                end

                # Calculate the row index in spP_order_n
                # This maps the (orig_row_f, X_col_idx) pair in spX_order_n's grid to a linear index
                # Formula: (row_in_X - 1) * num_cols_in_X + col_in_X
                # P_row_idx = (orig_row_f - 1) * X_ncols_n + X_col_idx
                P_row_idx = (X_col_idx - 1) * nϵ + orig_row_f

                # The column index in spP_order_n is the parameter index
                P_col_idx = p_col

                push!(sparse_rows_n_P, P_row_idx)
                push!(sparse_cols_n_P, P_col_idx)
                push!(sparse_vals_n_P, p_val)

                k_temp_P += 1 # Increment linear index counter for spP_of_flatX_nzval_curr.nzval
            end
        end

        # Construct the P-derivative sparse matrix for order n
        # Dimensions are (rows of spX_order_n * cols of spX_order_n) x np
        spP_order_n = sparse!(sparse_rows_n_P, sparse_cols_n_P, sparse_vals_n_P, P_nrows_n, P_ncols_n)

        # Store the pair (X-matrix, P-matrix) for order n
        push!(results, (spX_order_n, spP_order_n))


        # Prepare for the next iteration (order n+1)
        # The nzvals for the next X-Jacobian step are the nzvals of the current flat X-Jacobian
        nzvals_prev = sp_flat_curr_X_rn.nzval
        # The map for the next step should provide info for order n derivatives
        nz_to_indices_prev = nz_to_indices_curr

    end # End of loop for orders n = 2 to max_perturbation_order
end

return results #, (𝔛, 𝔓) # Return results as a tuple of (X_matrix, P_matrix) pairs


spX_order_1_mapped[:,18:21]


SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
# calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]

# par = vcat(parameters,calibrated_parameters)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

vcat(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx)
# X = [SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; SS[dyn_ss_idx]; par; shocks_ss]




union(𝓂.var,𝓂.exo_past,𝓂.exo_future,𝓂.exo)

SS_and_pars_names = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)

nx = length(SS_and_pars_names)

np = length(𝓂.parameter_values)

nε = 𝓂.timings.nExo
    
nϵ = length(𝓂.dyn_equations)

Symbolics.@variables 𝔛[1:nx] 𝔓[1:np] # ε[1:nε]

𝔛ˢ = Symbolics.scalarize(𝔛)
𝔓ˢ = Symbolics.scalarize(𝔓)
εˢ = zeros(nε)

ϵˢ = zeros(Symbolics.Num, nϵ)

StSt = 𝔛ˢ[1:end - length(𝓂.calibration_equations)]
par = vcat(𝔓ˢ, 𝔛ˢ[(end - length(𝓂.calibration_equations)+1):end])

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

𝔙 = vcat(StSt[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]],εˢ)
𝔓ᵈ = vcat(par, StSt[dyn_ss_idx])

calc! = @RuntimeGeneratedFunction(funcs)

SS_and_pars_names = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)

nx = length(SS_and_pars_names)

np = length(𝓂.parameter_values)

nε = 𝓂.timings.nExo
    
nϵ = length(𝓂.dyn_equations)

Symbolics.@variables 𝔛[1:nx] 𝔓[1:np] # ε[1:nε]

𝔛ˢ = Symbolics.scalarize(𝔛)
𝔓ˢ = Symbolics.scalarize(𝔓)
εˢ = zeros(nε)

ϵˢ = zeros(Symbolics.Num, nϵ)

SS = 𝔛ˢ[1:end - length(𝓂.calibration_equations)]
par = vcat(𝔓ˢ, 𝔛ˢ[(end - length(𝓂.calibration_equations)+1):end])

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

𝔙 = vcat(SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]],εˢ)
𝔓ᵈ = vcat(par, SS[dyn_ss_idx])








include("../models/Guerrieri_Iacoviello_2017.jl")
𝓂 = Guerrieri_Iacoviello_2017

SS(𝓂)

using LinearAlgebra
Guerrieri_Iacoviello_2017.∂SS_equations_∂parameters[1]
lusp = lu(Guerrieri_Iacoviello_2017.∂SS_equations_∂SS_and_pars[1])




include("../models/NAWM_EAUS_2008.jl")
SS(NAWM_EAUS_2008)

include("../models/QUEST3_2009.jl")
SS(QUEST3_2009)

include("../models/Caldara_et_al_2012.jl")
SS(Caldara_et_al_2012)

include("../models/Smets_Wouters_2003.jl")
SS(Smets_Wouters_2003)

include("../models/Smets_Wouters_2007.jl")
SS(Smets_Wouters_2007)

include("../models/GNSS_2010.jl")
SS(GNSS_2010)

include("../models/Baxter_King_1993.jl")
SS(Baxter_King_1993)

include("../models/Backus_Kehoe_Kydland_1992.jl")
SS(Backus_Kehoe_Kydland_1992)

include("../models/Aguiar_Gopinath_2007.jl")
SS(Aguiar_Gopinath_2007)






unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)

ss_equations = vcat(𝓂.ss_equations, 𝓂.calibration_equations)



np = length(𝓂.parameters)
nu = length(unknowns)
nc = length(𝓂.calibration_equations_no_var)

Symbolics.@variables 𝔓[1:np] 𝔘[1:nu] ℭ[1:nc]

parameter_dict = Dict{Symbol, Symbol}()
back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
calib_vars = Symbol[]
calib_expr = []


for (i,v) in enumerate(𝓂.parameters)
    push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
    push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
end

for (i,v) in enumerate(unknowns)
    push!(parameter_dict, v => :($(Symbol("𝔘_$i"))))
    push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔘_$i"))), @__MODULE__) => 𝔘[i])
end

for (i,v) in enumerate(𝓂.calibration_equations_no_var)
    push!(calib_vars, v.args[1])
    push!(calib_expr, v.args[2])
    push!(parameter_dict, v.args[1] => :($(Symbol("ℭ_$i"))))
    push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("ℭ_$i"))), @__MODULE__) => ℭ[i])
end

calib_replacements = Dict{Symbol,Any}()
for (i,x) in enumerate(calib_vars)
    replacement = Dict(x => calib_expr[i])
    for ii in i+1:length(calib_vars)
        calib_expr[ii] = replace_symbols(calib_expr[ii], replacement)
    end
    push!(calib_replacements, x => calib_expr[i])
end


ss_equations_sub = ss_equations |> 
    x -> replace_symbols.(x, Ref(calib_replacements)) |> 
    x -> replace_symbols.(x, Ref(parameter_dict)) |> 
    x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
    x -> Symbolics.substitute.(x, Ref(back_to_array_dict))


lennz = length(ss_equations_sub)

if lennz > 1500
    parallel = Symbolics.ShardedForm(1500,4)
else
    parallel = Symbolics.SerialForm()
end

_, func_exprs = Symbolics.build_function(ss_equations_sub, 𝔓, 𝔘,
                                            cse = cse, 
                                            skipzeros = skipzeros, 
                                            parallel = parallel,
                                            expression_module = @__MODULE__,
                                            expression = Val(false))::Tuple{<:Function, <:Function}




SS(m, derivatives = false)

SS(m, derivatives = true)

get_solution(m)
get_std(m)

𝓂.∂SS_equations_∂SS_and_pars[2]
𝓂.∂SS_equations_∂SS_and_pars[1]


𝓂 = m
𝓂 = Smets_Wouters_2003
SS_and_pars = SS(𝓂, derivatives = false)

get_non_stochastic_steady_state_residuals(𝓂, collect(SS_and_pars))

m.SS_check_func


m.SS_solve_func
m.ss_solve_blocks
𝓂 = FS2000
𝓂 = NAWM_EAUS_2008
𝓂 = QUEST3_2009


@benchmark begin
    # clear_solution_caches!(m, :first_order)
    get_SS(m, silent = true, derivatives = false)
end setup = clear_solution_caches!(m, :first_order)

clear_solution_caches!(m, :first_order)

@profview for i in 1:10
    clear_solution_caches!(m, :first_order)
    get_SS(m, silent = true, derivatives = false)
end

include("../test/models/RBC_CME.jl")

include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")
m.SS_solve_func
m.ss_solve_blocks
m.ss_solve_blocks_in_place[1].ss_problem.jac_buffer

m.ss_solve_blocks_in_place[1].extended_ss_problem.jac_buffer


import MacroModelling: get_NSSS_and_parameters, clear_solution_caches!

include("../models/Smets_Wouters_2007.jl")

m = Smets_Wouters_2007


include("../models/NAWM_EAUS_2008.jl")

m = NAWM_EAUS_2008


m.SS_solve_func
SS(m, derivatives = false)

clear_solution_caches!(m, :first_order)
get_NSSS_and_parameters(m, m.parameter_values)

@benchmark get_NSSS_and_parameters(m, m.parameter_values) setup = clear_solution_caches!(m, :first_order)

@profview for i in 1:10
    clear_solution_caches!(m, :first_order)
    get_NSSS_and_parameters(m, m.parameter_values)
end

m.NSSS_solver_cache[end][1]

m.SS_solve_func

b = copy(m.ss_solve_blocks_in_place[1].ss_problem.func_buffer)
A = copy(m.ss_solve_blocks_in_place[1].ss_problem.jac_buffer)
# b = copy(m.parameter_values)


A \ A
dA = collect(A)
n = 200
A = rand(n,n)
B = rand(n,n)
X = rand(n,n)

@benchmark begin
prob = LinearProblem(A, B[:, 1])

sol_cache = init(prob)
# Solve for each column of B
for i in 1:3
    # Create a linear problem for the i-th column of B
    sol_cache.b = B[:, i]
    # Solve the system for the i-th column
    sol = LinearSolve.solve!(sol_cache)

    # Store the solution vector in the i-th column of X
    X[:, i] = sol.u
end
end
@benchmark A\B

using SparseArrays
spA = sprand(n,n,.1)

spB = sprand(n,n,.1)
B = collect(spB)

@benchmark spA\B


@benchmark begin
    prob = LinearProblem(spA, spB[:, 1])
    
    sol_cache = init(prob, UMFPACKFactorization())
    # Solve for each column of B
    for i in 1:n
        # Create a linear problem for the i-th column of B
        sol_cache.b .= spB[:, i]
        # Solve the system for the i-th column
        sol = LinearSolve.solve!(sol_cache)
    
        # Store the solution vector in the i-th column of X
        X[:, i] = sol.u
    end
end

# Create a linear problem
prob = LinearSolve.LinearProblem(A, B, MetalLUFactorization())

# Solve the system
sol = solve(prob)

# Access the solution matrix X
X = sol.u

prob = LinearProblem(A,b)
sol_cache = LinearSolve.init(prob, UMFPACKFactorization())
sol = LinearSolve.solve!(sol_cache)

A\b

prob = LinearProblem(dA,b)
sol_cache = LinearSolve.init(prob)
sol = LinearSolve.solve!(sol_cache)


A*sol - b
A\b - sol
A .= 1
sol_cache.A = A
sol = LinearSolve.solve!(sol_cache)

sum(sol)
LinearSolve.LinearProblem(A,b)
guess = [ 0.15666405734651248
1.2082329932960632
0.022590361445783146
9.438431853512776
1.4214505803483093]

# [0.9999998807907104, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
# MacroModelling.solver_parameters(1.9479518608134938, 0.02343520604394183, 5.125002799990568, 0.02387522857907376, 0.2239226474715968, 4.889172213411495, 1.747880258818237, 2.8683242331457, 0.938229356687311, 1.4890887655876235, 1.6261504814901664, 11.26863249187599, 36.05486169712279, 6.091535897587629, 11.73936761697657, 3.189349432626493, 0.21045178305336348, 0.17122196312330415, 13.251662547139363, 5.282429995876679, 1, 0.0, 2)
# true
a ≈ b
a = [-12.433767642918585 0.0 0.0 -5.3762485340939765 5.3762502416961295 0.0 0.0 0.0; 12.433767642918585 -5.3762502416961295 -28.399665542074878 -23.0234170079809 0.0 0.0 0.0 0.0; -3.764232002406259 0.0 5.370859922641225 1.2120486067971766e-7 0.0 0.0 0.0 0.0; 0.0 1.0177608119544763 0.0 0.0 -1.0177608119544763 0.0 0.0 0.0; 0.0 0.0 0.0 0.2544402029886191 -0.2544402029886191 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 -1.937937047481161 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 -1.0111874208078342 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.4135047829953282]
b = [-12.433767642918587 0.0 0.0 -5.3762485340939765 5.3762502416961295 0.0 0.0 0.0; 12.433767642918587 -5.3762502416961295 -28.399665542074878 -23.023417007980903 0.0 0.0 0.0 0.0; 0.0 1.0177608119544763 0.0 0.0 -1.0177608119544763 0.0 0.0 0.0; -3.764232002406259 -0.0 5.370859922641225 1.2120486067971766e-7 -0.0 -0.0 -0.0 -0.0; 0.0 0.0 0.0 0.2544402029886191 -0.2544402029886191 0.0 0.0 0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -1.937937047481161 -0.0 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -1.0111874208078342 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -1.4135047829953282]
[0.8813735027258436, 2.366350881077193, 2.3663746058124353, 2.366374605151466, 2.366374573597027, 1.2803606388441917, 0.14944312018495765, 0.8806644521167746]

[0.8813735027258436, 2.366350881077193, 2.3663746058124353, 2.366374605151466, 2.366374573597027, 1.2803606388441917, 0.14944312018495765, 0.8806644521167746]

# [-12.433767626388102 0.0 0.0 -5.376248529783475 5.376250070701516 0.0 0.0 0.0; 12.433767626388102 -5.376124918137118 -28.399665514162745 -23.023416985040242 0.0 0.0 0.0 0.0; 0.0 1.0177371208803547 0.0 0.0 -1.0177362718097986 0.0 0.0 0.0; -3.7642320012329327 -0.0 5.370859921823067 1.2120486068320023e-7 -0.0 -0.0 -0.0 -0.0; 0.0 0.0 0.0 0.2544402111672444 -0.25444021144994905 0.0 0.0 0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -1.937937047481161 -0.0 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -1.0111874208078342 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -1.4135047829953282]
# [0.8414630548452718, 2.067702729890952, 1.5444697302150523, 2.217027361455363, 2.120867377098202, 1.2803606388441917, 0.14944312018495765, 0.8806644521167747]
# [0.8414630548452863, 2.067702729890989, 1.5444697302150652, 2.217027361455346, 2.120867377098219, 1.2803606388441917, 0.14944312018495765, 0.8806644521167747]

[0.15603018290473208, 1.0212331164230357, 0.022588440484669205, 2.940731629734951, 1.1503870139467807, 1.2803606388441915, 0.14944312018495765, 0.8806644521167747]
[0.15603018290473153, 1.0212331164230348, 0.02258844048466909, 2.9407316297349495, 1.1503870139467793, 1.2803606388441915, 0.14944312018495765, 0.8806644521167747]

import MacroModelling: Tolerances, block_solver, levenberg_marquardt, solve_ss, newton
import LinearAlgebra as ℒ
import DataStructures: CircularBuffer

𝓂 = m
tol = Tolerances()
verbose = true
initial_parameters = 𝓂.parameter_values
cold_start = true
solver_parameters  = 𝓂.solver_parameters[2:end]

 initial_parameters = if typeof(initial_parameters) == Vector{Float64}
                  initial_parameters
              else
                  ℱ.value.(initial_parameters)
              end
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3682 =#
          initial_parameters_tmp = copy(initial_parameters)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3684 =#
          parameters = copy(initial_parameters)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3685 =#
          params_flt = copy(initial_parameters)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3687 =#
          current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - initial_parameters)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3688 =#
          closest_solution_init = 𝓂.NSSS_solver_cache[end]
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3690 =#
          for pars = 𝓂.NSSS_solver_cache
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3691 =#
              copy!(initial_parameters_tmp, pars[end])
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3693 =#
              ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3695 =#
              latest = sum(abs2, initial_parameters_tmp)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3696 =#
              if latest <= current_best
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3697 =#
                  current_best = latest
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3698 =#
                  closest_solution_init = pars
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3700 =#
          end
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
          range_iters = 0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3706 =#
          solution_error = 1.0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3707 =#
          solved_scale = 0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3709 =#
          scale = 1.0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3711 =#
          NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3712 =#
          push!(NSSS_solver_cache_scale, closest_solution_init)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3714 =#
        #   while range_iters <= if cold_start
        #                   1
        #               else
        #                   500
        #               end && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3715 =#
              range_iters += 1
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
              fail_fast_solvers_only = if range_iters > 1
                      true
                  else
                      false
                  end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3718 =#
              if abs(solved_scale - scale) < 0.0001
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3720 =#
                  # break
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3733 =#
              current_best = sum(abs2, (NSSS_solver_cache_scale[end])[end] - initial_parameters)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3734 =#
              closest_solution = NSSS_solver_cache_scale[end]
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3736 =#
              for pars = NSSS_solver_cache_scale
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3737 =#
                  copy!(initial_parameters_tmp, pars[end])
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3739 =#
                  ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3741 =#
                  latest = sum(abs2, initial_parameters_tmp)
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3743 =#
                  if latest <= current_best
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3744 =#
                      current_best = latest
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3745 =#
                      closest_solution = pars
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3747 =#
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3751 =#
              if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3752 =#
                  parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
              else
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3754 =#
                  parameters = copy(initial_parameters)
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3756 =#
              params_flt = parameters
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3760 =#
              EA_SIZE = parameters[22]
              EA_OMEGA = parameters[23]
              EA_BETA = parameters[24]
              EA_SIGMA = parameters[25]
              EA_KAPPA = parameters[26]
              EA_ZETA = parameters[27]
              EA_DELTA = parameters[28]
              EA_ETA = parameters[29]
              EA_ETAI = parameters[30]
              EA_ETAJ = parameters[31]
              EA_XII = parameters[32]
              EA_XIJ = parameters[33]
              EA_CHII = parameters[34]
              EA_CHIJ = parameters[35]
              EA_ALPHA = parameters[36]
              EA_THETA = parameters[38]
              EA_XIH = parameters[39]
              EA_XIX = parameters[40]
              EA_CHIH = parameters[41]
              EA_CHIX = parameters[42]
              EA_NUC = parameters[43]
              EA_MUC = parameters[44]
              EA_NUI = parameters[45]
              EA_MUI = parameters[46]
              EA_GAMMAV1 = parameters[47]
              EA_GAMMAV2 = parameters[48]
              EA_GAMMAU2 = parameters[51]
              EA_GAMMAB1 = parameters[54]
              EA_BYTARGET = parameters[55]
              EA_PHITB = parameters[56]
              EA_GYBAR = parameters[57]
              EA_TRYBAR = parameters[58]
              EA_TAUCBAR = parameters[59]
              EA_TAUKBAR = parameters[60]
              EA_TAUNBAR = parameters[61]
              EA_TAUWHBAR = parameters[62]
              EA_TAUWFBAR = parameters[63]
              EA_UPSILONT = parameters[64]
              EA_UPSILONTR = parameters[65]
              EA_PI4TARGET = parameters[66]
              EA_PHIRR = parameters[67]
              EA_PHIRPI = parameters[68]
              EA_BFYTARGET = parameters[70]
              EA_PYBAR = parameters[81]
              EA_YBAR = parameters[82]
              EA_PIBAR = parameters[84]
              EA_PSIBAR = parameters[85]
              EA_QBAR = parameters[86]
              EA_TAUDBAR = parameters[87]
              EA_ZBAR = parameters[88]
              US_SIZE = parameters[89]
              US_OMEGA = parameters[90]
              US_BETA = parameters[91]
              US_SIGMA = parameters[92]
              US_KAPPA = parameters[93]
              US_ZETA = parameters[94]
              US_DELTA = parameters[95]
              US_ETA = parameters[96]
              US_ETAI = parameters[97]
              US_ETAJ = parameters[98]
              US_XII = parameters[99]
              US_XIJ = parameters[100]
              US_CHII = parameters[101]
              US_CHIJ = parameters[102]
              US_ALPHA = parameters[103]
              US_THETA = parameters[105]
              US_XIH = parameters[106]
              US_XIX = parameters[107]
              US_CHIH = parameters[108]
              US_CHIX = parameters[109]
              US_NUC = parameters[110]
              US_MUC = parameters[111]
              US_NUI = parameters[112]
              US_MUI = parameters[113]
              US_GAMMAV1 = parameters[114]
              US_GAMMAV2 = parameters[115]
              US_GAMMAU2 = parameters[118]
              US_BYTARGET = parameters[122]
              US_PHITB = parameters[123]
              US_GYBAR = parameters[124]
              US_TRYBAR = parameters[125]
              US_TAUCBAR = parameters[126]
              US_TAUKBAR = parameters[127]
              US_TAUNBAR = parameters[128]
              US_TAUWHBAR = parameters[129]
              US_TAUWFBAR = parameters[130]
              US_UPSILONT = parameters[131]
              US_UPSILONTR = parameters[132]
              US_PI4TARGET = parameters[133]
              US_PHIRR = parameters[134]
              US_PHIRPI = parameters[135]
              US_PYBAR = parameters[147]
              US_TAUDBAR = parameters[148]
              US_YBAR = parameters[149]
              US_PIBAR = parameters[150]
              US_PSIBAR = parameters[151]
              US_QBAR = parameters[152]
              US_ZBAR = parameters[153]
              US_RER = parameters[154]
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3761 =#
              EA_OMEGA = min(max(EA_OMEGA, 2.220446049250313e-16), 1.0e12)
              EA_ALPHA = min(max(EA_ALPHA, 2.220446049250313e-16), 1.0e12)
              EA_NUC = min(max(EA_NUC, 2.220446049250313e-16), 1.0e12)
              EA_NUI = min(max(EA_NUI, 2.220446049250313e-16), 1.0e12)
              EA_PI4TARGET = min(max(EA_PI4TARGET, 2.220446049250313e-16), 1.0e12)
              EA_ZBAR = min(max(EA_ZBAR, 2.220446049250313e-16), 1.0e12)
              US_OMEGA = min(max(US_OMEGA, 2.220446049250313e-16), 1.0e12)
              US_ALPHA = min(max(US_ALPHA, 2.220446049250313e-16), 1.0e12)
              US_NUC = min(max(US_NUC, 2.220446049250313e-16), 1.0e12)
              US_NUI = min(max(US_NUI, 2.220446049250313e-16), 1.0e12)
              US_PI4TARGET = min(max(US_PI4TARGET, 2.220446049250313e-16), 1.0e12)
              US_ZBAR = min(max(US_ZBAR, 2.220446049250313e-16), 1.0e12)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3762 =#
              EA_RRSTAR = 1 / EA_BETA
              US_RRSTAR = 1 / US_BETA
              EA_interest_EXOG = EA_BETA ^ -1 * EA_PI4TARGET ^ (1 / 4)
              US_interest_EXOG = US_BETA ^ -1 * US_PI4TARGET ^ (1 / 4)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3763 =#
              NSSS_solver_cache_tmp = []
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3764 =#
              solution_error = 0.0
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3765 =#
              iters = 0
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3766 =#
              EA_RP = 0
              EA_RERDEP = 1
              EA_GAMMAIMC = 0
              EA_GAMMAIMCDAG = 1
              EA_GAMMAIMI = 0
              ➕₃₅ = min(max(2.220446049250313e-16, US_GAMMAV1 * US_GAMMAV2), 1.0e12)
              solution_error += abs(➕₃₅ - US_GAMMAV1 * US_GAMMAV2)
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              US_TAUWH = US_TAUWHBAR
              US_TRY = US_TRYBAR
              US_TR = US_PYBAR * US_TRY * US_YBAR
              US_TRI = US_TR * US_UPSILONTR
              US_TRJ = ((US_OMEGA * US_TRI + US_TR) - US_TRI) / US_OMEGA
              US_TAUC = US_TAUCBAR
              US_GY = US_GYBAR
              US_TAUD = US_TAUDBAR
              US_TAUWF = US_TAUWFBAR
              US_Z = US_ZBAR
              solution_error += abs(min(max(2.220446049250313e-16, US_Z), 1.0e12) - US_Z)
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3529 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3529 =#
                      println("Failed for bounded variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3529 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3529 =#
                  # continue
              end
              ➕₄₁ = min(max(2.220446049250313e-16, 1 - US_ALPHA), 1.0e12)
              solution_error += abs(➕₄₁ - (1 - US_ALPHA))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              US_GAMMAI = 0
              ➕₅₂ = min(max(2.220446049250313e-16, 1 - US_NUC), 1.0e12)
              solution_error += abs(➕₅₂ - (1 - US_NUC))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              US_GAMMAIMC = 0
              EA_GY = EA_GYBAR
              EA_GAMMAIMIDAG = 1
              EA_TAUK = EA_TAUKBAR
              EA_GAMMAI = 0
              EA_GAMMAIDER = 0
              EA_Z = EA_ZBAR
              solution_error += abs(min(max(2.220446049250313e-16, EA_Z), 1.0e12) - EA_Z)
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3529 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3529 =#
                      println("Failed for bounded variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3529 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3529 =#
                  # continue
              end
              ➕₂ = min(max(2.220446049250313e-16, EA_GAMMAV1 * EA_GAMMAV2), 1.0e12)
              solution_error += abs(➕₂ - EA_GAMMAV1 * EA_GAMMAV2)
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              EA_TAUN = EA_TAUNBAR
              EA_TRY = EA_TRYBAR
              EA_TR = EA_PYBAR * EA_TRY * EA_YBAR
              EA_TRI = EA_TR * EA_UPSILONTR
              EA_TRJ = ((EA_OMEGA * EA_TRI + EA_TR) - EA_TRI) / EA_OMEGA
              ➕₁₉ = min(max(2.220446049250313e-16, 1 - EA_NUC), 1.0e12)
              solution_error += abs(➕₁₉ - (1 - EA_NUC))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              EA_TAUWF = EA_TAUWFBAR
              EA_TAUC = EA_TAUCBAR
              EA_TAUWH = EA_TAUWHBAR
              EA_TAUD = EA_TAUDBAR
              ➕₁₂ = min(max(2.220446049250313e-16, 1 - EA_OMEGA), 1.0e12)
              solution_error += abs(➕₁₂ - (1 - EA_OMEGA))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              ➕₈ = min(max(2.220446049250313e-16, 1 - EA_ALPHA), 1.0e12)
              solution_error += abs(➕₈ - (1 - EA_ALPHA))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              US_GAMMAIMI = 0
              ➕₅₅ = min(max(2.220446049250313e-16, 1 - US_NUI), 1.0e12)
              solution_error += abs(➕₅₅ - (1 - US_NUI))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              US_TAUK = US_TAUKBAR
              ➕₄₅ = min(max(2.220446049250313e-16, 1 - US_OMEGA), 1.0e12)
              solution_error += abs(➕₄₅ - (1 - US_OMEGA))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              US_GAMMAIMIDAG = 1
              US_GAMMAIMCDAG = 1
              US_TAUN = US_TAUNBAR
              US_GAMMAIDER = 0
              ➕₂₂ = min(max(2.220446049250313e-16, 1 - EA_NUI), 1.0e12)
              solution_error += abs(➕₂₂ - (1 - EA_NUI))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3510 =#
                  # continue
              end
              ➕₆₈ = min(1.0e12, max(eps(), US_PI4TARGET))
              ➕₆₉ = min(1.0e12, max(eps(), EA_PI4TARGET))
              ➕₇₀ = min(1.0e12, max(eps(), EA_OMEGA))
              ➕₇₁ = min(1.0e12, max(eps(), EA_ALPHA))
              ➕₇₂ = min(1.0e12, max(eps(), EA_NUC))
              ➕₇₃ = min(1.0e12, max(eps(), US_ALPHA))
              ➕₇₄ = min(1.0e12, max(eps(), EA_NUI))
              ➕₇₅ = min(1.0e12, max(eps(), US_NUC))
              ➕₇₆ = min(1.0e12, max(eps(), US_NUI))
              ➕₇₇ = min(1.0e12, max(eps(), US_OMEGA))
              params_and_solved_vars = [EA_SIZE, EA_OMEGA, EA_BETA, EA_SIGMA, EA_KAPPA, EA_ZETA, EA_DELTA, EA_ETA, EA_ETAI, EA_ETAJ, EA_XII, EA_XIJ, EA_CHII, EA_CHIJ, EA_ALPHA, EA_THETA, EA_XIH, EA_XIX, EA_CHIH, EA_CHIX, EA_NUC, EA_MUC, EA_NUI, EA_MUI, EA_GAMMAV1, EA_GAMMAV2, EA_GAMMAU2, EA_GAMMAB1, EA_BYTARGET, EA_PHITB, EA_TAUKBAR, EA_UPSILONT, EA_PI4TARGET, EA_PHIRR, EA_PHIRPI, EA_BFYTARGET, EA_PYBAR, EA_YBAR, EA_PIBAR, EA_PSIBAR, EA_QBAR, US_SIZE, US_OMEGA, US_BETA, US_SIGMA, US_KAPPA, US_ZETA, US_DELTA, US_ETA, US_ETAI, US_ETAJ, US_XII, US_XIJ, US_CHII, US_CHIJ, US_ALPHA, US_THETA, US_XIH, US_XIX, US_CHIH, US_CHIX, US_NUC, US_MUC, US_NUI, US_MUI, US_GAMMAV1, US_GAMMAV2, US_GAMMAU2, US_BYTARGET, US_PHITB, US_TAUKBAR, US_UPSILONT, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PYBAR, US_YBAR, US_PIBAR, US_PSIBAR, US_QBAR, US_RER, EA_RRSTAR, US_RRSTAR, EA_GY, EA_TAUC, EA_TAUD, EA_TAUK, EA_TAUN, EA_TAUWF, EA_TAUWH, EA_TR, EA_TRJ, EA_Z, US_GY, US_TAUC, US_TAUD, US_TAUK, US_TAUN, US_TAUWF, US_TAUWH, US_TR, US_TRJ, US_Z, ➕₂, ➕₈, ➕₁₂, ➕₁₉, ➕₂₂, ➕₃₅, ➕₄₁, ➕₄₅, ➕₅₂, ➕₅₅, ➕₆₈, ➕₆₉, ➕₇₀, ➕₇₁, ➕₇₂, ➕₇₃, ➕₇₄, ➕₇₅, ➕₇₆, ➕₇₇]
              lbs = [-1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16]
              ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 600.0, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
              inits = [max.(lbs[1:length(closest_solution[1])], min.(ubs[1:length(closest_solution[1])], closest_solution[1])), closest_solution[2]]
              




            SS_solve_block = 𝓂.ss_solve_blocks_in_place[1];

            parameters_and_solved_vars = params_and_solved_vars

            guess_and_pars_solved_vars = inits

            n_block = 1

              solved_yet = false

              guess = guess_and_pars_solved_vars[1]
          
              sol_values = guess
          
              closest_parameters_and_solved_vars = sum(abs, guess_and_pars_solved_vars[2]) == Inf ? parameters_and_solved_vars : guess_and_pars_solved_vars[2]
          
              # res = ss_solve_blocks(parameters_and_solved_vars, guess)
          
              SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.func_buffer, guess, parameters_and_solved_vars, 0) # TODO: make the block a struct
              # TODO: do the function creation with Symbolics as this will solve the compilation bottleneck for large functions
          
              res = SS_solve_block.ss_problem.func_buffer
          
              sol_minimum  = ℒ.norm(res)
          
                  rel_sol_minimum = 1.0
              
              if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                  solved_yet = true
          
                  if verbose
                      println("Block: $n_block, - Solved using previous solution; residual norm: $sol_minimum")
                  end
              end
          
              total_iters = [0,0]
          
              SS_optimizer = levenberg_marquardt
            #   SS_optimizer = newton

              
              guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only




                xtol = tol.NSSS_xtol
                ftol = tol.NSSS_ftol
                rel_xtol = tol.NSSS_rel_xtol
                separate_starting_value = false
                solver_params = solver_parameters[1]
                extended_problem = true


    if separate_starting_value isa Float64
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], fill(separate_starting_value, length(guess))))
        sol_values_init[ubs[1:length(guess)] .<= 1] .= .1 # capture cases where part of values is small
    else
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], [g < 1e12 ? g : solver_params.starting_value for g in guess]))
    end


    
    fnj = extended_problem ? SS_solve_block.extended_ss_problem : SS_solve_block.ss_problem;
    parameters = solver_params
    upper_bounds = ubs
    lower_bounds = lbs
    initial_guess = extended_problem ? vcat(sol_values_init, closest_parameters_and_solved_vars) : sol_values_init
import MacroModelling: transform


    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    iterations = 250
    
    ϕ̄ = parameters.ϕ̄
    ϕ̂ = parameters.ϕ̂
    μ̄¹ = parameters.μ̄¹
    μ̄² = parameters.μ̄²
    p̄¹ = parameters.p̄¹
    p̄² = parameters.p̄²
    ρ = parameters.ρ
    ρ¹ = parameters.ρ¹
    ρ² = parameters.ρ²
    ρ³ = parameters.ρ³
    ν = parameters.ν
    λ¹ = parameters.λ¹
    λ² = parameters.λ²
    λ̂¹ = parameters.λ̂¹
    λ̂² = parameters.λ̂²
    λ̅¹ = parameters.λ̅¹
    λ̅² = parameters.λ̅²
    λ̂̅¹ = parameters.λ̂̅¹
    λ̂̅² = parameters.λ̂̅²
    transformation_level = parameters.transformation_level
    shift = parameters.shift
    backtracking_order = parameters.backtracking_order

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert all(lower_bounds .< upper_bounds)
    @assert backtracking_order ∈ [2,3] "Backtracking order can only be quadratic (2) or cubic (3)."

    max_linesearch_iterations = 600

    # function f̂(x) 
    #     f(undo_transform(x,transformation_level))  
    # #     # f(undo_transform(x,transformation_level,shift))  
    # end

    upper_bounds  = transform(upper_bounds,transformation_level)
    # upper_bounds  = transform(upper_bounds,transformation_level,shift)
    lower_bounds  = transform(lower_bounds,transformation_level)
    # lower_bounds  = transform(lower_bounds,transformation_level,shift)

    current_guess = copy(transform(initial_guess,transformation_level))
    current_guess_untransformed = copy(transform(initial_guess,transformation_level))
    # current_guess = copy(transform(initial_guess,transformation_level,shift))
    previous_guess = similar(current_guess)
    previous_guess_untransformed = similar(current_guess)
    guess_update = similar(current_guess)
    factor = similar(current_guess)
    # ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))
    ∇ = fnj.jac_buffer
    ∇̂ = similar(∇)

    if ∇ isa SparseMatrixCSC
        prob = 𝒮.LinearProblem(∇, guess_update, 𝒮.UMFPACKFactorization())
    else
        prob = 𝒮.LinearProblem(∇, guess_update)#, 𝒮.CholeskyFactorization)
    end

    sol_cache = 𝒮.init(prob)
    
    # prep = 𝒟.prepare_jacobian(f̂, backend, current_guess)

    largest_step = T(1.0)
    largest_residual = T(1.0)
    largest_relative_step = T(1.0)

    μ¹ = μ̄¹
    μ² = μ̄²

    p¹ = p̄¹
    p² = p̄²

    grad_iter = 0
    func_iter = 0


    if ∇ isa SparseMatrixCSC
        prob = 𝒮.LinearProblem(∇, guess_update, 𝒮.UMFPACKFactorization())
    else
        prob = 𝒮.LinearProblem(∇, guess_update)#, 𝒮.CholeskyFactorization)
    end

    sol_cache = 𝒮.init(prob)
    
    # prep = 𝒟.prepare_jacobian(f̂, backend, current_guess)

    largest_step = T(1.0)
    largest_residual = T(1.0)
    largest_relative_step = T(1.0)

    μ¹ = μ̄¹
    μ² = μ̄²

    p¹ = p̄¹
    p² = p̄²

    grad_iter = 0
    func_iter = 0

    for iter in 1:iterations
        # make the jacobian and f calls nonallocating
        copy!(current_guess_untransformed, current_guess)
        
        if transformation_level > 0
            factor .= 1
            for _ in 1:transformation_level
                factor .*= cosh.(current_guess_untransformed)
                current_guess_untransformed .= sinh.(current_guess_untransformed)
            end
        end

        fnj.jac(∇, current_guess_untransformed, parameters_and_solved_vars, transformation_level)
        # 𝒟.jacobian!(f̂, ∇, prep, backend, current_guess)

        



    sol_new_tmp, info = SS_optimizer(   extended_problem ? SS_solve_block.extended_ss_problem : SS_solve_block.ss_problem,
                                        extended_problem ? vcat(sol_values_init, closest_parameters_and_solved_vars) : sol_values_init,
                                        parameters_and_solved_vars,
                                        extended_problem ? lbs : lbs[1:length(guess)],
                                        extended_problem ? ubs : ubs[1:length(guess)],
                                        solver_params,
                                        tol = tol   )




              sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
              # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
              guesses[1], 
              solver_parameters[1],
                                                  true,
                                                  false)



              solution = block_solver(params_and_solved_vars, 1, 𝓂.ss_solve_blocks_in_place[1], inits, lbs, ubs, solver_parameters, fail_fast_solvers_only, cold_start, verbose)
              
              iters += (solution[2])[2]
              solution_error += (solution[2])[1]





          initial_parameters = if typeof(initial_parameters) == Vector{Float64}
                  initial_parameters
              else
                  ℱ.value.(initial_parameters)
              end
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3681 =#
          initial_parameters_tmp = copy(initial_parameters)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3683 =#
          parameters = copy(initial_parameters)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3684 =#
          params_flt = copy(initial_parameters)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3686 =#
          current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - initial_parameters)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3687 =#
          closest_solution_init = 𝓂.NSSS_solver_cache[end]
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3689 =#
          for pars = 𝓂.NSSS_solver_cache
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3690 =#
              copy!(initial_parameters_tmp, pars[end])
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3692 =#
              ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3694 =#
              latest = sum(abs2, initial_parameters_tmp)
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3695 =#
              if latest <= current_best
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3696 =#
                  current_best = latest
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3697 =#
                  closest_solution_init = pars
              end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3699 =#
          end
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3704 =#
          range_iters = 0
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3705 =#
          solution_error = 1.0
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3706 =#
          solved_scale = 0
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3708 =#
          scale = 1.0
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3710 =#
          NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3711 =#
          push!(NSSS_solver_cache_scale, closest_solution_init)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3713 =#
        #   while range_iters <= if cold_start
        #                   1
        #               else
        #                   500
        #               end && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3714 =#
              range_iters += 1
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3715 =#
              fail_fast_solvers_only = if range_iters > 1
                      true
                  else
                      false
                  end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3717 =#
            #   if abs(solved_scale - scale) < 0.0001
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3719 =#
                #   # break
            #   end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3732 =#
              current_best = sum(abs2, (NSSS_solver_cache_scale[end])[end] - initial_parameters)
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3733 =#
              closest_solution = NSSS_solver_cache_scale[end]
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3735 =#
              for pars = NSSS_solver_cache_scale
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3736 =#
                  copy!(initial_parameters_tmp, pars[end])
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3738 =#
                  ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3740 =#
                  latest = sum(abs2, initial_parameters_tmp)
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3742 =#
                  if latest <= current_best
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3743 =#
                      current_best = latest
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3744 =#
                      closest_solution = pars
                  end
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3746 =#
              end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3750 =#
              if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3751 =#
                  parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
              else
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3753 =#
                  parameters = copy(initial_parameters)
              end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3755 =#
              params_flt = parameters
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3759 =#
              cap_share = parameters[1]
              R_ss = parameters[2]
              I_K_ratio = parameters[3]
              phi_pi = parameters[4]
              Pi_real = parameters[7]
              rhoz = parameters[8]
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3760 =#
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3761 =#
              Pi_ss = R_ss - Pi_real
              rho_z_delta = rhoz
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3762 =#
              NSSS_solver_cache_tmp = []
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3763 =#
              solution_error = 0.0
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3764 =#
              iters = 0
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3765 =#
              Pi = Pi_ss
              ➕₃ = min(600, max(-1.0e12, R_ss - 1))
              solution_error += +(abs(➕₃ - (R_ss - 1)))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3520 =#
                  if verbose
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3520 =#
                      println("Failed for analytical variables with error $(solution_error)")
                  end
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3520 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3520 =#
                #   # continue
              end
              R = exp(➕₃)
              beta = Pi / R
              solution_error += abs(min(max(1.1920928955078125e-7, beta), 0.9999998807907104) - beta)
              if solution_error > tol.NSSS_acceptance_tol
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3528 =#
                  if verbose
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3528 =#
                      println("Failed for bounded variables with error $(solution_error)")
                  end
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3528 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3528 =#
                #   # continue
              end
              ➕₁ = min(max(2.220446049250313e-16, (R * beta) ^ (1 / phi_pi)), 1.0e12)
              solution_error += abs(➕₁ - (R * beta) ^ (1 / phi_pi))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3509 =#
                  if verbose
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3509 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3509 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3509 =#
                #   # continue
              end
              Pibar = Pi / ➕₁
              A = 1
              z_delta = 1
              params_and_solved_vars = [cap_share, I_K_ratio, beta]
              lbs = [1.1920928955078125e-7, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, 1.1920928955078125e-7]
              ubs = [0.9999998807907104, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 0.9999998807907104]
              inits = [max.(lbs[1:length(closest_solution[1])], min.(ubs[1:length(closest_solution[1])], closest_solution[1])), closest_solution[2]]






              solution = block_solver(params_and_solved_vars, 1, 𝓂.ss_solve_blocks_in_place[1], inits, lbs, ubs, solver_parameters, fail_fast_solvers_only, cold_start, verbose)

              




            SS_solve_block = 𝓂.ss_solve_blocks_in_place[1];

            parameters_and_solved_vars = params_and_solved_vars

            guess_and_pars_solved_vars = inits

            n_block = 1

              solved_yet = false

              guess = guess_and_pars_solved_vars[1]
          
              sol_values = guess
          
              closest_parameters_and_solved_vars = sum(abs, guess_and_pars_solved_vars[2]) == Inf ? parameters_and_solved_vars : guess_and_pars_solved_vars[2]
          
              # res = ss_solve_blocks(parameters_and_solved_vars, guess)
          
              SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.func_buffer, guess, parameters_and_solved_vars, 0) # TODO: make the block a struct
              # TODO: do the function creation with Symbolics as this will solve the compilation bottleneck for large functions
          
              res = SS_solve_block.ss_problem.func_buffer
          
              sol_minimum  = ℒ.norm(res)
          
                  rel_sol_minimum = 1.0
              
              if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                  solved_yet = true
          
                  if verbose
                      println("Block: $n_block, - Solved using previous solution; residual norm: $sol_minimum")
                  end
              end
          
              total_iters = [0,0]
          
              SS_optimizer = levenberg_marquardt
              SS_optimizer = newton

              
              guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only

              sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
              # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
              guesses[1], 
              solver_parameters[1],
                                                  false,
                                                  false)




              for g in guesses
                  for p in solver_parameters
                      for ext in [true, false] # try first the system where values and parameters can vary, next try the system where only values can vary
                          if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol# || rel_sol_minimum > rtol
                              if solved_yet # continue end
      
                              sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                              # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                                  g, 
                                                                  p,
                                                                  ext,
                                                                  false)
                                                                  
                              if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                                  solved_yet = true
                              end
                          end
                      end
                  end
              end



              SS_solve_block = 𝓂.ss_solve_blocks_in_place[1];

              parameters_and_solved_vars = params_and_solved_vars

              guess_and_pars_solved_vars = inits

              n_block = 1

    solved_yet = false

    guess = guess_and_pars_solved_vars[1]

    sol_values = guess

    closest_parameters_and_solved_vars = sum(abs, guess_and_pars_solved_vars[2]) == Inf ? parameters_and_solved_vars : guess_and_pars_solved_vars[2]

    # res = ss_solve_blocks(parameters_and_solved_vars, guess)

    SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.func_buffer, guess, parameters_and_solved_vars, 0) # TODO: make the block a struct
    # TODO: do the function creation with Symbolics as this will solve the compilation bottleneck for large functions

    res = SS_solve_block.ss_problem.func_buffer

    sol_minimum  = ℒ.norm(res)

    total_iters = [0,0]

    SS_optimizer = levenberg_marquardt

    SS_optimizer = newton

    # if cold_start
        guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only

        g = guesses[1]
        # g = rand(5) .+ 1
        p = solver_parameters[1]
        ext = false

        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
        g, 
        p,
        ext,
        false)

        sol_minimum < tol.NSSS_acceptance_tol
        SS_solve_block.ss_problem.jac_buffer

        SS_solve_block.ss_problem.func_buffer

        for g in guesses
            for p in solver_parameters
                for ext in [true, false] # try first the system where values and parameters can vary, next try the system where only values can vary
                    if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol# || rel_sol_minimum > rtol
                        if solved_yet # continue end
                        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                            g, 
                                                            p,
                                                            ext,
                                                            false)
                        if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                            solved_yet = true
                        end
                    end
                end
            end
        end



              iters += (solution[2])[2]
              solution_error += (solution[2])[1]

              if solution_error > tol.NSSS_acceptance_tol
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2926 =#
                if verbose
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2926 =#
                    println("Failed after solving block with error $(solution_error)")
                end
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2926 =#
                scale = scale * 0.3 + solved_scale * 0.7
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2926 =#
                # # continue
            end
            sol = solution[1]
            alpha = sol[1]
            c = sol[2]
            delta = sol[3]
            k = sol[4]
            y = sol[5]
            NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
                        sol
                    else
                        ℱ.value.(sol)
                    end]
            NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
                        params_and_solved_vars
                    else
                        ℱ.value.(params_and_solved_vars)
                    end]
            ➕₂ = min(max(2.220446049250313e-16, c - 1), 0.9999999999999998)
            solution_error += abs(➕₂ - (c - 1))
            if solution_error > tol.NSSS_acceptance_tol
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                if verbose
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                    println("Failed for analytical aux variables with error $(solution_error)")
                end
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                scale = scale * 0.3 + solved_scale * 0.7
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                # # continue
            end
            ➕₄ = min(2 - eps(), max(eps(), 2.0➕₂))
            solution_error += +(abs(➕₄ - 2.0➕₂))
            if solution_error > tol.NSSS_acceptance_tol
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                if verbose
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                    println("Failed for analytical variables with error $(solution_error)")
                end
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                scale = scale * 0.3 + solved_scale * 0.7
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                # continue
            end
            c_norminvcdf = -1.4142135623731 * erfcinv(➕₄)
            c_normlogpdf = -0.5 * c ^ 2 - 0.918938533204673
            log_ZZ_avg = 0
            ZZ_avg_fut = 1
            ZZ_avg = 1
            eps_z = 0
            eps_zᴸ⁽⁻¹⁾ = 0
            eps_zᴸ⁽¹⁾ = 0
            if length(NSSS_solver_cache_tmp) == 0
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3641 =#
                NSSS_solver_cache_tmp = [copy(params_flt)]
            else
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3641 =#
                NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)]
            end
            if current_best > 1.0e-8 && (solution_error < tol.NSSS_acceptance_tol && scale == 1)
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3651 =#
                reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3766 =#
            if solution_error < tol.NSSS_acceptance_tol
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3768 =#
                solved_scale = scale
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3769 =#
                if scale == 1
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3773 =#
                    return ([A, A, A, Pi, R, ZZ_avg, ZZ_avg_fut, c, c_norminvcdf, c_normlogpdf, eps_z, eps_z, eps_z, k, log_ZZ_avg, y, z_delta, beta, Pibar, alpha, delta], (solution_error, iters))
                else
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3775 =#
                    reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                end
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3778 =#
                if scale > 0.95
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3779 =#
                    scale = 1
                else
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3782 =#
                    scale = scale * 0.4 + 0.6
                end
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3792 =#
        end
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3793 =#
        return ([0.0], (1, 0))