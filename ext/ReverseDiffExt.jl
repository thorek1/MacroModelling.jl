module ReverseDiffExt

using MacroModelling
using ReverseDiff
using ReverseDiff: TrackedArray, TrackedReal, TrackedMatrix, TrackedVector
using ReverseDiff: value, deriv, track, tape, istracked, increment_deriv!, unseed!, pull_value!
using ReverseDiff: record!, SpecialInstruction, InstructionTape, special_forward_exec!, special_reverse_exec!

import MacroModelling: solve_sylvester_equation, solve_lyapunov_equation
import MacroModelling: calculate_jacobian, calculate_hessian, calculate_third_order_derivatives
import MacroModelling: get_NSSS_and_parameters
import MacroModelling: timings, ℳ, CalculationOptions, sylvester_caches, Sylvester_caches, higher_order_caches, Higher_order_caches

import LinearAlgebra
const ℒ = LinearAlgebra

#= 
ReverseDiff Extension for MacroModelling.jl

This extension provides ReverseDiff compatibility for the key functions used in 
DSGE model estimation by implementing custom SpecialInstruction types that use
the analytical pullbacks already defined in the package.

The approach:
1. Define wrapper structs to identify each function type
2. Implement TrackedArray methods that record SpecialInstructions
3. Implement special_forward_exec! for the forward pass
4. Implement special_reverse_exec! using the analytical pullbacks from rrules
=#

# =============================================================================
# Function type markers for SpecialInstruction dispatch
# We use singleton instances for dispatch
# =============================================================================

struct LyapunovSolver end
struct SylvesterSolver end
struct JacobianCalculation end
struct HessianCalculation end 
struct ThirdOrderCalculation end
struct NSSSCalculation end
struct NSSSCalculationFailed end  # For failed solutions - returns zero gradients
struct KalmanFilter end

const LYAPUNOV_SOLVER = LyapunovSolver()
const SYLVESTER_SOLVER = SylvesterSolver()
const JACOBIAN_CALCULATION = JacobianCalculation()
const HESSIAN_CALCULATION = HessianCalculation()
const THIRD_ORDER_CALCULATION = ThirdOrderCalculation()
const NSSS_CALCULATION = NSSSCalculation()
const NSSS_CALCULATION_FAILED = NSSSCalculationFailed()
const KALMAN_FILTER = KalmanFilter()

# =============================================================================
# Lyapunov equation solver
# Solves: A * X * A' + C = X
# Uses analytical pullback from rrule in algorithms/lyapunov.jl
# =============================================================================

# TrackedArray A, untracked C
function solve_lyapunov_equation(A::TrackedArray{V,D,2},
                                  C::AbstractMatrix{<:Real};
                                  lyapunov_algorithm::Symbol = :doubling,
                                  tol::AbstractFloat = 1e-14,
                                  acceptance_tol::AbstractFloat = 1e-12,
                                  verbose::Bool = false) where {V<:Real, D<:Real}
    
    tp = tape(A)
    A_val = value(A)
    
    # Compute primal
    P, solved = solve_lyapunov_equation(A_val, C,
                                        lyapunov_algorithm = lyapunov_algorithm,
                                        tol = tol,
                                        acceptance_tol = acceptance_tol,
                                        verbose = verbose)
    
    # Track output
    P_tracked = track(copy(P), D, tp)
    
    # Record instruction with cache for pullback
    cache = (A_val, C, P, lyapunov_algorithm, tol, verbose)
    record!(tp, SpecialInstruction, LYAPUNOV_SOLVER, (A,), P_tracked, cache)
    
    return (P_tracked, solved)
end

# TrackedArray A and C
function solve_lyapunov_equation(A::TrackedArray{V1,D1,2},
                                  C::TrackedArray{V2,D2,2};
                                  lyapunov_algorithm::Symbol = :doubling,
                                  tol::AbstractFloat = 1e-14,
                                  acceptance_tol::AbstractFloat = 1e-12,
                                  verbose::Bool = false) where {V1<:Real, D1<:Real, V2<:Real, D2<:Real}
    
    tp = tape(A)
    A_val = value(A)
    C_val = value(C)
    
    # Compute primal
    P, solved = solve_lyapunov_equation(A_val, C_val,
                                        lyapunov_algorithm = lyapunov_algorithm,
                                        tol = tol,
                                        acceptance_tol = acceptance_tol,
                                        verbose = verbose)
    
    # Track output
    P_tracked = track(copy(P), D1, tp)
    
    # Record instruction with cache for pullback
    cache = (A_val, C_val, P, lyapunov_algorithm, tol, verbose)
    record!(tp, SpecialInstruction, LYAPUNOV_SOLVER, (A, C), P_tracked, cache)
    
    return (P_tracked, solved)
end

# Untracked A, TrackedArray C
function solve_lyapunov_equation(A::AbstractMatrix{<:Real},
                                  C::TrackedArray{V,D,2};
                                  lyapunov_algorithm::Symbol = :doubling,
                                  tol::AbstractFloat = 1e-14,
                                  acceptance_tol::AbstractFloat = 1e-12,
                                  verbose::Bool = false) where {V<:Real, D<:Real}
    
    tp = tape(C)
    C_val = value(C)
    
    # Compute primal
    P, solved = solve_lyapunov_equation(A, C_val,
                                        lyapunov_algorithm = lyapunov_algorithm,
                                        tol = tol,
                                        acceptance_tol = acceptance_tol,
                                        verbose = verbose)
    
    # Track output
    P_tracked = track(copy(P), D, tp)
    
    # Record instruction with cache for pullback
    cache = (A, C_val, P, lyapunov_algorithm, tol, verbose)
    record!(tp, SpecialInstruction, LYAPUNOV_SOLVER, (C,), P_tracked, cache)
    
    return (P_tracked, solved)
end

# Forward execution - no-op since we compute in the function call
@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{LyapunovSolver})
    # Value already computed during recording
    return nothing
end

# Reverse execution - analytical pullback
@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{LyapunovSolver})
    inputs = instruction.input
    output = instruction.output
    A_val, C_val, P, lyapunov_algorithm, tol, verbose = instruction.cache
    
    ∂P = deriv(output)
    
    if ℒ.norm(∂P) >= tol
        # Analytical pullback from rrule: ∂C = solve_lyapunov(A', ∂P)
        ∂C, _ = solve_lyapunov_equation(A_val', ∂P,
                                        lyapunov_algorithm = lyapunov_algorithm,
                                        tol = tol,
                                        verbose = verbose)
        
        # ∂A = ∂C * A * P' + ∂C' * A * P
        ∂A = ∂C * A_val * P' + ∂C' * A_val * P
        
        # Accumulate gradients based on which inputs are tracked
        if length(inputs) == 1
            # Either A or C is tracked
            input = inputs[1]
            if size(value(input)) == size(∂A)
                increment_deriv!(input, ∂A)
            else
                increment_deriv!(input, ∂C)
            end
        else
            # Both A and C are tracked
            A_tracked, C_tracked = inputs
            increment_deriv!(A_tracked, ∂A)
            increment_deriv!(C_tracked, ∂C)
        end
    end
    
    unseed!(output)
    return nothing
end

# =============================================================================
# Sylvester equation solver
# Solves: A * X * B + C = X
# Uses analytical pullback from rrule in algorithms/sylvester.jl
# =============================================================================

# All combinations of tracked/untracked for A, B, C
# A tracked, B and C untracked
function solve_sylvester_equation(A::TrackedArray{V,D,2},
                                   B::AbstractMatrix{<:Real},
                                   C::AbstractMatrix{<:Real};
                                   initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                   sylvester_algorithm::Symbol = :doubling,
                                   acceptance_tol::AbstractFloat = 1e-10,
                                   tol::AbstractFloat = 1e-14,
                                   𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                   verbose::Bool = false) where {V<:Real, D<:Real}
    
    tp = tape(A)
    A_val = value(A)
    
    P, solved = solve_sylvester_equation(A_val, B, C,
                                        sylvester_algorithm = sylvester_algorithm,
                                        tol = tol,
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose,
                                        initial_guess = initial_guess)
    
    P_tracked = track(copy(P), D, tp)
    cache = (A_val, B, C, P, sylvester_algorithm, tol, 𝕊ℂ, verbose, :A)
    record!(tp, SpecialInstruction, SYLVESTER_SOLVER, (A,), P_tracked, cache)
    
    return (P_tracked, solved)
end

# A and C tracked, B untracked  
function solve_sylvester_equation(A::TrackedArray{V1,D1,2},
                                   B::AbstractMatrix{<:Real},
                                   C::TrackedArray{V2,D2,2};
                                   initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                   sylvester_algorithm::Symbol = :doubling,
                                   acceptance_tol::AbstractFloat = 1e-10,
                                   tol::AbstractFloat = 1e-14,
                                   𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                   verbose::Bool = false) where {V1<:Real, D1<:Real, V2<:Real, D2<:Real}
    
    tp = tape(A)
    A_val = value(A)
    C_val = value(C)
    
    P, solved = solve_sylvester_equation(A_val, B, C_val,
                                        sylvester_algorithm = sylvester_algorithm,
                                        tol = tol,
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose,
                                        initial_guess = initial_guess)
    
    P_tracked = track(copy(P), D1, tp)
    cache = (A_val, B, C_val, P, sylvester_algorithm, tol, 𝕊ℂ, verbose, :AC)
    record!(tp, SpecialInstruction, SYLVESTER_SOLVER, (A, C), P_tracked, cache)
    
    return (P_tracked, solved)
end

# All three tracked
function solve_sylvester_equation(A::TrackedArray{V1,D1,2},
                                   B::TrackedArray{V2,D2,2},
                                   C::TrackedArray{V3,D3,2};
                                   initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                   sylvester_algorithm::Symbol = :doubling,
                                   acceptance_tol::AbstractFloat = 1e-10,
                                   tol::AbstractFloat = 1e-14,
                                   𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                   verbose::Bool = false) where {V1<:Real, D1<:Real, V2<:Real, D2<:Real, V3<:Real, D3<:Real}
    
    tp = tape(A)
    A_val = value(A)
    B_val = value(B)
    C_val = value(C)
    
    P, solved = solve_sylvester_equation(A_val, B_val, C_val,
                                        sylvester_algorithm = sylvester_algorithm,
                                        tol = tol,
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose,
                                        initial_guess = initial_guess)
    
    P_tracked = track(copy(P), D1, tp)
    cache = (A_val, B_val, C_val, P, sylvester_algorithm, tol, 𝕊ℂ, verbose, :ABC)
    record!(tp, SpecialInstruction, SYLVESTER_SOLVER, (A, B, C), P_tracked, cache)
    
    return (P_tracked, solved)
end

# A and B tracked, C untracked
function solve_sylvester_equation(A::TrackedArray{V1,D1,2},
                                   B::TrackedArray{V2,D2,2},
                                   C::AbstractMatrix{<:Real};
                                   initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                   sylvester_algorithm::Symbol = :doubling,
                                   acceptance_tol::AbstractFloat = 1e-10,
                                   tol::AbstractFloat = 1e-14,
                                   𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                   verbose::Bool = false) where {V1<:Real, D1<:Real, V2<:Real, D2<:Real}
    
    tp = tape(A)
    A_val = value(A)
    B_val = value(B)
    
    P, solved = solve_sylvester_equation(A_val, B_val, C,
                                        sylvester_algorithm = sylvester_algorithm,
                                        tol = tol,
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose,
                                        initial_guess = initial_guess)
    
    P_tracked = track(copy(P), D1, tp)
    cache = (A_val, B_val, C, P, sylvester_algorithm, tol, 𝕊ℂ, verbose, :AB)
    record!(tp, SpecialInstruction, SYLVESTER_SOLVER, (A, B), P_tracked, cache)
    
    return (P_tracked, solved)
end

# B and C tracked, A untracked
function solve_sylvester_equation(A::AbstractMatrix{<:Real},
                                   B::TrackedArray{V1,D1,2},
                                   C::TrackedArray{V2,D2,2};
                                   initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                   sylvester_algorithm::Symbol = :doubling,
                                   acceptance_tol::AbstractFloat = 1e-10,
                                   tol::AbstractFloat = 1e-14,
                                   𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                   verbose::Bool = false) where {V1<:Real, D1<:Real, V2<:Real, D2<:Real}
    
    tp = tape(B)
    B_val = value(B)
    C_val = value(C)
    
    P, solved = solve_sylvester_equation(A, B_val, C_val,
                                        sylvester_algorithm = sylvester_algorithm,
                                        tol = tol,
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose,
                                        initial_guess = initial_guess)
    
    P_tracked = track(copy(P), D1, tp)
    cache = (A, B_val, C_val, P, sylvester_algorithm, tol, 𝕊ℂ, verbose, :BC)
    record!(tp, SpecialInstruction, SYLVESTER_SOLVER, (B, C), P_tracked, cache)
    
    return (P_tracked, solved)
end

# B tracked only
function solve_sylvester_equation(A::AbstractMatrix{<:Real},
                                   B::TrackedArray{V,D,2},
                                   C::AbstractMatrix{<:Real};
                                   initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                   sylvester_algorithm::Symbol = :doubling,
                                   acceptance_tol::AbstractFloat = 1e-10,
                                   tol::AbstractFloat = 1e-14,
                                   𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                   verbose::Bool = false) where {V<:Real, D<:Real}
    
    tp = tape(B)
    B_val = value(B)
    
    P, solved = solve_sylvester_equation(A, B_val, C,
                                        sylvester_algorithm = sylvester_algorithm,
                                        tol = tol,
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose,
                                        initial_guess = initial_guess)
    
    P_tracked = track(copy(P), D, tp)
    cache = (A, B_val, C, P, sylvester_algorithm, tol, 𝕊ℂ, verbose, :B)
    record!(tp, SpecialInstruction, SYLVESTER_SOLVER, (B,), P_tracked, cache)
    
    return (P_tracked, solved)
end

# C tracked only
function solve_sylvester_equation(A::AbstractMatrix{<:Real},
                                   B::AbstractMatrix{<:Real},
                                   C::TrackedArray{V,D,2};
                                   initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                   sylvester_algorithm::Symbol = :doubling,
                                   acceptance_tol::AbstractFloat = 1e-10,
                                   tol::AbstractFloat = 1e-14,
                                   𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                   verbose::Bool = false) where {V<:Real, D<:Real}
    
    tp = tape(C)
    C_val = value(C)
    
    P, solved = solve_sylvester_equation(A, B, C_val,
                                        sylvester_algorithm = sylvester_algorithm,
                                        tol = tol,
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose,
                                        initial_guess = initial_guess)
    
    P_tracked = track(copy(P), D, tp)
    cache = (A, B, C_val, P, sylvester_algorithm, tol, 𝕊ℂ, verbose, :C)
    record!(tp, SpecialInstruction, SYLVESTER_SOLVER, (C,), P_tracked, cache)
    
    return (P_tracked, solved)
end

# Forward execution - no-op
@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{SylvesterSolver})
    return nothing
end

# Reverse execution - analytical pullback from sylvester rrule
@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{SylvesterSolver})
    inputs = instruction.input
    output = instruction.output
    A_val, B_val, C_val, P, sylvester_algorithm, tol, 𝕊ℂ, verbose, tracked_flag = instruction.cache
    
    ∂P = deriv(output)
    
    if ℒ.norm(∂P) >= tol
        # Analytical pullback: ∂C = solve_sylvester(A', B', ∂P)
        ∂C, _ = solve_sylvester_equation(A_val', B_val', ∂P,
                                        sylvester_algorithm = sylvester_algorithm,
                                        tol = tol,
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose)
        
        # ∂A = ∂C * B' * P'
        # ∂B = P' * A' * ∂C
        ∂A = ∂C * B_val' * P'
        ∂B = P' * A_val' * ∂C
        
        # Accumulate gradients based on which inputs are tracked
        if tracked_flag == :A
            increment_deriv!(inputs[1], ∂A)
        elseif tracked_flag == :B
            increment_deriv!(inputs[1], ∂B)
        elseif tracked_flag == :C
            increment_deriv!(inputs[1], ∂C)
        elseif tracked_flag == :AB
            increment_deriv!(inputs[1], ∂A)
            increment_deriv!(inputs[2], ∂B)
        elseif tracked_flag == :AC
            increment_deriv!(inputs[1], ∂A)
            increment_deriv!(inputs[2], ∂C)
        elseif tracked_flag == :BC
            increment_deriv!(inputs[1], ∂B)
            increment_deriv!(inputs[2], ∂C)
        elseif tracked_flag == :ABC
            increment_deriv!(inputs[1], ∂A)
            increment_deriv!(inputs[2], ∂B)
            increment_deriv!(inputs[3], ∂C)
        end
    end
    
    unseed!(output)
    return nothing
end

# =============================================================================
# Jacobian, Hessian, Third Order Derivatives - TrackedVector parameters
# Uses analytical pullbacks from rrules in MacroModelling.jl
# =============================================================================

function calculate_jacobian(parameters::TrackedArray{V,D,1},
                            SS_and_pars::AbstractVector,
                            𝓂::ℳ) where {V<:Real, D<:Real}
    
    tp = tape(parameters)
    params_val = value(parameters)
    SS_val = SS_and_pars isa TrackedArray ? value(SS_and_pars) : SS_and_pars
    
    jacobian = calculate_jacobian(params_val, SS_val, 𝓂)
    
    jacobian_tracked = track(copy(Array(jacobian)), D, tp)
    cache = (params_val, SS_val, 𝓂)
    record!(tp, SpecialInstruction, JACOBIAN_CALCULATION, (parameters,), jacobian_tracked, cache)
    
    return jacobian_tracked
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{JacobianCalculation})
    return nothing
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{JacobianCalculation})
    parameters = instruction.input[1]
    output = instruction.output
    params_val, SS_val, 𝓂 = instruction.cache
    
    ∂∇₁ = deriv(output)
    
    # Use precomputed Jacobian of Jacobian from model
    𝓂.jacobian_parameters[2](𝓂.jacobian_parameters[1], params_val, SS_val)
    ∂parameters = 𝓂.jacobian_parameters[1]' * vec(∂∇₁)
    
    increment_deriv!(parameters, ∂parameters)
    unseed!(output)
    return nothing
end

function calculate_hessian(parameters::TrackedArray{V,D,1},
                           SS_and_pars::AbstractVector,
                           𝓂::ℳ) where {V<:Real, D<:Real}
    
    tp = tape(parameters)
    params_val = value(parameters)
    SS_val = SS_and_pars isa TrackedArray ? value(SS_and_pars) : SS_and_pars
    
    hessian = calculate_hessian(params_val, SS_val, 𝓂)
    
    hessian_tracked = track(copy(Array(hessian)), D, tp)
    cache = (params_val, SS_val, 𝓂)
    record!(tp, SpecialInstruction, HESSIAN_CALCULATION, (parameters,), hessian_tracked, cache)
    
    return hessian_tracked
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{HessianCalculation})
    return nothing
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{HessianCalculation})
    parameters = instruction.input[1]
    output = instruction.output
    params_val, SS_val, 𝓂 = instruction.cache
    
    ∂∇₂ = deriv(output)
    
    𝓂.hessian_parameters[2](𝓂.hessian_parameters[1], params_val, SS_val)
    ∂parameters = 𝓂.hessian_parameters[1]' * vec(∂∇₂)
    
    increment_deriv!(parameters, ∂parameters)
    unseed!(output)
    return nothing
end

function calculate_third_order_derivatives(parameters::TrackedArray{V,D,1},
                                            SS_and_pars::AbstractVector,
                                            𝓂::ℳ) where {V<:Real, D<:Real}
    
    tp = tape(parameters)
    params_val = value(parameters)
    SS_val = SS_and_pars isa TrackedArray ? value(SS_and_pars) : SS_and_pars
    
    third_order = calculate_third_order_derivatives(params_val, SS_val, 𝓂)
    
    third_order_tracked = track(copy(Array(third_order)), D, tp)
    cache = (params_val, SS_val, 𝓂)
    record!(tp, SpecialInstruction, THIRD_ORDER_CALCULATION, (parameters,), third_order_tracked, cache)
    
    return third_order_tracked
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{ThirdOrderCalculation})
    return nothing
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{ThirdOrderCalculation})
    parameters = instruction.input[1]
    output = instruction.output
    params_val, SS_val, 𝓂 = instruction.cache
    
    ∂∇₃ = deriv(output)
    
    𝓂.third_order_derivatives_parameters[2](𝓂.third_order_derivatives_parameters[1], params_val, SS_val)
    ∂parameters = 𝓂.third_order_derivatives_parameters[1]' * vec(∂∇₃)
    
    increment_deriv!(parameters, ∂parameters)
    unseed!(output)
    return nothing
end

# =============================================================================
# NSSS and parameters
# Uses analytical pullback from rrule in MacroModelling.jl
# =============================================================================

function get_NSSS_and_parameters(𝓂::ℳ,
                                  parameter_values::TrackedArray{V,D,1};
                                  opts::CalculationOptions = MacroModelling.merge_calculation_options()) where {V<:Real, D<:Real}
    
    tp = tape(parameter_values)
    params_val = value(parameter_values)
    
    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(params_val, 𝓂, opts.tol, opts.verbose, false, 𝓂.solver_parameters)
    
    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        # Record a failed instruction that returns zero gradients
        SS_tracked = track(copy(SS_and_pars), D, tp)
        cache = (length(params_val),)
        record!(tp, SpecialInstruction, NSSS_CALCULATION_FAILED, (parameter_values,), SS_tracked, cache)
        return (SS_tracked, (solution_error, iters))
    end
    
    # Compute the JVP matrix for the pullback (from the rrule)
    SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
    SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
    SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
    
    unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,MacroModelling.get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))
    
    C = SS_and_pars[indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag)]
    
    if eltype(𝓂.∂SS_equations_∂parameters[1]) != eltype(params_val)
        jac_buffer = similar(𝓂.∂SS_equations_∂parameters[1], eltype(params_val))
        jac_buffer .= 0
    else
        jac_buffer = 𝓂.∂SS_equations_∂parameters[1]
    end
    𝓂.∂SS_equations_∂parameters[2](jac_buffer, params_val, C)
    ∂SS_equations_∂parameters = jac_buffer
    
    if eltype(𝓂.∂SS_equations_∂SS_and_pars[1]) != eltype(SS_and_pars)
        jac_buffer2 = similar(𝓂.∂SS_equations_∂SS_and_pars[1], eltype(SS_and_pars))
        jac_buffer2 .= 0
    else
        jac_buffer2 = 𝓂.∂SS_equations_∂SS_and_pars[1]
    end
    𝓂.∂SS_equations_∂SS_and_pars[2](jac_buffer2, params_val, C)
    ∂SS_equations_∂SS_and_pars = jac_buffer2
    
    ∂SS_equations_∂SS_and_pars_lu = MacroModelling.RF.lu(∂SS_equations_∂SS_and_pars, check = false)
    
    if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
        # Record a failed instruction that returns zero gradients
        SS_tracked = track(copy(SS_and_pars), D, tp)
        cache = (length(params_val),)
        record!(tp, SpecialInstruction, NSSS_CALCULATION_FAILED, (parameter_values,), SS_tracked, cache)
        return (SS_tracked, (10.0, iters))
    end
    
    JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)
    
    jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))
    
    for (i,v) in enumerate(SS_and_pars_names)
        if v in unknowns
            jvp[i,:] = JVP[indexin([v], unknowns),:]
        end
    end
    
    SS_tracked = track(copy(SS_and_pars), D, tp)
    cache = (jvp,)
    record!(tp, SpecialInstruction, NSSS_CALCULATION, (parameter_values,), SS_tracked, cache)
    
    return (SS_tracked, (solution_error, iters))
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{NSSSCalculation})
    return nothing
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{NSSSCalculation})
    parameter_values = instruction.input[1]
    output = instruction.output
    jvp, = instruction.cache
    
    ∂SS = deriv(output)
    ∂parameters = jvp' * ∂SS
    
    increment_deriv!(parameter_values, ∂parameters)
    unseed!(output)
    return nothing
end

# Failed NSSS calculation - returns zero gradients
@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{NSSSCalculationFailed})
    return nothing
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{NSSSCalculationFailed})
    parameter_values = instruction.input[1]
    output = instruction.output
    n_params, = instruction.cache
    
    # Return zero gradients for failed solutions
    increment_deriv!(parameter_values, zeros(n_params))
    unseed!(output)
    return nothing
end

end # module

