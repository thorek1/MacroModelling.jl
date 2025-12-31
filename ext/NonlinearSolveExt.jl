module NonlinearSolveExt

import NonlinearSolve
import MacroModelling: function_and_jacobian, solver_parameters, Tolerances
import LinearAlgebra as ℒ
using DispatchDoctor
import DocStringExtensions: SIGNATURES

import MacroModelling: nonlinearsolve_solver

@stable default_mode = "disable" begin

"""
$(SIGNATURES)
Wrapper function to use NonlinearSolve.jl solvers for finding the non-stochastic steady state (NSSS).

This function follows the same interface as the built-in `levenberg_marquardt` and `newton` solvers.
It uses NonlinearSolve.jl's solvers with user-specified or default algorithms.

# Arguments
- `fnj::function_and_jacobian`: Struct containing the residual function and Jacobian function.
- `initial_guess::Vector{T}`: Initial guess for the solution.
- `parameters_and_solved_vars::Vector{T}`: Parameters and already solved variables.
- `lower_bounds::Vector{T}`: Lower bounds for the variables.
- `upper_bounds::Vector{T}`: Upper bounds for the variables.
- `parameters::solver_parameters`: Solver parameters (transformation_level is used).

# Keyword Arguments
- `tol::Tolerances`: Tolerance settings for the solver.
- `algorithm`: NonlinearSolve.jl algorithm to use. Default is `nothing` which lets NonlinearSolve choose automatically.

# Returns
- `Tuple{Vector{T}, Tuple{Int, Int, T, T}}`: Solution vector and tuple of (grad_iterations, func_iterations, relative_tolerance, residual_norm).

# Examples
```julia
using MacroModelling
using NonlinearSolve

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# The nonlinearsolve_solver function is now available and can be used
# as an alternative steady state solver through the block_solver infrastructure.
```
"""
function nonlinearsolve_solver(
    fnj::function_and_jacobian,
    initial_guess::Vector{T},
    parameters_and_solved_vars::Vector{T},
    lower_bounds::Vector{T},
    upper_bounds::Vector{T},
    parameters::solver_parameters;
    tol::Tolerances = Tolerances(),
    algorithm = nothing
)::Tuple{Vector{T}, Tuple{Int, Int, T, T}} where {T <: AbstractFloat}

    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol
    
    transformation_level = parameters.transformation_level
    
    # Transform initial guess and bounds
    u_bounds = copy(upper_bounds)
    l_bounds = copy(lower_bounds)
    current_guess = copy(initial_guess)
    
    for _ in 1:transformation_level
        u_bounds .= asinh.(u_bounds)
        l_bounds .= asinh.(l_bounds)
        current_guess .= asinh.(current_guess)
    end
    
    # Clamp initial guess to bounds
    current_guess .= max.(l_bounds, min.(u_bounds, current_guess))
    
    # Preallocate buffers for transformations
    u_untransformed_buffer = similar(current_guess)
    factor_buffer = similar(current_guess)
    
    # Define the residual function for NonlinearSolve
    # NonlinearSolve expects f(res, u, p) for in-place form
    function residual!(res, u, p)
        # Undo transformation
        copyto!(u_untransformed_buffer, u)
        for _ in 1:transformation_level
            u_untransformed_buffer .= sinh.(u_untransformed_buffer)
        end
        
        # Evaluate the residual function
        fnj.func(res, u_untransformed_buffer, p)
        return nothing
    end
    
    # Define the Jacobian function for NonlinearSolve  
    # NonlinearSolve expects jac(J, u, p) for in-place form
    function jacobian!(J, u, p)
        # Undo transformation
        copyto!(u_untransformed_buffer, u)
        fill!(factor_buffer, one(T))
        
        for _ in 1:transformation_level
            factor_buffer .*= cosh.(u_untransformed_buffer)
            u_untransformed_buffer .= sinh.(u_untransformed_buffer)
        end
        
        # Evaluate the Jacobian
        fnj.jac(J, u_untransformed_buffer, p)
        
        # Scale columns by the transformation factor (chain rule)
        if transformation_level > 0
            for j in 1:size(J, 2)
                @inbounds for i in 1:size(J, 1)
                    J[i, j] *= factor_buffer[j]
                end
            end
        end
        return nothing
    end
    
    # Create NonlinearFunction with in-place Jacobian
    nlf = NonlinearSolve.NonlinearFunction{true}(residual!; jac = jacobian!)
    
    # Create NonlinearProblem
    prob = NonlinearSolve.NonlinearProblem(nlf, current_guess, parameters_and_solved_vars)
    
    # Solve with specified or default algorithm
    # Note: maxiters = 250 matches the hardcoded iteration limit in the built-in 
    # levenberg_marquardt and newton solvers
    sol = if isnothing(algorithm)
        NonlinearSolve.solve(prob; abstol = ftol, reltol = rel_xtol, maxiters = 250)
    else
        NonlinearSolve.solve(prob, algorithm; abstol = ftol, reltol = rel_xtol, maxiters = 250)
    end
    
    # Extract solution
    solution = sol.u
    
    # Undo transformation for the solution
    best_current_guess = copy(solution)
    for _ in 1:transformation_level
        best_current_guess .= sinh.(best_current_guess)
    end
    
    # Clamp to original bounds
    best_current_guess .= max.(lower_bounds, min.(upper_bounds, best_current_guess))
    
    # Compute final residual
    fnj.func(fnj.func_buffer, best_current_guess, parameters_and_solved_vars)
    largest_residual = ℒ.norm(fnj.func_buffer)
    
    # Estimate relative change
    # Note: NonlinearSolve.jl doesn't expose per-iteration step sizes in the same way as the
    # built-in solvers. For consistent interface, we return 0.0 for successful solves
    # (indicating convergence) and 1.0 for failures. The residual norm in info[4] provides 
    # the primary convergence measure.
    largest_relative_step = NonlinearSolve.SciMLBase.successful_retcode(sol.retcode) ? T(0.0) : T(1.0)
    
    # Get iteration count if available from solver statistics
    nsteps = 0
    nf = 0
    if hasproperty(sol, :stats) && !isnothing(sol.stats)
        stats = sol.stats
        nsteps = hasproperty(stats, :nsteps) ? stats.nsteps : 0
        nf = hasproperty(stats, :nf) ? stats.nf : 0
    end
    
    return best_current_guess, (nsteps, nf, largest_relative_step, largest_residual)
end

end # @stable

end # module
