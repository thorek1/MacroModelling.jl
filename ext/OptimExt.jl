module OptimExt

import MacroModelling
import Optim

# Helper function for LBFGS optimization objective
function _minimize_distance_to_conditions(X::Vector{S}, p)::S where S
    Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, precision_factor = p

    S_shocks = convert(Vector{S}, Shocks)
    S_shocks[Free_shock_idx] .= X

    new_State = State_update(State, S_shocks)

    cond_vars = Pruning ? sum(new_State) : new_State

    return precision_factor * sum(abs2, Conditions - cond_vars[Cond_var_idx])
end

"""
    find_shocks_conditional_forecast(::Val{:LBFGS}, 
                                    initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
                                    shocks::Vector{Float64},
                                    conditions::Vector{Float64},
                                    cond_var_idx::Vector{Int},
                                    free_shock_idx::Vector{Int},
                                    pruning::Bool,
                                    S₁, S₂, S₃, timings; verbose::Bool = false)

Find shocks that satisfy conditional forecast constraints using LBFGS optimizer.

Note: This is the Optim-based implementation. It requires the Optim.jl extension.

# Arguments
- `initial_state`: Initial state vector (or vector of vectors for pruning)
- `shocks`: Initial shock vector
- `conditions`: Target values for conditioned variables
- `cond_var_idx`: Indices of conditioned variables
- `free_shock_idx`: Indices of free shocks to be determined
- `pruning`: Whether pruning is used
- `S₁`: First-order solution matrix
- `S₂`, `S₃`: Higher-order perturbation matrices (not used in LBFGS, for compatibility only)
- `timings`: Model timings structure

# Returns
- `x`: Vector of optimal shock values
- `matched`: Boolean indicating if optimization converged successfully
- Set `verbose = true` to show optimizer traces and fallback attempts
"""
function MacroModelling.find_shocks_conditional_forecast(::Val{:LBFGS},
                                         initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
                                         shocks::Vector{Float64},
                                         conditions::Vector{Float64},
                                         cond_var_idx::Vector{Int},
                                         free_shock_idx::Vector{Int},
                                         state_update::Function,
                                        #  pruning::Bool,
                                         S₁, S₂, S₃, timings; verbose::Bool = false)
                                         
    pruning = typeof(initial_state) <: Vector{Vector{Float64}}
    precision_factor = 1.0
    
    # Pack parameters for objective function
    p = (conditions, state_update, shocks, cond_var_idx, free_shock_idx, initial_state, pruning, precision_factor)

    options = Optim.Options(f_abstol = eps(), g_tol = 1e-30, show_trace = verbose, store_trace = verbose)
    verbose && @info "LBFGS line-search attempt" free_shocks = length(free_shock_idx) conditioned = length(cond_var_idx)
    
    # First attempt: LBFGS with line search
    res = Optim.optimize(x -> _minimize_distance_to_conditions(x, p), 
                      zeros(length(free_shock_idx)), 
                      Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)), 
                      options; 
                      autodiff = :forward) 
    
    matched = Optim.minimum(res) < 1e-12
    
    # Second attempt: LBFGS without line search if first failed
    if !matched
        verbose && @info "LBFGS retry without line search" objective = Optim.minimum(res)
        res = Optim.optimize(x -> _minimize_distance_to_conditions(x, p), 
                          zeros(length(free_shock_idx)), 
                          Optim.LBFGS(), 
                          options; 
                          autodiff = :forward) 
        
        matched = Optim.minimum(res) < 1e-12
    end
    
    verbose && @info "LBFGS finished" objective = Optim.minimum(res) matched = matched iterations = Optim.iterations(res)
    x = Optim.minimizer(res)
    
    return x, matched
end

"""
    find_SS_solver_parameters!(::Val{:SAMIN}, 𝓂::ℳ; maxtime::Int = 120, maxiter::Int = 2500000, tol::Tolerances = Tolerances(), verbosity = 0)

Find optimal steady state solver parameters using Optim's SAMIN algorithm.

This function optimizes solver parameters to minimize runtime while maintaining solver accuracy.
It uses Simulated Annealing with Metropolis acceptance (SAMIN) from Optim.jl.

# Arguments
- `𝓂`: Model structure
- `maxtime`: Maximum time in seconds for optimization
- `maxiter`: Maximum number of iterations
- `tol`: Tolerance structure
- `verbosity`: Verbosity level for output
"""
function MacroModelling.find_SS_solver_parameters!(::Val{:SAMIN}, 𝓂::MacroModelling.ℳ; 
                                                    maxtime::Int = 120, 
                                                    maxiter::Int = 2500000, 
                                                    tol::MacroModelling.Tolerances = MacroModelling.Tolerances(), 
                                                    verbosity = 0)
    pars = rand(20) .+ 1
    pars[20] -= 1

    lbs = fill(eps(), length(pars))
    lbs[20] = -20

    ubs = fill(100.0, length(pars))
    
    # Use Optim SAMIN algorithm
    sol = Optim.optimize(x -> MacroModelling.calculate_SS_solver_runtime_and_loglikelihood(x, 𝓂, tol = tol), 
                        lbs, ubs, pars, 
                        Optim.SAMIN(verbosity = verbosity, nt = 5, ns = 5), 
                        Optim.Options(time_limit = maxtime, iterations = maxiter))::Optim.MultivariateOptimizationResults

    pars = Optim.minimizer(sol)::Vector{Float64}

    par_inputs = MacroModelling.solver_parameters(pars..., 1, 0.0, 2)

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, tol, false, true, [par_inputs])

    if solution_error < tol.NSSS_acceptance_tol
        push!(𝓂.solver_parameters, par_inputs)
        return true
    else 
        return false
    end
end

end  # module OptimExt
