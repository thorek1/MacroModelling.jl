# Legacy LBFGS solver for conditional forecasts
# This file contains the original LBFGS-based implementation for backward compatibility

"""
    minimize_distance_to_conditions(X::Vector{S}, p)::S where S

Helper function for LBFGS optimization in conditional forecasts.
Computes the squared distance between conditions and model predictions.

# Arguments
- `X`: Vector of free shock values
- `p`: Tuple containing (Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, precision_factor)

# Returns
- Squared distance metric to be minimized
"""
function minimize_distance_to_conditions(X::Vector{S}, p)::S where S
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
                                    S₁, S₂, S₃, timings)

Find shocks that satisfy conditional forecast constraints using LBFGS optimizer.

Note: This is the legacy implementation. It reconstructs the state_update function
from the first-order solution matrix S₁.

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
"""
function find_shocks_conditional_forecast(::Val{:LBFGS},
                                         initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
                                         shocks::Vector{Float64},
                                         conditions::Vector{Float64},
                                         cond_var_idx::Vector{Int},
                                         free_shock_idx::Vector{Int},
                                         state_update::Function,
                                        #  pruning::Bool,
                                         S₁, S₂, S₃, timings)
                                         
    pruning = typeof(initial_state) <: Vector{Vector{Float64}}
    precision_factor = 1.0
    
    # Pack parameters for objective function
    p = (conditions, state_update, shocks, cond_var_idx, free_shock_idx, initial_state, pruning, precision_factor)
    
    # First attempt: LBFGS with line search
    res = @suppress begin 
        Optim.optimize(x -> minimize_distance_to_conditions(x, p), 
                      zeros(length(free_shock_idx)), 
                      Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                      Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                      autodiff = :forward) 
    end
    
    matched = Optim.minimum(res) < 1e-12
    
    # Second attempt: LBFGS without line search if first failed
    if !matched
        res = @suppress begin 
            Optim.optimize(x -> minimize_distance_to_conditions(x, p), 
                          zeros(length(free_shock_idx)), 
                          Optim.LBFGS(), 
                          Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                          autodiff = :forward) 
        end
        
        matched = Optim.minimum(res) < 1e-12
    end
    
    x = Optim.minimizer(res)
    
    return x, matched
end
