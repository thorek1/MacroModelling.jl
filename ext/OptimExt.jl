# OptimExt.jl - Extension for Optim.jl integration with MacroModelling.jl
# This extension provides an alternative LBFGS-based solver for conditional forecasts
# Note: Optim is already a required dependency for MacroModelling.jl (used in SS solver)
# so this extension is always loaded. It provides an alternative solver method.

module OptimExt

using MacroModelling
using MacroModelling: find_shocks_conditional_forecast
using Optim
using LineSearches

# LBFGS-based solver for conditional forecasts
# This is an alternative solver that uses Optim's LBFGS implementation
# It can be useful when the default Lagrange-Newton method fails to converge
function find_shocks_conditional_forecast(::Val{:LBFGS},
                                         state_update::Function,
                                         initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
                                         all_shocks::Vector{Float64},
                                         conditions::Vector{Float64},
                                         cond_var_idx::Vector{Int},
                                         free_shock_idx::Vector{Int},
                                         pruning::Bool;
                                         max_iter::Int = 1000,
                                         tol::Float64 = 1e-13)
    
    # Objective function: squared distance from conditions
    function objective(free_shocks::Vector{S}) where S
        shocks_copy = copy(all_shocks)
        shocks_copy[free_shock_idx] .= free_shocks
        new_state = state_update(initial_state, convert(typeof(free_shocks), shocks_copy))
        cond_vars = pruning ? sum(new_state) : new_state
        
        return sum(abs2, cond_vars[cond_var_idx] - conditions)
    end
    
    # Try with linesearch first
    res = try
        Optim.optimize(objective, 
                      zeros(length(free_shock_idx)), 
                      Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                      Optim.Options(f_abstol = eps(), g_tol= 1e-30, iterations = max_iter); 
                      autodiff = :forward)
    catch
        nothing
    end
    
    matched = !isnothing(res) && Optim.minimum(res) < 1e-12
    
    # Try without linesearch if first attempt failed
    if !matched
        res = try
            Optim.optimize(objective, 
                          zeros(length(free_shock_idx)), 
                          Optim.LBFGS(), 
                          Optim.Options(f_abstol = eps(), g_tol= 1e-30, iterations = max_iter); 
                          autodiff = :forward)
        catch
            nothing
        end
        
        matched = !isnothing(res) && Optim.minimum(res) < 1e-12
    end
    
    if matched
        return Optim.minimizer(res), true
    else
        return zeros(length(free_shock_idx)), false
    end
end

end # module OptimExt
