module MacroModelling


import DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF, TYPEDSIGNATURES, TYPEDFIELDS
# import StatsFuns: normcdf
import ThreadedSparseArrays
using PrecompileTools
import SpecialFunctions: erfcinv, erfc
import SpecialFunctions
import SymPyPythonCall as SPyPyC
import Symbolics
import Accessors
# import NaNMath
# import Memoization: @memoize
# import LRUCache: LRU

# for find shocks
# import JuMP
# import MadNLP
# import Ipopt
# import AbstractDifferentiation as 𝒜
import DifferentiationInterface as 𝒟
import ForwardDiff as ℱ
backend = 𝒟.AutoForwardDiff()
# import Diffractor: DiffractorForwardBackend
# 𝒷 = 𝒜.ForwardDiffBackend
# 𝒷 = Diffractor.DiffractorForwardBackend

import Polyester
import NLopt
import Optim, LineSearches
# import Zygote
import SparseArrays: SparseMatrixCSC, SparseVector, AbstractSparseArray, AbstractSparseMatrix, sparse! #, sparse, spzeros, droptol!, sparsevec, spdiagm, findnz#, sparse!
import LinearAlgebra as ℒ
import LinearAlgebra: mul!
# import Octavian: matmul!
# import TriangularSolve as TS
# import ComponentArrays as 𝒞
import Combinatorics: combinations
import BlockTriangularForm
import Subscripts: super, sub
import Krylov
import LinearOperators
import DataStructures: CircularBuffer
import SpeedMapping: speedmapping
import Suppressor: @suppress
import REPL
import Unicode
import MatrixEquations # good overview: https://cscproxy.mpi-magdeburg.mpg.de/mpcsc/benner/talks/Benner-Melbourne2019.pdf
# import NLboxsolve: nlboxsolve
# using NamedArrays
# using AxisKeys

import ChainRulesCore: @ignore_derivatives, ignore_derivatives, rrule, NoTangent, @thunk
import RecursiveFactorization as RF

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

using Requires

import Reexport
Reexport.@reexport import AxisKeys: KeyedArray, axiskeys, rekey
Reexport.@reexport import SparseArrays: sparse, spzeros, droptol!, sparsevec, spdiagm, findnz


# Type definitions
const Symbol_input = Union{Symbol,Vector{Symbol},Matrix{Symbol},Tuple{Symbol,Vararg{Symbol}}}
const String_input = Union{String,Vector{String},Matrix{String},Tuple{String,Vararg{String}}}
const ParameterType = Union{Nothing,
                            Pair{Symbol, Float64},
                            Pair{String, Float64},
                            Tuple{Pair{Symbol, Float64}, Vararg{Pair{Symbol, Float64}}},
                            Tuple{Pair{String, Float64}, Vararg{Pair{String, Float64}}},
                            Vector{Pair{Symbol, Float64}},
                            Vector{Pair{String, Float64}},
                            Pair{Symbol, Int},
                            Pair{String, Int},
                            Tuple{Pair{Symbol, Int}, Vararg{Pair{Symbol, Int}}},
                            Tuple{Pair{String, Int}, Vararg{Pair{String, Int}}},
                            Vector{Pair{Symbol, Int}},
                            Vector{Pair{String, Int}},
                            Pair{Symbol, Real},
                            Pair{String, Real},
                            Tuple{Pair{Symbol, Real}, Vararg{Pair{Symbol, Real}}},
                            Tuple{Pair{String, Real}, Vararg{Pair{String, Real}}},
                            Vector{Pair{Symbol, Real}},
                            Vector{Pair{String, Real}},
                            Dict{Symbol, Float64},
                            Tuple{Int, Vararg{Int}},
                            Matrix{Int},
                            Tuple{Float64, Vararg{Float64}},
                            Matrix{Float64},
                            Tuple{Real, Vararg{Real}},
                            Matrix{Real},
                            Vector{Float64} }


# Imports
include("common_docstrings.jl")
include("structures.jl")
include("sylvester.jl")
include("lyapunov.jl")
include("find_shocks.jl")
include("macros.jl")
include("get_functions.jl")
include("dynare.jl")
include("inspect.jl")

function __init__()
    @require StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd" include("plotting.jl")
    @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("priors.jl")
end


export @model, @parameters, solve!
export plot_irfs, plot_irf, plot_IRF, plot_simulations, plot_solution, plot_simulation #, plot
export plot_conditional_variance_decomposition, plot_forecast_error_variance_decomposition, plot_fevd, plot_model_estimates, plot_shock_decomposition
export get_irfs, get_irf, get_IRF, simulate, get_simulation, get_simulations
export get_conditional_forecast, plot_conditional_forecast
export get_solution, get_first_order_solution, get_perturbation_solution, get_second_order_solution, get_third_order_solution
export get_steady_state, get_SS, get_ss, get_non_stochastic_steady_state, get_stochastic_steady_state, get_SSS, steady_state, SS, SSS, ss, sss
export get_non_stochastic_steady_state_residuals, get_residuals, check_residuals
export get_moments, get_statistics, get_covariance, get_standard_deviation, get_variance, get_var, get_std, get_stdev, get_cov, var, std, stdev, cov, get_mean #, mean
export get_autocorrelation, get_correlation, get_variance_decomposition, get_corr, get_autocorr, get_var_decomp, corr, autocorr
export get_fevd, fevd, get_forecast_error_variance_decomposition, get_conditional_variance_decomposition
export calculate_jacobian, calculate_hessian, calculate_third_order_derivatives
export calculate_first_order_solution, calculate_second_order_solution, calculate_third_order_solution #, calculate_jacobian_manual, calculate_jacobian_sparse, calculate_jacobian_threaded
export get_shock_decomposition, get_estimated_shocks, get_estimated_variables, get_estimated_variable_standard_deviations, get_loglikelihood
export plotlyjs_backend, gr_backend
export Beta, InverseGamma, Gamma, Normal

export translate_mod_file, translate_dynare_file, import_model, import_dynare
export write_mod_file, write_dynare_file, write_to_dynare_file, write_to_dynare, export_dynare, export_to_dynare, export_mod_file, export_model

export get_equations, get_steady_state_equations, get_dynamic_equations, get_calibration_equations, get_parameters, get_calibrated_parameters, get_parameters_in_equations, get_parameters_defined_by_parameters, get_parameters_defining_parameters, get_calibration_equation_parameters, get_variables, get_nonnegativity_auxilliary_variables, get_dynamic_auxilliary_variables, get_shocks, get_state_variables, get_jump_variables
# Internal
export irf, girf

# Remove comment for debugging
# export riccati_forward, block_solver, remove_redundant_SS_vars!, write_parameters_input!, parse_variables_input_to_index, undo_transformer , transformer, calculate_third_order_stochastic_steady_state, calculate_second_order_stochastic_steady_state, filter_and_smooth
# export create_symbols_eqs!, solve_steady_state!, write_functions_mapping!, solve!, parse_algorithm_to_state_update, block_solver, block_solver_AD, calculate_covariance, calculate_jacobian, calculate_first_order_solution, expand_steady_state, calculate_quadratic_iteration_solution, calculate_linear_time_iteration_solution, get_symbols, calculate_covariance_AD, parse_shocks_input_to_index


# StatsFuns
norminvcdf(p) = -erfcinv(2*p) * 1.4142135623730951
norminv(p::Number) = norminvcdf(p)
qnorm(p::Number) = norminvcdf(p)
normlogpdf(z) = -(abs2(z) + 1.8378770664093453)/2
normpdf(z) = exp(-abs2(z)/2) * 0.3989422804014327
normcdf(z) = erfc(-z * 0.7071067811865475)/2
pnorm(p::Number) = normcdf(p)
dnorm(p::Number) = normpdf(p)




Base.show(io::IO, 𝓂::ℳ) = println(io, 
                "Model:        ", 𝓂.model_name, 
                "\nVariables", 
                "\n Total:       ", 𝓂.timings.nVars,
                "\n  Auxiliary:  ", length(𝓂.exo_present) + length(𝓂.aux),
                "\n States:      ", 𝓂.timings.nPast_not_future_and_mixed,
                "\n  Auxiliary:  ",  length(intersect(𝓂.timings.past_not_future_and_mixed, 𝓂.aux_present)),
                "\n Jumpers:     ", 𝓂.timings.nFuture_not_past_and_mixed, # 𝓂.timings.mixed, 
                "\n  Auxiliary:  ", length(intersect(𝓂.timings.future_not_past_and_mixed, union(𝓂.aux_present, 𝓂.aux_future))),
                "\nShocks:       ", 𝓂.timings.nExo,
                "\nParameters:   ", length(𝓂.parameters_in_equations),
                if 𝓂.calibration_equations == Expr[]
                    ""
                else
                    "\nCalibration\nequations:    " * repr(length(𝓂.calibration_equations))
                end,
                # "\n¹: including auxilliary variables"
                # "\nVariable bounds (upper,lower,any): ",sum(𝓂.upper_bounds .< Inf),", ",sum(𝓂.lower_bounds .> -Inf),", ",length(𝓂.bounds),
                # "\nNon-stochastic-steady-state found: ",!𝓂.solution.outdated_NSSS
                )

check_for_dynamic_variables(ex::Int) = false
check_for_dynamic_variables(ex::Float64) = false
check_for_dynamic_variables(ex::Symbol) = occursin(r"₍₁₎|₍₀₎|₍₋₁₎",string(ex))

function check_for_dynamic_variables(ex::Expr)
    dynamic_indicator = Bool[]

    postwalk(x -> 
        x isa Expr ?
            x.head == :ref ? 
                occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    x :
                begin
                    push!(dynamic_indicator,true)
                    x
                end :
            x :
        x,
    ex)

    any(dynamic_indicator)
end


function transform_expression(expr::Expr)
    # Dictionary to store the transformations for reversing
    reverse_transformations = Dict{Symbol, Expr}()

    # Counter for generating unique placeholders
    unique_counter = Ref(0)

    # Step 1: Replace min/max calls and record their original form
    function replace_min_max(expr)
        if expr isa Expr && expr.head == :call && (expr.args[1] == :min || expr.args[1] == :max)
            # Replace min/max functions with a placeholder
            # placeholder = Symbol("minimal__P", unique_counter[])
            placeholder = :minmax__P
            unique_counter[] += 1

            # Store the original min/max call for reversal
            reverse_transformations[placeholder] = expr

            return placeholder
        else
            return expr
        end
    end

    # Step 2: Transform :ref fields in the rest of the expression
    function transform_ref_fields(expr)
        if expr isa Expr && expr.head == :ref && isa(expr.args[1], Symbol)
            # Handle :ref expressions
            if isa(expr.args[2], Number) || isa(expr.args[2], Symbol)           
                if expr.args[2] < 0
                    new_symbol = Symbol(expr.args[1], "__", abs(expr.args[2]))
                else
                    new_symbol = Symbol(expr.args[1], "_", expr.args[2])
                end
            else
                # Generate a unique placeholder for complex :ref
                unique_counter[] += 1
                placeholder = Symbol("__placeholder", unique_counter[])
                new_symbol = placeholder
            end

            # Record the reverse transformation
            reverse_transformations[new_symbol] = expr

            return new_symbol
        else
            return expr
        end
    end


    # Replace equality sign with minus
    function replace_equality_with_minus(expr)
        if expr isa Expr && expr.head == :(=)
            return Expr(:call, :-, expr.args...)
        else
            return expr
        end
    end

    # Apply transformations
    expr = postwalk(replace_min_max, expr)
    expr = postwalk(transform_ref_fields, expr)
    transformed_expr = postwalk(replace_equality_with_minus, expr)
    
    return transformed_expr, reverse_transformations
end


function reverse_transformation(transformed_expr::Expr, reverse_dict::Dict{Symbol, Expr})
    # Function to replace the transformed symbols with their original form
    function revert_symbol(expr)
        if expr isa Symbol && haskey(reverse_dict, expr)
            return reverse_dict[expr]
        else
            return expr
        end
    end

    # Revert the expression using postwalk
    reverted_expr = postwalk(revert_symbol, transformed_expr)

    return reverted_expr
end


function transform_obc(ex::Expr; avoid_solve::Bool = false)
    transformed_expr, reverse_dict = transform_expression(ex)

    for symbs in get_symbols(transformed_expr)
        eval(:($symbs = SPyPyC.symbols($(string(symbs)), real = true, finite = true)))
    end

    eq = eval(transformed_expr)

    if avoid_solve && count_ops(Meta.parse(string(eq))) > 15
        soll = nothing
    else
        soll = try SPyPyC.solve(eq, eval(:minmax__P))
        catch
        end
    end

    if length(soll) == 1
        sorted_minmax = Expr(:call, reverse_dict[:minmax__P].args[1], :($(reverse_dict[:minmax__P].args[2]) - $(Meta.parse(string(soll[1])))),  :($(reverse_dict[:minmax__P].args[3]) - $(Meta.parse(string(soll[1])))))
        return reverse_transformation(sorted_minmax, reverse_dict)
    else
        @error "Occasionally binding constraint not well-defined. See documentation for examples."
    end
end


function obc_constraint_optim_fun(res::Vector{S}, X::Vector{S}, jac::Matrix{S}, p) where S
    𝓂 = p[4]

    if length(jac) > 0
        # jac .= 𝒜.jacobian(𝒷(), xx -> 𝓂.obc_violation_function(xx, p), X)[1]'
        jac .= 𝒟.jacobian(xx -> 𝓂.obc_violation_function(xx, p), backend, X)'
    end

    res .= 𝓂.obc_violation_function(X, p)
end

function obc_objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
    if length(grad) > 0
        grad .= 2 .* X
    end
    
    sum(abs2, X)
end



function match_conditions(res::Vector{S}, X::Vector{S}, jac::Matrix{S}, p) where S
    Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, precision_factor = p

    if length(jac) > 0
        # jac .= 𝒜.jacobian(𝒷(), xx -> begin
        #                                 Shocks[Free_shock_idx] .= xx

        #                                 new_State = State_update(State, convert(typeof(xx), Shocks))

        #                                 cond_vars = Pruning ? sum(new_State) : new_State
                                        
        #                                 return precision_factor .* abs.(Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
        #                             end, X)[1]'
        jac .= 𝒟.jacobian(xx -> begin
                                    Shocks[Free_shock_idx] .= xx

                                    new_State = State_update(State, convert(typeof(xx), Shocks))

                                    cond_vars = Pruning ? sum(new_State) : new_State
                                    
                                    return precision_factor .* abs.(Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
                                end, backend, X)'
    end

    Shocks[Free_shock_idx] .= X

    new_State = State_update(State, convert(typeof(X), Shocks))

    cond_vars = Pruning ? sum(new_State) : new_State

    res .= precision_factor .* abs.(Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
end


function minimize_distance_to_conditions(X::Vector{S}, p)::S where S
    Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, precision_factor = p

    Shocks[Free_shock_idx] .= X

    new_State = State_update(State, convert(typeof(X), Shocks))

    cond_vars = Pruning ? sum(new_State) : new_State

    return precision_factor * sum(abs2, Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
end



function minimize_distance_to_conditions!(X::Vector{S}, grad::Vector{S}, p) where S
    Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, precision_factor = p

    if length(grad) > 0
        # grad .= 𝒜.gradient(𝒷(), xx -> begin
        #                                 Shocks[Free_shock_idx] .= xx

        #                                 new_State = State_update(State, convert(typeof(xx), Shocks))

        #                                 cond_vars = Pruning ? sum(new_State) : new_State
                                        
        #                                 return precision_factor * sum(abs2, Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
        #                             end, X)[1]
        grad .= 𝒟.gradient(xx -> begin
                                        Shocks[Free_shock_idx] .= xx

                                        new_State = State_update(State, convert(typeof(xx), Shocks))

                                        cond_vars = Pruning ? sum(new_State) : new_State
                                        
                                        return precision_factor * sum(abs2, Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
                                    end, backend, X)

    end

    Shocks[Free_shock_idx] .= X

    new_State = State_update(State, convert(typeof(X), Shocks))

    cond_vars = Pruning ? sum(new_State) : new_State

    return precision_factor * sum(abs2, Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
end


# function match_conditions(res::Vector{S}, X::Vector{S}, jac::Matrix{S}, p) where S
#     conditions, state_update, shocks, cond_var_idx, free_shock_idx, state, 𝒷 = p
    
#     if length(jac) > 0
#         jac .= 𝒜.jacobian(𝒷(), xx -> begin
#                                         shocks[free_shock_idx] .= xx
#                                         return abs2.(conditions[cond_var_idx] - state_update(state, convert(typeof(xx), shocks))[cond_var_idx])
#                                     end, X)[1]'
#     end

#     shocks[free_shock_idx] .= X

#     res .= abs2.(conditions[cond_var_idx] - state_update(state, convert(typeof(X), shocks))[cond_var_idx])
# end





function minimize_distance_to_initial_data!(X::Vector{S}, grad::Vector{S}, data::Vector{T}, state::Union{Vector{T},Vector{Vector{T}}}, state_update::Function, warmup_iters::Int, cond_var_idx::Vector{Union{Nothing, Int64}}, precision_factor::Float64) where {S, T}
    if state isa Vector{T}
        pruning = false
    else
        pruning = true
    end

    if length(grad) > 0
        # grad .= 𝒜.gradient(𝒷(), xx -> begin
        #                                 state_copy = deepcopy(state)

        #                                 XX = reshape(xx, length(X) ÷ warmup_iters, warmup_iters)

        #                                 for i in 1:warmup_iters
        #                                     state_copy = state_update(state_copy, XX[:,i])
        #                                 end

        #                                 return precision_factor .* sum(abs2, data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
        #                             end, X)[1]
        grad .= 𝒟.gradient(xx -> begin
                                        state_copy = deepcopy(state)

                                        XX = reshape(xx, length(X) ÷ warmup_iters, warmup_iters)

                                        for i in 1:warmup_iters
                                            state_copy = state_update(state_copy, XX[:,i])
                                        end

                                        return precision_factor .* sum(abs2, data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
                                    end, backend, X)
    end

    state_copy = deepcopy(state)

    XX = reshape(X, length(X) ÷ warmup_iters, warmup_iters)

    for i in 1:warmup_iters
        state_copy = state_update(state_copy, XX[:,i])
    end

    return precision_factor .* sum(abs2, data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
end




function match_initial_data!(res::Vector{S}, X::Vector{S}, jac::Matrix{S}, data::Vector{T}, state::Union{Vector{T},Vector{Vector{T}}}, state_update::Function, warmup_iters::Int, cond_var_idx::Vector{Union{Nothing, Int64}}, precision_factor::Float64) where {S, T}
    if state isa Vector{T}
        pruning = false
    else
        pruning = true
    end

    if length(jac) > 0
        # jac .= 𝒜.jacobian(𝒷(), xx -> begin
        #                                 state_copy = deepcopy(state)

        #                                 XX = reshape(xx, length(X) ÷ warmup_iters, warmup_iters)

        #                                 for i in 1:warmup_iters
        #                                     state_copy = state_update(state_copy, XX[:,i])
        #                                 end

        #                                 return precision_factor .* abs.(data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
        #                             end, X)[1]'
        jac .= 𝒟.jacobian(xx -> begin
                                        state_copy = deepcopy(state)

                                        XX = reshape(xx, length(X) ÷ warmup_iters, warmup_iters)

                                        for i in 1:warmup_iters
                                            state_copy = state_update(state_copy, XX[:,i])
                                        end

                                        return precision_factor .* abs.(data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
                                    end, backend, X)'
    end

    if length(res) > 0
        state_copy = deepcopy(state)

        XX = reshape(X, length(X) ÷ warmup_iters, warmup_iters)

        for i in 1:warmup_iters
            state_copy = state_update(state_copy, XX[:,i])
        end

        res .= abs.(data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
    end
end



function minimize_distance_to_initial_data(X::Vector{S}, data::Vector{T}, state::Union{Vector{T},Vector{Vector{T}}}, state_update::Function, warmup_iters::Int, cond_var_idx::Vector{Union{Nothing, Int64}}, precision_factor::Float64, pruning::Bool)::S where {S, T}
    state_copy = deepcopy(state)

    XX = reshape(X, length(X) ÷ warmup_iters, warmup_iters)

    for i in 1:warmup_iters
        state_copy = state_update(state_copy, XX[:,i])
    end

    return precision_factor .* sum(abs2, data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
end





function minimize_distance_to_data(X::Vector{S}, Data::Vector{T}, State::Union{Vector{T},Vector{Vector{T}}}, state_update::Function, cond_var_idx::Vector{Union{Nothing, Int64}}, precision_factor::Float64, pruning::Bool)::S where {S, T}
    return precision_factor .* sum(abs2, Data - (pruning ? sum(state_update(State, X)) : state_update(State, X))[cond_var_idx])
end


function minimize_distance_to_data!(X::Vector{S}, grad::Vector{S}, Data::Vector{T}, State::Union{Vector{T},Vector{Vector{T}}}, state_update::Function, cond_var_idx::Vector{Union{Nothing, Int64}}, precision_factor::Float64) where {S, T}
    if State isa Vector{T}
        pruning = false
    else
        pruning = true
    end
    
    if length(grad) > 0
        # grad .= 𝒜.gradient(𝒷(), xx -> precision_factor .* sum(abs2, Data - (pruning ? sum(state_update(State, xx)) : state_update(State, xx))[cond_var_idx]), X)[1]
        grad .= 𝒟.gradient(xx -> precision_factor .* sum(abs2, Data - (pruning ? sum(state_update(State, xx)) : state_update(State, xx))[cond_var_idx]), backend, X)
    end

    return precision_factor .* sum(abs2, Data - (pruning ? sum(state_update(State, X)) : state_update(State, X))[cond_var_idx])
end



function match_data_sequence!(res::Vector{S}, X::Vector{S}, jac::Matrix{S}, Data::Vector{T}, State::Union{Vector{T},Vector{Vector{T}}}, state_update::Function, cond_var_idx::Vector{Union{Nothing, Int64}}, precision_factor::Float64) where {S, T}
    if State isa Vector{T}
        pruning = false
    else
        pruning = true
    end
    
    if length(jac) > 0
        # jac .= 𝒜.jacobian(𝒷(), xx -> precision_factor .* abs.(Data - (pruning ? sum(state_update(State, xx)) : state_update(State, xx))[cond_var_idx]), X)[1]'
        jac .= 𝒟.jacobian(xx -> precision_factor .* abs.(Data - (pruning ? sum(state_update(State, xx)) : state_update(State, xx))[cond_var_idx]), backend, X)[1]'
    end

    if length(res) > 0
        res .= precision_factor .* abs.(Data - (pruning ? sum(state_update(State, X)) : state_update(State, X))[cond_var_idx])
    end
end


function set_up_obc_violation_function!(𝓂)
    present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎$")))

    sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))

    # write indices in auxiliary objects
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎")))

    dyn_var_present = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_var_present_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

    SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

    dyn_var_present_idx = indexin(dyn_var_present   , SS_and_pars_names)

    alll = []
    for (i,var) in enumerate(present_varss)
        if !(match(r"^χᵒᵇᶜ", string(var)) === nothing)
            push!(alll,:($var = Y[$(dyn_var_present_idx[i]),1:max(periods, 1)]))
        end
    end

    calc_obc_violation = :(function calculate_obc_violation(x, p)
        state, state_update, reference_steady_state, 𝓂, algorithm, periods, shock_values = p

        T = 𝓂.timings

        Y = zeros(typeof(x[1]), T.nVars, periods+1)

        shock_values = convert(typeof(x), shock_values)

        shock_values[contains.(string.(T.exo),"ᵒᵇᶜ")] .= x

        zero_shock = zero(shock_values)

        if algorithm ∈ [:pruned_second_order, :pruned_third_order]
            states = state_update(state, shock_values)
            Y[:,1] = sum(states)
        else
            Y[:,1] = state_update(state, shock_values)
        end

        for t in 1:periods
            if algorithm ∈ [:pruned_second_order, :pruned_third_order]
                states = state_update(states, zero_shock)
                Y[:,t+1] = sum(states)
            else
                Y[:,t+1] = state_update(Y[:,t], zero_shock)
            end
        end

        Y .+= reference_steady_state[1:T.nVars]

        $(alll...)

        constraint_values = Vector[]

        $(𝓂.obc_violation_equations...)

        return vcat(constraint_values...)
    end)

    𝓂.obc_violation_function = @RuntimeGeneratedFunction(calc_obc_violation)

    return nothing
end


function check_for_minmax(expr)
    contains_minmax = Bool[]

    postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        x.args[1] ∈ [:max,:min] ?
                            begin
                                push!(contains_minmax,true)
                                x
                            end :
                        x :
                    x :
                x,
    expr)

    any(contains_minmax)
end


function write_obc_violation_equations(𝓂)
    eqs = Expr[]
    for (i,eq) in enumerate(𝓂.dyn_equations)
        if check_for_minmax(eq)
            minmax_fixed_eqs = postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        length(x.args) == 3 ?
                            x.args[3] isa Expr ?
                                x.args[3].args[1] ∈ [:Min, :min, :Max, :max] ?
                                    begin
                                        plchldr = Symbol(replace(string(x.args[2]), "₍₀₎" => ""))

                                        ineq_plchldr_1 = x.args[3].args[2] isa Symbol ? Symbol(replace(string(x.args[3].args[2]), "₍₀₎" => "")) : x.args[3].args[2]

                                        arg1 = x.args[3].args[2]
                                        arg2 = x.args[3].args[3]

                                        dyn_1 = check_for_dynamic_variables(x.args[3].args[2])
                                        dyn_2 = check_for_dynamic_variables(x.args[3].args[3])

                                        cond1 = Expr[]
                                        cond2 = Expr[]

                                        maximisation = contains(string(plchldr), "⁺")
                                        
                                        # if dyn_1
                                        #     if maximisation
                                        #         push!(cond1, :(push!(constraint_values, $(x.args[3].args[2]))))
                                        #         # push!(cond2, :(push!(constraint_values, $(x.args[3].args[2]))))
                                        #     else
                                        #         push!(cond1, :(push!(constraint_values, -$(x.args[3].args[2]))))
                                        #         # push!(cond2, :(push!(constraint_values, -$(x.args[3].args[2])))) # RBC
                                        #     end
                                        # end

                                        # if dyn_2
                                        #     if maximisation
                                        #         push!(cond1, :(push!(constraint_values, $(x.args[3].args[3]))))
                                        #         # push!(cond2, :(push!(constraint_values, $(x.args[3].args[3])))) # testmax
                                        #     else
                                        #         push!(cond1, :(push!(constraint_values, -$(x.args[3].args[3]))))
                                        #         # push!(cond2, :(push!(constraint_values, -$(x.args[3].args[3])))) # RBC
                                        #     end
                                        # end


                                        if maximisation
                                            push!(cond1, :(push!(constraint_values, [sum($(x.args[3].args[2]) .* $(x.args[3].args[3]))])))
                                            push!(cond1, :(push!(constraint_values, $(x.args[3].args[2]))))
                                            push!(cond1, :(push!(constraint_values, $(x.args[3].args[3]))))
                                            # push!(cond1, :(push!(constraint_values, max.($(x.args[3].args[2]), $(x.args[3].args[3])))))
                                        else
                                            push!(cond1, :(push!(constraint_values, [sum($(x.args[3].args[2]) .* $(x.args[3].args[3]))])))
                                            push!(cond1, :(push!(constraint_values, -$(x.args[3].args[2]))))
                                            push!(cond1, :(push!(constraint_values, -$(x.args[3].args[3]))))
                                            # push!(cond1, :(push!(constraint_values, min.($(x.args[3].args[2]), $(x.args[3].args[3])))))
                                        end

                                        # if maximisation
                                        #     push!(cond1, :(push!(shock_sign_indicators, true)))
                                        #     # push!(cond2, :(push!(shock_sign_indicators, true)))
                                        # else
                                        #     push!(cond1, :(push!(shock_sign_indicators, false)))
                                        #     # push!(cond2, :(push!(shock_sign_indicators, false)))
                                        # end

                                        # :(if isapprox($plchldr, $ineq_plchldr_1, atol = 1e-12)
                                        #     $(Expr(:block, cond1...))
                                        # else
                                        #     $(Expr(:block, cond2...))
                                        # end)
                                        :($(Expr(:block, cond1...)))
                                    end :
                                x :
                            x :
                        x :
                    x :
                x,
            eq)

            push!(eqs, minmax_fixed_eqs)
        end
    end

    return eqs
end

# function parse_obc_shock_bounds(expr::Expr)
#     # Determine the order of the shock and bound in the expression
#     shock_first = isa(expr.args[2], Symbol)
    
#     # Extract the shock and bound from the expression
#     shock = shock_first ? expr.args[2] : expr.args[3]
#     bound_expr = shock_first ? expr.args[3] : expr.args[2]
    
#     # Evaluate the bound expression to get a numerical value
#     bound = eval(bound_expr) |> Float64
    
#     # Determine whether the bound is a lower or upper bound based on the comparison operator and order
#     is_upper_bound = (expr.args[1] in (:<, :≤) && shock_first) || (expr.args[1] in (:>, :≥) && !shock_first)
    
#     return shock, is_upper_bound, bound
# end



function mat_mult_kron(A::AbstractArray{T},B::AbstractArray{T},C::AbstractArray{T}; tol::AbstractFloat = eps()) where T <: Real
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    B̄ = collect(B)
    C̄ = collect(C)

    vals = T[]
    rows = Int[]
    cols = Int[]

    for row in 1:size(A,1)
        idx_mat, vals_mat = A[row,:] |> findnz

        if length(vals_mat) == 0 continue end

        for col in 1:(n_colB*n_colC)
            col_1, col_2 = divrem((col - 1) % (n_colB*n_colC), n_colC) .+ 1

            mult_val = 0.0

            for (i,idx) in enumerate(idx_mat)
                i_1, i_2 = divrem((idx - 1) % (n_rowB*n_rowC), n_rowC) .+ 1
                
                mult_val += vals_mat[i] * B̄[i_1,col_1] * C̄[i_2,col_2]
            end

            if abs(mult_val) > tol
                push!(vals,mult_val)
                push!(rows,row)
                push!(cols,col)
            end
        end
    end

    sparse(rows,cols,vals,size(A,1),n_colB*n_colC)
end

function kron³(A::SparseMatrixCSC{T}, M₃::third_order_auxilliary_matrices) where T <: Real
    rows, cols, vals = findnz(A)

    # Dictionary to accumulate sums of values for each coordinate
    result_dict = Dict{Tuple{Int, Int}, T}()

    # Using a single iteration over non-zero elements
    nvals = length(vals)

    lk = ReentrantLock()

    Polyester.@batch for i in 1:nvals
        for j in 1:nvals
            for k in 1:nvals
                r1, c1, v1 = rows[i], cols[i], vals[i]
                r2, c2, v2 = rows[j], cols[j], vals[j]
                r3, c3, v3 = rows[k], cols[k], vals[k]
                
                sorted_cols = [c1, c2, c3]
                sorted_rows = [r1, r2, r3] # a lot of time spent here
                sort!(sorted_rows, rev = true) # a lot of time spent here
                
                if haskey(M₃.𝐈₃, sorted_cols) # && haskey(M₃.𝐈₃, sorted_rows) # a lot of time spent here
                    row_idx = M₃.𝐈₃[sorted_rows]
                    col_idx = M₃.𝐈₃[sorted_cols]

                    key = (row_idx, col_idx)

                    begin
                        lock(lk)
                        try
                            if haskey(result_dict, key)
                                result_dict[key] += v1 * v2 * v3
                            else
                                result_dict[key] = v1 * v2 * v3
                            end
                        finally
                            unlock(lk)
                        end
                    end
                end
            end
        end
    end

    # Extract indices and values from the dictionary
    result_rows = Int[]
    result_cols = Int[]
    result_vals = T[]

    for (ks, valu) in result_dict
        push!(result_rows, ks[1])
        push!(result_cols, ks[2])
        push!(result_vals, valu)
    end
    
    # Create the sparse matrix from the collected indices and values
    if VERSION >= v"1.10"
        return sparse!(result_rows, result_cols, result_vals, size(M₃.𝐂₃, 2), size(M₃.𝐔₃, 1))
    else
        return sparse(result_rows, result_cols, result_vals, size(M₃.𝐂₃, 2), size(M₃.𝐔₃, 1))
    end
end

function A_mult_kron_power_3_B(A::AbstractArray{R},B::AbstractArray{T}; tol::AbstractFloat = eps()) where {R <: Real, T <: Real}
    n_row = size(B,1)
    n_col = size(B,2)

    B̄ = collect(B)

    vals = T[]
    rows = Int[]
    cols = Int[]

    for row in 1:size(A,1)
        idx_mat, vals_mat = A[row,:] |> findnz

        if length(vals_mat) == 0 continue end

        for col in 1:size(B,2)^3
            col_1, col_3 = divrem((col - 1) % (n_col^2), n_col) .+ 1
            col_2 = ((col - 1) ÷ (n_col^2)) + 1

            mult_val = 0.0

            for (i,idx) in enumerate(idx_mat)
                i_1, i_3 = divrem((idx - 1) % (n_row^2), n_row) .+ 1
                i_2 = ((idx - 1) ÷ (n_row^2)) + 1
                mult_val += vals_mat[i] * B̄[i_1,col_1] * B̄[i_2,col_2] * B̄[i_3,col_3]
            end

            if abs(mult_val) > tol
                push!(vals,mult_val)
                push!(rows,row)
                push!(cols,col)
            end
        end
    end

    sparse(rows,cols,vals,size(A,1),size(B,2)^3)
end


function translate_symbol_to_ascii(x::Symbol)
    ss = Unicode.normalize(replace(string(x),  "◖" => "__", "◗" => "__"), :NFD)

    outstr = ""

    for i in ss
        out = REPL.symbol_latex(string(i))[2:end]
        if out == ""
            outstr *= string(i)
        else
            outstr *= replace(out,  
                        r"\!" => s"_",
                        r"\(" => s"_",
                        r"\)" => s"_",
                        r"\^" => s"_",
                        r"\_\^" => s"_",
                        r"\+" => s"plus",
                        r"\-" => s"minus",
                        r"\*" => s"times")
            if i != ss[end]
                outstr *= "_"
            end
        end
    end

    return outstr
end


function translate_expression_to_ascii(exp::Expr)
    postwalk(x -> 
                x isa Symbol ?
                    begin
                        x_tmp = translate_symbol_to_ascii(x)

                        if x_tmp == string(x)
                            x
                        else
                            Symbol(x_tmp)
                        end
                    end :
                x,
    exp)
end
                


# function jacobian_wrt_values(A, B)
#     # does this without creating dense arrays: reshape(permutedims(reshape(ℒ.I - ℒ.kron(A, B) ,size(B,1), size(A,1), size(A,1), size(B,1)), [2, 3, 4, 1]), size(A,1) * size(B,1), size(A,1) * size(B,1))

#     # Compute the Kronecker product and subtract from identity
#     C = ℒ.I - ℒ.kron(A, B)

#     # Extract the row, column, and value indices from C
#     rows, cols, vals = findnz(C)

#     # Lists to store the 2D indices after the operations
#     final_rows = zeros(Int,length(rows))
#     final_cols = zeros(Int,length(rows))

#     Threads.@threads for i = 1:length(rows)
#         # Convert the 1D row index to its 2D components
#         i1, i2 = divrem(rows[i]-1, size(B,1)) .+ 1

#         # Convert the 1D column index to its 2D components
#         j1, j2 = divrem(cols[i]-1, size(A,1)) .+ 1

#         # Convert the 4D index (i1, j2, j1, i2) to a 2D index in the final matrix
#         final_col, final_row = divrem(Base._sub2ind((size(A,1), size(A,1), size(B,1), size(B,1)), i1, j2, j1, i2) - 1, size(A,1) * size(B,1)) .+ 1

#         # Store the 2D indices
#         final_rows[i] = final_row
#         final_cols[i] = final_col
#     end

#     return sparse(final_rows, final_cols, vals, size(A,1) * size(B,1), size(A,1) * size(B,1))
# end




# function jacobian_wrt_A(A, X)
#     # does this without creating dense arrays: reshape(permutedims(reshape(ℒ.I - ℒ.kron(A, B) ,size(B,1), size(A,1), size(A,1), size(B,1)), [2, 3, 4, 1]), size(A,1) * size(B,1), size(A,1) * size(B,1))

#     # Compute the Kronecker product and subtract from identity
#     C = ℒ.kron(ℒ.I(size(A,1)), sparse(A * X))

#     # Extract the row, column, and value indices from C
#     rows, cols, vals = findnz(C)

#     # Lists to store the 2D indices after the operations
#     final_rows = zeros(Int,length(rows))
#     final_cols = zeros(Int,length(rows))

#     Threads.@threads for i = 1:length(rows)
#         # Convert the 1D row index to its 2D components
#         i1, i2 = divrem(rows[i]-1, size(A,1)) .+ 1

#         # Convert the 1D column index to its 2D components
#         j1, j2 = divrem(cols[i]-1, size(A,1)) .+ 1

#         # Convert the 4D index (i1, j2, j1, i2) to a 2D index in the final matrix
#         final_col, final_row = divrem(Base._sub2ind((size(A,1), size(A,1), size(A,1), size(A,1)), i2, i1, j1, j2) - 1, size(A,1) * size(A,1)) .+ 1

#         # Store the 2D indices
#         final_rows[i] = final_row
#         final_cols[i] = final_col
#     end

#     r,c,_ = findnz(A) 
    
#     non_zeros_only = spzeros(Int,size(A,1)^2,size(A,1)^2)
    
#     non_zeros_only[CartesianIndex.(r .+ (c.-1) * size(A,1), r .+ (c.-1) * size(A,1))] .= 1
    
#     return sparse(final_rows, final_cols, vals, size(A,1) * size(A,1), size(A,1) * size(A,1)) + ℒ.kron(sparse(X * A'), ℒ.I(size(A,1)))' * non_zeros_only
# end


# # higher order solutions moment helper functions

# function warshall_algorithm!(R::SparseMatrixCSC{Bool,Int64})
#     # Size of the matrix
#     n, m = size(R)
    
#     @assert n == m "Warshall algorithm only works for square matrices."

#     # The core idea of the Warshall algorithm is to consider each node (in this case, block)
#     # as an intermediate node and check if a path can be created between two nodes by using the
#     # intermediate node.
    
#     # k is the intermediate node (or block).
#     for k in 1:n
#         # i is the starting node (or block).
#         for i in 1:n
#             # j is the ending node (or block).
#             for j in 1:n
#                 # If there is a direct path from i to k AND a direct path from k to j, 
#                 # then a path from i to j exists via k.
#                 # Thus, set the value of R[i, j] to 1 (true).
#                 R[i, j] = R[i, j] || (R[i, k] && R[k, j])
#             end
#         end
#     end
    
#     # Return the transitive closure matrix.
#     return R
# end

function combine_pairs(v::Vector{Pair{Vector{Symbol}, Vector{Symbol}}})
    i = 1
    while i <= length(v)
        subset_found = false
        for j in i+1:length(v)
            # Check if v[i].second is subset of v[j].second or vice versa
            if all(elem -> elem in v[j].second, v[i].second) || all(elem -> elem in v[i].second, v[j].second)
                # Combine the first elements and assign to the one with the larger second element
                if length(v[i].second) > length(v[j].second)
                    v[i] = v[i].first ∪ v[j].first => v[i].second
                else
                    v[j] = v[i].first ∪ v[j].first => v[j].second
                end
                # Remove the one with the smaller second element
                deleteat!(v, length(v[i].second) > length(v[j].second) ? j : i)
                subset_found = true
                break
            end
        end
        # If no subset was found for v[i], move to the next element
        if !subset_found
            i += 1
        end
    end
    return v
end

function determine_efficient_order(𝐒₁::Matrix{<: Real}, 
    T::timings, 
    variables::Union{Symbol_input,String_input};
    tol::AbstractFloat = eps())

    orders = Pair{Vector{Symbol}, Vector{Symbol}}[]

    nˢ = T.nPast_not_future_and_mixed
    
    if variables == :full_covar
        return [T.var => T.past_not_future_and_mixed]
    else
        var_idx = MacroModelling.parse_variables_input_to_index(variables, T)
        observables = T.var[var_idx]
    end

    for obs in observables
        obs_in_var_idx = indexin([obs],T.var)
        dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0

        while dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁[indexin(T.past_not_future_and_mixed, T.var),1:nˢ]) .> tol) != dependencies_in_states
            dependencies_in_states = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁[indexin(T.past_not_future_and_mixed, T.var),1:nˢ]) .> tol)
        end

        dependencies = T.past_not_future_and_mixed[dependencies_in_states]

        push!(orders,[obs] => sort(dependencies))
    end

    sort!(orders, by = x -> length(x[2]), rev = true)

    return combine_pairs(orders)
end

# function determine_efficient_order(∇₁::SparseMatrixCSC{<: Real}, 
#                                     T::timings, 
#                                     variables::Union{Symbol_input,String_input};
#                                     tol::AbstractFloat = eps())

#     droptol!(∇₁, tol)

#     if variables == :full_covar
#         return [T.var => T.var]
#     else
#         var_idx = parse_variables_input_to_index(variables, T)
#         observables = T.var[var_idx]
#     end

#     expand = [  spdiagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
#                 spdiagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 
    
#     ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
#     ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
#     ∇₋ = ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]

#     incidence = abs.(∇₊) + abs.(∇₀) + abs.(∇₋)

#     Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence))
#     R̂ = []
#     for i in 1:n_blocks
#         [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1] - 1]
#     end
#     push!(R̂,1)
    
#     vars = hcat(P, R̂)'
#     eqs  = hcat(Q, R̂)'
    
#     dependency_matrix = incidence[vars[1,:], eqs[1,:]] .!= 0
    
#     warshall_algorithm!(dependency_matrix)

#     solve_order = Vector{Symbol}[]
#     already_solved_for = Set{Symbol}()
#     corresponding_dependencies = Vector{Symbol}[]

#     for obs in intersect(T.var[eqs[1,:]], observables)
#         dependencies = T.var[eqs[1,:]][findall(dependency_matrix[indexin([obs], T.var[eqs[1,:]])[1],:])]
#         to_be_solved_for = setdiff(intersect(observables, dependencies), already_solved_for)
#         if length(to_be_solved_for) > 0
#             push!(solve_order, to_be_solved_for)
#             push!(corresponding_dependencies, dependencies)
#         end
#         push!(already_solved_for, intersect(observables, dependencies)...)
#     end

#     return solve_order .=> corresponding_dependencies
# end

function get_and_check_observables(𝓂::ℳ, data::KeyedArray{Float64})::Vector{Symbol}
    @assert size(data,1) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    observables = collect(axiskeys(data,1))

    @assert observables isa Vector{String} || observables isa Vector{Symbol}  "Make sure that the data has variables names as rows. They can be either Strings or Symbols."

    observables_symbols = observables isa String_input ? observables .|> Meta.parse .|> replace_indices : observables

    @assert length(setdiff(observables_symbols, 𝓂.var)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(observables_symbols,𝓂.var))

    sort!(observables_symbols)
    
    return observables_symbols
end

function bivariate_moment(moment::Vector{Int}, rho::Int)::Int
    if (moment[1] + moment[2]) % 2 == 1
        return 0
    end

    result = 1
    coefficient = 1
    odd_value = 2 * (moment[1] % 2)

    for j = 1:min(moment[1] ÷ 2, moment[2] ÷ 2)
        coefficient *= 2 * (moment[1] ÷ 2 + 1 - j) * (moment[2] ÷ 2 + 1 - j) * rho^2 / (j * (2 * j - 1 + odd_value))
        result += coefficient
    end

    if odd_value == 2
        result *= rho
    end

    result *= prod(1:2:moment[1]) * prod(1:2:moment[2])

    return result
end


function product_moments(V, ii, nu)
    s = sum(nu)

    if s == 0
        return 1
    elseif isodd(s)
        return 0
    end

    mask = .!(nu .== 0)
    nu = nu[mask]
    ii = ii[mask]
    V = V[ii, ii]

    m, s2 = length(ii), s / 2

    if m == 1
        return (V^s2 * prod(1:2:s-1))[1]
    elseif m == 2
        if V[1,1]==0 || V[2,2]==0
            return 0
        end
        rho = V[1, 2] / sqrt(V[1, 1] * V[2, 2])
        return (V[1, 1]^(nu[1] / 2) * V[2, 2]^(nu[2] / 2) * bivariate_moment(nu, Int(rho)))[1]
    end

    inu = sortperm(nu, rev=true)

    sort!(nu, rev=true)

    V = V[inu, inu]

    x = zeros(Int, 1, m)
    V = V / 2
    nu2 = nu' / 2
    p = 2
    q = nu2 * V * nu2'
    y = 0

    for _ in 1:round(Int, prod(nu .+ 1) / 2)
        y += p * q^s2
        for j in 1:m
            if x[j] < nu[j]
                x[j] += 1
                p = -round(p * (nu[j] + 1 - x[j]) / x[j])
                q -= (2 * (nu2 - x) * V[:, j] .+ V[j, j])[1]
                break
            else
                x[j] = 0
                p = isodd(nu[j]) ? -p : p
                q += (2 * nu[j] * (nu2 - x) * V[:, j] .- nu[j]^2 * V[j, j])[1]
            end
        end
    end

    return y / prod(1:s2)
end


function multiplicate(p::Int, order::Int)
    # precompute p powers
    pⁿ = [p^i for i in 0:order-1]

    DP = spzeros(Bool, p^order, prod(p - 1 .+ (1:order)) ÷ factorial(order))

    binom_p_ord = binomial(p + order - 1, order)

    # Initialize index and binomial arrays
    indexes = ones(Int, order)  # Vector to hold current indexes
    binomials = zeros(Int, order)  # Vector to hold binomial values

    # Helper function to handle the nested loops
    function loop(level::Int)
        for i=1:p
            indexes[level] = i
            binomials[level] = binomial(p + level - 1 - i, level)

            if level < order  # If not at innermost loop yet, continue nesting
                loop(level + 1)
            else  # At innermost loop, perform calculation
                n = sum((indexes[k] - 1) * pⁿ[k] for k in 1:order)
                m = binom_p_ord - sum(binomials[k] for k in 1:order)
                DP[n+1, m] = 1  # Arrays are 1-indexed in Julia
            end
        end
    end

    loop(1)  # Start the recursive loop

    return DP
end


function generateSumVectors(vectorLength::Int, totalSum::Int)
    # Base case: if vectorLength is 1, return totalSum
    if vectorLength == 1
        return [totalSum]
    end

    # Recursive case: generate all possible vectors for smaller values of vectorLength and totalSum
    return [[currentInt; smallerVector...]' for currentInt in totalSum:-1:0 for smallerVector in generateSumVectors(vectorLength-1, totalSum-currentInt)]
end


function match_pattern(strings::Union{Set,Vector}, pattern::Regex)
    return filter(r -> match(pattern, string(r)) !== nothing, strings)
end


function count_ops(expr)
    op_count = 0
    postwalk(x -> begin
        if x isa Expr && x.head == :call
            op_count += 1
        end
        x
    end, expr)
    return op_count
end

# try: run optim only if there is a violation / capture case with small shocks and set them to zero
function parse_occasionally_binding_constraints(equations_block; max_obc_horizon::Int = 40, avoid_solve::Bool = false)
    # precision_factor = 1e  #factor to force the optimiser to have non-relevatn shocks at zero

    eqs = []
    obc_shocks = Expr[]

    for arg in equations_block.args
        if isa(arg,Expr)
            if check_for_minmax(arg)
                arg_trans = transform_obc(arg)
            else
                arg_trans = arg
            end

            eq = postwalk(x -> 
                    x isa Expr ?
                        x.head == :call ? 
                            x.args[1] == :max ?
                                begin

                                    obc_vars_left = Expr(:ref, Meta.parse("χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝˡ" ), 0)
                                    obc_vars_right = Expr(:ref, Meta.parse("χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝʳ" ), 0)

                                    if !(x.args[2] isa Symbol) && check_for_dynamic_variables(x.args[2])
                                        push!(eqs, :($obc_vars_left = $(x.args[2])))
                                    else
                                        obc_vars_left = x.args[2]
                                    end

                                    if !(x.args[3] isa Symbol) && check_for_dynamic_variables(x.args[3])
                                        push!(eqs, :($obc_vars_right = $(x.args[3])))
                                    else
                                        obc_vars_right = x.args[3]
                                    end

                                    obc_inequality = Expr(:ref, Meta.parse("Χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ" ), 0)

                                    push!(eqs, :($obc_inequality = $(Expr(x.head, x.args[1], obc_vars_left, obc_vars_right))))

                                    obc_shock = Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0)

                                    push!(obc_shocks, obc_shock)

                                    :($obc_inequality - $obc_shock)
                                end :
                            x.args[1] == :min ?
                                begin
                                    obc_vars_left = Expr(:ref, Meta.parse("χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝˡ" ), 0)
                                    obc_vars_right = Expr(:ref, Meta.parse("χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝʳ" ), 0)

                                    if !(x.args[2] isa Symbol) && check_for_dynamic_variables(x.args[2])
                                        push!(eqs, :($obc_vars_left = $(x.args[2])))
                                    else
                                        obc_vars_left = x.args[2]
                                    end

                                    if !(x.args[3] isa Symbol) && check_for_dynamic_variables(x.args[3])
                                        push!(eqs, :($obc_vars_right = $(x.args[3])))
                                    else
                                        obc_vars_right = x.args[3]
                                    end

                                    obc_inequality = Expr(:ref, Meta.parse("Χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ" ), 0)

                                    push!(eqs, :($obc_inequality = $(Expr(x.head, x.args[1], obc_vars_left, obc_vars_right))))

                                    obc_shock = Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0)

                                    push!(obc_shocks, obc_shock)

                                    :($obc_inequality - $obc_shock)
                                end :
                            x :
                        x :
                    x,
            arg_trans)

            push!(eqs, eq)
        end
    end

    for obc in obc_shocks
        # push!(eqs, :($(obc) = $(Expr(:ref, obc.args[1], -1)) * 0.3 + $(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(max_obc_horizon)) * "⁾"), 0))))
        push!(eqs, :($(obc) = $(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(max_obc_horizon)) * "⁾"), 0))))

        push!(eqs, :($(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻⁰⁾"), 0)) = activeᵒᵇᶜshocks * $(Expr(:ref, Meta.parse(string(obc.args[1]) * "⁽" * super(string(max_obc_horizon)) * "⁾"), :x))))

        for i in 1:max_obc_horizon
            push!(eqs, :($(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(i)) * "⁾"), 0)) = $(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(i-1)) * "⁾"), -1)) + activeᵒᵇᶜshocks * $(Expr(:ref, Meta.parse(string(obc.args[1]) * "⁽" * super(string(max_obc_horizon-i)) * "⁾"), :x))))
        end
    end

    return Expr(:block, eqs...)
end



function get_relevant_steady_states(𝓂::ℳ, algorithm::Symbol)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    full_NSSS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    full_NSSS[indexin(𝓂.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

    if any(x -> contains(string(x), "◖"), full_NSSS)
        full_NSSS_decomposed = decompose_name.(full_NSSS)
        full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
    end

    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, return_variables_only = true, derivatives = false)

    reference_steady_state = [s ∈ 𝓂.exo_present ? 0 : relevant_SS(s) for s in full_NSSS]

    relevant_NSSS = get_steady_state(𝓂, algorithm = :first_order, return_variables_only = true, derivatives = false)

    NSSS = [s ∈ 𝓂.exo_present ? 0 : relevant_NSSS(s) for s in full_NSSS]

    SSS_delta = NSSS - reference_steady_state

    return reference_steady_state, NSSS, SSS_delta
end

# compatibility with SymPy
Max = max
Min = min

function simplify(ex::Expr)
    ex_ss = convert_to_ss_equation(ex)

    for x in get_symbols(ex_ss)
	    eval(:($x = SPyPyC.symbols($(string(x)), real = true, finite = true)))
    end

	parsed = ex_ss |> eval |> string |> Meta.parse

    postwalk(x ->   x isa Expr ? 
                        x.args[1] == :conjugate ? 
                            x.args[2] : 
                        x : 
                    x, parsed)
end

function convert_to_ss_equation(eq::Expr)
    postwalk(x -> 
        x isa Expr ? 
            x.head == :(=) ? 
                Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                    x.head == :ref ?
                        occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 :
                x.args[1] : 
            x.head == :call ?
                x.args[1] == :* ?
                    x.args[2] isa Int ?
                        x.args[3] isa Int ?
                            x :
                        :($(x.args[3]) * $(x.args[2])) : # avoid 2X syntax. doesnt work with sympy
                    x :
                x :
            unblock(x) : 
        x,
    eq)
end


function replace_indices_inside_for_loop(exxpr,index_variable,indices,concatenate, operator)
    @assert operator ∈ [:+,:*] "Only :+ and :* allowed as operators in for loops."
    calls = []
    indices = indices.args[1] == :(:) ? eval(indices) : [indices.args...]
    for idx in indices
        push!(calls, postwalk(x -> begin
            x isa Expr ?
                x.head == :ref ?
                    @capture(x, name_{index_}[time_]) ?
                        index == index_variable ?
                            :($(Expr(:ref, Symbol(string(name) * "◖" * string(idx) * "◗"),time))) :
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ get_symbols(time) ?
                                :($(Expr(:ref, Expr(:curly,name,index), Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            x :
                        x :
                    @capture(x, name_[time_]) ?
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ get_symbols(time) ?
                                :($(Expr(:ref, name, Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            x :
                        x :
                    x :
                @capture(x, name_{index_}) ?
                    index == index_variable ?
                        :($(Symbol(string(name) * "◖" * string(idx) * "◗"))) :
                    x :
                x :
            @capture(x, name_) ?
                name == index_variable && idx isa Int ?
                    :($idx) :
                x :
            x
        end,
        exxpr))
    end

    if concatenate
        return :($(Expr(:call, operator, calls...)))
    else
        return calls
    end
end


replace_indices(x::Symbol) = x

replace_indices(x::String) = Symbol(replace(x, "{" => "◖", "}" => "◗"))

replace_indices_in_symbol(x::Symbol) = replace(string(x), "◖" => "{", "◗" => "}")

function replace_indices(exxpr::Expr)
    postwalk(x -> begin
        @capture(x, name_{index_}) ?
            :($(Symbol(string(name) * "◖" * string((index)) * "◗"))) :
        x
        end,
    exxpr)
end


function write_out_for_loops(arg::Expr)
    postwalk(x -> begin
                    x = unblock(x)
                    x isa Expr ?
                        x.head == :for ?
                            x.args[2] isa Array ?
                                length(x.args[2]) >= 1 ?
                                    x.args[1].head == :block ?
                                        [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] :
                                    [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] :
                                x :
                            x.args[2].head ∉ [:(=), :block] ?
                                x.args[1].head == :block ?
                                    replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[2].args[1]), 
                                                        (x.args[1].args[2].args[2]),
                                                        true,
                                                        x.args[1].args[1].args[2].value) : # for loop part of equation
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    (x.args[1].args[2]),
                                                    true,
                                                    :+) : # for loop part of equation
                            x.args[1].head == :block ?
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[2].args[1]), 
                                                    (x.args[1].args[2].args[2]),
                                                    false,
                                                    x.args[1].args[1].args[2].value) : # for loop part of equation
                            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                Symbol(x.args[1].args[1]), 
                                                (x.args[1].args[2]),
                                                false,
                                                :+) :
                        x :
                    x
                end,
    arg)
end


function parse_for_loops(equations_block)
    eqs = Expr[]
    for arg in equations_block.args
        if isa(arg,Expr)
            parsed_eqs = write_out_for_loops(arg)
            if parsed_eqs isa Expr
                push!(eqs,unblock(replace_indices(parsed_eqs)))
            elseif parsed_eqs isa Array
                for B in parsed_eqs
                    if B isa Array
                        for b in B
                            push!(eqs,unblock(replace_indices(b)))
                        end
                    elseif B isa Expr
                        if B.head == :block
                            for b in B.args
                                if b isa Expr
                                    push!(eqs,replace_indices(b))
                                end
                            end
                        else
                            push!(eqs,unblock(replace_indices(B)))
                        end
                    else
                        push!(eqs,unblock(replace_indices(B)))
                    end
                end
            end

        end
    end
    return Expr(:block,eqs...)
end



function decompose_name(name::Symbol)
    name = string(name)
    matches = eachmatch(r"◖([\p{L}\p{N}]+)◗|([\p{L}\p{N}]+[^◖◗]*)", name)

    result = []
    nested = []

    for m in matches
        if m.captures[1] !== nothing
            push!(nested, m.captures[1])
        else
            if !isempty(nested)
                push!(result, Symbol.(nested))
                nested = []
            end
            push!(result, Symbol(m.captures[2]))
        end
    end

    if !isempty(nested)
        push!(result, (nested))
    end

    return result
end



function get_possible_indices_for_name(name::Symbol, all_names::Vector{Symbol})
    indices = filter(x -> length(x) < 3 && x[1] == name, decompose_name.(all_names))

    indexset = []

    for i in indices
        if length(i) > 1
            push!(indexset, Symbol.(i[2])...)
        end
    end

    return indexset
end



function expand_calibration_equations(calibration_equation_parameters::Vector{Symbol}, calibration_equations::Vector{Expr}, ss_calib_list::Vector, par_calib_list::Vector, all_names::Vector{Symbol})
    expanded_parameters = Symbol[]
    expanded_equations = Expr[]
    expanded_ss_var_list = []
    expanded_par_var_list = []

    for (u,par) in enumerate(calibration_equation_parameters)
        indices_in_calibration_equation = Set()
        indexed_names = []
        for i in get_symbols(calibration_equations[u])
            indices = get_possible_indices_for_name(i, all_names)
            if indices != Any[]
                push!(indices_in_calibration_equation, indices)
                push!(indexed_names,i)
            end
        end

        par_indices = get_possible_indices_for_name(par, all_names)
        
        if length(par_indices) > 0
            push!(indices_in_calibration_equation, par_indices)
        end
        
        @assert length(indices_in_calibration_equation) <= 1 "Calibration equations cannot have more than one index in the equations or for the parameter."
        
        if length(indices_in_calibration_equation) == 0
            push!(expanded_parameters,par)
            push!(expanded_equations,calibration_equations[u])
            push!(expanded_ss_var_list,ss_calib_list[u])
            push!(expanded_par_var_list,par_calib_list[u])
        else
            for i in collect(indices_in_calibration_equation)[1]
                expanded_ss_var = Set()
                expanded_par_var = Set()
                push!(expanded_parameters, Symbol(string(par) * "◖" * string(i) * "◗"))
                push!(expanded_equations, postwalk(x -> x ∈ indexed_names ? Symbol(string(x) * "◖" * string(i) * "◗") : x, calibration_equations[u]))
                for ss in ss_calib_list[u]
                    if ss ∈ indexed_names
                        push!(expanded_ss_var,Symbol(string(ss) * "◖" * string(i) * "◗"))
                    else
                        push!(expanded_ss_var,ss)
                        push!(expanded_par_var,par_calib_list[u])
                    end
                end
                push!(expanded_ss_var_list, expanded_ss_var)
                push!(expanded_par_var_list, expanded_par_var)
            end
        end
    end

    return expanded_parameters, expanded_equations, expanded_ss_var_list, expanded_par_var_list
end



function expand_indices(compressed_inputs::Vector{Symbol}, compressed_values::Vector{T}, expanded_list::Vector{Symbol}) where T
    expanded_inputs = Symbol[]
    expanded_values = T[]

    for (i,par) in enumerate(compressed_inputs)
        par_idx = findall(x -> string(par) == x, first.(split.(string.(expanded_list ), "◖")))

        if length(par_idx) > 1
            for idx in par_idx
                push!(expanded_inputs, expanded_list[idx])
                push!(expanded_values, compressed_values[i])
            end
        else#if par ∈ expanded_list ## breaks parameters defind in parameter block
            push!(expanded_inputs, par)
            push!(expanded_values, compressed_values[i])
        end
    end
    return expanded_inputs, expanded_values
end


function minmax!(x::Vector{Float64},lb::Vector{Float64},ub::Vector{Float64})
    @inbounds for i in eachindex(x)
        x[i] = max(lb[i], min(x[i], ub[i]))
    end
end



# transformation of NSSS problem
function transform(x::Vector{T}, option::Int, shift::AbstractFloat) where T <: Real
    if option == 4
        return asinh.(asinh.(asinh.(asinh.(x .+ shift))))
    elseif option == 3
        return asinh.(asinh.(asinh.(x .+ shift)))
    elseif option == 2
        return asinh.(asinh.(x .+ shift))
    elseif option == 1
        return asinh.(x .+ shift)
    elseif option == 0
        return x .+ shift
    end
end

function transform(x::Vector{T}, option::Int) where T <: Real
    if option == 4
        return asinh.(asinh.(asinh.(asinh.(x))))
    elseif option == 3
        return asinh.(asinh.(asinh.(x)))
    elseif option == 2
        return asinh.(asinh.(x))
    elseif option == 1
        return asinh.(x)
    elseif option == 0
        return x
    end
end

function undo_transform(x::Vector{T}, option::Int, shift::AbstractFloat) where T <: Real
    if option == 4
        return sinh.(sinh.(sinh.(sinh.(x)))) .- shift
    elseif option == 3
        return sinh.(sinh.(sinh.(x))) .- shift
    elseif option == 2
        return sinh.(sinh.(x)) .- shift
    elseif option == 1
        return sinh.(x) .- shift
    elseif option == 0
        return x .- shift
    end
end

function undo_transform(x::Vector{T}, option::Int) where T <: Real
    if option == 4
        return sinh.(sinh.(sinh.(sinh.(x))))
    elseif option == 3
        return sinh.(sinh.(sinh.(x)))
    elseif option == 2
        return sinh.(sinh.(x))
    elseif option == 1
        return sinh.(x)
    elseif option == 0
        return x
    end
end


function levenberg_marquardt(f::Function, 
    initial_guess::Array{T,1}, 
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1},
    parameters::solver_parameters
    ) where {T <: AbstractFloat}
    # issues with optimization: https://www.gurobi.com/documentation/8.1/refman/numerics_gurobi_guidelines.html

    xtol = parameters.xtol
    ftol = parameters.ftol
    rel_xtol = parameters.rel_xtol
    iterations = parameters.iterations
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

    max_linesearch_iterations = 1000

    function f̂(x) 
        f(undo_transform(x,transformation_level))  
        # f(undo_transform(x,transformation_level,shift))  
    end

    upper_bounds  = transform(upper_bounds,transformation_level)
    # upper_bounds  = transform(upper_bounds,transformation_level,shift)
    lower_bounds  = transform(lower_bounds,transformation_level)
    # lower_bounds  = transform(lower_bounds,transformation_level,shift)

    current_guess = copy(transform(initial_guess,transformation_level))
    # current_guess = copy(transform(initial_guess,transformation_level,shift))
    previous_guess = similar(current_guess)
    guess_update = similar(current_guess)

    ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))
    ∇̂ = similar(∇)

    prep = 𝒟.prepare_jacobian(f̂, backend, current_guess)

    largest_step = zero(T)
    largest_residual = zero(T)

    μ¹ = μ̄¹
    μ² = μ̄²

    p¹ = p̄¹
    p² = p̄²

	for iter in 1:iterations
        # make the jacobian and f calls nonallocating
        𝒟.jacobian!(f̂, ∇, backend, current_guess, prep)

        previous_guess .= current_guess

        # ∇̂ .= ∇' * ∇
        ℒ.mul!(∇̂, ∇', ∇)

        μ¹s = μ¹ * sum(abs2, f̂(current_guess))^p¹

        for i in 1:size(∇̂,1)
            ∇̂[i,i] += μ¹s + μ² * ∇̂[i,i]^p²
        end
        # ∇̂ .+= μ¹ * sum(abs2, f̂(current_guess))^p¹ * ℒ.I + μ² * ℒ.Diagonal(∇̂).^p²

        if !all(isfinite,∇̂)
            return undo_transform(current_guess,transformation_level), (iter, Inf, Inf, upper_bounds)
            # return undo_transform(current_guess,transformation_level,shift), (iter, Inf, Inf, upper_bounds)
        end

        ∇̄ = ℒ.cholesky!(∇̂, check = false)

        if !ℒ.issuccess(∇̄)
            return undo_transform(current_guess,transformation_level), (iter, Inf, Inf, upper_bounds)
            # ∇̄ = ℒ.svd(∇̂)
        end

        ℒ.mul!(guess_update, ∇', f̂(current_guess))
        ℒ.ldiv!(∇̄, guess_update)
        ℒ.axpy!(-1, guess_update, current_guess)
        # current_guess .-= ∇̄ \ ∇' * f̂(current_guess)

        minmax!(current_guess, lower_bounds, upper_bounds)

        P = sum(abs2, f̂(previous_guess))
        P̃ = P

        P̋ = sum(abs2, f̂(current_guess))

        α = 1.0
        ᾱ = 1.0

        ν̂ = ν

        guess_update .= current_guess - previous_guess
        g = f̂(previous_guess)' * ∇ * guess_update
        U = sum(abs2,guess_update)

        if P̋ > ρ * P 
            linesearch_iterations = 0
            while P̋ > (1 + ν̂ - ρ¹ * α^2) * P̃ + ρ² * α^2 * g - ρ³ * α^2 * U && linesearch_iterations < max_linesearch_iterations
                if backtracking_order == 2
                    # Quadratic backtracking line search
                    α̂ = -g * α^2 / (2 * (P̋ - P̃ - g * α))
                elseif backtracking_order == 3
                    # Cubic backtracking line search
                    a = (ᾱ^2 * (P̋ - P̃ - g * α) - α^2 * (P - P̃ - g * ᾱ)) / (ᾱ^2 * α^2 * (α - ᾱ))
                    b = (α^3 * (P - P̃ - g * ᾱ) - ᾱ^3 * (P̋ - P̃ - g * α)) / (ᾱ^2 * α^2 * (α - ᾱ))

                    if isapprox(a, zero(a), atol=eps())
                        α̂ = g / (2 * b)
                    else
                        # discriminant
                        d = max(b^2 - 3 * a * g, 0)
                        # quadratic equation root
                        α̂ = (sqrt(d) - b) / (3 * a)
                    end

                    ᾱ = α
                end

                α̂ = min(α̂, ϕ̄ * α)
                α = max(α̂, ϕ̂ * α)
                
                copy!(current_guess, previous_guess)
                ℒ.axpy!(α, guess_update, current_guess)
                # current_guess .= previous_guess + α * guess_update
                minmax!(current_guess, lower_bounds, upper_bounds)
                
                P = P̋

                P̋ = sum(abs2,f̂(current_guess))

                ν̂ *= α

                linesearch_iterations += 1
            end

            μ¹ *= λ̅¹
            μ² *= λ̅²

            p¹ *= λ̂̅¹
            p² *= λ̂̅²
        else
            μ¹ = min(μ¹ / λ¹, μ̄¹)
            μ² = min(μ² / λ², μ̄²)

            p¹ = min(p¹ / λ̂¹, p̄¹)
            p² = min(p² / λ̂², p̄²)
        end

        largest_step = maximum(abs, previous_guess - current_guess)
        largest_relative_step = maximum(abs, (previous_guess - current_guess) ./ previous_guess)
        largest_residual = maximum(abs, f(undo_transform(current_guess,transformation_level)))
        # largest_residual = maximum(abs, f(undo_transform(current_guess,transformation_level,shift)))

        if largest_step <= xtol || largest_residual <= ftol || largest_relative_step <= rel_xtol
            return undo_transform(current_guess,transformation_level), (iter, largest_step, largest_residual, f(undo_transform(current_guess,transformation_level)))
            # return undo_transform(current_guess,transformation_level,shift), (iter, largest_step, largest_residual, f(undo_transform(current_guess,transformation_level,shift)))
        end
    end

    best_guess = undo_transform(current_guess,transformation_level)
    # best_guess = undo_transform(current_guess,transformation_level,shift)

    return best_guess, (iterations, largest_step, largest_residual, f(best_guess))
end


function expand_steady_state(SS_and_pars::Vector{M}, 𝓂::ℳ) where M
    all_variables = @ignore_derivatives sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    ignore_derivatives() do
        all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    end

    NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    X = zeros(Int, length(all_variables), length(SS_and_pars))

    ignore_derivatives() do
        for (i,s) in enumerate(all_variables)
            idx = indexin([s],NSSS_labels)
            X[i,idx...] = 1
        end
    end
    
    return X * SS_and_pars
end




# function add_auxilliary_variables_to_steady_state(SS_and_pars::Vector{Float64},𝓂::ℳ)
#     all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

#     all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
#     vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))

#     [SS_and_pars[indexin([s],vars_in_ss_equations)...] for s in all_variables]
# end


function create_symbols_eqs!(𝓂::ℳ)
    # create symbols in module scope
    symbols_in_dynamic_equations = reduce(union,get_symbols.(𝓂.dyn_equations))

    symbols_in_dynamic_equations_wo_subscripts = Symbol.(replace.(string.(symbols_in_dynamic_equations),r"₍₋?(₀|₁|ₛₛ|ₓ)₎$"=>""))

    symbols_in_ss_equations = reduce(union,get_symbols.(𝓂.ss_aux_equations))

    symbols_in_equation = union(𝓂.parameters_in_equations,𝓂.parameters,𝓂.parameters_as_function_of_parameters,symbols_in_dynamic_equations,symbols_in_dynamic_equations_wo_subscripts,symbols_in_ss_equations)#,𝓂.dynamic_variables_future)

    symbols_pos = []
    symbols_neg = []
    symbols_none = []

    for symb in symbols_in_equation
        if haskey(𝓂.bounds, symb)
            if 𝓂.bounds[symb][1] >= 0
                push!(symbols_pos, symb)
            elseif 𝓂.bounds[symb][2] <= 0
                push!(symbols_neg, symb)
            else 
                push!(symbols_none, symb)
            end
        else
            push!(symbols_none, symb)
        end
    end

    for pos in symbols_pos
        eval(:($pos = SPyPyC.symbols($(string(pos)), real = true, finite = true, positive = true)))
    end
    for neg in symbols_neg
        eval(:($neg = SPyPyC.symbols($(string(neg)), real = true, finite = true, negative = true)))
    end
    for none in symbols_none
        eval(:($none = SPyPyC.symbols($(string(none)), real = true, finite = true)))
    end

    symbolics(map(x->eval(:($x)),𝓂.ss_aux_equations),
                map(x->eval(:($x)),𝓂.dyn_equations),
                # map(x->eval(:($x)),𝓂.dyn_equations_future),

                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_present_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_past_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift_var_future_list),

                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_shift2_var_past_list),

                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_present_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_past_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_var_future_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_ss_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_list),

                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_future_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_present_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dyn_exo_past_list),

                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_future_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_present_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.dyn_past_list),

                map(x->Set(eval(:([$(x...)]))),𝓂.var_present_list_aux_SS),
                map(x->Set(eval(:([$(x...)]))),𝓂.var_past_list_aux_SS),
                map(x->Set(eval(:([$(x...)]))),𝓂.var_future_list_aux_SS),
                map(x->Set(eval(:([$(x...)]))),𝓂.ss_list_aux_SS),

                map(x->Set(eval(:([$(x...)]))),𝓂.var_list_aux_SS),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dynamic_variables_list),
                # map(x->Set(eval(:([$(x...)]))),𝓂.dynamic_variables_future_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.par_list_aux_SS),

                map(x->eval(:($x)),𝓂.calibration_equations),
                map(x->eval(:($x)),𝓂.calibration_equations_parameters),
                # map(x->eval(:($x)),𝓂.parameters),

                # Set(eval(:([$(𝓂.var_present...)]))),
                # Set(eval(:([$(𝓂.var_past...)]))),
                # Set(eval(:([$(𝓂.var_future...)]))),
                Set(eval(:([$(𝓂.vars_in_ss_equations...)]))),
                Set(eval(:([$(𝓂.var...)]))),
                Set(eval(:([$(𝓂.➕_vars...)]))),

                map(x->Set(eval(:([$(x...)]))),𝓂.ss_calib_list),
                map(x->Set(eval(:([$(x...)]))),𝓂.par_calib_list),

                [Set() for _ in 1:length(𝓂.ss_aux_equations)],
                # [Set() for _ in 1:length(𝓂.calibration_equations)],
                # [Set() for _ in 1:length(𝓂.ss_aux_equations)],
                # [Set() for _ in 1:length(𝓂.calibration_equations)]
                )
end



function remove_redundant_SS_vars!(𝓂::ℳ, Symbolics::symbolics; avoid_solve::Bool = false)
    ss_equations = Symbolics.ss_equations

    # check variables which appear in two time periods. they might be redundant in steady state
    redundant_vars = intersect.(
        union.(
            intersect.(Symbolics.var_future_list,Symbolics.var_present_list),
            intersect.(Symbolics.var_future_list,Symbolics.var_past_list),
            intersect.(Symbolics.var_present_list,Symbolics.var_past_list),
            intersect.(Symbolics.ss_list,Symbolics.var_present_list),
            intersect.(Symbolics.ss_list,Symbolics.var_past_list),
            intersect.(Symbolics.ss_list,Symbolics.var_future_list)
        ),
    Symbolics.var_list)

    redundant_idx = getindex(1:length(redundant_vars), (length.(redundant_vars) .> 0) .& (length.(Symbolics.var_list) .> 1))

    for i in redundant_idx
        for var_to_solve_for in redundant_vars[i]            
            if avoid_solve && count_ops(Meta.parse(string(ss_equations[i]))) > 15
                soll = nothing
            else
                soll = try SPyPyC.solve(ss_equations[i],var_to_solve_for)
                catch
                end
            end

            if isnothing(soll)
                continue
            end
            
            if length(soll) == 0 || soll == SPyPyC.Sym[0] # take out variable if it is redundant from that euation only
                push!(Symbolics.var_redundant_list[i],var_to_solve_for)
                ss_equations[i] = ss_equations[i].subs(var_to_solve_for,1).replace(SPyPyC.Sym(ℯ),exp(1)) # replace euler constant as it is not translated to julia properly
            end

        end
    end

end


function write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)
    # ➕_vars = Symbol[]
    unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()

    vars_to_exclude = [vcat(Symbol.(vars_to_solve), 𝓂.➕_vars),Symbol[]]

    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_rebust_to_domain_errors(Meta.parse.(string.(eqs_to_solve)), vars_to_exclude, 𝓂.bounds, 𝓂.➕_vars, unique_➕_eqs)


    push!(𝓂.solved_vars, Symbol.(vars_to_solve))
    push!(𝓂.solved_vals, rewritten_eqs)


    syms_in_eqs = Set{Symbol}()

    for i in vcat(ss_and_aux_equations_dep, ss_and_aux_equations, rewritten_eqs)
        push!(syms_in_eqs, get_symbols(i)...)
    end

    setdiff!(syms_in_eqs,𝓂.➕_vars)

    syms_in_eqs2 = Set{Symbol}()

    for i in ss_and_aux_equations
        push!(syms_in_eqs2, get_symbols(i)...)
    end

    ➕_vars_alread_in_eqs = intersect(𝓂.➕_vars,reduce(union,get_symbols.(Meta.parse.(string.(eqs_to_solve)))))

    union!(syms_in_eqs, intersect(union(➕_vars_alread_in_eqs, syms_in_eqs2), 𝓂.➕_vars))

    push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

    calib_pars = Expr[]
    calib_pars_input = Symbol[]
    relevant_pars = union(intersect(reduce(union, vcat(𝓂.par_list_aux_SS, 𝓂.par_calib_list)[eq_idx_in_block_to_solve]), syms_in_eqs),intersect(syms_in_eqs, 𝓂.➕_vars))
    
    union!(relevant_pars_across, relevant_pars)

    iii = 1
    for parss in union(𝓂.parameters, 𝓂.parameters_as_function_of_parameters)
        if :($parss) ∈ relevant_pars
            push!(calib_pars, :($parss = parameters_and_solved_vars[$iii]))
            push!(calib_pars_input, :($parss))
            iii += 1
        end
    end

    guess = Expr[]
    result = Expr[]

    sorted_vars = sort(Symbol.(vars_to_solve))

    for (i, parss) in enumerate(sorted_vars) 
        push!(guess,:($parss = guess[$i]))
        push!(result,:($parss = sol[$i]))
    end

    # separate out auxilliary variables (nonnegativity)
    # nnaux = []
    # nnaux_linear = []
    # nnaux_error = []
    # push!(nnaux_error, :(aux_error = 0))
    solved_vals = Expr[]
    partially_solved_block = Expr[]

    other_vrs_eliminated_by_sympy = Set{Symbol}()

    for (i,val) in enumerate(𝓂.solved_vals[end])
        if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
            val = vcat(𝓂.ss_aux_equations, 𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
            # push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
            push!(other_vrs_eliminated_by_sympy, val.args[2])
            # push!(nnaux_linear,:($val))
            # push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
        end
    end



    for (i,val) in enumerate(rewritten_eqs)
        push!(solved_vals, postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
    end

    # if length(nnaux) > 1
    #     all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

    #     nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
        
    #     inc_matrix = fill(0,length(all_symbols),length(all_symbols))

    #     for i in 1:length(all_symbols)
    #         for k in 1:length(nn_symbols)
    #             inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
    #         end
    #     end

    #     QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

    #     nnaux = nnaux[QQ]
    #     nnaux_linear = nnaux_linear[QQ]
    # end

    other_vars = Expr[]
    other_vars_input = Symbol[]
    other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars),
                                        sort(𝓂.solved_vars[end]) ),
                                union(syms_in_eqs, other_vrs_eliminated_by_sympy ) )
                                # union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

    for var in other_vrs
        push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
        push!(other_vars_input,:($(var)))
        iii += 1
    end

    # solved_vals[end] = Expr(:call, :+, solved_vals[end], ss_and_aux_equations_error_dep...)

    funcs = :(function block(parameters_and_solved_vars::Vector, guess::Vector)
            $(guess...) 
            $(calib_pars...) # add those variables which were previously solved and are used in the equations
            $(other_vars...) # take only those that appear in equations - DONE

            $(ss_and_aux_equations_dep...)
            # return [$(solved_vals...),$(nnaux_linear...)]
            return [$(solved_vals...)]
        end)

    push!(NSSS_solver_cache_init_tmp, [haskey(𝓂.guess, v) ? 𝓂.guess[v] : Inf for v in sorted_vars])
    push!(NSSS_solver_cache_init_tmp, [Inf])

    # WARNING: infinite bounds are transformed to 1e12
    lbs = Float64[]
    ubs = Float64[]

    limit_boundaries = 1e12

    for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
        if haskey(𝓂.bounds,i)
            push!(lbs,𝓂.bounds[i][1])
            push!(ubs,𝓂.bounds[i][2])
        else
            push!(lbs,-limit_boundaries)
            push!(ubs, limit_boundaries)
        end
    end

    push!(SS_solve_func,ss_and_aux_equations...)

    push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...), $(other_vars_input...)]))

    push!(SS_solve_func,:(lbs = [$(lbs...)]))
    push!(SS_solve_func,:(ubs = [$(ubs...)]))
            
    n_block = length(𝓂.ss_solve_blocks) + 1   
        
    push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

    push!(SS_solve_func,:(solution = block_solver(params_and_solved_vars,
                                                            $(n_block), 
                                                            𝓂.ss_solve_blocks[$(n_block)], 
                                                            # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                            # f, 
                                                            inits,
                                                            lbs, 
                                                            ubs,
                                                            solver_parameters,
                                                            # fail_fast_solvers_only = fail_fast_solvers_only,
                                                            cold_start,
                                                            verbose)))
                                                            
    push!(SS_solve_func,:(iters += solution[2][2])) 
    push!(SS_solve_func,:(solution_error += solution[2][1])) 

    if length(ss_and_aux_equations_error) > 0
        push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, ss_and_aux_equations_error...))))
    end

    push!(SS_solve_func,:(sol = solution[1]))

    push!(SS_solve_func,:($(result...)))   

    push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
    push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

    push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
end




function partial_solve(eqs_to_solve, vars_to_solve, incidence_matrix_subset; avoid_solve::Bool = false)
    for n in length(eqs_to_solve)-1:-1:2
        for eq_combo in combinations(1:length(eqs_to_solve), n)
            var_indices_to_select_from = findall([sum(incidence_matrix_subset[:,eq_combo],dims = 2)...] .> 0)

            var_indices_in_remaining_eqs = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),eq_combo)],dims = 2)...] .> 0) 

            for var_combo in combinations(var_indices_to_select_from, n)
                remaining_vars_in_remaining_eqs = setdiff(var_indices_in_remaining_eqs, var_combo)
                # println("Solving for: ",vars_to_solve[var_combo]," in: ",eqs_to_solve[eq_combo])
                if length(remaining_vars_in_remaining_eqs) == length(eqs_to_solve) - n # not sure whether this condition needs to be there. could be because if the last remaining vars not solved for in the block is not present in the remaining block he will not be able to solve it for the same reasons he wasnt able to solve the unpartitioned block 
                    if avoid_solve && count_ops(Meta.parse(string(eqs_to_solve[eq_combo]))) > 15
                        soll = nothing
                    else
                        soll = try SPyPyC.solve(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
                        catch
                        end
                    end
                    
                    if !(isnothing(soll) || length(soll) == 0)
                        soll_collected = soll isa Dict ? collect(values(soll)) : collect(soll[end])
                        
                        return (vars_to_solve[setdiff(1:length(eqs_to_solve),var_combo)],
                                vars_to_solve[var_combo],
                                eqs_to_solve[setdiff(1:length(eqs_to_solve),eq_combo)],
                                soll_collected)
                    end
                end
            end
        end
    end
end



function make_equation_rebust_to_domain_errors(eqs,#::Vector{Union{Symbol,Expr}}, 
                                                vars_to_exclude::Vector{Vector{Symbol}}, 
                                                bounds::Dict{Symbol,Tuple{Float64,Float64}}, 
                                                ➕_vars::Vector{Symbol}, 
                                                unique_➕_eqs,#::Dict{Union{Expr,Symbol},Symbol}();
                                                precompile::Bool = false)
    ss_and_aux_equations = Expr[]
    ss_and_aux_equations_dep = Expr[]
    ss_and_aux_equations_error = Expr[]
    ss_and_aux_equations_error_dep = Expr[]
    rewritten_eqs = Union{Expr,Symbol}[]
    # write down ss equations including nonnegativity auxilliary variables
    # find nonegative variables, parameters, or terms
    for eq in eqs
        if eq isa Symbol
            push!(rewritten_eqs, eq)
        elseif eq isa Expr
            rewritten_eq = postwalk(x -> 
                x isa Expr ? 
                    # x.head == :(=) ? 
                    #     Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                    #         x.head == :ref ?
                    #             occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
                    #     x : 
                    x.head == :call ?
                        x.args[1] == :* ?
                            x.args[2] isa Int ?
                                x.args[3] isa Int ?
                                    x :
                                Expr(:call, :*, x.args[3:end]..., x.args[2]) : # 2beta => beta * 2 
                            x :
                        x.args[1] ∈ [:^] ?
                            !(x.args[3] isa Int) ?
                                x.args[2] isa Symbol ? # nonnegative parameters 
                                    x.args[2] ∈ vars_to_exclude[1] ?
                                        begin
                                            bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
                                            x 
                                        end :
                                    begin
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if x.args[2] in vars_to_exclude[1]
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1e12)) : (eps(), 1e12)
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                        
                                        :($(replacement) ^ $(x.args[3]))
                                    end :
                                x.args[2] isa Float64 ?
                                    x :
                                x.args[2].head == :call ? # nonnegative expressions
                                    begin
                                        if precompile
                                            replacement = x.args[2]
                                        else
                                            replacement = simplify(x.args[2])
                                        end

                                        if !(replacement isa Int) # check if the nonnegative term is just a constant
                                            if haskey(unique_➕_eqs, x.args[2])
                                                replacement = unique_➕_eqs[x.args[2]]
                                            else
                                                if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                    push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                    push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                                else
                                                    push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                    push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                                end

                                                bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1e12)) : (eps(), 1e12)
                                                push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                                replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                                unique_➕_eqs[x.args[2]] = replacement
                                            end
                                        end

                                        :($(replacement) ^ $(x.args[3]))
                                    end :
                                x :
                            x :
                        x.args[2] isa Float64 ?
                            x :
                        x.args[1] ∈ [:log] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                x.args[2] ∈ vars_to_exclude[1] ?
                                    begin
                                        bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
                                        x 
                                    end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end

                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1e12)) : (eps(), 1e12)
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                        unique_➕_eqs[x.args[2]] = replacement
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1e12)) : (eps(), 1e12)
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:norminvcdf, :norminv, :qnorm] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                x.args[2] ∈ vars_to_exclude[1] ?
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1-eps())) : (eps(), 1 - eps())
                                    x 
                                end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1-eps(),max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1-eps(),max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end

                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1 - eps())) : (eps(), 1 - eps())
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                        unique_➕_eqs[x.args[2]] = replacement
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1-eps(),max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1-eps(),max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1 - eps())) : (eps(), 1 - eps())
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:exp] ?
                            x.args[2] isa Symbol ? # have exp terms bound so they dont go to Inf
                                x.args[2] ∈ vars_to_exclude[1] ?
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12), min(bounds[x.args[2]][2], 700)) : (-1e12, 700)
                                    x 
                                end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(700,max(-1e12,$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(700,max(-1e12,$(x.args[2]))))) 
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end
                                        
                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], -1e12), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 700)) : (-1e12, 700)
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                        unique_➕_eqs[x.args[2]] = replacement
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # have exp terms bound so they dont go to Inf
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(700,max(-1e12,$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(700,max(-1e12,$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end
                                            
                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], -1e12), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 700)) : (-1e12, 700)
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:erfcinv] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                x.args[2] ∈ vars_to_exclude[1] ?
                                    begin
                                        bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 2 - eps())) : (eps(), 2 - eps())
                                        x 
                                    end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(2-eps(),max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(2-eps(),max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end
                                        
                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 2 - eps())) : (eps(), 2 - eps())
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                        unique_➕_eqs[x.args[2]] = replacement
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(2-eps(),max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(2-eps(),max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end
                                            
                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 2 - eps())) : (eps(), 2 - eps())
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x :
                    x :
                x,
            eq)
            push!(rewritten_eqs,rewritten_eq)
        else
            @assert typeof(eq) in [Symbol, Expr]
        end
    end

    vars_to_exclude_from_block = vcat(vars_to_exclude...)

    found_new_dependecy = true

    while found_new_dependecy
        found_new_dependecy = false

        for ssauxdep in ss_and_aux_equations_dep
            push!(vars_to_exclude_from_block, ssauxdep.args[1])
        end

        for (iii, ssaux) in enumerate(ss_and_aux_equations)
            if !isempty(intersect(get_symbols(ssaux), vars_to_exclude_from_block))
                found_new_dependecy = true
                push!(vars_to_exclude_from_block, ssaux.args[1])
                push!(ss_and_aux_equations_dep, ssaux)
                push!(ss_and_aux_equations_error_dep, ss_and_aux_equations_error[iii])
                deleteat!(ss_and_aux_equations, iii)
                deleteat!(ss_and_aux_equations_error, iii)
            end
        end
    end

    return rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep
end




function write_ss_check_function!(𝓂::ℳ)
    # vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_equations)),union(𝓂.parameters_in_equations))))

    unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)

    ss_equations = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

    pars = []
    for (i, p) in enumerate(𝓂.parameters)
        push!(pars, :($p = parameters[$i]))
    end

    unknwns = []
    for (i, u) in enumerate(unknowns)
        push!(unknwns, :($u = unknowns[$i]))
    end

    solve_exp = :(function solve_SS(parameters::Vector{Real}, unknowns::Vector{Real})
        $(pars...)
        $(𝓂.calibration_equations_no_var...)
        $(unknwns...)
        return [$(ss_equations...)]
    end)

    𝓂.SS_check_func = @RuntimeGeneratedFunction(solve_exp)
end


function solve_steady_state!(𝓂::ℳ, symbolic_SS, Symbolics::symbolics; verbose::Bool = false, avoid_solve::Bool = false)
    write_ss_check_function!(𝓂)

    unknowns = union(Symbolics.vars_in_ss_equations, Symbolics.calibration_equations_parameters)

    @assert length(unknowns) <= length(Symbolics.ss_equations) + length(Symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = fill(0,length(unknowns),length(unknowns))

    eq_list = vcat(union.(setdiff.(union.(Symbolics.var_list,
                                        Symbolics.ss_list),
                                    Symbolics.var_redundant_list),
                            Symbolics.par_list),
                    union.(Symbolics.ss_calib_list,
                            Symbolics.par_calib_list))

    for i in 1:length(unknowns)
        for k in 1:length(unknowns)
            incidence_matrix[i,k] = collect(unknowns)[i] ∈ collect(eq_list)[k]
        end
    end

    Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence_matrix))
    R̂ = []
    for i in 1:n_blocks
        [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1] - 1]
    end
    push!(R̂,1)

    vars = hcat(P, R̂)'
    eqs = hcat(Q, R̂)'

    # @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations for: " * repr([collect(Symbol.(unknowns))[vars[1,eqs[1,:] .< 0]]...]) # repr([vcat(Symbolics.ss_equations,Symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
    @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant euqations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now." # repr([vcat(Symbolics.ss_equations,Symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
    
    n = n_blocks

    ss_equations = vcat(Symbolics.ss_equations,Symbolics.calibration_equations)# .|> SPyPyC.Sym
    # println(ss_equations)

    SS_solve_func = []

    atoms_in_equations = Set{Symbol}()
    atoms_in_equations_list = []
    relevant_pars_across = Symbol[]
    NSSS_solver_cache_init_tmp = []

    min_max_errors = []

    unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()

    while n > 0 
        if length(eqs[:,eqs[2,:] .== n]) == 2
            var_to_solve_for = collect(unknowns)[vars[:,vars[2,:] .== n][1]]

            eq_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1]]

            # eliminate min/max from equations if solving for variables inside min/max. set to the variable we solve for automatically
            parsed_eq_to_solve_for = eq_to_solve |> string |> Meta.parse

            minmax_fixed_eqs = postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        x.args[1] ∈ [:Max,:Min] ?
                            Symbol(var_to_solve_for) ∈ get_symbols(x.args[2]) ?
                                x.args[2] :
                            Symbol(var_to_solve_for) ∈ get_symbols(x.args[3]) ?
                                x.args[3] :
                            x :
                        x :
                    x :
                x,
            parsed_eq_to_solve_for)

            if parsed_eq_to_solve_for != minmax_fixed_eqs
                [push!(atoms_in_equations, a) for a in setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs))]
                push!(min_max_errors,:(solution_error += abs($parsed_eq_to_solve_for)))
                eq_to_solve = eval(minmax_fixed_eqs)
            end
            
            if avoid_solve && count_ops(Meta.parse(string(eq_to_solve))) > 15
                soll = nothing
            else
                soll = try SPyPyC.solve(eq_to_solve,var_to_solve_for)
                catch
                end
            end
            
            if isnothing(soll) || isempty(soll)
                println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
                
                eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

                write_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)
                # write_domain_safe_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
            elseif soll[1].is_number == true
                ss_equations = [eq.subs(var_to_solve_for,soll[1]) for eq in ss_equations]
                
                push!(𝓂.solved_vars,Symbol(var_to_solve_for))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))

                if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = max(eps(),$(𝓂.solved_vals[end]))))
                else
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                end

                push!(atoms_in_equations_list,[])
            else
                push!(𝓂.solved_vars,Symbol(var_to_solve_for))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))
                
                [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
                push!(atoms_in_equations_list, Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)),Symbol.(soll[1].atoms()))))

                if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars)
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = min(max($(𝓂.bounds[𝓂.solved_vars[end]][1]), $(𝓂.solved_vals[end])), $(𝓂.bounds[𝓂.solved_vars[end]][2]))))
                    push!(SS_solve_func,:(solution_error += $(Expr(:call,:abs, Expr(:call, :-, 𝓂.solved_vars[end], 𝓂.solved_vals[end])))))
                    unique_➕_eqs[𝓂.solved_vals[end]] = 𝓂.solved_vars[end]
                else
                    vars_to_exclude = [vcat(Symbol.(var_to_solve_for), 𝓂.➕_vars), Symbol[]]
                    
                    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_rebust_to_domain_errors([𝓂.solved_vals[end]], vars_to_exclude, 𝓂.bounds, 𝓂.➕_vars, unique_➕_eqs)
    
                    if length(vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)) > 0
                        push!(SS_solve_func,vcat(ss_and_aux_equations, ss_and_aux_equations_dep)...)
                        push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)...))))
                    end
                    
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(rewritten_eqs[1])))
                end

                if haskey(𝓂.bounds, 𝓂.solved_vars[end]) && 𝓂.solved_vars[end] ∉ 𝓂.➕_vars
                    push!(SS_solve_func,:(solution_error += abs(min(max($(𝓂.bounds[𝓂.solved_vars[end]][1]), $(𝓂.solved_vars[end])), $(𝓂.bounds[𝓂.solved_vars[end]][2])) - $(𝓂.solved_vars[end]))))
                end
            end
        else
            vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]

            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false
            
            if symbolic_SS
                if avoid_solve && count_ops(Meta.parse(string(eqs_to_solve))) > 15
                    soll = nothing
                else
                    soll = try SPyPyC.solve(eqs_to_solve,vars_to_solve)
                    catch
                    end
                end

                if isnothing(soll) || length(soll) == 0 || length(intersect((union(SPyPyC.free_symbols.(soll[1])...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
                    if verbose println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.") end

                    numerical_sol = true
                else
                    if verbose println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.") end
                    
                    atoms = reduce(union,map(x->x.atoms(),collect(soll[1])))

                    for a in atoms push!(atoms_in_equations, Symbol(a)) end
                    
                    for (k, vars) in enumerate(vars_to_solve)
                        push!(𝓂.solved_vars,Symbol(vars))
                        push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

                        push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
                        push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                    end
                end
            end

            eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

            incidence_matrix_subset = incidence_matrix[vars[:,vars[2,:] .== n][1,:], eq_idx_in_block_to_solve]

            # try symbolically and use numerical if it does not work
            if numerical_sol || !symbolic_SS
                pv = sortperm(vars_to_solve, by = Symbol)
                pe = sortperm(eqs_to_solve, by = string)

                if length(pe) > 5
                    write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)
                    # write_domain_safe_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)
                else
                    solved_system = partial_solve(eqs_to_solve[pe], vars_to_solve[pv], incidence_matrix_subset[pv,pe], avoid_solve = avoid_solve)
                    
                    # if !isnothing(solved_system) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "LambertW")) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "Heaviside")) 
                    #     write_reduced_block_solution!(𝓂, SS_solve_func, solved_system, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, 
                    #     𝓂.➕_vars, unique_➕_eqs)  
                    # else
                        write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)  
                        # write_domain_safe_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
                    # end
                end

                if !symbolic_SS && verbose
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
                end
            end
        end
        n -= 1
    end

    push!(NSSS_solver_cache_init_tmp, fill(Inf, length(𝓂.parameters)))
    push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_init_tmp)

    unknwns = Symbol.(collect(unknowns))

    parameters_only_in_par_defs = Set()
    # add parameters from parameter definitions
    if length(𝓂.calibration_equations_no_var) > 0
		atoms = reduce(union,get_symbols.(𝓂.calibration_equations_no_var))
	    [push!(atoms_in_equations, a) for a in atoms]
	    [push!(parameters_only_in_par_defs, a) for a in atoms]
	end
    
    # 𝓂.par = union(𝓂.par,setdiff(parameters_only_in_par_defs,𝓂.parameters_as_function_of_parameters))
    
    parameters_in_equations = []

    for (i, parss) in enumerate(𝓂.parameters) 
        if parss ∈ union(atoms_in_equations, relevant_pars_across)
            push!(parameters_in_equations,:($parss = parameters[$i]))
        end
    end
    
    dependencies = []
    for (i, a) in enumerate(atoms_in_equations_list)
        push!(dependencies,𝓂.solved_vars[i] => intersect(a, union(𝓂.var,𝓂.parameters)))
    end

    push!(dependencies,:SS_relevant_calibration_parameters => intersect(reduce(union,atoms_in_equations_list),𝓂.parameters))

    𝓂.SS_dependencies = dependencies
    

    
    dyn_exos = []
    for dex in union(𝓂.exo_past,𝓂.exo_future)
        push!(dyn_exos,:($dex = 0))
    end

    push!(SS_solve_func,:($(dyn_exos...)))
    
    push!(SS_solve_func, min_max_errors...)
    # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, params_scaled_flt)))
    
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [copy(params_flt)] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp...,copy(params_flt)] end))
    

    # push!(SS_solve_func,:(for pars in 𝓂.NSSS_solver_cache
    #                             latest = sqrt(sum(abs2,pars[end] - params_flt))# / max(sum(abs2,pars[end]), sum(abs,params_flt))
    #                             if latest <= current_best
    #                                 current_best = latest
    #                             end
    #                         end))

    push!(SS_solve_func,:(if (current_best > 1e-5) && (solution_error < 1e-12)
                                reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
                                # solved_scale = scale
                            end))
    # push!(SS_solve_func,:(if length(𝓂.NSSS_solver_cache) > 100 popfirst!(𝓂.NSSS_solver_cache) end))
    
    # push!(SS_solve_func,:(SS_init_guess = ([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)])))

    # push!(SS_solve_func,:(𝓂.SS_init_guess = typeof(SS_init_guess) == Vector{Float64} ? SS_init_guess : ℱ.value.(SS_init_guess)))

    # push!(SS_solve_func,:(return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]))))


    # fix parameter bounds
    par_bounds = []

    for varpar in intersect(𝓂.parameters,union(atoms_in_equations, relevant_pars_across))
        if haskey(𝓂.bounds, varpar)
            push!(par_bounds, :($varpar = min(max($varpar,$(𝓂.bounds[varpar][1])),$(𝓂.bounds[varpar][2]))))
        end
    end

    solve_exp = :(function solve_SS(parameters::Vector{Real}, 
                                    𝓂::ℳ,
                                    # fail_fast_solvers_only::Bool, 
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})

                    # params_flt = typeof(parameters) == Vector{Float64} ? parameters : ℱ.value.(parameters)
                    # current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                    # closest_solution_init = 𝓂.NSSS_solver_cache[end]
                    # for pars in 𝓂.NSSS_solver_cache
                    #     latest = sum(abs2,pars[end] - params_flt)
                    #     if latest <= current_best
                    #         current_best = latest
                    #         closest_solution_init = pars
                    #     end
                    # end
                    # solved_scale = 0
                    # range_length = [1]#fail_fast_solvers_only ? [1] : [ 1, 2, 4, 8,16,32]
                    # for r in range_length
                        # rangee = ignore_derivatives(range(0,1,r+1))
                        # for scale in rangee[2:end]
                            # if scale <= solved_scale continue end
                            params_flt = typeof(parameters) == Vector{Float64} ? parameters : ℱ.value.(parameters)
                            current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                            closest_solution = 𝓂.NSSS_solver_cache[end]
                            for pars in 𝓂.NSSS_solver_cache
                                latest = sum(abs2,pars[end] - params_flt)
                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end

                            # params = all(isfinite.(closest_solution_init[end])) && parameters != closest_solution_init[end] ? scale * parameters + (1 - scale) * closest_solution_init[end] : parameters

                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.calibration_equations_no_var...)
                            NSSS_solver_cache_tmp = []
                            solution_error = 0.0
                            iters = 0
                            $(SS_solve_func...)
                            # if scale == 1
                                # return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                # NSSS_solution = [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)]
                                # NSSS_solution[abs.(NSSS_solution) .< 1e-12] .= 0 # doesnt work with Zygote
                                return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                            # end
                        # end
                    # end
                end)

    𝓂.SS_solve_func = @RuntimeGeneratedFunction(solve_exp)
    # 𝓂.SS_solve_func = eval(solve_exp)

    return nothing
end




function solve_steady_state!(𝓂::ℳ; verbose::Bool = false)
    unknowns = union(𝓂.vars_in_ss_equations, 𝓂.calibration_equations_parameters)

    @assert length(unknowns) <= length(𝓂.ss_aux_equations) + length(𝓂.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = fill(0,length(unknowns),length(unknowns))

    eq_list = vcat(union.(union.(𝓂.var_list_aux_SS,
                                        𝓂.ss_list_aux_SS),
                            𝓂.par_list_aux_SS),
                    union.(𝓂.ss_calib_list,
                            𝓂.par_calib_list))

    for i in 1:length(unknowns)
        for k in 1:length(unknowns)
            incidence_matrix[i,k] = collect(unknowns)[i] ∈ collect(eq_list)[k]
        end
    end

    Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence_matrix))
    R̂ = []
    for i in 1:n_blocks
        [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1] - 1]
    end
    push!(R̂,1)

    vars = hcat(P, R̂)'
    eqs = hcat(Q, R̂)'

    # @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations for: " * repr([collect(Symbol.(unknowns))[vars[1,eqs[1,:] .< 0]]...]) # repr([vcat(𝓂.ss_equations,𝓂.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
    @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant euqations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now." # repr([vcat(𝓂.ss_equations,𝓂.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
    
    n = n_blocks

    ss_equations = vcat(𝓂.ss_aux_equations,𝓂.calibration_equations)

    SS_solve_func = []

    atoms_in_equations = Set{Symbol}()
    atoms_in_equations_list = []
    relevant_pars_across = []
    NSSS_solver_cache_init_tmp = []

    n_block = 1

    while n > 0
        vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]

        eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

        # try symbolically and use numerical if it does not work
        if verbose
            println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
        end
        
        push!(𝓂.solved_vars,Symbol.(vars_to_solve))
        push!(𝓂.solved_vals,Meta.parse.(string.(eqs_to_solve)))

        syms_in_eqs = Set()

        for i in eqs_to_solve
            push!(syms_in_eqs, get_symbols(i)...)
        end

        # println(syms_in_eqs)
        push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

        calib_pars = []
        calib_pars_input = []
        relevant_pars = reduce(union,vcat(𝓂.par_list_aux_SS,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
        relevant_pars_across = union(relevant_pars_across,relevant_pars)
        
        iii = 1
        for parss in union(𝓂.parameters,𝓂.parameters_as_function_of_parameters)
            # valss   = 𝓂.parameter_values[i]
            if :($parss) ∈ relevant_pars
                push!(calib_pars,:($parss = parameters_and_solved_vars[$iii]))
                push!(calib_pars_input,:($parss))
                iii += 1
            end
        end


        guess = []
        result = []
        sorted_vars = sort(𝓂.solved_vars[end])
        # sorted_vars = sort(setdiff(𝓂.solved_vars[end],𝓂.➕_vars))
        for (i, parss) in enumerate(sorted_vars) 
            push!(guess,:($parss = guess[$i]))
            # push!(guess,:($parss = undo_transformer(guess[$i])))
            push!(result,:($parss = sol[$i]))
        end

        
        # separate out auxilliary variables (nonnegativity)
        nnaux = []
        nnaux_linear = []
        nnaux_error = []
        push!(nnaux_error, :(aux_error = 0))
        solved_vals = Expr[]
        
        eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]


        other_vrs_eliminated_by_sympy = Set()

        for (i,val) in enumerate(𝓂.solved_vals[end])
            if typeof(val) ∈ [Symbol,Float64,Int]
                push!(solved_vals,val)
            else
                if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
                    val = vcat(𝓂.ss_aux_equations,𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
                    push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                    push!(other_vrs_eliminated_by_sympy, val.args[2])
                    push!(nnaux_linear,:($val))
                    push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
                else
                    push!(solved_vals,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
                end
            end
        end

        # println(other_vrs_eliminated_by_sympy)
        # sort nnaux vars so that they enter in right order. avoid using a variable before it is declared
        # println(nnaux)
        if length(nnaux) > 1
            all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

            nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
            
            inc_matrix = fill(0,length(all_symbols),length(all_symbols))

            for i in 1:length(all_symbols)
                for k in 1:length(nn_symbols)
                    inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
                end
            end

            QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

            nnaux = nnaux[QQ]
            nnaux_linear = nnaux_linear[QQ]
        end


        other_vars = []
        other_vars_input = []
        # other_vars_inverse = []
        other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars),
                                            sort(𝓂.solved_vars[end]) ),
                                union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

        for var in other_vrs
            # var_idx = findfirst(x -> x == var, union(𝓂.var,𝓂.calibration_equations_parameters))
            push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
            push!(other_vars_input,:($(var)))
            iii += 1
            # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
        end

        funcs = :(function block(parameters_and_solved_vars::Vector, guess::Vector)
                # if guess isa Tuple guess = guess[1] end
                # guess = undo_transformer(guess,lbs,ubs, option = transformer_option) 
                # println(guess)
                $(guess...) 
                $(calib_pars...) # add those variables which were previously solved and are used in the equations
                $(other_vars...) # take only those that appear in equations - DONE

                # $(aug_lag...)
                # $(nnaux...)
                # $(nnaux_linear...)
                return [$(solved_vals...),$(nnaux_linear...)]
            end)

        push!(NSSS_solver_cache_init_tmp,fill(1.205996189998029, length(sorted_vars)))
        push!(NSSS_solver_cache_init_tmp,[Inf])

        # WARNING: infinite bounds are transformed to 1e12
        lbs = []
        ubs = []
        
        limit_boundaries = 1e12

        for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
            if haskey(𝓂.bounds, i)
                push!(lbs,𝓂.bounds[i][1] == -Inf ? -limit_boundaries+rand() : 𝓂.bounds[i][1])
                push!(ubs,𝓂.bounds[i][2] ==  Inf ?  limit_boundaries-rand() : 𝓂.bounds[i][2])
            else
                push!(lbs,-limit_boundaries+rand())
                push!(ubs,limit_boundaries+rand())
            end
        end

        push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...),$(other_vars_input...)]))

        push!(SS_solve_func,:(lbs = [$(lbs...)]))
        push!(SS_solve_func,:(ubs = [$(ubs...)]))
        
        push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

        push!(SS_solve_func,:(solution = block_solver(length(params_and_solved_vars) == 0 ? [0.0] : params_and_solved_vars,
                                                                $(n_block), 
                                                                𝓂.ss_solve_blocks[$(n_block)], 
                                                                # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                                # f, 
                                                                inits,
                                                                lbs, 
                                                                ubs,
                                                                solver_parameters,
                                                                cold_start,
                                                                # fail_fast_solvers_only = fail_fast_solvers_only,
                                                                verbose)))
        
        # push!(SS_solve_func,:(solution = block_solver_RD(length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)])))#, 
        
        push!(SS_solve_func,:(iters += solution[2][2])) 
        push!(SS_solve_func,:(solution_error += solution[2][1])) 
        push!(SS_solve_func,:(sol = solution[1]))

        # push!(SS_solve_func,:(solution = block_solver_RD(length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)])))#, 
        
        # push!(SS_solve_func,:(solution_error += sum(abs2,𝓂.ss_solve_blocks[$(n_block)](length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)],solution))))

        push!(SS_solve_func,:($(result...)))   
        
        push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
        push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

        push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
        
        n_block += 1
        
        n -= 1
    end

    push!(NSSS_solver_cache_init_tmp,[Inf])
    push!(NSSS_solver_cache_init_tmp,fill(Inf,length(𝓂.parameters)))
    push!(𝓂.NSSS_solver_cache,NSSS_solver_cache_init_tmp)

    unknwns = Symbol.(collect(unknowns))

    parameters_only_in_par_defs = Set()
    # add parameters from parameter definitions
    if length(𝓂.calibration_equations_no_var) > 0
		atoms = reduce(union,get_symbols.(𝓂.calibration_equations_no_var))
	    [push!(atoms_in_equations, a) for a in atoms]
	    [push!(parameters_only_in_par_defs, a) for a in atoms]
	end
    
    # 𝓂.par = union(𝓂.par,setdiff(parameters_only_in_par_defs,𝓂.parameters_as_function_of_parameters))
    
    parameters_in_equations = []

    for (i, parss) in enumerate(𝓂.parameters) 
        if parss ∈ union(atoms_in_equations, relevant_pars_across)
            push!(parameters_in_equations,:($parss = parameters[$i]))
        end
    end
    
    dependencies = []
    for (i, a) in enumerate(atoms_in_equations_list)
        push!(dependencies,𝓂.solved_vars[i] => intersect(a, union(𝓂.var,𝓂.parameters)))
    end

    push!(dependencies,:SS_relevant_calibration_parameters => intersect(reduce(union,atoms_in_equations_list),𝓂.parameters))

    𝓂.SS_dependencies = dependencies

    
    dyn_exos = []
    for dex in union(𝓂.exo_past,𝓂.exo_future)
        push!(dyn_exos,:($dex = 0))
    end

    push!(SS_solve_func,:($(dyn_exos...)))

    # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, params_scaled_flt)))
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [copy(params_flt)] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)] end))
    
    push!(SS_solve_func,:(current_best = sqrt(sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt))))# / max(sum(abs2,𝓂.NSSS_solver_cache[end][end]), sum(abs2,params_flt))))

    push!(SS_solve_func,:(for pars in 𝓂.NSSS_solver_cache
                                latest = sqrt(sum(abs2,pars[end] - params_flt))# / max(sum(abs2,pars[end]), sum(abs,params_flt))
                                if latest <= current_best
                                    current_best = latest
                                end
                            end))

    push!(SS_solve_func,:(if (current_best > 1e-5) && (solution_error < 1e-12)
                                reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
                                # solved_scale = scale
                            end))

    # fix parameter bounds
    par_bounds = []
    
    for varpar in intersect(𝓂.parameters,union(atoms_in_equations, relevant_pars_across))
        if haskey(𝓂.bounds, varpar)
            push!(par_bounds, :($varpar = min(max($varpar,$(𝓂.bounds[varpar][1])),$(𝓂.bounds[varpar][2]))))
        end
    end

    solve_exp = :(function solve_SS(parameters::Vector{Real}, 
                                    𝓂::ℳ, 
                                    # fail_fast_solvers_only::Bool, 
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})

                    # params_flt = typeof(parameters) == Vector{Float64} ? parameters : ℱ.value.(parameters)
                    # current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                    # closest_solution_init = 𝓂.NSSS_solver_cache[end]
                    # for pars in 𝓂.NSSS_solver_cache
                    #     latest = sum(abs2,pars[end] - params_flt)
                    #     if latest <= current_best
                    #         current_best = latest
                    #         closest_solution_init = pars
                    #     end
                    # end
                    # solved_scale = 0
                    # range_length = [1]#fail_fast_solvers_only ? [1] : [ 1, 2, 4, 8,16,32]
                    # for r in range_length
                    #     rangee = ignore_derivatives(range(0,1,r+1))
                    #     for scale in rangee[2:end]
                    #         if scale <= solved_scale continue end
                            params_flt = typeof(parameters) == Vector{Float64} ? parameters : ℱ.value.(parameters)
                            current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                            closest_solution = 𝓂.NSSS_solver_cache[end]
                            for pars in 𝓂.NSSS_solver_cache
                                latest = sum(abs2,pars[end] - params_flt)
                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end
                            # params = all(isfinite.(closest_solution_init[end])) && parameters != closest_solution_init[end] ? scale * parameters + (1 - scale) * closest_solution_init[end] : parameters

                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.calibration_equations_no_var...)
                            NSSS_solver_cache_tmp = []
                            iters = 0
                            solution_error = 0.0
                            $(SS_solve_func...)
                            # if scale == 1
                                # return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)] , (solution_error, iters)
                            # end
                    #     end
                    # end
                end)

    𝓂.SS_solve_func = @RuntimeGeneratedFunction(solve_exp)
    # 𝓂.SS_solve_func = eval(solve_exp)

    return nothing
end


function reverse_diff_friendly_push!(x,y)
    @ignore_derivatives push!(x,y)
end

function calculate_SS_solver_runtime_and_loglikelihood(pars::Vector{Float64}, 𝓂::ℳ, tol::AbstractFloat = 1e-12)::Float64
    log_lik = 0.0
    log_lik -= -sum(pars[1:19])                                 # logpdf of a gamma dist with mean and variance 1
    log_lik -= -log(5 * sqrt(2 * π)) - (pars[20]^2 / (2 * 5^2)) # logpdf of a normal dist with mean = 0 and variance = 5^2

    pars[1:2] = sort(pars[1:2], rev = true)

    par_inputs = solver_parameters(eps(), eps(), eps(), 250, pars..., 1, 0.0, 2)

    runtime = @elapsed outmodel = try 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, true, [par_inputs]) catch end

    runtime = outmodel isa Tuple{Vector{Float64}, Tuple{Float64, Int64}} ? 
                    (outmodel[2][1] > tol) || !isfinite(outmodel[2][1]) ? 
                        10 : 
                    runtime : 
                10

    return log_lik / 1e4 + runtime * 1e3
end


function find_SS_solver_parameters!(𝓂::ℳ; maxtime::Int = 60, maxiter::Int = 250000, tol::AbstractFloat = 1e-12)
    pars = rand(20) .+ 1
    pars[20] -= 1

    lbs = fill(eps(), length(pars))
    lbs[20] = -20

    ubs = fill(100.0,length(pars))

    sol = Optim.optimize(x -> calculate_SS_solver_runtime_and_loglikelihood(x, 𝓂), 
                            lbs, ubs, pars, 
                            Optim.SAMIN(verbosity = 0), 
                            Optim.Options(time_limit = maxtime, iterations = maxiter))

    pars = Optim.minimizer(sol)

    par_inputs = solver_parameters(eps(), eps(), eps(), 250, pars..., 1, 0.0, 2)

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, true, [par_inputs])

    if solution_error < tol
        push!(𝓂.solver_parameters, par_inputs)
        return true
    else 
        return false
    end
end


function select_fastest_SS_solver_parameters!(𝓂::ℳ; tol::AbstractFloat = 1e-12)
    best_param = 𝓂.solver_parameters[1]

    best_time = Inf

    solved = false

    for p in 𝓂.solver_parameters
        total_time = 0.0
        
        for _ in 1:10
            start_time = time()

            SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, true, [p])

            elapsed_time = time() - start_time

            total_time += elapsed_time
            
            if solution_error > tol
                total_time = 1e7
                break
            end
        end

        if total_time < best_time
            best_time = total_time
            best_param = p
        end

        solved = true
    end

    if solved
        pushfirst!(𝓂.solver_parameters, best_param)
    end
end


function solve_ss(SS_optimizer::Function,
                    ss_solve_blocks::Function,
                    parameters_and_solved_vars::Vector{Float64},
                    closest_parameters_and_solved_vars::Vector{Float64},
                    lbs::Vector{Float64},
                    ubs::Vector{Float64},
                    tol::AbstractFloat,
                    total_iters::Int,
                    n_block::Int,
                    verbose::Bool,
                    guess::Vector{Float64},
                    solver_params::solver_parameters,
                    extended_problem::Bool,
                    separate_starting_value::Union{Bool,Float64})
    if separate_starting_value isa Float64
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], fill(separate_starting_value, length(guess))))
        sol_values_init[ubs[1:length(guess)] .<= 1] .= .1 # capture cases where part of values is small
    else
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], [g < 1e12 ? g : solver_params.starting_value for g in guess]))
    end

    if extended_problem
        function ext_function_to_optimize(guesses)
            gss = guesses[1:length(guess)]
    
            parameters_and_solved_vars_guess = guesses[length(guess)+1:end]
    
            res = ss_solve_blocks(parameters_and_solved_vars, gss)
    
            return vcat(res, parameters_and_solved_vars .- parameters_and_solved_vars_guess)
        end
    else
        function function_to_optimize(guesses) ss_solve_blocks(parameters_and_solved_vars, guesses) end
    end

    sol_new_tmp, info = SS_optimizer(   extended_problem ? ext_function_to_optimize : function_to_optimize,
                                        extended_problem ? vcat(sol_values_init, closest_parameters_and_solved_vars) : sol_values_init,
                                        extended_problem ? lbs : lbs[1:length(guess)],
                                        extended_problem ? ubs : ubs[1:length(guess)],
                                        solver_params   )

    sol_new = isnothing(sol_new_tmp) ? sol_new_tmp : sol_new_tmp[1:length(guess)]

    sol_minimum = isnan(sum(abs, info[4])) ? Inf : sum(abs, info[4])

    sol_values = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], sol_new))

    total_iters += info[1]
    
    extended_problem_str = extended_problem ? "(extended problem) " : ""

    if separate_starting_value isa Bool
        starting_value_str = ""
    else
        starting_value_str = "and starting point: $separate_starting_value"
    end

    if all(guess .< 1e12) && separate_starting_value isa Bool
        any_guess_str = "previous solution, "
    elseif any(guess .< 1e12) && separate_starting_value isa Bool
        any_guess_str = "provided guess, "
    else
        any_guess_str = ""
    end

    max_resid = maximum(abs,ss_solve_blocks(parameters_and_solved_vars, sol_values))

    if sol_minimum < tol && verbose
        println("Block: $n_block - Solved $(extended_problem_str)using ",string(SS_optimizer),", $(any_guess_str)$(starting_value_str); maximum residual = $max_resid")
    end

    return sol_values, sol_minimum
end


function block_solver(parameters_and_solved_vars::Vector{Float64}, 
                        n_block::Int, 
                        ss_solve_blocks::Function, 
                        # SS_optimizer, 
                        # f::OptimizationFunction, 
                        guess_and_pars_solved_vars::Vector{Vector{Float64}}, 
                        lbs::Vector{Float64}, 
                        ubs::Vector{Float64},
                        parameters::Vector{solver_parameters},
                        cold_start::Bool,
                        verbose::Bool;
                        tol::AbstractFloat = 1e-12 #, # eps(),
                        # timeout = 120,
                        # starting_points::Vector{Float64} = [1.205996189998029, 0.7688, 0.897, 1.2],#, 0.9, 0.75, 1.5, -0.5, 2.0, .25]
                        # fail_fast_solvers_only = true,
                        # verbose::Bool = false
                        )
    guess = guess_and_pars_solved_vars[1]

    sol_values = guess

    closest_parameters_and_solved_vars = sum(abs, guess_and_pars_solved_vars[2]) == Inf ? parameters_and_solved_vars : guess_and_pars_solved_vars[2]

    sol_minimum  = sum(abs, ss_solve_blocks(parameters_and_solved_vars, guess))
    
    if sol_minimum < tol
        if verbose
            println("Block: $n_block, - Solved using previous solution; maximum residual = ", maximum(abs, ss_solve_blocks(parameters_and_solved_vars, guess)))
        end
    end

    total_iters = 0

    SS_optimizer = levenberg_marquardt

    if cold_start
        guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only

        for g in guesses
            for p in parameters
                for ext in [true, false] # try first the system where values and parameters can vary, next try the system where only values can vary
                    if sol_minimum > tol
                        sol_values, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                            g, 
                                                            p,
                                                            ext,
                                                            false)
                    end
                end
            end
        end
    else !cold_start
        for ext in [false, true] # try first the system where only values can vary, next try the system where values and parameters can vary
            if sol_minimum > tol
                sol_values, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                    guess, 
                                                    parameters[1],
                                                    ext,
                                                    false)
            end
        end

        for p in unique(parameters) # take unique because some parameters might appear more than once
            for s in [p.starting_value, 1.206, 1.5, 2.0, 0.897, 0.7688]#, .9, .75, 1.5, -.5, 2, .25] # try first the guess and then different starting values
                # for ext in [false, true] # try first the system where only values can vary, next try the system where values and parameters can vary
                    if sol_minimum > tol
                        sol_values, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                            guess, 
                                                            p,
                                                            false,
                                                            s)
                    end
                # end
            end
        end
    end

    return sol_values, (sol_minimum, total_iters)
end


# function second_order_stochastic_steady_state_iterative_solution_forward(𝐒₁𝐒₂::SparseVector{Float64};  dims::Vector{Tuple{Int,Int}},  𝓂::ℳ, tol::AbstractFloat = eps())
#     len𝐒₁ = dims[1][1] * dims[1][2]

#     𝐒₁ = reshape(𝐒₁𝐒₂[1 : len𝐒₁],dims[1])
#     𝐒₂ = sparse(reshape(𝐒₁𝐒₂[len𝐒₁ + 1 : end],dims[2]))
        
#     state = zeros(𝓂.timings.nVars)
#     shock = zeros(𝓂.timings.nExo)

#     aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
#     1
#     shock]

#     sol = @suppress begin
#         speedmapping(state; 
#                     m! = (SSS, sss) -> begin 
#                                         aug_state .= [sss[𝓂.timings.past_not_future_and_mixed_idx]
#                                                     1
#                                                     shock]

#                                         SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
#                     end, 
#         tol = tol, maps_limit = 10000)
#     end

#     return sol.minimizer, sol.converged
# end


function calculate_second_order_stochastic_steady_state(parameters::Vector{M}, 𝓂::ℳ; verbose::Bool = false, pruning::Bool = false, sylvester_algorithm::Symbol = :doubling, tol::AbstractFloat = 1e-12)::Tuple{Vector{M}, Bool, Vector{M}, M, AbstractMatrix{M}, SparseMatrixCSC{M}, AbstractMatrix{M}, SparseMatrixCSC{M}} where M
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, verbose = verbose)
    
    all_SS = expand_steady_state(SS_and_pars,𝓂)

    if solution_error > tol || isnan(solution_error)
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix
    
    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
    if !solved
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂) * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂
    
    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, sylvester_algorithm = sylvester_algorithm, verbose = verbose)

    if !solved2
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]

    aug_state₁ = sparse([zeros(𝓂.timings.nPast_not_future_and_mixed); 1; zeros(𝓂.timings.nExo)])

    tmp = (ℒ.I - 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed])

    tmp̄ = @ignore_derivatives ℒ.lu(tmp, check = false)

    if !ℒ.issuccess(tmp̄)
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    SSSstates = tmp \ (𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[𝓂.timings.past_not_future_and_mixed_idx]

    if pruning
        state = 𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] * SSSstates + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2
        converged = true
    else
        nᵉ = 𝓂.timings.nExo

        s_in_s⁺ = @ignore_derivatives BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))

        kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
        
        A = 𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed]
        B̂ = 𝐒₂[:,kron_s⁺_s⁺]
    
        SSSstates, converged = calculate_second_order_stochastic_steady_state(Val(:Newton), 𝐒₁, 𝐒₂, SSSstates, 𝓂)
        state = A * SSSstates + B̂ * ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1)) / 2
        # state, converged = second_order_stochastic_steady_state_iterative_solution([sparsevec(𝐒₁); vec(𝐒₂)]; dims = [size(𝐒₁); size(𝐒₂)], 𝓂 = 𝓂)
    end

    # all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    # all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
    # NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
    
    # all_SS = [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
    # we need all variables for the stochastic steady state because even leads and lags have different SSS then the non-lead-lag ones (contrary to the no stochastic steady state) and we cannot recover them otherwise

    return all_SS + state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
end



function calculate_second_order_stochastic_steady_state(::Val{:Newton}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14)
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 - x)

        if i > 3 && isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    return x, isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2, x, rtol = tol)
end




function calculate_second_order_stochastic_steady_state(::Val{:Newton}, 
                                                        𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                                        𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                                        x::Vector{ℱ.Dual{Z,S,N}},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14) where {Z,S,N}

    𝐒₁̂ = ℱ.value.(𝐒₁)
    𝐒₂̂ = ℱ.value.(𝐒₂)
    x̂ = ℱ.value.(x)
    
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    A = 𝐒₁̂[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
 
    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 - x̂)

        if i > 5 && isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2, x̂, rtol = tol)
            break
        end
        
        # x̂ += Δx
        ℒ.axpy!(-1, Δx, x̂)
    end

    ∂x = zeros(length(x̂), N)
    
    for i in 1:N
        ∂𝐒₁ = ℱ.partials.(𝐒₁, i)
        ∂𝐒₂ = ℱ.partials.(𝐒₂, i)

        ∂A = ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
        ∂B̂ = ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]

        tmp = ∂A * x̂ + ∂B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2

        TMP = A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)

        ∂x[:,i] = -TMP \ tmp
    end
    
    return reshape(map(x̂, eachrow(∂x)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2, x̂, rtol = tol)
end



function rrule(::typeof(calculate_second_order_stochastic_steady_state),
                                                        ::Val{:Newton}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14)
    
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    
    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 - x)

        if i > 5 && isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    solved = isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2, x, rtol = tol)         

    # println(x)

    ∂𝐒₁ =  zero(𝐒₁)
    ∂𝐒₂ =  zero(𝐒₂)

    function second_order_stochastic_steady_state_pullback(∂x)
        S = -∂x[1]' / (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed] = S' * x'
        
        ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺] = S' * ℒ.kron(vcat(x,1), vcat(x,1))' / 2

        return NoTangent(), NoTangent(), ∂𝐒₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent()
    end

    return (x, solved), second_order_stochastic_steady_state_pullback
end




# function third_order_stochastic_steady_state_iterative_solution_forward(𝐒₁𝐒₂𝐒₃::SparseVector{Float64}; dims::Vector{Tuple{Int,Int}}, 𝓂::ℳ, tol::AbstractFloat = eps())
#     len𝐒₁ = dims[1][1] * dims[1][2]
#     len𝐒₂ = dims[2][1] * dims[2][2]

#     𝐒₁ = reshape(𝐒₁𝐒₂𝐒₃[1 : len𝐒₁],dims[1])
#     𝐒₂ = sparse(reshape(𝐒₁𝐒₂𝐒₃[len𝐒₁ .+ (1 : len𝐒₂)],dims[2]))
#     𝐒₃ = sparse(reshape(𝐒₁𝐒₂𝐒₃[len𝐒₁ + len𝐒₂ + 1 : end],dims[3]))

#     state = zeros(𝓂.timings.nVars)
#     shock = zeros(𝓂.timings.nExo)

#     aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
#     1
#     shock]

#     sol = @suppress begin
#         speedmapping(state; 
#                     m! = (SSS, sss) -> begin 
#                                         aug_state .= [sss[𝓂.timings.past_not_future_and_mixed_idx]
#                                                     1
#                                                     shock]

#                                         SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
#                     end, 
#         tol = tol, maps_limit = 10000)
#     end

#     return sol.minimizer, sol.converged
# end


function calculate_third_order_stochastic_steady_state( parameters::Vector{M}, 
                                                        𝓂::ℳ; 
                                                        verbose::Bool = false, 
                                                        pruning::Bool = false, 
                                                        sylvester_algorithm::Symbol = :doubling, 
                                                        tol::AbstractFloat = 1e-12)::Tuple{Vector{M}, Bool, Vector{M}, M, AbstractMatrix{M}, SparseMatrixCSC{M}, SparseMatrixCSC{M}, AbstractMatrix{M}, SparseMatrixCSC{M}, SparseMatrixCSC{M}} where M
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, verbose = verbose)

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    if solution_error > tol || isnan(solution_error)
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix
    
    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
    if !solved
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂) * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂
    
    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol, sylvester_algorithm = sylvester_algorithm, verbose= verbose)

    if !solved2
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂) * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃
            
    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝓂.solution.perturbation.second_order_auxilliary_matrices, 𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

    if !solved3
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]

    aug_state₁ = sparse([zeros(𝓂.timings.nPast_not_future_and_mixed); 1; zeros(𝓂.timings.nExo)])
    
    tmp = (ℒ.I - 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed])

    tmp̄ = @ignore_derivatives ℒ.lu(tmp, check = false)

    if !ℒ.issuccess(tmp̄)
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    SSSstates = tmp \ (𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[𝓂.timings.past_not_future_and_mixed_idx]

    if pruning
        state = 𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] * SSSstates + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2
        converged = true
    else
        nᵉ = 𝓂.timings.nExo

        s_in_s⁺ = @ignore_derivatives BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))

        kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
        
        kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
        
        A = 𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed]
        B̂ = 𝐒₂[:,kron_s⁺_s⁺]
        Ĉ = 𝐒₃[:,kron_s⁺_s⁺_s⁺]
    
        SSSstates, converged = calculate_third_order_stochastic_steady_state(Val(:Newton), 𝐒₁, 𝐒₂, 𝐒₃, SSSstates, 𝓂)
        
        state = A * SSSstates + B̂ * ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1)) / 2 + Ĉ * ℒ.kron(vcat(SSSstates,1),  ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1))) / 6
        # state, converged = third_order_stochastic_steady_state_iterative_solution([sparsevec(𝐒₁); vec(𝐒₂); vec(𝐒₃)]; dims = [size(𝐒₁); size(𝐒₂); size(𝐒₃)], 𝓂 = 𝓂)
        # state, converged = third_order_stochastic_steady_state_iterative_solution_forward([sparsevec(𝐒₁); vec(𝐒₂); vec(𝐒₃)]; dims = [size(𝐒₁); size(𝐒₂); size(𝐒₃)], 𝓂 = 𝓂)
    end

    # all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    # all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
    # NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
    
    # all_SS = [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
    # we need all variables for the stochastic steady state because even leads and lags have different SSS then the non-lead-lag ones (contrary to the no stochastic steady state) and we cannot recover them otherwise

    return all_SS + state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃
end


function calculate_third_order_stochastic_steady_state(::Val{:Newton}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        𝐒₃::AbstractSparseMatrix{Float64},
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14)
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
    
    kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)
    
    A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))
        
        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6 - x)

        if i > 5 && isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    return x, isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6, x, rtol = tol)
end


function calculate_third_order_stochastic_steady_state(::Val{:Newton}, 
                                                        𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                                        𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                                        𝐒₃::AbstractSparseMatrix{ℱ.Dual{Z,S,N}},
                                                        x::Vector{ℱ.Dual{Z,S,N}},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14) where {Z,S,N}
# TODO: check whether this works with SParseMatrices
    𝐒₁̂ = ℱ.value.(𝐒₁)
    𝐒₂̂ = ℱ.value.(𝐒₂)
    𝐒₃̂ = ℱ.value.(𝐒₃)
    x̂ = ℱ.value.(x)
    
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
    
    kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)
    
    A = 𝐒₁̂[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃̂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x̂,1), vcat(x̂,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6 - x̂)

        if i > 5 && isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6, x̂, rtol = tol)
            break
        end
        
        # x̂ += Δx
        ℒ.axpy!(-1, Δx, x̂)
    end

    ∂x = zeros(length(x̂), N)
    
    for i in 1:N
        ∂𝐒₁ = ℱ.partials.(𝐒₁, i)
        ∂𝐒₂ = ℱ.partials.(𝐒₂, i)
        ∂𝐒₃ = ℱ.partials.(𝐒₃, i)

        ∂A = ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
        ∂B̂ = ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
        ∂Ĉ = ∂𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

        tmp = ∂A * x̂ + ∂B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + ∂Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6

        TMP = A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x̂,1), vcat(x̂,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)

        ∂x[:,i] = -TMP \ tmp
    end
    
    return reshape(map(x̂, eachrow(∂x)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6, x̂, rtol = tol)
end




function rrule(::typeof(calculate_third_order_stochastic_steady_state),
                                                        ::Val{:Newton}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        𝐒₃::AbstractSparseMatrix{Float64},
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14)
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
    
    kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)
    
    A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))
        
        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6 - x)

        if i > 5 && isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    solved = isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6, x, rtol = tol)         

    ∂𝐒₁ =  zero(𝐒₁)
    ∂𝐒₂ =  zero(𝐒₂)
    ∂𝐒₃ =  zero(𝐒₃)

    function third_order_stochastic_steady_state_pullback(∂x)
        S = -∂x[1]' / (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed] = S' * x'
        
        ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺] = S' * ℒ.kron(vcat(x,1), vcat(x,1))' / 2

        ∂𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺] = S' * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))' / 6

        return NoTangent(), NoTangent(), ∂𝐒₁, ∂𝐒₂, ∂𝐒₃, NoTangent(), NoTangent(), NoTangent()
    end

    return (x, solved), third_order_stochastic_steady_state_pullback
end


function solve!(𝓂::ℳ; 
    parameters::ParameterType = nothing, 
    dynamics::Bool = false, 
    algorithm::Symbol = :riccati, 
    obc::Bool = false,
    verbose::Bool = false,
    silent::Bool = false,
    tol::AbstractFloat = 1e-12)

    @assert algorithm ∈ all_available_algorithms
    
    write_parameters_input!(𝓂, parameters, verbose = verbose)

    if 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝛔 == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0) && 
        algorithm ∈ [:second_order, :pruned_second_order]
        start_time = time()
        if !silent print("Take symbolic derivatives up to second order:\t\t\t\t") end
        write_functions_mapping!(𝓂, 2)
        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    elseif 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0) && algorithm ∈ [:third_order, :pruned_third_order]
        start_time = time()
        if !silent print("Take symbolic derivatives up to third order:\t\t\t\t") end
        write_functions_mapping!(𝓂, 3)
        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    end

    if dynamics
        obc_not_solved = isnothing(𝓂.solution.perturbation.first_order.state_update_obc)
        if  ((:riccati             == algorithm) && ((:riccati             ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:first_order         == algorithm) && ((:first_order         ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:second_order        == algorithm) && ((:second_order        ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:pruned_second_order == algorithm) && ((:pruned_second_order ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:third_order         == algorithm) && ((:third_order         ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:pruned_third_order  == algorithm) && ((:pruned_third_order  ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))

            SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, verbose = verbose) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

            if solution_error > tol
                @warn "Could not find non stochastic steady steady."
            end

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
            
            S₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
            
            @assert solved "Could not find stable first order solution."

            state_update₁ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            shock]
                return S₁ * aug_state # you need a return statement for forwarddiff to work
            end
            
            if obc
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1, verbose = false)

                ∇̂₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
            
                Ŝ₁, solved = calculate_first_order_solution(∇̂₁; T = 𝓂.timings)

                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 0, verbose = false)

                state_update₁̂ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                shock]
                    return Ŝ₁ * aug_state # you need a return statement for forwarddiff to work
                end
            else
                state_update₁̂ = nothing
            end
            
            𝓂.solution.perturbation.first_order = perturbation_solution(S₁, state_update₁, state_update₁̂)
            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:riccati, :first_order])

            𝓂.solution.non_stochastic_steady_state = SS_and_pars
            𝓂.solution.outdated_NSSS = solution_error > tol

        end

        obc_not_solved = isnothing(𝓂.solution.perturbation.second_order.state_update_obc)
        if  ((:second_order  == algorithm) && ((:second_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:third_order  == algorithm) && ((:third_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))
            

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, verbose = verbose)
            
            if !converged  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

            state_update₂ = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            1
                            shock]
                return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
            end

            if obc
                Ŝ₁̂ = [Ŝ₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) Ŝ₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
            
                state_update₂̂ = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                1
                                shock]
                    return Ŝ₁̂ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                end
            else
                state_update₂̂ = nothing
            end

            𝓂.solution.perturbation.second_order = second_order_perturbation_solution(𝐒₂, stochastic_steady_state, state_update₂, state_update₂̂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:second_order])
        end
        
        obc_not_solved = isnothing(𝓂.solution.perturbation.pruned_second_order.state_update_obc)
        if  ((:pruned_second_order  == algorithm) && ((:pruned_second_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:pruned_third_order  == algorithm) && ((:pruned_third_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, verbose = verbose, pruning = true)

            if !converged  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

            state_update₂ = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
                aug_state₁ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 1; shock]
                aug_state₂ = [pruned_states[2][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
                
                return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
            end

            if obc
                Ŝ₁̂ = [Ŝ₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) Ŝ₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
            
                state_update₂̂ = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
                    aug_state₁ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 1; shock]
                    aug_state₂ = [pruned_states[2][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
                    
                    return [Ŝ₁̂ * aug_state₁, Ŝ₁̂ * aug_state₂ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
                end
            else
                state_update₂̂ = nothing
            end

            𝓂.solution.perturbation.pruned_second_order = second_order_perturbation_solution(𝐒₂, stochastic_steady_state, state_update₂, state_update₂̂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:pruned_second_order])
        end
        
        obc_not_solved = isnothing(𝓂.solution.perturbation.third_order.state_update_obc)
        if  ((:third_order  == algorithm) && ((:third_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))
            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, verbose = verbose)

            if !converged  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

            state_update₃ = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                1
                                shock]
                return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
            end

            if obc
                Ŝ₁̂ = [Ŝ₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) Ŝ₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
            
                state_update₃̂ = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                    1
                                    shock]
                    return Ŝ₁̂ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
                end
            else
                state_update₃̂ = nothing
            end

            𝓂.solution.perturbation.third_order = third_order_perturbation_solution(𝐒₃, stochastic_steady_state, state_update₃, state_update₃̂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:third_order])
        end

        obc_not_solved = isnothing(𝓂.solution.perturbation.pruned_third_order.state_update_obc)
        if ((:pruned_third_order  == algorithm) && ((:pruned_third_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, verbose = verbose, pruning = true)

            if !converged  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

            state_update₃ = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
                aug_state₁ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 1; shock]
                aug_state₁̂ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 0; shock]
                aug_state₂ = [pruned_states[2][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
                aug_state₃ = [pruned_states[3][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
                
                kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)
                
                return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * kron_aug_state₁ / 2, 𝐒₁ * aug_state₃ + 𝐒₂ * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒₃ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
            end

            if obc
                Ŝ₁̂ = [Ŝ₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) Ŝ₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]
            
                state_update₃̂ = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
                    aug_state₁ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 1; shock]
                    aug_state₁̂ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 0; shock]
                    aug_state₂ = [pruned_states[2][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
                    aug_state₃ = [pruned_states[3][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
                    
                    kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)
                    
                    return [Ŝ₁̂ * aug_state₁, Ŝ₁̂ * aug_state₂ + 𝐒₂ * kron_aug_state₁ / 2, Ŝ₁̂ * aug_state₃ + 𝐒₂ * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒₃ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # strictly following Andreasen et al. (2018)
                end
            else
                state_update₃̂ = nothing
            end

            𝓂.solution.perturbation.pruned_third_order = third_order_perturbation_solution(𝐒₃, stochastic_steady_state, state_update₃, state_update₃̂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:pruned_third_order])
        end
        
        obc_not_solved = isnothing(𝓂.solution.perturbation.quadratic_iteration.state_update_obc)
        if  ((:binder_pesaran  == algorithm) && ((:binder_pesaran   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:quadratic_iteration  == algorithm) && ((:quadratic_iteration   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))
            
            SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, verbose = verbose) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

            if solution_error > tol
                @warn "Could not find non stochastic steady steady."
            end

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)#|> Matrix
            
            S₁, converged = calculate_quadratic_iteration_solution(∇₁; T = 𝓂.timings)
            
            state_update₁ₜ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            shock]
                return S₁ * aug_state # you need a return statement for forwarddiff to work
            end
            
            if obc
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1, verbose = false)

                ∇̂₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
            
                Ŝ₁, converged = calculate_quadratic_iteration_solution(∇₁; T = 𝓂.timings)
            
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 0, verbose = false)

                state_update₁̂ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                shock]
                    return Ŝ₁ * aug_state # you need a return statement for forwarddiff to work
                end
            else
                state_update₁̂ = nothing
            end

            𝓂.solution.perturbation.quadratic_iteration = perturbation_solution(S₁, state_update₁ₜ, state_update₁̂)
            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:quadratic_iteration, :binder_pesaran])

            𝓂.solution.non_stochastic_steady_state = SS_and_pars
            𝓂.solution.outdated_NSSS = solution_error > tol
            
        end

        obc_not_solved = isnothing(𝓂.solution.perturbation.linear_time_iteration.state_update_obc)
        if  ((:linear_time_iteration  == algorithm) && ((:linear_time_iteration   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))
            
            SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, verbose = verbose) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

            if solution_error > tol
                @warn "Could not find non stochastic steady steady."
            end

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
            
            S₁ = calculate_linear_time_iteration_solution(∇₁; T = 𝓂.timings)
            
            state_update₁ₜ = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            shock]
                return S₁ * aug_state # you need a return statement for forwarddiff to work
            end
            
            if obc
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1)

                ∇̂₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
            
                Ŝ₁, converged = calculate_linear_time_iteration_solution(∇₁; T = 𝓂.timings)
            
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 0)

                state_update₁̂ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                shock]
                    return Ŝ₁ * aug_state # you need a return statement for forwarddiff to work
                end
            else
                state_update₁̂ = nothing
            end

            𝓂.solution.perturbation.linear_time_iteration = perturbation_solution(S₁, state_update₁ₜ, state_update₁̂)
            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:linear_time_iteration])

            𝓂.solution.non_stochastic_steady_state = SS_and_pars
            𝓂.solution.outdated_NSSS = solution_error > tol
        end
    end
    
    return nothing
end




function create_second_order_auxilliary_matrices(T::timings)
    # Indices and number of variables
    n₋ = T.nPast_not_future_and_mixed
    nₑ = T.nExo

    # setup compression matrices for hessian matrix
    nₑ₋ = T.nPast_not_future_and_mixed + T.nVars + T.nFuture_not_past_and_mixed + T.nExo
    colls2 = [nₑ₋ * (i-1) + k for i in 1:nₑ₋ for k in 1:i]
    𝐂∇₂ = sparse(colls2, 1:length(colls2), 1)
    𝐔∇₂ = 𝐂∇₂' * sparse([i <= k ? (k - 1) * nₑ₋ + i : (i - 1) * nₑ₋ + k for k in 1:nₑ₋ for i in 1:nₑ₋], 1:nₑ₋^2, 1)

    # set up vector to capture volatility effect
    nₑ₋ = n₋ + 1 + nₑ
    redu = sparsevec(nₑ₋ - nₑ + 1:nₑ₋, 1)
    redu_idxs = findnz(ℒ.kron(redu, redu))[1]
    𝛔 = @views sparse(redu_idxs[Int.(range(1,nₑ^2,nₑ))], fill(n₋ * (nₑ₋ + 1) + 1, nₑ), 1, nₑ₋^2, nₑ₋^2)
    
    # setup compression matrices for transition matrix
    colls2 = [nₑ₋ * (i-1) + k for i in 1:nₑ₋ for k in 1:i]
    𝐂₂ = sparse(colls2, 1:length(colls2), 1)
    𝐔₂ = 𝐂₂' * sparse([i <= k ? (k - 1) * nₑ₋ + i : (i - 1) * nₑ₋ + k for k in 1:nₑ₋ for i in 1:nₑ₋], 1:nₑ₋^2, 1)

    return second_order_auxilliary_matrices(𝛔, 𝐂₂, 𝐔₂, 𝐔∇₂)
end



function add_sparse_entries!(P, perm)
    n = size(P, 1)
    for i in 1:n
        P[perm[i], i] += 1.0
    end
end


function create_third_order_auxilliary_matrices(T::timings, ∇₃_col_indices::Vector{Int})    
    # Indices and number of variables
    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    n = T.nVars
    nₑ = T.nExo

    n̄ = n₋ + n + n₊ + nₑ

    # compression matrices for third order derivatives matrix
    nₑ₋ = T.nPast_not_future_and_mixed + T.nVars + T.nFuture_not_past_and_mixed + T.nExo
    colls3 = [nₑ₋^2 * (i-1) + nₑ₋ * (k-1) + l for i in 1:nₑ₋ for k in 1:i for l in 1:k]
    𝐂∇₃ = sparse(colls3, 1:length(colls3) , 1.0)
    
    idxs = Int[]
    for k in 1:nₑ₋
        for j in 1:nₑ₋
            for i in 1:nₑ₋
                sorted_ids = sort([k,j,i])
                push!(idxs, (sorted_ids[3] - 1) * nₑ₋ ^ 2 + (sorted_ids[2] - 1) * nₑ₋ + sorted_ids[1])
            end
        end
    end
    
    𝐔∇₃ = 𝐂∇₃' * sparse(idxs,1:nₑ₋ ^ 3, 1)

    # compression matrices for third order transition matrix
    nₑ₋ = n₋ + 1 + nₑ
    colls3 = [nₑ₋^2 * (i-1) + nₑ₋ * (k-1) + l for i in 1:nₑ₋ for k in 1:i for l in 1:k]
    𝐂₃ = sparse(colls3, 1:length(colls3) , 1.0)
    
    idxs = Int[]
    for k in 1:nₑ₋
        for j in 1:nₑ₋
            for i in 1:nₑ₋
                sorted_ids = sort([k,j,i])
                push!(idxs, (sorted_ids[3] - 1) * nₑ₋ ^ 2 + (sorted_ids[2] - 1) * nₑ₋ + sorted_ids[1])
            end
        end
    end
    
    𝐔₃ = 𝐂₃' * sparse(idxs,1:nₑ₋ ^ 3, 1)
    
    # Precompute 𝐈₃
    𝐈₃ = Dict{Vector{Int}, Int}()
    idx = 1
    for i in 1:nₑ₋
        for k in 1:i 
            for l in 1:k
                𝐈₃[[i,k,l]] = idx
                idx += 1
            end
        end
    end

    # permutation matrices
    M = reshape(1:nₑ₋^3,1,nₑ₋,nₑ₋,nₑ₋)

    𝐏 = spzeros(nₑ₋^3, nₑ₋^3)  # Preallocate the sparse matrix

    # Create the permutations directly
    add_sparse_entries!(𝐏, PermutedDimsArray(M, (1, 4, 2, 3)))
    add_sparse_entries!(𝐏, PermutedDimsArray(M, (1, 2, 4, 3)))
    add_sparse_entries!(𝐏, PermutedDimsArray(M, (1, 2, 3, 4)))

    # 𝐏 = @views sparse(reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 4, 2, 3])],nₑ₋^3,nₑ₋^3)
    #                     + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 4, 3])],nₑ₋^3,nₑ₋^3)
    #                     + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 3, 4])],nₑ₋^3,nₑ₋^3))

    𝐏₁ₗ = sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(2,1,3))),:])
    𝐏₁ᵣ = sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(2,1,3)))])

    𝐏₁ₗ̂  = @views sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(1,3,2))),:])
    𝐏₂ₗ̂  = @views sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(3,1,2))),:])

    𝐏₁ₗ̄ = @views sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2))),:])
    𝐏₂ₗ̄ = @views sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2))),:])


    𝐏₁ᵣ̃ = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2)))])
    𝐏₂ᵣ̃ = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2)))])

    ∇₃_col_indices_extended = findnz(sparse(ones(Int,length(∇₃_col_indices)),∇₃_col_indices,ones(Int,length(∇₃_col_indices)),1,size(𝐔∇₃,1)) * 𝐔∇₃)[2]

    nonnull_columns = Set{Int}()
    for i in 1:n̄ 
        for j in i:n̄ 
            for k in j:n̄ 
                if n̄^2 * (i - 1)  + n̄ * (j - 1) + k in ∇₃_col_indices_extended
                    push!(nonnull_columns,i)
                    push!(nonnull_columns,j)
                    push!(nonnull_columns,k)
                end
            end
        end
    end
            
    𝐒𝐏 = sparse(collect(nonnull_columns), collect(nonnull_columns), 1, n̄, n̄)

    return third_order_auxilliary_matrices(𝐂₃, 𝐔₃, 𝐈₃, 𝐔∇₃, 𝐏, 𝐏₁ₗ, 𝐏₁ᵣ, 𝐏₁ₗ̂, 𝐏₂ₗ̂, 𝐏₁ₗ̄, 𝐏₂ₗ̄, 𝐏₁ᵣ̃, 𝐏₂ᵣ̃, 𝐒𝐏)
end



function write_sparse_derivatives_function(rows::Vector{Int},columns::Vector{Int},values::Vector{Symbolics.Num},nrows::Int,ncolumns::Int,::Val{:Symbolics})
    vals_expr = Symbolics.toexpr.(values)

    @RuntimeGeneratedFunction(
        :(𝔛 -> sparse(
                        $rows, 
                        $columns, 
                        [$(vals_expr...)], 
                        $nrows, 
                        $ncolumns
                    )
        )
    )
end

function write_sparse_derivatives_function(rows::Vector{Int},columns::Vector{Int},values::Vector{Symbolics.Num},nrows::Int,ncolumns::Int,::Val{:string})
    vals_expr = Meta.parse(string(values))

    vals_expr.args[1] = :Float64

    @RuntimeGeneratedFunction(
        :(𝔛 -> sparse(
                        $rows, 
                        $columns,
                        $vals_expr, 
                        $nrows, 
                        $ncolumns
                    )
        )
    )
end


function write_derivatives_function(values::Vector{Symbolics.Num}, ::Val{:string})
    vals_expr = Meta.parse(string(values))
    
    @RuntimeGeneratedFunction(:(𝔛 -> $(Expr(:vect, vals_expr.args[2:end]...))))
end

function write_derivatives_function(values::Symbolics.Num, ::Val{:string})
    vals_expr = Meta.parse(string(values))
    
    @RuntimeGeneratedFunction(:(𝔛 -> $vals_expr.args))
end

function write_derivatives_function(values::Vector{Symbolics.Num}, position::UnitRange{Int}, ::Val{:string})
    vals_expr = Meta.parse(string(values))

    @RuntimeGeneratedFunction(:(𝔛 -> ($(Expr(:vect, vals_expr.args[2:end]...)), $position)))
end # needed for JET tests

function write_derivatives_function(values::Vector{Symbolics.Num}, position::Int, ::Val{:string})
    vals_expr = Meta.parse(string(values))

    @RuntimeGeneratedFunction(:(𝔛 -> ($(Expr(:vect, vals_expr.args[2:end]...)), $position)))
end

function write_derivatives_function(values::Symbolics.Num, position::Int, ::Val{:string})
    vals_expr = Meta.parse(string(values))

    @RuntimeGeneratedFunction(:(𝔛 -> ($vals_expr, $position)))
end

function write_derivatives_function(values::Symbolics.Num, position::UnitRange{Int}, ::Val{:string})
    vals_expr = Meta.parse(string(values))
    position  = position[1]
    @RuntimeGeneratedFunction(:(𝔛 -> ($vals_expr, $position)))
end # needed for JET tests

function write_derivatives_function(values::Vector{Symbolics.Num}, ::Val{:Symbolics})
    vals_expr = Symbolics.toexpr.(values)

    @RuntimeGeneratedFunction(:(𝔛 -> [$(vals_expr...)]))
end

function write_derivatives_function(values::Symbolics.Num, ::Val{:Symbolics})
    vals_expr = Symbolics.toexpr.(values)

    @RuntimeGeneratedFunction(:(𝔛 -> $vals_expr))
end

# TODO: check why this takes so much longer than previous implementation
function write_functions_mapping!(𝓂::ℳ, max_perturbation_order::Int; max_exprs_per_func::Int = 1)
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
    # dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.dyn_ss_list))) .* "₍ₛₛ₎")
    
    future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
    present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
    past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
    exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
    # stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))
    
    vars_raw = [dyn_future_list[indexin(sort(future),future)]...,
                dyn_present_list[indexin(sort(present),present)]...,
                dyn_past_list[indexin(sort(past),past)]...,
                dyn_exo_list[indexin(sort(exo),exo)]...]

    Symbolics.@syms norminvcdf(x) norminv(x) qnorm(x) normlogpdf(x) normpdf(x) normcdf(x) pnorm(x) dnorm(x) erfc(x) erfcinv(x)

    # overwrite SymPyCall names
    input_args = vcat(future_varss,
                        present_varss,
                        past_varss,
                        ss_varss,
                        𝓂.parameters,
                        𝓂.calibration_equations_parameters,
                        shock_varss)

    eval(:(Symbolics.@variables $(input_args...)))

    Symbolics.@variables 𝔛[1:length(input_args)]

    SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
    
    calib_eq_no_vars = reduce(union, get_symbols.(𝓂.calibration_equations_no_var), init = []) |> collect
    
    eval(:(Symbolics.@variables $((vcat(SS_and_pars_names_lead_lag, calib_eq_no_vars))...)))

    vars = eval(:(Symbolics.@variables $(vars_raw...)))

    eqs = Symbolics.parse_expr_to_symbolic.(𝓂.dyn_equations,(@__MODULE__,))

    final_indices = vcat(𝓂.parameters, SS_and_pars_names_lead_lag)

    input_X = Pair{Symbolics.Num, Symbolics.Num}[]
    input_X_no_time = Pair{Symbolics.Num, Symbolics.Num}[]
    
    for (v,input) in enumerate(input_args)
        push!(input_X, eval(input) => eval(𝔛[v]))
    
        if input ∈ shock_varss
            push!(input_X_no_time, eval(𝔛[v]) => 0)
        else
            input_no_time = Symbol(replace(string(input), r"₍₁₎$"=>"", r"₍₀₎$"=>"" , r"₍₋₁₎$"=>"", r"₍ₛₛ₎$"=>"", r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

            vv = indexin([input_no_time], final_indices)
            
            if vv[1] isa Int
                push!(input_X_no_time, eval(𝔛[v]) => eval(𝔛[vv[1]]))
            end
        end
    end

    vars_X = map(x -> Symbolics.substitute(x, input_X), vars)

    calib_eqs = Dict([(eval(calib_eq.args[1]) => eval(calib_eq.args[2])) for calib_eq in reverse(𝓂.calibration_equations_no_var)])

    eqs_sub = Symbolics.Num[]
    for subst in eqs
        for _ in calib_eqs
            for calib_eq in calib_eqs
                subst = Symbolics.substitute(subst, calib_eq)
            end
        end
        # subst = Symbolics.fixpoint_sub(subst, calib_eqs)
        subst = Symbolics.substitute(subst, input_X)
        push!(eqs_sub, subst)
    end
    
    if max_perturbation_order >= 2 
        nk = length(vars_raw)
        second_order_idxs = [nk * (i-1) + k for i in 1:nk for k in 1:i]
        if max_perturbation_order == 3
            third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]
        end
    end

    first_order = Symbolics.Num[]
    second_order = Symbolics.Num[]
    third_order = Symbolics.Num[]
    row1 = Int[]
    row2 = Int[]
    row3 = Int[]
    column1 = Int[]
    column2 = Int[]
    column3 = Int[]

    # Polyester.@batch for rc1 in 0:length(vars_X) * length(eqs_sub) - 1
    # for rc1 in 0:length(vars_X) * length(eqs_sub) - 1
    for (c1, var1) in enumerate(vars_X)
        for (r, eq) in enumerate(eqs_sub)
        # r, c1 = divrem(rc1, length(vars_X)) .+ 1
        # var1 = vars_X[c1]
        # eq = eqs_sub[r]
            if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
                deriv_first = Symbolics.derivative(eq, var1)
                
                push!(first_order, deriv_first)
                push!(row1, r)
                # push!(row1, r...)
                push!(column1, c1)
                if max_perturbation_order >= 2 
                    for (c2, var2) in enumerate(vars_X)
                        if (((c1 - 1) * length(vars) + c2) ∈ second_order_idxs) && (Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first)))
                            deriv_second = Symbolics.derivative(deriv_first, var2)
                            
                            push!(second_order, deriv_second)
                            push!(row2, r)
                            push!(column2, Int.(indexin([(c1 - 1) * length(vars) + c2], second_order_idxs))...)
                            if max_perturbation_order == 3
                                for (c3, var3) in enumerate(vars_X)
                                    if (((c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3) ∈ third_order_idxs) && (Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second)))
                                        deriv_third = Symbolics.derivative(deriv_second,var3)

                                        push!(third_order, deriv_third)
                                        push!(row3, r)
                                        push!(column3, Int.(indexin([(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3], third_order_idxs))...)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    if max_perturbation_order >= 1
        if 𝓂.model_jacobian[2] == Int[]
            write_auxilliary_indices!(𝓂)

            write_derivatives_of_ss_equations!(𝓂::ℳ, max_exprs_per_func = max_exprs_per_func)

            # derivative of jacobian wrt SS_and_pars and parameters
            eqs_static = map(x -> Symbolics.substitute(x, input_X_no_time), first_order)

            ∂jacobian_∂SS_and_pars = Symbolics.sparsejacobian(eqs_static, eval.(𝔛[1:(length(final_indices))]), simplify = false) # |> findnz

            idx_conversion = (row1 + length(eqs) * (column1 .- 1))

            cols, rows, vals = findnz(∂jacobian_∂SS_and_pars) #transposed

            converted_cols = idx_conversion[cols]

            perm_vals = sortperm(converted_cols) # sparse reorders the rows and cols and sorts by column. need to do that also for the values

            min_n_funcs = length(vals) ÷ max_exprs_per_func + 1

            funcs = Function[]

            lk = ReentrantLock()

            if min_n_funcs == 1
                push!(funcs, write_derivatives_function(vals[perm_vals], 1:length(vals), Val(:string)))
            else
                Polyester.@batch minbatch = 20 for i in 1:min(min_n_funcs, length(vals))
                    indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(vals) : i * max_exprs_per_func)

                    indices = length(indices) == 1 ? indices[1] : indices

                    func = write_derivatives_function(vals[perm_vals][indices], indices, Val(:string))

                    begin
                        lock(lk)
                        try
                            push!(funcs, func)
                        finally
                            unlock(lk)
                        end
                    end
                end
            end

            𝓂.model_jacobian_SS_and_pars_vars = (funcs, sparse(rows, converted_cols, zero(cols), length(final_indices), length(eqs) * length(vars)))

            # first order
            min_n_funcs = length(first_order) ÷ max_exprs_per_func + 1

            funcs = Function[]

            lk = ReentrantLock()

            if min_n_funcs == 1
                push!(funcs, write_derivatives_function(first_order, 1:length(first_order), Val(:string)))
            else
                Polyester.@batch minbatch = 20 for i in 1:min(min_n_funcs, length(first_order))
                    indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(first_order) : i * max_exprs_per_func)

                    indices = length(indices) == 1 ? indices[1] : indices

                    func = write_derivatives_function(first_order[indices], indices, Val(:string))

                    begin
                        lock(lk)
                        try
                            push!(funcs, func)
                        finally
                            unlock(lk)
                        end
                    end
                end
            end

            𝓂.model_jacobian = (funcs, row1 .+ (column1 .- 1) .* length(eqs_sub),  zeros(length(eqs_sub), length(vars)))
        end
    end
        
    if max_perturbation_order >= 2
    # second order
        if 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝛔 == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            𝓂.solution.perturbation.second_order_auxilliary_matrices = create_second_order_auxilliary_matrices(𝓂.timings)

            perm_vals = sortperm(column2) # sparse reorders the rows and cols and sorts by column. need to do that also for the values

            min_n_funcs = length(second_order) ÷ max_exprs_per_func + 1

            funcs = Function[]
        
            lk = ReentrantLock()

            if min_n_funcs == 1
                push!(funcs, write_derivatives_function(second_order[perm_vals], 1:length(second_order), Val(:string)))
            else
                Polyester.@batch minbatch = 20 for i in 1:min(min_n_funcs, length(second_order))
                    indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(second_order) : i * max_exprs_per_func)
            
                    indices = length(indices) == 1 ? indices[1] : indices

                    func = write_derivatives_function(second_order[perm_vals][indices], indices, Val(:string))

                    begin
                        lock(lk)
                        try
                            push!(funcs, func)
                        finally
                            unlock(lk)
                        end
                    end
                end
            end

            𝓂.model_hessian = (funcs, sparse(row2, column2, zero(column2), length(eqs_sub), size(𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂,1)))
        end

        # derivative of hessian wrt SS_and_pars and parameters
        eqs_static = map(x -> Symbolics.substitute(x, input_X_no_time), second_order)

        ∂hessian_∂SS_and_pars = Symbolics.sparsejacobian(eqs_static, eval.(𝔛[1:(length(final_indices))]), simplify = false) # |> findnz

        idx_conversion = (row2 + length(eqs) * (column2 .- 1))

        cols, rows, vals = findnz(∂hessian_∂SS_and_pars) #transposed

        converted_cols = idx_conversion[cols]

        perm_vals = sortperm(converted_cols) # sparse reorders the rows and cols and sorts by column. need to do that also for the values

        min_n_funcs = length(vals) ÷ max_exprs_per_func + 1

        funcs = Function[]

        lk = ReentrantLock()

        if min_n_funcs == 1
            push!(funcs, write_derivatives_function(vals[perm_vals], 1:length(vals), Val(:string)))
        else
            Polyester.@batch minbatch = 20 for i in 1:min(min_n_funcs, length(vals))
                indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(vals) : i * max_exprs_per_func)

                indices = length(indices) == 1 ? indices[1] : indices

                func = write_derivatives_function(vals[perm_vals][indices], indices, Val(:string))

                begin
                    lock(lk)
                    try
                        push!(funcs, func)
                    finally
                        unlock(lk)
                    end
                end
            end
        end

        𝓂.model_hessian_SS_and_pars_vars = (funcs, sparse(rows, converted_cols, zero(cols), length(final_indices), length(eqs) * size(𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂,1)))

    end

    if max_perturbation_order == 3
    # third order
        if 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            𝓂.solution.perturbation.third_order_auxilliary_matrices = create_third_order_auxilliary_matrices(𝓂.timings, unique(column3))
        
            perm_vals = sortperm(column3) # sparse reorders the rows and cols and sorts by column. need to do that also for the values

            min_n_funcs = length(third_order) ÷ max_exprs_per_func + 1

            funcs = Function[]
        
            lk = ReentrantLock()
            
            if min_n_funcs == 1
                push!(funcs, write_derivatives_function(third_order[perm_vals], 1:length(third_order), Val(:string)))
            else
                Polyester.@batch minbatch = 20 for i in 1:min(min_n_funcs, length(third_order))
                    indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(third_order) : i * max_exprs_per_func)
            
                    if length(indices) == 1
                        indices = indices[1]
                    end

                    func = write_derivatives_function(third_order[perm_vals][indices], indices, Val(:string))

                    begin
                        lock(lk)
                        try
                            push!(funcs, func)
                        finally
                            unlock(lk)
                        end
                    end
                end
            end

            𝓂.model_third_order_derivatives = (funcs, sparse(row3, column3, zero(column3), length(eqs_sub), size(𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃,1)))
        end

        # derivative of third order wrt SS_and_pars and parameters
        eqs_static = map(x -> Symbolics.substitute(x, input_X_no_time), third_order)

        ∂third_order_∂SS_and_pars = Symbolics.sparsejacobian(eqs_static, eval.(𝔛[1:(length(final_indices))]), simplify = false) # |> findnz

        idx_conversion = (row3 + length(eqs) * (column3 .- 1))

        cols, rows, vals = findnz(∂third_order_∂SS_and_pars) #transposed

        converted_cols = idx_conversion[cols]

        perm_vals = sortperm(converted_cols) # sparse reorders the rows and cols and sorts by column. need to do that also for the values

        min_n_funcs = length(vals) ÷ max_exprs_per_func + 1

        funcs = Function[]

        lk = ReentrantLock()

        if min_n_funcs == 1
            push!(funcs, write_derivatives_function(vals[perm_vals], 1:length(vals), Val(:string)))
        else
            Polyester.@batch minbatch = 20 for i in 1:min(min_n_funcs, length(vals))
                indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(vals) : i * max_exprs_per_func)

                indices = length(indices) == 1 ? indices[1] : indices

                func = write_derivatives_function(vals[perm_vals][indices], indices, Val(:string))

                begin
                    lock(lk)
                    try
                        push!(funcs, func)
                    finally
                        unlock(lk)
                    end
                end
            end
        end

        𝓂.model_third_order_derivatives_SS_and_pars_vars = (funcs, sparse(rows, converted_cols, zero(cols), length(final_indices), length(eqs) * size(𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃,1)))
    end

    return nothing
end


function write_derivatives_of_ss_equations!(𝓂::ℳ; max_exprs_per_func::Int = 1)
    # derivative of SS equations wrt parameters and SS_and_pars
    # unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
    SS_and_pars = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))

    ss_equations = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

    Symbolics.@syms norminvcdf(x) norminv(x) qnorm(x) normlogpdf(x) normpdf(x) normcdf(x) pnorm(x) dnorm(x) erfc(x) erfcinv(x)

    # overwrite SymPyCall names
    other_pars = setdiff(union(𝓂.parameters_in_equations, 𝓂.parameters_as_function_of_parameters), 𝓂.parameters)

    if length(other_pars) > 0
        eval(:(Symbolics.@variables $(other_pars...)))
    end

    vars = eval(:(Symbolics.@variables $(SS_and_pars...)))

    pars = eval(:(Symbolics.@variables $(𝓂.parameters...)))

    input_args = vcat(𝓂.parameters, SS_and_pars)
    
    Symbolics.@variables 𝔛[1:length(input_args)]

    input_X_no_time = Pair{Symbolics.Num, Symbolics.Num}[]

    for (v,input) in enumerate(input_args)
        push!(input_X_no_time, eval(input) => eval(𝔛[v]))
    end

    ss_eqs = Symbolics.parse_expr_to_symbolic.(ss_equations,(@__MODULE__,))

    calib_eqs = Dict([(eval(calib_eq.args[1]) => eval(calib_eq.args[2])) for calib_eq in reverse(𝓂.calibration_equations_no_var)])

    eqs = Symbolics.Num[]
    for subst in ss_eqs
        for _ in calib_eqs # to completely substitute all calibration equations
            for calib_eq in calib_eqs
                subst = Symbolics.substitute(subst, calib_eq)
            end
        end
        # subst = Symbolics.fixpoint_sub(subst, calib_eqs)
        subst = Symbolics.substitute(subst, input_X_no_time)
        push!(eqs, subst)
    end
    
    ∂SS_equations_∂parameters = Symbolics.sparsejacobian(eqs, eval.(𝔛[1:length(pars)])) |> findnz

    min_n_funcs = length(∂SS_equations_∂parameters[3]) ÷ max_exprs_per_func + 1

    funcs = Function[]

    lk = ReentrantLock()

    if min_n_funcs == 1
        push!(funcs, write_derivatives_function(∂SS_equations_∂parameters[3], 1:length(∂SS_equations_∂parameters[3]), Val(:string)))
    else
        Polyester.@batch minbatch = 20 for i in 1:min(min_n_funcs, length(∂SS_equations_∂parameters[3]))
            indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(∂SS_equations_∂parameters[3]) : i * max_exprs_per_func)

            indices = length(indices) == 1 ? indices[1] : indices

            func = write_derivatives_function(∂SS_equations_∂parameters[3][indices], indices, Val(:string))

            begin
                lock(lk)
                try
                    push!(funcs, func)
                finally
                    unlock(lk)
                end
            end
        end
    end

    𝓂.∂SS_equations_∂parameters = (funcs, sparse(∂SS_equations_∂parameters[1], ∂SS_equations_∂parameters[2], zeros(Float64,length(∂SS_equations_∂parameters[3])), length(eqs), length(pars)))

    # 𝓂.∂SS_equations_∂parameters = write_sparse_derivatives_function(∂SS_equations_∂parameters[1], 
    #                                                                     ∂SS_equations_∂parameters[2], 
    #                                                                     ∂SS_equations_∂parameters[3],
    #                                                                     length(eqs), 
    #                                                                     length(pars),
    #                                                                     Val(:string));

    ∂SS_equations_∂SS_and_pars = Symbolics.sparsejacobian(eqs, eval.(𝔛[length(pars)+1:end])) |> findnz

    min_n_funcs = length(∂SS_equations_∂SS_and_pars[3]) ÷ max_exprs_per_func + 1

    funcs = Function[]

    lk = ReentrantLock()

    if min_n_funcs == 1
        push!(funcs, write_derivatives_function(∂SS_equations_∂SS_and_pars[3], 1:length(∂SS_equations_∂SS_and_pars[3]), Val(:string)))
    else
        Polyester.@batch minbatch = 20 for i in 1:min(min_n_funcs, length(∂SS_equations_∂SS_and_pars[3]))
            indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(∂SS_equations_∂SS_and_pars[3]) : i * max_exprs_per_func)

            indices = length(indices) == 1 ? indices[1] : indices

            func = write_derivatives_function(∂SS_equations_∂SS_and_pars[3][indices], indices, Val(:string))

            begin
                lock(lk)
                try
                    push!(funcs, func)
                finally
                    unlock(lk)
                end
            end
        end
    end

    𝓂.∂SS_equations_∂SS_and_pars = (funcs, ∂SS_equations_∂SS_and_pars[1] .+ (∂SS_equations_∂SS_and_pars[2] .- 1) .* length(eqs), zeros(length(eqs), length(vars)))

    # 𝓂.∂SS_equations_∂SS_and_pars = write_sparse_derivatives_function(∂SS_equations_∂SS_and_pars[1], 
    #                                                                     ∂SS_equations_∂SS_and_pars[2], 
    #                                                                     ∂SS_equations_∂SS_and_pars[3],
    #                                                                     length(eqs), 
    #                                                                     length(vars),
    #                                                                     Val(:string));
end

function write_auxilliary_indices!(𝓂::ℳ)
    # write indices in auxiliary objects
    dyn_var_future_list  = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₁₎")))
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎")))
    dyn_var_past_list    = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₋₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₋₁₎")))
    dyn_exo_list         = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₓ₎")))
    dyn_ss_list          = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₛₛ₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₛₛ₎")))

    dyn_var_future  = Symbol.(string.(sort(collect(reduce(union,dyn_var_future_list)))))
    dyn_var_present = Symbol.(string.(sort(collect(reduce(union,dyn_var_present_list)))))
    dyn_var_past    = Symbol.(string.(sort(collect(reduce(union,dyn_var_past_list)))))
    dyn_exo         = Symbol.(string.(sort(collect(reduce(union,dyn_exo_list)))))
    dyn_ss          = Symbol.(string.(sort(collect(reduce(union,dyn_ss_list)))))

    SS_and_pars_names = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)

    dyn_var_future_idx  = indexin(dyn_var_future    , SS_and_pars_names)
    dyn_var_present_idx = indexin(dyn_var_present   , SS_and_pars_names)
    dyn_var_past_idx    = indexin(dyn_var_past      , SS_and_pars_names)
    dyn_ss_idx          = indexin(dyn_ss            , SS_and_pars_names)

    shocks_ss = zeros(length(dyn_exo))

    𝓂.solution.perturbation.auxilliary_indices = auxilliary_indices(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx, dyn_ss_idx, shocks_ss)
end

write_parameters_input!(𝓂::ℳ, parameters::Nothing; verbose::Bool = true) = return parameters
write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,Float64}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Pair{String,Float64}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters[1] |> Meta.parse |> replace_indices => parameters[2]), verbose = verbose)



write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Float64},Vararg{Pair{Symbol,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
# write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Union{Symbol,String},Union{Float64,Int}},Vararg{Pair{Union{Symbol,String},Union{Float64,Int}}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
# write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Int},Vararg{Pair{String,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{String,Float64},Vararg{Pair{String,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters])
, verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{Symbol, Float64}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{String, Float64}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)


write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,Int}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Pair{String,Int}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}((parameters[1] |> Meta.parse |> replace_indices) => parameters[2]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Int},Vararg{Pair{Symbol,Int}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{String,Int},Vararg{Pair{String,Int}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{Symbol, Int}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{String, Int}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)


write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,Real}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Pair{String,Real}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}((parameters[1] |> Meta.parse |> replace_indices) => parameters[2]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Real},Vararg{Pair{Symbol,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{String,Real},Vararg{Pair{String,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{Symbol, Real}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{String, Real}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)



function write_parameters_input!(𝓂::ℳ, parameters::Dict{Symbol,Float64}; verbose::Bool = true)
    if length(setdiff(collect(keys(parameters)),𝓂.parameters))>0
        println("Parameters not part of the model: ",setdiff(collect(keys(parameters)),𝓂.parameters))
        for kk in setdiff(collect(keys(parameters)),𝓂.parameters)
            delete!(parameters,kk)
        end
    end

    bounds_broken = false

    for (par,val) in parameters
        if haskey(𝓂.bounds,par)
            if val > 𝓂.bounds[par][2]
                println("Calibration is out of bounds for $par < $(𝓂.bounds[par][2])\t parameter value: $val")
                bounds_broken = true
                continue
            end
            if val < 𝓂.bounds[par][1]
                println("Calibration is out of bounds for $par > $(𝓂.bounds[par][1])\t parameter value: $val")
                bounds_broken = true
                continue
            end
        end
    end

    if bounds_broken
        println("Parameters unchanged.")
    else
        ntrsct_idx = map(x-> getindex(1:length(𝓂.parameter_values),𝓂.parameters .== x)[1],collect(keys(parameters)))
        
        if !all(𝓂.parameter_values[ntrsct_idx] .== collect(values(parameters))) && !(𝓂.parameters[ntrsct_idx] == [:activeᵒᵇᶜshocks])
            if verbose println("Parameter changes: ") end
            𝓂.solution.outdated_algorithms = Set(all_available_algorithms)
        end
            
        for i in 1:length(parameters)
            if 𝓂.parameter_values[ntrsct_idx[i]] != collect(values(parameters))[i]
                if collect(keys(parameters))[i] ∈ 𝓂.SS_dependencies[end][2] && 𝓂.solution.outdated_NSSS == false
                    𝓂.solution.outdated_NSSS = true
                end
                
                if verbose println("\t",𝓂.parameters[ntrsct_idx[i]],"\tfrom ",𝓂.parameter_values[ntrsct_idx[i]],"\tto ",collect(values(parameters))[i]) end

                𝓂.parameter_values[ntrsct_idx[i]] = collect(values(parameters))[i]
            end
        end
    end

    if 𝓂.solution.outdated_NSSS == true && verbose println("New parameters changed the steady state.") end
end


write_parameters_input!(𝓂::ℳ, parameters::Tuple{Int,Vararg{Int}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Float64.(vec(collect(parameters))), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Matrix{Int}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Float64.(vec(collect(parameters))), verbose = verbose)

write_parameters_input!(𝓂::ℳ, parameters::Tuple{Float64,Vararg{Float64}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, vec(collect(parameters)), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Matrix{Float64}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, vec(collect(parameters)), verbose = verbose)

write_parameters_input!(𝓂::ℳ, parameters::Tuple{Real,Vararg{Real}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Float64.(vec(collect(parameters))), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Matrix{Real}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Float64.(vec(collect(parameters))), verbose = verbose)



function write_parameters_input!(𝓂::ℳ, parameters::Vector{Float64}; verbose::Bool = true)
    if length(parameters) > length(𝓂.parameter_values)
        println("Model has "*string(length(𝓂.parameter_values))*" parameters. "*string(length(parameters))*" were provided. The following will be ignored: "*string(parameters[length(𝓂.parameter_values)+1:end]...))

        parameters = parameters[1:length(𝓂.parameter_values)]
    end

    bounds_broken = false

    for (par,val) in Dict(𝓂.parameters .=> parameters)
        if haskey(𝓂.bounds,par)
            if val > 𝓂.bounds[par][2]
                println("Calibration is out of bounds for $par < $(𝓂.bounds[par][2])\t parameter value: $val")
                bounds_broken = true
                continue
            end
            if val < 𝓂.bounds[par][1]
                println("Calibration is out of bounds for $par > $(𝓂.bounds[par][1])\t parameter value: $val")
                bounds_broken = true
                continue
            end
        end
    end

    if bounds_broken
        println("Parameters unchanged.")
    else
        if !all(parameters .== 𝓂.parameter_values[1:length(parameters)])
            𝓂.solution.outdated_algorithms = Set(all_available_algorithms)

            match_idx = []
            for (i, v) in enumerate(parameters)
                if v != 𝓂.parameter_values[i]
                    push!(match_idx,i)
                end
            end
            
            changed_vals = parameters[match_idx]
            changed_pars = 𝓂.parameters[match_idx]

            # for p in changes_pars
            #     if p ∈ 𝓂.SS_dependencies[end][2] && 𝓂.solution.outdated_NSSS == false
                    𝓂.solution.outdated_NSSS = true # fix the SS_dependencies
                    # println("SS outdated.")
            #     end
            # end

            if verbose 
                println("Parameter changes: ")
                for (i,m) in enumerate(match_idx)
                    println("\t",changed_pars[i],"\tfrom ",𝓂.parameter_values[m],"\tto ",changed_vals[i])
                end
            end

            𝓂.parameter_values[match_idx] = parameters[match_idx]
        end
    end
    if 𝓂.solution.outdated_NSSS == true && verbose println("New parameters changed the steady state.") end
end


function create_timings_for_estimation!(𝓂::ℳ, observables::Vector{Symbol})
    dyn_equations = 𝓂.dyn_equations

    vars_to_exclude = setdiff(𝓂.timings.present_only, observables)

    # Mapping variables to their equation index
    variable_to_equation = Dict{Symbol, Vector{Int}}()
    for var in vars_to_exclude
        for (eq_idx, vars_set) in enumerate(𝓂.dyn_var_present_list)
        # for var in vars_set
            if var in vars_set
                if haskey(variable_to_equation, var)
                    push!(variable_to_equation[var],eq_idx)
                else
                    variable_to_equation[var] = [eq_idx]
                end
            end
        end
    end

    # cols_to_exclude = indexin(𝓂.timings.var, setdiff(𝓂.timings.present_only, observables))
    cols_to_exclude = indexin(setdiff(𝓂.timings.present_only, observables), 𝓂.timings.var)

    present_idx = 𝓂.timings.nFuture_not_past_and_mixed .+ (setdiff(range(1, 𝓂.timings.nVars), cols_to_exclude))

    dyn_var_future_list  = deepcopy(𝓂.dyn_var_future_list)
    dyn_var_present_list = deepcopy(𝓂.dyn_var_present_list)
    dyn_var_past_list    = deepcopy(𝓂.dyn_var_past_list)
    dyn_exo_list         = deepcopy(𝓂.dyn_exo_list)
    dyn_ss_list          = deepcopy(𝓂.dyn_ss_list)

    rows_to_exclude = Int[]

    for vidx in values(variable_to_equation)
        for v in vidx
            if v ∉ rows_to_exclude
                push!(rows_to_exclude, v)

                for vv in vidx
                    dyn_var_future_list[vv] = union(dyn_var_future_list[vv], dyn_var_future_list[v])
                    dyn_var_present_list[vv] = union(dyn_var_present_list[vv], dyn_var_present_list[v])
                    dyn_var_past_list[vv] = union(dyn_var_past_list[vv], dyn_var_past_list[v])
                    dyn_exo_list[vv] = union(dyn_exo_list[vv], dyn_exo_list[v])
                    dyn_ss_list[vv] = union(dyn_ss_list[vv], dyn_ss_list[v])
                end

                break
            end
        end
    end

    rows_to_include = setdiff(1:𝓂.timings.nVars, rows_to_exclude)

    all_symbols = setdiff(reduce(union,collect.(get_symbols.(dyn_equations)))[rows_to_include], vars_to_exclude)
    parameters_in_equations = sort(setdiff(all_symbols, match_pattern(all_symbols,r"₎$")))
    
    dyn_var_future  =  sort(setdiff(collect(reduce(union,dyn_var_future_list[rows_to_include])), vars_to_exclude))
    dyn_var_present =  sort(setdiff(collect(reduce(union,dyn_var_present_list[rows_to_include])), vars_to_exclude))
    dyn_var_past    =  sort(setdiff(collect(reduce(union,dyn_var_past_list[rows_to_include])), vars_to_exclude))
    dyn_var_ss      =  sort(setdiff(collect(reduce(union,dyn_ss_list[rows_to_include])), vars_to_exclude))

    all_dyn_vars        = union(dyn_var_future, dyn_var_present, dyn_var_past)

    @assert length(setdiff(dyn_var_ss, all_dyn_vars)) == 0 "The following variables are (and cannot be) defined only in steady state (`[ss]`): $(setdiff(dyn_var_ss, all_dyn_vars))"

    all_vars = union(all_dyn_vars, dyn_var_ss)

    present_only              = sort(setdiff(dyn_var_present,union(dyn_var_past,dyn_var_future)))
    future_not_past           = sort(setdiff(dyn_var_future, dyn_var_past))
    past_not_future           = sort(setdiff(dyn_var_past, dyn_var_future))
    mixed                     = sort(setdiff(dyn_var_present, union(present_only, future_not_past, past_not_future)))
    future_not_past_and_mixed = sort(union(future_not_past,mixed))
    past_not_future_and_mixed = sort(union(past_not_future,mixed))
    present_but_not_only      = sort(setdiff(dyn_var_present,present_only))
    mixed_in_past             = sort(intersect(dyn_var_past, mixed))
    not_mixed_in_past         = sort(setdiff(dyn_var_past,mixed_in_past))
    mixed_in_future           = sort(intersect(dyn_var_future, mixed))
    exo                       = sort(collect(reduce(union,dyn_exo_list)))
    var                       = sort(dyn_var_present)
    aux_tmp                   = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_present))
    aux                       = aux_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_tmp)]
    exo_future                = dyn_var_future[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_future)]
    exo_present               = dyn_var_present[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_present)]
    exo_past                  = dyn_var_past[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_past)]

    nPresent_only              = length(present_only)
    nMixed                     = length(mixed)
    nFuture_not_past_and_mixed = length(future_not_past_and_mixed)
    nPast_not_future_and_mixed = length(past_not_future_and_mixed)
    nPresent_but_not_only      = length(present_but_not_only)
    nVars                      = length(all_vars)
    nExo                       = length(collect(exo))

    present_only_idx              = indexin(present_only,var)
    present_but_not_only_idx      = indexin(present_but_not_only,var)
    future_not_past_and_mixed_idx = indexin(future_not_past_and_mixed,var)
    past_not_future_and_mixed_idx = indexin(past_not_future_and_mixed,var)
    mixed_in_future_idx           = indexin(mixed_in_future,dyn_var_future)
    mixed_in_past_idx             = indexin(mixed_in_past,dyn_var_past)
    not_mixed_in_past_idx         = indexin(not_mixed_in_past,dyn_var_past)
    past_not_future_idx           = indexin(past_not_future,var)

    reorder       = indexin(var, [present_only; past_not_future; future_not_past_and_mixed])
    dynamic_order = indexin(present_but_not_only, [past_not_future; future_not_past_and_mixed])

    @assert length(intersect(union(var,exo),parameters_in_equations)) == 0 "Parameters and variables cannot have the same name. This is the case for: " * repr(sort([intersect(union(var,exo),parameters_in_equations)...]))

    T = timings(present_only,
                future_not_past,
                past_not_future,
                mixed,
                future_not_past_and_mixed,
                past_not_future_and_mixed,
                present_but_not_only,
                mixed_in_past,
                not_mixed_in_past,
                mixed_in_future,
                exo,
                var,
                aux,
                exo_present,

                nPresent_only,
                nMixed,
                nFuture_not_past_and_mixed,
                nPast_not_future_and_mixed,
                nPresent_but_not_only,
                nVars,
                nExo,

                present_only_idx,
                present_but_not_only_idx,
                future_not_past_and_mixed_idx,
                not_mixed_in_past_idx,
                past_not_future_and_mixed_idx,
                mixed_in_past_idx,
                mixed_in_future_idx,
                past_not_future_idx,

                reorder,
                dynamic_order)

    push!(𝓂.estimation_helper, observables => T)
end



function calculate_jacobian(parameters::Vector{M}, SS_and_pars::Vector{N}, 𝓂::ℳ)::Matrix{M} where {M,N}
    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
    calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
    
    par = vcat(parameters,calibrated_parameters)
    
    dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
    dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
    dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
    dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

    shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

    X = [SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; SS[dyn_ss_idx]; par; shocks_ss]
    
    # vals = M[]

    # for f in 𝓂.model_jacobian[1]
    #     push!(vals, f(X)...)
    # end

    vals = zeros(M, length(𝓂.model_jacobian[1]))
    
    # lk = ReentrantLock()

    Polyester.@batch minbatch = 200 for f in 𝓂.model_jacobian[1]
    # for f in 𝓂.model_jacobian[1]
        # val, idx = f(X)#::Tuple{<: Real, Int}
        out = f(X)#::Tuple{Vector{<: Real}, UnitRange{Int64}}
        
        # begin
        #     lock(lk)
        #     try
                @inbounds vals[out[2]] = out[1]
                # @inbounds vals[idx] = val
        #     finally
        #         unlock(lk)
        #     end
        # end
    end

    if eltype(𝓂.model_jacobian[3]) ≠ M
        Accessors.@reset 𝓂.model_jacobian[3] = convert(Matrix{M}, 𝓂.model_jacobian[3])
    end

    𝓂.model_jacobian[3][𝓂.model_jacobian[2]] .= vals

    return 𝓂.model_jacobian[3]
end


function rrule(::typeof(calculate_jacobian), parameters, SS_and_pars, 𝓂)
    jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    function calculate_jacobian_pullback(∂∇₁)
        X = [parameters; SS_and_pars]

        # vals = Float64[]

        # for f in 𝓂.model_jacobian_SS_and_pars_vars[1]
        #     push!(vals, f(X)...)
        # end

        vals = zeros(Float64, length(𝓂.model_jacobian_SS_and_pars_vars[1]))

        # lk = ReentrantLock()

        Polyester.@batch minbatch = 200 for f in 𝓂.model_jacobian_SS_and_pars_vars[1]
            out = f(X)

            # begin
            #     lock(lk)
            #     try
                    @inbounds vals[out[2]] = out[1]
            #     finally
            #         unlock(lk)
            #     end
            # end
        end
    
        Accessors.@reset 𝓂.model_jacobian_SS_and_pars_vars[2].nzval = vals
        
        analytical_jacobian_SS_and_pars_vars = 𝓂.model_jacobian_SS_and_pars_vars[2] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

        cols_unique = unique(findnz(analytical_jacobian_SS_and_pars_vars)[2])

        v∂∇₁ = ∂∇₁[cols_unique]

        ∂parameters_and_SS_and_pars = analytical_jacobian_SS_and_pars_vars[:,cols_unique] * v∂∇₁

        return NoTangent(), ∂parameters_and_SS_and_pars[1:length(parameters)], ∂parameters_and_SS_and_pars[length(parameters)+1:end], NoTangent()
    end

    return jacobian, calculate_jacobian_pullback
end


function calculate_hessian(parameters::Vector{M}, SS_and_pars::Vector{N}, 𝓂::ℳ) where {M,N}
    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
    calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
    
    par = vcat(parameters,calibrated_parameters)

    dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
    dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
    dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
    dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

    shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

    # nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)
        
    # return sparse(reshape(𝒜.jacobian(𝒷(), x -> 𝒜.jacobian(𝒷(), x -> (𝓂.model_function(x, par, SS)), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^2))#, SS_and_pars
    # return 𝓂.model_hessian([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SS[dyn_ss_idx]]) * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂

    # second_out =  [f([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SS[dyn_ss_idx]]) for f in 𝓂.model_hessian]
    
    # vals = [i[1] for i in second_out]
    # rows = [i[2] for i in second_out]
    # cols = [i[3] for i in second_out]

    X = [SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; SS[dyn_ss_idx]; par; shocks_ss]
    
    # vals = M[]

    # for f in 𝓂.model_hessian[1]
    #     push!(vals, f(X)...)
    # end

    vals = zeros(M, length(𝓂.model_hessian[1]))

    # lk = ReentrantLock()

    Polyester.@batch minbatch = 200 for f in 𝓂.model_hessian[1]
        out = f(X)
        
        # begin
        #     lock(lk)
        #     try
                @inbounds vals[out[2]] = out[1]
        #     finally
        #         unlock(lk)
        #     end
        # end
    end

    Accessors.@reset 𝓂.model_hessian[2].nzval = vals
    
    return 𝓂.model_hessian[2]# * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂

    # vals = M[]
    # rows = Int[]
    # cols = Int[]

    # for f in 𝓂.model_hessian
    #     output = f(input)

    #     push!(vals, output[1]...)
    #     push!(rows, output[2]...)
    #     push!(cols, output[3]...)
    # end

    # vals = convert(Vector{M}, vals)

    # # nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)
    # # sparse(rows, cols, vals, length(𝓂.dyn_equations), nk^2)
    # sparse!(rows, cols, vals, length(𝓂.dyn_equations), size(𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂,1)) * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂
end


function rrule(::typeof(calculate_hessian), parameters, SS_and_pars, 𝓂)
    hessian = calculate_hessian(parameters, SS_and_pars, 𝓂)

    function calculate_hessian_pullback(∂∇₁)
        X = [parameters; SS_and_pars]

        vals = zeros(Float64, length(𝓂.model_hessian_SS_and_pars_vars[1]))

        Polyester.@batch minbatch = 200 for f in 𝓂.model_hessian_SS_and_pars_vars[1]
            out = f(X)
            
            @inbounds vals[out[2]] = out[1]
        end
    
        Accessors.@reset 𝓂.model_hessian_SS_and_pars_vars[2].nzval = vals
        
        analytical_hessian_SS_and_pars_vars = 𝓂.model_hessian_SS_and_pars_vars[2] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

        cols_unique = unique(findnz(analytical_hessian_SS_and_pars_vars)[2])

        v∂∇₁ = ∂∇₁[cols_unique]

        ∂parameters_and_SS_and_pars = analytical_hessian_SS_and_pars_vars[:,cols_unique] * v∂∇₁

        return NoTangent(), ∂parameters_and_SS_and_pars[1:length(parameters)], ∂parameters_and_SS_and_pars[length(parameters)+1:end], NoTangent()
    end

    return hessian, calculate_hessian_pullback
end


function calculate_third_order_derivatives(parameters::Vector{M}, SS_and_pars::Vector{N}, 𝓂::ℳ) where {M,N}
    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
    calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
    
    par = vcat(parameters,calibrated_parameters)

    dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
    dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
    dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
    dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

    shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

    # return sparse(reshape(𝒜.jacobian(𝒷(), x -> 𝒜.jacobian(𝒷(), x -> 𝒜.jacobian(𝒷(), x -> 𝓂.model_function(x, par, SS), x), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^3))#, SS_and_pars
    # return 𝓂.model_third_order_derivatives([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SS[dyn_ss_idx]]) * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃
    
    
    # third_out =  [f([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx]) for f in 𝓂.model_third_order_derivatives]
    
    # vals = [i[1] for i in third_out]
    # rows = [i[2] for i in third_out]
    # cols = [i[3] for i in third_out]

    # vals = convert(Vector{M}, vals)
    
    X = [SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; SS[dyn_ss_idx]; par; shocks_ss]
    
    # vals = M[]

    # for f in 𝓂.model_third_order_derivatives[1]
    #     push!(vals, f(X)...)
    # end
    
    vals = zeros(M, length(𝓂.model_third_order_derivatives[1]))

    # lk = ReentrantLock()

    Polyester.@batch minbatch = 200 for f in 𝓂.model_third_order_derivatives[1]
        out = f(X)
        
        # begin
        #     lock(lk)
        #     try
                @inbounds vals[out[2]] = out[1]
        #     finally
        #         unlock(lk)
        #     end
        # end
    end

    Accessors.@reset 𝓂.model_third_order_derivatives[2].nzval = vals
    
    return 𝓂.model_third_order_derivatives[2]# * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃

    # vals = M[]
    # rows = Int[]
    # cols = Int[]

    # for f in 𝓂.model_third_order_derivatives
    #     output = f(input)

    #     push!(vals, output[1]...)
    #     push!(rows, output[2]...)
    #     push!(cols, output[3]...)
    # end

    # # # nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)
    # # # sparse(rows, cols, vals, length(𝓂.dyn_equations), nk^3)
    # sparse(rows, cols, vals, length(𝓂.dyn_equations), size(𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃,1)) * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃
end


function rrule(::typeof(calculate_third_order_derivatives), parameters, SS_and_pars, 𝓂)
    third_order_derivatives = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

    function calculate_third_order_derivatives_pullback(∂∇₁)
        X = [parameters; SS_and_pars]

        vals = zeros(Float64, length(𝓂.model_third_order_derivatives_SS_and_pars_vars[1]))

        Polyester.@batch minbatch = 200 for f in 𝓂.model_third_order_derivatives_SS_and_pars_vars[1]
            out = f(X)
            
            @inbounds vals[out[2]] = out[1]
        end
    
        Accessors.@reset 𝓂.model_third_order_derivatives_SS_and_pars_vars[2].nzval = vals
        
        analytical_third_order_derivatives_SS_and_pars_vars = 𝓂.model_third_order_derivatives_SS_and_pars_vars[2] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

        cols_unique = unique(findnz(analytical_third_order_derivatives_SS_and_pars_vars)[2])

        v∂∇₁ = ∂∇₁[cols_unique]

        ∂parameters_and_SS_and_pars = analytical_third_order_derivatives_SS_and_pars_vars[:,cols_unique] * v∂∇₁

        return NoTangent(), ∂parameters_and_SS_and_pars[1:length(parameters)], ∂parameters_and_SS_and_pars[length(parameters)+1:end], NoTangent()
    end

    return third_order_derivatives, calculate_third_order_derivatives_pullback
end


function calculate_linear_time_iteration_solution(∇₁::AbstractMatrix{Float64}; T::timings, tol::AbstractFloat = eps())
    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
            ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
    ∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    maxiter = 1000

    F = zero(∇₋)
    S = zero(∇₋)
    # F = randn(size(∇₋))
    # S = randn(size(∇₋))
    
    error = one(tol) + tol
    iter = 0

    while error > tol && iter <= maxiter
        F̂ = -(∇₊ * F + ∇₀) \ ∇₋
        Ŝ = -(∇₋ * S + ∇₀) \ ∇₊
        
        error = maximum(∇₊ * F̂ * F̂ + ∇₀ * F̂ + ∇₋)
        
        F = F̂
        S = Ŝ
        
        iter += 1
    end

    if iter == maxiter
        outmessage = "Convergence Failed. Max Iterations Reached. Error: $error"
    elseif maximum(abs,ℒ.eigen(F).values) > 1.0
        outmessage = "No Stable Solution Exists!"
    elseif maximum(abs,ℒ.eigen(S).values) > 1.0
        outmessage = "Multiple Solutions Exist!"
    end

    Q = -(∇₊ * F + ∇₀) \ ∇ₑ

    @views hcat(F[:,T.past_not_future_and_mixed_idx],Q)
end



function calculate_quadratic_iteration_solution(∇₁::AbstractMatrix{Float64}; T::timings, tol::AbstractFloat = eps())
    # see Binder and Pesaran (1997) for more details on this approach
    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
            ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
    ∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    
    ∇̂₀ =  RF.lu(∇₀)
    
    A = ∇̂₀ \ ∇₋
    B = ∇̂₀ \ ∇₊

    C = similar(A)
    C̄ = similar(A)

    E = similar(C)

    sol = @suppress begin
        speedmapping(zero(A); m! = (C̄, C) -> begin 
                                                ℒ.mul!(E, C, C)
                                                ℒ.mul!(C̄, B, E)
                                                ℒ.axpy!(1, A, C̄)
                                            end,
                                            # C̄ .=  A + B * C^2, 
        tol = tol, maps_limit = 10000)
    end

    C = -sol.minimizer

    D = -(∇₊ * C + ∇₀) \ ∇ₑ

    @views hcat(C[:,T.past_not_future_and_mixed_idx],D), sol.converged
end



function calculate_quadratic_iteration_solution_AD(∇₁::AbstractMatrix{S}; T::timings, tol::AbstractFloat = 1e-12) where S
    # see Binder and Pesaran (1997) for more details on this approach
    expand = @ignore_derivatives [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
            ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
    ∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    A = ∇₀ \ ∇₋
    B = ∇₀ \ ∇₊

    # A = sparse(∇̂₀ \ ∇₋) # sparsity desnt make it faster
    # B = sparse(∇̂₀ \ ∇₊)

    # droptol!(A,eps())
    # droptol!(B,eps())

    C = copy(A)
    C̄ = similar(A)

    maxiter = 10000  # Maximum number of iterations

    error = one(tol) + tol
    iter = 0

    while error > tol && iter <= maxiter
        C̄ = copy(C)  # Store the current C̄ before updating it
        
        # Update C̄ based on the given formula
        C = A + B * C^2
        
        # Check for convergence
        if iter % 100 == 0
            error = maximum(abs, C - C̄)
        end

        iter += 1
    end

    C̄ = ℒ.lu(∇₊ * -C + ∇₀, check = false)

    if !ℒ.issuccess(C̄)
        return -C, false
    end

    D = -inv(C̄) * ∇ₑ

    return hcat(-C[:, T.past_not_future_and_mixed_idx], D), error <= tol
end


function riccati_forward(∇₁::Matrix{Float64}; T::timings, explosive::Bool = false)::Tuple{Matrix{Float64},Bool}
    n₀₊ = zeros(T.nVars, T.nFuture_not_past_and_mixed)
    n₀₀ = zeros(T.nVars, T.nVars)
    n₀₋ = zeros(T.nVars, T.nPast_not_future_and_mixed)
    n₋₋ = zeros(T.nPast_not_future_and_mixed, T.nPast_not_future_and_mixed)
    nₚ₋ = zeros(T.nPresent_only, T.nPast_not_future_and_mixed)
    nₜₚ = zeros(T.nVars - T.nPresent_only, T.nPast_not_future_and_mixed)
    
    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
    Qinv = Q.Q'

    mul!(n₀₊, Qinv, ∇₊)
    mul!(n₀₀, Qinv, ∇₀)
    mul!(n₀₋, Qinv, ∇₋)
    A₊ = n₀₊
    A₀ = n₀₀
    A₋ = n₀₋

    dynIndex = T.nPresent_only+1:T.nVars

    Ã₊  = A₊[dynIndex,:]
    Ã₋  = A₋[dynIndex,:]
    Ã₀₊ = A₀[dynIndex, T.future_not_past_and_mixed_idx]
    @views mul!(nₜₚ, A₀[dynIndex, T.past_not_future_idx], ℒ.I(T.nPast_not_future_and_mixed)[T.not_mixed_in_past_idx,:])
    Ã₀₋ = nₜₚ

    Z₊ = zeros(T.nMixed, T.nFuture_not_past_and_mixed)
    I₊ = ℒ.I(T.nFuture_not_past_and_mixed)[T.mixed_in_future_idx,:]

    Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
    I₋ = ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

    D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))

    ℒ.rmul!(Ã₋,-1)
    ℒ.rmul!(Ã₀₊,-1)
    E = vcat(hcat(Ã₋,Ã₀₊), hcat(Z₋, I₊))

    # this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
    schdcmp = try
        ℒ.schur!(D, E)
    catch
        return zeros(T.nVars,T.nPast_not_future_and_mixed), false
    end

    if explosive # returns false for NaN gen. eigenvalue which is correct here bc they are > 1
        eigenselect = abs.(schdcmp.β ./ schdcmp.α) .>= 1

        ℒ.ordschur!(schdcmp, eigenselect)

        Z₂₁ = schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
        Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
        T₁₁    = schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        Ẑ₁₁ = RF.lu(Z₁₁, check = false)

        if !ℒ.issuccess(Ẑ₁₁)
            Ẑ₁₁ = ℒ.svd(Z₁₁, check = false)
        end

        if !ℒ.issuccess(Ẑ₁₁)
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    else
        eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

        try
            ℒ.ordschur!(schdcmp, eigenselect)
        catch
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end

        Z₂₁ = schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
        Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
        T₁₁    = schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        Ẑ₁₁ = RF.lu(Z₁₁, check = false)

        if !ℒ.issuccess(Ẑ₁₁)
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    end

    if VERSION >= v"1.9"
        Ŝ₁₁ = RF.lu!(S₁₁, check = false)
    else
        Ŝ₁₁ = RF.lu(S₁₁, check = false)
    end

    if !ℒ.issuccess(Ŝ₁₁)
        return zeros(T.nVars,T.nPast_not_future_and_mixed), false
    end

    # D      = Z₂₁ / Ẑ₁₁
    ℒ.rdiv!(Z₂₁, Ẑ₁₁)
    D = Z₂₁

    # L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
    ℒ.ldiv!(Ŝ₁₁, T₁₁)
    mul!(n₋₋, Z₁₁, T₁₁)
    ℒ.rdiv!(n₋₋, Ẑ₁₁)
    L = n₋₋

    sol = vcat(L[T.not_mixed_in_past_idx,:], D)

    Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = A₋[1:T.nPresent_only,:]

    
    if VERSION >= v"1.9"
        Ā̂₀ᵤ = RF.lu!(Ā₀ᵤ, check = false)
    else
        Ā̂₀ᵤ = RF.lu(Ā₀ᵤ, check = false)
    end

    if !ℒ.issuccess(Ā̂₀ᵤ)
        return zeros(T.nVars,T.nPast_not_future_and_mixed), false
    #     Ā̂₀ᵤ = ℒ.svd(collect(Ā₀ᵤ))
    end

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    if T.nPresent_only > 0
        mul!(A₋ᵤ, Ã₀ᵤ, sol[T.dynamic_order,:], 1, 1)
        mul!(nₚ₋, A₊ᵤ, D)
        mul!(A₋ᵤ, nₚ₋, L, 1, 1)
        ℒ.ldiv!(Ā̂₀ᵤ, A₋ᵤ)
        ℒ.rmul!(A₋ᵤ,-1)
    end
    A    = vcat(A₋ᵤ, sol)

    return A[T.reorder,:], true
end




function riccati_forward(∇₁::Matrix{ℱ.Dual{Z,S,N}}; T::timings, explosive::Bool = false) where {Z,S,N}
    # unpack: AoS -> SoA
    ∇̂₁ = ℱ.value.(∇₁)

    expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    A = ∇̂₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    B = ∇̂₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

    x, solved = riccati_forward(∇̂₁;T = T, explosive = explosive)

    X = x * expand[2]

    AXB = A * X + B
    
    AXBfact = RF.lu(AXB, check = false)

    if !ℒ.issuccess(AXBfact)
        AXBfact = ℒ.svd(AXB)
    end

    invAXB = inv(AXBfact)

    AA = invAXB * A

    X² = X * X

    X̃ = zeros(length(x), N)

    p = zero(∇̂₁)

    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        p .= ℱ.partials.(∇₁, i)

        dA = p[:,1:T.nFuture_not_past_and_mixed] * expand[1]
        dB = p[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
        dC = p[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
    
        CC = invAXB * (dA * X² + dC + dB * X)
    
        dX, solved = solve_sylvester_equation(AA, -X, -CC, sylvester_algorithm = :sylvester)

        X̃[:,i] = vec(dX[:,T.past_not_future_and_mixed_idx])
    end
    
    return reshape(map(x, eachrow(X̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x)), solved
end


function rrule(::typeof(riccati_forward), ∇₁; T, explosive = false)
    # Forward pass to compute the output and intermediate values needed for the backward pass
    A, solved = riccati_forward(∇₁, T = T, explosive = explosive)

    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
                    ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    Â = A * expand[2]
    
    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

    ∂∇₁ = zero(∇₁)
    
    invtmp = inv(-Â' * ∇₊' - ∇₀')
    
    tmp2 = invtmp * ∇₊'

    function first_order_solution_pullback(∂A)
        tmp1 = invtmp * ∂A[1] * expand[2]

        ss, solved = solve_sylvester_equation(tmp2, Â', -tmp1, sylvester_algorithm = :sylvester)

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .= (ss * Â' * Â')[:,T.future_not_past_and_mixed_idx]
        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .= ss * Â'
        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] .= ss[:,T.past_not_future_and_mixed_idx]

        return NoTangent(), ∂∇₁
    end

    return (A, solved), first_order_solution_pullback
end



function calculate_first_order_solution(∇₁::Matrix{Float64}; 
                                        T::timings, 
                                        explosive::Bool = false)::Tuple{Matrix{Float64}, Bool}
    A, solved = riccati_forward(∇₁; T = T, explosive = explosive)

    if !solved
        return hcat(A, zeros(size(A,1),T.nExo)), solved
    end

    Jm = @view(ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:])
    
    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:]
    ∇₀ = copy(∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)])
    ∇ₑ = copy(∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end])
    
    M = similar(∇₀)
    mul!(M, A, Jm)
    mul!(∇₀, ∇₊, M, 1, 1)
    C = RF.lu!(∇₀, check = false)
    # C = RF.lu!(∇₊ * A * Jm + ∇₀, check = false)
    
    if !ℒ.issuccess(C)
        return hcat(A, zeros(size(A,1),T.nExo)), solved
    end
    
    ℒ.ldiv!(C, ∇ₑ)
    ℒ.rmul!(∇ₑ, -1)
    # B = -(C \ ∇ₑ) # otherwise Zygote doesnt diff it

    return hcat(A, ∇ₑ), solved
end



function rrule(::typeof(calculate_first_order_solution), ∇₁; T, explosive = false)
    # Forward pass to compute the output and intermediate values needed for the backward pass
    𝐒ᵗ, solved = riccati_forward(∇₁, T = T, explosive = explosive)

    if !solved
        return (hcat(𝐒ᵗ, zeros(size(𝐒ᵗ,1),T.nExo)), solved), x -> NoTangent(), NoTangent(), NoTangent()
    end

    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
                    ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇ₑ = @view ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    
    M̂ = RF.lu(∇₊ * 𝐒ᵗ * expand[2] + ∇₀, check = false)
    
    if !ℒ.issuccess(M̂)
        return (hcat(𝐒ᵗ, zeros(size(𝐒ᵗ,1),T.nExo)), solved), x -> NoTangent(), NoTangent(), NoTangent()
    end
    
    M = inv(M̂)
    
    𝐒ᵉ = -M * ∇ₑ # otherwise Zygote doesnt diff it

    𝐒̂ᵗ = 𝐒ᵗ * expand[2]
   
    tmp2 = -M' * ∇₊'

    function first_order_solution_pullback(∂𝐒) 
        ∂∇₁ = zero(∇₁)

        ∂𝐒ᵗ = ∂𝐒[1][:,1:T.nPast_not_future_and_mixed]
        ∂𝐒ᵉ = ∂𝐒[1][:,T.nPast_not_future_and_mixed + 1:end]

        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1:end] .= -M' * ∂𝐒ᵉ

        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .= M' * ∂𝐒ᵉ * ∇ₑ' * M'

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .= (M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]' * 𝐒ᵗ')[:,T.future_not_past_and_mixed_idx]

        ∂𝐒ᵗ .+= ∇₊' * M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]'

        tmp1 = M' * ∂𝐒ᵗ * expand[2]

        ss, solved = solve_sylvester_equation(tmp2, 𝐒̂ᵗ', -tmp1, sylvester_algorithm = :sylvester)

        if !solved
            NoTangent(), NoTangent(), NoTangent()
        end

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .+= (ss * 𝐒̂ᵗ' * 𝐒̂ᵗ')[:,T.future_not_past_and_mixed_idx]
        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .+= ss * 𝐒̂ᵗ'
        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] .+= ss[:,T.past_not_future_and_mixed_idx]

        return NoTangent(), ∂∇₁, NoTangent()
    end

    return (hcat(𝐒ᵗ, 𝐒ᵉ), solved), first_order_solution_pullback
end

function calculate_first_order_solution(∇₁::Matrix{ℱ.Dual{Z,S,N}}; T::timings, explosive::Bool = false)::Tuple{Matrix{ℱ.Dual{Z,S,N}},Bool} where {Z,S,N}
    A, solved = riccati_forward(∇₁; T = T, explosive = explosive)

    if !solved
        return hcat(A, zeros(size(A,1),T.nExo)), solved
    end

    Jm = @view(ℒ.diagm(ones(S,T.nVars))[T.past_not_future_and_mixed_idx,:])
    
    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(S,T.nVars))[T.future_not_past_and_mixed_idx,:]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇ₑ = @view ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    B = -((∇₊ * A * Jm + ∇₀) \ ∇ₑ)

    return hcat(A, B), solved
end


function separate_values_and_partials_from_sparsevec_dual(V::SparseVector{ℱ.Dual{Z,S,N}}; tol::AbstractFloat = eps()) where {Z,S,N}
    nrows = length(V)
    ncols = length(V.nzval[1].partials)

    rows = Int[]
    cols = Int[]

    prtls = Float64[]

    for (i,v) in enumerate(V.nzind)
        for (k,w) in enumerate(V.nzval[i].partials)
            if abs(w) > tol
                push!(rows,v)
                push!(cols,k)
                push!(prtls,w)
            end
        end
    end

    vvals = sparsevec(V.nzind,[i.value for i in V.nzval],nrows)
    ps = sparse(rows,cols,prtls,nrows,ncols)

    return vvals, ps
end


function calculate_second_order_solution(∇₁::AbstractMatrix{<: Real}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
                                            𝑺₁::AbstractMatrix{<: Real},#first order solution
                                            M₂::second_order_auxilliary_matrices;  # aux matrices
                                            T::timings,
                                            sylvester_algorithm::Symbol = :doubling,
                                            tol::AbstractFloat = eps(),
                                            verbose::Bool = false)
    # inspired by Levintal

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
    
    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];


    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = ∇₂ * sparse(ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 
    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = (mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + mat_mult_kron(∇₂, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 

    X = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
    droptol!(X,tol)

    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    B = spinv * ∇₁₊
    droptol!(B,tol)

    C = (M₂.𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + M₂.𝐔₂ * M₂.𝛔) * M₂.𝐂₂
    droptol!(C,tol)

    𝐒₂, solved = solve_sylvester_equation(B, C, X, sylvester_algorithm = sylvester_algorithm, verbose = verbose)

    𝐒₂ = sparse(𝐒₂)

    if !solved
        return 𝐒₂, solved
    end

    𝐒₂ *= M₂.𝐔₂

    return 𝐒₂, solved
end




function rrule(::typeof(calculate_second_order_solution), 
                    ∇₁::AbstractMatrix{<: Real}, #first order derivatives
                    ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
                    𝑺₁::AbstractMatrix{<: Real},#first order solution
                    M₂::second_order_auxilliary_matrices;  # aux matrices
                    T::timings,
                    sylvester_algorithm::Symbol = :doubling,
                    tol::AbstractFloat = eps(),
                    verbose::Bool = false)
    # inspired by Levintal

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
    
    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];


    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = ∇₂ * sparse(ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 
    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = (mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + mat_mult_kron(∇₂, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 

    X = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
    droptol!(X,tol)

    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    B = spinv * ∇₁₊
    droptol!(B,tol)

    C = (M₂.𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + M₂.𝐔₂ * M₂.𝛔) * M₂.𝐂₂
    droptol!(C,tol)

    𝐒₂, solved = solve_sylvester_equation(B, C, X, sylvester_algorithm = sylvester_algorithm, verbose = verbose)

    𝐒₂ = sparse(𝐒₂)

    if !solved
        return 𝐒₂, solved
    end

    # 𝐒₂ *= M₂.𝐔₂

    Bt = sparse(B')

    Ct = sparse(C')
    
    function second_order_solution_pullback(∂𝐒₂_solved) 
        ∂∇₁ = zero(∇₁)
        ∂𝐒₁ = zero(𝐒₁)
        ∂spinv = zero(spinv)
        ∂𝐒₁₋╱𝟏ₑ = zero(𝐒₁₋╱𝟏ₑ)
        ∂𝐒₁₊╱𝟎 = zero(𝐒₁₊╱𝟎)
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = zero(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)

        ∂𝐒₂ = ∂𝐒₂_solved[1]
        
        # droptol!(∂𝐒₂, eps())

        ∂𝐒₂ *= M₂.𝐔₂'

        ∂X, solved = solve_sylvester_equation(Bt, Ct, ∂𝐒₂, sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

        ∂X = sparse(∂X)

        ∂B = ∂X * C' * 𝐒₂'

        ∂C = 𝐒₂' * B' * ∂X

        # C = (M₂.𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + M₂.𝐔₂ * M₂.𝛔) * M₂.𝐂₂
        ∂kron𝐒₁₋╱𝟏ₑ = M₂.𝐔₂' * ∂C * M₂.𝐂₂'

        re∂kron𝐒₁₋╱𝟏ₑ = reshape(∂kron𝐒₁₋╱𝟏ₑ, size(𝐒₁₋╱𝟏ₑ,1), size(𝐒₁₋╱𝟏ₑ,1), size(𝐒₁₋╱𝟏ₑ,2), size(𝐒₁₋╱𝟏ₑ,2))

        ei = 1
        for e in eachslice(re∂kron𝐒₁₋╱𝟏ₑ; dims = (1,3))
            ∂𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(𝐒₁₋╱𝟏ₑ,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kron𝐒₁₋╱𝟏ₑ; dims = (2,4))
            ∂𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(𝐒₁₋╱𝟏ₑ,e)
            ei += 1
        end

        # B = spinv * ∇₁₊
        ∂∇₁₊ = spinv' * ∂B
        ∂spinv += ∂B * ∇₁₊'
        
        # ∇₁₊ =  sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])
        ∂∇₁[:,1:n₊] += ∂∇₁₊ * spdiagm(ones(n))[i₊,:]'

        # X = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
        ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = spinv' * ∂X
        ∂spinv += ∂X * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹'


        # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = ∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) * M₂.𝐂₂  + ∇₂ * ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔 * M₂.𝐂₂
        # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = (mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + mat_mult_kron(∇₂, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 
        ∂∇₂ = ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ * M₂.𝐂₂' * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)' + ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ * M₂.𝐂₂' * M₂.𝛔' * ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)'
        
        ∂kron𝐒₁₊╱𝟎 = ∇₂' * ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ * M₂.𝐂₂' * M₂.𝛔'

        re∂kron𝐒₁₊╱𝟎 = reshape(∂kron𝐒₁₊╱𝟎, size(𝐒₁₊╱𝟎,1), size(𝐒₁₊╱𝟎,1), size(𝐒₁₊╱𝟎,2), size(𝐒₁₊╱𝟎,2))

        ei = 1
        for e in eachslice(re∂kron𝐒₁₊╱𝟎; dims = (1,3))
            ∂𝐒₁₊╱𝟎[ei] += ℒ.dot(𝐒₁₊╱𝟎,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kron𝐒₁₊╱𝟎; dims = (2,4))
            ∂𝐒₁₊╱𝟎[ei] += ℒ.dot(𝐒₁₊╱𝟎,e)
            ei += 1
        end
        
        ∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = ∇₂' * ∂∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ * M₂.𝐂₂'

        re∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = reshape(∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,1), size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,1), size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,2), size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,2))

        ei = 1
        for e in eachslice(re∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋; dims = (1,3))
            ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[ei] += ℒ.dot(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, e)
            ei += 1
        end
        
        ei = 1
        for e in eachslice(re∂kron⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋; dims = (2,4))
            ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[ei] += ℒ.dot(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, e)
            ei += 1
        end

        # spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
        ∂∇₁₊𝐒₁➕∇₁₀ = -spinv' * ∂spinv * spinv'

        # ∇₁₊𝐒₁➕∇₁₀ =  -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]
        ∂∇₁[:,1:n₊] -= ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.diagm(ones(n))[i₋,:]' * 𝐒₁[i₊,1:n₋]'
        ∂∇₁[:,range(1,n) .+ n₊] -= ∂∇₁₊𝐒₁➕∇₁₀

        ∂𝐒₁[i₊,1:n₋] -= ∇₁[:,1:n₊]' * ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.diagm(ones(n))[i₋,:]'

        # 𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
        #                 zeros(n₋ + n + nₑ, nₑ₋)];
        ∂𝐒₁[i₊,:] += ∂𝐒₁₊╱𝟎[1:length(i₊),:]

        ###### ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ =  [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
        # ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ =  [ℒ.I(size(𝐒₁,1))[i₊,:] * 𝐒₁ * 𝐒₁₋╱𝟏ₑ
        #                     𝐒₁
        #                     spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];
        ∂𝐒₁ += spdiagm(ones(size(𝐒₁,1)))[:,i₊] * ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[1:length(i₊),:] * 𝐒₁₋╱𝟏ₑ'
        ∂𝐒₁ += ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[length(i₊) .+ (1:size(𝐒₁,1)),:]
        
        ∂𝐒₁₋╱𝟏ₑ += 𝐒₁' * spdiagm(ones(size(𝐒₁,1)))[:,i₊] * ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[1:length(i₊),:]

        # 𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
        ∂𝐒₁[i₋,:] += ∂𝐒₁₋╱𝟏ₑ[1:length(i₋), :]

        # 𝐒₁ = [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]
        ∂𝑺₁ = [∂𝐒₁[:,1:n₋] ∂𝐒₁[:,n₋+2:end]]

        return NoTangent(), ∂∇₁, ∂∇₂, ∂𝑺₁, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return (𝐒₂ * M₂.𝐔₂, solved), second_order_solution_pullback
end



function calculate_third_order_solution(∇₁::AbstractMatrix{<: Real}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
                                            ∇₃::SparseMatrixCSC{<: Real}, #third order derivatives
                                            𝑺₁::AbstractMatrix{<: Real}, #first order solution
                                            𝐒₂::SparseMatrixCSC{<: Real}, #second order solution
                                            M₂::second_order_auxilliary_matrices,  # aux matrices second order
                                            M₃::third_order_auxilliary_matrices;  # aux matrices third order
                                            T::timings,
                                            sylvester_algorithm::Symbol = :doubling,
                                            tol::AbstractFloat = eps(),
                                            verbose::Bool = false)
    # inspired by Levintal

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]


    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    B = spinv * ∇₁₊
    droptol!(B,tol)


    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)
    
    # C = M₃.𝐔₃ * tmpkron + M₃.𝐔₃ * M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐔₃ * M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃
    # C += M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) # no speed up here from A_mult_kron_power_3_B; this is the bottleneck. ideally have this return reduced space directly. TODO: make kron3 faster
    # C *= M₃.𝐂₃
    C = M₃.𝐔₃ * (tmpkron + M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃ + ℒ.kron(𝐒₁₋╱𝟏ₑ, kron𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃
    droptol!(C,tol)
    
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    # kronaux = ℒ.kron(aux, aux)
    # 𝐗₃ = ∇₃ * ℒ.kron(kronaux, aux)
    𝐗₃ = A_mult_kron_power_3_B(∇₃, aux)

    tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)
    out = ∇₃ * tmpkron + ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron * M₃.𝐏₁ᵣ̃  + ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron * M₃.𝐏₂ᵣ̃
    𝐗₃ += out
    
    tmpkron10 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
    tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
    tmpkron11 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * M₂.𝛔)

    𝐗₃ += ∇₂ * (
      tmpkron10
    + tmpkron1 * tmpkron2
    + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
    + tmpkron11
     ) * M₃.𝐏# |> findnz
    
    𝐗₃ += @views ∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]) * M₃.𝐏
    droptol!(𝐗₃,tol)
    
    X = spinv * 𝐗₃ * M₃.𝐂₃
    droptol!(X,tol)
    
    𝐒₃, solved = solve_sylvester_equation(B, C, X, sylvester_algorithm = sylvester_algorithm, verbose= verbose)
    
    𝐒₃ = sparse(𝐒₃)

    if !solved
        return 𝐒₃, solved
    end

    𝐒₃ *= M₃.𝐔₃

    return 𝐒₃, solved
end




function rrule(::typeof(calculate_third_order_solution), 
                ∇₁::AbstractMatrix{<: Real}, #first order derivatives
                ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
                ∇₃::SparseMatrixCSC{<: Real}, #third order derivatives
                𝑺₁::AbstractMatrix{<: Real}, #first order solution
                𝐒₂::SparseMatrixCSC{<: Real}, #second order solution
                M₂::second_order_auxilliary_matrices,  # aux matrices second order
                M₃::third_order_auxilliary_matrices;  # aux matrices third order
                T::timings,
                sylvester_algorithm::Symbol = :doubling,
                tol::AbstractFloat = eps(),
                verbose::Bool = false)    
    # inspired by Levintal

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]


    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    B = spinv * ∇₁₊
    droptol!(B,tol)

    
    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)
    
    # C = M₃.𝐔₃ * tmpkron + M₃.𝐔₃ * M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐔₃ * M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃
    # C += M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ, kron𝐒₁₋╱𝟏ₑ) # no speed up here from A_mult_kron_power_3_B; this is the bottleneck. ideally have this return reduced space directly. TODO: make kron3 faster
    # C *= M₃.𝐂₃
    C = M₃.𝐔₃ * (tmpkron + M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃ + ℒ.kron(𝐒₁₋╱𝟏ₑ, kron𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃
    droptol!(C,tol)
    
    
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    kronaux = ℒ.kron(aux, aux)
    𝐗₃ = ∇₃ * ℒ.kron(kronaux, aux)
    # 𝐗₃ = A_mult_kron_power_3_B(∇₃, aux)

    tmpkron0 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔
    tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, tmpkron0)
    out = ∇₃ * tmpkron22 + ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron22 * M₃.𝐏₁ᵣ̃  + ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron22 * M₃.𝐏₂ᵣ̃
    𝐗₃ += out
    
    tmpkron10 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
    tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
    𝐒₂₊╱𝟎𝛔 = 𝐒₂₊╱𝟎 * M₂.𝛔
    tmpkron11 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎𝛔)

    𝐗₃ += ∇₂ * (tmpkron10
     + tmpkron1 * tmpkron2
     + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
     + tmpkron11
     ) * M₃.𝐏# |> findnz
    
    𝐒₂₋╱𝟎 = @views [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    tmpkron12 = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
    𝐗₃ += ∇₁₊ * 𝐒₂ * tmpkron12 * M₃.𝐏
    droptol!(𝐗₃,tol)
    
    X = spinv * 𝐗₃ * M₃.𝐂₃
    droptol!(X,tol)
    

    𝐒₃, solved = solve_sylvester_equation(B, C, X, sylvester_algorithm = sylvester_algorithm, verbose = verbose)
    
    𝐒₃ = sparse(𝐒₃)

    if !solved
        return (𝐒₃, solved), x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() 
    end

    Bt = sparse(B')

    Ct = sparse(C')

    function third_order_solution_pullback(∂𝐒₃_solved) 
        ∂∇₁ = zero(∇₁)
        ∂∇₂ = zero(∇₂)
        ∂∇₃ = zero(∇₃)
        ∂𝐒₁ = zero(𝐒₁)
        ∂𝐒₂ = zero(𝐒₂)
        ∂spinv = zero(spinv)
        ∂𝐒₁₋╱𝟏ₑ = zero(𝐒₁₋╱𝟏ₑ)
        ∂kron𝐒₁₋╱𝟏ₑ = zero(kron𝐒₁₋╱𝟏ₑ)
        ∂𝐒₁₊╱𝟎 = zero(𝐒₁₊╱𝟎)
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = zero(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋)
        ∂tmpkron = zero(tmpkron)
        ∂tmpkron22 = zero(tmpkron22)
        ∂kronaux = zero(kronaux)
        ∂aux = zero(aux)
        ∂tmpkron0 = zero(tmpkron0)
        ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = zero(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
        ∂𝐒₂₊╱𝟎 = zero(𝐒₂₊╱𝟎)
        ∂𝐒₂₊╱𝟎𝛔 = zero(𝐒₂₊╱𝟎𝛔)
        ∂∇₁₊ = zero(∇₁₊)
        ∂𝐒₂₋╱𝟎 = zero(𝐒₂₋╱𝟎)


        ∂𝐒₃ = ∂𝐒₃_solved[1]

        ∂𝐒₃ *= M₃.𝐔₃'

        ∂X, solved = solve_sylvester_equation(Bt, Ct, ∂𝐒₃, sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

        ∂X = sparse(∂X)

        ∂B = ∂X * C' * 𝐒₃'

        ∂C = 𝐒₃' * B' * ∂X

        # X = spinv * 𝐗₃ * M₃.𝐂₃
        ∂𝐗₃ = spinv' * ∂X * M₃.𝐂₃'
        ∂spinv += ∂X * M₃.𝐂₃' * 𝐗₃'

        # 𝐗₃ = ∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux) 
        # + ∇₃ * tmpkron22 
        # + ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron22 * M₃.𝐏₁ᵣ̃ 
        # + ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron22 * M₃.𝐏₂ᵣ̃
        # + ∇₂ * (tmpkron10 + tmpkron1 * tmpkron2 + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ + tmpkron11) * M₃.𝐏
        # + ∇₁₊ * 𝐒₂ * tmpkron12 * M₃.𝐏

        # ∇₁₊ * 𝐒₂ * tmpkron12 * M₃.𝐏
        ∂∇₁₊ += ∂𝐗₃ * M₃.𝐏' * tmpkron12' * 𝐒₂'
        ∂𝐒₂ += ∇₁₊' * ∂𝐗₃ * M₃.𝐏' * tmpkron12'
        ∂tmpkron12 = 𝐒₂' * ∇₁₊' * ∂𝐗₃ * M₃.𝐏'

        # tmpkron12 = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎)
        re∂tmpkron12 = reshape(∂tmpkron12, 
                                size(𝐒₂₋╱𝟎,1), 
                                size(𝐒₁₋╱𝟏ₑ,1), 
                                size(𝐒₂₋╱𝟎,2),
                                size(𝐒₁₋╱𝟏ₑ,2))

        ei = 1
        for e in eachslice(re∂tmpkron12; dims = (1,3))
            ∂𝐒₂₋╱𝟎[ei] += ℒ.dot(𝐒₁₋╱𝟏ₑ,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂tmpkron12; dims = (2,4))
            ∂𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(𝐒₂₋╱𝟎,e)
            ei += 1
        end
        
        
        # ∇₂ * (tmpkron10 + tmpkron1 * tmpkron2 + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ + tmpkron11) * M₃.𝐏

        ∂∇₂ += ∂𝐗₃ * M₃.𝐏' * (
           tmpkron10
         + tmpkron1 * tmpkron2
         + tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
         + tmpkron11
         )'

        ∂tmpkron10 = ∇₂' * ∂𝐗₃ * M₃.𝐏'

        # tmpkron10 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
        re∂tmpkron10 = reshape(∂tmpkron10, 
                                size(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎,1), 
                                size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,1), 
                                size(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎,2),
                                size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,2))

        ei = 1
        for e in eachslice(re∂tmpkron10; dims = (1,3))
            ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎[ei] += ℒ.dot(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂tmpkron10; dims = (2,4))
            ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[ei] += ℒ.dot(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎,e)
            ei += 1
        end


        ∂tmpkron1 = ∇₂' * ∂𝐗₃ * M₃.𝐏' * tmpkron2' + ∇₂' * ∂𝐗₃ * M₃.𝐏' * M₃.𝐏₁ᵣ' * tmpkron2' * M₃.𝐏₁ₗ'

        ∂tmpkron2 = tmpkron1' * ∇₂' * ∂𝐗₃ * M₃.𝐏' + M₃.𝐏₁ₗ' * tmpkron1' * ∇₂' * ∂𝐗₃ * M₃.𝐏' * M₃.𝐏₁ᵣ'

        ∂tmpkron11 = ∇₂' * ∂𝐗₃ * M₃.𝐏'

        # tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
        re∂tmpkron1 = reshape(∂tmpkron1, 
                                size(𝐒₂₊╱𝟎,1), 
                                size(𝐒₁₊╱𝟎,1), 
                                size(𝐒₂₊╱𝟎,2),
                                size(𝐒₁₊╱𝟎,2))

        ei = 1
        for e in eachslice(re∂tmpkron1; dims = (1,3))
            ∂𝐒₂₊╱𝟎[ei] += ℒ.dot(𝐒₁₊╱𝟎,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂tmpkron1; dims = (2,4))
            ∂𝐒₁₊╱𝟎[ei] += ℒ.dot(𝐒₂₊╱𝟎,e)
            ei += 1
        end


        # tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
        re∂tmpkron2 = reshape(∂tmpkron2, 
                                size(𝐒₁₋╱𝟏ₑ,1), 
                                size(M₂.𝛔,1), 
                                size(𝐒₁₋╱𝟏ₑ,2),
                                size(M₂.𝛔,2))

        ei = 1
        for e in eachslice(re∂tmpkron2; dims = (1,3))
            ∂𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(M₂.𝛔,e)
            ei += 1
        end


        # tmpkron11 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * M₂.𝛔)
        re∂tmpkron11 = reshape(∂tmpkron11, 
                                size(𝐒₂₊╱𝟎𝛔,1), 
                                size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,1), 
                                size(𝐒₂₊╱𝟎𝛔,2),
                                size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,2))

        ei = 1
        for e in eachslice(re∂tmpkron11; dims = (1,3))
            ∂𝐒₂₊╱𝟎𝛔[ei] += ℒ.dot(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂tmpkron11; dims = (2,4))
            ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[ei] += ℒ.dot(𝐒₂₊╱𝟎𝛔,e)
            ei += 1
        end

        ∂𝐒₂₊╱𝟎 += ∂𝐒₂₊╱𝟎𝛔 * M₂.𝛔'


        # out = ∇₃ * tmpkron22 
        # + ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron22 * M₃.𝐏₁ᵣ̃ 
        # + ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron22 * M₃.𝐏₂ᵣ̃

        ∂∇₃ += ∂𝐗₃ * tmpkron22' + ∂𝐗₃ * M₃.𝐏₁ᵣ̃' * tmpkron22' * M₃.𝐏₁ₗ̂' + ∂𝐗₃ * M₃.𝐏₂ᵣ̃' * tmpkron22' * M₃.𝐏₂ₗ̂'

        ∂tmpkron22 += ∇₃' * ∂𝐗₃ + M₃.𝐏₁ₗ̂' * ∇₃' * ∂𝐗₃ * M₃.𝐏₁ᵣ̃' + M₃.𝐏₂ₗ̂' * ∇₃' * ∂𝐗₃ * M₃.𝐏₂ᵣ̃'

        # tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)
        
        re∂tmpkron22 = reshape(∂tmpkron22, 
                                size(tmpkron0,1), 
                                size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,1), 
                                size(tmpkron0,2),
                                size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,2))

        ei = 1
        for e in eachslice(re∂tmpkron22; dims = (1,3))
            ∂tmpkron0[ei] += ℒ.dot(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂tmpkron22; dims = (2,4))
            ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[ei] += ℒ.dot(tmpkron0,e)
            ei += 1
        end

        ∂kron𝐒₁₊╱𝟎 = ∂tmpkron0 * M₂.𝛔'

        re∂kron𝐒₁₊╱𝟎 = reshape(∂kron𝐒₁₊╱𝟎, 
                                size(𝐒₁₊╱𝟎,1), 
                                size(𝐒₁₊╱𝟎,1), 
                                size(𝐒₁₊╱𝟎,2), 
                                size(𝐒₁₊╱𝟎,2))

        ei = 1
        for e in eachslice(re∂kron𝐒₁₊╱𝟎; dims = (2,4))
            ∂𝐒₁₊╱𝟎[ei] += ℒ.dot(𝐒₁₊╱𝟎,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kron𝐒₁₊╱𝟎; dims = (1,3))
            ∂𝐒₁₊╱𝟎[ei] += ℒ.dot(𝐒₁₊╱𝟎,e)
            ei += 1
        end

        # -∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux)
        ∂∇₃ += ∂𝐗₃ * ℒ.kron(ℒ.kron(aux, aux), aux)'
        ∂kronkronaux = ∇₃' * ∂𝐗₃

        re∂kronkronaux = reshape(∂kronkronaux, 
                                size(aux,1), 
                                size(kronaux,1), 
                                size(aux,2),
                                size(kronaux,2))

        ei = 1
        for e in eachslice(re∂kronkronaux; dims = (1,3))
            ∂aux[ei] += ℒ.dot(kronaux,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kronkronaux; dims = (2,4))
            ∂kronaux[ei] += ℒ.dot(aux,e)
            ei += 1
        end


        re∂kronaux = reshape(∂kronaux, 
                            size(aux,1), 
                            size(aux,1), 
                            size(aux,2), 
                            size(aux,2))

        ei = 1
        for e in eachslice(re∂kronaux; dims = (2,4))
            ∂aux[ei] += ℒ.dot(aux,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kronaux; dims = (1,3))
            ∂aux[ei] += ℒ.dot(aux,e)
            ei += 1
        end

        # aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋
        ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ += M₃.𝐒𝐏' * ∂aux

        # 𝐒₂₋╱𝟎 = @views [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]
        ∂𝐒₂[i₋,:] += ∂𝐒₂₋╱𝟎[1:length(i₋),:]

        # 𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
        #     zeros(n₋ + n + nₑ, nₑ₋^2)]
        ∂𝐒₂[i₊,:] += ∂𝐒₂₊╱𝟎[1:length(i₊),:]


        # ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = [
            ## (𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            ## ℒ.diagm(ones(n))[i₊,:] * (𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])
            # ℒ.diagm(ones(n))[i₊,:] * 𝐒₂k𝐒₁₋╱𝟏ₑ
            # 𝐒₂
            # zeros(n₋ + nₑ, nₑ₋^2)
        # ];
        ∂𝐒₂k𝐒₁₋╱𝟏ₑ = ℒ.diagm(ones(n))[i₊,:]' * ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎[1:length(i₊),:]

        ∂𝐒₂ += ∂⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎[length(i₊) .+ (1:size(𝐒₂,1)),:]

        ∂𝐒₂ += ∂𝐒₂k𝐒₁₋╱𝟏ₑ * kron𝐒₁₋╱𝟏ₑ'

        ∂kron𝐒₁₋╱𝟏ₑ += 𝐒₂' * ∂𝐒₂k𝐒₁₋╱𝟏ₑ

        
        # 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)]
        # 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * 𝐒₂╱𝟎
        ∂𝐒₁ += ∂𝐒₂k𝐒₁₋╱𝟏ₑ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)]'
        
        # ∂𝐒₂[i₋,:] += spdiagm(ones(size(𝐒₂,1)))[i₋,:]' * 𝐒₁' * ∂𝐒₂k𝐒₁₋╱𝟏ₑ[1:length(i₋),:]
        ∂𝐒₂╱𝟎 = 𝐒₁' * ∂𝐒₂k𝐒₁₋╱𝟏ₑ
        ∂𝐒₂[i₋,:] += ∂𝐒₂╱𝟎[1:length(i₋),:]

        ###
        # C = M₃.𝐔₃ * (tmpkron + M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃ + ℒ.kron(𝐒₁₋╱𝟏ₑ, kron𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃
        ∂tmpkron += M₃.𝐔₃' * ∂C * M₃.𝐂₃'
        ∂tmpkron += M₃.𝐏₁ₗ̄' * M₃.𝐔₃' * ∂C * M₃.𝐂₃' * M₃.𝐏₁ᵣ̃'
        ∂tmpkron += M₃.𝐏₂ₗ̄' * M₃.𝐔₃' * ∂C * M₃.𝐂₃' * M₃.𝐏₂ᵣ̃'

        ∂kronkron𝐒₁₋╱𝟏ₑ = M₃.𝐔₃' * ∂C * M₃.𝐂₃'

        re∂kronkron𝐒₁₋╱𝟏ₑ = reshape(∂kronkron𝐒₁₋╱𝟏ₑ, 
                                    size(kron𝐒₁₋╱𝟏ₑ,1), 
                                    size(𝐒₁₋╱𝟏ₑ,1), 
                                    size(kron𝐒₁₋╱𝟏ₑ,2), 
                                    size(𝐒₁₋╱𝟏ₑ,2))

        ei = 1
        for e in eachslice(re∂kronkron𝐒₁₋╱𝟏ₑ; dims = (2,4))
            ∂𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(kron𝐒₁₋╱𝟏ₑ,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kronkron𝐒₁₋╱𝟏ₑ; dims = (1,3))
            ∂kron𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(𝐒₁₋╱𝟏ₑ,e)
            ei += 1
        end

        re∂kron𝐒₁₋╱𝟏ₑ = reshape(∂kron𝐒₁₋╱𝟏ₑ, 
                                size(𝐒₁₋╱𝟏ₑ,1), 
                                size(𝐒₁₋╱𝟏ₑ,1), 
                                size(𝐒₁₋╱𝟏ₑ,2), 
                                size(𝐒₁₋╱𝟏ₑ,2))

        ei = 1
        for e in eachslice(re∂kron𝐒₁₋╱𝟏ₑ; dims = (2,4))
            ∂𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(𝐒₁₋╱𝟏ₑ,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kron𝐒₁₋╱𝟏ₑ; dims = (1,3))
            ∂𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(𝐒₁₋╱𝟏ₑ,e)
            ei += 1
        end 

        # tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
        re∂tmpkron = reshape(∂tmpkron, 
                            size(M₂.𝛔,1), # this needs to correspond to the second entry in the kron call
                            size(𝐒₁₋╱𝟏ₑ,1),  # this needs to correspond to the first entry in the kron call
                            size(M₂.𝛔,2), 
                            size(𝐒₁₋╱𝟏ₑ,2))

        ei = 1
        for e in eachslice(re∂tmpkron; dims = (2,4))
            ∂𝐒₁₋╱𝟏ₑ[ei] += ℒ.dot(M₂.𝛔,e)
            ei += 1
        end


        # B = spinv * ∇₁₊
        ∂∇₁₊ += spinv' * ∂B
        ∂spinv += ∂B * ∇₁₊'
        
        # ∇₁₊ =  sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])
        ∂∇₁[:,1:n₊] += ∂∇₁₊ * spdiagm(ones(n))[i₊,:]'

        # spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
        ∂∇₁₊𝐒₁➕∇₁₀ = -spinv' * ∂spinv * spinv'

        # ∇₁₊𝐒₁➕∇₁₀ =  -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]
        ∂∇₁[:,1:n₊] -= ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.diagm(ones(n))[i₋,:]' * 𝐒₁[i₊,1:n₋]'
        ∂∇₁[:,range(1,n) .+ n₊] -= ∂∇₁₊𝐒₁➕∇₁₀

        ∂𝐒₁[i₊,1:n₋] -= ∇₁[:,1:n₊]' * ∂∇₁₊𝐒₁➕∇₁₀ * ℒ.diagm(ones(n))[i₋,:]'

        # # 𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
        # #                 zeros(n₋ + n + nₑ, nₑ₋)];
        ∂𝐒₁[i₊,:] += ∂𝐒₁₊╱𝟎[1:length(i₊),:]

        # ###### ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ =  [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
        # # ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ =  [ℒ.I(size(𝐒₁,1))[i₊,:] * 𝐒₁ * 𝐒₁₋╱𝟏ₑ
        # #                     𝐒₁
        # #                     spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];
        ∂𝐒₁ += spdiagm(ones(size(𝐒₁,1)))[:,i₊] * ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[1:length(i₊),:] * 𝐒₁₋╱𝟏ₑ'
        ∂𝐒₁ += ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[length(i₊) .+ (1:size(𝐒₁,1)),:]
        
        ∂𝐒₁₋╱𝟏ₑ += 𝐒₁' * spdiagm(ones(size(𝐒₁,1)))[:,i₊] * ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[1:length(i₊),:]

        # 𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
        ∂𝐒₁[i₋,:] += ∂𝐒₁₋╱𝟏ₑ[1:length(i₋), :]

        # 𝐒₁ = [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]
        ∂𝑺₁ = [∂𝐒₁[:,1:n₋] ∂𝐒₁[:,n₋+2:end]]

        return NoTangent(), ∂∇₁, ∂∇₂, ∂∇₃, ∂𝑺₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return (𝐒₃ * M₃.𝐔₃, solved), third_order_solution_pullback
end



function irf(state_update::Function, 
    obc_state_update::Function,
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64},
    T::timings; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    negative_shock::Bool = false)

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks which are not part of the model."
        
        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,T)
    end

    var_idx = parse_variables_input_to_index(variables, T)

    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    always_solved = true

    if shocks == :simulate
        shock_history = randn(T.nExo,periods)

        shock_history[contains.(string.(T.exo),"ᵒᵇᶜ"),:] .= 0

        Y = zeros(T.nVars,periods,1)

        past_states = initial_state
        
        for t in 1:periods
            past_states, past_shocks, solved  = obc_state_update(past_states, shock_history[:,t], state_update)

            if !solved @warn "No solution in period: $t" end#. Possible reasons: 1. infeasability 2. too long spell of binding constraint. To address the latter try setting max_obc_horizon to a larger value (default: 40): @model <name> max_obc_horizon=40 begin ... end" end

            always_solved = always_solved && solved

            if !always_solved break end

            Y[:,t,1] = pruning ? sum(past_states) : past_states

            shock_history[:,t] = past_shocks
        end

        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = [:simulate])
    elseif shocks == :none
        Y = zeros(T.nVars,periods,1)

        shck = T.nExo == 0 ? Vector{Float64}(undef, 0) : zeros(T.nExo)
        
        past_states = initial_state
        
        for t in 1:periods
            past_states, _, solved  = obc_state_update(past_states, shck, state_update)

            if !solved @warn "No solution in period: $t" end#. Possible reasons: 1. infeasability 2. too long spell of binding constraint. To address the latter try setting max_obc_horizon to a larger value (default: 40): @model <name> max_obc_horizon=40 begin ... end" end

            always_solved = always_solved && solved

            if !always_solved break end

            Y[:,t,1] = pruning ? sum(past_states) : past_states
        end

        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = [:none])
    else
        Y = zeros(T.nVars,periods,length(shock_idx))

        for (i,ii) in enumerate(shock_idx)
            if shocks != :simulate && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -1 : 1
            end

            past_states = initial_state
            
            for t in 1:periods
                past_states, past_shocks, solved = obc_state_update(past_states, shock_history[:,t], state_update)

                if !solved @warn "No solution in period: $t" end#. Possible reasons: 1. infeasability 2. too long spell of binding constraint. To address the latter try setting max_obc_horizon to a larger value (default: 40): @model <name> max_obc_horizon=40 begin ... end" end

                always_solved = always_solved && solved

                if !always_solved break end

                Y[:,t,i] = pruning ? sum(past_states) : past_states

                shock_history[:,t] = past_shocks
            end
        end

        axis2 = shocks isa Union{Symbol_input,String_input} ? 
                    shock_idx isa Int ? 
                        [T.exo[shock_idx]] : 
                    T.exo[shock_idx] : 
                [:Shock_matrix]

        if any(x -> contains(string(x), "◖"), axis2)
            axis2_decomposed = decompose_name.(axis2)
            axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        end
    
        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = axis2)
    end
end




function irf(state_update::Function, 
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64},
    T::timings; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    negative_shock::Bool = false)

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks which are not part of the model."
        
        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,T)
    end

    var_idx = parse_variables_input_to_index(variables, T)

    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    if shocks == :simulate
        shock_history = randn(T.nExo,periods)

        shock_history[contains.(string.(T.exo),"ᵒᵇᶜ"),:] .= 0

        Y = zeros(T.nVars,periods,1)

        initial_state = state_update(initial_state,shock_history[:,1])

        Y[:,1,1] = pruning ? sum(initial_state) : initial_state

        for t in 1:periods-1
            initial_state = state_update(initial_state,shock_history[:,t+1])

            Y[:,t+1,1] = pruning ? sum(initial_state) : initial_state
        end

        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = [:simulate])
    elseif shocks == :none
        Y = zeros(T.nVars,periods,1)

        shck = T.nExo == 0 ? Vector{Float64}(undef, 0) : zeros(T.nExo)

        initial_state = state_update(initial_state, shck)

        Y[:,1,1] = pruning ? sum(initial_state) : initial_state

        for t in 1:periods-1
            initial_state = state_update(initial_state, shck)

            Y[:,t+1,1] = pruning ? sum(initial_state) : initial_state
        end

        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = [:none])
    else
        Y = zeros(T.nVars,periods,length(shock_idx))

        for (i,ii) in enumerate(shock_idx)
            initial_state_copy = deepcopy(initial_state)
            
            if shocks != :simulate && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -1 : 1
            end

            initial_state_copy = state_update(initial_state_copy, shock_history[:,1])

            Y[:,1,i] = pruning ? sum(initial_state_copy) : initial_state_copy

            for t in 1:periods-1
                initial_state_copy = state_update(initial_state_copy, shock_history[:,t+1])

                Y[:,t+1,i] = pruning ? sum(initial_state_copy) : initial_state_copy
            end
        end

        axis2 = shocks isa Union{Symbol_input,String_input} ? 
                    shock_idx isa Int ? 
                        [T.exo[shock_idx]] : 
                    T.exo[shock_idx] : 
                [:Shock_matrix]
        
        if any(x -> contains(string(x), "◖"), axis2)
            axis2_decomposed = decompose_name.(axis2)
            axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        end
    
        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = axis2)
    end
end



function girf(state_update::Function,
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64}, 
    T::timings; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    negative_shock::Bool = false, 
    warmup_periods::Int = 100, 
    draws::Int = 50)

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks which are not part of the model."

        shock_history = zeros(T.nExo, periods + 1)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,T)
    end

    var_idx = parse_variables_input_to_index(variables, T)

    Y = zeros(T.nVars, periods + 1, length(shock_idx))

    for (i,ii) in enumerate(shock_idx)
        initial_state_copy = deepcopy(initial_state)

        for draw in 1:draws
            initial_state_copy² = deepcopy(initial_state_copy)

            for i in 1:warmup_periods
                initial_state_copy² = state_update(initial_state_copy², randn(T.nExo))
            end

            Y₁ = zeros(T.nVars, periods + 1)
            Y₂ = zeros(T.nVars, periods + 1)

            baseline_noise = randn(T.nExo)

            if shocks != :simulate && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -1 : 1
            end

            if pruning
                initial_state_copy² = state_update(initial_state_copy², baseline_noise)

                initial_state₁ = deepcopy(initial_state_copy²)
                initial_state₂ = deepcopy(initial_state_copy²)

                Y₁[:,1] = initial_state_copy² |> sum
                Y₂[:,1] = initial_state_copy² |> sum
            else
                Y₁[:,1] = state_update(initial_state_copy², baseline_noise)
                Y₂[:,1] = state_update(initial_state_copy², baseline_noise)
            end

            for t in 1:periods
                baseline_noise = randn(T.nExo)

                if pruning
                    initial_state₁ = state_update(initial_state₁, baseline_noise)
                    initial_state₂ = state_update(initial_state₂, baseline_noise + shock_history[:,t])

                    Y₁[:,t+1] = initial_state₁ |> sum
                    Y₂[:,t+1] = initial_state₂ |> sum
                else
                    Y₁[:,t+1] = state_update(Y₁[:,t],baseline_noise)
                    Y₂[:,t+1] = state_update(Y₂[:,t],baseline_noise + shock_history[:,t])
                end
            end

            Y[:,:,i] += Y₂ - Y₁
        end
        Y[:,:,i] /= draws
    end
    
    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    axis2 = shocks isa Union{Symbol_input,String_input} ? 
                shock_idx isa Int ? 
                    [T.exo[shock_idx]] : 
                T.exo[shock_idx] : 
            [:Shock_matrix]

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    return KeyedArray(Y[var_idx,2:end,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = axis2)
end


function parse_variables_input_to_index(variables::Union{Symbol_input,String_input}, T::timings)
    
    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    if variables == :all_excluding_auxilliary_and_obc
        return indexin(setdiff(T.var[.!contains.(string.(T.var),"ᵒᵇᶜ")],T.aux),sort(union(T.var,T.aux,T.exo_present)))
        # return indexin(setdiff(setdiff(T.var,T.exo_present),T.aux),sort(union(T.var,T.aux,T.exo_present)))
    elseif variables == :all_excluding_obc
        return indexin(T.var[.!contains.(string.(T.var),"ᵒᵇᶜ")],sort(union(T.var,T.aux,T.exo_present)))
    elseif variables == :all
        return 1:length(union(T.var,T.aux,T.exo_present))
    elseif variables isa Matrix{Symbol}
        if length(setdiff(variables,T.var)) > 0
            return @warn "Following variables are not part of the model: " * join(string.(setdiff(variables,T.var)),", ")
        end
        return getindex(1:length(T.var),convert(Vector{Bool},vec(sum(variables .== T.var,dims= 2))))
    elseif variables isa Vector{Symbol}
        if length(setdiff(variables,T.var)) > 0
            return @warn "Following variables are not part of the model: " * join(string.(setdiff(variables,T.var)),", ")
        end
        return getindex(1:length(T.var),convert(Vector{Bool},vec(sum(reshape(variables,1,length(variables)) .== T.var,dims= 2))))
    elseif variables isa Tuple{Symbol,Vararg{Symbol}}
        if length(setdiff(variables,T.var)) > 0
            return @warn "Following variables are not part of the model: " * join(string.(setdiff(Symbol.(collect(variables)),T.var)), ", ")
        end
        return getindex(1:length(T.var),convert(Vector{Bool},vec(sum(reshape(collect(variables),1,length(variables)) .== T.var,dims= 2))))
    elseif variables isa Symbol
        if length(setdiff([variables],T.var)) > 0
            return @warn "Following variable is not part of the model: " * join(string(setdiff([variables],T.var)[1]),", ")
        end
        return getindex(1:length(T.var),variables .== T.var)
    else
        return @warn "Invalid argument in variables"
    end
end


function parse_shocks_input_to_index(shocks::Union{Symbol_input,String_input}, T::timings)#::Union{UnitRange{Int64}, Int64, Vector{Int64}}
    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks == :all
        shock_idx = 1:T.nExo
    elseif shocks == :all_excluding_obc
        shock_idx = findall(.!contains.(string.(T.exo),"ᵒᵇᶜ"))
    elseif shocks == :none
        shock_idx = 1
    elseif shocks == :simulate
        shock_idx = 1
    elseif shocks isa Matrix{Symbol}
        if length(setdiff(shocks,T.exo)) > 0
            @warn "Following shocks are not part of the model: " * join(string.(setdiff(shocks,T.exo)),", ")
            shock_idx = Int64[]
        else
            shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(shocks .== T.exo,dims= 2))))
        end
    elseif shocks isa Vector{Symbol}
        if length(setdiff(shocks,T.exo)) > 0
            @warn "Following shocks are not part of the model: " * join(string.(setdiff(shocks,T.exo)),", ")
            shock_idx = Int64[]
        else
            shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(shocks,1,length(shocks)) .== T.exo, dims= 2))))
        end
    elseif shocks isa Tuple{Symbol, Vararg{Symbol}}
        if length(setdiff(shocks,T.exo)) > 0
            @warn "Following shocks are not part of the model: " * join(string.(setdiff(Symbol.(collect(shocks)),T.exo)),", ")
            shock_idx = Int64[]
        else
            shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(collect(shocks),1,length(shocks)) .== T.exo,dims= 2))))
        end
    elseif shocks isa Symbol
        if length(setdiff([shocks],T.exo)) > 0
            @warn "Following shock is not part of the model: " * join(string(setdiff([shocks],T.exo)[1]),", ")
            shock_idx = Int64[]
        else
            shock_idx = getindex(1:T.nExo,shocks .== T.exo)
        end
    else
        @warn "Invalid argument in shocks"
        shock_idx = Int64[]
    end
    return shock_idx
end


function parse_algorithm_to_state_update(algorithm::Symbol, 𝓂::ℳ, occasionally_binding_constraints::Bool)::Tuple{Function,Bool}
    if occasionally_binding_constraints
        if :linear_time_iteration == algorithm
            state_update = 𝓂.solution.perturbation.linear_time_iteration.state_update_obc
            pruning = false
        elseif algorithm ∈ [:riccati, :first_order]
            state_update = 𝓂.solution.perturbation.first_order.state_update_obc
            pruning = false
        elseif :second_order == algorithm
            state_update = 𝓂.solution.perturbation.second_order.state_update_obc
            pruning = false
        elseif :pruned_second_order == algorithm
            state_update = 𝓂.solution.perturbation.pruned_second_order.state_update_obc
            pruning = true
        elseif :third_order == algorithm
            state_update = 𝓂.solution.perturbation.third_order.state_update_obc
            pruning = false
        elseif :pruned_third_order == algorithm
            state_update = 𝓂.solution.perturbation.pruned_third_order.state_update_obc
            pruning = true
        end
    else
        if :linear_time_iteration == algorithm
            state_update = 𝓂.solution.perturbation.linear_time_iteration.state_update
            pruning = false
        elseif algorithm ∈ [:riccati, :first_order]
            state_update = 𝓂.solution.perturbation.first_order.state_update
            pruning = false
        elseif :second_order == algorithm
            state_update = 𝓂.solution.perturbation.second_order.state_update
            pruning = false
        elseif :pruned_second_order == algorithm
            state_update = 𝓂.solution.perturbation.pruned_second_order.state_update
            pruning = true
        elseif :third_order == algorithm
            state_update = 𝓂.solution.perturbation.third_order.state_update
            pruning = false
        elseif :pruned_third_order == algorithm
            state_update = 𝓂.solution.perturbation.pruned_third_order.state_update
            pruning = true
        end
    end

    return state_update, pruning
end



function calculate_covariance(parameters::Vector{<: Real}, 
                                𝓂::ℳ; 
                                lyapunov_algorithm::Symbol = :doubling, 
                                verbose::Bool = false)
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, verbose = verbose)
    
	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) 

    sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    A = @views sol[:, 1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    C = @views sol[:, 𝓂.timings.nPast_not_future_and_mixed+1:end]
    
    CC = C * C'

    covar_raw, _ = solve_lyapunov_equation(A, CC, lyapunov_algorithm = lyapunov_algorithm, verbose = verbose)

    return covar_raw, sol , ∇₁, SS_and_pars
end




function calculate_mean(parameters::Vector{T}, 
                        𝓂::ℳ; 
                        verbose::Bool = false, 
                        algorithm = :pruned_second_order, 
                        sylvester_algorithm::Symbol = :doubling, 
                        tol::Float64 = eps()) where T <: Real
    # Theoretical mean identical for 2nd and 3rd order pruned solution.
    @assert algorithm ∈ [:linear_time_iteration, :riccati, :first_order, :quadratic_iteration, :binder_pesaran, :pruned_second_order, :pruned_third_order] "Theoretical mean only available for first order, pruned second and third order perturbation solutions."

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, verbose = verbose)
    
    if algorithm ∈ [:linear_time_iteration, :riccati, :first_order, :quadratic_iteration, :binder_pesaran]
        return SS_and_pars[1:𝓂.timings.nVars], solution_error
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix
    
    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂) * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂
    
    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

    nᵉ = 𝓂.timings.nExo
    nˢ = 𝓂.timings.nPast_not_future_and_mixed

    s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
    v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))
    
    kron_states     = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_shocks     = ℒ.kron(e_in_s⁺, e_in_s⁺)
    kron_volatility = ℒ.kron(v_in_s⁺, v_in_s⁺)

    # first order
    states_to_variables¹ = sparse(𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed])

    states_to_states¹ = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx, 1:𝓂.timings.nPast_not_future_and_mixed]
    shocks_to_states¹ = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx, (𝓂.timings.nPast_not_future_and_mixed + 1):end]

    # second order
    states_to_variables²        = 𝐒₂[:, kron_states]
    shocks_to_variables²        = 𝐒₂[:, kron_shocks]
    volatility_to_variables²    = 𝐒₂[:, kron_volatility]

    states_to_states²       = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx, kron_states] |> collect
    shocks_to_states²       = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx, kron_shocks]
    volatility_to_states²   = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx, kron_volatility]

    kron_states_to_states¹ = ℒ.kron(states_to_states¹, states_to_states¹) |> collect
    kron_shocks_to_states¹ = ℒ.kron(shocks_to_states¹, shocks_to_states¹)

    n_sts = 𝓂.timings.nPast_not_future_and_mixed

    # Set up in pruned state transition matrices
    pruned_states_to_pruned_states = [  states_to_states¹       zeros(T,n_sts, n_sts)   zeros(T,n_sts, n_sts^2)
                                        zeros(T,n_sts, n_sts)   states_to_states¹       states_to_states² / 2
                                        zeros(T,n_sts^2, 2 * n_sts)                     kron_states_to_states¹   ]

    pruned_states_to_variables = [states_to_variables¹  states_to_variables¹  states_to_variables² / 2]

    pruned_states_vol_and_shock_effect = [  zeros(T,n_sts) 
                                            vec(volatility_to_states²) / 2 + shocks_to_states² / 2 * vec(ℒ.I(𝓂.timings.nExo))
                                            kron_shocks_to_states¹ * vec(ℒ.I(𝓂.timings.nExo))]

    variables_vol_and_shock_effect = (vec(volatility_to_variables²) + shocks_to_variables² * vec(ℒ.I(𝓂.timings.nExo))) / 2

    ## First-order moments, ie mean of variables
    mean_of_pruned_states   = (ℒ.I - pruned_states_to_pruned_states) \ pruned_states_vol_and_shock_effect
    mean_of_variables   = SS_and_pars[1:𝓂.timings.nVars] + pruned_states_to_variables * mean_of_pruned_states + variables_vol_and_shock_effect
    
    return mean_of_variables, 𝐒₁, ∇₁, 𝐒₂, ∇₂
end




function calculate_second_order_moments(
    parameters::Vector{<: Real}, 
    𝓂::ℳ; 
    covariance::Bool = true,
    verbose::Bool = false, 
    sylvester_algorithm::Symbol = :doubling,
    lyapunov_algorithm::Symbol = :doubling,
    tol::AbstractFloat = eps())

    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose, lyapunov_algorithm = lyapunov_algorithm)

    nᵉ = 𝓂.timings.nExo

    nˢ = 𝓂.timings.nPast_not_future_and_mixed

    iˢ = 𝓂.timings.past_not_future_and_mixed_idx

    Σᶻ₁ = Σʸ₁[iˢ, iˢ]

    # precalc second order
    ## mean
    I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

    ## covariance
    E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

    quadrup = multiplicate(nᵉ, 4)

    comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

    comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

    for j = 1:size(comb⁴,1)
        E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
    end

    e⁴ = quadrup * E_e⁴

    # second order
    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂) * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol, sylvester_algorithm = sylvester_algorithm, verbose = verbose)

    s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
    v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
    kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
    kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

    # first order
    s_to_y₁ = 𝐒₁[:, 1:nˢ]
    e_to_y₁ = 𝐒₁[:, (nˢ + 1):end]
    
    s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
    e_to_s₁ = 𝐒₁[iˢ, (nˢ + 1):end]


    # second order
    s_s_to_y₂ = 𝐒₂[:, kron_s_s]
    e_e_to_y₂ = 𝐒₂[:, kron_e_e]
    v_v_to_y₂ = 𝐒₂[:, kron_v_v]
    s_e_to_y₂ = 𝐒₂[:, kron_s_e]

    s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
    e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
    v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
    s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

    s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
    e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
    s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

    # # Set up in pruned state transition matrices
    ŝ_to_ŝ₂ = [ s_to_s₁             zeros(nˢ, nˢ + nˢ^2)
                zeros(nˢ, nˢ)       s_to_s₁             s_s_to_s₂ / 2
                zeros(nˢ^2, 2*nˢ)   s_to_s₁_by_s_to_s₁                  ]

    ê_to_ŝ₂ = [ e_to_s₁         zeros(nˢ, nᵉ^2 + nᵉ * nˢ)
                zeros(nˢ,nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
                zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁]

    ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2]

    ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

    ŝv₂ = [ zeros(nˢ) 
            vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec(ℒ.I(nᵉ))
            e_to_s₁_by_e_to_s₁ * vec(ℒ.I(nᵉ))]

    yv₂ = (vec(v_v_to_y₂) + e_e_to_y₂ * vec(ℒ.I(nᵉ))) / 2

    ## Mean
    μˢ⁺₂ = (ℒ.I - ŝ_to_ŝ₂) \ ŝv₂
    Δμˢ₂ = vec((ℒ.I - s_to_s₁) \ (s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂ + e_e_to_s₂ * vec(ℒ.I(nᵉ))) / 2))
    μʸ₂  = SS_and_pars[1:𝓂.timings.nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

    if !covariance
        return μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂
    end

    # Covariance
    Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
            zeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     zeros(nᵉ^2, nᵉ * nˢ)
            zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

    C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

    Σᶻ₂, info = solve_lyapunov_equation(ŝ_to_ŝ₂, C, lyapunov_algorithm = lyapunov_algorithm, verbose = verbose)

    # if Σᶻ₂ isa DenseMatrix
    #     Σᶻ₂ = sparse(Σᶻ₂)
    # end

    Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'

    autocorr_tmp = ŝ_to_ŝ₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_ŝ₂ * Γ₂ * ê_to_y₂'

    return Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂
end






function calculate_third_order_moments(parameters::Vector{T}, 
                                            observables::Union{Symbol_input,String_input},
                                            𝓂::ℳ; 
                                            covariance::Bool = true,
                                            autocorrelation::Bool = false,
                                            autocorrelation_periods::U = 1:5,
                                            verbose::Bool = false, 
                                            dependencies_tol::AbstractFloat = 1e-12, 
                                            sylvester_algorithm::Symbol = :doubling,
                                            lyapunov_algorithm::Symbol = :doubling,
                                            tol::AbstractFloat = eps()) where {U, T <: Real}

    second_order_moments = calculate_second_order_moments(parameters, 
                                                            𝓂, 
                                                            verbose = verbose, 
                                                            sylvester_algorithm = sylvester_algorithm, 
                                                            lyapunov_algorithm = lyapunov_algorithm)

    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = second_order_moments

    if !covariance && !autocorrelation
        return μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂
    end

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂) * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
                                                𝓂.solution.perturbation.second_order_auxilliary_matrices, 
                                                𝓂.solution.perturbation.third_order_auxilliary_matrices; 
                                                T = 𝓂.timings, 
                                                sylvester_algorithm = sylvester_algorithm,
                                                tol = tol, 
                                                verbose= verbose)

    orders = determine_efficient_order(𝐒₁, 𝓂.timings, observables, tol = dependencies_tol)

    nᵉ = 𝓂.timings.nExo

    # precalc second order
    ## covariance
    E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

    quadrup = multiplicate(nᵉ, 4)

    comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

    comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

    for j = 1:size(comb⁴,1)
        E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
    end

    e⁴ = quadrup * E_e⁴

    # precalc third order
    sextup = multiplicate(nᵉ, 6)
    E_e⁶ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4 * (nᵉ + 4)÷5 * (nᵉ + 5)÷6)

    comb⁶   = reduce(vcat, generateSumVectors(nᵉ, 6))

    comb⁶ = comb⁶ isa Int64 ? reshape([comb⁶],1,1) : comb⁶

    for j = 1:size(comb⁶,1)
        E_e⁶[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁶[j,:])
    end

    e⁶ = sextup * E_e⁶

    Σʸ₃ = zeros(T, size(Σʸ₂))

    if autocorrelation
        autocorr = zeros(T, size(Σʸ₂,1), length(autocorrelation_periods))
    end

    # Threads.@threads for ords in orders 
    for ords in orders 
        variance_observable, dependencies_all_vars = ords

        sort!(variance_observable)

        sort!(dependencies_all_vars)

        dependencies = intersect(𝓂.timings.past_not_future_and_mixed, dependencies_all_vars)

        obs_in_y = indexin(variance_observable, 𝓂.timings.var)

        dependencies_in_states_idx = indexin(dependencies, 𝓂.timings.past_not_future_and_mixed)

        dependencies_in_var_idx = Int.(indexin(dependencies, 𝓂.timings.var))

        nˢ = length(dependencies)

        iˢ = dependencies_in_var_idx

        Σ̂ᶻ₁ = Σʸ₁[iˢ, iˢ]

        dependencies_extended_idx = vcat(dependencies_in_states_idx, 
                dependencies_in_states_idx .+ 𝓂.timings.nPast_not_future_and_mixed, 
                findall(ℒ.kron(𝓂.timings.past_not_future_and_mixed .∈ (intersect(𝓂.timings.past_not_future_and_mixed,dependencies),), 𝓂.timings.past_not_future_and_mixed .∈ (intersect(𝓂.timings.past_not_future_and_mixed,dependencies),))) .+ 2*𝓂.timings.nPast_not_future_and_mixed)
        
        Σ̂ᶻ₂ = Σᶻ₂[dependencies_extended_idx, dependencies_extended_idx]
        
        Δ̂μˢ₂ = Δμˢ₂[dependencies_in_states_idx]

        s_in_s⁺ = BitVector(vcat(𝓂.timings.past_not_future_and_mixed .∈ (dependencies,), zeros(Bool, nᵉ + 1)))
        e_in_s⁺ = BitVector(vcat(zeros(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), ones(Bool, nᵉ)))
        v_in_s⁺ = BitVector(vcat(zeros(Bool, 𝓂.timings.nPast_not_future_and_mixed), 1, zeros(Bool, nᵉ)))

        # precalc second order
        ## mean
        I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

        e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ*nˢ)), nˢ*nᵉ^2, nˢ*nᵉ^2))
        e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ*nˢ^2, nᵉ*nˢ^2))
        ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
        s_s  = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))

        # first order
        s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
        e_to_y₁ = 𝐒₁[obs_in_y,:][:, (𝓂.timings.nPast_not_future_and_mixed + 1):end]
        
        s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
        e_to_s₁ = 𝐒₁[iˢ, (𝓂.timings.nPast_not_future_and_mixed + 1):end]

        # second order
        kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
        kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
        kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

        s_s_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_s]
        e_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_e_e]
        s_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_e]

        s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
        e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
        v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
        s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

        s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
        e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
        s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

        # third order
        kron_s_v = ℒ.kron(s_in_s⁺, v_in_s⁺)
        kron_e_v = ℒ.kron(e_in_s⁺, v_in_s⁺)

        s_s_s_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, s_in_s⁺)]
        s_s_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, e_in_s⁺)]
        s_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_e, e_in_s⁺)]
        e_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_e, e_in_s⁺)]
        s_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_v, v_in_s⁺)]
        e_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_v, v_in_s⁺)]

        s_s_s_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, s_in_s⁺)]
        s_s_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, e_in_s⁺)]
        s_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_e, e_in_s⁺)]
        e_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_e, e_in_s⁺)]
        s_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_v, v_in_s⁺)]
        e_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_v, v_in_s⁺)]

        # Set up pruned state transition matrices
        ŝ_to_ŝ₃ = [  s_to_s₁                zeros(nˢ, 2*nˢ + 2*nˢ^2 + nˢ^3)
                                            zeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2   zeros(nˢ, nˢ + nˢ^2 + nˢ^3)
                                            zeros(nˢ^2, 2 * nˢ)               s_to_s₁_by_s_to_s₁  zeros(nˢ^2, nˢ + nˢ^2 + nˢ^3)
                                            s_v_v_to_s₃ / 2    zeros(nˢ, nˢ + nˢ^2)      s_to_s₁       s_s_to_s₂    s_s_s_to_s₃ / 6
                                            ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    zeros(nˢ^2, 2*nˢ + nˢ^2)     s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2)    
                                            zeros(nˢ^3, 3*nˢ + 2*nˢ^2)   ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁)]

        ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(nˢ^2, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                        ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                        zeros(nˢ^3, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss   ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es  ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6]

        ê_to_y₃ = [e_to_y₁ + e_v_v_to_y₃ / 2  e_e_to_y₂ / 2  s_e_to_y₂   s_e_to_y₂     s_s_e_to_y₃ / 2    s_e_e_to_y₃ / 2    e_e_e_to_y₃ / 6]

        μˢ₃δμˢ₁ = reshape((ℒ.I - s_to_s₁_by_s_to_s₁) \ vec( 
                                    (s_s_to_s₂  * reshape(ss_s * vec(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂'),nˢ^2, nˢ) +
                                    s_s_s_to_s₃ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ^3, nˢ) / 6 +
                                    s_e_e_to_s₃ * ℒ.kron(Σ̂ᶻ₁, vec(ℒ.I(nᵉ))) / 2 +
                                    s_v_v_to_s₃ * Σ̂ᶻ₁ / 2) * s_to_s₁' +
                                    (s_e_to_s₂  * ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ)) +
                                    e_e_e_to_s₃ * reshape(e⁴, nᵉ^3, nᵉ) / 6 +
                                    s_s_e_to_s₃ * ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ)) / 2 +
                                    e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                    ), nˢ, nˢ)

        Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σ̂ᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    reshape(e⁴, nᵉ, nᵉ^3)
                spzeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σ̂ᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δ̂μˢ₂ * Δ̂μˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δ̂μˢ₂, reshape(e⁴, nᵉ, nᵉ^3))
                ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σ̂ᶻ₁), reshape(e⁴, nᵉ, nᵉ^3))
                spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σ̂ᶻ₁, reshape(e⁴, nᵉ^2, nᵉ^2))    spzeros(nˢ*nᵉ^2,nᵉ^3)
                reshape(e⁴, nᵉ^3, nᵉ)  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', reshape(e⁴, nᵉ^3, nᵉ))     ℒ.kron(vec(Σ̂ᶻ₁)', reshape(e⁴, nᵉ^3, nᵉ))  spzeros(nᵉ^3, nˢ*nᵉ^2)     reshape(e⁶, nᵉ^3, nᵉ^3)]


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
                ℒ.kron(Σ̂ᶻ₁,vec(ℒ.I(nᵉ)))   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec(ℒ.I(nᵉ)))    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec(ℒ.I(nᵉ)))  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec(ℒ.I(nᵉ)))
                spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]
        
        droptol!(ŝ_to_ŝ₃, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())
        
        A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
        droptol!(A, eps())

        C = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A + A'
        droptol!(C, eps())

        Σᶻ₃, info = solve_lyapunov_equation(ŝ_to_ŝ₃, C, lyapunov_algorithm = lyapunov_algorithm, verbose = verbose)

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃' + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

        for obs in variance_observable
            Σʸ₃[indexin([obs], 𝓂.timings.var), indexin(variance_observable, 𝓂.timings.var)] = Σʸ₃tmp[indexin([obs], variance_observable), :]
        end

        if autocorrelation
            autocorr_tmp = ŝ_to_ŝ₃ * Eᴸᶻ' * ê_to_y₃' + ê_to_ŝ₃ * Γ₃ * ê_to_y₃'

            s_to_s₁ⁱ = zero(s_to_s₁)
            s_to_s₁ⁱ += ℒ.diagm(ones(nˢ))

            ŝ_to_ŝ₃ⁱ = zero(ŝ_to_ŝ₃)
            ŝ_to_ŝ₃ⁱ += ℒ.diagm(ones(size(Σᶻ₃,1)))

            Σᶻ₃ⁱ = Σᶻ₃

            for i in autocorrelation_periods
                Σᶻ₃ⁱ .= ŝ_to_ŝ₃ * Σᶻ₃ⁱ + ê_to_ŝ₃ * Eᴸᶻ
                s_to_s₁ⁱ *= s_to_s₁

                Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
                ℒ.kron(s_to_s₁ⁱ * Σ̂ᶻ₁,vec(ℒ.I(nᵉ)))   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(s_to_s₁ⁱ * μˢ₃δμˢ₁',vec(ℒ.I(nᵉ)))    ℒ.kron(s_to_s₁ⁱ * reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec(ℒ.I(nᵉ)))  ℒ.kron(s_to_s₁ⁱ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec(ℒ.I(nᵉ)))
                spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]

                for obs in variance_observable
                    autocorr[indexin([obs], 𝓂.timings.var), i] .= (ℒ.diag(ŝ_to_y₃ * Σᶻ₃ⁱ * ŝ_to_y₃' + ŝ_to_y₃ * ŝ_to_ŝ₃ⁱ * autocorr_tmp + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃') ./ ℒ.diag(Σʸ₃tmp))[indexin([obs], variance_observable)]
                end

                ŝ_to_ŝ₃ⁱ *= ŝ_to_ŝ₃
            end

        end
    end

    if autocorrelation
        return Σʸ₃, μʸ₂, autocorr, SS_and_pars
    else
        return Σʸ₃, μʸ₂, SS_and_pars
    end

end

function find_variables_to_exclude(𝓂::ℳ, observables::Vector{Symbol})
    # reduce system
    vars_to_exclude = setdiff(𝓂.timings.present_only, observables)

    # Mapping variables to their equation index
    variable_to_equation = Dict{Symbol, Vector{Int}}()
    for var in vars_to_exclude
        for (eq_idx, vars_set) in enumerate(𝓂.dyn_var_present_list)
        # for var in vars_set
            if var in vars_set
                if haskey(variable_to_equation, var)
                    push!(variable_to_equation[var],eq_idx)
                else
                    variable_to_equation[var] = [eq_idx]
                end
            end
        end
    end

    return variable_to_equation
end


function create_broadcaster(indices::Vector{Int}, n::Int)
    broadcaster = spzeros(n, length(indices))
    for (i, vid) in enumerate(indices)
        broadcaster[vid,i] = 1.0
    end
    return broadcaster  
end


# Specialization for :kalman filter
function calculate_loglikelihood(::Val{:kalman}, algorithm, observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations, filter_algorithm, verbose)
    return calculate_kalman_filter_loglikelihood(observables, 𝐒, data_in_deviations, TT, presample_periods = presample_periods, initial_covariance = initial_covariance, verbose = verbose)
end

# Specialization for :inversion filter
function calculate_loglikelihood(::Val{:inversion}, algorithm, observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations, filter_algorithm, verbose)
    return calculate_inversion_filter_loglikelihood(Val(algorithm), state, 𝐒, data_in_deviations, observables, TT, warmup_iterations = warmup_iterations, presample_periods = presample_periods, filter_algorithm = filter_algorithm)
end

function get_NSSS_and_parameters(𝓂::ℳ, parameter_values::Vector{S}; verbose::Bool = false, tol::AbstractFloat = 1e-12) where S <: Float64
    𝓂.SS_solve_func(parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)
end


function rrule(::typeof(get_NSSS_and_parameters), 𝓂, parameter_values; verbose = false,  tol::AbstractFloat = 1e-12)
    SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)

    if solution_error > tol || isnan(solution_error)
        return (SS_and_pars, (solution_error, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
        
    SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
    
    # unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
    unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))
    # ∂SS_equations_∂parameters = try 𝓂.∂SS_equations_∂parameters(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
    # catch
    #     return (SS_and_pars, (10, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    # end

    X = [parameter_values; SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]]
    
    # vals = Float64[]

    # for f in 𝓂.∂SS_equations_∂parameters[1]
    #     push!(vals, f(X)...)
    # end
    
    vals = zeros(Float64, length(𝓂.∂SS_equations_∂parameters[1]))

    # lk = ReentrantLock()

    Polyester.@batch minbatch = 200 for f in 𝓂.∂SS_equations_∂parameters[1]
        out = f(X)
        
        # begin
        #     lock(lk)
        #     try
                @inbounds vals[out[2]] = out[1]
        #     finally
        #         unlock(lk)
        #     end
        # end
    end

    Accessors.@reset 𝓂.∂SS_equations_∂parameters[2].nzval = vals
    
    ∂SS_equations_∂parameters = 𝓂.∂SS_equations_∂parameters[2]

    # vals = Float64[]

    # for f in 𝓂.∂SS_equations_∂SS_and_pars[1]
    #     push!(vals, f(X)...)
    # end

    vals = zeros(Float64, length(𝓂.∂SS_equations_∂SS_and_pars[1]))

    # lk = ReentrantLock()

    Polyester.@batch minbatch = 200 for f in 𝓂.∂SS_equations_∂SS_and_pars[1]
        out = f(X)
        
        # begin
        #     lock(lk)
        #     try
                @inbounds vals[out[2]] = out[1]
        #     finally
        #         unlock(lk)
        #     end
        # end
    end

    𝓂.∂SS_equations_∂SS_and_pars[3] .*= 0
    𝓂.∂SS_equations_∂SS_and_pars[3][𝓂.∂SS_equations_∂SS_and_pars[2]] .+= vals

    ∂SS_equations_∂SS_and_pars = 𝓂.∂SS_equations_∂SS_and_pars[3]

    # ∂SS_equations_∂parameters = 𝓂.∂SS_equations_∂parameters(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
    # ∂SS_equations_∂SS_and_pars = 𝓂.∂SS_equations_∂SS_and_pars(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
    
    ∂SS_equations_∂SS_and_pars_lu = RF.lu!(∂SS_equations_∂SS_and_pars, check = false)

    if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
        return (SS_and_pars, (10, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]

    jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))
    
    for (i,v) in enumerate(SS_and_pars_names)
        if v in unknowns
            jvp[i,:] = JVP[indexin([v], unknowns),:]
        end
    end
    # try block-gmres here
    function get_non_stochastic_steady_state_pullback(∂SS_and_pars)
        # println(∂SS_and_pars)
        return NoTangent(), NoTangent(), jvp' * ∂SS_and_pars[1], NoTangent()
    end

    return (SS_and_pars, (solution_error, iters)), get_non_stochastic_steady_state_pullback
end
    




function get_NSSS_and_parameters(𝓂::ℳ, parameter_values_dual::Vector{ℱ.Dual{Z,S,N}}; verbose::Bool = false, tol::AbstractFloat = 1e-12) where {Z,S,N}
    parameter_values = ℱ.value.(parameter_values_dual)

    SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)

    # if solution_error > tol || isnan(solution_error)
    #     return (SS_and_pars, (solution_error, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    # end

    SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
        
    SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
    
    # unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
    unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))
    # ∂SS_equations_∂parameters = try 𝓂.∂SS_equations_∂parameters(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
    # catch
    #     return (SS_and_pars, (10, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    # end

    X = [parameter_values; SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]]
    
    # vals = Float64[]

    # for f in 𝓂.∂SS_equations_∂parameters[1]
    #     push!(vals, f(X)...)
    # end
    
    vals = zeros(Float64, length(𝓂.∂SS_equations_∂parameters[1]))

    # lk = ReentrantLock()

    Polyester.@batch minbatch = 200 for f in 𝓂.∂SS_equations_∂parameters[1]
        out = f(X)
        
        # begin
        #     lock(lk)
        #     try
                @inbounds vals[out[2]] = out[1]
        #     finally
        #         unlock(lk)
        #     end
        # end
    end

    Accessors.@reset 𝓂.∂SS_equations_∂parameters[2].nzval = vals
    
    ∂SS_equations_∂parameters = 𝓂.∂SS_equations_∂parameters[2]

    # vals = Float64[]

    # for f in 𝓂.∂SS_equations_∂SS_and_pars[1]
    #     push!(vals, f(X)...)
    # end

    vals = zeros(Float64, length(𝓂.∂SS_equations_∂SS_and_pars[1]))

    # lk = ReentrantLock()

    Polyester.@batch minbatch = 200 for f in 𝓂.∂SS_equations_∂SS_and_pars[1]
        out = f(X)
        
        # begin
        #     lock(lk)
        #     try
                @inbounds vals[out[2]] = out[1]
        #     finally
        #         unlock(lk)
        #     end
        # end
    end

    𝓂.∂SS_equations_∂SS_and_pars[3] .*= 0
    𝓂.∂SS_equations_∂SS_and_pars[3][𝓂.∂SS_equations_∂SS_and_pars[2]] .+= vals

    ∂SS_equations_∂SS_and_pars = 𝓂.∂SS_equations_∂SS_and_pars[3]

    # ∂SS_equations_∂parameters = 𝓂.∂SS_equations_∂parameters(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
    # ∂SS_equations_∂SS_and_pars = 𝓂.∂SS_equations_∂SS_and_pars(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
    
    ∂SS_equations_∂SS_and_pars_lu = RF.lu!(∂SS_equations_∂SS_and_pars, check = false)

    if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
        return (SS_and_pars, (10, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]

    jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))
    
    for (i,v) in enumerate(SS_and_pars_names)
        if v in unknowns
            jvp[i,:] = JVP[indexin([v], unknowns),:]
        end
    end

    ∂SS_and_pars = zeros(length(SS_and_pars), N)

    for i in 1:N
        parameter_values_partials = ℱ.partials.(parameter_values_dual, i)

        ∂SS_and_pars[:,i] = jvp * parameter_values_partials
    end
    
    return reshape(map(SS_and_pars, eachrow(∂SS_and_pars)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(SS_and_pars)), (solution_error, iters)
end




function calculate_kalman_filter_loglikelihood(observables::Vector{Symbol}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                presample_periods::Int = 0, 
                                                initial_covariance::Symbol = :theoretical,
                                                verbose::Bool = false)::S where S <: Real
    obs_idx = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(T.aux,T.var,T.exo_present))))

    calculate_kalman_filter_loglikelihood(obs_idx, 𝐒, data_in_deviations, T, presample_periods = presample_periods, initial_covariance = initial_covariance, verbose = verbose)
end

function calculate_kalman_filter_loglikelihood(observables::Vector{String}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                presample_periods::Int = 0, 
                                                initial_covariance::Symbol = :theoretical,
                                                verbose::Bool = false)::S where S <: Real
    obs_idx = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(T.aux,T.var,T.exo_present))))

    calculate_kalman_filter_loglikelihood(obs_idx, 𝐒, data_in_deviations, T, presample_periods = presample_periods, initial_covariance = initial_covariance, verbose = verbose)
end

function calculate_kalman_filter_loglikelihood(observables_index::Vector{Int}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                presample_periods::Int = 0,
                                                initial_covariance::Symbol = :theoretical,
                                                lyapunov_algorithm::Symbol = :doubling,
                                                verbose::Bool = false)::S where S <: Real
    observables_and_states = @ignore_derivatives sort(union(T.past_not_future_and_mixed_idx,observables_index))

    A = 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(S, length(observables_and_states)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

    C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior
    P = get_initial_covariance(Val(initial_covariance), A, 𝐁, lyapunov_algorithm = lyapunov_algorithm, verbose = verbose)

    return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
end

# TODO: use higher level wrapper, like for lyapunov/sylvester
# Specialization for :theoretical
function get_initial_covariance(::Val{:theoretical}, A::AbstractMatrix{S}, B::AbstractMatrix{S}; lyapunov_algorithm::Symbol = :doubling, verbose::Bool = false)::AbstractMatrix{S} where S <: Real
    P, _ = solve_lyapunov_equation(A, B, lyapunov_algorithm = lyapunov_algorithm, verbose = verbose)
    return P
end


# Specialization for :diagonal
function get_initial_covariance(::Val{:diagonal}, A::AbstractMatrix{S}, B::AbstractMatrix{S}; lyapunov_algorithm::Symbol = :doubling, verbose::Bool = false)::Matrix{S} where S <: Real
    P = @ignore_derivatives collect(ℒ.I(size(A, 1)) * 10.0)
    return P
end


function run_kalman_iterations(A::Matrix{S}, 𝐁::Matrix{S}, C::Matrix{Float64}, P::Matrix{S}, data_in_deviations::Matrix{S}; presample_periods::Int = 0)::S where S <: Float64
    u = zeros(S, size(C,2))

    z = C * u

    ztmp = similar(z)

    loglik = S(0.0)

    utmp = similar(u)

    Ctmp = similar(C)

    F = similar(C * C')

    K = similar(C')
    # Ktmp = similar(C')

    tmp = similar(P)
    Ptmp = similar(P)

    for t in 1:size(data_in_deviations, 2)
        ℒ.axpby!(1, data_in_deviations[:, t], -1, z)
        # v = data_in_deviations[:, t] - z

        mul!(Ctmp, C, P) # use Octavian.jl
        mul!(F, Ctmp, C')
        # F = C * P * C'

        luF = RF.lu!(F, check = false) ###

        if !ℒ.issuccess(luF)
            return -Inf
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            return -Inf
        end

        # invF = inv(luF) ###

        if t > presample_periods
            ℒ.ldiv!(ztmp, luF, z)
            loglik += log(Fdet) + ℒ.dot(z', ztmp) ###
            # loglik += log(Fdet) + z' * invF * z###
            # loglik += log(Fdet) + v' * invF * v###
        end

        # mul!(Ktmp, P, C')
        # mul!(K, Ktmp, invF)
        mul!(K, P, C')
        ℒ.rdiv!(K, luF)
        # K = P * Ct / luF
        # K = P * C' * invF

        mul!(tmp, K, C)
        mul!(Ptmp, tmp, P)
        ℒ.axpy!(-1, Ptmp, P)

        mul!(Ptmp, A, P)
        mul!(P, Ptmp, A')
        ℒ.axpy!(1, 𝐁, P)
        # P = A * (P - K * C * P) * A' + 𝐁

        mul!(u, K, z, 1, 1)
        mul!(utmp, A, u)
        u .= utmp
        # u = A * (u + K * v)

        mul!(z, C, u)
        # z = C * u
    end

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
end



function run_kalman_iterations(A::Matrix{S}, 𝐁::Matrix{S}, C::Matrix{Float64}, P::Matrix{S}, data_in_deviations::Matrix{S}; presample_periods::Int = 0)::S where S <: ℱ.Dual
    u = zeros(S, size(C,2))

    z = C * u

    loglik = S(0.0)

    F = similar(C * C')

    K = similar(C')

    for t in 1:size(data_in_deviations, 2)
        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false) ###

        if !ℒ.issuccess(luF)
            return -Inf
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            return -Inf
        end

        invF = inv(luF) ###

        if t > presample_periods
            loglik += log(Fdet) + ℒ.dot(v, invF, v)###
        end

        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)

        z = C * u
    end

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
end


function rrule(::typeof(run_kalman_iterations), A, 𝐁, C, P, data_in_deviations; presample_periods = 0)
    T = size(data_in_deviations, 2) + 1

    z = zeros(size(data_in_deviations, 1))

    ū = zeros(size(C,2))

    P̄ = deepcopy(P) 

    temp_N_N = similar(P)

    PCtmp = similar(C')

    F = similar(C * C')

    u = [similar(ū) for _ in 1:T] # used in backward pass

    P = [copy(P̄) for _ in 1:T] # used in backward pass

    CP = [zero(C) for _ in 1:T] # used in backward pass

    K = [similar(C') for _ in 1:T] # used in backward pass

    invF = [similar(F) for _ in 1:T] # used in backward pass

    v = [zeros(size(data_in_deviations, 1)) for _ in 1:T] # used in backward pass

    loglik = 0.0

    for t in 2:T
        v[t] .= data_in_deviations[:, t-1] .- z#[t-1]

        # CP[t] .= C * P̄[t-1]
        mul!(CP[t], C, P̄)#[t-1])
    
        # F[t] .= CP[t] * C'
        mul!(F, CP[t], C')
    
        luF = RF.lu(F, check = false)
    
        if !ℒ.issuccess(luF)
            return -Inf, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            return -Inf, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        # invF[t] .= inv(luF)
        copy!(invF[t], inv(luF))
        
        if t - 1 > presample_periods
            loglik += log(Fdet) + ℒ.dot(v[t], invF[t], v[t])
        end

        # K[t] .= P̄[t-1] * C' * invF[t]
        mul!(PCtmp, P̄, C')
        mul!(K[t], PCtmp, invF[t])

        # P[t] .= P̄[t-1] - K[t] * CP[t]
        mul!(P[t], K[t], CP[t], -1, 0)
        P[t] .+= P̄
    
        # P̄[t] .= A * P[t] * A' + 𝐁
        mul!(temp_N_N, P[t], A')
        mul!(P̄, A, temp_N_N)
        P̄ .+= 𝐁

        # u[t] .= K[t] * v[t] + ū[t-1]
        mul!(u[t], K[t], v[t])
        u[t] .+= ū
        
        # ū[t] .= A * u[t]
        mul!(ū, A, u[t])

        # z[t] .= C * ū[t]
        mul!(z, C, ū)
    end

    llh = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
    
    # initialise derivative variables
    ∂A = zero(A)
    ∂F = zero(F)
    ∂Faccum = zero(F)
    ∂P = zero(P̄)
    ∂ū = zero(ū)
    ∂v = zero(v[1])
    ∂𝐁 = zero(𝐁)
    ∂data_in_deviations = zero(data_in_deviations)
    vtmp = zero(v[1])
    Ptmp = zero(P[1])

    # pullback
    function kalman_pullback(∂llh)
        ℒ.rmul!(∂A, 0)
        ℒ.rmul!(∂Faccum, 0)
        ℒ.rmul!(∂P, 0)
        ℒ.rmul!(∂ū, 0)
        ℒ.rmul!(∂𝐁, 0)

        for t in T:-1:2
            if t > presample_periods + 1
                # ∂llh∂F
                # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
                # ∂F = invF[t]' - invF[t]' * v[t] * v[t]' * invF[t]'
                mul!(∂F, v[t], v[t]')
                mul!(invF[1], invF[t]', ∂F) # using invF[1] as temporary storage
                mul!(∂F, invF[1], invF[t]')
                ℒ.axpby!(1, invF[t]', -1, ∂F)
        
                # ∂llh∂ū
                # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
                # z[t] .= C * ū[t]
                # ∂v = (invF[t]' + invF[t]) * v[t]
                copy!(invF[1], invF[t]' .+ invF[t])
                # copy!(invF[1], invF[t]) # using invF[1] as temporary storage
                # ℒ.axpy!(1, invF[t]', invF[1]) # using invF[1] as temporary storage
                mul!(∂v, invF[1], v[t])
                # mul!(∂ū∂v, C', v[1])
            else
                ℒ.rmul!(∂F, 0)
                ℒ.rmul!(∂v, 0)
            end
        
            # ∂F∂P
            # F[t] .= C * P̄[t-1] * C'
            # ∂P += C' * (∂F + ∂Faccum) * C
            ℒ.axpy!(1, ∂Faccum, ∂F)
            mul!(PCtmp, C', ∂F) 
            mul!(∂P, PCtmp, C, 1, 1) 
        
            # ∂ū∂P
            # K[t] .= P̄[t-1] * C' * invF[t]
            # u[t] .= K[t] * v[t] + ū[t-1]
            # ū[t] .= A * u[t]
            # ∂P += A' * ∂ū * v[t]' * invF[t]' * C
            mul!(CP[1], invF[t]', C) # using CP[1] as temporary storage
            mul!(PCtmp, ∂ū , v[t]')
            mul!(P[1], PCtmp , CP[1]) # using P[1] as temporary storage
            mul!(∂P, A', P[1], 1, 1) 
        
            # ∂ū∂data
            # v[t] .= data_in_deviations[:, t-1] .- z
            # z[t] .= C * ū[t]
            # ∂data_in_deviations[:,t-1] = -C * ∂ū
            mul!(u[1], A', ∂ū)
            mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
            ℒ.axpy!(1, ∂v, v[1])
            ∂data_in_deviations[:,t-1] .= v[1]
            # mul!(∂data_in_deviations[:,t-1], C, ∂ū, -1, 0) # cannot assign to columns in matrix, must be whole matrix 

            # ∂ū∂ū
            # z[t] .= C * ū[t]
            # v[t] .= data_in_deviations[:, t-1] .- z
            # K[t] .= P̄[t-1] * C' * invF[t]
            # u[t] .= K[t] * v[t] + ū[t-1]
            # ū[t] .= A * u[t]
            # step to next iteration
            # ∂ū = A' * ∂ū - C' * K[t]' * A' * ∂ū
            mul!(u[1], A', ∂ū) # using u[1] as temporary storage
            mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
            mul!(∂ū, C', v[1])
            mul!(u[1], C', v[1], -1, 1)
            copy!(∂ū, u[1])
        
            # ∂llh∂ū
            # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
            # v[t] .= data_in_deviations[:, t-1] .- z
            # z[t] .= C * ū[t]
            # ∂ū -= ∂ū∂v
            mul!(u[1], C', ∂v) # using u[1] as temporary storage
            ℒ.axpy!(-1, u[1], ∂ū)
        
            if t > 2
                # ∂ū∂A
                # ū[t] .= A * u[t]
                # ∂A += ∂ū * u[t-1]'
                mul!(∂A, ∂ū, u[t-1]', 1, 1)
        
                # ∂P̄∂A and ∂P̄∂𝐁
                # P̄[t] .= A * P[t] * A' + 𝐁
                # ∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]
                mul!(P[1], A, P[t-1]')
                mul!(Ptmp ,∂P, P[1])
                mul!(P[1], A, P[t-1])
                mul!(Ptmp ,∂P', P[1], 1, 1)
                ℒ.axpy!(1, Ptmp, ∂A)
        
                # ∂𝐁 += ∂P
                ℒ.axpy!(1, ∂P, ∂𝐁)
        
                # ∂P∂P
                # P[t] .= P̄[t-1] - K[t] * C * P̄[t-1]
                # P̄[t] .= A * P[t] * A' + 𝐁
                # step to next iteration
                # ∂P = A' * ∂P * A
                mul!(P[1], ∂P, A) # using P[1] as temporary storage
                mul!(∂P, A', P[1])
        
                # ∂P̄∂P
                # K[t] .= P̄[t-1] * C' * invF[t]
                # P[t] .= P̄[t-1] - K[t] * CP[t]
                # ∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 
                mul!(PCtmp, ∂P, K[t-1])
                mul!(CP[1], K[t-1]', ∂P) # using CP[1] as temporary storage
                mul!(∂P, PCtmp, C, -1, 1)
                mul!(∂P, C', CP[1], -1, 1)
        
                # ∂ū∂F
                # K[t] .= P̄[t-1] * C' * invF[t]
                # u[t] .= K[t] * v[t] + ū[t-1]
                # ū[t] .= A * u[t]
                # ∂Faccum = -invF[t-1]' * CP[t-1] * A' * ∂ū * v[t-1]' * invF[t-1]'
                mul!(u[1], A', ∂ū) # using u[1] as temporary storage
                mul!(v[1], CP[t-1], u[1]) # using v[1] as temporary storage
                mul!(vtmp, invF[t-1]', v[1], -1, 0)
                mul!(invF[1], vtmp, v[t-1]') # using invF[1] as temporary storage
                mul!(∂Faccum, invF[1], invF[t-1]')
        
                # ∂P∂F
                # K[t] .= P̄[t-1] * C' * invF[t]
                # P[t] .= P̄[t-1] - K[t] * CP[t]
                # ∂Faccum -= invF[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invF[t-1]'
                mul!(CP[1], invF[t-1]', CP[t-1]) # using CP[1] as temporary storage
                mul!(PCtmp, CP[t-1]', invF[t-1]')
                mul!(K[1], ∂P, PCtmp) # using K[1] as temporary storage
                mul!(∂Faccum, CP[1], K[1], -1, 1)
        
            end
        end
        
        ℒ.rmul!(∂P, -∂llh/2)
        ℒ.rmul!(∂A, -∂llh/2)
        ℒ.rmul!(∂𝐁, -∂llh/2)
        ℒ.rmul!(∂data_in_deviations, -∂llh/2)

        return NoTangent(), ∂A, ∂𝐁, NoTangent(), ∂P, ∂data_in_deviations, NoTangent()
    end
    
    return llh, kalman_pullback
end




function check_bounds(parameter_values::Vector{S}, 𝓂::ℳ)::Bool where S <: Real
    if length(𝓂.bounds) > 0 
        for (k,v) in 𝓂.bounds
            if k ∈ 𝓂.parameters
                if min(max(parameter_values[indexin([k], 𝓂.parameters)][1], v[1]), v[2]) != parameter_values[indexin([k], 𝓂.parameters)][1]
                    return true
                end
            end
        end
    end

    return false
end

function get_relevant_steady_state_and_state_update(::Val{:second_order}, parameter_values::Vector{S}, 𝓂::ℳ, tol::AbstractFloat) where S <: Real
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(parameter_values, 𝓂)

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = collect(sss) - all_SS

    TT = 𝓂.timings

    return TT, SS_and_pars, [𝐒₁, 𝐒₂], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:pruned_second_order}, parameter_values::Vector{S}, 𝓂::ℳ, tol::AbstractFloat)::Tuple{timings, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{S}}, Bool} where S <: Real
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(parameter_values, 𝓂, pruning = true)

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = [zeros(𝓂.timings.nVars), collect(sss) - all_SS]

    TT = 𝓂.timings

    return TT, SS_and_pars, [𝐒₁, 𝐒₂], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:third_order}, parameter_values::Vector{S}, 𝓂::ℳ, tol::AbstractFloat) where S <: Real
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(parameter_values, 𝓂)

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = collect(sss) - all_SS

    TT = 𝓂.timings

    return TT, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:pruned_third_order}, parameter_values::Vector{S}, 𝓂::ℳ, tol::AbstractFloat)::Tuple{timings, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{S}}, Bool} where S <: Real
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(parameter_values, 𝓂, pruning = true)

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = [zeros(𝓂.timings.nVars), collect(sss) - all_SS, zeros(𝓂.timings.nVars)]

    TT = 𝓂.timings

    return TT, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged
end


function get_relevant_steady_state_and_state_update(::Val{:first_order}, parameter_values::Vector{S}, 𝓂::ℳ, tol::AbstractFloat)::Tuple{timings, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{Float64}}, Bool} where S <: Real
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameter_values, tol = tol)

    state = zeros(𝓂.timings.nVars)

    TT = 𝓂.timings

    if solution_error > tol || isnan(solution_error)
        return TT, SS_and_pars, zeros(S, 0, 0), [state], false
    end

    ∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂)# |> Matrix

    # ∇₁ = Matrix{S}(sp∇₁)

    𝐒₁, solved = calculate_first_order_solution(∇₁; T = TT)

    return TT, SS_and_pars, 𝐒₁, [state], solved
end

    # reduce_system = false

    # if reduce_system
    #     variable_to_equation = @ignore_derivatives find_variables_to_exclude(𝓂, observables)
    
    #     rows_to_exclude = Int[]
    #     cant_exclude = Symbol[]

    #     for (ks, vidx) in variable_to_equation
    #         iidd =  @ignore_derivatives indexin([ks] ,𝓂.timings.var)[1]
    #         if !isnothing(iidd)
    #             # if all(.!(∇₁[vidx, 𝓂.timings.nFuture_not_past_and_mixed .+ iidd] .== 0))
    #             if minimum(abs, ∇₁[vidx, 𝓂.timings.nFuture_not_past_and_mixed .+ iidd]) / maximum(abs, ∇₁[vidx, 𝓂.timings.nFuture_not_past_and_mixed .+ iidd]) > 1e-12
    #                 for v in vidx
    #                     if v ∉ rows_to_exclude
    #                         @ignore_derivatives push!(rows_to_exclude, v)
    #                         # ∇₁[vidx,:] .-= ∇₁[v,:]' .* ∇₁[vidx, 𝓂.timings.nFuture_not_past_and_mixed .+ iidd] ./ ∇₁[v, 𝓂.timings.nFuture_not_past_and_mixed .+ iidd]
    #                         broadcaster = @ignore_derivatives create_broadcaster(vidx, size(∇₁,1))
    #                         # broadcaster = spzeros(size(∇₁,1), length(vidx))
    #                         # for (i, vid) in enumerate(vidx)
    #                         #     broadcaster[vid,i] = 1.0
    #                         # end
    #                         ∇₁ -= broadcaster * (∇₁[v,:]' .* ∇₁[vidx, 𝓂.timings.nFuture_not_past_and_mixed .+ iidd] ./ ∇₁[v, 𝓂.timings.nFuture_not_past_and_mixed .+ iidd])
    #                         break
    #                     end
    #                 end
    #             else
    #                 @ignore_derivatives push!(cant_exclude, ks)
    #             end
    #         end
    #     end

    #     rows_to_include = @ignore_derivatives setdiff(1:𝓂.timings.nVars, rows_to_exclude)
    
    #     cols_to_exclude = @ignore_derivatives indexin(setdiff(𝓂.timings.present_only, union(observables, cant_exclude)), 𝓂.timings.var)

    #     present_idx = @ignore_derivatives 𝓂.timings.nFuture_not_past_and_mixed .+ (setdiff(range(1, 𝓂.timings.nVars), cols_to_exclude))

    #     ∇₁ = Matrix{S}(∇₁[rows_to_include, vcat(1:𝓂.timings.nFuture_not_past_and_mixed, present_idx , 𝓂.timings.nFuture_not_past_and_mixed + 𝓂.timings.nVars + 1 : size(∇₁,2))])
    
    #     @ignore_derivatives if !haskey(𝓂.estimation_helper, union(observables, cant_exclude)) create_timings_for_estimation!(𝓂, union(observables, cant_exclude)) end

    #     TT = @ignore_derivatives 𝓂.estimation_helper[union(observables, cant_exclude)]
    # else






    
function calculate_inversion_filter_loglikelihood(::Val{:first_order},
                                                    state::Vector{Vector{Float64}}, 
                                                    𝐒::Matrix{Float64}, 
                                                    data_in_deviations::Matrix{Float64}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    filter_algorithm::Symbol = :LagrangeNewton)
    # first order
    state = copy(state[1])

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0
    
    if warmup_iterations > 0
        if warmup_iterations >= 1
            jac = 𝐒[cond_var_idx,end-T.nExo+1:end]
            if warmup_iterations >= 2
                jac = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], jac)
                if warmup_iterations >= 3
                    Sᵉ = 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                    for e in 1:warmup_iterations-2
                        jac = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], jac)
                        Sᵉ *= 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                    end
                end
            end
        end
    
        jacdecomp = ℒ.svd(jac)

        x = jacdecomp \ data_in_deviations[:,1]
    
        warmup_shocks = reshape(x, T.nExo, warmup_iterations)
    
        for i in 1:warmup_iterations-1
            ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], warmup_shocks[:,i]))
            # state = state_update(state, warmup_shocks[:,i])
        end

        for i in 1:warmup_iterations
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor))
            end
        end
    
        shocks² += sum(abs2,x)
    end

    y = zeros(length(cond_var_idx))
    x = zeros(T.nExo)
    jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

    if T.nExo == length(observables)
        jacdecomp = RF.lu(jac, check = false)
        if !ℒ.issuccess(jacdecomp)
            return -Inf
        end
        logabsdets = ℒ.logabsdet(jac ./ precision_factor)[1]
        invjac = inv(jacdecomp)
    else
        jacdecomp = ℒ.svd(jac)
        
        logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jac ./ precision_factor))
        invjac = inv(jacdecomp)
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods
    
    𝐒obs = 𝐒[cond_var_idx,1:end-T.nExo]

    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒obs, state[T.past_not_future_and_mixed_idx])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
        ℒ.mul!(x, invjac, y)

        # x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

        if i > presample_periods
            shocks² += sum(abs2,x)
        end

        ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], x))
        # state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)
    end
    # TODO: use subset of observables and states when propagating states (see kalman filter)

    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
    # return -(logabsdets + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end



function rrule(::typeof(calculate_inversion_filter_loglikelihood), 
                ::Val{:first_order}, 
                state::Vector{Vector{Float64}}, 
                𝐒::Matrix{Float64}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}}, 
                T::timings; 
                warmup_iterations::Int = 0, 
                presample_periods::Int = 0,
                filter_algorithm::Symbol = :LagrangeNewton)
    # first order
    state = copy(state[1])

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    obs_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    t⁻ = T.past_not_future_and_mixed_idx

    shocks² = 0.0
    logabsdets = 0.0

    @assert warmup_iterations == 0 "Warmup iterations not yet implemented for reverse-mode automatic differentiation."
    # TODO: implement warmup iterations

    state = [copy(state) for _ in 1:size(data_in_deviations,2)+1]

    shocks² = 0.0
    logabsdets = 0.0

    y = zeros(length(obs_idx))
    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

    jac = 𝐒[obs_idx,end-T.nExo+1:end]

    if T.nExo == length(observables)
        logabsdets = ℒ.logabsdet(-jac' ./ precision_factor)[1]
        jacdecomp = ℒ.lu(jac)
        invjac = inv(jacdecomp)
    else
        logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(-jac' ./ precision_factor))
        jacdecomp = ℒ.svd(jac)
        invjac = inv(jacdecomp)
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods

    @views 𝐒obs = 𝐒[obs_idx,1:end-T.nExo]

    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒obs, state[i][t⁻])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
        ℒ.mul!(x[i],invjac,y)
        # x = 𝐒[obs_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[obs_idx,1:end-T.nExo] * state[t⁻])

        if i > presample_periods
            shocks² += sum(abs2,x[i])
        end

        ℒ.mul!(state[i+1], 𝐒, vcat(state[i][t⁻], x[i]))
        # state[i+1] =  𝐒 * vcat(state[i][t⁻], x[i])
    end

    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
    
    if llh < -1e10
        return -Inf, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    ∂𝐒 = zero(𝐒)
    
    ∂𝐒ᵗ⁻ = copy(∂𝐒[t⁻,:])

    ∂data_in_deviations = zero(data_in_deviations)
    
    ∂data = zeros(length(t⁻), size(data_in_deviations,2) - 1)

    ∂state = zero(state[1])

    # precomputed matrices
    M¹  = 𝐒[obs_idx, 1:end-T.nExo]' * invjac' 
    M²  = 𝐒[t⁻,1:end-T.nExo]' - M¹ * 𝐒[t⁻,end-T.nExo+1:end]'
    M³  = invjac' * 𝐒[t⁻,end-T.nExo+1:end]'

    ∂Stmp = [M¹ for _ in 1:size(data_in_deviations,2)-1]

    for t in 2:size(data_in_deviations,2)-1
        ∂Stmp[t] = M² * ∂Stmp[t-1]
    end

    tmp1 = zeros(Float64, T.nExo, length(t⁻) + T.nExo)
    tmp2 = zeros(Float64, length(t⁻), length(t⁻) + T.nExo)
    tmp3 = zeros(Float64, length(t⁻) + T.nExo)

    ∂𝐒t⁻        = copy(tmp2)
    # ∂𝐒obs_idx   = copy(tmp1)

    # TODO: optimize allocations
    # pullback
    function inversion_pullback(∂llh)
        for t in reverse(axes(data_in_deviations,2))
            ∂state[t⁻]                                  .= M² * ∂state[t⁻]

            if t > presample_periods
                ∂state[t⁻]                              += M¹ * x[t]

                ∂data_in_deviations[:,t]                -= invjac' * x[t]

                ∂𝐒[obs_idx, :]                          += invjac' * x[t] * vcat(state[t][t⁻], x[t])'

                if t > 1
                    ∂data[:,t:end]                      .= M² * ∂data[:,t:end]
                    
                    ∂data[:,t-1]                        += M¹ * x[t]
            
                    ∂data_in_deviations[:,t-1]          += M³ * ∂data[:,t-1:end] * ones(size(data_in_deviations,2) - t + 1)

                    for tt in t-1:-1:1
                        for (i,v) in enumerate(t⁻)
                            copyto!(tmp3::Vector{Float64}, i::Int, state[tt]::Vector{Float64}, v::Int, 1)
                        end
                        
                        copyto!(tmp3, length(t⁻) + 1, x[tt], 1, T.nExo)

                        mul!(tmp1,  x[t], tmp3')

                        mul!(∂𝐒t⁻,  ∂Stmp[t-tt], tmp1, 1, 1)
                        
                    end
                end
            end
        end

        ∂𝐒[t⁻,:]                            += ∂𝐒t⁻
                        
        ∂𝐒[obs_idx, :]                      -= M³ * ∂𝐒t⁻
        
        ∂𝐒[obs_idx,end-T.nExo+1:end] -= (size(data_in_deviations,2) - presample_periods) * invjac' / 2

        return NoTangent(), NoTangent(), [∂state * ∂llh], ∂𝐒 * ∂llh, ∂data_in_deviations * ∂llh, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    
    return llh, inversion_pullback
end




function calculate_inversion_filter_loglikelihood(::Val{:pruned_second_order},
                                                    state::Vector{Vector{Float64}}, 
                                                    𝐒::Vector{AbstractMatrix{Float64}}, 
                                                    data_in_deviations::Matrix{Float64}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    filter_algorithm::Symbol = :LagrangeNewton)
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = @ignore_derivatives indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺  = @ignore_derivatives BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = @ignore_derivatives BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺  = @ignore_derivatives BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = @ignore_derivatives setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻  = 𝐒[2][cond_var_idx, var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]

    kron_buffer = zeros(T.nExo^2)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    for i in axes(data_in_deviations, 2)
        state¹⁻ = state₁

        state¹⁻_vol = vcat(state¹⁻, 1)

        state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        # shock_independent = copy(data_in_deviations[:,i])

        # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        # ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)

        # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

        shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒¹⁻ * state²⁻ + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)  

        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
        
        init_guess = @ignore_derivatives zeros(size(𝐒ⁱ, 2))

        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
                     
        # if matched println("$filter_algorithm: $matched; current x: $x") end      
        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    return -Inf # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
                end 
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)
    
        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
            end

            shocks² += sum(abs2,x)
        end

        aug_state₁ = [state₁; 1; x]
        aug_state₂ = [state₂; 0; zero(x)]

        state₁, state₂ = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
    end

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf and Fair and Taylor (1983)
    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end



function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:pruned_second_order},
                state::Vector{Vector{Float64}}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                filter_algorithm::Symbol = :LagrangeNewton)# where S <: Real
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒⁻¹ᵉ = 𝐒[1][T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]

    kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]
    
    J = ℒ.I(T.nExo)
    
    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))
    
    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
    
    state¹⁻ = state₁

    state¹⁻_vol = vcat(state¹⁻, 1)

    state²⁻ = state₂

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
   
    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
    
    aug_state₁ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    aug_state₂ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    
    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[1])), x[1])
    
    jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]
    
    λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]
    
    λ[1] = tmp' \ x[1] * 2
    
    fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
                -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
    fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronxλ_tmp = ℒ.kron(x[1], λ[1])
    
    kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]
    
    for i in axes(data_in_deviations,2)
        state¹⁻ = state₁
    
        state¹⁻_vol = vcat(state¹⁻, 1)
    
        state²⁻ = state₂
        
        shock_independent = copy(data_in_deviations[:,i])
    
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
    
        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
    
        init_guess = zeros(size(𝐒ⁱ, 2))
    
        x[i], matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kronxx[i],
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
    
        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])
    
        λ[i] = jacc[i]' \ x[i] * 2
        # ℒ.ldiv!(λ[i], tmp', x[i])
        # ℒ.rmul!(λ[i], 2)
    
        fXλp[i] = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc[i]'
                    -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ℒ.kron!(kronxx[i], x[i], x[i])
    
        ℒ.kron!(kronxλ[i], x[i], λ[i])
    
        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
            end
    
            shocks² += sum(abs2,x[i])
        end
    
        aug_state₁[i] = [state₁; 1; x[i]]
        aug_state₂[i] = [state₂; 0; zero(x[1])]

        state₁, state₂ = [𝐒⁻¹ * aug_state₁[i], 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * ℒ.kron(aug_state₁[i], aug_state₁[i]) / 2] # strictly following Andreasen et al. (2018)
    end
    
    

    ∂data_in_deviations = similar(data_in_deviations)

    function inversion_filter_loglikelihood_pullback(∂llh) 
        ∂𝐒ⁱ = zero(𝐒ⁱ)
        ∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)

        ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
        ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)

        ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
        ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)

        ∂𝐒⁻¹ = zero(𝐒⁻¹)
        ∂𝐒⁻² = zero(𝐒⁻²)

        ∂𝐒¹⁻ = zero(𝐒¹⁻)

        ∂state¹⁻_vol = zero(state¹⁻_vol)
        ∂x = zero(x[1])
        ∂state = [zeros(T.nPast_not_future_and_mixed), zeros(T.nPast_not_future_and_mixed)]

        for i in reverse(axes(data_in_deviations,2))
            # state₁, state₂ = [𝐒⁻¹ * aug_state₁[i], 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * ℒ.kron(aug_state₁[i], aug_state₁[i]) / 2]
            # state₁ = 𝐒⁻¹ * aug_state₁[i]
            ∂𝐒⁻¹ += ∂state[1] * aug_state₁[i]'
            ∂aug_state₁ = 𝐒⁻¹' * ∂state[1]

            # state₂ = 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * ℒ.kron(aug_state₁[i], aug_state₁[i]) / 2
            ∂𝐒⁻¹ += ∂state[2] * aug_state₂[i]'

            ∂aug_state₂ = 𝐒⁻¹' * ∂state[2]

            ∂𝐒⁻² += ∂state[2] * ℒ.kron(aug_state₁[i], aug_state₁[i])' / 2

            ∂kronaug_state₁ = 𝐒⁻²' * ∂state[2] / 2

            re∂kronaug_state₁ = reshape(∂kronaug_state₁, 
                                    length(aug_state₁[i]), 
                                    length(aug_state₁[i]))

            ei = 1
            for e in eachslice(re∂kronaug_state₁; dims = (1))
                ∂aug_state₁[ei] += ℒ.dot(aug_state₁[i],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronaug_state₁; dims = (2))
                ∂aug_state₁[ei] += ℒ.dot(aug_state₁[i],e)
                ei += 1
            end

            if i > 1 && i < size(data_in_deviations,2)
                ∂state[1] *= 0
                ∂state[2] *= 0
            end
            

            # aug_state₁ = [state₁; 1; x]
            ∂state[1] += ∂aug_state₁[1:length(∂state[1])] # TODO: cleanup length and size references

            ∂x = ∂aug_state₁[T.nPast_not_future_and_mixed+2:end]

            # aug_state₂ = [state₂; 0; zero(x)] # TODO: dont allocate new vector here
            ∂state[2] += ∂aug_state₂[1:length(∂state[1])]

            # shocks² += sum(abs2,x[i])
            if i < size(data_in_deviations,2)
                ∂x -= copy(x[i])
            else
                ∂x += copy(x[i])
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            ∂jacc = inv(jacc[i])'

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
            ∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

            re∂kronIx = reshape(∂kronIx, 
                                    T.nExo, 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIx; dims = (1,3))
                if i < size(data_in_deviations,2)
                    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
                else
                    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                end
                ei += 1
            end

            ∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

            # find_shocks
            ∂xλ = vcat(∂x, zero(λ[i]))

            S = fXλp[i]' \ ∂xλ

            if i < size(data_in_deviations,2)
                S *= -1
            end

            ∂shock_independent = S[T.nExo+1:end] # fine

            ∂𝐒ⁱ = (S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]') # fine
            ∂𝐒ⁱ -= ∂jacc / 2 # fine

            ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] *  kronxλ[i]' - S[T.nExo+1:end] * kronxx[i]'

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
            ∂state¹⁻_vol *= 0
            ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

            re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
                ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                ei += 1
            end

            ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), [aug_state₁[i][1:T.nPast_not_future_and_mixed]; 1])'

            # shock_independent = copy(data_in_deviations[:,i])
            ∂data_in_deviations[:,i] = ∂shock_independent


            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            ∂𝐒¹⁻ᵛ -= ∂shock_independent * [aug_state₁[i][1:T.nPast_not_future_and_mixed]; 1]'

            ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)
            ∂𝐒¹⁻ -= ∂shock_independent * aug_state₂[i][1:T.nPast_not_future_and_mixed]'

            ∂state[2] -= 𝐒¹⁻' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
            ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron([aug_state₁[i][1:T.nPast_not_future_and_mixed]; 1], [aug_state₁[i][1:T.nPast_not_future_and_mixed]; 1])' / 2

            ∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

            re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    length(state¹⁻_vol))

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
                ∂state¹⁻_vol[ei] += ℒ.dot([aug_state₁[i][1:T.nPast_not_future_and_mixed]; 1],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
                ∂state¹⁻_vol[ei] += ℒ.dot([aug_state₁[i][1:T.nPast_not_future_and_mixed]; 1],e) # fine
                ei += 1
            end

            # state¹⁻_vol = vcat(state¹⁻, 1)
            ∂state[1] += ∂state¹⁻_vol[1:end-1]
        end

        ∂𝐒 = [copy(𝐒[1]) * 0, copy(𝐒[2]) * 0]

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
        ∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒ⁱ²ᵉ / 2

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed] += ∂𝐒¹⁻

        ∂𝐒[1] *= ∂llh
        ∂𝐒[2] *= ∂llh

        ∂state[1] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[1] * ∂llh
        ∂state[2] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[2] * ∂llh

        return NoTangent(), NoTangent(),  ∂state, ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
    end

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    return llh, inversion_filter_loglikelihood_pullback
end




function calculate_inversion_filter_loglikelihood(::Val{:second_order},
                                                    state::Vector{Float64}, 
                                                    𝐒::Vector{AbstractMatrix{Float64}}, 
                                                    data_in_deviations::Matrix{Float64}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    filter_algorithm::Symbol = :LagrangeNewton)# where S <: Real
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    # s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    # tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    # var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    # 𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    # 𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    # 𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = state[T.past_not_future_and_mixed_idx]

    kron_buffer = zeros(T.nExo^2)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    for i in axes(data_in_deviations,2)
        state¹⁻ = state#[T.past_not_future_and_mixed_idx]

        state¹⁻_vol = vcat(state¹⁻, 1)
        
        shock_independent = copy(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

        init_guess = zeros(size(𝐒ⁱ, 2))

        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
                                
        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    return -Inf # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
                end 
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        jacc = -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x))

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
            end

            shocks² += sum(abs2,x)
        end

        aug_state = [state; 1; x]

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 - data_in_deviations[:,i]
        # println("Match with data: $res")

        state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
    end

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end



function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:second_order},
                state::Vector{Float64}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                filter_algorithm::Symbol = :LagrangeNewton)# where S <: Real
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒⁻¹ᵉ = 𝐒[1][T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    stt = state[T.past_not_future_and_mixed_idx]

    kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]
    
    J = ℒ.I(T.nExo)
    
    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))
    
    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
    
    state¹⁻ = stt
    
    state¹⁻_vol = vcat(state¹⁻, 1)
    
    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
    
    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
    
    aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    
    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[1])), x[1])
    
    jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]
    
    λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]
    
    λ[1] = tmp' \ x[1] * 2
    
    fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
                -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
    fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronxλ_tmp = ℒ.kron(x[1], λ[1])
    
    kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]
    
    
    for i in axes(data_in_deviations,2)
        state¹⁻ = stt
    
        state¹⁻_vol = vcat(state¹⁻, 1)
        
        shock_independent = copy(data_in_deviations[:,i])
    
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
    
        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
    
        init_guess = zeros(size(𝐒ⁱ, 2))
    
        x[i], matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kronxx[i],
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
    
        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])
    
        λ[i] = jacc[i]' \ x[i] * 2
        # ℒ.ldiv!(λ[i], tmp', x[i])
        # ℒ.rmul!(λ[i], 2)
    
        fXλp[i] = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc[i]'
                    -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ℒ.kron!(kronxx[i], x[i], x[i])
    
        ℒ.kron!(kronxλ[i], x[i], λ[i])
    
        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
            end
    
            shocks² += sum(abs2,x[i])
        end
    
        aug_state[i] = [stt; 1; x[i]]
    
        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2
    end
    
    

    ∂state = similar(state)

    ∂𝐒 = copy(𝐒)

    ∂data_in_deviations = similar(data_in_deviations)

    function inversion_filter_loglikelihood_pullback(∂llh)
        ∂𝐒ⁱ = zero(𝐒ⁱ)
        ∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)

        ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
        ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)

        ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
        ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)

        ∂𝐒⁻¹ = zero(𝐒⁻¹)
        ∂𝐒⁻² = zero(𝐒⁻²)

        ∂state¹⁻_vol = zero(state¹⁻_vol)
        # ∂x = zero(x[1])
        ∂state = zeros(T.nPast_not_future_and_mixed)

        for i in reverse(axes(data_in_deviations,2))
            # stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
            ∂𝐒⁻¹ += ∂state * aug_state[i]'
            
            ∂𝐒⁻² += ∂state * ℒ.kron(aug_state[i], aug_state[i])' / 2

            ∂aug_state = 𝐒⁻¹' * ∂state
            ∂kronaug_state  = 𝐒⁻²' * ∂state / 2

            re∂kronaug_state = reshape(∂kronaug_state, 
                                    length(aug_state[i]), 
                                    length(aug_state[i]))

            ei = 1
            for e in eachslice(re∂kronaug_state; dims = (1))
                ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronaug_state; dims = (2))
                ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
                ei += 1
            end

            if i > 1 && i < size(data_in_deviations,2)
                ∂state *= 0
            end

            # aug_state[i] = [stt; 1; x[i]]
            ∂state += ∂aug_state[1:length(∂state)]

            # aug_state[i] = [stt; 1; x[i]]
            ∂x = ∂aug_state[T.nPast_not_future_and_mixed+2:end]

            # shocks² += sum(abs2,x[i])
            if i < size(data_in_deviations,2)
                ∂x -= copy(x[i])
            else
                ∂x += copy(x[i])
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            ∂jacc = inv(jacc[i])'

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
            ∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

            re∂kronIx = reshape(∂kronIx, 
                                    T.nExo, 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIx; dims = (1,3))
                if i < size(data_in_deviations,2)
                    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
                else
                    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                end
                ei += 1
            end

            ∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

            # find_shocks
            ∂xλ = vcat(∂x, zero(λ[i]))

            S = fXλp[i]' \ ∂xλ

            if i < size(data_in_deviations,2)
                S *= -1
            end

            ∂shock_independent = S[T.nExo+1:end] # fine

            ∂𝐒ⁱ = S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]' # fine
            ∂𝐒ⁱ -= ∂jacc / 2 # fine

            ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] *  kronxλ[i]' - S[T.nExo+1:end] * kronxx[i]'

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
            ∂state¹⁻_vol *= 0
            ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

            re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
                ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                ei += 1
            end

            ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), [aug_state[i][1:T.nPast_not_future_and_mixed];1])'

            # shock_independent = copy(data_in_deviations[:,i])
            ∂data_in_deviations[:,i] = ∂shock_independent


            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            ∂𝐒¹⁻ᵛ -= ∂shock_independent * [aug_state[i][1:T.nPast_not_future_and_mixed];1]'

            ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
            ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron([aug_state[i][1:T.nPast_not_future_and_mixed];1], [aug_state[i][1:T.nPast_not_future_and_mixed];1])' / 2

            ∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

            re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    length(state¹⁻_vol))

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
                ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:T.nPast_not_future_and_mixed];1],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
                ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:T.nPast_not_future_and_mixed];1],e) # fine
                ei += 1
            end

            # state¹⁻_vol = vcat(state¹⁻, 1)
            ∂state += ∂state¹⁻_vol[1:end-1]
        end

        ∂𝐒 = [copy(𝐒[1]) * 0, copy(𝐒[2]) * 0]

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
        ∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒ⁱ²ᵉ / 2

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²

        ∂𝐒[1] *= ∂llh
        ∂𝐒[2] *= ∂llh

        return NoTangent(), NoTangent(),  ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state * ∂llh, ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
    end

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    return llh, inversion_filter_loglikelihood_pullback
end




function calculate_inversion_filter_loglikelihood(::Val{:pruned_third_order},
                                                    state::Vector{Vector{Float64}}, 
                                                    𝐒::Vector{AbstractMatrix{Float64}}, 
                                                    data_in_deviations::Matrix{Float64}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    filter_algorithm::Symbol = :LagrangeNewton)
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))

    tmp = ℒ.kron(e_in_s⁺, s_in_s⁺) |> sparse
    shockvar_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind

    shockvar²_idxs = setdiff(union(shock_idxs), shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind

    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind

    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²⁻ᵛᵉ = 𝐒[2][cond_var_idx,shockvar_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²⁻ᵛᵉ   = length(𝐒²⁻ᵛᵉ.nzval)   / length(𝐒²⁻ᵛᵉ) > .1 ? collect(𝐒²⁻ᵛᵉ)   : 𝐒²⁻ᵛᵉ
    𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
    var_vol³_idxs = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs3 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shock³_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shockvar1_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
    shockvar2_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
    shockvar3_idxs = tmp.nzind

    shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)#, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    𝐒³⁻ᵛ = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ  = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³  = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = length(𝐒³⁻ᵛ.nzval)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = length(𝐒³⁻ᵉ.nzval)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = length(𝐒³ᵉ.nzval)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = length(𝐒⁻³.nzval)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    state[1] = state[1][T.past_not_future_and_mixed_idx]
    state[2] = state[2][T.past_not_future_and_mixed_idx]
    state[3] = state[3][T.past_not_future_and_mixed_idx]

    kron_buffer = zeros(T.nExo^2)

    kron_buffer² = zeros(T.nExo^3)

    II = sparse(ℒ.I(T.nExo^2))
    
    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, kron_buffer)

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    for i in axes(data_in_deviations,2)
        state¹⁻ = state[1]

        state¹⁻_vol = vcat(state¹⁻, 1)

        state²⁻ = state[2]#[T.past_not_future_and_mixed_idx]

        state³⁻ = state[3]#[T.past_not_future_and_mixed_idx]

        shock_independent = copy(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state¹⁻, state²⁻), -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state²⁻) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess = zeros(size(𝐒ⁱ, 2))


        # x² , matched = find_shocks(Val(filter_algorithm), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
        #                         println(x²)

        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer²,
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 200
                                )
                                
                                # println(x)
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && (!matched || ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂) || (matched && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x)))
        #             # println("$i - $filter_algorithm restart - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             # println("$i - $backup_solver - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         # else
        #         #     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #         #     norm1 = ℒ.norm(y)

        #         #     norm2 = ℒ.norm(shock_independent)

        #             # println("$i - $filter_algorithm - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")#, residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     # else
        #     #     println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end

        if !matched
            return -Inf # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
        end 
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
                # println("$filter_algorithm: $matched; current x: $x")
                # if !matched
                #     x, matched = find_shocks(Val(:COBYLA), 
                #                             x,
                #                             kron_buffer,
                #                             kron_buffer²,
                #                             kron_buffer2,
                #                             kron_buffer3,
                #                             J,
                #                             𝐒ⁱ,
                #                             𝐒ⁱ²ᵉ,
                #                             𝐒ⁱ³ᵉ,
                #                             shock_independent)
                # end
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x))), $(ℒ.norm(x2)-ℒ.norm(x))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x))), $(ℒ.norm(x3)-ℒ.norm(x))")
        # end

        jacc = -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x)))

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
            end

            shocks² += sum(abs2,x)
        end

        aug_state₁ = [state[1]; 1; x]
        aug_state₁̂ = [state[1]; 0; x]
        aug_state₂ = [state[2]; 0; zero(x)]
        aug_state₃ = [state[3]; 0; zero(x)]
        
        kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

        # res = 𝐒[1][cond_var_idx,:] * aug_state₁   +   𝐒[1][cond_var_idx,:] * aug_state₂ + 𝐒[2][cond_var_idx,:] * kron_aug_state₁ / 2   +   𝐒[1][cond_var_idx,:] * aug_state₃ + 𝐒[2][cond_var_idx,:] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3][cond_var_idx,:] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")
        
        # println(ℒ.norm(x))

        state = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2, 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
        # println(sum(state))
    end

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end




function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:pruned_third_order},
                state::Vector{Vector{Float64}}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                filter_algorithm::Symbol = :LagrangeNewton)
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))

    tmp = ℒ.kron(e_in_s⁺, s_in_s⁺) |> sparse
    shockvar_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind

    shockvar²_idxs = setdiff(union(shock_idxs), shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind

    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind

    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²⁻ᵛᵉ = 𝐒[2][cond_var_idx,shockvar_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²⁻ᵛᵉ   = length(𝐒²⁻ᵛᵉ.nzval)   / length(𝐒²⁻ᵛᵉ) > .1 ? collect(𝐒²⁻ᵛᵉ)   : 𝐒²⁻ᵛᵉ
    𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
    var_vol³_idxs = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs3 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shock³_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shockvar1_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
    shockvar2_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
    shockvar3_idxs = tmp.nzind

    shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)#, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    𝐒³⁻ᵛ = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ  = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³  = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = length(𝐒³⁻ᵛ.nzval)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = length(𝐒³⁻ᵉ.nzval)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = length(𝐒³ᵉ.nzval)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = length(𝐒⁻³.nzval)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]
    state₃ = state[3][T.past_not_future_and_mixed_idx]

    kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]
    
    J = ℒ.I(T.nExo)
    
    II = sparse(ℒ.I(T.nExo^2))

    kronxxx = [zeros(T.nExo^3) for _ in 1:size(data_in_deviations,2)]

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))
    
    kron_buffer3 = ℒ.kron(J, zeros(T.nExo^2))

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
    
    state¹⁻ = state₁

    state¹⁻_vol = vcat(state¹⁻, 1)

    state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

    state³⁻ = state₃#[T.past_not_future_and_mixed_idx]
    
    state¹⁻_vol = vcat(state¹⁻, 1)
    
    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
    
    𝐒ⁱ²ᵉ = [zero(𝐒²ᵉ) for _ in 1:size(data_in_deviations,2)]

    aug_state₁ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    aug_state₁̂ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    aug_state₂ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    aug_state₃ = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]

    kron_aug_state₁ = [zeros(size(𝐒⁻¹,2)^2) for _ in 1:size(data_in_deviations,2)]

    jacc_tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[1] * ℒ.kron(ℒ.I(T.nExo), x[1])
    
    jacc = [zero(jacc_tmp) for _ in 1:size(data_in_deviations,2)]
    
    λ = [zeros(size(jacc_tmp, 1)) for _ in 1:size(data_in_deviations,2)]
    
    λ[1] = jacc_tmp' \ x[1] * 2
    
    fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ[1]' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc_tmp'
                -jacc_tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
    fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronxλ_tmp = ℒ.kron(x[1], λ[1])
    
    kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]
    
    kronxxλ_tmp = ℒ.kron(x[1], kronxλ_tmp)
    
    kronxxλ = [kronxxλ_tmp for _ in 1:size(data_in_deviations,2)]

    II = sparse(ℒ.I(T.nExo^2))

    lI = 2 * ℒ.I(size(𝐒ⁱ, 2))

    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    for i in axes(data_in_deviations,2)
        state¹⁻ = state₁

        state¹⁻_vol = vcat(state¹⁻, 1)

        state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        state³⁻ = state₃#[T.past_not_future_and_mixed_idx]

        shock_independent = copy(data_in_deviations[:,i])
    
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state¹⁻, state²⁻), -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state²⁻) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        init_guess = zeros(size(𝐒ⁱ, 2))
    
        x[i], matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kronxx[i],
                                kronxxx[i],
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ[i],
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
    
        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[i] * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), kronxx[i])
    
        λ[i] = jacc[i]' \ x[i] * 2
        # ℒ.ldiv!(λ[i], tmp', x[i])
        # ℒ.rmul!(λ[i], 2)
        fXλp[i] = [reshape((2 * 𝐒ⁱ²ᵉ[i] + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(II, x[i]))' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - lI  jacc[i]'
                    -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ℒ.kron!(kronxx[i], x[i], x[i])
    
        ℒ.kron!(kronxλ[i], x[i], λ[i])
    
        ℒ.kron!(kronxxλ[i], x[i], kronxλ[i])

        ℒ.kron!(kronxxx[i], x[i], kronxx[i])

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
            end
    
            shocks² += sum(abs2,x[i])
        end
    
        aug_state₁[i] = [state₁; 1; x[i]]
        aug_state₁̂[i] = [state₁; 0; x[i]]
        aug_state₂[i] = [state₂; 0; zeros(T.nExo)]
        aug_state₃[i] = [state₃; 0; zeros(T.nExo)]

        kron_aug_state₁[i] = ℒ.kron(aug_state₁[i], aug_state₁[i])

        state₁, state₂, state₃ = [𝐒⁻¹ * aug_state₁[i], 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * kron_aug_state₁[i] / 2, 𝐒⁻¹ * aug_state₃[i] + 𝐒⁻² * ℒ.kron(aug_state₁̂[i], aug_state₂[i]) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁[i], aug_state₁[i]) / 6]
    end
    
    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

    ∂state = similar(state)

    ∂𝐒 = copy(𝐒)

    ∂data_in_deviations = similar(data_in_deviations)

    function inversion_filter_loglikelihood_pullback(∂llh)
        ∂𝐒ⁱ = zero(𝐒ⁱ)
        ∂𝐒²ᵉ = zero(𝐒²ᵉ)
        ∂𝐒ⁱ³ᵉ = zero(𝐒ⁱ³ᵉ)

        ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
        ∂𝐒¹⁻ = zero(𝐒¹⁻)
        ∂𝐒²⁻ = zero(𝐒²⁻)
        ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
        ∂𝐒²⁻ᵛᵉ = zero(𝐒²⁻ᵛᵉ)
        ∂𝐒³⁻ᵉ = zero(𝐒³⁻ᵉ)
        ∂𝐒³⁻ᵉ² = zero(𝐒³⁻ᵉ²)

        ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
        ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
        ∂𝐒³⁻ᵛ = zero(𝐒³⁻ᵛ)
        
        ∂𝐒⁻¹ = zero(𝐒⁻¹)
        ∂𝐒⁻² = zero(𝐒⁻²)
        ∂𝐒⁻³ = zero(𝐒⁻³)

        ∂aug_state₁̂ = zero(aug_state₁̂[1])
        ∂state¹⁻_vol = zero(state¹⁻_vol)
        ∂x = zero(x[1])
        ∂kronxx = zero(kronxx[1])
        ∂kronstate¹⁻_vol = zeros(length(state¹⁻_vol)^2)
        ∂state = [zeros(T.nPast_not_future_and_mixed), zeros(T.nPast_not_future_and_mixed), zeros(T.nPast_not_future_and_mixed)]

        for i in reverse(axes(data_in_deviations,2))
            # state₁ = 𝐒⁻¹ * aug_state₁[i]
            ∂𝐒⁻¹ += ∂state[1] * aug_state₁[i]'

            ∂aug_state₁ = 𝐒⁻¹' * ∂state[1]

            # state₂ = 𝐒⁻¹ * aug_state₂[i] + 𝐒⁻² * kron_aug_state₁[i] / 2
            ∂𝐒⁻¹ += ∂state[2] * aug_state₂[i]'

            ∂aug_state₂ = 𝐒⁻¹' * ∂state[2]

            ∂𝐒⁻² += ∂state[2] * kron_aug_state₁[i]' / 2

            ∂kronaug_state₁ = 𝐒⁻²' * ∂state[2] / 2

            # state₃ = 𝐒⁻¹ * aug_state₃[i] + 𝐒⁻² * ℒ.kron(aug_state₁̂[i], aug_state₂[i]) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁[i],aug_state₁[i]) / 6
            ∂𝐒⁻¹ += ∂state[3] * aug_state₃[i]'

            ∂aug_state₃ = 𝐒⁻¹' * ∂state[3]

            ∂𝐒⁻² += ∂state[3] * ℒ.kron(aug_state₁̂[i], aug_state₂[i])'

            ∂aug_state₁̂ *= 0

            ∂kronaug_state₁̂₂ = 𝐒⁻²' * ∂state[3]

            re∂kronaug_state₁̂₂ = reshape(∂kronaug_state₁̂₂, 
                                    length(aug_state₁[i]), 
                                    length(aug_state₁[i]))

            ei = 1
            for e in eachslice(re∂kronaug_state₁̂₂; dims = (1))
                ∂aug_state₁̂[ei] += ℒ.dot(aug_state₂[i],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronaug_state₁̂₂; dims = (2))
                ∂aug_state₂[ei] += ℒ.dot(aug_state₁̂[i],e)
                ei += 1
            end

            ∂𝐒⁻³ += ∂state[3] * ℒ.kron(kron_aug_state₁[i],aug_state₁[i])' / 6

            ∂kronkronaug_state₁ = 𝐒⁻³' * ∂state[3] / 6

            re∂kronkronaug_state₁ = reshape(∂kronkronaug_state₁, 
                                            length(aug_state₁[i]), 
                                            length(aug_state₁[i])^2)
    
            ei = 1
            for e in eachslice(re∂kronkronaug_state₁; dims = (1))
                ∂aug_state₁[ei] += ℒ.dot(kron_aug_state₁[i],e)
                ei += 1
            end
            
            ei = 1
            for e in eachslice(re∂kronkronaug_state₁; dims = (2))
                ∂kronaug_state₁[ei] += ℒ.dot(aug_state₁[i],e)
                ei += 1
            end
    
            # kron_aug_state₁[i] = ℒ.kron(aug_state₁[i], aug_state₁[i])
            re∂kronaug_state₁ = reshape(∂kronaug_state₁, 
                                    length(aug_state₁[i]), 
                                    length(aug_state₁[i]))

            ei = 1
            for e in eachslice(re∂kronaug_state₁; dims = (1))
                ∂aug_state₁[ei] += ℒ.dot(aug_state₁[i],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronaug_state₁; dims = (2))
                ∂aug_state₁[ei] += ℒ.dot(aug_state₁[i],e)
                ei += 1
            end

            if i > 1 && i < size(data_in_deviations,2)
                ∂state[1] *= 0
                ∂state[2] *= 0
                ∂state[3] *= 0
            end

            # aug_state₁[i] = [state₁; 1; x[i]]
            ∂state[1] += ∂aug_state₁[1:length(∂state[1])] # TODO: cleanup length and size references

            ∂x = ∂aug_state₁[T.nPast_not_future_and_mixed+2:end]

            # aug_state₁̂[i] = [state₁; 0; x[i]]
            ∂state[1] += ∂aug_state₁̂[1:length(∂state[1])]

            ∂x += ∂aug_state₁̂[T.nPast_not_future_and_mixed+2:end]

            # aug_state₂[i] = [state₂; 0; zeros(T.nExo)]
            ∂state[2] += ∂aug_state₂[1:length(∂state[1])]
            
            # aug_state₃[i] = [state₃; 0; zeros(T.nExo)]
            ∂state[3] += ∂aug_state₃[1:length(∂state[1])]

            # shocks² += sum(abs2,x[i])
            if i < size(data_in_deviations,2)
                ∂x -= copy(x[i])
            else
                ∂x += copy(x[i])
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            ∂jacc = inv(jacc[i])'

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x))
            ∂𝐒ⁱ = -∂jacc / 2 # fine

            ∂kronIx = 𝐒ⁱ²ᵉ[i]' * ∂jacc

            re∂kronIx = reshape(∂kronIx, 
                                    T.nExo, 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIx; dims = (1,3))
                if i < size(data_in_deviations,2)
                    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
                else
                    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                end
                ei += 1
            end

            ∂𝐒ⁱ²ᵉ = -∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

            ∂kronIxx = 𝐒ⁱ³ᵉ' * ∂jacc * 3 / 2
            
            re∂kronIxx = reshape(∂kronIxx, 
                                    T.nExo^2, 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ∂kronxx *= 0

            ei = 1
            for e in eachslice(re∂kronIxx; dims = (1,3))
                if i < size(data_in_deviations,2)
                    ∂kronxx[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
                else
                    ∂kronxx[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                end
                ei += 1
            end

            re∂kronxx = reshape(∂kronxx, 
                                    T.nExo, 
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronxx; dims = (2))
                ∂x[ei] += ℒ.dot(x[i],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronxx; dims = (1))
                ∂x[ei] += ℒ.dot(x[i],e)
                ei += 1
            end

            ∂𝐒ⁱ³ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), kronxx[i])' * 3 / 2

            # find_shocks
            ∂xλ = vcat(∂x, zero(λ[i]))

            S = fXλp[i]' \ ∂xλ

            if i < size(data_in_deviations,2)
                S *= -1
            end

            ∂shock_independent = S[T.nExo+1:end] # fine

            ∂𝐒ⁱ += S[1:T.nExo] * λ[i]' - S[T.nExo + 1:end] * x[i]' # fine

            ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] * kronxλ[i]' - S[T.nExo + 1:end] * kronxx[i]'

            ∂𝐒ⁱ³ᵉ += 3 * S[1:T.nExo] * kronxxλ[i]' - S[T.nExo + 1:end] * kronxxx[i]'

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state²⁻) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
            ∂kronstate¹⁻_vol *= 0

            state¹⁻_vol = [aug_state₁[i][1:T.nPast_not_future_and_mixed];1] # define here as it is used multiple times later
            state¹⁻ = aug_state₁[i][1:T.nPast_not_future_and_mixed]
            state²⁻ = aug_state₂[i][1:T.nPast_not_future_and_mixed]
            state³⁻ = aug_state₃[i][1:T.nPast_not_future_and_mixed]

            ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ∂state¹⁻_vol *= 0

            ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

            re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
                ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                ei += 1
            end

            ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

            ∂kronIstate²⁻ = 𝐒²⁻ᵛᵉ' * ∂𝐒ⁱ

            re∂kronIstate²⁻ = reshape(∂kronIstate²⁻, 
                                    length(state²⁻), 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIstate²⁻; dims = (1,3))
                ∂state[2][ei] += ℒ.dot(ℒ.I(T.nExo),e)
                ei += 1
            end

            ∂𝐒²⁻ᵛᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state²⁻)'

            ∂kronIstate¹⁻_volstate¹⁻_vol = 𝐒³⁻ᵉ²' * ∂𝐒ⁱ / 2

            re∂kronIstate¹⁻_volstate¹⁻_vol = reshape(∂kronIstate¹⁻_volstate¹⁻_vol, 
                                    length(state¹⁻_vol)^2, 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIstate¹⁻_volstate¹⁻_vol; dims = (1,3))
                ∂kronstate¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e) # ∂kronstate¹⁻_vol is dealt with later
                ei += 1
            end

            ∂𝐒³⁻ᵉ² += ∂𝐒ⁱ * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol)' / 2
            
            # 𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2
            ∂𝐒²ᵉ += ∂𝐒ⁱ²ᵉ / 2
            
            ∂𝐒³⁻ᵉ += ∂𝐒ⁱ²ᵉ * ℒ.kron(II, state¹⁻_vol)' / 2
            
            ∂kronIIstate¹⁻_vol = 𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ / 2

            re∂kronIIstate¹⁻_vol = reshape(∂kronIIstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    T.nExo^2, 
                                    1,
                                    T.nExo^2)

            ei = 1
            for e in eachslice(re∂kronIIstate¹⁻_vol; dims = (1,3))
                ∂state¹⁻_vol[ei] += ℒ.dot(II,e)
                ei += 1
            end

            # shock_independent = copy(data_in_deviations[:,i])
            ∂data_in_deviations[:,i] = ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            ∂𝐒¹⁻ᵛ -= ∂shock_independent * state¹⁻_vol'

            ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)
            ∂𝐒¹⁻ -= ∂shock_independent * state²⁻'

            ∂state[2] -= 𝐒¹⁻' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)
            ∂𝐒¹⁻ -= ∂shock_independent * state³⁻'

            ∂state[3] -= 𝐒¹⁻' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
            ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron(state¹⁻_vol, state¹⁻_vol)' / 2

            ∂kronstate¹⁻_vol -= 𝐒²⁻ᵛ' * ∂shock_independent / 2

            # ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state¹⁻, state²⁻), -1, 1)
            ∂𝐒²⁻ -= ∂shock_independent * ℒ.kron(state¹⁻, state²⁻)'

            ∂kronstate¹⁻²⁻ = -𝐒²⁻' * ∂shock_independent

            re∂kronstate¹⁻²⁻ = reshape(∂kronstate¹⁻²⁻, 
                                    length(state¹⁻), 
                                    length(state¹⁻))

            ei = 1
            for e in eachslice(re∂kronstate¹⁻²⁻; dims = (2))
                ∂state[1][ei] += ℒ.dot(state²⁻,e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronstate¹⁻²⁻; dims = (1))
                ∂state[2][ei] += ℒ.dot(state¹⁻,e) # fine
                ei += 1
            end

            # ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
            ∂𝐒³⁻ᵛ -= ∂shock_independent * ℒ.kron(ℒ.kron(state¹⁻_vol, state¹⁻_vol), state¹⁻_vol)' / 6

            ∂kronstate¹⁻_volstate¹⁻_vol = -𝐒³⁻ᵛ' * ∂shock_independent / 6

            re∂kronstate¹⁻_volstate¹⁻_vol = reshape(∂kronstate¹⁻_volstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    length(state¹⁻_vol)^2)
                            
            ei = 1
            for e in eachslice(re∂kronstate¹⁻_volstate¹⁻_vol; dims = (2))
                ∂kronstate¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_volstate¹⁻_vol; dims = (1))
                ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.kron(state¹⁻_vol, state¹⁻_vol),e) # fine
                ei += 1
            end        

            re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    length(state¹⁻_vol))

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
                ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
                ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e) # fine
                ei += 1
            end

            # state¹⁻_vol = vcat(state¹⁻, 1)
            ∂state[1] += ∂state¹⁻_vol[1:end-1]
        end

        ∂𝐒 = [copy(𝐒[1]) * 0, copy(𝐒[2]) * 0, copy(𝐒[3]) * 0]

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed] += ∂𝐒¹⁻
        ∂𝐒[2][cond_var_idx,var²_idxs] += ∂𝐒²⁻
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
        ∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒²ᵉ
        ∂𝐒[2][cond_var_idx,shockvar_idxs] += ∂𝐒²⁻ᵛᵉ
        ∂𝐒[3][cond_var_idx,shockvar³2_idxs] += ∂𝐒³⁻ᵉ²
        ∂𝐒[3][cond_var_idx,shockvar³_idxs] += ∂𝐒³⁻ᵉ
        ∂𝐒[3][cond_var_idx,shock³_idxs] += ∂𝐒ⁱ³ᵉ / 6 # 𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ
        ∂𝐒[3][cond_var_idx,var_vol³_idxs] += ∂𝐒³⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²
        ∂𝐒[3][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻³

        ∂𝐒[1] *= ∂llh
        ∂𝐒[2] *= ∂llh
        ∂𝐒[3] *= ∂llh

        ∂state[1] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[1] * ∂llh
        ∂state[2] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[2] * ∂llh
        ∂state[3] = ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state[3] * ∂llh

        return NoTangent(), NoTangent(), ∂state, ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
    end

    return llh, inversion_filter_loglikelihood_pullback
end





function calculate_inversion_filter_loglikelihood(::Val{:third_order},
                                                    state::Vector{Float64}, 
                                                    𝐒::Vector{AbstractMatrix{Float64}}, 
                                                    data_in_deviations::Matrix{Float64}, 
                                                    observables::Union{Vector{String}, Vector{Symbol}},
                                                    T::timings; 
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    filter_algorithm::Symbol = :LagrangeNewton)
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))

    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind

    shockvar²_idxs = setdiff(union(shock_idxs), shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind

    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind

    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = state[T.past_not_future_and_mixed_idx]

    tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
    var_vol³_idxs = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs3 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shock³_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shockvar1_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
    shockvar2_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
    shockvar3_idxs = tmp.nzind

    shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)#, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    𝐒³⁻ᵛ  = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ  = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ   = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³   = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = length(𝐒³⁻ᵛ.nzval)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = length(𝐒³⁻ᵉ.nzval)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = length(𝐒³ᵉ.nzval)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = length(𝐒⁻³.nzval)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    kron_buffer = zeros(T.nExo^2)

    kron_buffer² = zeros(T.nExo^3)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, kron_buffer)

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    II = sparse(ℒ.I(T.nExo^2))

    for i in axes(data_in_deviations,2)
        state¹⁻ = state

        state¹⁻_vol = vcat(state¹⁻, 1)
        
        shock_independent = copy(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess = zeros(size(𝐒ⁱ, 2))

        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer²,
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 200
                                )
                                
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂)
        #             println("$i - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             println("$i - $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         else
        #             y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #             norm1 = ℒ.norm(y)

        #             norm2 = ℒ.norm(shock_independent)

        #             println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     else
        #         println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end

        if !matched
            return -Inf # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
        end 
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
            #     println("$filter_algorithm: $matched; current x: $x")
            #     if !matched
            #         x, matched = find_shocks(Val(:COBYLA), 
            #                                 x,
            #                                 kron_buffer,
            #                                 kron_buffer²,
            #                                 kron_buffer2,
            #                                 kron_buffer3,
            #                                 J,
            #                                 𝐒ⁱ,
            #                                 𝐒ⁱ²ᵉ,
            #                                 𝐒ⁱ³ᵉ,
            #                                 shock_independent)
            #         println("COBYLA: $matched; current x: $x")
            #     end
            # end
        # end

        # x2, mat = find_shocks(Val(:COBYLA), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
            
        # x3, mat2 = find_shocks(Val(filter_algorithm), 
        #                         x2,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # # if mat
        #     println("COBYLA - $mat: $x2, $(ℒ.norm(x2))")
        # # end
        # # if mat2
        #     println("LagrangeNewton restart - $mat2: $x3, $(ℒ.norm(x3))")
        # # end

        jacc = -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x)))
    
        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
            end

            shocks² += sum(abs2,x)
        end

        aug_state = [state; 1; x]

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 + 𝐒[3][cond_var_idx, :] * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")

        state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
        # state = state_update(state, x)
    end

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end




function rrule(::typeof(calculate_inversion_filter_loglikelihood),
                ::Val{:third_order},
                state::Vector{Float64}, 
                𝐒::Vector{AbstractMatrix{Float64}}, 
                data_in_deviations::Matrix{Float64}, 
                observables::Union{Vector{String}, Vector{Symbol}},
                T::timings; 
                warmup_iterations::Int = 0,
                presample_periods::Int = 0,
                filter_algorithm::Symbol = :LagrangeNewton)
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
    sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))
    
    tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind
    
    tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
    shock²_idxs = tmp.nzind
    
    shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

    tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
    var_vol²_idxs = tmp.nzind
    
    tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    var²_idxs = tmp.nzind
    
    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒⁻¹ᵉ = 𝐒[1][T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻  = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
    var_vol³_idxs = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs2 = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs3 = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shock³_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shockvar1_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
    shockvar2_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
    shockvar3_idxs = tmp.nzind

    shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)#, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

    𝐒³⁻ᵛ  = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ  = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ   = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³   = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = length(𝐒³⁻ᵛ.nzval)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = length(𝐒³⁻ᵉ.nzval)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = length(𝐒³ᵉ.nzval)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = length(𝐒⁻³.nzval)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    stt = state[T.past_not_future_and_mixed_idx]

    kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]
    
    J = ℒ.I(T.nExo)
    
    kronxxx = [zeros(T.nExo^3) for _ in 1:size(data_in_deviations,2)]

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))
    
    kron_buffer3 = ℒ.kron(J, zeros(T.nExo^2))

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
    
    state¹⁻ = stt
    
    state¹⁻_vol = vcat(state¹⁻, 1)
    
    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
    
    𝐒ⁱ²ᵉ = [zero(𝐒²ᵉ) for _ in 1:size(data_in_deviations,2)]

    aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]
    
    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[1] * ℒ.kron(ℒ.I(T.nExo), x[1])
    
    jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]
    
    λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]
    
    λ[1] = tmp' \ x[1] * 2
    
    fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ[1]' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
                -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
    fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]
    
    kronxλ_tmp = ℒ.kron(x[1], λ[1])
    
    kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]
    
    kronxxλ_tmp = ℒ.kron(x[1], kronxλ_tmp)
    
    kronxxλ = [kronxxλ_tmp for _ in 1:size(data_in_deviations,2)]

    II = sparse(ℒ.I(T.nExo^2))

    lI = 2 * ℒ.I(size(𝐒ⁱ, 2))

    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    for i in axes(data_in_deviations,2)
        state¹⁻ = stt
    
        state¹⁻_vol = vcat(state¹⁻, 1)
        
        shock_independent = copy(data_in_deviations[:,i])
    
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
    
        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        init_guess = zeros(size(𝐒ⁱ, 2))
    
        x[i], matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kronxx[i],
                                kronxxx[i],
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ[i],
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
    
        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[i] * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), kronxx[i])
    
        λ[i] = jacc[i]' \ x[i] * 2
        # ℒ.ldiv!(λ[i], tmp', x[i])
        # ℒ.rmul!(λ[i], 2)
        fXλp[i] = [reshape((2 * 𝐒ⁱ²ᵉ[i] + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(II, x[i]))' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - lI  jacc[i]'
                    -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ℒ.kron!(kronxx[i], x[i], x[i])
    
        ℒ.kron!(kronxλ[i], x[i], λ[i])
    
        ℒ.kron!(kronxxλ[i], x[i], kronxλ[i])

        ℒ.kron!(kronxxx[i], x[i], kronxx[i])

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
            end
    
            shocks² += sum(abs2,x[i])
        end
    
        aug_state[i] = [stt; 1; x[i]]
    
        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
    end
    
    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2


    ∂state = similar(state)

    ∂𝐒 = copy(𝐒)

    ∂data_in_deviations = similar(data_in_deviations)

    function inversion_filter_loglikelihood_pullback(∂llh)
        ∂𝐒ⁱ = zero(𝐒ⁱ)
        ∂𝐒²ᵉ = zero(𝐒²ᵉ)
        ∂𝐒ⁱ³ᵉ = zero(𝐒ⁱ³ᵉ)

        ∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
        ∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
        ∂𝐒³⁻ᵉ = zero(𝐒³⁻ᵉ)
        ∂𝐒³⁻ᵉ² = zero(𝐒³⁻ᵉ²)

        ∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
        ∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
        ∂𝐒³⁻ᵛ = zero(𝐒³⁻ᵛ)
        
        ∂𝐒⁻¹ = zero(𝐒⁻¹)
        ∂𝐒⁻² = zero(𝐒⁻²)
        ∂𝐒⁻³ = zero(𝐒⁻³)

        ∂state¹⁻_vol = zero(state¹⁻_vol)
        ∂x = zero(x[1])
        ∂kronxx = zero(kronxx[1])
        ∂kronstate¹⁻_vol = zeros(length(state¹⁻_vol)^2)
        ∂state = zeros(T.nPast_not_future_and_mixed)

        for i in reverse(axes(data_in_deviations,2))
            # stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
            ∂𝐒⁻¹ += ∂state * aug_state[i]'
            
            ∂𝐒⁻² += ∂state * ℒ.kron(aug_state[i], aug_state[i])' / 2

            ∂𝐒⁻³ += ∂state * ℒ.kron(ℒ.kron(aug_state[i], aug_state[i]), aug_state[i])' / 6
            
            ∂aug_state = 𝐒⁻¹' * ∂state
            ∂kronaug_state = 𝐒⁻²' * ∂state / 2
            ∂kronkronaug_state = 𝐒⁻³' * ∂state / 6
    
            re∂kronkronaug_state = reshape(∂kronkronaug_state, 
                                            length(aug_state[i]), 
                                            length(aug_state[i])^2)
    
            ei = 1
            for e in eachslice(re∂kronkronaug_state; dims = (1))
                ∂aug_state[ei] += ℒ.dot(ℒ.kron(aug_state[i], aug_state[i]),e)
                ei += 1
            end
            
            ei = 1
            for e in eachslice(re∂kronkronaug_state; dims = (2))
                ∂kronaug_state[ei] += ℒ.dot(aug_state[i],e)
                ei += 1
            end
    
            re∂kronaug_state = reshape(∂kronaug_state, 
                                    length(aug_state[i]), 
                                    length(aug_state[i]))

            ei = 1
            for e in eachslice(re∂kronaug_state; dims = (1))
                ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronaug_state; dims = (2))
                ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
                ei += 1
            end

            if i > 1 && i < size(data_in_deviations,2)
                ∂state *= 0
            end

            # aug_state[i] = [stt; 1; x[i]]
            ∂state += ∂aug_state[1:length(∂state)]

            # aug_state[i] = [stt; 1; x[i]]
            ∂x = ∂aug_state[T.nPast_not_future_and_mixed+2:end]

            # shocks² += sum(abs2,x[i])
            if i < size(data_in_deviations,2)
                ∂x -= copy(x[i])
            else
                ∂x += copy(x[i])
            end

            # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
            ∂jacc = inv(jacc[i])'

            # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x))
            ∂𝐒ⁱ = -∂jacc / 2 # fine

            ∂kronIx = 𝐒ⁱ²ᵉ[i]' * ∂jacc

            re∂kronIx = reshape(∂kronIx, 
                                    T.nExo, 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIx; dims = (1,3))
                if i < size(data_in_deviations,2)
                    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
                else
                    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                end
                ei += 1
            end

            ∂𝐒ⁱ²ᵉ = -∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

            ∂kronIxx = 𝐒ⁱ³ᵉ' * ∂jacc * 3 / 2
            
            re∂kronIxx = reshape(∂kronIxx, 
                                    T.nExo^2, 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ∂kronxx *= 0

            ei = 1
            for e in eachslice(re∂kronIxx; dims = (1,3))
                if i < size(data_in_deviations,2)
                    ∂kronxx[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
                else
                    ∂kronxx[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                end
                ei += 1
            end

            re∂kronxx = reshape(∂kronxx, 
                                    T.nExo, 
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronxx; dims = (2))
                ∂x[ei] += ℒ.dot(x[i],e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronxx; dims = (1))
                ∂x[ei] += ℒ.dot(x[i],e)
                ei += 1
            end

            ∂𝐒ⁱ³ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), kronxx[i])' * 3 / 2


            # find_shocks
            ∂xλ = vcat(∂x, zero(λ[i]))

            S = fXλp[i]' \ ∂xλ

            if i < size(data_in_deviations,2)
                S *= -1
            end

            ∂shock_independent = S[T.nExo+1:end] # fine

            ∂𝐒ⁱ += S[1:T.nExo] * λ[i]' - S[T.nExo + 1:end] * x[i]' # fine

            ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] * kronxλ[i]' - S[T.nExo + 1:end] * kronxx[i]'

            ∂𝐒ⁱ³ᵉ += 3 * S[1:T.nExo] * kronxxλ[i]' - S[T.nExo + 1:end] * kronxxx[i]'

            # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
            ∂kronstate¹⁻_vol *= 0

            state¹⁻_vol = [aug_state[i][1:T.nPast_not_future_and_mixed];1] # define here as it is used multiple times later

            ∂𝐒¹ᵉ += ∂𝐒ⁱ

            ∂state¹⁻_vol *= 0

            ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

            re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
                ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
                ei += 1
            end

            ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

            ∂kronIstate¹⁻_volstate¹⁻_vol = 𝐒³⁻ᵉ²' * ∂𝐒ⁱ / 2

            re∂kronIstate¹⁻_volstate¹⁻_vol = reshape(∂kronIstate¹⁻_volstate¹⁻_vol, 
                                    length(state¹⁻_vol)^2, 
                                    T.nExo, 
                                    1,
                                    T.nExo)

            ei = 1
            for e in eachslice(re∂kronIstate¹⁻_volstate¹⁻_vol; dims = (1,3))
                ∂kronstate¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e) # ∂kronstate¹⁻_vol is dealt with later
                ei += 1
            end

            ∂𝐒³⁻ᵉ² += ∂𝐒ⁱ * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol)' / 2
            

            # 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2
            ∂𝐒²ᵉ += ∂𝐒ⁱ²ᵉ / 2
            
            ∂𝐒³⁻ᵉ += ∂𝐒ⁱ²ᵉ * ℒ.kron(II, state¹⁻_vol)' / 2
            
            ∂kronIIstate¹⁻_vol = 𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ / 2

            re∂kronIIstate¹⁻_vol = reshape(∂kronIIstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    T.nExo^2, 
                                    1,
                                    T.nExo^2)

            ei = 1
            for e in eachslice(re∂kronIIstate¹⁻_vol; dims = (1,3))
                ∂state¹⁻_vol[ei] += ℒ.dot(II,e)
                ei += 1
            end


            # shock_independent = copy(data_in_deviations[:,i])
            ∂data_in_deviations[:,i] = ∂shock_independent


            # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
            ∂𝐒¹⁻ᵛ -= ∂shock_independent * state¹⁻_vol'

            ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

            # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
            ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron(state¹⁻_vol, state¹⁻_vol)' / 2

            ∂kronstate¹⁻_vol -= 𝐒²⁻ᵛ' * ∂shock_independent / 2

            # ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
            ∂𝐒³⁻ᵛ -= ∂shock_independent * ℒ.kron(ℒ.kron(state¹⁻_vol, state¹⁻_vol), state¹⁻_vol)' / 6

            ∂kronstate¹⁻_volstate¹⁻_vol = -𝐒³⁻ᵛ' * ∂shock_independent / 6

            re∂kronstate¹⁻_volstate¹⁻_vol = reshape(∂kronstate¹⁻_volstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    length(state¹⁻_vol)^2)
                            
            ei = 1
            for e in eachslice(re∂kronstate¹⁻_volstate¹⁻_vol; dims = (2))
                ∂kronstate¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_volstate¹⁻_vol; dims = (1))
                ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.kron(state¹⁻_vol, state¹⁻_vol),e) # fine
                ei += 1
            end        

            re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                                    length(state¹⁻_vol), 
                                    length(state¹⁻_vol))

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
                ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
                ei += 1
            end

            ei = 1
            for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
                ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e) # fine
                ei += 1
            end

            # state¹⁻_vol = vcat(state¹⁻, 1)
            ∂state += ∂state¹⁻_vol[1:end-1]
        end

        ∂𝐒 = [copy(𝐒[1]) * 0, copy(𝐒[2]) * 0, copy(𝐒[3]) * 0]

        ∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
        ∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
        ∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒²ᵉ
        ∂𝐒[3][cond_var_idx,shockvar³2_idxs] += ∂𝐒³⁻ᵉ²
        ∂𝐒[3][cond_var_idx,shockvar³_idxs] += ∂𝐒³⁻ᵉ
        ∂𝐒[3][cond_var_idx,shock³_idxs] += ∂𝐒ⁱ³ᵉ / 6 # 𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        ∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
        ∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ
        ∂𝐒[3][cond_var_idx,var_vol³_idxs] += ∂𝐒³⁻ᵛ

        ∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
        ∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²
        ∂𝐒[3][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻³

        ∂𝐒[1] *= ∂llh
        ∂𝐒[2] *= ∂llh
        ∂𝐒[3] *= ∂llh

        return NoTangent(), NoTangent(), ℒ.I(T.nVars)[:,T.past_not_future_and_mixed_idx] * ∂state * ∂llh, ∂𝐒, ∂data_in_deviations * ∂llh, NoTangent(),  NoTangent(),  NoTangent(),  NoTangent(), NoTangent()
    end

    return llh, inversion_filter_loglikelihood_pullback
end





function inversion_filter(𝓂::ℳ, 
    data_in_deviations::AbstractArray{Float64},
    algorithm::Symbol; 
    warmup_iterations::Int = 0,
    verbose::Bool = false, 
    tol::AbstractFloat = 1e-12)
    
    observables = collect(axiskeys(data_in_deviations,1))

    data_in_deviations = collect(data_in_deviations)
    
    @assert observables isa Vector{String} || observables isa Vector{Symbol} "Make sure that the data has variables names as rows. They can be either Strings or Symbols."

    sort!(observables)

    observables = observables isa String_input ? observables .|> Meta.parse .|> replace_indices : observables

    # solve model given the parameters
    if algorithm == :second_order
        sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂)

        if !converged 
            @error "No solution for these parameters."
        end

        all_SS = expand_steady_state(SS_and_pars,𝓂)

        state = collect(sss) - all_SS

        state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
            aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                        1
                        shock]
            return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
        end
    elseif algorithm == :pruned_second_order
        sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, pruning = true)

        if !converged 
            @error "No solution for these parameters."
        end

        all_SS = expand_steady_state(SS_and_pars,𝓂)

        state = [zeros(𝓂.timings.nVars), collect(sss) - all_SS]

        state_update = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
            aug_state₁ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 1; shock]
            aug_state₂ = [pruned_states[2][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
            
            return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
        end
    elseif algorithm == :third_order
        sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂)

        if !converged 
            @error "No solution for these parameters."
        end

        all_SS = expand_steady_state(SS_and_pars,𝓂)

        state = collect(sss) - all_SS

        state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
            aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            1
                            shock]
            return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
        end
    elseif algorithm == :pruned_third_order
        sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, pruning = true)

        if !converged 
            @error "No solution for these parameters."
        end

        all_SS = expand_steady_state(SS_and_pars,𝓂)

        state = [zeros(𝓂.timings.nVars), collect(sss) - all_SS, zeros(𝓂.timings.nVars)]

        state_update = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
            aug_state₁ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 1; shock]
            aug_state₁̂ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 0; shock]
            aug_state₂ = [pruned_states[2][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
            aug_state₃ = [pruned_states[3][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
            
            kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)
            
            return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * kron_aug_state₁ / 2, 𝐒₁ * aug_state₃ + 𝐒₂ * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒₃ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
        end
    else
        SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, verbose = verbose)

        if solution_error > tol || isnan(solution_error)
            @error "No solution for these parameters."
        end

        state = zeros(𝓂.timings.nVars)

        ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix

        𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
        
        if !solved 
            @error "No solution for these parameters."
        end

        state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
            aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                        shock]
            return 𝐒₁ * aug_state # you need a return statement for forwarddiff to work
        end
    end

    if state isa Vector{Float64}
        pruning = false
    else
        pruning = true
    end

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))

    states = zeros(𝓂.timings.nVars, n_obs)
    shocks = zeros(𝓂.timings.nExo, n_obs)

    precision_factor = 1.0

    if warmup_iterations > 0
        res = @suppress begin Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
                            zeros(𝓂.timings.nExo * warmup_iterations), 
                            Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                            Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                            autodiff = :forward) end
    
        matched = Optim.minimum(res) < 1e-12
    
        if !matched
            res = @suppress begin Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
                                zeros(𝓂.timings.nExo * warmup_iterations), 
                                Optim.LBFGS(), 
                                Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                                autodiff = :forward) end

            matched = Optim.minimum(res) < 1e-12
        end

        @assert matched "Numerical stabiltiy issues for restrictions in warmup iterations."
    
        x = Optim.minimizer(res)
    
        warmup_shocks = reshape(x, 𝓂.timings.nExo, warmup_iterations)
    
        for i in 1:warmup_iterations-1
            state = state_update(state, warmup_shocks[:,i])
        end
    end
    
    initial_state = state

    for i in axes(data_in_deviations,2)
        res = @suppress begin Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                            zeros(𝓂.timings.nExo), 
                            Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                            Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                            autodiff = :forward) end
    
        matched = Optim.minimum(res) < 1e-12
    
        if !matched
            res = @suppress begin Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                            zeros(𝓂.timings.nExo), 
                            Optim.LBFGS(), 
                            Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                            autodiff = :forward) end

            matched = Optim.minimum(res) < 1e-12
        end

        @assert matched "Numerical stabiltiy issues for restrictions in period $i."
    
        x = Optim.minimizer(res)
    
        state = state_update(state, x)

        states[:,i] = pruning ? sum(state) : state
        shocks[:,i] = x
    end
        
    return states, shocks, initial_state
end


function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:first_order}, # algo
    ::Val{:kalman}; # filter
    warmup_iterations::Int = 0,
    smooth::Bool = true,
    verbose::Bool = false)

    obs_axis = collect(axiskeys(data_in_deviations,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, obs_symbols; verbose = verbose)

    variables           = filtered_and_smoothed[smooth ? 1 : 5]
    standard_deviations = filtered_and_smoothed[smooth ? 2 : 6]
    shocks              = filtered_and_smoothed[smooth ? 3 : 7]
    decomposition       = filtered_and_smoothed[smooth ? 4 : 8]

    return variables, shocks, standard_deviations, decomposition
end


function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:first_order}, # algo
    ::Val{:inversion}; # filter
    warmup_iterations::Int = 0,
    smooth::Bool = true,
    verbose::Bool = false)

    algorithm = :first_order

    variables, shocks, initial_state = inversion_filter(𝓂, data_in_deviations, algorithm, warmup_iterations = warmup_iterations)

    state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)

    decomposition = zeros(𝓂.timings.nVars, 𝓂.timings.nExo + 2, size(data_in_deviations, 2))

    decomposition[:,end,:] .= variables

    for i in 1:𝓂.timings.nExo
        sck = zeros(𝓂.timings.nExo)
        sck[i] = shocks[i, 1]
        decomposition[:,i,1] = state_update(initial_state , sck)
    end

    decomposition[:,end - 1,1] .= decomposition[:,end,1] - sum(decomposition[:,1:end-2,1], dims=2)

    for i in 2:size(data_in_deviations,2)
        for ii in 1:𝓂.timings.nExo
            sck = zeros(𝓂.timings.nExo)
            sck[ii] = shocks[ii, i]
            decomposition[:,ii,i] = state_update(decomposition[:,ii, i-1], sck)
        end

        decomposition[:,end - 1,i] .= decomposition[:,end,i] - sum(decomposition[:,1:end-2,i], dims=2)
    end

    return variables, shocks, [], decomposition
end


function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:second_order}, # algo
    ::Val{:inversion}; # filter
    warmup_iterations::Int = 0,
    smooth::Bool = true,
    verbose::Bool = false)

    variables, shocks, initial_state = inversion_filter(𝓂, data_in_deviations, :second_order, warmup_iterations = warmup_iterations)

    return variables, shocks, [], []
end


function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:pruned_second_order}, # algo
    ::Val{:inversion}; # filter
    warmup_iterations::Int = 0,
    smooth::Bool = true,
    verbose::Bool = false)

    algorithm = :pruned_second_order

    variables, shocks, initial_state = inversion_filter(𝓂, data_in_deviations, algorithm, warmup_iterations = warmup_iterations)

    state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)

    states = [initial_state for _ in 1:𝓂.timings.nExo + 1]

    decomposition = zeros(𝓂.timings.nVars, 𝓂.timings.nExo + 3, size(data_in_deviations, 2))

    decomposition[:, end, :] .= variables

    for i in 1:𝓂.timings.nExo
        sck = zeros(𝓂.timings.nExo)
        sck[i] = shocks[i, 1]
        states[i] = state_update(initial_state , sck)
        decomposition[:,i,1] = sum(states[i])
    end

    states[end] = state_update(initial_state, shocks[:, 1])

    decomposition[:, end - 2, 1] = sum(states[end]) - sum(decomposition[:, 1:end - 3, 1], dims = 2)
    decomposition[:, end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:, 1:end - 2, 1], dims = 2)

    for i in 2:size(data_in_deviations, 2)
        for ii in 1:𝓂.timings.nExo
            sck = zeros(𝓂.timings.nExo)
            sck[ii] = shocks[ii, i]
            states[ii] = state_update(states[ii] , sck)
            decomposition[:, ii, i] = sum(states[ii])
        end

        states[end] = state_update(states[end] , shocks[:, i])

        decomposition[:, end - 2, i] = sum(states[end]) - sum(decomposition[:, 1:end - 3, i], dims = 2)
        decomposition[:, end - 1, i] .= decomposition[:, end, i] - sum(decomposition[:, 1:end - 2, i], dims = 2)
    end

    return variables, shocks, [], decomposition
end

function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:third_order}, # algo
    ::Val{:inversion}; # filter
    warmup_iterations::Int = 0,
    smooth::Bool = true,
    verbose::Bool = false)

    variables, shocks, initial_state = inversion_filter(𝓂, data_in_deviations, :third_order, warmup_iterations = warmup_iterations)

    return variables, shocks, [], []
end


function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:pruned_third_order}, # algo
    ::Val{:inversion}; # filter
    warmup_iterations::Int = 0,
    smooth::Bool = true,
    verbose::Bool = false)

    algorithm = :pruned_third_order

    variables, shocks, initial_state = inversion_filter(𝓂, data_in_deviations, algorithm, warmup_iterations = warmup_iterations)

    state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)

    states = [initial_state for _ in 1:𝓂.timings.nExo + 1]

    decomposition = zeros(𝓂.timings.nVars, 𝓂.timings.nExo + 3, size(data_in_deviations, 2))

    decomposition[:, end, :] .= variables

    for i in 1:𝓂.timings.nExo
        sck = zeros(𝓂.timings.nExo)
        sck[i] = shocks[i, 1]
        states[i] = state_update(initial_state , sck)
        decomposition[:,i,1] = sum(states[i])
    end

    states[end] = state_update(initial_state, shocks[:, 1])

    decomposition[:,end - 2, 1] = sum(states[end]) - sum(decomposition[:,1:end - 3, 1], dims = 2)
    decomposition[:,end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:,1:end - 2, 1], dims = 2)

    for i in 2:size(data_in_deviations, 2)
        for ii in 1:𝓂.timings.nExo
            sck = zeros(𝓂.timings.nExo)
            sck[ii] = shocks[ii, i]
            states[ii] = state_update(states[ii] , sck)
            decomposition[:, ii, i] = sum(states[ii])
        end

        states[end] = state_update(states[end] , shocks[:, i])
        
        decomposition[:,end - 2, i] = sum(states[end]) - sum(decomposition[:,1:end - 3, i], dims = 2)
        decomposition[:,end - 1, i] .= decomposition[:, end, i] - sum(decomposition[:,1:end - 2, i], dims = 2)
    end

    return variables, shocks, [], decomposition
end


function filter_and_smooth(𝓂::ℳ, 
                            data_in_deviations::AbstractArray{Float64}, 
                            observables::Vector{Symbol}; 
                            verbose::Bool = false, 
                            lyapunov_algorithm::Symbol = :doubling,
                            tol::AbstractFloat = 1e-12)
    # Based on Durbin and Koopman (2012)
    # https://jrnold.github.io/ssmodels-in-stan/filtering-and-smoothing.html#smoothing

    @assert length(observables) == size(data_in_deviations)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    sort!(observables)

    solve!(𝓂, verbose = verbose)

    parameters = 𝓂.parameter_values

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, verbose = verbose)
    
    @assert solution_error < tol "Could not solve non stochastic steady state." 

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix

    sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    B = @views sol[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(𝓂.timings.nVars))[sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),:]

    𝐁 = B * B'

    P̄ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose, lyapunov_algorithm = lyapunov_algorithm)[1]

    n_obs = size(data_in_deviations,2)

    v = zeros(size(C,1), n_obs)
    μ = zeros(size(A,1), n_obs+1) # filtered_states
    P = zeros(size(A,1), size(A,1), n_obs+1) # filtered_covariances
    σ = zeros(size(A,1), n_obs) # filtered_standard_deviations
    iF= zeros(size(C,1), size(C,1), n_obs)
    L = zeros(size(A,1), size(A,1), n_obs)
    ϵ = zeros(size(B,2), n_obs) # filtered_shocks

    P[:, :, 1] = P̄

    # Kalman Filter
    for t in axes(data_in_deviations,2)
        v[:, t]     .= data_in_deviations[:, t] - C * μ[:, t]

        F̄ = ℒ.lu(C * P[:, :, t] * C', check = false)

        if !ℒ.issuccess(F̄) 
            @warn "Kalman filter stopped in period $t due to numerical stabiltiy issues."
            break
        end

        iF[:, :, t] .= inv(F̄)
        PCiF         = P[:, :, t] * C' * iF[:, :, t]
        L[:, :, t]  .= A - A * PCiF * C
        P[:, :, t+1].= A * P[:, :, t] * L[:, :, t]' + 𝐁
        σ[:, t]     .= sqrt.(abs.(ℒ.diag(P[:, :, t+1]))) # small numerical errors in this computation
        μ[:, t+1]   .= A * (μ[:, t] + PCiF * v[:, t])
        ϵ[:, t]     .= B' * C' * iF[:, :, t] * v[:, t]
    end


    # Historical shock decompositionm (filter)
    filter_decomposition = zeros(size(A,1), size(B,2)+2, n_obs)

    filter_decomposition[:,end,:] .= μ[:, 2:end]
    filter_decomposition[:,1:end-2,1] .= B .* repeat(ϵ[:, 1]', size(A,1))
    filter_decomposition[:,end-1,1] .= filter_decomposition[:,end,1] - sum(filter_decomposition[:,1:end-2,1],dims=2)

    for i in 2:size(data_in_deviations,2)
        filter_decomposition[:,1:end-2,i] .= A * filter_decomposition[:,1:end-2,i-1]
        filter_decomposition[:,1:end-2,i] .+= B .* repeat(ϵ[:, i]', size(A,1))
        filter_decomposition[:,end-1,i] .= filter_decomposition[:,end,i] - sum(filter_decomposition[:,1:end-2,i],dims=2)
    end
    
    μ̄ = zeros(size(A,1), n_obs) # smoothed_states
    σ̄ = zeros(size(A,1), n_obs) # smoothed_standard_deviations
    ϵ̄ = zeros(size(B,2), n_obs) # smoothed_shocks

    r = zeros(size(A,1))
    N = zeros(size(A,1), size(A,1))

    # Kalman Smoother
    for t in n_obs:-1:1
        r       .= C' * iF[:, :, t] * v[:, t] + L[:, :, t]' * r
        μ̄[:, t] .= μ[:, t] + P[:, :, t] * r
        N       .= C' * iF[:, :, t] * C + L[:, :, t]' * N * L[:, :, t]
        σ̄[:, t] .= sqrt.(abs.(ℒ.diag(P[:, :, t] - P[:, :, t] * N * P[:, :, t]'))) # can go negative
        ϵ̄[:, t] .= B' * r
    end

    # Historical shock decompositionm (smoother)
    smooth_decomposition = zeros(size(A,1), size(B,2)+2, n_obs)

    smooth_decomposition[:,end,:] .= μ̄
    smooth_decomposition[:,1:end-2,1] .= B .* repeat(ϵ̄[:, 1]', size(A,1))
    smooth_decomposition[:,end-1,1] .= smooth_decomposition[:,end,1] - sum(smooth_decomposition[:,1:end-2,1],dims=2)

    for i in 2:size(data_in_deviations,2)
        smooth_decomposition[:,1:end-2,i] .= A * smooth_decomposition[:,1:end-2,i-1]
        smooth_decomposition[:,1:end-2,i] .+= B .* repeat(ϵ̄[:, i]', size(A,1))
        smooth_decomposition[:,end-1,i] .= smooth_decomposition[:,end,i] - sum(smooth_decomposition[:,1:end-2,i],dims=2)
    end

    return μ̄, σ̄, ϵ̄, smooth_decomposition, μ[:, 2:end], σ, ϵ, filter_decomposition
end


# if VERSION >= v"1.9"
#     @setup_workload begin
#         # Putting some things in `setup` can reduce the size of the
#         # precompile file and potentially make loading faster.
#         @model FS2000 precompile = true begin
#             dA[0] = exp(gam + z_e_a  *  e_a[x])
#             log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  e_m[x]
#             - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0
#             W[0] = l[0] / n[0]
#             - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0
#             R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]
#             1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0
#             c[0] + k[0] = exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  *  e_a[x])) * k[-1]
#             P[0] * c[0] = m[0]
#             m[0] - 1 + d[0] = l[0]
#             e[0] = exp(z_e_a  *  e_a[x])
#             y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x]))
#             gy_obs[0] = dA[0] * y[0] / y[-1]
#             gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]
#             log_gy_obs[0] = log(gy_obs[0])
#             log_gp_obs[0] = log(gp_obs[0])
#         end

#         @parameters FS2000 silent = true precompile = true begin  
#             alp     = 0.356
#             bet     = 0.993
#             gam     = 0.0085
#             mst     = 1.0002
#             rho     = 0.129
#             psi     = 0.65
#             del     = 0.01
#             z_e_a   = 0.035449
#             z_e_m   = 0.008862
#         end
        
#         ENV["GKSwstype"] = "nul"

#         @compile_workload begin
#             # all calls in this block will be precompiled, regardless of whether
#             # they belong to your package or not (on Julia 1.8 and higher)
#             @model RBC precompile = true begin
#                 1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
#                 c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
#                 z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
#             end

#             @parameters RBC silent = true precompile = true begin
#                 δ = 0.02
#                 α = 0.5
#             end

#             get_SS(FS2000, silent = true)
#             get_SS(FS2000, parameters = :alp => 0.36, silent = true)
#             get_solution(FS2000, silent = true)
#             get_solution(FS2000, parameters = :alp => 0.35)
#             get_standard_deviation(FS2000)
#             get_correlation(FS2000)
#             get_autocorrelation(FS2000)
#             get_variance_decomposition(FS2000)
#             get_conditional_variance_decomposition(FS2000)
#             get_irf(FS2000)

#             data = simulate(FS2000)([:c,:k],:,:simulate)
#             get_loglikelihood(FS2000, data, FS2000.parameter_values)
#             get_mean(FS2000, silent = true)
#             # get_SSS(FS2000, silent = true)
#             # get_SSS(FS2000, algorithm = :third_order, silent = true)

#             # import StatsPlots
#             # plot_irf(FS2000)
#             # plot_solution(FS2000,:k) # fix warning when there is no sensitivity and all values are the same. triggers: no strict ticks found...
#             # plot_conditional_variance_decomposition(FS2000)
#         end
#     end
# end

end
