module MacroModelling


import DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF, TYPEDSIGNATURES, TYPEDFIELDS
# import StatsFuns: normcdf
import ThreadedSparseArrays
using PrecompileTools
import SpecialFunctions: erfcinv, erfc
import SymPyPythonCall as SPyPyC
import Symbolics

import AbstractDifferentiation as 𝒜
import ForwardDiff as ℱ
# import Diffractor: DiffractorForwardBackend
𝒷 = 𝒜.ForwardDiffBackend
# 𝒷 = Diffractor.DiffractorForwardBackend

import NLopt
import Optim, LineSearches
# import Zygote
import SparseArrays: SparseMatrixCSC, SparseVector, AbstractSparseArray#, sparse, spzeros, droptol!, sparsevec, spdiagm, findnz#, sparse!
import LinearAlgebra as ℒ
# import ComponentArrays as 𝒞
import BlockTriangularForm
import Subscripts: super, sub
import Krylov
import LinearOperators
import DataStructures: CircularBuffer
import ImplicitDifferentiation as ℐ
import SpeedMapping: speedmapping
import Suppressor: @suppress
import REPL
import Unicode
import MatrixEquations # good overview: https://cscproxy.mpi-magdeburg.mpg.de/mpcsc/benner/talks/Benner-Melbourne2019.pdf
# import NLboxsolve: nlboxsolve
# using NamedArrays
# using AxisKeys

import ChainRulesCore: @ignore_derivatives, ignore_derivatives
import RecursiveFactorization as RF

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

using Requires

import Reexport
Reexport.@reexport using AxisKeys
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
include("macros.jl")
include("get_functions.jl")
include("dynare.jl")
include("inspect.jl")

function __init__()
    @require StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd" include("plotting.jl")
    @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("priors.jl")
end


export @model, @parameters, solve!
export plot_irfs, plot_irf, plot_IRF, plot_simulations, plot_solution, plot_simulation#, plot
export plot_conditional_variance_decomposition, plot_forecast_error_variance_decomposition, plot_fevd, plot_model_estimates, plot_shock_decomposition
export get_irfs, get_irf, get_IRF, simulate, get_simulation, get_simulations
export get_conditional_forecast, plot_conditional_forecast
export get_solution, get_first_order_solution, get_perturbation_solution, get_second_order_solution, get_third_order_solution, get_eigenvalues
export get_steady_state, get_SS, get_ss, get_non_stochastic_steady_state, get_stochastic_steady_state, get_SSS, steady_state, SS, SSS, ss, sss
export get_moments, get_statistics, get_covariance, get_standard_deviation, get_variance, get_var, get_std, get_stdev, get_cov, var, std, stdev, cov, get_mean#, mean
export get_autocorrelation, get_correlation, get_variance_decomposition, get_corr, get_autocorr, get_var_decomp, corr, autocorr
export get_fevd, fevd, get_forecast_error_variance_decomposition, get_conditional_variance_decomposition
export calculate_jacobian, calculate_hessian, calculate_third_order_derivatives
export calculate_first_order_solution, calculate_second_order_solution, calculate_third_order_solution#, calculate_jacobian_manual, calculate_jacobian_sparse, calculate_jacobian_threaded
export get_shock_decomposition, get_estimated_shocks, get_estimated_variables, get_estimated_variable_standard_deviations, get_loglikelihood
export plotlyjs_backend, gr_backend
export Beta, InverseGamma, Gamma, Normal

export translate_mod_file, translate_dynare_file, import_model, import_dynare
export write_mod_file, write_dynare_file, write_to_dynare_file, write_to_dynare, export_dynare, export_to_dynare, export_mod_file, export_model

export get_equations, get_steady_state_equations, get_dynamic_equations, get_calibration_equations, get_parameters, get_calibrated_parameters, get_parameters_in_equations, get_parameters_defined_by_parameters, get_parameters_defining_parameters, get_calibration_equation_parameters, get_variables, get_nonnegativity_auxilliary_variables, get_dynamic_auxilliary_variables, get_shocks, get_state_variables, get_jump_variables
# Internal
export irf, girf

# Remove comment for debugging
# export riccati_forward, block_solver, remove_redundant_SS_vars!, write_parameters_input!, parse_variables_input_to_index, undo_transformer , transformer, SSS_third_order_parameter_derivatives, SSS_second_order_parameter_derivatives, calculate_third_order_stochastic_steady_state, calculate_second_order_stochastic_steady_state, filter_and_smooth
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


function transform_obc(ex::Expr)
    transformed_expr, reverse_dict = transform_expression(ex)

    for symbs in get_symbols(transformed_expr)
        eval(:($symbs = SPyPyC.symbols($(string(symbs)), real = true, finite = true)))
    end

    eq = eval(transformed_expr)

    soll = try SPyPyC.solve(eq, eval(:minmax__P))
    catch
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
        jac .= 𝒜.jacobian(𝒷(), xx -> 𝓂.obc_violation_function(xx, p), X)[1]'
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
    Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, 𝒷, precision_factor = p

    if length(jac) > 0
        jac .= 𝒜.jacobian(𝒷(), xx -> begin
                                        Shocks[Free_shock_idx] .= xx

                                        new_State = State_update(State, convert(typeof(xx), Shocks))

                                        cond_vars = Pruning ? sum(new_State) : new_State
                                        
                                        return precision_factor .* abs.(Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
                                    end, X)[1]'
    end

    Shocks[Free_shock_idx] .= X

    new_State = State_update(State, convert(typeof(X), Shocks))

    cond_vars = Pruning ? sum(new_State) : new_State

    res .= precision_factor .* abs.(Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
end


function minimize_distance_to_conditions(X::Vector{S}, p)::S where S
    Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, 𝒷, precision_factor = p

    Shocks[Free_shock_idx] .= X

    new_State = State_update(State, convert(typeof(X), Shocks))

    cond_vars = Pruning ? sum(new_State) : new_State

    return precision_factor * sum(abs2, Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
end



function minimize_distance_to_conditions!(X::Vector{S}, grad::Vector{S}, p) where S
    Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, 𝒷, precision_factor = p

    if length(grad) > 0
        grad .= 𝒜.gradient(𝒷(), xx -> begin
                                        Shocks[Free_shock_idx] .= xx

                                        new_State = State_update(State, convert(typeof(xx), Shocks))

                                        cond_vars = Pruning ? sum(new_State) : new_State
                                        
                                        return precision_factor * sum(abs2, Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
                                    end, X)[1]
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
        grad .= 𝒜.gradient(𝒷(), xx -> begin
                                        state_copy = deepcopy(state)

                                        XX = reshape(xx, length(X) ÷ warmup_iters, warmup_iters)

                                        for i in 1:warmup_iters
                                            state_copy = state_update(state_copy, XX[:,i])
                                        end

                                        return precision_factor .* sum(abs2, data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
                                    end, X)[1]
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
        jac .= 𝒜.jacobian(𝒷(), xx -> begin
                                        state_copy = deepcopy(state)

                                        XX = reshape(xx, length(X) ÷ warmup_iters, warmup_iters)

                                        for i in 1:warmup_iters
                                            state_copy = state_update(state_copy, XX[:,i])
                                        end

                                        return precision_factor .* abs.(data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])
                                    end, X)[1]'
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
        grad .= 𝒜.gradient(𝒷(), xx -> precision_factor .* sum(abs2, Data - (pruning ? sum(state_update(State, xx)) : state_update(State, xx))[cond_var_idx]), X)[1]
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
        jac .= 𝒜.jacobian(𝒷(), xx -> precision_factor .* abs.(Data - (pruning ? sum(state_update(State, xx)) : state_update(State, xx))[cond_var_idx]), X)[1]'
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
                


function jacobian_wrt_values(A, B)
    # does this without creating dense arrays: reshape(permutedims(reshape(ℒ.I - ℒ.kron(A, B) ,size(B,1), size(A,1), size(A,1), size(B,1)), [2, 3, 4, 1]), size(A,1) * size(B,1), size(A,1) * size(B,1))

    # Compute the Kronecker product and subtract from identity
    C = ℒ.I - ℒ.kron(A, B)

    # Extract the row, column, and value indices from C
    rows, cols, vals = findnz(C)

    # Lists to store the 2D indices after the operations
    final_rows = zeros(Int,length(rows))
    final_cols = zeros(Int,length(rows))

    Threads.@threads for i = 1:length(rows)
        # Convert the 1D row index to its 2D components
        i1, i2 = divrem(rows[i]-1, size(B,1)) .+ 1

        # Convert the 1D column index to its 2D components
        j1, j2 = divrem(cols[i]-1, size(A,1)) .+ 1

        # Convert the 4D index (i1, j2, j1, i2) to a 2D index in the final matrix
        final_col, final_row = divrem(Base._sub2ind((size(A,1), size(A,1), size(B,1), size(B,1)), i1, j2, j1, i2) - 1, size(A,1) * size(B,1)) .+ 1

        # Store the 2D indices
        final_rows[i] = final_row
        final_cols[i] = final_col
    end

    return sparse(final_rows, final_cols, vals, size(A,1) * size(B,1), size(A,1) * size(B,1))
end




function jacobian_wrt_A(A, X)
    # does this without creating dense arrays: reshape(permutedims(reshape(ℒ.I - ℒ.kron(A, B) ,size(B,1), size(A,1), size(A,1), size(B,1)), [2, 3, 4, 1]), size(A,1) * size(B,1), size(A,1) * size(B,1))

    # Compute the Kronecker product and subtract from identity
    C = ℒ.kron(ℒ.I(size(A,1)), sparse(A * X))

    # Extract the row, column, and value indices from C
    rows, cols, vals = findnz(C)

    # Lists to store the 2D indices after the operations
    final_rows = zeros(Int,length(rows))
    final_cols = zeros(Int,length(rows))

    Threads.@threads for i = 1:length(rows)
        # Convert the 1D row index to its 2D components
        i1, i2 = divrem(rows[i]-1, size(A,1)) .+ 1

        # Convert the 1D column index to its 2D components
        j1, j2 = divrem(cols[i]-1, size(A,1)) .+ 1

        # Convert the 4D index (i1, j2, j1, i2) to a 2D index in the final matrix
        final_col, final_row = divrem(Base._sub2ind((size(A,1), size(A,1), size(A,1), size(A,1)), i2, i1, j1, j2) - 1, size(A,1) * size(A,1)) .+ 1

        # Store the 2D indices
        final_rows[i] = final_row
        final_cols[i] = final_col
    end

    r,c,_ = findnz(A) 
    
    non_zeros_only = spzeros(Int,size(A,1)^2,size(A,1)^2)
    
    non_zeros_only[CartesianIndex.(r .+ (c.-1) * size(A,1), r .+ (c.-1) * size(A,1))] .= 1
    
    return sparse(final_rows, final_cols, vals, size(A,1) * size(A,1), size(A,1) * size(A,1)) + ℒ.kron(sparse(X * A'), ℒ.I(size(A,1)))' * non_zeros_only
end


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

    sort!(observables)
    
    return observables
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

# function convert_superscript_to_integer(str::String)
#     # Regular expression to match superscript numbers in brackets
#     regex = r"⁽([⁰¹²³⁴⁵⁶⁷⁸⁹]+)⁾$"

#     # Mapping of superscript characters to their integer values
#     superscript_map = Dict('⁰'=>0, '¹'=>1, '²'=>2, '³'=>3, '⁴'=>4, '⁵'=>5, '⁶'=>6, '⁷'=>7, '⁸'=>8, '⁹'=>9)

#     # Check for a match and process if found
#     if occursin(regex, str)
#         matched = match(regex, str).captures[1]
#         # Convert each superscript character to a digit and concatenate
#         digits = [superscript_map[c] for c in matched]
#         # Convert array of digits to integer
#         return parse(Int, join(digits))
#     else
#         return nothing
#     end
# end

function match_pattern(strings::Union{Set,Vector}, pattern::Regex)
    return filter(r -> match(pattern, string(r)) !== nothing, strings)
end

# try: run optim only if there is a violation / capture case with small shocks and set them to zero
function parse_occasionally_binding_constraints(equations_block; max_obc_horizon::Int = 40)
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


function get_and_check_initial_state(𝓂::ℳ, initial_state::Union{Vector{Vector{Float64}}, Vector{Float64}, Symbol}, reference_steady_state::Vector{Float64}, NSSS::Vector{Float64}, SSS_delta::Vector{Float64}, algorithm::Symbol)::Union{Vector{Vector{Float64}}, Vector{Float64}}
	if initial_state isa Symbol
		if initial_state ∈ [:relevant_SS, :relevant_ss, :relevant_steady_state]
            init_state = zeros(𝓂.timings.nVars) - SSS_delta
		elseif initial_state ∈ [:SSS, :sss, :stochastic_steady_state]
            init_state = zeros(𝓂.timings.nVars) - SSS_delta
			
            if algorithm ∈ [:second_order, :third_order, :pruned_second_order, :pruned_third_order]
				@warn "Algorithm: $algorithm has no stochastic steady state. Continuing with the non stochastic steady state as the initial state."
				init_state = zeros(𝓂.timings.nVars)
			end
		elseif initial_state ∈ [:NSSS, :nsss, :non_stochastic_steady_state]
			init_state = zeros(𝓂.timings.nVars)
		elseif initial_state == :mean
            if algorithm == :first_order
			    init_state = zeros(𝓂.timings.nVars)
            else
                @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] "Mean only available for first order, pruned second order, or pruned third order solution."
			end
		else
			@assert initial_state ∈ [:NSSS, :nsss, :non_stochastic_steady_state, :SSS, :sss, :stochastic_steady_state, :relevant_SS, :relevant_ss, :relevant_steady_state] "No valid input of type Symbol."
		end
	end
	
    if initial_state isa Vector{Float64}
        init_state = initial_state - NSSS
    elseif initial_state isa Vector{Vector{Float64}}
        if algorithm ∉ [:pruned_second_order, :pruned_third_order]
            @assert initial_state isa Vector{Float64} "The solution algorithm has one state vector: initial_state must be a Vector{Float64}."
		else
			init_state = initial_state
        end
    end
    
    if algorithm == :pruned_second_order
        init_state = init_state isa Vector{Float64} ? [zeros(𝓂.timings.nVars), init_state] : init_state
    elseif algorithm == :pruned_third_order
        init_state = init_state isa Vector{Float64} ? [zeros(𝓂.timings.nVars), init_state, zeros(𝓂.timings.nVars)] : init_state
    end

    return init_state
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
    upper_bounds::Array{T,1}; 
    parameters::solver_parameters = solver_parameters(eps(), eps(), 250, 2.9912988764832833, 0.8725, 0.0027, 0.028948770826150612, 8.04, 4.076413176215408, 0.06375413238034794, 0.24284340766769424, 0.5634017580097571, 0.009549630552246828, 0.6342888355132347, 0.5275522227754195, 1.0, 0.06178989216048817, 0.5234277812131813, 0.422, 0.011209254402846185, 0.5047, 0.6020757011698457, 1, 0.0, 2)
    ) where {T <: AbstractFloat}
    # issues with optimization: https://www.gurobi.com/documentation/8.1/refman/numerics_gurobi_guidelines.html

    xtol = parameters.xtol
    ftol = parameters.ftol
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

    largest_step = zero(T)
    largest_residual = zero(T)

    μ¹ = μ̄¹
    μ² = μ̄²

    p¹ = p̄¹
    p² = p̄²

	for iter in 1:iterations
        ∇ .= 𝒜.jacobian(𝒷(), f̂,current_guess)[1]

        previous_guess .= current_guess

        ∇̂ .= ∇' * ∇

        ∇̂ .+= μ¹ * sum(abs2, f̂(current_guess))^p¹ * ℒ.I + μ² * ℒ.Diagonal(∇̂).^p²

        if !all(isfinite,∇̂)
            return undo_transform(current_guess,transformation_level), (iter, Inf, Inf, upper_bounds)
            # return undo_transform(current_guess,transformation_level,shift), (iter, Inf, Inf, upper_bounds)
        end

        ∇̄ = RF.lu(∇̂, check = false)

        if !ℒ.issuccess(∇̄)
            ∇̄ = ℒ.svd(∇̂)
        end

        current_guess .-= ∇̄ \ ∇' * f̂(current_guess)

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

                current_guess .= previous_guess + α * guess_update
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
        largest_residual = maximum(abs, f(undo_transform(current_guess,transformation_level)))
        # largest_residual = maximum(abs, f(undo_transform(current_guess,transformation_level,shift)))

        if largest_step <= xtol || largest_residual <= ftol
            return undo_transform(current_guess,transformation_level), (iter, largest_step, largest_residual, f(undo_transform(current_guess,transformation_level)))
            # return undo_transform(current_guess,transformation_level,shift), (iter, largest_step, largest_residual, f(undo_transform(current_guess,transformation_level,shift)))
        end
    end

    best_guess = undo_transform(current_guess,transformation_level)
    # best_guess = undo_transform(current_guess,transformation_level,shift)

    return best_guess, (iterations, largest_step, largest_residual, f(best_guess))
end


function expand_steady_state(SS_and_pars::Vector{M},𝓂::ℳ) where M
    all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
    NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
    
    [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
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

    l_bnds = Dict(𝓂.bounded_vars .=> 𝓂.lower_bounds)
    u_bnds = Dict(𝓂.bounded_vars .=> 𝓂.upper_bounds)

    symbols_pos = []
    symbols_neg = []
    symbols_none = []

    for symb in symbols_in_equation
        if symb in 𝓂.bounded_vars
            if l_bnds[symb] >= 0
                push!(symbols_pos, symb)
            elseif u_bnds[symb] <= 0
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



function remove_redundant_SS_vars!(𝓂::ℳ, Symbolics::symbolics)
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
            soll = try SPyPyC.solve(ss_equations[i],var_to_solve_for)
            catch
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




function solve_steady_state!(𝓂::ℳ, symbolic_SS, Symbolics::symbolics; verbose::Bool = false)
    unknowns = union(Symbolics.vars_in_ss_equations,Symbolics.calibration_equations_parameters)

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
    relevant_pars_across = []
    NSSS_solver_cache_init_tmp = []

    min_max_errors = []

    n_block = 1

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

            soll = try SPyPyC.solve(eq_to_solve,var_to_solve_for)
            catch
            end
            
            if isnothing(soll)
                # println("Could not solve single variables case symbolically.")
                println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
                # solve numerically
                continue
            # elseif PythonCall.pyconvert(Bool,soll[1].is_number)

            elseif soll[1].is_number == true
                # ss_equations = ss_equations.subs(var_to_solve,soll[1])
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
                
                # atoms = reduce(union,soll[1].atoms())

                [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
                push!(atoms_in_equations_list, Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)),Symbol.(soll[1].atoms()))))

                # println(atoms_in_equations)
                # push!(atoms_in_equations, soll[1].atoms())

                if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = min(max($(𝓂.lower_bounds[indexin([𝓂.solved_vars[end]],𝓂.bounded_vars)][1]),$(𝓂.solved_vals[end])),$(𝓂.upper_bounds[indexin([𝓂.solved_vars[end]],𝓂.bounded_vars)][1]))))
                else
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                end
            end

            # push!(single_eqs,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
            # solve symbolically
        else

            vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]

            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false
            
            if symbolic_SS
                soll = try SPyPyC.solve(eqs_to_solve,vars_to_solve)
                # soll = try solve(SPyPyC.Sym(eqs_to_solve),var_order)#,check=false,force = true,manual=true)
                catch
                end

                if isnothing(soll)
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # continue
                elseif length(soll) == 0
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # continue
                # elseif length(intersect(vars_to_solve,reduce(union,map(x->x.atoms(),collect(soll[1]))))) > 0
                elseif length(intersect((union(SPyPyC.free_symbols.(soll[1])...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
                # elseif length(intersect(union(SPyPyC.free_symbols.(soll[1])...),vars_to_solve)) > 0
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # println("Could not solve for: ",intersect(var_list,reduce(union,map(x->x.atoms(),solll)))...)
                    # break_ind = true
                    # break
                else
                    if verbose
                        println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.")
                    end
                    # relevant_pars = reduce(union,vcat(𝓂.par_list,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
                    # relevant_pars = reduce(union,map(x->x.atoms(),collect(soll[1])))
                    atoms = reduce(union,map(x->x.atoms(),collect(soll[1])))
                    # println(atoms)
                    [push!(atoms_in_equations, Symbol(a)) for a in atoms]
                    
                    for (k, vars) in enumerate(vars_to_solve)
                        push!(𝓂.solved_vars,Symbol(vars))
                        push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

                        push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
                        # push!(atoms_in_equations_list, Set(Symbol.(soll[vars].atoms()))) # to be fixed
                        push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                    end
                end


            end
                
            # try symbolically and use numerical if it does not work
            if numerical_sol || !symbolic_SS
                if !symbolic_SS && verbose
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
                end
                
                push!(𝓂.solved_vars,Symbol.(vars_to_solve))
                push!(𝓂.solved_vals,Meta.parse.(string.(eqs_to_solve)))

                syms_in_eqs = Set()

                for i in eqs_to_solve
                    # push!(syms_in_eqs, Symbol.(PythonCall.pystr.(i.atoms()))...)
                    push!(syms_in_eqs, Symbol.(SPyPyC.:↓(SPyPyC.free_symbols(i)))...)
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
                solved_vals = []
                
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

                # println(intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars), sort(𝓂.solved_vars[end]) ), union(syms_in_eqs, other_vrs_eliminated_by_sympy ) ))
                # println(other_vrs)
                for var in other_vrs
                    # var_idx = findfirst(x -> x == var, union(𝓂.var,𝓂.calibration_equations_parameters))
                    push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
                    push!(other_vars_input,:($(var)))
                    iii += 1
                    # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
                end

                # augment system for bound constraint violations
                # aug_lag = []
                # aug_lag_penalty = []
                # push!(aug_lag_penalty, :(bound_violation_penalty = 0))

                # for varpar in intersect(𝓂.bounded_vars,union(other_vrs,sorted_vars,relevant_pars))
                #     i = indexin([varpar],𝓂.bounded_vars)
                #     push!(aug_lag,:($varpar = min(max($varpar,$(𝓂.lower_bounds[i...])),$(𝓂.upper_bounds[i...]))))
                #     push!(aug_lag_penalty,:(bound_violation_penalty += max(0,$(𝓂.lower_bounds[i...]) - $varpar) + max(0,$varpar - $(𝓂.upper_bounds[i...]))))
                # end


                # add it also to output from optimisation, in case you use optimiser without bounds
                # aug_lag_results = []

                # for varpar in intersect(𝓂.bounded_vars,sorted_vars)
                #     i = indexin([varpar],𝓂.bounded_vars)
                #     push!(aug_lag_results,:($varpar = min(max($varpar,𝓂.lower_bounds[$i...]),𝓂.upper_bounds[$i...])))
                # end

                # funcs_no_transform = :(function block(parameters_and_solved_vars::Vector{Float64}, guess::Vector{Float64})
                #         # if guess isa Tuple guess = guess[1] end
                #         # guess = undo_transformer(guess) 
                #         # println(guess)
                #         $(guess...) 
                #         $(calib_pars...) # add those variables which were previously solved and are used in the equations
                #         $(other_vars...) # take only those that appear in equations - DONE

                #         # $(aug_lag...)
                #         # $(nnaux_linear...)
                #         return [$(solved_vals...),$(nnaux_linear...)]
                #     end)

# println(solved_vals)
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

                # push!(solved_vals,:(aux_error))
                # push!(solved_vals,:(bound_violation_penalty))

                #funcs_optim = :(function block(guess::Vector{Float64},transformer_parameters_and_solved_vars::Tuple{Vector{Float64},Int})
                    #guess = undo_transformer(guess,option = transformer_parameters_and_solved_vars[2])
                    #parameters_and_solved_vars = transformer_parameters_and_solved_vars[1]
                #  $(guess...) 
                # $(calib_pars...) # add those variables which were previously solved and are used in the equations
                #   $(other_vars...) # take only those that appear in equations - DONE

                    # $(aug_lag_penalty...)
                    # $(aug_lag...)
                    # $(nnaux...) # not needed because the aux vars are inputs
                    # $(nnaux_error...)
                    #return sum(abs2,[$(solved_vals...),$(nnaux_linear...)])
                #end)
            
                push!(NSSS_solver_cache_init_tmp,fill(0.897,length(sorted_vars)))

                # WARNING: infinite bounds are transformed to 1e12
                lbs = []
                ubs = []
                
                limit_boundaries = 1e12

                for i in sorted_vars
                    if i ∈ 𝓂.bounded_vars
                        push!(lbs,𝓂.lower_bounds[i .== 𝓂.bounded_vars][1] == -Inf ? -limit_boundaries+rand() : 𝓂.lower_bounds[i .== 𝓂.bounded_vars][1])
                        push!(ubs,𝓂.upper_bounds[i .== 𝓂.bounded_vars][1] ==  Inf ?  limit_boundaries-rand() : 𝓂.upper_bounds[i .== 𝓂.bounded_vars][1])
                    else
                        push!(lbs,-limit_boundaries+rand())
                        push!(ubs,limit_boundaries+rand())
                    end
                end
                push!(SS_solve_func,:(lbs = [$(lbs...)]))
                push!(SS_solve_func,:(ubs = [$(ubs...)]))
                # push!(SS_solve_func,:(𝓂.SS_init_guess = initial_guess))
                # push!(SS_solve_func,:(f = OptimizationFunction(𝓂.ss_solve_blocks_optim[$(n_block)], Optimization.AutoForwardDiff())))
                # push!(SS_solve_func,:(inits = max.(lbs,min.(ubs,𝓂.SS_init_guess[$([findfirst(x->x==y,union(𝓂.var,𝓂.calibration_equations_parameters)) for y in sorted_vars])]))))
                # push!(SS_solve_func,:(closest_solution = 𝓂.NSSS_solver_cache[findmin([sum(abs2,pars[end] - params_flt) for pars in 𝓂.NSSS_solver_cache])[2]]))
                # push!(SS_solve_func,:(inits = [transformer(max.(lbs,min.(ubs, closest_solution[$(n_block)] ))),closest_solution[end]]))
                push!(SS_solve_func,:(inits = max.(lbs,min.(ubs, closest_solution[$(n_block)]))))

                # push!(SS_solve_func,:(println([$(calib_pars_input...),$(other_vars_input...)])))

                if VERSION >= v"1.9"
                    push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())))
                else
                    push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver())))
                end

                push!(SS_solve_func,:(solution = block_solver_AD([$(calib_pars_input...),$(other_vars_input...)],
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
                
                # push!(SS_solve_func,:(solution = block_solver_RD([$(calib_pars_input...),$(other_vars_input...)])))#, 
                        # $(n_block), 
                        # 𝓂.ss_solve_blocks[$(n_block)], 
                        # # 𝓂.SS_optimizer, 
                        # f, 
                        # inits,
                        # lbs, 
                        # ubs,
                        # fail_fast_solvers_only = fail_fast_solvers_only,
                        # verbose = verbose)))
                push!(SS_solve_func,:(iters += solution[2][2])) 
                push!(SS_solve_func,:(solution_error += solution[2][1])) 
                push!(SS_solve_func,:(sol = solution[1]))
                # push!(SS_solve_func,:(solution_error += sum(abs2,𝓂.ss_solve_blocks[$(n_block)]([$(calib_pars_input...),$(other_vars_input...)],solution))))
                # push!(SS_solve_func,:(sol = solution))

                # push!(SS_solve_func,:(println(sol))) 

                push!(SS_solve_func,:($(result...)))   
                # push!(SS_solve_func,:($(aug_lag_results...))) 

                # push!(SS_solve_func,:(NSSS_solver_cache_tmp = []))
                # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol))))
                push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))

                push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
                # push!(𝓂.ss_solve_blocks_no_transform,@RuntimeGeneratedFunction(funcs_no_transform))
                # push!(𝓂.ss_solve_blocks_optim,@RuntimeGeneratedFunction(funcs_optim))
                
                n_block += 1
            end
        end
        n -= 1
    end

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
            push!(parameters_in_equations,:($parss = params[$i]))
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
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [params_scaled_flt] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp...,params_scaled_flt] end))
    
    push!(SS_solve_func,:(current_best = sqrt(sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt))))# / max(sum(abs2,𝓂.NSSS_solver_cache[end][end]), sum(abs2,params_flt))))

    push!(SS_solve_func,:(for pars in 𝓂.NSSS_solver_cache
                                latest = sqrt(sum(abs2,pars[end] - params_flt))# / max(sum(abs2,pars[end]), sum(abs,params_flt))
                                if latest <= current_best
                                    current_best = latest
                                end
                            end))

    push!(SS_solve_func,:(if (current_best > 1e-5) && (solution_error < eps(Float64))
                                reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
                                solved_scale = scale
                            end))
    # push!(SS_solve_func,:(if length(𝓂.NSSS_solver_cache) > 100 popfirst!(𝓂.NSSS_solver_cache) end))
    
    # push!(SS_solve_func,:(SS_init_guess = ([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)])))

    # push!(SS_solve_func,:(𝓂.SS_init_guess = typeof(SS_init_guess) == Vector{Float64} ? SS_init_guess : ℱ.value.(SS_init_guess)))

    # push!(SS_solve_func,:(return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]))))


    # fix parameter bounds
    par_bounds = []
    
    for varpar in intersect(𝓂.bounded_vars, intersect(𝓂.parameters,union(atoms_in_equations, relevant_pars_across)))
        i = indexin([varpar],𝓂.bounded_vars)
        push!(par_bounds, :($varpar = min(max($varpar,$(𝓂.lower_bounds[i...])),$(𝓂.upper_bounds[i...]))))
    end


    solve_exp = :(function solve_SS(parameters::Vector{Real}, 
                                    𝓂::ℳ,
                                    # fail_fast_solvers_only::Bool, 
                                    verbose::Bool, 
                                    cold_start::Union{Bool,Float64},
                                    solver_parameters::solver_parameters)

                    params_flt = typeof(parameters) == Vector{Float64} ? parameters : ℱ.value.(parameters)
                    current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                    closest_solution_init = 𝓂.NSSS_solver_cache[end]
                    for pars in 𝓂.NSSS_solver_cache
                        latest = sum(abs2,pars[end] - params_flt)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end
                    solved_scale = 0
                    range_length = [1]#fail_fast_solvers_only ? [1] : [ 1, 2, 4, 8,16,32]
                    for r in range_length
                        rangee = ignore_derivatives(range(0,1,r+1))
                        for scale in rangee[2:end]
                            if scale <= solved_scale continue end
                            current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                            closest_solution = 𝓂.NSSS_solver_cache[end]
                            for pars in 𝓂.NSSS_solver_cache
                                latest = sum(abs2,pars[end] - params_flt)
                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end
                            params = all(isfinite.(closest_solution_init[end])) && parameters != closest_solution_init[end] ? scale * parameters + (1 - scale) * closest_solution_init[end] : parameters
                            params_scaled_flt = typeof(params) == Vector{Float64} ? params : ℱ.value.(params)
                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.calibration_equations_no_var...)
                            NSSS_solver_cache_tmp = []
                            solution_error = 0.0
                            iters = 0
                            $(SS_solve_func...)
                            if scale == 1
                                # return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                # NSSS_solution = [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)]
                                # NSSS_solution[abs.(NSSS_solution) .< 1e-12] .= 0 # doesnt work with Zygote
                                return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                            end
                        end
                    end
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
        solved_vals = []
        
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

        push!(NSSS_solver_cache_init_tmp,fill(0.897,length(sorted_vars)))

        # WARNING: infinite bounds are transformed to 1e12
        lbs = []
        ubs = []
        
        limit_boundaries = 1e12

        for i in sorted_vars
            if i ∈ 𝓂.bounded_vars
                push!(lbs,𝓂.lower_bounds[i .== 𝓂.bounded_vars][1] == -Inf ? -limit_boundaries+rand() : 𝓂.lower_bounds[i .== 𝓂.bounded_vars][1])
                push!(ubs,𝓂.upper_bounds[i .== 𝓂.bounded_vars][1] ==  Inf ?  limit_boundaries-rand() : 𝓂.upper_bounds[i .== 𝓂.bounded_vars][1])
            else
                push!(lbs,-limit_boundaries+rand())
                push!(ubs,limit_boundaries+rand())
            end
        end
        push!(SS_solve_func,:(lbs = [$(lbs...)]))
        push!(SS_solve_func,:(ubs = [$(ubs...)]))
        
        push!(SS_solve_func,:(inits = max.(lbs,min.(ubs, closest_solution[$(n_block)]))))

        if VERSION >= v"1.9"
            push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())))
        else
            push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver())))
        end

        push!(SS_solve_func,:(solution = block_solver_AD(length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)],
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

        push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
        
        n_block += 1
        
        n -= 1
    end

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
            push!(parameters_in_equations,:($parss = params[$i]))
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
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [params_scaled_flt] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp...,params_scaled_flt] end))
    
    push!(SS_solve_func,:(current_best = sqrt(sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt))))# / max(sum(abs2,𝓂.NSSS_solver_cache[end][end]), sum(abs2,params_flt))))

    push!(SS_solve_func,:(for pars in 𝓂.NSSS_solver_cache
                                latest = sqrt(sum(abs2,pars[end] - params_flt))# / max(sum(abs2,pars[end]), sum(abs,params_flt))
                                if latest <= current_best
                                    current_best = latest
                                end
                            end))

    push!(SS_solve_func,:(if (current_best > 1e-5) && (solution_error < eps(Float64))
                                reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
                                solved_scale = scale
                            end))

    # fix parameter bounds
    par_bounds = []
    
    for varpar in intersect(𝓂.bounded_vars, intersect(𝓂.parameters,union(atoms_in_equations, relevant_pars_across)))
        i = indexin([varpar],𝓂.bounded_vars)
        push!(par_bounds, :($varpar = min(max($varpar,$(𝓂.lower_bounds[i...])),$(𝓂.upper_bounds[i...]))))
    end


    solve_exp = :(function solve_SS(parameters::Vector{Real}, 
                                    𝓂::ℳ, 
                                    # fail_fast_solvers_only::Bool, 
                                    verbose::Bool, 
                                    cold_start::Union{Bool,Float64},
                                    solver_parameters::solver_parameters)

                    params_flt = typeof(parameters) == Vector{Float64} ? parameters : ℱ.value.(parameters)
                    current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                    closest_solution_init = 𝓂.NSSS_solver_cache[end]
                    for pars in 𝓂.NSSS_solver_cache
                        latest = sum(abs2,pars[end] - params_flt)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end
                    solved_scale = 0
                    range_length = [1]#fail_fast_solvers_only ? [1] : [ 1, 2, 4, 8,16,32]
                    for r in range_length
                        rangee = ignore_derivatives(range(0,1,r+1))
                        for scale in rangee[2:end]
                            if scale <= solved_scale continue end
                            current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - params_flt)
                            closest_solution = 𝓂.NSSS_solver_cache[end]
                            for pars in 𝓂.NSSS_solver_cache
                                latest = sum(abs2,pars[end] - params_flt)
                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end
                            params = all(isfinite.(closest_solution_init[end])) && parameters != closest_solution_init[end] ? scale * parameters + (1 - scale) * closest_solution_init[end] : parameters
                            params_scaled_flt = typeof(params) == Vector{Float64} ? params : ℱ.value.(params)
                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.calibration_equations_no_var...)
                            NSSS_solver_cache_tmp = []
                            iters = 0
                            solution_error = 0.0
                            $(SS_solve_func...)
                            if scale == 1
                                # return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)] , (solution_error, iters)
                            end
                        end
                    end
                end)

    𝓂.SS_solve_func = @RuntimeGeneratedFunction(solve_exp)
    # 𝓂.SS_solve_func = eval(solve_exp)

    return nothing
end


function reverse_diff_friendly_push!(x,y)
    @ignore_derivatives push!(x,y)
end


function block_solver(parameters_and_solved_vars::Vector{Float64}, 
                        n_block::Int, 
                        ss_solve_blocks::Function, 
                        # SS_optimizer, 
                        # f::OptimizationFunction, 
                        guess::Vector{Float64}, 
                        lbs::Vector{Float64}, 
                        ubs::Vector{Float64},
                        parameters::solver_parameters,
                        cold_start::Union{Bool,Float64},
                        verbose::Bool;
                        tol::AbstractFloat = eps(),
                        # timeout = 120,
                        starting_points::Vector{Float64} = [0.7688, 0.897, 1.2, .9, .75, 1.5, -.5, 2, .25]
                        # fail_fast_solvers_only = true,
                        # verbose::Bool = false
                        )
    
    if cold_start isa Float64
        sol_values = max.(lbs,min.(ubs, fill(cold_start,length(guess))))
        sol_minimum = 1
    elseif cold_start isa Bool 
        if cold_start
            sol_values = max.(lbs,min.(ubs, fill(0.0,length(guess))))
            sol_minimum  = sum(abs2,ss_solve_blocks(parameters_and_solved_vars,sol_values))
        else
            sol_values = guess
            sol_minimum  = sum(abs2,ss_solve_blocks(parameters_and_solved_vars,sol_values))
            iters = 0

            if verbose && sol_minimum < tol
                println("Block: ",n_block," - Solved using previous solution; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars, sol_values)))
            end
        end
    end

    # try modified LM to solve hard SS problems
    if (sol_minimum > tol)# | (maximum(abs,ss_solve_blocks(sol_values,parameters_and_solved_vars)) > tol))
        SS_optimizer = levenberg_marquardt

        previous_sol_init = max.(lbs,min.(ubs, sol_values))
        
        sol_new, info = SS_optimizer(
                                        x->ss_solve_blocks(parameters_and_solved_vars, x),
                                        previous_sol_init,
                                        lbs,
                                        ubs,
                                        parameters = parameters
                                    ) # alternatively use .001)#, μ = μ, p = p)# catch e end

        sol_minimum = isnan(sum(abs2,info[4])) ? Inf : sum(abs2,info[4])
        sol_values = max.(lbs,min.(ubs, sol_new ))
        iters = info[1]

        if sol_minimum < tol
            if verbose
                println("Block: ",n_block," - Solved using ",string(SS_optimizer)," and previous best non-converged solution; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars, sol_values)))
            end
        elseif cold_start isa Bool && !cold_start
            # if the previous non-converged best guess as a starting point does not work, try the standard starting points
            for starting_point in starting_points
                if sol_minimum > tol
                    standard_inits = max.(lbs,min.(ubs, fill(starting_point,length(guess))))
                    standard_inits[ubs .<= 1] .= .1 # capture cases where part of values is small

                    sol_new, info = SS_optimizer(
                        x->ss_solve_blocks(parameters_and_solved_vars, x),
                        standard_inits,
                        lbs,
                        ubs,
                        parameters = parameters
                        )# catch e end
                
                    sol_minimum = isnan(sum(abs2,info[4])) ? Inf : sum(abs2,info[4])
                    sol_values = max.(lbs,min.(ubs, sol_new))
                    iters = info[1]

                    if sol_minimum < tol && verbose
                        println("Block: ",n_block," - Solved using ",string(SS_optimizer)," and starting point: ",starting_point,"; maximum residual = ",maximum(abs,ss_solve_blocks(parameters_and_solved_vars, sol_values)))
                    end

                else 
                    break
                end
            end
        end
    end

    return sol_values, (sol_minimum, iters)
end

# needed for Julia 1.8
function block_solver(parameters_and_solved_vars::Vector{ℱ.Dual{Z,S,N}}, 
    n_block::Int, 
    ss_solve_blocks::Function, 
    # SS_optimizer, 
    # f::OptimizationFunction, 
    guess::Vector{Float64}, 
    lbs::Vector{Float64}, 
    ubs::Vector{Float64},
    parameters::solver_parameters,
    cold_start::Union{Bool,Float64},
    verbose::Bool ;
    tol::AbstractFloat = eps(),
    # timeout = 120,
    starting_points::Vector{Float64} = [0.7688, 0.897, 1.2, .9, .75, 1.5, -.5, 2, .25]
    # fail_fast_solvers_only = true,
    # verbose::Bool = false
    ) where {Z,S,N}

    # unpack: AoS -> SoA
    inp = ℱ.value.(parameters_and_solved_vars)

    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, parameters_and_solved_vars)'

    if verbose println("Solution for derivatives.") end
    # get f(vs)
    val, (min, iter) = block_solver(inp, 
                        n_block, 
                        ss_solve_blocks, 
                        # SS_optimizer, 
                        # f, 
                        guess, 
                        lbs, 
                        ubs,
                        parameters,
                        cold_start,
                        verbose;
                        tol = tol,
                        # timeout = timeout,
                        starting_points = starting_points)

    if min > tol
        jvp = fill(0,length(val),length(inp)) * ps
    else
        # get J(f, vs) * ps (cheating). Write your custom rule here
        B = 𝒜.jacobian(𝒷(), x -> ss_solve_blocks(x,val), inp)[1]
        A = 𝒜.jacobian(𝒷(), x -> ss_solve_blocks(inp,x), val)[1]
        # B = Zygote.jacobian(x -> ss_solve_blocks(x,transformer(val, option = 0),0), inp)[1]
        # A = Zygote.jacobian(x -> ss_solve_blocks(inp,transformer(x, option = 0),0), val)[1]

        Â = RF.lu(A, check = false)

        if !ℒ.issuccess(Â)
            Â = ℒ.svd(A)
        end
        
        jvp = -(Â \ B) * ps
    end

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(val)), (min, iter)
end



function second_order_stochastic_steady_state_iterative_solution_forward(𝐒₁𝐒₂::SparseVector{Float64};  dims::Vector{Tuple{Int,Int}},  𝓂::ℳ, tol::AbstractFloat = eps())
    len𝐒₁ = dims[1][1] * dims[1][2]

    𝐒₁ = reshape(𝐒₁𝐒₂[1 : len𝐒₁],dims[1])
    𝐒₂ = sparse(reshape(𝐒₁𝐒₂[len𝐒₁ + 1 : end],dims[2]))
        
    state = zeros(𝓂.timings.nVars)
    shock = zeros(𝓂.timings.nExo)

    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
    1
    shock]

    sol = @suppress begin
        speedmapping(state; 
                    m! = (SSS, sss) -> begin 
                                        aug_state .= [sss[𝓂.timings.past_not_future_and_mixed_idx]
                                                    1
                                                    shock]

                                        SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                    end, 
        tol = tol, maps_limit = 10000)
    end

    return sol.minimizer, sol.converged
end


function second_order_stochastic_steady_state_iterative_solution_conditions(𝐒₁𝐒₂::SparseVector, SSS, converged::Bool; dims::Vector{Tuple{Int,Int}}, 𝓂::ℳ, tol::AbstractFloat = eps())
    len𝐒₁ = dims[1][1] * dims[1][2]

    𝐒₁ = reshape(𝐒₁𝐒₂[1 : len𝐒₁],dims[1])
    𝐒₂ = sparse(reshape(𝐒₁𝐒₂[len𝐒₁ + 1 : end],dims[2]))

    shock = zeros(𝓂.timings.nExo)

    aug_state = [SSS[𝓂.timings.past_not_future_and_mixed_idx]
    1
    shock]

    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 - SSS
end


function second_order_stochastic_steady_state_iterative_solution_forward(𝐒₁𝐒₂::SparseVector{ℱ.Dual{Z,S,N}}; dims::Vector{Tuple{Int,Int}}, 𝓂::ℳ, tol::AbstractFloat = eps()) where {Z,S,N}
    S₁S₂, ps = separate_values_and_partials_from_sparsevec_dual(𝐒₁𝐒₂)

    # get f(vs)
    val, converged = second_order_stochastic_steady_state_iterative_solution_forward(S₁S₂; dims = dims, 𝓂 = 𝓂, tol = tol)

    if converged
        # get J(f, vs) * ps (cheating). Write your custom rule here
        B = 𝒜.jacobian(𝒷(), x -> second_order_stochastic_steady_state_iterative_solution_conditions(x, val, converged; dims = dims, 𝓂 = 𝓂, tol = tol), S₁S₂)[1]
        A = 𝒜.jacobian(𝒷(), x -> second_order_stochastic_steady_state_iterative_solution_conditions(S₁S₂, x, converged; dims = dims, 𝓂 = 𝓂, tol = tol), val)[1]

        Â = RF.lu(A, check = false)

        if !ℒ.issuccess(Â)
            Â = ℒ.svd(A)
        end
        
        jvp = -(Â \ B) * ps
    else
        jvp = fill(0,length(val),length(𝐒₁𝐒₂)) * ps
    end

    # lm = LinearMap{Float64}(x -> A * reshape(x, size(B)), length(B))

    # jvp = - sparse(reshape(ℐ.gmres(lm, sparsevec(B)), size(B))) * ps
    # jvp *= -ps

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val)), converged
end


second_order_stochastic_steady_state_iterative_solution = ℐ.ImplicitFunction(second_order_stochastic_steady_state_iterative_solution_forward,
                                                                                    second_order_stochastic_steady_state_iterative_solution_conditions; 
                                                                                    linear_solver = ℐ.DirectLinearSolver())


function calculate_second_order_stochastic_steady_state(parameters::Vector{M}, 𝓂::ℳ; verbose::Bool = false, pruning::Bool = false, tol::AbstractFloat = eps())::Tuple{Vector{M}, Bool, Vector{M}, Float64, AbstractMatrix{M}, SparseMatrixCSC{M}, AbstractMatrix{M}, SparseMatrixCSC{M}} where M
    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
    if solution_error > tol || isnan(solution_error)
        return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix
    
    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
    if !solved
        return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)
    
    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)

    if !solved2
        return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]

    if pruning
        aug_state₁ = sparse([zeros(𝓂.timings.nPast_not_future_and_mixed); 1; zeros(𝓂.timings.nExo)])

        tmp = (ℒ.I - 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed])

        tmp̄ = RF.lu(tmp, check = false)

        if !ℒ.issuccess(tmp̄)
            return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
        end

        SSSstates = tmp̄ \ (𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[𝓂.timings.past_not_future_and_mixed_idx]

        state = 𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] * SSSstates + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2
        converged = true
    else
        state, converged = second_order_stochastic_steady_state_iterative_solution([sparsevec(𝐒₁); vec(𝐒₂)]; dims = [size(𝐒₁); size(𝐒₂)], 𝓂 = 𝓂)
    end

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    # all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    # all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
    # NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
    
    # all_SS = [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
    # we need all variables for the stochastic steady state because even leads and lags have different SSS then the non-lead-lag ones (contrary to the no stochastic steady state) and we cannot recover them otherwise

    return all_SS + state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
end



function third_order_stochastic_steady_state_iterative_solution_forward(𝐒₁𝐒₂𝐒₃::SparseVector{Float64}; dims::Vector{Tuple{Int,Int}}, 𝓂::ℳ, tol::AbstractFloat = eps())
    len𝐒₁ = dims[1][1] * dims[1][2]
    len𝐒₂ = dims[2][1] * dims[2][2]

    𝐒₁ = reshape(𝐒₁𝐒₂𝐒₃[1 : len𝐒₁],dims[1])
    𝐒₂ = sparse(reshape(𝐒₁𝐒₂𝐒₃[len𝐒₁ .+ (1 : len𝐒₂)],dims[2]))
    𝐒₃ = sparse(reshape(𝐒₁𝐒₂𝐒₃[len𝐒₁ + len𝐒₂ + 1 : end],dims[3]))

    state = zeros(𝓂.timings.nVars)
    shock = zeros(𝓂.timings.nExo)

    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
    1
    shock]

    sol = @suppress begin
        speedmapping(state; 
                    m! = (SSS, sss) -> begin 
                                        aug_state .= [sss[𝓂.timings.past_not_future_and_mixed_idx]
                                                    1
                                                    shock]

                                        SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
                    end, 
        tol = tol, maps_limit = 10000)
    end

    return sol.minimizer, sol.converged
end


function third_order_stochastic_steady_state_iterative_solution_conditions(𝐒₁𝐒₂𝐒₃::SparseVector, SSS, converged::Bool; dims::Vector{Tuple{Int,Int}}, 𝓂::ℳ, tol::AbstractFloat = eps())
    len𝐒₁ = dims[1][1] * dims[1][2]
    len𝐒₂ = dims[2][1] * dims[2][2]

    𝐒₁ = reshape(𝐒₁𝐒₂𝐒₃[1 : len𝐒₁],dims[1])
    𝐒₂ = sparse(reshape(𝐒₁𝐒₂𝐒₃[len𝐒₁ .+ (1 : len𝐒₂)],dims[2]))
    𝐒₃ = sparse(reshape(𝐒₁𝐒₂𝐒₃[len𝐒₁ + len𝐒₂ + 1 : end],dims[3]))

    shock = zeros(𝓂.timings.nExo)

    aug_state = [SSS[𝓂.timings.past_not_future_and_mixed_idx]
    1
    shock]

    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6 - SSS
end

third_order_stochastic_steady_state_iterative_solution = ℐ.ImplicitFunction(third_order_stochastic_steady_state_iterative_solution_forward,
                                                                                third_order_stochastic_steady_state_iterative_solution_conditions; 
                                                                                linear_solver = ℐ.DirectLinearSolver())

function third_order_stochastic_steady_state_iterative_solution_forward(𝐒₁𝐒₂𝐒₃::SparseVector{ℱ.Dual{Z,S,N}}; dims::Vector{Tuple{Int,Int}}, 𝓂::ℳ, tol::AbstractFloat = eps()) where {Z,S,N}
    S₁S₂S₃, ps = separate_values_and_partials_from_sparsevec_dual(𝐒₁𝐒₂𝐒₃)

    # get f(vs)
    val, converged = third_order_stochastic_steady_state_iterative_solution_forward(S₁S₂S₃; dims = dims, 𝓂 = 𝓂, tol = tol)

    if converged
        # get J(f, vs) * ps (cheating). Write your custom rule here
        B = 𝒜.jacobian(𝒷(), x -> third_order_stochastic_steady_state_iterative_solution_conditions(x, val, converged; dims = dims, 𝓂 = 𝓂, tol = tol), S₁S₂S₃)[1]
        A = 𝒜.jacobian(𝒷(), x -> third_order_stochastic_steady_state_iterative_solution_conditions(S₁S₂S₃, x, converged; dims = dims, 𝓂 = 𝓂, tol = tol), val)[1]
        
        Â = RF.lu(A, check = false)
    
        if !ℒ.issuccess(Â)
            Â = ℒ.svd(A)
        end
        
        jvp = -(Â \ B) * ps
    else
        jvp = fill(0,length(val),length(𝐒₁𝐒₂𝐒₃)) * ps
    end

    # lm = LinearMap{Float64}(x -> A * reshape(x, size(B)), length(B))

    # jvp = - sparse(reshape(ℐ.gmres(lm, sparsevec(B)), size(B))) * ps
    # jvp *= -ps

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val)), converged
end


function calculate_third_order_stochastic_steady_state(parameters::Vector{M}, 𝓂::ℳ; verbose::Bool = false, pruning::Bool = false, tol::AbstractFloat = eps())::Tuple{Vector{M}, Bool, Vector{M}, Float64, AbstractMatrix{M}, SparseMatrixCSC{M}, SparseMatrixCSC{M}, AbstractMatrix{M}, SparseMatrixCSC{M}, SparseMatrixCSC{M}} where M
    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
    if solution_error > tol || isnan(solution_error)
        return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix
    
    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
    if !solved
        return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)
    
    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    if !solved2
        return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)
            
    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝓂.solution.perturbation.second_order_auxilliary_matrices, 𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    if !solved3
        return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]

    if pruning
        aug_state₁ = sparse([zeros(𝓂.timings.nPast_not_future_and_mixed); 1; zeros(𝓂.timings.nExo)])
        
        tmp = (ℒ.I - 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed])

        tmp̄ = RF.lu(tmp, check = false)

        if !ℒ.issuccess(tmp̄)
            return SS_and_pars, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
        end

        SSSstates = tmp̄ \ (𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[𝓂.timings.past_not_future_and_mixed_idx]

        state = 𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] * SSSstates + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2
        converged = true
    else
        state, converged = third_order_stochastic_steady_state_iterative_solution([sparsevec(𝐒₁); vec(𝐒₂); vec(𝐒₃)]; dims = [size(𝐒₁); size(𝐒₂); size(𝐒₃)], 𝓂 = 𝓂)
    end

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    # all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    # all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
    # NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
    
    # all_SS = [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
    # we need all variables for the stochastic steady state because even leads and lags have different SSS then the non-lead-lag ones (contrary to the no stochastic steady state) and we cannot recover them otherwise

    return all_SS + state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃
end




function solve!(𝓂::ℳ; 
    parameters::ParameterType = nothing, 
    dynamics::Bool = false, 
    algorithm::Symbol = :riccati, 
    obc::Bool = false,
    verbose::Bool = false,
    silent::Bool = false,
    tol::AbstractFloat = eps())

    @assert algorithm ∈ all_available_algorithms
    
    write_parameters_input!(𝓂, parameters, verbose = verbose)

    if 𝓂.model_hessian == Function[] && algorithm ∈ [:second_order, :pruned_second_order]
        start_time = time()
        write_functions_mapping!(𝓂, 2)
        if !silent println("Take symbolic derivatives up to second order:\t",round(time() - start_time, digits = 3), " seconds") end
    elseif 𝓂.model_third_order_derivatives == Function[] && algorithm ∈ [:third_order, :pruned_third_order]
        start_time = time()
        write_functions_mapping!(𝓂, 3)
        if !silent println("Take symbolic derivatives up to third order:\t",round(time() - start_time, digits = 3), " seconds") end
    end

    if dynamics
        obc_not_solved = isnothing(𝓂.solution.perturbation.first_order.state_update_obc)
        if  ((:riccati             == algorithm) && ((:riccati             ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:first_order         == algorithm) && ((:first_order         ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:second_order        == algorithm) && ((:second_order        ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:pruned_second_order == algorithm) && ((:pruned_second_order ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:third_order         == algorithm) && ((:third_order         ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:pruned_third_order  == algorithm) && ((:pruned_third_order  ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))

            SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

            if solution_error > tol
                @warn "Could not find non stochastic steady steady."
            end

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix
            
            S₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
            
            @assert solved "Could not find stable first order solution."

            state_update₁ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            shock]
                return S₁ * aug_state # you need a return statement for forwarddiff to work
            end
            
            if obc
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1, verbose = false)

                ∇̂₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix
            
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
            
            if !converged @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

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

            if !converged @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

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

            if !converged @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

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

            if !converged @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

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
            
            SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

            if solution_error > tol
                @warn "Could not find non stochastic steady steady."
            end

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix
            
            S₁, converged = calculate_quadratic_iteration_solution(∇₁; T = 𝓂.timings)
            
            state_update₁ₜ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            shock]
                return S₁ * aug_state # you need a return statement for forwarddiff to work
            end
            
            if obc
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1, verbose = false)

                ∇̂₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix
            
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

            SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

            if solution_error > tol
                @warn "Could not find non stochastic steady steady."
            end

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix
            
            S₁ = calculate_linear_time_iteration_solution(∇₁; T = 𝓂.timings)
            
            state_update₁ₜ = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            shock]
                return S₁ * aug_state # you need a return statement for forwarddiff to work
            end
            
            if obc
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1)

                ∇̂₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix
            
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
    
    idxs = []
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
    
    idxs = []
    for k in 1:nₑ₋
        for j in 1:nₑ₋
            for i in 1:nₑ₋
                sorted_ids = sort([k,j,i])
                push!(idxs, (sorted_ids[3] - 1) * nₑ₋ ^ 2 + (sorted_ids[2] - 1) * nₑ₋ + sorted_ids[1])
            end
        end
    end
    
    𝐔₃ = 𝐂₃' * sparse(idxs,1:nₑ₋ ^ 3, 1)
    
    # permutation matrices
    M = reshape(1:nₑ₋^3,1,nₑ₋,nₑ₋,nₑ₋)
    𝐏 = @views sparse(reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 4, 2, 3])],nₑ₋^3,nₑ₋^3)
                        + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 4, 3])],nₑ₋^3,nₑ₋^3)
                        + reshape(spdiagm(ones(nₑ₋^3))[:,PermutedDimsArray(M,[1, 2, 3, 4])],nₑ₋^3,nₑ₋^3))

    𝐏₁ₗ = sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(2,1,3))),:])
    𝐏₁ᵣ = sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(2,1,3)))])

    𝐏₁ₗ̂  = @views sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(1,3,2))),:])
    𝐏₂ₗ̂  = @views sparse(spdiagm(ones(n̄^3))[vec(permutedims(reshape(1:n̄^3,n̄,n̄,n̄),(3,1,2))),:])

    𝐏₁ₗ̄ = @views sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2))),:])
    𝐏₂ₗ̄ = @views sparse(spdiagm(ones(nₑ₋^3))[vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2))),:])


    𝐏₁ᵣ̃ = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(1,3,2)))])
    𝐏₂ᵣ̃ = @views sparse(spdiagm(ones(nₑ₋^3))[:,vec(permutedims(reshape(1:nₑ₋^3,nₑ₋,nₑ₋,nₑ₋),(3,1,2)))])

    ∇₃_col_indices_extended = findnz(sparse(ones(Int,length(∇₃_col_indices)),∇₃_col_indices,ones(Int,length(∇₃_col_indices)),1,size(𝐔∇₃,1)) * 𝐔∇₃)[2]

    nonnull_columns = Set()
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

    return third_order_auxilliary_matrices(𝐂₃, 𝐔₃, 𝐔∇₃, 𝐏, 𝐏₁ₗ, 𝐏₁ᵣ, 𝐏₁ₗ̂, 𝐏₂ₗ̂, 𝐏₁ₗ̄, 𝐏₂ₗ̄, 𝐏₁ᵣ̃, 𝐏₂ᵣ̃, 𝐒𝐏)
end

function write_functions_mapping!(𝓂::ℳ, max_perturbation_order::Int)
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

    steady_state = []
    for (i, var) in enumerate(ss_varss)
        push!(steady_state,:($var = X̄[$i]))
        # ii += 1
    end

    ii = 1

    alll = []
    for var in future_varss
        push!(alll,:($var = X[$ii]))
        ii += 1
    end

    for var in present_varss
        push!(alll,:($var = X[$ii]))
        ii += 1
    end

    for var in past_varss
        push!(alll,:($var = X[$ii]))
        ii += 1
    end

    for var in shock_varss
        push!(alll,:($var = X[$ii]))
        ii += 1
    end


    # paras = []
    # push!(paras,:((;$(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)...)) = params))

    paras = []
    for (i, parss) in enumerate(vcat(𝓂.parameters,𝓂.calibration_equations_parameters))
        push!(paras,:($parss = params[$i]))
    end

    # # watch out with naming of parameters in model and functions
    # mod_func2 = :(function model_function_uni_redux(X::Vector, params::Vector{Number}, X̄::Vector)
    #     $(alll...)
    #     $(paras...)
	# 	$(𝓂.calibration_equations_no_var...)
    #     $(steady_state...)
    #     [$(𝓂.dyn_equations...)]
    # end)


    # 𝓂.model_function = @RuntimeGeneratedFunction(mod_func2)
    # 𝓂.model_function = eval(mod_func2)

    dyn_future_list = collect(reduce(union, 𝓂.dyn_future_list))
    dyn_present_list = collect(reduce(union, 𝓂.dyn_present_list))
    dyn_past_list = collect(reduce(union, 𝓂.dyn_past_list))
    dyn_exo_list = collect(reduce(union,𝓂.dyn_exo_list))
    
    future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
    present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
    past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
    exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
    
    vars_raw = [dyn_future_list[indexin(sort(future),future)]...,
            dyn_present_list[indexin(sort(present),present)]...,
            dyn_past_list[indexin(sort(past),past)]...,
            dyn_exo_list[indexin(sort(exo),exo)]...]

    # overwrite SymPyCall names
    eval(:(Symbolics.@variables $(reduce(union,get_symbols.(𝓂.dyn_equations))...)))

    vars = eval(:(Symbolics.@variables $(vars_raw...)))

    eqs = Symbolics.parse_expr_to_symbolic.(𝓂.dyn_equations,(@__MODULE__,))

    if max_perturbation_order >= 2 
        nk = length(vars_raw)
        second_order_idxs = [nk * (i-1) + k for i in 1:nk for k in 1:i]
        if max_perturbation_order == 3
            third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]
        end
    end

    first_order = []
    second_order = []
    third_order = []
    row1 = Int[]
    row2 = Int[]
    row3 = Int[]
    column1 = Int[]
    column2 = Int[]
    column3 = Int[]
    # column3ext = Int[]
    i1 = 1
    i2 = 1
    i3 = 1
    
    for (c1,var1) in enumerate(vars)
        for (r,eq) in enumerate(eqs)
            if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
                deriv_first = Symbolics.derivative(eq,var1)
                # if deriv_first != 0 
                #     deriv_expr = Meta.parse(string(deriv_first.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
                #     push!(first_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                    push!(first_order, Symbolics.toexpr(deriv_first))
                    push!(row1,r)
                    push!(column1,c1)
                    i1 += 1
                    if max_perturbation_order >= 2 
                        for (c2,var2) in enumerate(vars)
                            # if Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first))
                            if (((c1 - 1) * length(vars) + c2) ∈ second_order_idxs) && (Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first)))
                                deriv_second = Symbolics.derivative(deriv_first,var2)
                                # if deriv_second != 0 
                                #     deriv_expr = Meta.parse(string(deriv_second.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
                                #     push!(second_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                                    push!(second_order,Symbolics.toexpr(deriv_second))
                                    push!(row2,r)
                                    # push!(column2,(c1 - 1) * length(vars) + c2)
                                    push!(column2, Int.(indexin([(c1 - 1) * length(vars) + c2], second_order_idxs))...)
                                    i2 += 1
                                    if max_perturbation_order == 3
                                        for (c3,var3) in enumerate(vars)
                                            # if Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second))
                                                # push!(column3ext,(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3)
                                                if (((c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3) ∈ third_order_idxs) && (Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second)))
                                                    deriv_third = Symbolics.derivative(deriv_second,var3)
                                                    # if deriv_third != 0 
                                                    #     deriv_expr = Meta.parse(string(deriv_third.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
                                                    #     push!(third_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                                                        push!(third_order,Symbolics.toexpr(deriv_third))
                                                        push!(row3,r)
                                                        # push!(column3,(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3)
                                                        push!(column3, Int.(indexin([(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3], third_order_idxs))...)
                                                        i3 += 1
                                                    # end
                                                end
                                            # end
                                        end
                                    end
                                # end
                            end
                        end
                    end
                # end
            end
        end
    end


    mod_func3 = :(function model_jacobian(X::Vector, params::Vector{Real}, X̄::Vector)
        $(alll...)
        $(paras...)
        $(𝓂.calibration_equations_no_var...)
        $(steady_state...)
        sparse([$(row1...)], [$(column1...)], [$(first_order...)], $(length(eqs)), $(length(vars)))
    end)

    𝓂.model_jacobian = @RuntimeGeneratedFunction(mod_func3)
    # 𝓂.model_jacobian = FWrap{Tuple{Vector{Float64}, Vector{Number}, Vector{Float64}}, SparseMatrixCSC{Float64}}(@RuntimeGeneratedFunction(mod_func3))

    # 𝓂.model_jacobian = eval(mod_func3)


    if max_perturbation_order >= 2 && 𝓂.model_hessian == Function[]
        # if length(row2) == 0 
        #     out = :(spzeros($(length(eqs)), $(length(second_order_idxs))))
        # else 
        #     out = :(sparse([$(row2...)], [$(column2...)], [$(second_order...)], $(length(eqs)), $(length(second_order_idxs))))
        # end

        # mod_func4 = :(function model_hessian(X::Vector, params::Vector{Real}, X̄::Vector)
        #     $(alll...)
        #     $(paras...)
        #     $(𝓂.calibration_equations_no_var...)
        #     $(steady_state...)
        #     $out
        # end)

        for (l,second) in enumerate(second_order)
            exx = :(function(X::Vector, params::Vector{Real}, X̄::Vector)
            $(alll...)
            $(paras...)
            $(𝓂.calibration_equations_no_var...)
            $(steady_state...)
            return $second, $(row2[l]), $(column2[l])
            end)
            push!(𝓂.model_hessian,@RuntimeGeneratedFunction(exx))
        end

        𝓂.solution.perturbation.second_order_auxilliary_matrices = create_second_order_auxilliary_matrices(𝓂.timings)


        # 𝓂.model_hessian = @RuntimeGeneratedFunction(mod_func4)
        # 𝓂.model_hessian = eval(mod_func4)
    end

    if max_perturbation_order == 3 && 𝓂.model_third_order_derivatives == Function[]
        # if length(row3) == 0 
        #     out = :(spzeros($(length(eqs)), $(length(third_order_idxs))))
        # else 
        #     out = :(sparse([$(row3...)], [$(column3...)], [$(third_order...)], $(length(eqs)), $(length(third_order_idxs))))
        # end

        # mod_func5 = :(function model_hessian(X::Vector, params::Vector{Real}, X̄::Vector)
        #     $(alll...)
        #     $(paras...)
        #     $(𝓂.calibration_equations_no_var...)
        #     $(steady_state...)
        #     $out
        # end)

        for (l,third) in enumerate(third_order)
            exx = :(function(X::Vector, params::Vector{Real}, X̄::Vector)
            $(alll...)
            $(paras...)
            $(𝓂.calibration_equations_no_var...)
            $(steady_state...)
            return $third, $(row3[l]), $(column3[l])
            end)
            push!(𝓂.model_third_order_derivatives,@RuntimeGeneratedFunction(exx))
        end

        𝓂.solution.perturbation.third_order_auxilliary_matrices = create_third_order_auxilliary_matrices(𝓂.timings, unique(column3))

    end


    # write indices in auxiliary objects
    dyn_var_future_list  = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₁₎")))
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎")))
    dyn_var_past_list    = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₋₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₋₁₎")))
    dyn_exo_list         = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₓ₎")))
    dyn_ss_list          = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₛₛ₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₛₛ₎")))

    dyn_var_future  = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_var_future_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    dyn_var_present = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_var_present_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    dyn_var_past    = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_var_past_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    dyn_exo         = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_exo_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    dyn_ss          = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_ss_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

    SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)


    dyn_var_future_idx  = indexin(dyn_var_future    , SS_and_pars_names)
    dyn_var_present_idx = indexin(dyn_var_present   , SS_and_pars_names)
    dyn_var_past_idx    = indexin(dyn_var_past      , SS_and_pars_names)
    dyn_ss_idx          = indexin(dyn_ss            , SS_and_pars_names)

    shocks_ss = zeros(length(dyn_exo))

    𝓂.solution.perturbation.auxilliary_indices = auxilliary_indices(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx, dyn_ss_idx, shocks_ss)


    # 𝓂.model_third_order_derivatives = @RuntimeGeneratedFunction(mod_func5)
    # 𝓂.model_third_order_derivatives = eval(mod_func5)


    # calib_eqs = []
    # for (i, eqs) in enumerate(𝓂.solved_vals) 
    #     varss = 𝓂.solved_vars[i]
    #     push!(calib_eqs,:($varss = $eqs))
    # end

    # for varss in 𝓂.exo
    #     push!(calib_eqs,:($varss = 0))
    # end

    # calib_pars = []
    # for (i, parss) in enumerate(𝓂.parameters)
    #     push!(calib_pars,:($parss = parameters[$i]))
    # end

    # var_out = []
    # ii =  1
    # for var in 𝓂.var
    #     push!(var_out,:($var = SS[$ii]))
    #     ii += 1
    # end

    # par_out = []
    # for cal in 𝓂.calibration_equations_parameters
    #     push!(par_out,:($cal = SS[$ii]))
    #     ii += 1
    # end

    # calib_pars = []
    # for (i, parss) in enumerate(𝓂.parameters)
    #     push!(calib_pars,:($parss = parameters[$i]))
    # end

    # test_func = :(function test_SS(parameters::Vector{Float64}, SS::Vector{Float64})
    #     $(calib_pars...) 
    #     $(var_out...)
    #     $(par_out...)
    #     [$(𝓂.ss_equations...),$(𝓂.calibration_equations...)]
    # end)

    # 𝓂.solution.valid_steady_state_solution = @RuntimeGeneratedFunction(test_func)

    # 𝓂.solution.outdated_algorithms = Set([:linear_time_iteration, :riccati, :quadratic_iteration, :first_order, :second_order, :third_order])
    return nothing
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

    for i in 1:length(parameters)
        bnd_idx = findfirst(x->x==collect(keys(parameters))[i],𝓂.bounded_vars)
        if !isnothing(bnd_idx)
            if collect(values(parameters))[i] > 𝓂.upper_bounds[bnd_idx]
                # println("Calibration is out of bounds for ",collect(keys(parameters))[i],":\t",collect(values(parameters))[i]," > ",𝓂.upper_bounds[bnd_idx] + eps())
                println("Bounds error for ",collect(keys(parameters))[i]," < ",𝓂.upper_bounds[bnd_idx] + eps(),"\tparameter value: ",collect(values(parameters))[i])
                bounds_broken = true
                continue
            end
            if collect(values(parameters))[i] < 𝓂.lower_bounds[bnd_idx]
                # println("Calibration is out of bounds for ",collect(keys(parameters))[i],":\t",collect(values(parameters))[i]," < ",𝓂.lower_bounds[bnd_idx] - eps())
                println("Bounds error for ",collect(keys(parameters))[i]," > ",𝓂.lower_bounds[bnd_idx] + eps(),"\tparameter value: ",collect(values(parameters))[i])
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

    for i in 1:length(parameters)
        bnd_idx = findfirst(x -> x == 𝓂.parameters[i], 𝓂.bounded_vars)
        if !isnothing(bnd_idx)
            if collect(values(parameters))[i] > 𝓂.upper_bounds[bnd_idx]
                println("Bounds error for ",𝓂.parameters[i]," < ",𝓂.upper_bounds[bnd_idx] + eps(),"\tparameter value: ",𝓂.parameter_values[i])
                bounds_broken = true
                continue
            end
            if collect(values(parameters))[i] < 𝓂.lower_bounds[bnd_idx]
                println("Bounds error for ",𝓂.parameters[i]," > ",𝓂.lower_bounds[bnd_idx] + eps(),"\tparameter value: ",𝓂.parameter_values[i])
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


# helper for get functions
function SSS_third_order_parameter_derivatives(parameters::Vector{ℱ.Dual{Z,S,N}}, parameters_idx, 𝓂::ℳ; verbose::Bool = false, pruning::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    SSS = calculate_third_order_stochastic_steady_state(params, 𝓂, verbose = verbose, pruning = pruning)

    if !SSS[2] @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

    return SSS
end


# helper for get functions
function SSS_third_order_parameter_derivatives(parameters::ℱ.Dual{Z,S,N}, parameters_idx::Int, 𝓂::ℳ; verbose::Bool = false, pruning::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    SSS = calculate_third_order_stochastic_steady_state(params, 𝓂, verbose = verbose, pruning = pruning)

    if !SSS[2] @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

    return SSS
end


# helper for get functions
function SSS_second_order_parameter_derivatives(parameters::Vector{ℱ.Dual{Z,S,N}}, parameters_idx, 𝓂::ℳ; verbose::Bool = false, pruning::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    SSS = calculate_second_order_stochastic_steady_state(params, 𝓂, verbose = verbose, pruning = pruning)

    if !SSS[2] @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

    return SSS
end


# helper for get functions
function SSS_second_order_parameter_derivatives(parameters::ℱ.Dual{Z,S,N}, parameters_idx::Int, 𝓂::ℳ; verbose::Bool = false, pruning::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    SSS = calculate_second_order_stochastic_steady_state(params, 𝓂, verbose = verbose, pruning = pruning)

    if !SSS[2] @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

    return SSS
end


# helper for get functions
function SS_parameter_derivatives(parameters::Vector{ℱ.Dual{Z,S,N}}, parameters_idx, 𝓂::ℳ; verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    𝓂.SS_solve_func(params, 𝓂, verbose, false, 𝓂.solver_parameters)
end


# helper for get functions
function SS_parameter_derivatives(parameters::ℱ.Dual{Z,S,N}, parameters_idx::Int, 𝓂::ℳ; verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    𝓂.SS_solve_func(params, 𝓂, verbose, false, 𝓂.solver_parameters)
end


# helper for get functions
function covariance_parameter_derivatives(parameters::Vector{ℱ.Dual{Z,S,N}}, parameters_idx, 𝓂::ℳ; verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    convert(Vector{ℱ.Dual{Z,S,N}},max.(ℒ.diag(calculate_covariance(params, 𝓂, verbose = verbose)[1]),eps(Float64)))
end


# helper for get functions
function covariance_parameter_derivatives(parameters::ℱ.Dual{Z,S,N}, parameters_idx::Int, 𝓂::ℳ; verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    convert(Vector{ℱ.Dual{Z,S,N}},max.(ℒ.diag(calculate_covariance(params, 𝓂, verbose = verbose)[1]),eps(Float64)))
end




# helper for get functions
function covariance_parameter_derivatives_second_order(parameters::Vector{ℱ.Dual{Z,S,N}}, parameters_idx, 𝓂::ℳ; verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    convert(Vector{ℱ.Dual{Z,S,N}},max.(ℒ.diag(calculate_second_order_moments(params, 𝓂, verbose = verbose)[1]),eps(Float64)))
end


# helper for get functions
function covariance_parameter_derivatives_second_order(parameters::ℱ.Dual{Z,S,N}, parameters_idx::Int, 𝓂::ℳ; verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    convert(Vector{ℱ.Dual{Z,S,N}},max.(ℒ.diag(calculate_second_order_moments(params, 𝓂, verbose = verbose)[1]),eps(Float64)))
end


# helper for get functions
function covariance_parameter_derivatives_third_order(parameters::Vector{ℱ.Dual{Z,S,N}}, 
                                                        variables::Union{Symbol_input,String_input}, 
                                                        parameters_idx, 
                                                        𝓂::ℳ;
                                                        dependencies_tol::AbstractFloat = 1e-12,
                                                        verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    convert(Vector{ℱ.Dual{Z,S,N}},max.(ℒ.diag(calculate_third_order_moments(params, variables, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)[1]),eps(Float64)))
end


# helper for get functions
function covariance_parameter_derivatives_third_order(parameters::ℱ.Dual{Z,S,N}, 
                                                        variables::Union{Symbol_input,String_input}, 
                                                        parameters_idx::Int, 
                                                        𝓂::ℳ; 
                                                        dependencies_tol::AbstractFloat = 1e-12,
                                                        verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    convert(Vector{ℱ.Dual{Z,S,N}},max.(ℒ.diag(calculate_third_order_moments(params, variables, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)[1]),eps(Float64)))
end


# helper for get functions
function mean_parameter_derivatives(parameters::Vector{ℱ.Dual{Z,S,N}}, parameters_idx, 𝓂::ℳ; algorithm::Symbol = :pruned_second_order, verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    convert(Vector{ℱ.Dual{Z,S,N}}, calculate_mean(params, 𝓂, algorithm = algorithm, verbose = verbose)[1])
end


# helper for get functions
function mean_parameter_derivatives(parameters::ℱ.Dual{Z,S,N}, parameters_idx::Int, 𝓂::ℳ; algorithm::Symbol = :pruned_second_order, verbose::Bool = false) where {Z,S,N}
    params = copy(𝓂.parameter_values)
    params = convert(Vector{ℱ.Dual{Z,S,N}},params)
    params[parameters_idx] = parameters
    convert(Vector{ℱ.Dual{Z,S,N}}, calculate_mean(params, 𝓂, algorithm = algorithm, verbose = verbose)[1])
end



function calculate_jacobian(parameters::Vector{M}, SS_and_pars::AbstractArray{N}, 𝓂::ℳ) where {M,N}
    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
    calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
    # par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))
    par = vcat(parameters,calibrated_parameters)
    
    dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
    dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
    dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
    dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

    shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

    # return 𝒜.jacobian(𝒷(), x -> 𝓂.model_function(x, par, SS), [SS_future; SS_present; SS_past; shocks_ss])#, SS_and_pars
    # return Matrix(𝓂.model_jacobian(([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx])))
    return 𝓂.model_jacobian([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx])
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
    # return 𝓂.model_hessian([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx])

    second_out =  [f([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx]) for f in 𝓂.model_hessian]
    
    vals = [i[1] for i in second_out]
    rows = [i[2] for i in second_out]
    cols = [i[3] for i in second_out]

    vals = convert(Vector{M}, vals)

    # nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)
    # sparse(rows, cols, vals, length(𝓂.dyn_equations), nk^2)
    sparse(rows, cols, vals, length(𝓂.dyn_equations), size(𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂,1)) * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂
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
    # return 𝓂.model_third_order_derivatives([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx])
    
    
    third_out =  [f([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx]) for f in 𝓂.model_third_order_derivatives]
    
    vals = [i[1] for i in third_out]
    rows = [i[2] for i in third_out]
    cols = [i[3] for i in third_out]

    vals = convert(Vector{M}, vals)

    # nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)
    # sparse(rows, cols, vals, length(𝓂.dyn_equations), nk^3)
    sparse(rows, cols, vals, length(𝓂.dyn_equations), size(𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃,1)) * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃
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

    sol = @suppress begin
        speedmapping(zero(A); m! = (C̄, C) -> C̄ .=  A + B * C^2, tol = tol, maps_limit = 10000)
    end

    C = -sol.minimizer

    D = -(∇₊ * C + ∇₀) \ ∇ₑ

    @views hcat(C[:,T.past_not_future_and_mixed_idx],D), sol.converged
end



function riccati_forward(∇₁::Matrix{Float64}; T::timings, explosive::Bool = false)::Tuple{Matrix{Float64},Bool}
    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    Q    = ℒ.qr(collect(∇₀[:,T.present_only_idx]))
    Qinv = Q.Q'

    A₊ = Qinv * ∇₊
    A₀ = Qinv * ∇₀
    A₋ = Qinv * ∇₋

    dynIndex = T.nPresent_only+1:T.nVars

    Ã₊  = @view A₊[dynIndex,:]
    Ã₋  = @view A₋[dynIndex,:]
    Ã₀₊ = @view A₀[dynIndex, T.future_not_past_and_mixed_idx]
    Ã₀₋ = @views A₀[dynIndex, T.past_not_future_idx] * ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.not_mixed_in_past_idx,:]
    
    Z₊ = zeros(T.nMixed,T.nFuture_not_past_and_mixed)
    I₊ = @view ℒ.diagm(ones(T.nFuture_not_past_and_mixed))[T.mixed_in_future_idx,:]

    Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
    I₋ = @view ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

    D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
    E = vcat(hcat(-Ã₋,-Ã₀₊), hcat(Z₋, I₊))
    # this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
    schdcmp = try
        ℒ.schur(D, E)
    catch
        return zeros(T.nVars,T.nPast_not_future_and_mixed), false
    end
    
    if explosive # returns false for NaN gen. eigenvalue which is correct here bc they are > 1
        eigenselect = abs.(schdcmp.β ./ schdcmp.α) .>= 1

        ℒ.ordschur!(schdcmp, eigenselect)

        Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
        Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
        T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

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

        Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
        Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
        T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]


        Ẑ₁₁ = RF.lu(Z₁₁, check = false)

        if !ℒ.issuccess(Ẑ₁₁)
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    end
    
    Ŝ₁₁ = RF.lu(S₁₁, check = false)

    if !ℒ.issuccess(Ŝ₁₁)
        return zeros(T.nVars,T.nPast_not_future_and_mixed), false
    end
    
    D      = Z₂₁ / Ẑ₁₁
    L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁

    sol = @views vcat(L[T.not_mixed_in_past_idx,:], D)

    Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = @view A₋[1:T.nPresent_only,:]

    Ā̂₀ᵤ = RF.lu(Ā₀ᵤ, check = false)

    if !ℒ.issuccess(Ā̂₀ᵤ)
        Ā̂₀ᵤ = ℒ.svd(collect(Ā₀ᵤ))
    end

    A    = @views vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    
    return @view(A[T.reorder,:]), true
end

function riccati_conditions(∇₁::AbstractMatrix{M}, sol_d::AbstractMatrix{N}, solved::Bool; T::timings, explosive::Bool = false) where {M,N}
    expand = @ignore_derivatives [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    A = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    B = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    C = ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]

    sol_buf = sol_d * expand[2]

    sol_buf2 = sol_buf * sol_buf

    err1 = A * sol_buf2 + B * sol_buf + C

    err1[:,T.past_not_future_and_mixed_idx]
end


function riccati_forward(∇₁::Matrix{ℱ.Dual{Z,S,N}}; T::timings, explosive::Bool = false) where {Z,S,N}
    # unpack: AoS -> SoA
    ∇̂₁ = ℱ.value.(∇₁)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, ∇₁)'

    val, solved = riccati_forward(∇̂₁;T = T, explosive = explosive)

    if solved
        # get J(f, vs) * ps (cheating). Write your custom rule here
        B = 𝒜.jacobian(𝒷(), x -> riccati_conditions(x, val, solved; T = T), ∇̂₁)[1]
        A = 𝒜.jacobian(𝒷(), x -> riccati_conditions(∇̂₁, x, solved; T = T), val)[1]


        Â = RF.lu(A, check = false)

        if !ℒ.issuccess(Â)
            Â = ℒ.svd(A)
        end
        
        jvp = -(Â \ B) * ps
    else
        jvp = fill(0,length(val),length(∇̂₁)) * ps
    end

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val)), solved
end


riccati_AD_direct = ℐ.ImplicitFunction(riccati_forward, 
                                    riccati_conditions; 
                                    linear_solver = ℐ.DirectLinearSolver())

riccati_AD = ℐ.ImplicitFunction(riccati_forward, riccati_conditions) # doesnt converge!?


function calculate_first_order_solution(∇₁::Matrix{S}; T::timings, explosive::Bool = false)::Tuple{Matrix{S},Bool} where S <: Real
    A, solved = riccati_AD_direct(∇₁; T = T, explosive = explosive)

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
                                            tol::AbstractFloat = eps())
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

    # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = - ∇₂ * sparse(ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 
    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = -(mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + mat_mult_kron(∇₂, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 

    X = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
    droptol!(X,tol)

    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    B = spinv * ∇₁₊
    droptol!(B,tol)

    C = (M₂.𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + M₂.𝐔₂ * M₂.𝛔) * M₂.𝐂₂
    droptol!(C,tol)

    r1,c1,v1 = findnz(B)
    r2,c2,v2 = findnz(C)
    r3,c3,v3 = findnz(X)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    push!(coordinates,(r2,c2))
    push!(coordinates,(r3,c3))
    
    values = vcat(v1, v2, v3)

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(B))
    push!(dimensions,size(C))
    push!(dimensions,size(X))

    solver = length(X.nzval) / length(X) < .1 ? :sylvester : :gmres

    𝐒₂, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = solver, sparse_output = true)

    if !solved
        return 𝐒₂, solved
    end

    𝐒₂ *= M₂.𝐔₂

    return 𝐒₂, solved
end



function calculate_third_order_solution(∇₁::AbstractMatrix{<: Real}, #first order derivatives
                                            ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
                                            ∇₃::SparseMatrixCSC{<: Real}, #third order derivatives
                                            𝑺₁::AbstractMatrix{<: Real}, #first order solution
                                            𝐒₂::SparseMatrixCSC{<: Real}, #second order solution
                                            M₂::second_order_auxilliary_matrices,  # aux matrices second order
                                            M₃::third_order_auxilliary_matrices;  # aux matrices third order
                                            T::timings,
                                            tol::AbstractFloat = eps())
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

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    # 𝐗₃ = -∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux)
    𝐗₃ = -A_mult_kron_power_3_B(∇₃, aux)

    tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)
    out = - ∇₃ * tmpkron - ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron * M₃.𝐏₁ᵣ̃ - ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron * M₃.𝐏₂ᵣ̃
    𝐗₃ += out
    
    # tmp𝐗₃ = -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
    tmp𝐗₃ = -mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)

    tmpkron1 = -∇₂ *  ℒ.kron(𝐒₁₊╱𝟎,𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔,𝐒₁₋╱𝟏ₑ)
    out2 = tmpkron1 * tmpkron2 +  tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
    
    𝐗₃ += (tmp𝐗₃ + out2 + -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * M₂.𝛔)) * M₃.𝐏# |> findnz
    
    𝐗₃ += @views -∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]) * M₃.𝐏
    droptol!(𝐗₃,tol)
    
    X = spinv * 𝐗₃ * M₃.𝐂₃
    droptol!(X,tol)
    
    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
    
    C = M₃.𝐔₃ * tmpkron + M₃.𝐔₃ * M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐔₃ * M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃
    C += M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) # no speed up here from A_mult_kron_power_3_B
    C *= M₃.𝐂₃
    droptol!(C,tol)

    r1,c1,v1 = findnz(B)
    r2,c2,v2 = findnz(C)
    r3,c3,v3 = findnz(X)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    push!(coordinates,(r2,c2))
    push!(coordinates,(r3,c3))
    
    values = vcat(v1, v2, v3)

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(B))
    push!(dimensions,size(C))
    push!(dimensions,size(X))

    𝐒₃, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres, sparse_output = true)

    if !solved
        return 𝐒₃, solved
    end

    𝐒₃ *= M₃.𝐔₃

    return 𝐒₃, solved
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

        axis2 = shocks isa Union{Symbol_input,String_input} ? [T.exo[shock_idx]...] : [:Shock_matrix]
        
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

        axis2 = shocks isa Union{Symbol_input,String_input} ? [T.exo[shock_idx]...] : [:Shock_matrix]
        
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

    axis2 = shocks isa Union{Symbol_input,String_input} ? [T.exo[shock_idx]...] : [:Shock_matrix]
        
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


function parse_shocks_input_to_index(shocks::Union{Symbol_input,String_input}, T::timings)
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
            return @warn "Following shocks are not part of the model: " * join(string.(setdiff(shocks,T.exo)),", ")
        end
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(shocks .== T.exo,dims= 2))))
    elseif shocks isa Vector{Symbol}
        if length(setdiff(shocks,T.exo)) > 0
            return @warn "Following shocks are not part of the model: " * join(string.(setdiff(shocks,T.exo)),", ")
        end
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(shocks,1,length(shocks)) .== T.exo, dims= 2))))
    elseif shocks isa Tuple{Symbol, Vararg{Symbol}}
        if length(setdiff(shocks,T.exo)) > 0
            return @warn "Following shocks are not part of the model: " * join(string.(setdiff(Symbol.(collect(shocks)),T.exo)),", ")
        end
        shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(collect(shocks),1,length(shocks)) .== T.exo,dims= 2))))
    elseif shocks isa Symbol
        if length(setdiff([shocks],T.exo)) > 0
            return @warn "Following shock is not part of the model: " * join(string(setdiff([shocks],T.exo)[1]),", ")
        end
        shock_idx = getindex(1:T.nExo,shocks .== T.exo)
    else
        return @warn "Invalid argument in shocks"
    end
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



function calculate_covariance(parameters::Vector{<: Real}, 𝓂::ℳ; verbose::Bool = false)
    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters, 0)
    
	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) 

    sol, solved = calculate_first_order_solution(Matrix(∇₁); T = 𝓂.timings)

    # covar_raw, solved_cov = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = collect(1:𝓂.timings.nVars))

    A = @views sol[:, 1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    
    C = @views sol[:, 𝓂.timings.nPast_not_future_and_mixed+1:end]
    
    CC = C * C'

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    
    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(A))
    push!(dimensions,size(CC))
    
    values = vcat(vec(A), vec(collect(-CC)))

    covar_raw, _ = solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)

    return covar_raw, sol , ∇₁, SS_and_pars
end




function calculate_mean(parameters::Vector{T}, 𝓂::ℳ; verbose::Bool = false, algorithm = :pruned_second_order, tol::Float64 = eps()) where T <: Real
    # Theoretical mean identical for 2nd and 3rd order pruned solution.
    @assert algorithm ∈ [:linear_time_iteration, :riccati, :first_order, :quadratic_iteration, :binder_pesaran, :pruned_second_order, :pruned_third_order] "Theoretical mean only available for first order, pruned second and third order perturbation solutions."

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
    if algorithm ∈ [:linear_time_iteration, :riccati, :first_order, :quadratic_iteration, :binder_pesaran]
        return SS_and_pars[1:𝓂.timings.nVars], solution_error
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix
    
    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)
    
    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

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




function solve_matrix_equation_forward(ABC::Vector{Float64};
    coords::Vector{Tuple{Vector{Int}, Vector{Int}}},
    dims::Vector{Tuple{Int,Int}},
    sparse_output::Bool = false,
    solver::Symbol = :doubling)

    if length(coords) == 1
        lengthA = length(coords[1][1])
        vA = ABC[1:lengthA]
        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        C = reshape(ABC[lengthA+1:end],dims[2]...)
        if solver != :doubling
            B = A'
        end
    elseif length(coords) == 3
        lengthA = length(coords[1][1])
        lengthB = length(coords[2][1])

        vA = ABC[1:lengthA]
        vB = ABC[lengthA .+ (1:lengthB)]
        vC = ABC[lengthA + lengthB + 1:end]

        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        B = sparse(coords[2]...,vB,dims[2]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        C = sparse(coords[3]...,vC,dims[3]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
    else
        lengthA = dims[1][1] * dims[1][2]
        A = reshape(ABC[1:lengthA],dims[1]...)
        C = reshape(ABC[lengthA+1:end],dims[2]...)
        if solver != :doubling
            B = A'
        end
    end
    

    if solver ∈ [:gmres, :bicgstab]
        function sylvester!(sol,𝐱)
            𝐗 = reshape(𝐱, size(C))
            sol .= vec(A * 𝐗 * B - 𝐗)
            return sol
        end
        
        sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

        if solver == :gmres
            𝐂, info = Krylov.gmres(sylvester, [vec(C);])
        elseif solver == :bicgstab
            𝐂, info = Krylov.bicgstab(sylvester, [vec(C);])
        end
        solved = info.solved
    elseif solver == :iterative
        iter = 1
        change = 1
        𝐂  = C
        𝐂¹ = C
        while change > eps(Float32) && iter < 10000
            𝐂¹ = A * 𝐂 * B - C
            if !(𝐂¹ isa DenseMatrix)
                droptol!(𝐂¹, eps())
            end
            if iter > 500
                change = maximum(abs, 𝐂¹ - 𝐂)
            end
            𝐂 = 𝐂¹
            iter += 1
        end
        solved = change < eps(Float32)
    elseif solver == :doubling
        iter = 1
        change = 1
        𝐂  = -C
        𝐂¹ = -C
        while change > eps(Float32) && iter < 500
            𝐂¹ = A * 𝐂 * A' + 𝐂
            A *= A
            if !(A isa DenseMatrix)
                droptol!(A, eps())
            end
            if iter > 10
                change = maximum(abs, 𝐂¹ - 𝐂)
            end
            𝐂 = 𝐂¹
            iter += 1
        end
        solved = change < eps(Float32)
    elseif solver == :sylvester
        𝐂 = try MatrixEquations.sylvd(collect(-A),collect(B),-C)
        catch
            return sparse_output ? spzeros(0,0) : zeros(0,0), false
        end
        
        solved = isapprox(𝐂, A * 𝐂 * B - C, rtol = eps(Float32))
    elseif solver == :lyapunov
        𝐂 = MatrixEquations.lyapd(collect(A),-C)
        solved = isapprox(𝐂, A * 𝐂 * A' - C, rtol = eps(Float32))
    elseif solver == :speedmapping
        
        soll = @suppress begin
            speedmapping(collect(-C); m! = (X, x) -> X .= A * x * B - C, stabilize = true)
        end
        𝐂 = soll.minimizer

        solved = soll.converged
    end

    return sparse_output ? sparse(reshape(𝐂, size(C))) : reshape(𝐂, size(C)), solved # return info on convergence
end



function solve_matrix_equation_conditions(ABC::Vector{<: Real},
    X::AbstractMatrix{<: Real}, 
    solved::Bool;
    coords::Vector{Tuple{Vector{Int}, Vector{Int}}},
    dims::Vector{Tuple{Int,Int}},
    sparse_output::Bool = false,
    solver::Symbol = :doubling)

    solver = :gmres # ensure the AXB works always

    if length(coords) == 1
        lengthA = length(coords[1][1])
        vA = ABC[1:lengthA]
        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        C = reshape(ABC[lengthA+1:end],dims[2]...)
        if solver != :doubling
            B = A' |> sparse |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        end
    elseif length(coords) == 3
        lengthA = length(coords[1][1])
        lengthB = length(coords[2][1])
        
        vA = ABC[1:lengthA]
        vB = ABC[lengthA .+ (1:lengthB)]
        vC = ABC[lengthA + lengthB + 1:end]

        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        B = sparse(coords[2]...,vB,dims[2]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        C = sparse(coords[3]...,vC,dims[3]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
    else
        lengthA = dims[1][1] * dims[1][2]
        A = reshape(ABC[1:lengthA],dims[1]...)
        C = reshape(ABC[lengthA+1:end],dims[2]...)
        if solver != :doubling
            B = A'
        end
    end

    A * X * B - C - X
end



function solve_matrix_equation_forward(abc::Vector{ℱ.Dual{Z,S,N}};
    coords::Vector{Tuple{Vector{Int}, Vector{Int}}},
    dims::Vector{Tuple{Int,Int}},
    sparse_output::Bool = false,
    solver::Symbol = :doubling) where {Z,S,N}

    # unpack: AoS -> SoA
    ABC = ℱ.value.(abc)

    # you can play with the dimension here, sometimes it makes sense to transpose
    partial_values = zeros(length(abc), N)
    for i in 1:N
        partial_values[:,i] = ℱ.partials.(abc, i)
    end

    # get f(vs)
    val, solved = solve_matrix_equation_forward(ABC, coords = coords, dims = dims, sparse_output = sparse_output, solver = solver)

    if length(coords) == 1
        lengthA = length(coords[1][1])

        vA = ABC[1:lengthA]
        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        # C = reshape(ABC[lengthA+1:end],dims[2]...)
        droptol!(A,eps())

        B = sparse(A') |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

        partials = zeros(dims[1][1] * dims[1][2] + dims[2][1] * dims[2][2], size(partial_values,2))
        partials[vcat(coords[1][1] + (coords[1][2] .- 1) * dims[1][1], dims[1][1] * dims[1][2] + 1:end),:] = partial_values

        reshape_matmul_b = LinearOperators.LinearOperator(Float64, length(val) * size(partials,2), 2*size(A,1)^2 * size(partials,2), false, false, 
        (sol,𝐱) -> begin 
            𝐗 = reshape(𝐱, (2* size(A,1)^2,size(partials,2))) |> sparse

            b = hcat(jacobian_wrt_A(A, val), -ℒ.I(length(val)))
            droptol!(b,eps())

            sol .= vec(b * 𝐗)
            return sol
        end)
    elseif length(coords) == 3
        lengthA = length(coords[1][1])
        lengthB = length(coords[2][1])

        vA = ABC[1:lengthA]
        vB = ABC[lengthA .+ (1:lengthB)]
        # vC = ABC[lengthA + lengthB + 1:end]

        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        B = sparse(coords[2]...,vB,dims[2]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        # C = sparse(coords[3]...,vC,dims[3]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

        partials = spzeros(dims[1][1] * dims[1][2] + dims[2][1] * dims[2][2] + dims[3][1] * dims[3][2], size(partial_values,2))
        partials[vcat(
            coords[1][1] + (coords[1][2] .- 1) * dims[1][1], 
            coords[2][1] + (coords[2][2] .- 1) * dims[2][1] .+ dims[1][1] * dims[1][2], 
            coords[3][1] + (coords[3][2] .- 1) * dims[3][1] .+ dims[1][1] * dims[1][2] .+ dims[2][1] * dims[2][2]),:] = partial_values
        
        reshape_matmul_b = LinearOperators.LinearOperator(Float64, length(val) * size(partials,2), (length(A) + length(B) + length(val)) * size(partials,2), false, false, 
            (sol,𝐱) -> begin 
                𝐗 = reshape(𝐱, (length(A) + length(B) + length(val), size(partials,2))) |> sparse

                jacobian_A = ℒ.kron(val * B, ℒ.I(size(A,1)))
                jacobian_B = ℒ.kron(ℒ.I(size(B,1)), A * val)

                b = hcat(jacobian_A', jacobian_B, -ℒ.I(length(val)))
                droptol!(b,eps())

                sol .= vec(b * 𝐗)
                return sol
        end)
    else
        lengthA = dims[1][1] * dims[1][2]
        A = reshape(ABC[1:lengthA],dims[1]...) |> sparse
        droptol!(A, eps())
        # C = reshape(ABC[lengthA+1:end],dims[2]...)
        B = sparse(A') |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

        partials = partial_values

        reshape_matmul_b = LinearOperators.LinearOperator(Float64, length(val) * size(partials,2), 2*size(A,1)^2 * size(partials,2), false, false, 
        (sol,𝐱) -> begin 
            𝐗 = reshape(𝐱, (2* size(A,1)^2,size(partials,2))) |> sparse

            b = hcat(jacobian_wrt_A(A, val), -ℒ.I(length(val)))
            droptol!(b,eps())

            sol .= vec(b * 𝐗)
            return sol
        end)
    end
    
    # get J(f, vs) * ps (cheating). Write your custom rule here. This used to be the conditions but here they are analytically derived.
    reshape_matmul_a = LinearOperators.LinearOperator(Float64, length(val) * size(partials,2), length(val) * size(partials,2), false, false, 
        (sol,𝐱) -> begin 
        𝐗 = reshape(𝐱, (length(val),size(partials,2))) |> sparse

        a = jacobian_wrt_values(A, B)
        droptol!(a,eps())

        sol .= vec(a * 𝐗)
        return sol
    end)

    X, info = Krylov.gmres(reshape_matmul_a, vec(reshape_matmul_b * vec(partials)))#, atol = tol)

    jvp = reshape(X, (length(val), size(partials,2)))

    out = reshape(map(val, eachrow(jvp)) do v, p
            ℱ.Dual{Z}(v, p...) # Z is the tag
        end,size(val))

    # pack: SoA -> AoS
    return sparse_output ? sparse(out) : out, solved
end


solve_matrix_equation_AD = ℐ.ImplicitFunction(solve_matrix_equation_forward, 
                                                solve_matrix_equation_conditions)

solve_matrix_equation_AD_direct = ℐ.ImplicitFunction(solve_matrix_equation_forward, 
                                                solve_matrix_equation_conditions; 
                                                linear_solver = ℐ.DirectLinearSolver())



function calculate_second_order_moments(
    parameters::Vector{<: Real}, 
    𝓂::ℳ; 
    covariance::Bool = true,
    verbose::Bool = false, 
    tol::AbstractFloat = eps())

    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose)

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
    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

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

    r1,c1,v1 = findnz(sparse(ŝ_to_ŝ₂))

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(ŝ_to_ŝ₂))
    push!(dimensions,size(C))
    
    values = vcat(v1, vec(collect(-C)))

    Σᶻ₂, info = solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)
    
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
                                            tol::AbstractFloat = eps()) where {U, T <: Real}

    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(parameters, 𝓂, verbose = verbose)

    if !covariance && !autocorrelation
        return μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂
    end

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
                                                𝓂.solution.perturbation.second_order_auxilliary_matrices, 
                                                𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

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

        r1,c1,v1 = findnz(ŝ_to_ŝ₃)

        coordinates = Tuple{Vector{Int}, Vector{Int}}[]
        push!(coordinates,(r1,c1))
        
        dimensions = Tuple{Int, Int}[]
        push!(dimensions,size(ŝ_to_ŝ₃))
        push!(dimensions,size(C))
        
        values = vcat(v1, vec(collect(-C)))

        Σᶻ₃, info = solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)

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


function calculate_kalman_filter_loglikelihood(𝓂::ℳ, observables::Union{Vector{String}, Vector{Symbol}}, 𝐒₁::Matrix{S}, data_in_deviations::Matrix{S}) where S
    observables_and_states = @ignore_derivatives sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

    A = @views 𝐒₁[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = @views 𝐒₁[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior
    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    
    dimensions = [size(A),size(𝐁)]
    
    values = vcat(vec(A), vec(collect(-𝐁)))

    P, _ = solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)
    # P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))

    u = zeros(length(observables_and_states))
    # u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
    z = C * u
    
    loglik = 0.0

    for t in 1:size(data_in_deviations)[2]
        v = data_in_deviations[:,t] - z

        F = C * P * C'

        # F = (F + F') / 2

        # loglik += log(max(eps(),ℒ.det(F))) + v' * ℒ.pinv(F) * v
        # K = P * C' * ℒ.pinv(F)

        # loglik += log(max(eps(),ℒ.det(F))) + v' / F  * v
        Fdet = ℒ.det(F)

        if Fdet < eps() return -Inf end

        F̄ = ℒ.lu(F, check = false)

        if !ℒ.issuccess(F̄) return -Inf end

        invF = inv(F̄)

        loglik += log(Fdet) + v' * invF  * v
        
        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)
        
        z = C * u 
    end

    return -(loglik + length(data_in_deviations) * log(2 * 3.141592653589793)) / 2 
end



function calculate_inversion_filter_loglikelihood(𝓂::ℳ, state::Union{Vector{Float64},Vector{Vector{Float64}}}, state_update::Function, data_in_deviations::Matrix{Float64}, observables::Union{Vector{String}, Vector{Symbol}}, warmup_iterations::Int)
    if state isa Vector{Float64}
        pruning = false
    else
        pruning = true
    end

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))

    shocks² = 0.0
    logabsdets = 0.0

    if warmup_iterations > 0
        res = Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
                            zeros(𝓂.timings.nExo * warmup_iterations), 
                            Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                            Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                            autodiff = :forward)

        matched = Optim.minimum(res) < 1e-12

        if !matched # for robustness try other linesearch
            res = Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
                            zeros(𝓂.timings.nExo * warmup_iterations), 
                            Optim.LBFGS(), 
                            Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                            autodiff = :forward)
        
            matched = Optim.minimum(res) < 1e-12
        end

        if !matched return -Inf end

        x = Optim.minimizer(res)

        warmup_shocks = reshape(x, 𝓂.timings.nExo, warmup_iterations)

        for i in 1:warmup_iterations-1
            state = state_update(state, warmup_shocks[:,i])
        end
        
        res = zeros(0)

        jacc = zeros(𝓂.timings.nExo * warmup_iterations, length(observables))

        match_initial_data!(res, x, jacc, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor), zeros(size(data_in_deviations, 1))

        for i in 1:warmup_iterations
            if 𝓂.timings.nExo == length(observables)
                logabsdets += ℒ.logabsdet(jacc[(i - 1) * 𝓂.timings.nExo .+ (1:2),:] ./ precision_factor)[1]
            else
                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[(i - 1) * 𝓂.timings.nExo .+ (1:2),:] ./ precision_factor))
            end
        end

        shocks² += sum(abs2,x)
    end

    for i in axes(data_in_deviations,2)
        res = Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                        zeros(𝓂.timings.nExo), 
                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                        Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                        autodiff = :forward)

        matched = Optim.minimum(res) < 1e-12

        if !matched # for robustness try other linesearch
            res = Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                            zeros(𝓂.timings.nExo), 
                            Optim.LBFGS(), 
                            Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                            autodiff = :forward)
        
            matched = Optim.minimum(res) < 1e-12
        end

        if !matched return -Inf end

        x = Optim.minimizer(res)

        res  = zeros(0)

        jacc = zeros(𝓂.timings.nExo, length(observables))

        match_data_sequence!(res, x, jacc, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor)

        if 𝓂.timings.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x)

        state = state_update(state, x)
    end

    return -(logabsdets + shocks² + (𝓂.timings.nExo * (warmup_iterations + n_obs)) * log(2 * 3.141592653589793)) / 2
end


function inversion_filter(𝓂::ℳ, 
    data_in_deviations::AbstractArray{Float64},
    algorithm::Symbol; 
    warmup_iterations::Int = 0,
    verbose::Bool = false, 
    tol::AbstractFloat = eps())
    
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
        SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)

        if solution_error > tol || isnan(solution_error)
            @error "No solution for these parameters."
        end

        state = zeros(𝓂.timings.nVars)

        ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix

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
        
    return states, shocks
end


function filter_and_smooth(𝓂::ℳ, data_in_deviations::AbstractArray{Float64}, observables::Vector{Symbol}; verbose::Bool = false, tol::AbstractFloat = eps())
    # Based on Durbin and Koopman (2012)
    # https://jrnold.github.io/ssmodels-in-stan/filtering-and-smoothing.html#smoothing

    @assert length(observables) == size(data_in_deviations)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    sort!(observables)

    solve!(𝓂, verbose = verbose)

    parameters = 𝓂.parameter_values

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
    @assert solution_error < tol "Could not solve non stochastic steady state." 

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix

    sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    B = @views sol[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(𝓂.timings.nVars))[sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),:]

    𝐁 = B * B'

    P̄ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]

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
# @setup_workload begin
#     # Putting some things in `setup` can reduce the size of the
#     # precompile file and potentially make loading faster.
#     @model FS2000 precompile = true begin
#         dA[0] = exp(gam + z_e_a  *  e_a[x])
#         log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  e_m[x]
#         - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0
#         W[0] = l[0] / n[0]
#         - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0
#         R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]
#         1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0
#         c[0] + k[0] = exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  *  e_a[x])) * k[-1]
#         P[0] * c[0] = m[0]
#         m[0] - 1 + d[0] = l[0]
#         e[0] = exp(z_e_a  *  e_a[x])
#         y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x]))
#         gy_obs[0] = dA[0] * y[0] / y[-1]
#         gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]
#         log_gy_obs[0] = log(gy_obs[0])
#         log_gp_obs[0] = log(gp_obs[0])
#     end

#     @parameters FS2000 silent = true precompile = true begin  
#         alp     = 0.356
#         bet     = 0.993
#         gam     = 0.0085
#         mst     = 1.0002
#         rho     = 0.129
#         psi     = 0.65
#         del     = 0.01
#         z_e_a   = 0.035449
#         z_e_m   = 0.008862
#     end
    
#     ENV["GKSwstype"] = "nul"

#     @compile_workload begin
#         # all calls in this block will be precompiled, regardless of whether
#         # they belong to your package or not (on Julia 1.8 and higher)
#         @model RBC precompile = true begin
#             1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
#             c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
#             z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
#         end

#         @parameters RBC silent = true precompile = true begin
#             δ = 0.02
#             α = 0.5
#         end

#         get_SS(FS2000)
#         get_SS(FS2000, parameters = :alp => 0.36)
#         get_solution(FS2000)
#         get_solution(FS2000, parameters = :alp => 0.35)
#         get_standard_deviation(FS2000)
#         get_correlation(FS2000)
#         get_autocorrelation(FS2000)
#         get_variance_decomposition(FS2000)
#         get_conditional_variance_decomposition(FS2000)
#         get_irf(FS2000)

#         data = simulate(FS2000)([:c,:k],:,:simulate)
#         get_loglikelihood(FS2000, data, FS2000.parameter_values)
#         get_mean(FS2000, silent = true)
#         # get_SSS(FS2000, silent = true)
#         # get_SSS(FS2000, algorithm = :third_order, silent = true)

#         # import Plots, StatsPlots
#         # plot_irf(FS2000)
#         # plot_solution(FS2000,:k) # fix warning when there is no sensitivity and all values are the same. triggers: no strict ticks found...
#         # plot_conditional_variance_decomposition(FS2000)
#     end
# end

end
