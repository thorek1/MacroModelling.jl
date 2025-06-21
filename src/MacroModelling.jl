module MacroModelling


import DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF, TYPEDSIGNATURES, TYPEDFIELDS
# import StatsFuns: normcdf
import ThreadedSparseArrays
using PrecompileTools
import SpecialFunctions: erfcinv, erfc # can't use constants because of SymPy (e.g. sqrt2)
import SpecialFunctions
import SymPyPythonCall as SPyPyC
import PythonCall
import Symbolics
import Accessors
# import TimerOutputs
# import TimerOutputs: TimerOutput, @timeit, @timeit_debug
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

import LoopVectorization: @turbo
import Polyester
import NLopt
import Optim, LineSearches
# import Zygote
import SparseArrays: SparseMatrixCSC, SparseVector, AbstractSparseArray, AbstractSparseMatrix, sparse!, spzeros, nnz, issparse, nonzeros #, sparse, droptol!, sparsevec, spdiagm, findnz#, sparse!
import LinearAlgebra as ℒ
import LinearSolve as 𝒮
# import LinearAlgebra: mul!
# import Octavian: matmul!
# import TriangularSolve as TS
# import ComponentArrays as 𝒞
import Combinatorics: combinations
import BlockTriangularForm
import Subscripts: super, sub
import Krylov
import Krylov: GmresWorkspace, DqgmresWorkspace, BicgstabWorkspace
import LinearOperators
import DataStructures: CircularBuffer
import MacroTools: unblock, postwalk, prewalk, @capture, flatten

# import SpeedMapping: speedmapping
import Suppressor: @suppress
import REPL
import Unicode
import MatrixEquations # good overview: https://cscproxy.mpi-magdeburg.mpg.de/mpcsc/benner/talks/Benner-Melbourne2019.pdf
# import NLboxsolve: nlboxsolve
# using NamedArrays
# using AxisKeys

import ChainRulesCore: @ignore_derivatives, ignore_derivatives, rrule, NoTangent, @thunk, ProjectTo, unthunk
import RecursiveFactorization as RF

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

using Requires

import Reexport
Reexport.@reexport import AxisKeys: KeyedArray, axiskeys, rekey, NamedDimsArray
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


using DispatchDoctor
# @stable default_mode = "disable" begin

# Imports
include("common_docstrings.jl")
include("options_and_caches.jl")
include("structures.jl")
include("macros.jl")
include("get_functions.jl")
include("dynare.jl")
include("inspect.jl")
include("moments.jl")
include("perturbation.jl")

include("./algorithms/sylvester.jl")
include("./algorithms/lyapunov.jl")
include("./algorithms/nonlinear_solver.jl")
include("./algorithms/quadratic_matrix_equation.jl")

include("./filter/find_shocks.jl")
include("./filter/inversion.jl")
include("./filter/kalman.jl")


# end # DispatchDoctor

function __init__()
    @require StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd" include("plotting.jl")
    @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("priors.jl")
end


export @model, @parameters, solve!
export plot_irfs, plot_irf, plot_IRF, plot_simulations, plot_solution, plot_simulation, plot_girf #, plot
export plot_conditional_variance_decomposition, plot_forecast_error_variance_decomposition, plot_fevd, plot_model_estimates, plot_shock_decomposition
export get_irfs, get_irf, get_IRF, simulate, get_simulation, get_simulations, get_girf
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
export Beta, InverseGamma, Gamma, Normal, Cauchy
export Tolerances

export translate_mod_file, translate_dynare_file, import_model, import_dynare
export write_mod_file, write_dynare_file, write_to_dynare_file, write_to_dynare, export_dynare, export_to_dynare, export_mod_file, export_model

export get_equations, get_steady_state_equations, get_dynamic_equations, get_calibration_equations, get_parameters, get_calibrated_parameters, get_parameters_in_equations, get_parameters_defined_by_parameters, get_parameters_defining_parameters, get_calibration_equation_parameters, get_variables, get_nonnegativity_auxilliary_variables, get_dynamic_auxilliary_variables, get_shocks, get_state_variables, get_jump_variables
# Internal
export irf, girf

# Remove comment for debugging
# export block_solver, remove_redundant_SS_vars!, write_parameters_input!, parse_variables_input_to_index, undo_transformer , transformer, calculate_third_order_stochastic_steady_state, calculate_second_order_stochastic_steady_state, filter_and_smooth
# export create_symbols_eqs!, solve_steady_state!, write_functions_mapping!, solve!, parse_algorithm_to_state_update, block_solver, block_solver_AD, calculate_covariance, calculate_jacobian, calculate_first_order_solution, expand_steady_state, get_symbols, calculate_covariance_AD, parse_shocks_input_to_index

@stable default_mode = "disable" begin

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

end # dispatch_doctor

# ── norminvcdf, norminv & qnorm ──
# d/dp (norminvcdf(p)) = 1 / normpdf(norminvcdf(p))
function Symbolics.derivative(::typeof(norminvcdf), args::NTuple{1,Any}, ::Val{1})
    p = args[1]
    1 / normpdf(norminvcdf(p))
end
# norminv and qnorm are aliases of norminvcdf, so they share the same rule:
Symbolics.derivative(::typeof(norminv), args::NTuple{1,Any}, ::Val{1}) = 
    Symbolics.derivative(norminvcdf, args, Val{1}())
Symbolics.derivative(::typeof(qnorm),  args::NTuple{1,Any}, ::Val{1}) =
    Symbolics.derivative(norminvcdf, args, Val{1}())

# ── normlogpdf ──
# d/dz (normlogpdf(z)) = −z
function Symbolics.derivative(::typeof(normlogpdf), args::NTuple{1,Any}, ::Val{1})
    z = args[1]
    -z
end

# ── normpdf & dnorm ──
# normpdf(z) = (1/√(2π)) e^(−z²/2) ⇒ derivative = −z * normpdf(z)
function Symbolics.derivative(::typeof(normpdf), args::NTuple{1,Any}, ::Val{1})
    z = args[1]
    -z * normpdf(z)
end
# alias:
Symbolics.derivative(::typeof(dnorm), args::NTuple{1,Any}, ::Val{1}) = 
    Symbolics.derivative(normpdf, args, Val{1}())

# ── normcdf & pnorm ──
# d/dz (normcdf(z)) = normpdf(z)
function Symbolics.derivative(::typeof(normcdf), args::NTuple{1,Any}, ::Val{1})
    z = args[1]
    normpdf(z)
end
# alias:
Symbolics.derivative(::typeof(pnorm), args::NTuple{1,Any}, ::Val{1}) = 
    Symbolics.derivative(normcdf, args, Val{1}())

@stable default_mode = "disable" begin


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

end # dispatch_doctor

function mul_reverse_AD!(   C::Matrix{S},
                            A::AbstractMatrix{M},
                            B::AbstractMatrix{N}) where {S <: Real, M <: Real, N <: Real}
    ℒ.mul!(C,A,B)
end

function rrule( ::typeof(mul_reverse_AD!),
                C::Matrix{S},
                A::AbstractMatrix{M},
                B::AbstractMatrix{N}) where {S <: Real, M <: Real, N <: Real}
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)

    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        dA = @thunk(project_A(Ȳ * B'))
        dB = @thunk(project_B(A' * Ȳ))
        return NoTangent(), NoTangent(), dA, dB
    end

    return ℒ.mul!(C,A,B), times_pullback
end

@stable default_mode = "disable" begin

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


function replace_with_one(equation::SPyPyC.Sym{PythonCall.Core.Py}, variable::SPyPyC.Sym{PythonCall.Core.Py})::SPyPyC.Sym{PythonCall.Core.Py}
    # equation.subs(variable, 1).replace(SPyPyC.Sym(ℯ), exp(1))
    tmp = SPyPyC.subs(equation, variable, 1)

    return replace_e(tmp)
end

function replace_e(equation::SPyPyC.Sym{PythonCall.Core.Py})::SPyPyC.Sym{PythonCall.Core.Py}
    outraw =  SPyPyC.subs(equation, SPyPyC.Sym(ℯ), exp(1))

    if outraw isa SPyPyC.Sym{PythonCall.Core.Py}
        out = outraw
    else
        out = collect(outraw)[1]
    end
    
    return out
end

function replace_symbolic(equation::SPyPyC.Sym{PythonCall.Core.Py}, variable::SPyPyC.Sym{PythonCall.Core.Py}, replacement::SPyPyC.Sym{PythonCall.Core.Py})::SPyPyC.Sym{PythonCall.Core.Py}
    # equation.subs(variable, replacement)
    return SPyPyC.subs(equation, variable, replacement)
end

function solve_symbolically(equation::SPyPyC.Sym{PythonCall.Core.Py}, variable::SPyPyC.Sym{PythonCall.Core.Py})::Union{Nothing,Vector{SPyPyC.Sym{PythonCall.Core.Py}}}
    soll =  try SPyPyC.solve(equation, variable)
            catch
            end

    return soll
end

function solve_symbolically(equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}, variables::Vector{SPyPyC.Sym{PythonCall.Core.Py}})::Union{Nothing,Dict{SPyPyC.Sym{PythonCall.Core.Py}, SPyPyC.Sym{PythonCall.Core.Py}}}
    soll =  try SPyPyC.solve(equations, variables)
            catch
            end

    if soll == Any[]
        soll = Dict{SPyPyC.Sym{PythonCall.Core.Py}, SPyPyC.Sym{PythonCall.Core.Py}}()
    elseif soll isa Vector
        soll = Dict{SPyPyC.Sym{PythonCall.Core.Py}, SPyPyC.Sym{PythonCall.Core.Py}}(variables .=> soll[1])
    end
    
    return soll
end

function transform_obc(ex::Expr; avoid_solve::Bool = false)
    transformed_expr, reverse_dict = transform_expression(ex)

    for symbs in get_symbols(transformed_expr)
        eval(:($symbs = SPyPyC.symbols($(string(symbs)), real = true, finite = true)))
    end

    eq = eval(transformed_expr)

    if avoid_solve || count_ops(Meta.parse(string(eq))) > 15
        soll = nothing
    else
        soll = solve_symbolically(eq, eval(:minmax__P))
    end

    if !isempty(soll)
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

	return nothing
end

function obc_objective_optim_fun(X::Vector{S}, grad::Vector{S})::S where S
    if length(grad) > 0
        grad .= 2 .* X
    end
    
    sum(abs2, X)
end


function minimize_distance_to_conditions(X::Vector{S}, p)::S where S
    Conditions, State_update, Shocks, Cond_var_idx, Free_shock_idx, State, Pruning, precision_factor = p

    Shocks[Free_shock_idx] .= X

    new_State = State_update(State, convert(typeof(X), Shocks))

    cond_vars = Pruning ? sum(new_State) : new_State

    return precision_factor * sum(abs2, Conditions[Cond_var_idx] - cond_vars[Cond_var_idx])
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


function clear_solution_caches!(𝓂::ℳ, algorithm::Symbol)
    for i in [:first_order, :pruned_second_order, :second_order, :pruned_third_order, :third_order]
        push!(𝓂.solution.outdated_algorithms, i)
    end

    while length(𝓂.NSSS_solver_cache) > 1
        pop!(𝓂.NSSS_solver_cache)
    end

    𝓂.solution.outdated_NSSS = true
    𝓂.solution.perturbation.qme_solution = zeros(0,0)
    𝓂.solution.perturbation.second_order_solution = spzeros(0,0)
    𝓂.solution.perturbation.third_order_solution = spzeros(0,0)

    return nothing
end

function fill_kron_adjoint!(∂A::AbstractMatrix{R}, 
                            ∂B::AbstractMatrix{R}, 
                            ∂X::AbstractSparseMatrix{R}, 
                            A::AbstractMatrix{R}, 
                            B::AbstractMatrix{R}) where R <: Real
    @assert size(∂A) == size(A)
    @assert size(∂B) == size(B)
    @assert length(∂X) == length(B) * length(A) "∂X must have the same length as kron(B,A)"
    
    n1, m1 = size(B)
    n2 = size(A,1)
    
    # Precompute constants
    const_n1n2 = n1 * n2
    const_n1n2m1 = n1 * n2 * m1

    # Access the sparse matrix internal representation
    if ∂X isa SparseMatrixCSC
        colptr = ∂X.colptr  # Column pointers
        rowval = ∂X.rowval  # Row indices of non-zeros
        nzval  = ∂X.nzval   # Non-zero values
    else
        colptr = ∂X.A.colptr  # Column pointers
        rowval = ∂X.A.rowval  # Row indices of non-zeros
        nzval  = ∂X.A.nzval   # Non-zero values
    end
    
    # Iterate over columns of ∂X
    for col in 1:size(∂X, 2)
        # Iterate over the non-zeros in this column
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            linear_idx = (col - 1) * size(∂X, 1) + row

            @inbounds begin
                i = (linear_idx - 1) % n1 + 1
                k = ((linear_idx - 1) ÷ n1) % n2 + 1
                j = ((linear_idx - 1) ÷ const_n1n2) % m1 + 1
                l = ((linear_idx - 1) ÷ const_n1n2m1) + 1
                
                # Update ∂B and ∂A
                ∂A[k,l] += B[i,j] * val
                ∂B[i,j] += A[k,l] * val
            end
        end
    end
end


function fill_kron_adjoint!(∂A::AbstractMatrix{R}, 
                            ∂B::AbstractMatrix{R}, 
                            ∂X::DenseMatrix{R}, 
                            A::AbstractMatrix{R}, 
                            B::AbstractMatrix{R}) where R <: Real
    @assert size(∂A) == size(A)
    @assert size(∂B) == size(B)
    @assert length(∂X) == length(B) * length(A) "∂X must have the same length as kron(B,A)"
    
    re∂X = reshape(∂X, 
                    size(A,1), 
                    size(B,1), 
                    size(A,2), 
                    size(B,2))

    ei = 1
    for e in eachslice(re∂X; dims = (1,3))
        @inbounds ∂A[ei] += ℒ.dot(B,e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂X; dims = (2,4))
        @inbounds ∂B[ei] += ℒ.dot(A,e)
        ei += 1
    end
end



function fill_kron_adjoint!(∂A::V, ∂B::V, ∂X::V, A::V, B::V) where V <: Vector{<: Real}
    @assert size(∂A) == size(A)
    @assert size(∂B) == size(B)
    @assert length(∂X) == length(B) * length(A) "∂X must have the same length as kron(B,A)"
    
    re∂X = reshape(∂X, 
                    length(A), 
                    length(B))

    ei = 1
    for e in eachslice(re∂X; dims = 1)
        @inbounds ∂A[ei] += ℒ.dot(B,e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂X; dims = 2)
        @inbounds ∂B[ei] += ℒ.dot(A,e)
        ei += 1
    end
end


function fill_kron_adjoint_∂B!(∂X::AbstractSparseMatrix{R}, ∂B::AbstractArray{S}, A::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂B) * length(A) "∂X must have the same length as kron(B,A)"
    
    n1, m1 = size(∂B)
    n2 = size(A,1)
    
    # Precompute constants
    const_n1n2 = n1 * n2
    const_n1n2m1 = n1 * n2 * m1
    
    # Access the sparse matrix internal representation
    colptr = ∂X.colptr  # Column pointers
    rowval = ∂X.rowval  # Row indices of non-zeros
    nzval  = ∂X.nzval   # Non-zero values
    
    # Iterate over columns of ∂X
    for col in 1:size(∂X, 2)
        # Iterate over the non-zeros in this column
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            linear_idx = (col - 1) * size(∂X, 1) + row

            @inbounds begin
                i = (linear_idx - 1) % n1 + 1
                k = ((linear_idx - 1) ÷ n1) % n2 + 1
                j = ((linear_idx - 1) ÷ const_n1n2) % m1 + 1
                l = ((linear_idx - 1) ÷ const_n1n2m1) + 1
                
                # Update ∂B and ∂A
                ∂B[i,j] += A[k,l] * val
            end
        end
    end
end



function fill_kron_adjoint_∂B!(∂X::AbstractSparseMatrix{R}, ∂B::Vector{S}, A::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂B) * length(A) "∂X must have the same length as kron(B,A)"
    
    n1 = length(∂B)
    n2 = size(A,1)
    # println("hello")
    # Precompute constants
    const_n1n2 = n1 * n2
    
    # Access the sparse matrix internal representation
    colptr = ∂X.colptr  # Column pointers
    rowval = ∂X.rowval  # Row indices of non-zeros
    nzval  = ∂X.nzval   # Non-zero values
    
    # Iterate over columns of ∂X
    for col in 1:size(∂X, 2)
        # Iterate over the non-zeros in this column
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            linear_idx = (col - 1) * size(∂X, 1) + row

            @inbounds begin
                i = (linear_idx - 1) % n1 + 1
                k = ((linear_idx - 1) ÷ n1) % n2 + 1
                l = ((linear_idx - 1) ÷ const_n1n2) + 1
                
                # Update ∂B and ∂A
                ∂B[i] += A[k,l] * val
            end
        end
    end
end



function fill_kron_adjoint_∂B!(∂X::DenseMatrix{R}, ∂B::Vector{S}, A::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂B) * length(A) "∂X must have the same length as kron(B,A)"
        
    re∂X = reshape(∂X, 
                    size(A,1), 
                    length(∂B), 
                    size(A,2))

    ei = 1
    for e in eachslice(re∂X; dims = 2)
        @inbounds ∂B[ei] += ℒ.dot(A,e)
        ei += 1
    end
end


function fill_kron_adjoint_∂A!(∂X::DenseMatrix{R}, ∂A::Vector{S}, B::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(∂A) * length(B) "∂X must have the same length as kron(B,A)"
        
    re∂X = reshape(∂X, 
                    length(∂A), 
                    size(B,1), 
                    size(B,2))

    ei = 1
    for e in eachslice(re∂X; dims = 1)
        @inbounds ∂A[ei] += ℒ.dot(B,e)
        ei += 1
    end
end


function fill_kron_adjoint_∂A!(∂X::AbstractSparseMatrix{R}, ∂A::AbstractMatrix{S}, B::AbstractMatrix{T}) where {R <: Real, S <: Real, T <: Real}
    @assert length(∂X) == length(B) * length(∂A) "∂X must have the same length as kron(B,A)"
    
    n1, m1 = size(B)
    n2 = size(∂A,1)
    
    # Precompute constants
    const_n1n2 = n1 * n2
    const_n1n2m1 = n1 * n2 * m1
    
    # Access the sparse matrix internal representation
    colptr = ∂X.colptr  # Column pointers
    rowval = ∂X.rowval  # Row indices of non-zeros
    nzval  = ∂X.nzval   # Non-zero values
    
    # Iterate over columns of ∂X
    for col in 1:size(∂X, 2)
        # Iterate over the non-zeros in this column
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            val = nzval[idx]

            linear_idx = (col - 1) * size(∂X, 1) + row

            @inbounds begin
                i = (linear_idx - 1) % n1 + 1
                k = ((linear_idx - 1) ÷ n1) % n2 + 1
                j = ((linear_idx - 1) ÷ const_n1n2) % m1 + 1
                l = ((linear_idx - 1) ÷ const_n1n2m1) + 1
                
                # Update ∂B and ∂A
                ∂A[k,l] += B[i,j] * val
            end
        end
    end
end


function choose_matrix_format(A::ℒ.Diagonal{S, Vector{S}}; 
                                density_threshold::Float64 = .1, 
                                min_length::Int = 1000,
                                tol::R = 1e-14,
                                multithreaded::Bool = true)::Union{Matrix{S}, SparseMatrixCSC{S, Int}, ThreadedSparseArrays.ThreadedSparseMatrixCSC{S, Int, SparseMatrixCSC{S, Int}}} where {R <: AbstractFloat, S <: Real}
    if length(A) < 100
        a = convert(Matrix, A)
    else
        if multithreaded
            a = A |> sparse |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        else
            a = A |> sparse
        end
    end

    return a
end


function choose_matrix_format(A::ℒ.Adjoint{S, M}; 
                                density_threshold::Float64 = .1, 
                                min_length::Int = 1000,
                                tol::R = 1e-14,
                                multithreaded::Bool = true)::Union{Matrix{S}, SparseMatrixCSC{S, Int}, ThreadedSparseArrays.ThreadedSparseMatrixCSC{S, Int, SparseMatrixCSC{S, Int}}} where {R <: AbstractFloat, S <: Real, M <: AbstractMatrix{S}}
    choose_matrix_format(convert(typeof(transpose(A)),A), 
                        density_threshold = density_threshold, 
                        min_length = min_length, 
                        multithreaded = multithreaded,
                        tol = tol)
end

# function choose_matrix_format(A::ℒ.Adjoint{S, <: AbstractSparseMatrix{S}}; 
#                                 density_threshold::Float64 = .1, 
#                                 min_length::Int = 1000,
#                                 tol::R = 1e-14,
#                                 multithreaded::Bool = true)::Union{Matrix{S}, SparseMatrixCSC{S, Int}, ThreadedSparseArrays.ThreadedSparseMatrixCSC{S, Int, SparseMatrixCSC{S, Int}}} where {R <: AbstractFloat, S <: Real}
#     choose_matrix_format(convert(typeof(transpose(A)),A), 
#                         density_threshold = density_threshold, 
#                         min_length = min_length, 
#                         multithreaded = multithreaded,
#                         tol = tol)
# end

function choose_matrix_format(A::DenseMatrix{S}; 
                                density_threshold::Float64 = .1, 
                                min_length::Int = 1000,
                                tol::R = 1e-14,
                                multithreaded::Bool = true)::Union{Matrix{S}, SparseMatrixCSC{S, Int}, ThreadedSparseArrays.ThreadedSparseMatrixCSC{S, Int, SparseMatrixCSC{S, Int}}} where {R <: AbstractFloat, S <: Real}
    if sum(abs.(A) .> tol) / length(A) < density_threshold && length(A) > min_length
        if multithreaded
            a = sparse(A) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
            droptol!(a, tol)
        else
            a = convert(SparseMatrixCSC{S}, A)
            droptol!(a, tol)
        end
    else
        a = convert(Matrix, A)
    end

    return a
end

function choose_matrix_format(A::AbstractSparseMatrix{S}; 
                                density_threshold::Float64 = .1, 
                                min_length::Int = 1000,
                                tol::R = 1e-14,
                                multithreaded::Bool = true)::Union{Matrix{S}, SparseMatrixCSC{S, Int}, ThreadedSparseArrays.ThreadedSparseMatrixCSC{S, Int, SparseMatrixCSC{S, Int}}} where {R <: AbstractFloat, S <: Real}
    droptol!(A, tol)

    lennz = nnz(A)

    if lennz / length(A) > density_threshold || length(A) < min_length
        a = convert(Matrix, A)
    else
        if multithreaded
            if A isa ThreadedSparseArrays.ThreadedSparseMatrixCSC
                a = A
            else
                a = A |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
            end
        else
            if A isa ThreadedSparseArrays.ThreadedSparseMatrixCSC
                a = A.A
            else
                a = A
            end
        end
    end

    return a
end

function mat_mult_kron(A::AbstractSparseMatrix{R},
                        B::AbstractMatrix{T},
                        C::AbstractMatrix{T},
                        D::AbstractMatrix{S}) where {R <: Real, T <: Real, S <: Real}
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    X = zeros(T, size(A,1), size(D,2))

    # vals = T[]
    # rows = Int[]
    # cols = Int[]

    Ā = zeros(T, n_rowC, n_rowB)
    ĀB = zeros(T, n_rowC, n_colB)
    CĀB = zeros(T, n_colC, n_colB)
    vCĀB = zeros(T, n_colB * n_colC)
    # vCĀBD = zeros(size(D,2))

    rv = unique(A isa SparseMatrixCSC ? A.rowval : A.A.rowval)

    # Polyester.@batch threadlocal = (Vector{T}(), Vector{Int}(), Vector{Int}()) for row in rv |> unique
    @inbounds for row in rv
        @views copyto!(Ā, A[row, :])
        ℒ.mul!(ĀB, Ā, B)
        ℒ.mul!(CĀB, C', ĀB)
        copyto!(vCĀB, CĀB)
        @views ℒ.mul!(X[row,:], D', vCĀB)
    end

    return choose_matrix_format(X)
    #     ℒ.mul!(vCĀBD, D', vCĀB)

    #     for (i,v) in enumerate(vCĀBD)
    #         if abs(v) > eps()
    #             push!(rows, row)
    #             push!(cols, i)
    #             push!(vals, v)
    #         end
    #     end
    # end

    # if VERSION >= v"1.10"
    #     return sparse!(rows, cols, vals, size(A,1), size(D,2))   
    # else
    #     return sparse(rows, cols, vals, size(A,1), size(D,2))   
    # end
end


function mat_mult_kron(A::DenseMatrix{R},
                        B::AbstractMatrix{T},
                        C::AbstractMatrix{T},
                        D::AbstractMatrix{S}) where {R <: Real, T <: Real, S <: Real}
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    X = zeros(T, size(A,1), size(D,2))

    # vals = T[]
    # rows = Int[]
    # cols = Int[]

    Ā = zeros(T, n_rowC, n_rowB)
    ĀB = zeros(T, n_rowC, n_colB)
    CĀB = zeros(T, n_colC, n_colB)
    vCĀB = zeros(T, n_colB * n_colC)
    # vCĀBD = zeros(size(D,2))

    # rv = A isa SparseMatrixCSC ? A.rowval : A.A.rowval

    # Polyester.@batch threadlocal = (Vector{T}(), Vector{Int}(), Vector{Int}()) for row in rv |> unique
    r = 1
    @inbounds for row in eachrow(A)
        @views copyto!(Ā, row)
        ℒ.mul!(ĀB, Ā, B)
        ℒ.mul!(CĀB, C', ĀB)
        copyto!(vCĀB, CĀB)
        @views ℒ.mul!(X[row,:], D', vCĀB)
        r += 1
    end

    return choose_matrix_format(X)
    #     ℒ.mul!(vCĀBD, D', vCĀB)

    #     for (i,v) in enumerate(vCĀBD)
    #         if abs(v) > eps()
    #             push!(rows, row)
    #             push!(cols, i)
    #             push!(vals, v)
    #         end
    #     end
    # end

    # if VERSION >= v"1.10"
    #     return sparse!(rows, cols, vals, size(A,1), size(D,2))   
    # else
    #     return sparse(rows, cols, vals, size(A,1), size(D,2))   
    # end
end

function mat_mult_kron(A::AbstractSparseMatrix{R},
                        B::AbstractMatrix{T},
                        C::AbstractMatrix{T};
                        sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        sparse::Bool = false) where {R <: Real, T <: Real}
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    if sparse
        nnzA = nnz(A)
        nnzB = sum(abs.(B) .> eps())
        nnzC = sum(abs.(C) .> eps())

        p = nnzA * nnzB * nnzC / (length(A) * length(B) * length(C))
        
        if length(sparse_preallocation[1]) == 0
            estimated_nnz = Int(ceil((1-(1-p)^size(A,1))*size(A,1) * n_colB * n_colC))

            resize!(sparse_preallocation[1], estimated_nnz)
            resize!(sparse_preallocation[2], estimated_nnz)
            resize!(sparse_preallocation[3], estimated_nnz)

            I = sparse_preallocation[1]
            J = sparse_preallocation[2]
            V = sparse_preallocation[3]
        else
            estimated_nnz = length(sparse_preallocation[3])

            resize!(sparse_preallocation[1], estimated_nnz)

            I = sparse_preallocation[1]
            J = sparse_preallocation[2]
            V = sparse_preallocation[3]
        end
    else
        X = zeros(T, size(A,1), n_colB * n_colC)
    end

    Ā = zeros(T, n_rowC, n_rowB)
    ĀB = zeros(T, n_rowC, n_colB)
    CĀB = zeros(T, n_colC, n_colB)

    rv = A isa SparseMatrixCSC ? A.rowval : A.A.rowval

    α = .7 # speed of Vector increase
    k = 0

    # Polyester.@batch threadlocal = (Vector{T}(), Vector{Int}(), Vector{Int}()) for row in rv |> unique
    @inbounds for row in rv |> unique
        @views copyto!(Ā, A[row, :])
        ℒ.mul!(ĀB, Ā, B)
        ℒ.mul!(CĀB, C', ĀB)
        
        if sparse
            for (i,v) in enumerate(CĀB)
                if abs(v) > eps()
                    k += 1

                    if k > estimated_nnz
                        estimated_nnz += min(size(A,1) * n_colB * n_colC, max(10000, Int(ceil((α - 1) * estimated_nnz + (1 - α) * size(A,1) * n_colB * n_colC))))
                        
                        resize!(I, estimated_nnz)
                        resize!(J, estimated_nnz)
                        resize!(V, estimated_nnz)
                    end

                    I[k] = row
                    J[k] = i
                    V[k] = v
                end
            end
        else
            @views copyto!(X[row,:], CĀB)
        end
    end

    if sparse
        resize!(I, k)
        resize!(J, k)
        resize!(V, k)

        klasttouch = sparse_preallocation[4] # Vector{Ti}(undef, n)
        csrrowptr  = sparse_preallocation[5] # Vector{Ti}(undef, m + 1)
        csrcolval  = sparse_preallocation[6] # Vector{Ti}(undef, length(I))
        csrnzval   = sparse_preallocation[7] # Vector{Tv}(undef, length(I))

        resize!(klasttouch, n_colB * n_colC)
        resize!(csrrowptr, size(A, 1) + 1)
        resize!(csrcolval, length(I))
        resize!(csrnzval, length(I))

        out = sparse!(I, J, V, size(A, 1), n_colB * n_colC, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
        # out = sparse!(I, J, V, size(A, 1), n_colB * n_colC)   
    else
        out = choose_matrix_format(X)
    end
    
    return out
end




function mat_mult_kron(A::DenseMatrix{R},
                        B::AbstractMatrix{T},
                        C::AbstractMatrix{T}) where {R <: Real, T <: Real}
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    X = zeros(T, size(A,1), n_colB * n_colC)

    # vals = T[]
    # rows = Int[]
    # cols = Int[]

    Ā = zeros(T, n_rowC, n_rowB)
    ĀB = zeros(T, n_rowC, n_colB)
    CĀB = zeros(T, n_colC, n_colB)

    # Polyester.@batch threadlocal = (Vector{T}(), Vector{Int}(), Vector{Int}()) for row in rv |> unique
    r = 1
    @inbounds for row in eachrow(A)
        @views copyto!(Ā, row)
        ℒ.mul!(ĀB, Ā, B)
        ℒ.mul!(CĀB, C', ĀB)
        
        @views copyto!(X[r,:], CĀB)
        r += 1
    end

    return choose_matrix_format(X)
    #     for (i,v) in enumerate(CĀB)
    #         if abs(v) > eps()
    #             push!(rows, row)
    #             push!(cols, i)
    #             push!(vals, v)
    #         end
    #     end
    # end

    # if VERSION >= v"1.10"
    #     return sparse!(rows,cols,vals,size(A,1),n_colB*n_colC)   
    # else
    #     return sparse(rows,cols,vals,size(A,1),n_colB*n_colC)   
    # end
end

function sparse_preallocated!(Ŝ::Matrix{T}; ℂ::higher_order_caches{T,F} = Higher_order_caches()) where {T <: Real, F <: AbstractFloat}
    if !(eltype(ℂ.tmp_sparse_prealloc6[3]) == T)
        ℂ.tmp_sparse_prealloc6 = Higher_order_caches(T = T, S = F)
    end

    I           = ℂ.tmp_sparse_prealloc6[1]
    J           = ℂ.tmp_sparse_prealloc6[2]
    V           = ℂ.tmp_sparse_prealloc6[3]

    klasttouch  = ℂ.tmp_sparse_prealloc6[4] # Vector{Ti}(undef, n)
    csrrowptr   = ℂ.tmp_sparse_prealloc6[5] # Vector{Ti}(undef, m + 1)
    csrcolval   = ℂ.tmp_sparse_prealloc6[6] # Vector{Ti}(undef, length(I))
    csrnzval    = ℂ.tmp_sparse_prealloc6[7] # Vector{Tv}(undef, length(I))

    resize!(I, length(Ŝ))
    resize!(J, length(Ŝ))
    resize!(V, length(Ŝ))
    resize!(klasttouch, length(Ŝ))

    copyto!(V,Ŝ) # this is key to reduce allocations

    klasttouch .= abs.(V) .> eps() # this is key to reduce allocations

    m, n = size(Ŝ)

    idx_redux = 0
    @inbounds for (idx,val) in enumerate(klasttouch)
        if val == 1
            idx_redux += 1
            j, i = divrem(idx - 1, m)
            I[idx_redux] = i + 1
            J[idx_redux] = j + 1
            klasttouch[idx_redux] = idx
        end
    end

    resize!(I, idx_redux)
    resize!(J, idx_redux)
    resize!(V, idx_redux)
    resize!(klasttouch, idx_redux)

    V = Ŝ[klasttouch]

    resize!(klasttouch, n)
    resize!(csrrowptr, m + 1)
    resize!(csrcolval, idx_redux)
    resize!(csrnzval, idx_redux)

    out = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)

    return out
end

end # dispatch_doctor

function rrule(::typeof(sparse_preallocated!), Ŝ::Matrix{T}; ℂ::higher_order_caches{T,F} = Higher_order_caches()) where {T <: Real, F <: AbstractFloat}
    project_Ŝ = ProjectTo(Ŝ)

    function sparse_preallocated_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        ΔŜ = project_Ŝ(ΔΩ)
        return NoTangent(), ΔŜ, NoTangent()
    end

    return sparse_preallocated!(Ŝ, ℂ = ℂ), sparse_preallocated_pullback
end

@stable default_mode = "disable" begin

function sparse_preallocated!(Ŝ::Matrix{ℱ.Dual{Z,S,N}}; ℂ::higher_order_caches{T,F} = Higher_order_caches()) where {Z,S,N,T <: Real, F <: AbstractFloat}
    sparse(Ŝ)
end


function compressed_kron³(a::AbstractMatrix{T};
                    rowmask::Vector{Int} = Int[],
                    colmask::Vector{Int} = Int[],
                    # timer::TimerOutput = TimerOutput(),
                    tol::AbstractFloat = eps(),
                    sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[])) where T <: Real
    # @timeit_debug timer "Compressed 3rd kronecker power" begin
          
    # @timeit_debug timer "Preallocation" begin
    
    a_is_adjoint = typeof(a) <: ℒ.Adjoint{T,Matrix{T}}
    
    if a_is_adjoint
        â = copy(a')
        a = sparse(a')
        
        rmask = colmask
        colmask = rowmask
        rowmask = rmask
    elseif typeof(a) <: DenseMatrix{T}
        â = copy(a)
        a = sparse(a)
    else
        â = convert(Matrix, a)  # Convert to dense matrix for faster access
    end
    # Get the number of rows and columns
    n_rows, n_cols = size(a)
    
    # Calculate the number of unique triplet indices for rows and columns
    m3_rows = n_rows * (n_rows + 1) * (n_rows + 2) ÷ 6    # For rows: i ≤ j ≤ k
    m3_cols = n_cols * (n_cols + 1) * (n_cols + 2) ÷ 6    # For columns: i ≤ j ≤ k

    if rowmask == Int[0] || colmask == Int[0]
        if a_is_adjoint
            return spzeros(T, m3_cols, m3_rows)
        else
            return spzeros(T, m3_rows, m3_cols)
        end
    end
    # Initialize arrays to collect indices and values
    # Estimate an upper bound for non-zero entries to preallocate arrays
    lennz = nnz(a) # a isa ThreadedSparseArrays.ThreadedSparseMatrixCSC ? length(a.A.nzval) : length(a.nzval)

    m3_c = length(colmask) > 0 ? length(colmask) : m3_cols
    m3_r = length(rowmask) > 0 ? length(rowmask) : m3_rows

    m3_exp = (length(colmask) > 0 || length(rowmask) > 0) ? 3 : 4

    if length(sparse_preallocation[1]) == 0
        estimated_nnz = floor(Int, max(m3_r * m3_c * (lennz / length(a)) ^ m3_exp, 10000))

        resize!(sparse_preallocation[1], estimated_nnz)
        resize!(sparse_preallocation[2], estimated_nnz)
        resize!(sparse_preallocation[3], estimated_nnz)

        I = sparse_preallocation[1]
        J = sparse_preallocation[2]
        V = sparse_preallocation[3]
    else
        estimated_nnz = length(sparse_preallocation[3])

        resize!(sparse_preallocation[1], estimated_nnz)

        I = sparse_preallocation[1]
        J = sparse_preallocation[2]
        V = sparse_preallocation[3]
    end

    # k = Threads.Atomic{Int}(0)  # Counter for non-zero entries
    # k̄ = Threads.Atomic{Int}(0)  # effectively slower than the non-threaded version

    k = 0

    # end # timeit_debug

    # @timeit_debug timer "findnz" begin
                
    # Find unique non-zero row and column indices
    rowinds, colinds, _ = findnz(a)
    ui = unique(rowinds)
    uj = unique(colinds)
       
    # end # timeit_debug

    # @timeit_debug timer "Loop" begin
    # Triple nested loops for (i1 ≤ j1 ≤ k1) and (i2 ≤ j2 ≤ k2)
    # Polyester.@batch threadlocal=(Vector{Int}(), Vector{Int}(), Vector{T}()) for i1 in ui
    # Polyester.@batch minbatch = 10 for i1 in ui
    # Threads.@threads for i1 in ui
    norowmask = length(rowmask) == 0
    nocolmask = length(colmask) == 0

    for i1 in ui
        for j1 in ui
            if j1 ≤ i1
                for k1 in ui
                    if k1 ≤ j1

                        row = (i1-1) * i1 * (i1+1) ÷ 6 + (j1-1) * j1 ÷ 2 + k1

                        if norowmask || row in rowmask
                            for i2 in uj
                                for j2 in uj
                                    if j2 ≤ i2
                                        for k2 in uj
                                            if k2 ≤ j2

                                                col = (i2-1) * i2 * (i2+1) ÷ 6 + (j2-1) * j2 ÷ 2 + k2

                                                if nocolmask || col in colmask
                                                    # @timeit_debug timer "Multiplication" begin
                                                    @inbounds aii = â[i1, i2]
                                                    @inbounds aij = â[i1, j2]
                                                    @inbounds aik = â[i1, k2]
                                                    @inbounds aji = â[j1, i2]
                                                    @inbounds ajj = â[j1, j2]
                                                    @inbounds ajk = â[j1, k2]
                                                    @inbounds aki = â[k1, i2]
                                                    @inbounds akj = â[k1, j2]
                                                    @inbounds akk = â[k1, k2]

                                                    # Compute the six unique products
                                                    # val = 0.0
                                                    # val += aii * ajj * akk
                                                    # val += aij * aji * akk
                                                    # val += aik * ajj * aki
                                                    # val += aij * ajk * aki
                                                    # val += aik * aji * akj
                                                    # val += aii * ajk * akj

                                                    val = aii * (ajj * akk + ajk * akj) + aij * (aji * akk + ajk * aki) + aik * (aji * akj + ajj * aki)
                                                    # end # timeit_debug

                                                    # @timeit_debug timer "Save in vector" begin
                                                        
                                                    # Only add non-zero values to the sparse matrix
                                                    if abs(val) > tol
                                                        # Threads.atomic_add!(k, 1)
                                                        # Threads.atomic_max!(k̄, k[])

                                                        if i1 == j1
                                                            if i1 == k1
                                                                divisor = 6
                                                            else
                                                                divisor = 2
                                                            end
                                                        else
                                                            if i1 ≠ k1 && j1 ≠ k1
                                                                divisor = 1
                                                            else
                                                                divisor = 2
                                                            end
                                                        end
                                                        # push!(threadlocal[1],row)
                                                        # push!(threadlocal[2],col)
                                                        # push!(threadlocal[3],val / divisor)
                                                        # I[k[]] = row
                                                        # J[k[]] = col
                                                        # V[k[]] = val / divisor 

                                                        k += 1

                                                        if k > estimated_nnz
                                                            estimated_nnz += Int(ceil(max(1000, estimated_nnz * .1)))
                                                            estimated_nnz = min(m3_cols * m3_rows, estimated_nnz)
                                                            resize!(I, estimated_nnz)
                                                            resize!(J, estimated_nnz)
                                                            resize!(V, estimated_nnz)
                                                        end

                                                        I[k] = row
                                                        J[k] = col
                                                        V[k] = val / divisor 
                                                    end

                                                    # end # timeit_debug
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # end # timeit_debug

    # @timeit_debug timer "Resize" begin

    # out = map(fetch, threadlocal)

    # I = mapreduce(v -> v[1], vcat, out)
    # J = mapreduce(v -> v[2], vcat, out)
    # V = mapreduce(v -> v[3], vcat, out)

    # # Resize the index and value arrays to the actual number of entries
    # resize!(I, k̄[])
    # resize!(J, k̄[])
    # resize!(V, k̄[]) 
    resize!(I, k)
    resize!(J, k)
    resize!(V, k)

    # end # timeit_debug
    # end # timeit_debug

    # Create the sparse matrix from the collected indices and values
    if a_is_adjoint
        klasttouch = sparse_preallocation[4] # Vector{Ti}(undef, n)
        csrrowptr  = sparse_preallocation[5] # Vector{Ti}(undef, m + 1)
        csrcolval  = sparse_preallocation[6] # Vector{Ti}(undef, length(I))
        csrnzval   = sparse_preallocation[7] # Vector{Tv}(undef, length(I))

        resize!(klasttouch, m3_rows)
        resize!(csrrowptr, m3_cols + 1)
        resize!(csrcolval, length(J))
        resize!(csrnzval, length(J))

        out = sparse!(J, I, V, m3_cols, m3_rows, +, klasttouch, csrrowptr, csrcolval, csrnzval, J, I, V)
        # out = sparse!(J, I, V, m3_cols, m3_rows)
    else
        klasttouch = sparse_preallocation[4] # Vector{Ti}(undef, n)
        csrrowptr  = sparse_preallocation[5] # Vector{Ti}(undef, m + 1)
        csrcolval  = sparse_preallocation[6] # Vector{Ti}(undef, length(I))
        csrnzval   = sparse_preallocation[7] # Vector{Tv}(undef, length(I))

        resize!(klasttouch, m3_cols)
        resize!(csrrowptr, m3_rows + 1)
        resize!(csrcolval, length(I))
        resize!(csrnzval, length(I))

        out = sparse!(I, J, V, m3_rows, m3_cols, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
        # out = sparse!(I, J, V, m3_rows, m3_cols)
    end

    return out
end


function kron³(A::AbstractSparseMatrix{T}, M₃::third_order_auxilliary_matrices) where T <: Real
    rows, cols, vals = findnz(A)

    # Dictionary to accumulate sums of values for each coordinate
    result_dict = Dict{Tuple{Int, Int}, T}()

    # Using a single iteration over non-zero elements
    nvals = length(vals)

    lk = ReentrantLock()

    Polyester.@batch for i in 1:nvals
    # for i in 1:nvals
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

                    # begin
                    #     lock(lk)
                    #     try
                            if haskey(result_dict, key)
                                result_dict[key] += v1 * v2 * v3
                            else
                                result_dict[key] = v1 * v2 * v3
                            end
                    #     finally
                    #         unlock(lk)
                    #     end
                    # end
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
    return sparse!(result_rows, result_cols, result_vals, size(M₃.𝐂₃, 2), size(M₃.𝐔₃, 1))
end

function A_mult_kron_power_3_B(A::AbstractSparseMatrix{R},
                                B::Union{ℒ.Adjoint{T,Matrix{T}},DenseMatrix{T}}; 
                                tol::AbstractFloat = eps()) where {R <: Real, T <: Real}
    n_row = size(B,1)
    n_col = size(B,2)

    vals = T[]
    rows = Int[]
    cols = Int[]

    Ar, Ac, Av = findnz(A)

    for row in unique(Ar)
        idx_mat, vals_mat = A[row,:] |> findnz

        for col in 1:size(B,2)^3
            col_1, col_3 = divrem((col - 1) % (n_col^2), n_col) .+ 1
            col_2 = ((col - 1) ÷ (n_col^2)) + 1

            mult_val = 0.0

            for (i,idx) in enumerate(idx_mat)
                i_1, i_3 = divrem((idx - 1) % (n_row^2), n_row) .+ 1
                i_2 = ((idx - 1) ÷ (n_row^2)) + 1
                @inbounds mult_val += vals_mat[i] * B[i_1,col_1] * B[i_2,col_2] * B[i_3,col_3]
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
        var_idx = MacroModelling.parse_variables_input_to_index(variables, T) |> sort
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


function get_and_check_observables(𝓂::ℳ, data::KeyedArray{Float64})::Vector{Symbol}
    @assert size(data,1) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    observables = collect(axiskeys(data,1))

    @assert observables isa Vector{String} || observables isa Vector{Symbol}  "Make sure that the data has variables names as rows. They can be either Strings or Symbols."

    observables_symbols = observables isa String_input ? observables .|> Meta.parse .|> replace_indices : observables

    @assert length(setdiff(observables_symbols, 𝓂.var)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(observables_symbols,𝓂.var))

    sort!(observables_symbols)
    
    return observables_symbols
end

function x_kron_II!(buffer::Matrix{T}, x::Vector{T}) where T
    n = length(x)
    m = size(buffer,2)

    # @assert size(buffer, 1) == n^3 "Buffer must have n^2 rows."
    # @assert size(buffer, 2) == n^2 "Buffer must have n columns."

    @turbo for j in 1:m
         for i in 1:n
            buffer[(j - 1) * n + i, j] = x[i]
        end
    end
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


function product_moments(V, ii, nu)::Int
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


function generateSumVectors(vectorLength::Int, totalSum::Int)::Union{Vector{Int}, Vector{ℒ.Adjoint{Int, Vector{Int}}}}
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


function count_ops(expr)::Int
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



function get_relevant_steady_states(𝓂::ℳ, 
                                    algorithm::Symbol;
                                    opts::CalculationOptions = merge_calculation_options())::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    full_NSSS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    full_NSSS[indexin(𝓂.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

    if any(x -> contains(string(x), "◖"), full_NSSS)
        full_NSSS_decomposed = decompose_name.(full_NSSS)
        full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
    end

    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, return_variables_only = true, derivatives = false, 
                                    verbose = opts.verbose,
                                    tol = opts.tol,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³])

    reference_steady_state = [s ∈ 𝓂.exo_present ? 0 : relevant_SS(s) for s in full_NSSS]

    relevant_NSSS = get_steady_state(𝓂, algorithm = :first_order, return_variables_only = true, derivatives = false, 
                                    verbose = opts.verbose,
                                    tol = opts.tol,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³])

    NSSS = [s ∈ 𝓂.exo_present ? 0 : relevant_NSSS(s) for s in full_NSSS]

    SSS_delta = NSSS - reference_steady_state

    return reference_steady_state, NSSS, SSS_delta
end

# compatibility with SymPy
Max = max
Min = min

function simplify(ex::Expr)::Union{Expr,Symbol,Int}
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

function convert_to_ss_equation(eq::Expr)::Expr
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

end # dispatch_doctor

function evaluate_conditions(cond)
    if cond isa Bool
        return cond
    elseif cond isa Expr && cond.head == :call 
        a, b = cond.args[2], cond.args[3]

        if typeof(a) ∉ [Symbol, Number]
            a = eval(a)
        end

        if typeof(b) ∉ [Symbol, Number]
            b = eval(b)
        end
        
        if cond.args[1] == :(==)
            return a == b
        elseif cond.args[1] == :(!=)
            return a != b
        elseif cond.args[1] == :(<)
            return a < b
        elseif cond.args[1] == :(<=)
            return a <= b
        elseif cond.args[1] == :(>)
            return a > b
        elseif cond.args[1] == :(>=)
            return a >= b
        end
        # end
    end
    return nothing
end

function resolve_if_expr(ex::Expr)
    prewalk(ex) do node
        if node isa Expr && (node.head === :if || node.head === :elseif)
            cond = node.args[1]
            then_blk = node.args[2]
            if length(node.args) == 3
                else_blk = node.args[3]
            end
            val = evaluate_conditions(unblock(cond))

            if val === true
                # recurse into the selected branch
                return resolve_if_expr(unblock(then_blk))
            elseif val === false && length(node.args) == 3
                return resolve_if_expr(unblock(else_blk))
            elseif val === false && length(node.args) == 2
                return nothing
            elseif val === false && node.head === :elseif
                return resolve_if_expr(unblock(else_blk))
            end
        end
        return node
    end
end

# function remove_nothing(ex::Expr)
#     postwalk(ex) do node
#         # Only consider call-nodes with exactly two arguments
#         if node isa Expr && node.head === :call && length(node.args) == 3
#             fn, lhs, rhs = node.args
#             lhs2 = unblock(lhs)
#             rhs2 = unblock(rhs)

#             if rhs2 === :(nothing)
#                 # strip the call and recurse to clean deeper
#                 return remove_nothing(lhs2)
#             elseif lhs2 === :(nothing)
#                 return remove_nothing(rhs2)
#             # else
#             #     return remove_nothing(node.args)
#             end
#         end
#         return node
#     end
# end

function contains_equation(expr)
    found = false
    postwalk(expr) do x
        if x isa Expr && x.head == :(=)
            found = true
        end
        return x
    end
    return found
end

function remove_nothing(ex::Expr)
    postwalk(ex) do node
        # Only consider call-expressions
        if node isa Expr && node.head === :call && any(node.args .=== nothing)
            fn = node.args[1]
            # Unblock and collect all the operands
            # raw_args = map(arg -> unblock(arg), node.args[2:end])
            # Drop any nothing
            kept = filter(arg -> !(unblock(arg) === nothing), node.args[2:end])
            if isempty(kept)
                return nothing
            elseif length(kept) == 1
                return kept[1]
            else
            # elseif length(kept) < length(raw_args)
                return Expr(:call, fn, kept...)
            # else
            #     return node
            end
        end
        return node
    end
end

@stable default_mode = "disable" begin
    
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
                            :($(Expr(:ref, Symbol(string(name) * "{" * string(idx) * "}"),time))) :
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ get_symbols(time) ?
                                :($(Expr(:ref, Expr(:curly,name,index), Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            x :
                        x :
                    @capture(x, name_[time_]) ?
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ get_symbols(time) ?
                                :($(Expr(:ref, name, Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            # occursin("{" * string(index_variable) * "}", string(name)) ?
                            #     Expr(:ref, Symbol(replace(string(name), "{" * string(index_variable) * "}" => "◖" * string(idx) * "◗")), time) :
                            x :
                        # occursin("{" * string(index_variable) * "}", string(name)) ?
                        #     Expr(:ref, Symbol(replace(string(name), "{" * string(index_variable) * "}" => "◖" * string(idx) * "◗")), time) :
                        x :
                    x :
                x.head == :if ?
                    length(x.args) > 2 ?
                        Expr(:if,   postwalk(x -> x == index_variable ? idx : x, x.args[1]),
                                    replace_indices_inside_for_loop(x.args[2],index_variable,:([$idx]),false,:+) |> unblock,
                                    replace_indices_inside_for_loop(x.args[3],index_variable,:([$idx]),false,:+) |> unblock) :
                    Expr(:if,   postwalk(x -> x == index_variable ? idx : x, x.args[1]),
                                replace_indices_inside_for_loop(x.args[2],index_variable,:([$idx]),false,:+) |> unblock) :
                @capture(x, name_{index_}) ?
                    index == index_variable ?
                        :($(Symbol(string(name) * "{" * string(idx) * "}"))) :
                    x :
                x :
            @capture(x, name_) ?
                name == index_variable && idx isa Int ?
                    :($idx) :
                x isa Symbol ?
                    occursin("{" * string(index_variable) * "}", string(x)) ?
                Symbol(replace(string(x),  "{" * string(index_variable) * "}" => "{" * string(idx) * "}")) :
                    x :
                x :
            x
        end,
        exxpr))
    end
    
    if concatenate
        return :($(Expr(:call, operator, calls...)))
    else
        return :($(Expr(:block, calls...)))
        # return :($calls...)
        # return calls
    end
end


replace_indices(x::Symbol) = x

replace_indices(x::String) = Symbol(replace(x, "{" => "◖", "}" => "◗"))

replace_indices_in_symbol(x::Symbol) = replace(string(x), "◖" => "{", "◗" => "}")

function replace_indices(exxpr::Expr)::Union{Expr,Symbol}
    postwalk(x -> begin
        x isa Symbol ?
            replace_indices(string(x)) :
        x isa Expr ?
            x.head == :curly ?
                Symbol(string(x.args[1]) * "◖" * string(x.args[2]) * "◗") :
            x :
        x
    end, exxpr)
end

function write_out_for_loops(arg::Expr)::Expr
    postwalk(x -> begin
                    x = flatten(unblock(x))
                    x isa Expr ?
                        x.head == :for ?
                            x.args[2] isa Array ?
                                length(x.args[2]) >= 1 ?
                                    x.args[1].head == :block ?
                                        # begin println("here"); 
                                        [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] : # end :
                                    # begin println("here2"); 
                                    [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] : # end :
                                x :
                            x.args[2].head ∉ [:(=), :block] ?
                                x.args[1].head == :block ?
                                    # begin println("here3"); 
                                    replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[2].args[1]), 
                                                    (x.args[1].args[2].args[2]),
                                                    true,
                                                    x.args[1].args[1].args[2].value) : # end : # for loop part of equation
                                x.args[2].head == :if ?
                                    contains_equation(x.args[2]) ?
                                        # begin println("here5"); println(x)
                                        replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                            Symbol(x.args[1].args[1]), 
                                                            (x.args[1].args[2]),
                                                            false,
                                                            :+) : # end : # for loop part of equation
                                    # begin println("here6"); println(x)
                                    replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[1]), 
                                                        (x.args[1].args[2]),
                                                        true,
                                                        :+) : # end : # for loop part of equation
                                # begin println("here4"); println(x)
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    (x.args[1].args[2]),
                                                    true,
                                                    :+) : # end : # for loop part of equation
                            x.args[1].head == :block ?
                                # begin println("here5"); 
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[2].args[1]), 
                                                    (x.args[1].args[2].args[2]),
                                                    false,
                                                    x.args[1].args[1].args[2].value) : # end :
                                                # end 
                                                # : # for loop part of equation
                            # begin println(x); 
                            # begin println("here7"); println(x)
                            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                            Symbol(x.args[1].args[1]), 
                                            (x.args[1].args[2]),
                                            false,
                                            :+) : # end :
                                            # println(out); 
                                            # return out end 
                                            # :
                        x :
                    x
                end,
    arg) #|> unblock |> flatten
end

# function parse_for_loops(equations_block)
#     eqs = Expr[]  # Initialize an empty array to collect expressions

#     # Define a helper recursive function
#     function recurse(arg)
#             if arg isa Expr
#                 if arg.head == :block
#                     for b in arg.args
#                         if b isa Expr
#                             # If the result is an Expr, process and add to eqs
#                             push!(eqs, unblock(replace_indices(b)))
#                         elseif b isa Array
#                             recurse(b)
#                         end
#                     end
#                 end
#             elseif arg isa Array
#                 # If the result is an Array, iterate and recurse
#                 for B in arg
#                     println((B))
#                     recurse(B)
#                 end
#             end
#     end

#     for arg in equations_block.args
#         if isa(arg,Expr)
#             parsed_eqs = write_out_for_loops(arg)
#             recurse(parsed_eqs)
#         end
#     end

#     # Return the collected expressions as a block
#     return Expr(:block, eqs...)
# end


function parse_for_loops(equations_block)::Expr
    eqs = Expr[]
    for arg in equations_block.args
        if isa(arg,Expr)
            parsed_eqs = write_out_for_loops(arg)
            # println(parsed_eqs)
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
    return Expr(:block,eqs...) |> flatten
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



function create_symbols_eqs!(𝓂::ℳ)::symbolics
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
            if avoid_solve || count_ops(Meta.parse(string(ss_equations[i]))) > 15
                soll = nothing
            else
                soll = solve_symbolically(ss_equations[i],var_to_solve_for)
            end

            if isnothing(soll)
                continue
            end
            
            if isempty(soll) || soll == SPyPyC.Sym{PythonCall.Core.Py}[0] # take out variable if it is redundant from that euation only
                push!(Symbolics.var_redundant_list[i],var_to_solve_for)
                ss_equations[i] = replace_with_one(ss_equations[i], var_to_solve_for) # replace euler constant as it is not translated to julia properly
            end

        end
    end

end

function write_block_solution!(𝓂, 
                                SS_solve_func, 
                                vars_to_solve, 
                                eqs_to_solve, 
                                relevant_pars_across,
                                NSSS_solver_cache_init_tmp, 
                                eq_idx_in_block_to_solve, 
                                atoms_in_equations_list;
                                cse = true,
                                skipzeros = true, 
                                density_threshold::Float64 = .1,
                                min_length::Int = 10000)

    # ➕_vars = Symbol[]
    unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()

    vars_to_exclude = [vcat(Symbol.(vars_to_solve), 𝓂.➕_vars),Symbol[]]

    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_robust_to_domain_errors(Meta.parse.(string.(eqs_to_solve)), vars_to_exclude, 𝓂.bounds, 𝓂.➕_vars, unique_➕_eqs)


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

    # guess = Expr[]
    # untransformed_guess = Expr[]
    result = Expr[]
    # calib_pars = Expr[]

    calib_pars_input = Symbol[]

    relevant_pars = union(intersect(reduce(union, vcat(𝓂.par_list_aux_SS, 𝓂.par_calib_list)[eq_idx_in_block_to_solve]), syms_in_eqs),intersect(syms_in_eqs, 𝓂.➕_vars))
    
    union!(relevant_pars_across, relevant_pars)

    sorted_vars = sort(Symbol.(vars_to_solve))

    for (i, parss) in enumerate(sorted_vars) 
        # push!(guess,:($parss = guess[$i]))
        # push!(untransformed_guess,:($parss = undo_transform(guess[$i],transformation_level)))
        push!(result,:($parss = sol[$i]))
    end

    iii = 1
    for parss in union(𝓂.parameters, 𝓂.parameters_as_function_of_parameters)
        if :($parss) ∈ relevant_pars
            # push!(calib_pars, :($parss = parameters_and_solved_vars[$iii]))
            push!(calib_pars_input, :($parss))
            iii += 1
        end
    end

    # separate out auxilliary variables (nonnegativity)
    # nnaux = []
    # nnaux_linear = []
    # nnaux_error = []
    # push!(nnaux_error, :(aux_error = 0))
    solved_vals = Expr[]
    # solved_vals_in_place = Expr[]
    # partially_solved_block = Expr[]

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
        # push!(solved_vals_in_place, :(ℰ[$i] = $(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))))
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

    # other_vars = Expr[]
    other_vars_input = Symbol[]
    other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars),
                                        sort(𝓂.solved_vars[end]) ),
                                union(syms_in_eqs, other_vrs_eliminated_by_sympy ) )
                                # union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

    for var in other_vrs
        # push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
        push!(other_vars_input,:($(var)))
        iii += 1
    end

    parameters_and_solved_vars = vcat(calib_pars_input, other_vrs)

    ng = length(sorted_vars)
    np = length(parameters_and_solved_vars)
    nd = length(ss_and_aux_equations_dep)
    nx = iii - 1

    Symbolics.@variables 𝔊[1:ng] 𝔓[1:np]


    parameter_dict = Dict{Symbol, Symbol}()
    back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
    aux_vars = Symbol[]
    aux_expr = []


    for (i,v) in enumerate(sorted_vars)
        push!(parameter_dict, v => :($(Symbol("𝔊_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔊_$i"))), @__MODULE__) => 𝔊[i])
    end

    for (i,v) in enumerate(parameters_and_solved_vars)
        push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
    end

    for (i,v) in enumerate(ss_and_aux_equations_dep)
        push!(aux_vars, v.args[1])
        push!(aux_expr, v.args[2])
    end
    
    aux_replacements = Dict{Symbol,Any}()
    for (i,x) in enumerate(aux_vars)
        replacement = Dict(x => aux_expr[i])
        for ii in i+1:length(aux_vars)
            aux_expr[ii] = replace_symbols(aux_expr[ii], replacement)
        end
        push!(aux_replacements, x => aux_expr[i])
    end
    # aux_replacements = Dict{Symbol,Any}(aux_vars .=> aux_expr)

    replaced_solved_vals = solved_vals |> 
        x -> replace_symbols.(x, Ref(aux_replacements)) |> 
        x -> replace_symbols.(x, Ref(parameter_dict)) |> 
        x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
        x -> Symbolics.substitute.(x, Ref(back_to_array_dict))

    lennz = length(replaced_solved_vals)

    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, calc_block! = Symbolics.build_function(replaced_solved_vals, 𝔊, 𝔓,
                                                cse = cse, 
                                                skipzeros = skipzeros, 
                                                # nanmath = false,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    # 𝐷 = zeros(Symbolics.Num, nd)

    # ϵᵃ = zeros(nd)

    # calc_block_aux!(𝐷, 𝔊, 𝔓)

    ϵˢ = zeros(Symbolics.Num, ng)

    ϵ = zeros(ng)

    # calc_block!(ϵˢ, 𝔊, 𝔓, 𝐷)

    ∂block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals, 𝔊) # nϵ x nx

    lennz = nnz(∂block_∂parameters_and_solved_vars)

    if (lennz / length(∂block_∂parameters_and_solved_vars) > density_threshold) || (length(∂block_∂parameters_and_solved_vars) < min_length)
        derivatives_mat = convert(Matrix, ∂block_∂parameters_and_solved_vars)
        buffer = zeros(Float64, size(∂block_∂parameters_and_solved_vars))
    else
        derivatives_mat = ∂block_∂parameters_and_solved_vars
        buffer = similar(∂block_∂parameters_and_solved_vars, Float64)
        buffer.nzval .= 1
    end

    chol_buff = buffer * buffer'

    chol_buff += ℒ.I

    prob = 𝒮.LinearProblem(chol_buff, ϵ, 𝒮.CholeskyFactorization())

    chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization())

    prob = 𝒮.LinearProblem(buffer, ϵ, 𝒮.LUFactorization())

    lu_buffer = 𝒮.init(prob, 𝒮.LUFactorization())

    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end
    
    _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔊, 𝔓,
                                                cse = cse, 
                                                skipzeros = skipzeros, 
                                                # nanmath = false,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}


    Symbolics.@variables 𝔊[1:ng+nx]

    ext_diff = Symbolics.Num[]
    for i in 1:nx
        push!(ext_diff, 𝔓[i] - 𝔊[ng + i])
    end
    replaced_solved_vals_ext = vcat(replaced_solved_vals, ext_diff)

    _, calc_ext_block! = Symbolics.build_function(replaced_solved_vals_ext, 𝔊, 𝔓,
                                                cse = cse, 
                                                skipzeros = skipzeros, 
                                                # nanmath = false,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    ϵᵉ = zeros(ng + nx)
    
    # ϵˢᵉ = zeros(Symbolics.Num, ng + nx)

    # calc_block_aux!(𝐷, 𝔊, 𝔓)

    # Evaluate the function symbolically
    # calc_ext_block!(ϵˢᵉ, 𝔊, 𝔓, 𝐷)

    ∂ext_block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals_ext, 𝔊) # nϵ x nx

    lennz = nnz(∂ext_block_∂parameters_and_solved_vars)

    if (lennz / length(∂ext_block_∂parameters_and_solved_vars) > density_threshold) || (length(∂ext_block_∂parameters_and_solved_vars) < min_length)
        derivatives_mat_ext = convert(Matrix, ∂ext_block_∂parameters_and_solved_vars)
        ext_buffer = zeros(Float64, size(∂ext_block_∂parameters_and_solved_vars))
    else
        derivatives_mat_ext = ∂ext_block_∂parameters_and_solved_vars
        ext_buffer = similar(∂ext_block_∂parameters_and_solved_vars, Float64)
        ext_buffer.nzval .= 1
    end

    ext_chol_buff = ext_buffer * ext_buffer'

    ext_chol_buff += ℒ.I

    prob = 𝒮.LinearProblem(ext_chol_buff, ϵᵉ, 𝒮.CholeskyFactorization())

    ext_chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization())

    prob = 𝒮.LinearProblem(ext_buffer, ϵᵉ, 𝒮.LUFactorization())

    ext_lu_buffer = 𝒮.init(prob, 𝒮.LUFactorization())

    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end
    
    _, ext_func_exprs = Symbolics.build_function(derivatives_mat_ext, 𝔊, 𝔓,
                                                cse = cse, 
                                                skipzeros = skipzeros, 
                                                # nanmath = false,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    
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
            
    # n_block = length(𝓂.ss_solve_blocks) + 1 
    n_block = length(𝓂.ss_solve_blocks_in_place) + 1   
        
    push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

    push!(SS_solve_func,:(solution = block_solver(params_and_solved_vars,
                                                            $(n_block), 
                                                            𝓂.ss_solve_blocks_in_place[$(n_block)], 
                                                            # 𝓂.ss_solve_blocks[$(n_block)], 
                                                            # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                            # f, 
                                                            inits,
                                                            lbs, 
                                                            ubs,
                                                            solver_parameters,
                                                            fail_fast_solvers_only,
                                                            cold_start,
                                                            verbose)))
                                                            
    push!(SS_solve_func,:(iters += solution[2][2])) 
    push!(SS_solve_func,:(solution_error += solution[2][1])) 
    push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed after solving block with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))

    if length(ss_and_aux_equations_error) > 0
        push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, ss_and_aux_equations_error...))))
        push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for aux variables with error $(solution_error)") end; scale = scale * .3 + solved_scale * .7; continue end))
    end

    push!(SS_solve_func,:(sol = solution[1]))

    push!(SS_solve_func,:($(result...)))   

    push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
    push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

    
    push!(𝓂.ss_solve_blocks_in_place, ss_solve_block(
            function_and_jacobian(calc_block!::Function, ϵ, func_exprs::Function, buffer, chol_buffer, lu_buffer),
            function_and_jacobian(calc_ext_block!::Function, ϵᵉ, ext_func_exprs::Function, ext_buffer, ext_chol_buffer, ext_lu_buffer)
        )
    )
    
    return nothing
end




function partial_solve(eqs_to_solve::Vector{E}, vars_to_solve::Vector{T}, incidence_matrix_subset; avoid_solve::Bool = false)::Tuple{Vector{T}, Vector{T}, Vector{E}, Vector{T}} where {E, T}
    for n in length(eqs_to_solve)-1:-1:2
        for eq_combo in combinations(1:length(eqs_to_solve), n)
            var_indices_to_select_from = findall([sum(incidence_matrix_subset[:,eq_combo],dims = 2)...] .> 0)

            var_indices_in_remaining_eqs = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),eq_combo)],dims = 2)...] .> 0) 

            for var_combo in combinations(var_indices_to_select_from, n)
                remaining_vars_in_remaining_eqs = setdiff(var_indices_in_remaining_eqs, var_combo)
                # println("Solving for: ",vars_to_solve[var_combo]," in: ",eqs_to_solve[eq_combo])
                if length(remaining_vars_in_remaining_eqs) == length(eqs_to_solve) - n # not sure whether this condition needs to be there. could be because if the last remaining vars not solved for in the block is not present in the remaining block he will not be able to solve it for the same reasons he wasnt able to solve the unpartitioned block 
                    if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve[eq_combo]))) > 15
                        soll = nothing
                    else
                        soll = solve_symbolically(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
                    end
                    
                    if !(isnothing(soll) || isempty(soll))
                        soll_collected = collect(values(soll))
                        
                        return (vars_to_solve[setdiff(1:length(eqs_to_solve),var_combo)],
                                vars_to_solve[var_combo],
                                eqs_to_solve[setdiff(1:length(eqs_to_solve),eq_combo)],
                                soll_collected)
                    end
                end
            end
        end
    end
    
    return (T[], T[], E[], T[])
end



function make_equation_robust_to_domain_errors(eqs,#::Vector{Union{Symbol,Expr}}, 
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
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12), min(bounds[x.args[2]][2], 600)) : (-1e12, 600)
                                    x 
                                end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(600,max(-1e12,$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(600,max(-1e12,$(x.args[2]))))) 
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end
                                        
                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], -1e12), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 600)) : (-1e12, 600)
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
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(600,max(-1e12,$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(600,max(-1e12,$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end
                                            
                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], -1e12), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 600)) : (-1e12, 600)
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



function replace_symbols(exprs::T, remap::Dict{Symbol,S}) where {T,S}
    postwalk(node ->
          if node isa Symbol && haskey(remap, node)
              remap[node]
          else
              node
          end, 
          exprs)
end

function write_ss_check_function!(𝓂::ℳ;
                                    cse = true,
                                    skipzeros = true, 
                                    density_threshold::Float64 = .1,
                                    min_length::Int = 10000)
    unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)

    ss_equations = vcat(𝓂.ss_equations, 𝓂.calibration_equations)



    np = length(𝓂.parameters)
    nu = length(unknowns)
    # nc = length(𝓂.calibration_equations_no_var)

    Symbolics.@variables 𝔓[1:np] 𝔘[1:nu]# ℭ[1:nc]

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
        # push!(parameter_dict, v.args[1] => :($(Symbol("ℭ_$i"))))
        # push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("ℭ_$i"))), @__MODULE__) => ℭ[i])
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
                                                # nanmath = false, 
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}


    𝓂.SS_check_func = func_exprs


    # SS_and_pars = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))

    # eqs = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

    # nx = length(𝓂.parameter_values)

    # np = length(SS_and_pars)

    nϵˢ = length(ss_equations)

    # nc = length(𝓂.calibration_equations_no_var)

    # Symbolics.@variables 𝔛¹[1:nx] 𝔓¹[1:np]

    # ϵˢ = zeros(Symbolics.Num, nϵˢ)

    # calib_vals = zeros(Symbolics.Num, nc)

    # 𝓂.SS_calib_func(calib_vals, 𝔓)

    # 𝓂.SS_check_func(ϵˢ, 𝔓, 𝔘, calib_vals)

    ∂SS_equations_∂parameters = Symbolics.sparsejacobian(ss_equations_sub, 𝔓) # nϵ x nx

    lennz = nnz(∂SS_equations_∂parameters)

    if (lennz / length(∂SS_equations_∂parameters) > density_threshold) || (length(∂SS_equations_∂parameters) < min_length)
        derivatives_mat = convert(Matrix, ∂SS_equations_∂parameters)
        buffer = zeros(Float64, size(∂SS_equations_∂parameters))
    else
        derivatives_mat = ∂SS_equations_∂parameters
        buffer = similar(∂SS_equations_∂parameters, Float64)
        buffer.nzval .= 0
    end

    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end
    
    _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔓, 𝔘, 
                                                cse = cse, 
                                                skipzeros = skipzeros,
                                                # nanmath = false, 
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    𝓂.∂SS_equations_∂parameters = buffer, func_exprs



    ∂SS_equations_∂SS_and_pars = Symbolics.sparsejacobian(ss_equations_sub, 𝔘) # nϵ x nx

    lennz = nnz(∂SS_equations_∂SS_and_pars)

    if (lennz / length(∂SS_equations_∂SS_and_pars) > density_threshold) || (length(∂SS_equations_∂SS_and_pars) < min_length)
        derivatives_mat = convert(Matrix, ∂SS_equations_∂SS_and_pars)
        buffer = zeros(Float64, size(∂SS_equations_∂SS_and_pars))
    else
        derivatives_mat = ∂SS_equations_∂SS_and_pars
        buffer = similar(∂SS_equations_∂SS_and_pars, Float64)
        buffer.nzval .= 0
    end

    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔓, 𝔘, 
                                                cse = cse, 
                                                skipzeros = skipzeros, 
                                                # nanmath = false,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    𝓂.∂SS_equations_∂SS_and_pars = buffer, func_exprs

    return nothing
end


function solve_steady_state!(𝓂::ℳ, symbolic_SS, Symbolics::symbolics; verbose::Bool = false, avoid_solve::Bool = false)
    write_ss_check_function!(𝓂)

    unknowns = union(Symbolics.calibration_equations_parameters, Symbolics.vars_in_ss_equations)

    @assert length(unknowns) <= length(Symbolics.ss_equations) + length(Symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = spzeros(Int,length(unknowns),length(unknowns))

    eq_list = vcat(union.(setdiff.(union.(Symbolics.var_list,
                                        Symbolics.ss_list),
                                    Symbolics.var_redundant_list),
                            Symbolics.par_list),
                    union.(Symbolics.ss_calib_list,
                            Symbolics.par_calib_list))

    for (i,u) in enumerate(unknowns)
        for (k,e) in enumerate(eq_list)
            incidence_matrix[i,k] = u ∈ e
        end
    end

    Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(incidence_matrix)
    R̂ = Int[]
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
            var_to_solve_for = unknowns[vars[:,vars[2,:] .== n][1]]

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
                push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for min max terms in equations with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                eq_to_solve = eval(minmax_fixed_eqs)
            end
            
            if avoid_solve || count_ops(Meta.parse(string(eq_to_solve))) > 15
                soll = nothing
            else
                soll = solve_symbolically(eq_to_solve,var_to_solve_for)
            end

            if isnothing(soll) || isempty(soll)
                println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
                
                eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

                write_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)
                # write_domain_safe_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
            elseif soll[1].is_number == true
                ss_equations = [replace_symbolic(eq, var_to_solve_for, soll[1]) for eq in ss_equations]
                
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
                    push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for analytical aux variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                    
                    unique_➕_eqs[𝓂.solved_vals[end]] = 𝓂.solved_vars[end]
                else
                    vars_to_exclude = [vcat(Symbol.(var_to_solve_for), 𝓂.➕_vars), Symbol[]]
                    
                    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_robust_to_domain_errors([𝓂.solved_vals[end]], vars_to_exclude, 𝓂.bounds, 𝓂.➕_vars, unique_➕_eqs)
    
                    if length(vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)) > 0
                        push!(SS_solve_func,vcat(ss_and_aux_equations, ss_and_aux_equations_dep)...)
                        push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)...))))
                        push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for analytical variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                    end
                    
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(rewritten_eqs[1])))
                end

                if haskey(𝓂.bounds, 𝓂.solved_vars[end]) && 𝓂.solved_vars[end] ∉ 𝓂.➕_vars
                    push!(SS_solve_func,:(solution_error += abs(min(max($(𝓂.bounds[𝓂.solved_vars[end]][1]), $(𝓂.solved_vars[end])), $(𝓂.bounds[𝓂.solved_vars[end]][2])) - $(𝓂.solved_vars[end]))))
                    push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for bounded variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                end
            end
        else
            vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]

            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false
            
            if symbolic_SS
                if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve))) > 15
                    soll = nothing
                else
                    soll = solve_symbolically(eqs_to_solve,vars_to_solve)
                end

                if isnothing(soll) || isempty(soll) || length(intersect((union(SPyPyC.free_symbols.(collect(values(soll)))...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
                    if verbose println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.") end

                    numerical_sol = true
                else
                    if verbose println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.") end
                    
                    atoms = reduce(union,map(x->x.atoms(),collect(values(soll))))

                    for a in atoms push!(atoms_in_equations, Symbol(a)) end
                    
                    for vars in vars_to_solve
                        push!(𝓂.solved_vars,Symbol(vars))
                        push!(𝓂.solved_vals,Meta.parse(string(soll[vars]))) #using convert(Expr,x) leads to ugly expressions

                        push!(atoms_in_equations_list, Set(Symbol.(soll[vars].atoms())))
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

    unknwns = Symbol.(unknowns)

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
        push!(SS_solve_func,:(if (current_best > 1e-8) && (solution_error < tol.NSSS_acceptance_tol) && (scale == 1)
                                    reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
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

    solve_exp = :(function solve_SS(initial_parameters::Vector{Real}, 
                                    𝓂::ℳ,
                                    # fail_fast_solvers_only::Bool, 
                                    tol::Tolerances,
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})
                    initial_parameters = typeof(initial_parameters) == Vector{Float64} ? initial_parameters : ℱ.value.(initial_parameters)

                    initial_parameters_tmp = copy(initial_parameters)

                    parameters = copy(initial_parameters)
                    params_flt = copy(initial_parameters)
                    
                    current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - initial_parameters)
                    closest_solution_init = 𝓂.NSSS_solver_cache[end]
                    
                    for pars in 𝓂.NSSS_solver_cache
                        copy!(initial_parameters_tmp, pars[end])

                        ℒ.axpy!(-1,initial_parameters,initial_parameters_tmp)

                        latest = sum(abs2,initial_parameters_tmp)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end

                    # closest_solution = copy(closest_solution_init)
                    # solution_error = 1.0
                    # iters = 0
                    range_iters = 0
                    solution_error = 1.0
                    solved_scale = 0
                    # range_length = [ 1, 2, 4, 8,16,32,64,128,1024]
                    scale = 1.0

                    NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
                    push!(NSSS_solver_cache_scale, closest_solution_init)
                    # fail_fast_solvers_only = true
                    while range_iters <= (cold_start ? 1 : 500) && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
                        range_iters += 1
                        fail_fast_solvers_only = range_iters > 1 ? true : false

                        if abs(solved_scale - scale) < 1e-2
                            # println(NSSS_solver_cache_scale[end])
                            break 
                        end

                        # println("i: $range_iters - scale: $scale - solved_scale: $solved_scale")
                        # println(closest_solution[end])
                    # for range_ in range_length
                        # rangee = range(0,1,range_+1)
                        # for scale in rangee[2:end]
                            # scale = 6*scale^5 - 15*scale^4 + 10*scale^3 # smootherstep

                            # if scale <= solved_scale continue end

                            
                            current_best = sum(abs2,NSSS_solver_cache_scale[end][end] - initial_parameters)
                            closest_solution = NSSS_solver_cache_scale[end]

                            for pars in NSSS_solver_cache_scale
                                copy!(initial_parameters_tmp, pars[end])
                                
                                ℒ.axpy!(-1,initial_parameters,initial_parameters_tmp)

                                latest = sum(abs2,initial_parameters_tmp)

                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end

                            # println(closest_solution)

                            if all(isfinite,closest_solution[end]) && initial_parameters != closest_solution_init[end]
                                parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
                            else
                                parameters = copy(initial_parameters)
                            end
                            params_flt = parameters
                            
                            # println(parameters)

                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.calibration_equations_no_var...)
                            NSSS_solver_cache_tmp = []
                            solution_error = 0.0
                            iters = 0
                            $(SS_solve_func...)

                            if solution_error < tol.NSSS_acceptance_tol
                                # println("solved for $scale; $range_iters")
                                solved_scale = scale
                                if scale == 1
                                    # return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                    # NSSS_solution = [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)]
                                    # NSSS_solution[abs.(NSSS_solution) .< 1e-12] .= 0 # doesnt work with Zygote
                                    return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                                else
                                    reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                                end

                                if scale > .95
                                    scale = 1
                                else
                                    # scale = (scale + 1) / 2
                                    scale = scale * .4 + .6
                                end
                            # else
                            #     println("no sol")
                            #     scale  = (scale + solved_scale) / 2
                            #     println("scale $scale")
                            # elseif scale == 1 && range_ == range_length[end]
                            #     return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                            end
                    #     end
                    end
                    return [0.0], (1, 0)
                end)

    𝓂.SS_solve_func = @RuntimeGeneratedFunction(solve_exp)
    # 𝓂.SS_solve_func = eval(solve_exp)

    return nothing
end




function solve_steady_state!(𝓂::ℳ;
                            cse = true,
                            skipzeros = true,
                            density_threshold::Float64 = .1,
                            min_length::Int = 1000,
                            verbose::Bool = false)
    write_ss_check_function!(𝓂)

    unknowns = union(𝓂.vars_in_ss_equations, 𝓂.calibration_equations_parameters)

    @assert length(unknowns) <= length(𝓂.ss_aux_equations) + length(𝓂.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = spzeros(Int,length(unknowns),length(unknowns))

    eq_list = vcat(union.(union.(𝓂.var_list_aux_SS,
                                        𝓂.ss_list_aux_SS),
                            𝓂.par_list_aux_SS),
                    union.(𝓂.ss_calib_list,
                            𝓂.par_calib_list))

    for (i,u) in enumerate(unknowns)
        for (k,e) in enumerate(eq_list)
            incidence_matrix[i,k] = u ∈ e
        end
    end

    Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(incidence_matrix)
    R̂ = Int[]
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
        vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]

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

        # calib_pars = []
        calib_pars_input = []
        relevant_pars = reduce(union,vcat(𝓂.par_list_aux_SS,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
        relevant_pars_across = union(relevant_pars_across,relevant_pars)
        
        iii = 1
        for parss in union(𝓂.parameters,𝓂.parameters_as_function_of_parameters)
            # valss   = 𝓂.parameter_values[i]
            if :($parss) ∈ relevant_pars
                # push!(calib_pars,:($parss = parameters_and_solved_vars[$iii]))
                push!(calib_pars_input,:($parss))
                iii += 1
            end
        end


        # guess = Expr[]
        # untransformed_guess = Expr[]
        result = Expr[]
        sorted_vars = sort(𝓂.solved_vars[end])
        # sorted_vars = sort(setdiff(𝓂.solved_vars[end],𝓂.➕_vars))
        for (i, parss) in enumerate(sorted_vars) 
            # push!(guess,:($parss = guess[$i]))
            # push!(untransformed_guess,:($parss = undo_transform(guess[$i],transformation_level)))
            push!(result,:($parss = sol[$i]))
        end

        
        # separate out auxilliary variables (nonnegativity)
        nnaux = []
        # nnaux_linear = []
        # nnaux_error = []
        # push!(nnaux_error, :(aux_error = 0))
        solved_vals = Expr[]
        # solved_vals_in_place = Expr[]
        
        eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]


        other_vrs_eliminated_by_sympy = Set()

        for (i,val) in enumerate(𝓂.solved_vals[end])
            if typeof(val) ∈ [Symbol,Float64,Int]
                push!(solved_vals,val)
                # push!(solved_vals_in_place, :(ℰ[$i] = $val))
            else
                if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
                    val = vcat(𝓂.ss_aux_equations,𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
                    push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                    push!(other_vrs_eliminated_by_sympy, val.args[2])
                    # push!(nnaux_linear,:($val))
                    push!(solved_vals,:($val))
                    # push!(solved_vals_in_place,:(ℰ[$i] = $val))
                    # push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
                else
                    push!(solved_vals,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
                    # push!(solved_vals_in_place, :(ℰ[$i] = $(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))))
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
            # nnaux_linear = nnaux_linear[QQ]
        end


        # other_vars = []
        other_vars_input = []
        # other_vars_inverse = []
        other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars),
                                            sort(𝓂.solved_vars[end]) ),
                                union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

        for var in other_vrs
            # var_idx = findfirst(x -> x == var, union(𝓂.var,𝓂.calibration_equations_parameters))
            # push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
            push!(other_vars_input,:($(var)))
            iii += 1
            # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
        end

        parameters_and_solved_vars = vcat(calib_pars_input, other_vrs)

        ng = length(sorted_vars)
        np = length(parameters_and_solved_vars)
        nd = 0
        nx = iii - 1
    

        Symbolics.@variables 𝔊[1:ng] 𝔓[1:np]


        parameter_dict = Dict{Symbol, Symbol}()
        back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
        # aux_vars = Symbol[]
        # aux_expr = []
    
    
        for (i,v) in enumerate(sorted_vars)
            push!(parameter_dict, v => :($(Symbol("𝔊_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔊_$i"))), @__MODULE__) => 𝔊[i])
        end
    
        for (i,v) in enumerate(parameters_and_solved_vars)
            push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
        end
    
        # for (i,v) in enumerate(ss_and_aux_equations_dep)
        #     push!(aux_vars, v.args[1])
        #     push!(aux_expr, v.args[2])
        # end
    
        # aux_replacements = Dict(aux_vars .=> aux_expr)
    
        replaced_solved_vals = solved_vals |> 
            # x -> replace_symbols.(x, Ref(aux_replacements)) |> 
            x -> replace_symbols.(x, Ref(parameter_dict)) |> 
            x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
            x -> Symbolics.substitute.(x, Ref(back_to_array_dict))
    
        lennz = length(replaced_solved_vals)
    
        if lennz > 1500
            parallel = Symbolics.ShardedForm(1500,4)
        else
            parallel = Symbolics.SerialForm()
        end
    
        _, calc_block! = Symbolics.build_function(replaced_solved_vals, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    
        # 𝐷 = zeros(Symbolics.Num, nd)
    
        # ϵᵃ = zeros(nd)
    
        # calc_block_aux!(𝐷, 𝔊, 𝔓)
    
        ϵˢ = zeros(Symbolics.Num, ng)
    
        ϵ = zeros(ng)
    
        # calc_block!(ϵˢ, 𝔊, 𝔓, 𝐷)
    
        ∂block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals, 𝔊) # nϵ x nx
    
        lennz = nnz(∂block_∂parameters_and_solved_vars)
    
        if (lennz / length(∂block_∂parameters_and_solved_vars) > density_threshold) || (length(∂block_∂parameters_and_solved_vars) < min_length)
            derivatives_mat = convert(Matrix, ∂block_∂parameters_and_solved_vars)
            buffer = zeros(Float64, size(∂block_∂parameters_and_solved_vars))
        else
            derivatives_mat = ∂block_∂parameters_and_solved_vars
            buffer = similar(∂block_∂parameters_and_solved_vars, Float64)
            buffer.nzval .= 1
        end
    
        chol_buff = buffer * buffer'

        chol_buff += ℒ.I

        prob = 𝒮.LinearProblem(chol_buff, ϵ, 𝒮.CholeskyFactorization())

        chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization())

        prob = 𝒮.LinearProblem(buffer, ϵ, 𝒮.LUFactorization())

        lu_buffer = 𝒮.init(prob, 𝒮.LUFactorization())

        if lennz > 1500
            parallel = Symbolics.ShardedForm(1500,4)
        else
            parallel = Symbolics.SerialForm()
        end
        
        _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    
    
        Symbolics.@variables 𝔊[1:ng+nx]
    
        ext_diff = Symbolics.Num[]
        for i in 1:nx
            push!(ext_diff, 𝔓[i] - 𝔊[ng + i])
        end
        replaced_solved_vals_ext = vcat(replaced_solved_vals, ext_diff)
    
        _, calc_ext_block! = Symbolics.build_function(replaced_solved_vals_ext, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    
        ϵᵉ = zeros(ng + nx)
        
        # ϵˢᵉ = zeros(Symbolics.Num, ng + nx)
    
        # calc_block_aux!(𝐷, 𝔊, 𝔓)
    
        # Evaluate the function symbolically
        # calc_ext_block!(ϵˢᵉ, 𝔊, 𝔓, 𝐷)
    
        ∂ext_block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals_ext, 𝔊) # nϵ x nx
    
        lennz = nnz(∂ext_block_∂parameters_and_solved_vars)
    
        if (lennz / length(∂ext_block_∂parameters_and_solved_vars) > density_threshold) || (length(∂ext_block_∂parameters_and_solved_vars) < min_length)
            derivatives_mat_ext = convert(Matrix, ∂ext_block_∂parameters_and_solved_vars)
            ext_buffer = zeros(Float64, size(∂ext_block_∂parameters_and_solved_vars))
        else
            derivatives_mat_ext = ∂ext_block_∂parameters_and_solved_vars
            ext_buffer = similar(∂ext_block_∂parameters_and_solved_vars, Float64)
            ext_buffer.nzval .= 1
        end
    
        ext_chol_buff = ext_buffer * ext_buffer'

        ext_chol_buff += ℒ.I

        prob = 𝒮.LinearProblem(ext_chol_buff, ϵᵉ, 𝒮.CholeskyFactorization())

        ext_chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization())

        prob = 𝒮.LinearProblem(ext_buffer, ϵᵉ, 𝒮.LUFactorization())

        ext_lu_buffer = 𝒮.init(prob, 𝒮.LUFactorization())

        if lennz > 1500
            parallel = Symbolics.ShardedForm(1500,4)
        else
            parallel = Symbolics.SerialForm()
        end
        
        _, ext_func_exprs = Symbolics.build_function(derivatives_mat_ext, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    


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
                                                                𝓂.ss_solve_blocks_in_place[$(n_block)], 
                                                                # 𝓂.ss_solve_blocks[$(n_block)], 
                                                                # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                                # f, 
                                                                inits,
                                                                lbs, 
                                                                ubs,
                                                                solver_parameters,
                                                                fail_fast_solvers_only,
                                                                cold_start,
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


        push!(𝓂.ss_solve_blocks_in_place, 
            ss_solve_block(
                function_and_jacobian(calc_block!::Function, ϵ, func_exprs::Function, buffer, chol_buffer, lu_buffer),
                function_and_jacobian(calc_ext_block!::Function, ϵᵉ, ext_func_exprs::Function, ext_buffer, ext_chol_buffer, ext_lu_buffer)
            )
        )

        n_block += 1
        
        n -= 1
    end

    push!(NSSS_solver_cache_init_tmp,[Inf])
    push!(NSSS_solver_cache_init_tmp,fill(Inf,length(𝓂.parameters)))
    push!(𝓂.NSSS_solver_cache,NSSS_solver_cache_init_tmp)

    unknwns = Symbol.(unknowns)

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

    push!(SS_solve_func,:(if (current_best > 1e-8) && (solution_error < tol.NSSS_acceptance_tol)
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

    solve_exp = :(function solve_SS(initial_parameters::Vector{Real}, 
                                    𝓂::ℳ, 
                                    tol::Tolerances,
                                    # fail_fast_solvers_only::Bool, 
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})
                    initial_parameters = typeof(initial_parameters) == Vector{Float64} ? initial_parameters : ℱ.value.(initial_parameters)

                    parameters = copy(initial_parameters)
                    params_flt = copy(initial_parameters)
                    
                    current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - initial_parameters)
                    closest_solution_init = 𝓂.NSSS_solver_cache[end]
                    
                    for pars in 𝓂.NSSS_solver_cache
                        latest = sum(abs2,pars[end] - initial_parameters)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end

                    # closest_solution = closest_solution_init
                    # solution_error = 1.0
                    # iters = 0
                    range_iters = 0
                    solution_error = 1.0
                    solved_scale = 0
                    # range_length = [ 1, 2, 4, 8,16,32,64,128,1024]
                    scale = 1.0

                    while range_iters <= 500 && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
                        range_iters += 1
                        fail_fast_solvers_only = range_iters > 1 ? true : false

                    # for range_ in range_length
                        # rangee = range(0,1,range_+1)
                        # for scale in rangee[2:end]
                            # scale = 6*scale^5 - 15*scale^4 + 10*scale^3 # smootherstep

                            # if scale <= solved_scale continue end

                            current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - initial_parameters)
                            closest_solution = 𝓂.NSSS_solver_cache[end]

                            for pars in 𝓂.NSSS_solver_cache
                                latest = sum(abs2,pars[end] - initial_parameters)
                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end

                            # Zero initial value if startin without guess
                            if !isfinite(sum(abs,closest_solution[2]))
                                closest_solution = copy(closest_solution)
                                for i in 1:2:length(closest_solution)
                                    closest_solution[i] = zeros(length(closest_solution[i]))
                                end
                            end

                            # println(closest_solution)

                            if all(isfinite,closest_solution[end]) && initial_parameters != closest_solution_init[end]
                                parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
                            else
                                parameters = copy(initial_parameters)
                            end
                            params_flt = parameters
                            
                            # println(parameters)

                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.calibration_equations_no_var...)
                            NSSS_solver_cache_tmp = []
                            solution_error = 0.0
                            iters = 0
                            $(SS_solve_func...)

                            if solution_error < tol.NSSS_acceptance_tol
                                # println("solved for $scale; $range_iters")
                                solved_scale = scale
                                if scale == 1
                                    # return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                    # NSSS_solution = [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)]
                                    # NSSS_solution[abs.(NSSS_solution) .< 1e-12] .= 0 # doesnt work with Zygote
                                    return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                                else
                                    reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                                end

                                if scale > .95
                                    scale = 1
                                else
                                    # scale = (scale + 1) / 2
                                    scale = scale * .4 + .6
                                end
                            # else
                            #     println("no sol")
                            #     scale  = (scale + solved_scale) / 2
                            #     println("scale $scale")
                            # elseif scale == 1 && range_ == range_length[end]
                            #     return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                            end
                    #     end
                    end
                    return [0.0], (1, 0)
                end)

    𝓂.SS_solve_func = @RuntimeGeneratedFunction(solve_exp)
    # 𝓂.SS_solve_func = eval(solve_exp)

    return nothing
end


function reverse_diff_friendly_push!(x,y)
    @ignore_derivatives push!(x,y)
end

function calculate_SS_solver_runtime_and_loglikelihood(pars::Vector{Float64}, 𝓂::ℳ; tol::Tolerances = Tolerances())::Float64
    log_lik = 0.0
    log_lik -= -sum(pars[1:19])                                 # logpdf of a gamma dist with mean and variance 1
    σ = 5
    log_lik -= -log(σ * sqrt(2 * π)) - (pars[20]^2 / (2 * σ^2)) # logpdf of a normal dist with mean = 0 and variance = 5^2

    pars[1:2] = sort(pars[1:2], rev = true)

    par_inputs = solver_parameters(pars..., 1, 0.0, 2)

    while length(𝓂.NSSS_solver_cache) > 1
        pop!(𝓂.NSSS_solver_cache)
    end

    runtime = @elapsed outmodel = try 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, tol, false, true, [par_inputs]) catch end

    runtime = outmodel isa Tuple{Vector{Float64}, Tuple{Float64, Int64}} ? 
                    (outmodel[2][1] > tol.NSSS_acceptance_tol) || !isfinite(outmodel[2][1]) ? 
                        10 : 
                    runtime : 
                10

    return log_lik / 1e4 + runtime * 1e3
end


function find_SS_solver_parameters!(𝓂::ℳ; maxtime::Int = 120, maxiter::Int = 250000, tol::Tolerances = Tolerances(), verbosity = 0)
    pars = rand(20) .+ 1
    pars[20] -= 1

    lbs = fill(eps(), length(pars))
    lbs[20] = -20

    ubs = fill(100.0,length(pars))
    
    sol = Optim.optimize(x -> calculate_SS_solver_runtime_and_loglikelihood(x, 𝓂, tol = tol), 
                            lbs, ubs, pars, 
                            Optim.SAMIN(verbosity = verbosity, nt = 5, ns = 5), 
                            Optim.Options(time_limit = maxtime, iterations = maxiter))::Optim.MultivariateOptimizationResults

    pars = Optim.minimizer(sol)

    par_inputs = solver_parameters(pars..., 1, 0.0, 2)

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, tol, false, true, [par_inputs])

    if solution_error < tol.NSSS_acceptance_tol
        push!(𝓂.solver_parameters, par_inputs)
        return true
    else 
        return false
    end
end


function select_fastest_SS_solver_parameters!(𝓂::ℳ; tol::Tolerances = Tolerances())
    best_param = 𝓂.solver_parameters[1]

    best_time = Inf

    solved = false

    solved_NSSS = 𝓂.NSSS_solver_cache[end]

    for p in 𝓂.solver_parameters
        total_time = 0.0
        
        for _ in 1:100
            start_time = time()

            while length(𝓂.NSSS_solver_cache) > 1
                pop!(𝓂.NSSS_solver_cache)
            end

            SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, tol, false, true, [p])

            elapsed_time = time() - start_time

            total_time += elapsed_time
            
            if solution_error > tol.NSSS_acceptance_tol
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

    while length(𝓂.NSSS_solver_cache) > 1
        pop!(𝓂.NSSS_solver_cache)
    end

    push!(𝓂.NSSS_solver_cache, solved_NSSS)

    if solved
        pushfirst!(𝓂.solver_parameters, best_param)
    end
end


function solve_ss(SS_optimizer::Function,
                    # ss_solve_blocks::Function,
                    SS_solve_block::ss_solve_block,
                    parameters_and_solved_vars::Vector{T},
                    closest_parameters_and_solved_vars::Vector{T},
                    lbs::Vector{T},
                    ubs::Vector{T},
                    tol::Tolerances,
                    total_iters::Vector{Int},
                    n_block::Int,
                    verbose::Bool,
                    guess::Vector{T},
                    solver_params::solver_parameters,
                    extended_problem::Bool,
                    separate_starting_value::Union{Bool,T})::Tuple{Vector{T}, Vector{Int}, T, T} where T <: AbstractFloat
    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    if separate_starting_value isa Float64
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], fill(separate_starting_value, length(guess))))
        sol_values_init[ubs[1:length(guess)] .<= 1] .= .1 # capture cases where part of values is small
    else
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], [g < 1e12 ? g : solver_params.starting_value for g in guess]))
    end

    sol_new_tmp, info = SS_optimizer(   extended_problem ? SS_solve_block.extended_ss_problem : SS_solve_block.ss_problem,
    # if extended_problem
    #     function ext_function_to_optimize(guesses)
    #         gss = guesses[1:length(guess)]
    
    #         parameters_and_solved_vars_guess = guesses[length(guess)+1:end]
    
    #         res = ss_solve_blocks(parameters_and_solved_vars, gss)
    
    #         return vcat(res, parameters_and_solved_vars .- parameters_and_solved_vars_guess)
    #     end
    # else
    #     function function_to_optimize(guesses) ss_solve_blocks(parameters_and_solved_vars, guesses) end
    # end

    # sol_new_tmp, info = SS_optimizer(   extended_problem ? ext_function_to_optimize : function_to_optimize,
                                        extended_problem ? vcat(sol_values_init, closest_parameters_and_solved_vars) : sol_values_init,
                                        parameters_and_solved_vars,
                                        extended_problem ? lbs : lbs[1:length(guess)],
                                        extended_problem ? ubs : ubs[1:length(guess)],
                                        solver_params,
                                        tol = tol   )

    sol_new = isnothing(sol_new_tmp) ? sol_new_tmp : sol_new_tmp[1:length(guess)]

    sol_minimum = info[4] # isnan(sum(abs, info[4])) ? Inf : ℒ.norm(info[4])
    
    rel_sol_minimum = info[3]

    sol_values = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], sol_new))

    total_iters[1] += info[1]
    total_iters[2] += info[2]

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

    # max_resid = maximum(abs,ss_solve_blocks(parameters_and_solved_vars, sol_values))

    SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.func_buffer, sol_values, parameters_and_solved_vars)
    
    max_resid = maximum(abs, SS_solve_block.ss_problem.func_buffer)

    if sol_minimum < ftol && verbose
            println("Block: $n_block - Solved $(extended_problem_str) using ",string(SS_optimizer),", $(any_guess_str)$(starting_value_str); maximum residual = $max_resid")
    end
    
    return sol_values, total_iters, rel_sol_minimum, sol_minimum
end


function block_solver(parameters_and_solved_vars::Vector{T}, 
                        n_block::Int, 
                        # ss_solve_blocks::Function, 
                        SS_solve_block::ss_solve_block,
                        # SS_optimizer, 
                        # f::OptimizationFunction, 
                        guess_and_pars_solved_vars::Vector{Vector{T}}, 
                        lbs::Vector{T}, 
                        ubs::Vector{T},
                        parameters::Vector{solver_parameters},
                        fail_fast_solvers_only::Bool,
                        cold_start::Bool,
                        verbose::Bool ;
                        tol::Tolerances = Tolerances(),
                        # rtol::AbstractFloat = sqrt(eps()),
                        # timeout = 120,
                        # starting_points::Vector{Float64} = [1.205996189998029, 0.7688, 0.897, 1.2],#, 0.9, 0.75, 1.5, -0.5, 2.0, .25]
                        # verbose::Bool = false
                        )::Tuple{Vector{T},Tuple{T, Int}} where T <: AbstractFloat

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

    if !cold_start
        if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol
            # ∇ = 𝒟.jacobian(x->(ss_solve_blocks(parameters_and_solved_vars, x)), backend, guess)

            # ∇̂ = ℒ.lu!(∇, check = false)

            SS_solve_block.ss_problem.jac(SS_solve_block.ss_problem.jac_buffer, guess, parameters_and_solved_vars)

            ∇ = SS_solve_block.ss_problem.jac_buffer

            ∇̂ = ℒ.lu(∇, check = false)
            
            if ℒ.issuccess(∇̂)
                guess_update = ∇̂ \ res

                new_guess = guess - guess_update

                rel_sol_minimum = ℒ.norm(guess_update) / max(ℒ.norm(new_guess), sol_minimum)
            else
                rel_sol_minimum = 1.0
            end
        else
            rel_sol_minimum = 0.0
        end
    else
        rel_sol_minimum = 1.0
    end
    
    if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
        solved_yet = true

        if verbose
            println("Block: $n_block, - Solved using previous solution; residual norm: $sol_minimum")
        end
    end

    total_iters = [0,0]

    SS_optimizer = levenberg_marquardt

    if cold_start
        guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only

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
                            solved_yet = true
                        end
                    end
                end
            end
        end
    else !cold_start

        pars = (fail_fast_solvers_only ? [parameters[end]] : unique(parameters))
        
        for p in pars #[1:3] # take unique because some parameters might appear more than once
            start_vals = (fail_fast_solvers_only ? [false] : Any[false,p.starting_value, 1.206, 1.5, 0.7688, 2.0, 0.897])
            for s in start_vals #, .9, .75, 1.5, -.5, 2, .25] # try first the guess and then different starting values
                # for ext in [false, true] # try first the system where only values can vary, next try the system where values and parameters can vary
                for algo in [newton, levenberg_marquardt]
                    if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol # || rel_sol_minimum > rtol
                        if solved_yet continue end
                        # println("Block: $n_block pre GN - $ext - $sol_minimum - $rel_sol_minimum")
                        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(algo, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, 
                        # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(algo, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, 
                                                                            total_iters, 
                                                                            n_block, 
                                                                            false, # verbose
                                                                            guess, 
                                                                            p, 
                                                                            # parameters[1],
                                                                            false, # ext
                                                                            # false)
                                                                            s) 
                        if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol # || rel_sol_minimum > rtol)
                            solved_yet = true

                            if verbose
                                # println("Block: $n_block, - Solved with $algo using previous solution - $(indexin([ext],[false, true])[1])/2 - $ext - $sol_minimum - $rel_sol_minimum - $total_iters")
                                println("Block: $n_block, - Solved with $algo using previous solution - $sol_minimum - $rel_sol_minimum - $total_iters")
                            end
                        end                      
                    end
                end
            end
        end


        # if sol_minimum > tol# || rel_sol_minimum > rtol
        #     for p in unique(parameters)#[1:3] # take unique because some parameters might appear more than once
        #         # for s in [p.starting_value, 1.206, 1.5, 0.7688, 2.0, 0.897]#, .9, .75, 1.5, -.5, 2, .25] # try first the guess and then different starting values
        #             # for ext in [false, true] # try first the system where only values can vary, next try the system where values and parameters can vary
        #                 if sol_minimum > tol# || rel_sol_minimum > rtol
        #                     sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, 
        #                                                         false, # verbose
        #                                                         guess, 
        #                                                         p,
        #                                                         false,
        #                                                         false)
        #                                                         # s)
        #                     if !solved_yet && sol_minimum < tol# || rel_sol_minimum > rtol)     
        #                         solved_yet = true
        #                         if verbose
        #                             loop1 = unique(parameters)#[1:3]
        #                             loop2 = [p.starting_value, 1.206, 1.5, 0.7688, 2.0, 0.897]
        #                             p_in_loop1 = findfirst(x -> x == p, loop1)
        #                             s_in_loop2 = findfirst(x -> x == s, loop2)
        #                             if p_in_loop1 isa Nothing
        #                                 p_in_loop1 = 1
        #                             end
        #                             if s_in_loop2 isa Nothing
        #                                 s_in_loop2 = 1
        #                             end
        #                             n1 = (p_in_loop1 - 1) * length(loop2) + s_in_loop2
        #                             println("Block: $n_block, - Solved with modified Levenberg-Marquardt - $n1/$(length(loop2) *length(loop1)) - $sol_minimum - $rel_sol_minimum - $total_iters")
        #                         end
        #                     end 
        #                 end
        #             # end
        #         # end
        #     end
        # end
    end

    if verbose
        if !solved_yet
            println("Block: $n_block, - Solution not found after $(total_iters[1]) gradient evaluations and $(total_iters[2]) function evaluations; reltol: $rel_sol_minimum - tol: $sol_minimum")
        end
    end

    return sol_values, (sol_minimum, total_iters[1])
end


function calculate_second_order_stochastic_steady_state(parameters::Vector{M}, 
                                                        𝓂::ℳ; 
                                                        opts::CalculationOptions = merge_calculation_options(),
                                                        pruning::Bool = false)::Tuple{Vector{M}, Bool, Vector{M}, M, AbstractMatrix{M}, SparseMatrixCSC{M, Int}, AbstractMatrix{M}, SparseMatrixCSC{M, Int}} where M 
                                                        # timer::TimerOutput = TimerOutput(),
                                                        # tol::AbstractFloat = 1e-12)::Tuple{Vector{M}, Bool, Vector{M}, M, AbstractMatrix{M}, SparseMatrixCSC{M}, AbstractMatrix{M}, SparseMatrixCSC{M}} where M
    # @timeit_debug timer "Calculate NSSS" begin

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts) # , timer = timer)

    # end # timeit_debug
    
    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        # if verbose println("NSSS not found") end # handled within solve function
        return zeros(𝓂.timings.nVars), false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    # @timeit_debug timer "Calculate Jacobian" begin

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix
    
    # end # timeit_debug

    # @timeit_debug timer "Calculate first order solution" begin

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁;
                                                        T = 𝓂.timings,
                                                        opts = opts,
                                                        initial_guess = 𝓂.solution.perturbation.qme_solution,
                                                        decomposition = 𝓂.solution.perturbation.first_order_block_decomposition)

    if solved 𝓂.solution.perturbation.qme_solution = qme_sol end

    # end # timeit_debug

    if !solved
        if opts.verbose println("1st order solution not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    # @timeit_debug timer "Calculate Hessian" begin

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)# * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂
    
    # end # timeit_debug

    # @timeit_debug timer "Calculate second order solution" begin

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices,
                                                    𝓂.caches; 
                                                    T = 𝓂.timings, 
                                                    initial_guess = 𝓂.solution.perturbation.second_order_solution,
                                                    # timer = timer,
                                                    opts = opts)

    if eltype(𝐒₂) == Float64 && solved2 𝓂.solution.perturbation.second_order_solution = 𝐒₂ end

    𝐒₂ *= 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔₂

    if !issparse(𝐒₂)
        𝐒₂ = sparse(𝐒₂) # * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔₂)
    end

    # end # timeit_debug

    if !solved2
        if opts.verbose println("2nd order solution not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    # @timeit_debug timer "Calculate SSS" begin

    𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]

    aug_state₁ = sparse([zeros(𝓂.timings.nPast_not_future_and_mixed); 1; zeros(𝓂.timings.nExo)])

    tmp = (ℒ.I - 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed])

    tmp̄ = @ignore_derivatives ℒ.lu(tmp, check = false)

    if !ℒ.issuccess(tmp̄)
        if opts.verbose println("SSS not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
    end

    SSSstates = collect(tmp \ (𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[𝓂.timings.past_not_future_and_mixed_idx])

    if pruning
        state = 𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] * SSSstates + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2
        converged = true
    else
        nᵉ = 𝓂.timings.nExo

        s_in_s⁺ = @ignore_derivatives BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))

        kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
        
        A = 𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed]
        B̂ = 𝐒₂[:,kron_s⁺_s⁺]
    
        SSSstates, converged = calculate_second_order_stochastic_steady_state(Val(:newton), 𝐒₁, 𝐒₂, collect(SSSstates), 𝓂) # , timer = timer)
        
        if !converged
            if opts.verbose println("SSS not found") end
            return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0)
        end

        state = A * SSSstates + B̂ * ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1)) / 2
        # state, converged = second_order_stochastic_steady_state_iterative_solution([sparsevec(𝐒₁); vec(𝐒₂)]; dims = [size(𝐒₁); size(𝐒₂)], 𝓂 = 𝓂)
    end

    # end # timeit_debug

    # all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    # all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
    # NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
    
    # all_SS = [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
    # we need all variables for the stochastic steady state because even leads and lags have different SSS then the non-lead-lag ones (contrary to the no stochastic steady state) and we cannot recover them otherwise

    return all_SS + state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
end



function calculate_second_order_stochastic_steady_state(::Val{:newton}, 
                                                        𝐒₁::Matrix{R}, 
                                                        𝐒₂::AbstractSparseMatrix{R}, 
                                                        x::Vector{R},
                                                        𝓂::ℳ;
                                                        # timer::TimerOutput = TimerOutput(),
                                                        tol::AbstractFloat = 1e-14) where R <: AbstractFloat
    # @timeit_debug timer "Setup matrices" begin

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
    
    # end # timeit_debug
      
    # @timeit_debug timer "Iterations" begin

    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end

        x̂ = A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2

        Δx = ∂x̂ \ (x̂ - x)
        
        if i > 3 && isapprox(x̂, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    # end # timeit_debug

    return x, isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2, x, rtol = tol)
end




function calculate_second_order_stochastic_steady_state(::Val{:newton}, 
                                                        𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                                        𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                                        x::Vector{ℱ.Dual{Z,S,N}},
                                                        𝓂::ℳ;
                                                        # timer::TimerOutput = TimerOutput(),
                                                        tol::AbstractFloat = 1e-14)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}

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
 
    ∂x̄  = zeros(S, length(x̂), N)
    
    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            break
        end
        
        Δx = ∂x̂ \ (A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 - x̂)

        if i > 5 && isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2, x̂, rtol = tol)
            break
        end
        
        # x̂ += Δx
        ℒ.axpy!(-1, Δx, x̂)
    end

    solved = isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2, x̂, rtol = tol)

    if solved
        for i in 1:N
            ∂𝐒₁ = ℱ.partials.(𝐒₁, i)
            ∂𝐒₂ = ℱ.partials.(𝐒₂, i)

            ∂A = ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
            ∂B̂ = ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]

            tmp = ∂A * x̂ + ∂B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2

            TMP = A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)

            ∂x̄[:,i] = -TMP \ tmp
        end
    end
    
    return reshape(map(x̂, eachrow(∂x̄)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), solved
end

end # dispatch_doctor

function rrule(::typeof(calculate_second_order_stochastic_steady_state),
                                                        ::Val{:newton}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        # timer::TimerOutput = TimerOutput(),
                                                        tol::AbstractFloat = 1e-14)
    # @timeit_debug timer "Calculate SSS - forward" begin
    # @timeit_debug timer "Setup indices" begin

    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))
    
    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    
    # end # timeit_debug
      
    # @timeit_debug timer "Iterations" begin

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

    # end # timeit_debug
    # end # timeit_debug

    function second_order_stochastic_steady_state_pullback(∂x)
        # @timeit_debug timer "Calculate SSS - pullback" begin

        S = -∂x[1]' / (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed] = S' * x'
        
        ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺] = S' * ℒ.kron(vcat(x,1), vcat(x,1))' / 2

        # end # timeit_debug

        return NoTangent(), NoTangent(), ∂𝐒₁, ∂𝐒₂, NoTangent(), NoTangent(), NoTangent()
    end

    return (x, solved), second_order_stochastic_steady_state_pullback
end

@stable default_mode = "disable" begin

function calculate_third_order_stochastic_steady_state( parameters::Vector{M}, 
                                                        𝓂::ℳ; 
                                                        opts::CalculationOptions = merge_calculation_options(),
                                                        pruning::Bool = false)::Tuple{Vector{M}, Bool, Vector{M}, M, AbstractMatrix{M}, SparseMatrixCSC{M, Int}, SparseMatrixCSC{M, Int}, AbstractMatrix{M}, SparseMatrixCSC{M, Int}, SparseMatrixCSC{M, Int}} where M <: Real
                                                        # timer::TimerOutput = TimerOutput(),
                                                        # tol::AbstractFloat = 1e-12)::Tuple{Vector{M}, Bool, Vector{M}, M, AbstractMatrix{M}, SparseMatrixCSC{M}, SparseMatrixCSC{M}, AbstractMatrix{M}, SparseMatrixCSC{M}, SparseMatrixCSC{M}} where M
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts) # , timer = timer)
    
    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        if opts.verbose println("NSSS not found") end
        return zeros(𝓂.timings.nVars), false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end
    
    all_SS = expand_steady_state(SS_and_pars,𝓂)

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix
    
    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁;
                                                        T = 𝓂.timings,
                                                        opts = opts,
                                                        initial_guess = 𝓂.solution.perturbation.qme_solution,
                                                        decomposition = 𝓂.solution.perturbation.first_order_block_decomposition)
    
    if solved 𝓂.solution.perturbation.qme_solution = qme_sol end

    if !solved
        if opts.verbose println("1st order solution not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)# * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 
                                                    𝓂.solution.perturbation.second_order_auxilliary_matrices,
                                                    𝓂.caches;
                                                    T = 𝓂.timings,
                                                    initial_guess = 𝓂.solution.perturbation.second_order_solution,
                                                    # timer = timer,
                                                    opts = opts)

    if !solved2
        if opts.verbose println("2nd order solution not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end
    
    if eltype(𝐒₂) == Float64 && solved2 𝓂.solution.perturbation.second_order_solution = 𝐒₂ end

    𝐒₂ *= 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔₂

    if !issparse(𝐒₂)
        𝐒₂ = sparse(𝐒₂) # * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔₂)
    end
    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂) #, timer = timer)# * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃
            
    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
                                                𝓂.solution.perturbation.second_order_auxilliary_matrices, 
                                                𝓂.solution.perturbation.third_order_auxilliary_matrices,
                                                𝓂.caches; 
                                                T = 𝓂.timings, 
                                                initial_guess = 𝓂.solution.perturbation.third_order_solution,
                                                # timer = timer, 
                                                opts = opts)

    if !solved3
        if opts.verbose println("3rd order solution not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    if eltype(𝐒₃) == Float64 && solved3 𝓂.solution.perturbation.third_order_solution = 𝐒₃ end

    if length(𝓂.caches.third_order_caches.Ŝ) == 0 || !(eltype(𝐒₃) == eltype(𝓂.caches.third_order_caches.Ŝ))
        𝓂.caches.third_order_caches.Ŝ = 𝐒₃ * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔₃
    else
        mul_reverse_AD!(𝓂.caches.third_order_caches.Ŝ, 𝐒₃, 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔₃)
    end

    Ŝ = 𝓂.caches.third_order_caches.Ŝ

    𝐒₃̂ = sparse_preallocated!(Ŝ, ℂ = 𝓂.caches.third_order_caches)
    
    # 𝐒₃ *= 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔₃
    # 𝐒₃ = sparse_preallocated!(𝐒₃, ℂ = 𝓂.caches.third_order_caches)
    
    # 𝐒₃ = sparse(Ŝ) # * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔₃)

    𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]

    aug_state₁ = sparse([zeros(𝓂.timings.nPast_not_future_and_mixed); 1; zeros(𝓂.timings.nExo)])
    
    tmp = (ℒ.I - 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed])

    tmp̄ = @ignore_derivatives ℒ.lu(tmp, check = false)

    if !ℒ.issuccess(tmp̄)
        if opts.verbose println("SSS not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
    end

    SSSstates = collect(tmp \ (𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[𝓂.timings.past_not_future_and_mixed_idx])

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
        Ĉ = 𝐒₃̂[:,kron_s⁺_s⁺_s⁺]
    
        SSSstates, converged = calculate_third_order_stochastic_steady_state(Val(:newton), 𝐒₁, 𝐒₂, 𝐒₃̂, SSSstates, 𝓂)
        
        if !converged
            if opts.verbose println("SSS not found") end
            return all_SS, false, SS_and_pars, solution_error, zeros(0,0), spzeros(0,0), spzeros(0,0), zeros(0,0), spzeros(0,0), spzeros(0,0)
        end

        state = A * SSSstates + B̂ * ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1)) / 2 + Ĉ * ℒ.kron(vcat(SSSstates,1),  ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1))) / 6
        # state, converged = third_order_stochastic_steady_state_iterative_solution([sparsevec(𝐒₁); vec(𝐒₂); vec(𝐒₃)]; dims = [size(𝐒₁); size(𝐒₂); size(𝐒₃)], 𝓂 = 𝓂)
        # state, converged = third_order_stochastic_steady_state_iterative_solution_forward([sparsevec(𝐒₁); vec(𝐒₂); vec(𝐒₃)]; dims = [size(𝐒₁); size(𝐒₂); size(𝐒₃)], 𝓂 = 𝓂)
    end

    # all_variables = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    # all_variables[indexin(𝓂.aux,all_variables)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
    # NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]
    
    # all_SS = [SS_and_pars[indexin([s],NSSS_labels)...] for s in all_variables]
    # we need all variables for the stochastic steady state because even leads and lags have different SSS then the non-lead-lag ones (contrary to the no stochastic steady state) and we cannot recover them otherwise

    return all_SS + state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃̂
end


function calculate_third_order_stochastic_steady_state(::Val{:newton}, 
                                                        𝐒₁::Matrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        𝐒₃::AbstractSparseMatrix{Float64},
                                                        x::Vector{Float64},
                                                        𝓂::ℳ;
                                                        # timer::TimerOutput = TimerOutput(),
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


function calculate_third_order_stochastic_steady_state(::Val{:newton}, 
                                                        𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                                        𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                                        𝐒₃::AbstractSparseMatrix{ℱ.Dual{Z,S,N}},
                                                        x::Vector{ℱ.Dual{Z,S,N}},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
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

    ∂x̄  = zeros(S, length(x̂), N)
    
    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        ∂x = (A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x̂,1), vcat(x̂,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed))

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            break
        end
        
        Δx = ∂x̂ \ (A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6 - x̂)

        if i > 5 && isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6, x̂, rtol = tol)
            break
        end
        
        # x̂ += Δx
        ℒ.axpy!(-1, Δx, x̂)
    end

    solved = isapprox(A * x̂ + B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6, x̂, rtol = tol)
    
    if solved
        for i in 1:N
            ∂𝐒₁ = ℱ.partials.(𝐒₁, i)
            ∂𝐒₂ = ℱ.partials.(𝐒₂, i)
            ∂𝐒₃ = ℱ.partials.(𝐒₃, i)

            ∂A = ∂𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
            ∂B̂ = ∂𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
            ∂Ĉ = ∂𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

            tmp = ∂A * x̂ + ∂B̂ * ℒ.kron(vcat(x̂,1), vcat(x̂,1)) / 2 + ∂Ĉ * ℒ.kron(vcat(x̂,1), ℒ.kron(vcat(x̂,1), vcat(x̂,1))) / 6

            TMP = A + B * ℒ.kron(vcat(x̂,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x̂,1), vcat(x̂,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)

            ∂x̄[:,i] = -TMP \ tmp
        end
    end
    
    return reshape(map(x̂, eachrow(∂x̄)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), solved
end

end # dispatch_doctor

function rrule(::typeof(calculate_third_order_stochastic_steady_state),
                                                        ::Val{:newton}, 
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

@stable default_mode = "disable" begin

function solve!(𝓂::ℳ; 
                parameters::ParameterType = nothing, 
                dynamics::Bool = false, 
                algorithm::Symbol = :first_order, 
                opts::CalculationOptions = merge_calculation_options(),
                obc::Bool = false,
                silent::Bool = false) #,
                # quadratic_matrix_equation_algorithm::Symbol = :schur,
                # verbose::Bool = false,
                # timer::TimerOutput = TimerOutput(),
                # tol::AbstractFloat = 1e-12)

    @assert algorithm ∈ all_available_algorithms
    
    # @timeit_debug timer "Write parameter inputs" begin

    write_parameters_input!(𝓂, parameters, verbose = opts.verbose)

    # end # timeit_debug

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
        obc_not_solved = isnothing(𝓂.solution.perturbation.first_order.state_update_obc(zeros(𝓂.timings.nVars), zeros(𝓂.timings.nExo)))
        if  ((:first_order         == algorithm) && ((:first_order         ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:second_order        == algorithm) && ((:second_order        ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:pruned_second_order == algorithm) && ((:pruned_second_order ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:third_order         == algorithm) && ((:third_order         ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:pruned_third_order  == algorithm) && ((:pruned_third_order  ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))

            # @timeit_debug timer "Solve for NSSS (if necessary)" begin

            SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

            # end # timeit_debug

            @assert solution_error < opts.tol.NSSS_acceptance_tol "Could not find non stochastic steady steady."
            
            # @timeit_debug timer "Calculate Jacobian" begin

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
            
            # end # timeit_debug

            # @timeit_debug timer "Calculate first order solution" begin

            S₁, qme_sol, solved = calculate_first_order_solution(∇₁;
                                                                T = 𝓂.timings,
                                                                opts = opts,
                                                                initial_guess = 𝓂.solution.perturbation.qme_solution,
                                                                decomposition = 𝓂.solution.perturbation.first_order_block_decomposition)
            if solved 𝓂.solution.perturbation.qme_solution = qme_sol end

            # end # timeit_debug

            @assert solved "Could not find stable first order solution."

            state_update₁ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                            shock]
                return S₁ * aug_state # you need a return statement for forwarddiff to work
            end
            
            if obc
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1, verbose = false)

                ∇̂₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
            
                Ŝ₁, qme_sol, solved = calculate_first_order_solution(∇̂₁;
                                                                    T = 𝓂.timings,
                                                                    opts = opts,
                                                                    initial_guess = 𝓂.solution.perturbation.qme_solution,
                                                                    decomposition = 𝓂.solution.perturbation.first_order_block_decomposition)
                
                if solved 𝓂.solution.perturbation.qme_solution = qme_sol end

                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 0, verbose = false)

                state_update₁̂ = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
                    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                                shock]
                    return Ŝ₁ * aug_state # you need a return statement for forwarddiff to work
                end
            else
                state_update₁̂ = (x,y)->nothing
            end
            
            𝓂.solution.perturbation.first_order = perturbation_solution(S₁, state_update₁, state_update₁̂)
            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:first_order])

            𝓂.solution.non_stochastic_steady_state = SS_and_pars
            𝓂.solution.outdated_NSSS = solution_error > opts.tol.NSSS_acceptance_tol
        end

        obc_not_solved = isnothing(𝓂.solution.perturbation.second_order.state_update_obc(zeros(𝓂.timings.nVars), zeros(𝓂.timings.nExo)))
        if  ((:second_order  == algorithm) && ((:second_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:third_order  == algorithm) && ((:third_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))
            

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, opts = opts) # , timer = timer)
            
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
                state_update₂̂ = (x,y)->nothing
            end

            𝓂.solution.perturbation.second_order = second_order_perturbation_solution(stochastic_steady_state, state_update₂, state_update₂̂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:second_order])
        end
        
        obc_not_solved = isnothing(𝓂.solution.perturbation.pruned_second_order.state_update_obc([zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars)], zeros(𝓂.timings.nExo)))
        if  ((:pruned_second_order  == algorithm) && ((:pruned_second_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved))) ||
            ((:pruned_third_order  == algorithm) && ((:pruned_third_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, opts = opts, pruning = true) # , timer = timer)

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
                state_update₂̂ = (x,y)->nothing
            end

            𝓂.solution.perturbation.pruned_second_order = second_order_perturbation_solution(stochastic_steady_state, state_update₂, state_update₂̂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:pruned_second_order])
        end
        
        obc_not_solved = isnothing(𝓂.solution.perturbation.third_order.state_update_obc(zeros(𝓂.timings.nVars), zeros(𝓂.timings.nExo)))
        if  ((:third_order  == algorithm) && ((:third_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))
            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, opts = opts)

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
                state_update₃̂ = (x,y)->nothing
            end

            𝓂.solution.perturbation.third_order = third_order_perturbation_solution(stochastic_steady_state, state_update₃, state_update₃̂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:third_order])
        end

        obc_not_solved = isnothing(𝓂.solution.perturbation.pruned_third_order.state_update_obc([zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars)], zeros(𝓂.timings.nExo)))
        if ((:pruned_third_order  == algorithm) && ((:pruned_third_order   ∈ 𝓂.solution.outdated_algorithms) || (obc && obc_not_solved)))

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, opts = opts, pruning = true)

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
                state_update₃̂ = (x,y)->nothing
            end

            𝓂.solution.perturbation.pruned_third_order = third_order_perturbation_solution(stochastic_steady_state, state_update₃, state_update₃̂)

            𝓂.solution.outdated_algorithms = setdiff(𝓂.solution.outdated_algorithms,[:pruned_third_order])
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

    return third_order_auxilliary_matrices(𝐂₃, 𝐔₃, 𝐈₃, 𝐂∇₃, 𝐔∇₃, 𝐏, 𝐏₁ₗ, 𝐏₁ᵣ, 𝐏₁ₗ̂, 𝐏₂ₗ̂, 𝐏₁ₗ̄, 𝐏₂ₗ̄, 𝐏₁ᵣ̃, 𝐏₂ᵣ̃, 𝐒𝐏)
end

function take_nth_order_derivatives(
    dyn_equations::Vector{T},
    𝔙::Symbolics.Arr,
    𝔓::Symbolics.Arr,
    SS_mapping::Dict{T, T},
    nps::Int,
    nxs::Int;
    max_perturbation_order::Int = 1,
    output_compressed::Bool = true # Controls compression for X derivatives (order >= 2)
)::Vector{Tuple{SparseMatrixCSC{T, Int}, SparseMatrixCSC{T, Int}}} where T <: Symbolics.Num#, Tuple{Symbolics.Arr{Symbolics.Num, 1}, Symbolics.Arr{Symbolics.Num, 1}}}
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
    spP_of_flatX_nzval_order_1 = Symbolics.sparsejacobian(spX_order_1_sub.nzval, vcat(𝔓[1:nps], 𝔙[1:nxs])) # nnz(spX_order_1) x np

    # Determine dimensions for the Order 1 P matrix
    X_nrows_1 = nϵ
    X_ncols_1 = nx
    P_nrows_1 = X_nrows_1 * X_ncols_1
    P_ncols_1 = nps + nxs

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
            spP_of_flatX_nzval_curr = Symbolics.sparsejacobian(sp_flat_curr_X.nzval, vcat(𝔓[1:nps], 𝔙[1:nxs])) # nnz(sp_flat_curr_X) x np
            
            # Determine the desired dimensions of spP_order_n
            # Dimensions are (rows of spX_order_n * cols of spX_order_n) x np
            P_nrows_n = nϵ * X_ncols_n
            P_ncols_n = nps + nxs

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
end


# TODO: check why this takes so much longer than previous implementation
function write_functions_mapping!(𝓂::ℳ, max_perturbation_order::Int; 
                                    density_threshold::Float64 = .1, 
                                    min_length::Int = 1000,
                                    # parallel = Symbolics.SerialForm(),
                                    # parallel = Symbolics.ShardedForm(1500,4),
                                    cse = true,
                                    skipzeros = true)

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
                push!(SS_mapping, 𝔓[i] => 𝔙[nxs + i - nps - nc])
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

    derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs)


    ∇₁_dyn = derivatives[1][1]

    lennz = nnz(∇₁_dyn)

    if (lennz / length(∇₁_dyn) > density_threshold) || (length(∇₁_dyn) < min_length)
        derivatives_mat = convert(Matrix, ∇₁_dyn)
        buffer = zeros(Float64, size(∇₁_dyn))
    else
        derivatives_mat = ∇₁_dyn
        buffer = similar(∇₁_dyn, Float64)
        buffer.nzval .= 0
    end
    
    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end
    
    _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔓, 𝔙, 
                                            cse = cse, 
                                            skipzeros = skipzeros, 
                                            parallel = parallel,
                                            # nanmath = false,
                                            expression_module = @__MODULE__,
                                            expression = Val(false))::Tuple{<:Function, <:Function}

    𝓂.jacobian = buffer, func_exprs


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
                                                        # nanmath = false,
                                                        expression_module = @__MODULE__,
                                                        expression = Val(false))::Tuple{<:Function, <:Function}

    𝓂.jacobian_parameters =  buffer_parameters, func_∇₁_parameters
 

    ∇₁_SS_and_pars = derivatives[1][2][:,nps+1:end]

    lennz = nnz(∇₁_SS_and_pars)

    if (lennz / length(∇₁_SS_and_pars) > density_threshold) || (length(∇₁_SS_and_pars) < min_length)
        ∇₁_SS_and_pars_mat = convert(Matrix, ∇₁_SS_and_pars)
        buffer_SS_and_pars = zeros(Float64, size(∇₁_SS_and_pars))
    else
        ∇₁_SS_and_pars_mat = ∇₁_SS_and_pars
        buffer_SS_and_pars = similar(∇₁_SS_and_pars, Float64)
        buffer_SS_and_pars.nzval .= 0
    end

    if lennz > 1500
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, func_∇₁_SS_and_pars = Symbolics.build_function(∇₁_SS_and_pars_mat, 𝔓, 𝔙, 
                                                        cse = cse, 
                                                        skipzeros = skipzeros, 
                                                        parallel = parallel,
                                                        # nanmath = false,
                                                        expression_module = @__MODULE__,
                                                        expression = Val(false))::Tuple{<:Function, <:Function}

    𝓂.jacobian_SS_and_pars = buffer_SS_and_pars, func_∇₁_SS_and_pars

    # precompute block decomposition for first order solution
    rows, cols, _ = findnz(∇₁_dyn)
    pattern_full = sparse(rows, cols, ones(Int, length(rows)), size(∇₁_dyn,1), size(∇₁_dyn,2))

    T = 𝓂.timings
    dynIndex = T.nPresent_only+1:T.nVars
    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)
    f_in = indexin(T.future_not_past_and_mixed_idx, comb)
    Ir = ℒ.I(length(comb))

    Aplus_pattern = pattern_full[dynIndex,1:T.nFuture_not_past_and_mixed] * Ir[f_in,:]
    A0_pattern = pattern_full[dynIndex,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)][:,comb]
    block_pattern = (Aplus_pattern .+ A0_pattern) .> 0
    Q, P, R, _, _ = BlockTriangularForm.order(block_pattern)
    𝓂.solution.perturbation.first_order_block_decomposition = (Vector{Int}(Q), Vector{Int}(P), Vector{Int}(R))




    # if max_perturbation_order >= 1
    #     SS_and_pars = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))

    #     eqs = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

    #     nx = length(𝓂.parameter_values)

    #     np = length(SS_and_pars)

    #     nϵˢ = length(eqs)

    #     nc = length(𝓂.calibration_equations_no_var)

    #     Symbolics.@variables 𝔛¹[1:nx] 𝔓¹[1:np]

    #     ϵˢ = zeros(Symbolics.Num, nϵˢ)
    
    #     calib_vals = zeros(Symbolics.Num, nc)

    #     𝓂.SS_calib_func(calib_vals, 𝔛¹)
    
    #     𝓂.SS_check_func(ϵˢ, 𝔛¹, 𝔓¹, calib_vals)
    # println(ϵˢ)
    #     ∂SS_equations_∂parameters = Symbolics.sparsejacobian(ϵˢ, 𝔛¹) # nϵ x nx
    
    #     lennz = nnz(∂SS_equations_∂parameters)

    #     if (lennz / length(∂SS_equations_∂parameters) > density_threshold) || (length(∂SS_equations_∂parameters) < min_length)
    #         derivatives_mat = convert(Matrix, ∂SS_equations_∂parameters)
    #         buffer = zeros(Float64, size(∂SS_equations_∂parameters))
    #     else
    #         derivatives_mat = ∂SS_equations_∂parameters
    #         buffer = similar(∂SS_equations_∂parameters, Float64)
    #         buffer.nzval .= 0
    #     end

    #     if lennz > 1500
    #         parallel = Symbolics.ShardedForm(1500,4)
    #     else
    #         parallel = Symbolics.SerialForm()
    #     end
        
    #     _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔛¹, 𝔓¹, 
    #                                                 cse = cse, 
    #                                                 skipzeros = skipzeros, 
    #                                                 parallel = parallel,
    #                                                 # nanmath = false,
    #                                                 expression_module = @__MODULE__,
    #                                                 expression = Val(false))::Tuple{<:Function, <:Function}

    #     𝓂.∂SS_equations_∂parameters = buffer, func_exprs



    #     ∂SS_equations_∂SS_and_pars = Symbolics.sparsejacobian(ϵˢ, 𝔓¹) # nϵ x nx
    
    #     lennz = nnz(∂SS_equations_∂SS_and_pars)

    #     if (lennz / length(∂SS_equations_∂SS_and_pars) > density_threshold) || (length(∂SS_equations_∂SS_and_pars) < min_length)
    #         derivatives_mat = convert(Matrix, ∂SS_equations_∂SS_and_pars)
    #         buffer = zeros(Float64, size(∂SS_equations_∂SS_and_pars))
    #     else
    #         derivatives_mat = ∂SS_equations_∂SS_and_pars
    #         buffer = similar(∂SS_equations_∂SS_and_pars, Float64)
    #         buffer.nzval .= 0
    #     end

    #     if lennz > 1500
    #         parallel = Symbolics.ShardedForm(1500,4)
    #     else
    #         parallel = Symbolics.SerialForm()
    #     end

    #     _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔛¹, 𝔓¹, 
    #                                                 cse = cse, 
    #                                                 skipzeros = skipzeros, 
    #                                                 parallel = parallel,
    #                                                 # nanmath = false,
    #                                                 expression_module = @__MODULE__,
    #                                                 expression = Val(false))::Tuple{<:Function, <:Function}

    #     𝓂.∂SS_equations_∂SS_and_pars = buffer, func_exprs
    # end
        
    if max_perturbation_order >= 2
    # second order
        derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs; max_perturbation_order = 2, output_compressed = false)

        if 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝛔 == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            𝓂.solution.perturbation.second_order_auxilliary_matrices = create_second_order_auxilliary_matrices(𝓂.timings)

            ∇₂_dyn = derivatives[2][1]

            lennz = nnz(∇₂_dyn)

            if (lennz / length(∇₂_dyn) > density_threshold) || (length(∇₂_dyn) < min_length)
                derivatives_mat = convert(Matrix, ∇₂_dyn)
                buffer = zeros(Float64, size(∇₂_dyn))
            else
                derivatives_mat = ∇₂_dyn
                buffer = similar(∇₂_dyn, Float64)
                buffer.nzval .= 0
            end

            if lennz > 1500
                parallel = Symbolics.ShardedForm(1500,4)
            else
                parallel = Symbolics.SerialForm()
            end

            _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔓, 𝔙, 
                                                        cse = cse, 
                                                        skipzeros = skipzeros, 
                                                        parallel = parallel,
                                                        # nanmath = false,
                                                        expression_module = @__MODULE__,
                                                        expression = Val(false))::Tuple{<:Function, <:Function}

            𝓂.hessian = buffer, func_exprs


            ∇₂_parameters = derivatives[2][2][:,1:nps]

            lennz = nnz(∇₂_parameters)

            if (lennz / length(∇₂_parameters) > density_threshold) || (length(∇₂_parameters) < min_length)
                ∇₂_parameters_mat = convert(Matrix, ∇₂_parameters)
                buffer_parameters = zeros(Float64, size(∇₂_parameters))
            else
                ∇₂_parameters_mat = ∇₂_parameters
                buffer_parameters = similar(∇₂_parameters, Float64)
                buffer_parameters.nzval .= 0
            end

            if lennz > 1500
                parallel = Symbolics.ShardedForm(1500,4)
            else
                parallel = Symbolics.SerialForm()
            end

            _, func_∇₂_parameters = Symbolics.build_function(∇₂_parameters_mat, 𝔓, 𝔙, 
                                                                cse = cse, 
                                                                skipzeros = skipzeros, 
                                                                parallel = parallel,
                                                                # nanmath = false,
                                                                expression_module = @__MODULE__,
                                                                expression = Val(false))::Tuple{<:Function, <:Function}

            𝓂.hessian_parameters =  buffer_parameters, func_∇₂_parameters
        

            ∇₂_SS_and_pars = derivatives[2][2][:,nps+1:end]

            lennz = nnz(∇₂_SS_and_pars)

            if (lennz / length(∇₂_SS_and_pars) > density_threshold) || (length(∇₂_SS_and_pars) < min_length)
                ∇₂_SS_and_pars_mat = convert(Matrix, ∇₂_SS_and_pars)
                buffer_SS_and_pars = zeros(Float64, size(∇₂_SS_and_pars))
            else
                ∇₂_SS_and_pars_mat = ∇₂_SS_and_pars
                buffer_SS_and_pars = similar(∇₂_SS_and_pars, Float64)
                buffer_SS_and_pars.nzval .= 0
            end

            if lennz > 1500
                parallel = Symbolics.ShardedForm(1500,4)
            else
                parallel = Symbolics.SerialForm()
            end

            _, func_∇₂_SS_and_pars = Symbolics.build_function(∇₂_SS_and_pars_mat, 𝔓, 𝔙, 
                                                                cse = cse, 
                                                                skipzeros = skipzeros, 
                                                                parallel = parallel,
                                                                # nanmath = false,
                                                                expression_module = @__MODULE__,
                                                                expression = Val(false))::Tuple{<:Function, <:Function}

            𝓂.hessian_SS_and_pars = buffer_SS_and_pars, func_∇₂_SS_and_pars
        end
    end

    if max_perturbation_order == 3
        derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs; max_perturbation_order = max_perturbation_order, output_compressed = true)
    # third order
        if 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            I,J,V = findnz(derivatives[3][1])
            𝓂.solution.perturbation.third_order_auxilliary_matrices = create_third_order_auxilliary_matrices(𝓂.timings, unique(J))
        
            ∇₃_dyn = derivatives[3][1]

            lennz = nnz(∇₃_dyn)

            if (lennz / length(∇₃_dyn) > density_threshold) || (length(∇₃_dyn) < min_length)
                derivatives_mat = convert(Matrix, ∇₃_dyn)
                buffer = zeros(Float64, size(∇₃_dyn))
            else
                derivatives_mat = ∇₃_dyn
                buffer = similar(∇₃_dyn, Float64)
                buffer.nzval .= 0
            end

            if lennz > 1500
                parallel = Symbolics.ShardedForm(1500,4)
            else
                parallel = Symbolics.SerialForm()
            end

            _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔓, 𝔙, 
                                                        cse = cse, 
                                                        skipzeros = skipzeros, 
                                                        parallel = parallel,
                                                        # nanmath = false,
                                                        expression_module = @__MODULE__,
                                                        expression = Val(false))::Tuple{<:Function, <:Function}

            𝓂.third_order_derivatives = buffer, func_exprs


            ∇₃_parameters = derivatives[3][2][:,1:nps]

            lennz = nnz(∇₃_parameters)

            if (lennz / length(∇₃_parameters) > density_threshold) || (length(∇₃_parameters) < min_length)
                ∇₃_parameters_mat = convert(Matrix, ∇₃_parameters)
                buffer_parameters = zeros(Float64, size(∇₃_parameters))
            else
                ∇₃_parameters_mat = ∇₃_parameters
                buffer_parameters = similar(∇₃_parameters, Float64)
                buffer_parameters.nzval .= 0
            end

            if lennz > 1500
                parallel = Symbolics.ShardedForm(1500,4)
            else
                parallel = Symbolics.SerialForm()
            end

            _, func_∇₃_parameters = Symbolics.build_function(∇₃_parameters_mat, 𝔓, 𝔙, 
                                                                cse = cse, 
                                                                skipzeros = skipzeros, 
                                                                parallel = parallel,
                                                                # nanmath = false,
                                                                expression_module = @__MODULE__,
                                                                expression = Val(false))::Tuple{<:Function, <:Function}

            𝓂.third_order_derivatives_parameters =  buffer_parameters, func_∇₃_parameters
        

            ∇₃_SS_and_pars = derivatives[3][2][:,nps+1:end]

            lennz = nnz(∇₃_SS_and_pars)

            if (lennz / length(∇₃_SS_and_pars) > density_threshold) || (length(∇₃_SS_and_pars) < min_length)
                ∇₃_SS_and_pars_mat = convert(Matrix, ∇₃_SS_and_pars)
                buffer_SS_and_pars = zeros(Float64, size(∇₃_SS_and_pars))
            else
                ∇₃_SS_and_pars_mat = ∇₃_SS_and_pars
                buffer_SS_and_pars = similar(∇₃_SS_and_pars, Float64)
                buffer_SS_and_pars.nzval .= 0
            end

            if lennz > 1500
                parallel = Symbolics.ShardedForm(1500,4)
            else
                parallel = Symbolics.SerialForm()
            end

            _, func_∇₃_SS_and_pars = Symbolics.build_function(∇₃_SS_and_pars_mat, 𝔓, 𝔙, 
                                                                cse = cse, 
                                                                skipzeros = skipzeros, 
                                                                # nanmath = false,
                                                                parallel = parallel,
                                                                expression_module = @__MODULE__,
                                                                expression = Val(false))::Tuple{<:Function, <:Function}

            𝓂.third_order_derivatives_SS_and_pars = buffer_SS_and_pars, func_∇₃_SS_and_pars
        end
    end

    return nothing
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

    return nothing
end

write_parameters_input!(𝓂::ℳ, parameters::Nothing; verbose::Bool = true) = return parameters
write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,Float64}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Pair{String,Float64}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}(parameters[1] |> Meta.parse |> replace_indices => parameters[2]), verbose = verbose)



write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Float64},Vararg{Pair{Symbol,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
# write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Union{Symbol,String},Union{Float64,Int}},Vararg{Pair{Union{Symbol,String},Union{Float64,Int}}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
# write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Int},Vararg{Pair{String,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{String,Float64},Vararg{Pair{String,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters])
, verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{Symbol, Float64}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{String, Float64}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict{Symbol, Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)


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

    return nothing
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

    return nothing
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

    return nothing
end



function calculate_jacobian(parameters::Vector{M},
                            SS_and_pars::Vector{N},
                            𝓂::ℳ)::Matrix{M} where {M,N}
                            # timer::TimerOutput = TimerOutput())::Matrix{M} where {M,N}
    if eltype(𝓂.jacobian[1]) != M
        if 𝓂.jacobian[1] isa SparseMatrixCSC
            jac_buffer = similar(𝓂.jacobian[1],M)
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(M, size(𝓂.jacobian[1]))
        end
    else
        jac_buffer = 𝓂.jacobian[1]
    end
    
    𝓂.jacobian[2](jac_buffer, parameters, SS_and_pars)
    
    return jac_buffer
end

end # dispatch_doctor

function rrule(::typeof(calculate_jacobian), 
                parameters, 
                SS_and_pars, 
                𝓂)#;
                # timer::TimerOutput = TimerOutput())
    # @timeit_debug timer "Calculate jacobian - forward" begin

    jacobian = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    function calculate_jacobian_pullback(∂∇₁)
        # @timeit_debug timer "Calculate jacobian - reverse" begin

        𝓂.jacobian_parameters[2](𝓂.jacobian_parameters[1], parameters, SS_and_pars)
        𝓂.jacobian_SS_and_pars[2](𝓂.jacobian_SS_and_pars[1], parameters, SS_and_pars)

        ∂parameters = 𝓂.jacobian_parameters[1]' * vec(∂∇₁)
        ∂SS_and_pars = 𝓂.jacobian_SS_and_pars[1]' * vec(∂∇₁)

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent()
    end

    return jacobian, calculate_jacobian_pullback
end

@stable default_mode = "disable" begin

function calculate_hessian(parameters::Vector{M}, SS_and_pars::Vector{N}, 𝓂::ℳ)::SparseMatrixCSC{M, Int} where {M,N}
    if eltype(𝓂.hessian[1]) != M
        if 𝓂.hessian[1] isa SparseMatrixCSC
            hes_buffer = similar(𝓂.hessian[1],M)
            hes_buffer.nzval .= 0
        else
            hes_buffer = zeros(M, size(𝓂.hessian[1]))
        end
    else
        hes_buffer = 𝓂.hessian[1]
    end

    𝓂.hessian[2](hes_buffer, parameters, SS_and_pars)
    
    return hes_buffer
end

end # dispatch_doctor

function rrule(::typeof(calculate_hessian), parameters, SS_and_pars, 𝓂)
    hessian = calculate_hessian(parameters, SS_and_pars, 𝓂)

    function calculate_hessian_pullback(∂∇₂)
        # @timeit_debug timer "Calculate hessian - reverse" begin

        𝓂.hessian_parameters[2](𝓂.hessian_parameters[1], parameters, SS_and_pars)
        𝓂.hessian_SS_and_pars[2](𝓂.hessian_SS_and_pars[1], parameters, SS_and_pars)

        ∂parameters = 𝓂.hessian_parameters[1]' * vec(∂∇₂)
        ∂SS_and_pars = 𝓂.hessian_SS_and_pars[1]' * vec(∂∇₂)

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent()
    end

    return hessian, calculate_hessian_pullback
end

@stable default_mode = "disable" begin

function calculate_third_order_derivatives(parameters::Vector{M}, 
                                            SS_and_pars::Vector{N}, 
                                            𝓂::ℳ)::SparseMatrixCSC{M, Int} where {M,N}
    if eltype(𝓂.third_order_derivatives[1]) != M
        if 𝓂.third_order_derivatives[1] isa SparseMatrixCSC
            third_buffer = similar(𝓂.third_order_derivatives[1],M)
            third_buffer.nzval .= 0
        else
            third_buffer = zeros(M, size(𝓂.third_order_derivatives[1]))
        end
    else
        third_buffer = 𝓂.third_order_derivatives[1]
    end

    𝓂.third_order_derivatives[2](third_buffer, parameters, SS_and_pars)
    
    return third_buffer
end

end # dispatch_doctor

function rrule(::typeof(calculate_third_order_derivatives), parameters, SS_and_pars, 𝓂) # ;
    # timer::TimerOutput = TimerOutput())
    # @timeit_debug timer "3rd order derivatives - forward" begin
    third_order_derivatives = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂) #, timer = timer)
    # end # timeit_debug

    function calculate_third_order_derivatives_pullback(∂∇₃)
        # @timeit_debug timer "3rd order derivatives - pullback" begin
        𝓂.third_order_derivatives_parameters[2](𝓂.third_order_derivatives_parameters[1], parameters, SS_and_pars)
        𝓂.third_order_derivatives_SS_and_pars[2](𝓂.third_order_derivatives_SS_and_pars[1], parameters, SS_and_pars)

        ∂parameters = 𝓂.third_order_derivatives_parameters[1]' * vec(∂∇₃)
        ∂SS_and_pars = 𝓂.third_order_derivatives_SS_and_pars[1]' * vec(∂∇₃)

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂parameters, ∂SS_and_pars, NoTangent()
    end

    return third_order_derivatives, calculate_third_order_derivatives_pullback
end

@stable default_mode = "disable" begin

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


function irf(state_update::Function, 
    obc_state_update::Function,
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64},
    T::timings; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}

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

    var_idx = parse_variables_input_to_index(variables, T) |> sort

    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    always_solved = true

    if shocks == :simulate
        shock_history = randn(T.nExo,periods) * shock_size

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
            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -shock_size : shock_size
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
    shock_size::Real = 1,
    negative_shock::Bool = false)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}

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

    var_idx = parse_variables_input_to_index(variables, T) |> sort

    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    if shocks == :simulate
        shock_history = randn(T.nExo,periods) * shock_size

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
            
            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -shock_size : shock_size
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
    shock_size::Real = 1,
    negative_shock::Bool = false, 
    warmup_periods::Int = 100, 
    draws::Int = 50)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}

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

    var_idx = parse_variables_input_to_index(variables, T) |> sort

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

            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -shock_size : shock_size
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


function parse_variables_input_to_index(variables::Union{Symbol_input,String_input}, T::timings)::Union{UnitRange{Int}, Vector{Int}}
    variables = variables isa String_input ? variables .|> Meta.parse .|> replace_indices : variables

    if variables == :all_excluding_auxilliary_and_obc
        return Int.(indexin(setdiff(T.var[.!contains.(string.(T.var),"ᵒᵇᶜ")],union(T.aux, T.exo_present)),sort(union(T.var,T.aux,T.exo_present))))
        # return indexin(setdiff(setdiff(T.var,T.exo_present),T.aux),sort(union(T.var,T.aux,T.exo_present)))
    elseif variables == :all_excluding_obc
        return Int.(indexin(T.var[.!contains.(string.(T.var),"ᵒᵇᶜ")],sort(union(T.var,T.aux,T.exo_present))))
    elseif variables == :all
        return 1:length(union(T.var,T.aux,T.exo_present))
    elseif variables isa Matrix{Symbol}
        if length(setdiff(variables,T.var)) > 0
            @warn "Following variables are not part of the model: " * join(string.(setdiff(variables,T.var)),", ")
            return Int[]
        end
        return getindex(1:length(T.var),convert(Vector{Bool},vec(sum(variables .== T.var,dims= 2))))
    elseif variables isa Vector{Symbol}
        if length(setdiff(variables,T.var)) > 0
            @warn "Following variables are not part of the model: " * join(string.(setdiff(variables,T.var)),", ")
            return Int[]
        end
        return Int.(indexin(variables, T.var))
    elseif variables isa Tuple{Symbol,Vararg{Symbol}}
        if length(setdiff(variables,T.var)) > 0
            @warn "Following variables are not part of the model: " * join(string.(setdiff(Symbol.(collect(variables)),T.var)), ", ")
            return Int[]
        end
        return Int.(indexin(variables, T.var))
    elseif variables isa Symbol
        if length(setdiff([variables],T.var)) > 0
            @warn "Following variable is not part of the model: " * join(string(setdiff([variables],T.var)[1]),", ")
            return Int[]
        end
        return Int.(indexin([variables], T.var))
    else
        @warn "Invalid argument in variables"
        return Int[]
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

end # dispatch_doctor

# function Stateupdate(::Val{:first_order}, states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     return [P.first_order.solution_matrix * [states[1][T.past_not_future_and_mixed_idx]; shocks]]
# end

# function Stateupdate(::Val{:second_order}, states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     aug_state₁ = [states[1][T.past_not_future_and_mixed_idx]; shocks]

#     aug_state = [states[1][T.past_not_future_and_mixed_idx]; 1; shocks]

#     𝐒₁ = P.first_order.solution_matrix
#     𝐒₂ = P.second_order_solution * P.second_order_auxilliary_matrices.𝐔₂

#     return [𝐒₁ * aug_state₁ + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2]
# end

# function Stateupdate(::Val{:pruned_second_order}, pruned_states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     aug_state₁ = [pruned_states[1][T.past_not_future_and_mixed_idx]; 1; shocks]

#     aug_state₁̃ = [pruned_states[1][T.past_not_future_and_mixed_idx]; shocks]
#     aug_state₂̃ = [pruned_states[2][T.past_not_future_and_mixed_idx]; zero(shocks)]
    
#     𝐒₁ = P.first_order.solution_matrix
#     𝐒₂ = P.second_order_solution * P.second_order_auxilliary_matrices.𝐔₂

#     return [𝐒₁ * aug_state₁̃, 𝐒₁ * aug_state₂̃ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2]
# end

# function Stateupdate(::Val{:third_order}, states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     aug_state₁ = [states[1][T.past_not_future_and_mixed_idx]; shocks]

#     aug_state = [states[1][T.past_not_future_and_mixed_idx]; 1; shocks]

#     𝐒₁ = P.first_order.solution_matrix
#     𝐒₂ = P.second_order_solution * P.second_order_auxilliary_matrices.𝐔₂
#     𝐒₃ = P.third_order_solution * P.third_order_auxilliary_matrices.𝐔₃

#     kron_aug_state = ℒ.kron(aug_state, aug_state)

#     return [𝐒₁ * aug_state₁ + 𝐒₂ * kron_aug_state / 2 + 𝐒₃ * ℒ.kron(kron_aug_state, aug_state) / 6]
# end

# function Stateupdate(::Val{:pruned_third_order}, pruned_states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     aug_state₁ = [pruned_states[1][T.past_not_future_and_mixed_idx]; 1; shocks]
#     aug_state₁̂ = [pruned_states[1][T.past_not_future_and_mixed_idx]; 0; shocks]
#     aug_state₂ = [pruned_states[2][T.past_not_future_and_mixed_idx]; 0; zero(shocks)]

#     aug_state₁̃ = [pruned_states[1][T.past_not_future_and_mixed_idx]; shocks]
#     aug_state₂̃ = [pruned_states[2][T.past_not_future_and_mixed_idx]; zero(shocks)]
#     aug_state₃̃ = [pruned_states[3][T.past_not_future_and_mixed_idx]; zero(shocks)]
    
#     𝐒₁ = P.first_order.solution_matrix
#     𝐒₂ = P.second_order_solution * P.second_order_auxilliary_matrices.𝐔₂
#     𝐒₃ = P.third_order_solution * P.third_order_auxilliary_matrices.𝐔₃
    
#     kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

#     return [𝐒₁ * aug_state₁̃, 𝐒₁ * aug_state₂̃ + 𝐒₂ * kron_aug_state₁ / 2, 𝐒₁ * aug_state₃̃ + 𝐒₂ * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒₃ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
# end

function parse_algorithm_to_state_update(algorithm::Symbol, 𝓂::ℳ, occasionally_binding_constraints::Bool)::Tuple{Function, Bool}
    if occasionally_binding_constraints
        if algorithm == :first_order
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
        else
            # @assert false "Provided algorithm not valid. Valid algorithm: $all_available_algorithms"
            state_update = (x,y)->nothing
            pruning = false
        end
    else
        if algorithm == :first_order
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
        else
            # @assert false "Provided algorithm not valid. Valid algorithm: $all_available_algorithms"
            state_update = (x,y)->nothing
            pruning = false
        end
    end

    return state_update, pruning
end

@stable default_mode = "disable" begin

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

function get_NSSS_and_parameters(𝓂::ℳ, 
                                    parameter_values::Vector{S}; 
                                    opts::CalculationOptions = merge_calculation_options())::Tuple{Vector{S}, Tuple{S, Int}} where S <: Real
                                    # timer::TimerOutput = TimerOutput(),
    # @timeit_debug timer "Calculate NSSS" begin
    SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(parameter_values, 𝓂, opts.tol, opts.verbose, false, 𝓂.solver_parameters)

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        if opts.verbose 
            println("Failed to find NSSS") 
        end

        # return (SS_and_pars, (10.0, iters))#, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # end # timeit_debug
    return SS_and_pars, (solution_error, iters)
end

end # dispatch_doctor

function rrule(::typeof(get_NSSS_and_parameters), 
                𝓂, 
                parameter_values; 
                opts::CalculationOptions = merge_calculation_options()) 
                # timer::TimerOutput = TimerOutput(),
    # @timeit_debug timer "Calculate NSSS - forward" begin

    SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(parameter_values, 𝓂, opts.tol, opts.verbose, false, 𝓂.solver_parameters)

    # end # timeit_debug

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        return (SS_and_pars, (solution_error, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # @timeit_debug timer "Calculate NSSS - pullback" begin

    SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
        
    SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

    SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

    # unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
    unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))

    ∂ = parameter_values
    C = SS_and_pars[indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag)] # [dyn_ss_idx])

    if eltype(𝓂.∂SS_equations_∂parameters[1]) != eltype(parameter_values)
        if 𝓂.∂SS_equations_∂parameters[1] isa SparseMatrixCSC
            jac_buffer = similar(𝓂.∂SS_equations_∂parameters[1], eltype(parameter_values))
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(eltype(parameter_values), size(𝓂.∂SS_equations_∂parameters[1]))
        end
    else
        jac_buffer = 𝓂.∂SS_equations_∂parameters[1]
    end

    𝓂.∂SS_equations_∂parameters[2](jac_buffer, ∂, C)

    ∂SS_equations_∂parameters = jac_buffer

    
    if eltype(𝓂.∂SS_equations_∂SS_and_pars[1]) != eltype(SS_and_pars)
        if 𝓂.∂SS_equations_∂SS_and_pars[1] isa SparseMatrixCSC
            jac_buffer = similar(𝓂.∂SS_equations_∂SS_and_pars[1], eltype(SS_and_pars))
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(eltype(SS_and_pars), size(𝓂.∂SS_equations_∂SS_and_pars[1]))
        end
    else
        jac_buffer = 𝓂.∂SS_equations_∂SS_and_pars[1]
    end

    𝓂.∂SS_equations_∂SS_and_pars[2](jac_buffer, ∂, C)

    ∂SS_equations_∂SS_and_pars = jac_buffer

    ∂SS_equations_∂SS_and_pars_lu = RF.lu(∂SS_equations_∂SS_and_pars, check = false)

    if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
        return (SS_and_pars, (10.0, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]

    jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))
    
    for (i,v) in enumerate(SS_and_pars_names)
        if v in unknowns
            jvp[i,:] = JVP[indexin([v], unknowns),:]
        end
    end

    # end # timeit_debug
    # end # timeit_debug

    # try block-gmres here
    function get_non_stochastic_steady_state_pullback(∂SS_and_pars)
        # println(∂SS_and_pars)
        return NoTangent(), NoTangent(), jvp' * ∂SS_and_pars[1], NoTangent()
    end


    return (SS_and_pars, (solution_error, iters)), get_non_stochastic_steady_state_pullback
end

@stable default_mode = "disable" begin

function get_NSSS_and_parameters(𝓂::ℳ, 
                                parameter_values_dual::Vector{ℱ.Dual{Z,S,N}}; 
                                opts::CalculationOptions = merge_calculation_options())::Tuple{Vector{ℱ.Dual{Z,S,N}}, Tuple{S, Int}} where {Z, S <: AbstractFloat, N}
                                # timer::TimerOutput = TimerOutput(),
    parameter_values = ℱ.value.(parameter_values_dual)

    SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(parameter_values, 𝓂, opts.tol, opts.verbose, false, 𝓂.solver_parameters)

    ∂SS_and_pars = zeros(S, length(SS_and_pars), N)

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        if opts.verbose println("Failed to find NSSS") end

        solution_error = S(10.0)
    else
        SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
            
        SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
        
        SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

        # unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
        unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))
        

        ∂ = parameter_values
        C = SS_and_pars[indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag)] # [dyn_ss_idx])

        if eltype(𝓂.∂SS_equations_∂parameters[1]) != eltype(parameter_values)
            if 𝓂.∂SS_equations_∂parameters[1] isa SparseMatrixCSC
                jac_buffer = similar(𝓂.∂SS_equations_∂parameters[1], eltype(parameter_values))
                jac_buffer.nzval .= 0
            else
                jac_buffer = zeros(eltype(parameter_values), size(𝓂.∂SS_equations_∂parameters[1]))
            end
        else
            jac_buffer = 𝓂.∂SS_equations_∂parameters[1]
        end

        𝓂.∂SS_equations_∂parameters[2](jac_buffer, ∂, C)

        ∂SS_equations_∂parameters = jac_buffer

        
        if eltype(𝓂.∂SS_equations_∂SS_and_pars[1]) != eltype(parameter_values)
            if 𝓂.∂SS_equations_∂SS_and_pars[1] isa SparseMatrixCSC
                jac_buffer = similar(𝓂.∂SS_equations_∂SS_and_pars[1], eltype(SS_and_pars))
                jac_buffer.nzval .= 0
            else
                jac_buffer = zeros(eltype(SS_and_pars), size(𝓂.∂SS_equations_∂SS_and_pars[1]))
            end
        else
            jac_buffer = 𝓂.∂SS_equations_∂SS_and_pars[1]
        end

        𝓂.∂SS_equations_∂SS_and_pars[2](jac_buffer, ∂, C)

        ∂SS_equations_∂SS_and_pars = jac_buffer

        ∂SS_equations_∂SS_and_pars_lu = RF.lu(∂SS_equations_∂SS_and_pars, check = false)

        if !ℒ.issuccess(∂SS_equations_∂SS_and_pars_lu)
            if opts.verbose println("Failed to calculate implicit derivative of NSSS") end
            
            solution_error = S(10.0)
        else
            JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]

            jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))
            
            for (i,v) in enumerate(SS_and_pars_names)
                if v in unknowns
                    jvp[i,:] = JVP[indexin([v], unknowns),:]
                end
            end

            for i in 1:N
                parameter_values_partials = ℱ.partials.(parameter_values_dual, i)

                ∂SS_and_pars[:,i] = jvp * parameter_values_partials
            end
        end
    end
    
    return reshape(map(SS_and_pars, eachrow(∂SS_and_pars)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(SS_and_pars)), (solution_error, iters)
end




function check_bounds(parameter_values::Vector{S}, 𝓂::ℳ)::Bool where S <: Real
    if !all(isfinite,parameter_values) return true end

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

function get_relevant_steady_state_and_state_update(::Val{:second_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options()) where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(parameter_values, 𝓂, opts = opts) # timer = timer, 
    
    TT = 𝓂.timings

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        if opts.verbose println("Could not find 2nd order stochastic steady state") end
        return TT, SS_and_pars, [𝐒₁, 𝐒₂], collect(sss), converged
    end

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = collect(sss) - all_SS

    return TT, SS_and_pars, [𝐒₁, 𝐒₂], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:pruned_second_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options())::Tuple{timings, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{S}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(parameter_values, 𝓂, pruning = true, opts = opts) # timer = timer, 

    TT = 𝓂.timings

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        if opts.verbose println("Could not find 2nd order stochastic steady state") end
        return TT, SS_and_pars, [𝐒₁, 𝐒₂], [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars)], converged
    end

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = [zeros(𝓂.timings.nVars), collect(sss) - all_SS]

    return TT, SS_and_pars, [𝐒₁, 𝐒₂], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:third_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options())::Tuple{timings, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{S}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(parameter_values, 𝓂, opts = opts) # timer = timer,  

    TT = 𝓂.timings

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        if opts.verbose println("Could not find 3rd order stochastic steady state") end
        return TT, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], collect(sss), converged
    end

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = collect(sss) - all_SS

    return TT, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:pruned_third_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options())::Tuple{timings, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{S}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(parameter_values, 𝓂, pruning = true, opts = opts) # timer = timer, 

    TT = 𝓂.timings

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        if opts.verbose println("Could not find 3rd order stochastic steady state") end
        return TT, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars)], converged
    end

    all_SS = expand_steady_state(SS_and_pars,𝓂)

    state = [zeros(𝓂.timings.nVars), collect(sss) - all_SS, zeros(𝓂.timings.nVars)]

    return TT, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged
end


function get_relevant_steady_state_and_state_update(::Val{:first_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options())::Tuple{timings, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{Float64}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameter_values, opts = opts) # timer = timer, 

    state = zeros(𝓂.timings.nVars)

    TT = 𝓂.timings

    if solution_error > opts.tol.NSSS_acceptance_tol # || isnan(solution_error) if it's NaN the fisrt condition is false anyway
        # println("NSSS not found")
        return TT, SS_and_pars, zeros(S, 0, 0), [state], solution_error < opts.tol.NSSS_acceptance_tol
    end

    ∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂) # , timer = timer)# |> Matrix

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁;
                                                        T = TT,
                                                        # timer = timer,
                                                        initial_guess = 𝓂.solution.perturbation.qme_solution,
                                                        opts = opts,
                                                        decomposition = 𝓂.solution.perturbation.first_order_block_decomposition)

    if solved 𝓂.solution.perturbation.qme_solution = qme_sol end

    if !solved
        # println("NSSS not found")
        return TT, SS_and_pars, zeros(S, 0, 0), [state], solved
    end

    return TT, SS_and_pars, 𝐒₁, [state], solved
end

end # dispatch_doctor

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

#         get_SS(FS2000, silent = true)
#         get_SS(FS2000, parameters = :alp => 0.36, silent = true)
#         get_solution(FS2000, silent = true)
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
#         get_std(FS2000, silent = true)
#         # get_SSS(FS2000, silent = true)
#         # get_SSS(FS2000, algorithm = :third_order, silent = true)

#         # import StatsPlots
#         # plot_irf(FS2000)
#         # plot_solution(FS2000,:k) # fix warning when there is no sensitivity and all values are the same. triggers: no strict ticks found...
#         # plot_conditional_variance_decomposition(FS2000)
#     end
# end

end
