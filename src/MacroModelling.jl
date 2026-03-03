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

import Dates
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
# import Polyester
import NLopt
# import Zygote
import SparseArrays: SparseMatrixCSC, SparseVector, AbstractSparseArray, AbstractSparseMatrix, sparse!, spzeros, nnz, issparse, nonzeros #, sparse, droptol!, sparsevec, spdiagm, findnz#, sparse!
import LinearAlgebra as ℒ
import LinearSolve as 𝒮
import FastLapackInterface
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
import DataStructures: CircularBuffer, OrderedDict
import MacroTools: unblock, postwalk, prewalk, @capture, flatten

# import SpeedMapping: speedmapping
import Suppressor: @suppress
import REPL
import Unicode
import MatrixEquations # good overview: https://cscproxy.mpi-magdeburg.mpg.de/mpcsc/benner/talks/Benner-Melbourne2019.pdf
# import NLboxsolve: nlboxsolve
# using NamedArrays
# using AxisKeys

import ChainRulesCore: rrule, NoTangent, @thunk, ProjectTo, unthunk, AbstractZero
import RecursiveFactorization as RF

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

import Reexport
Reexport.@reexport import AxisKeys: KeyedArray, axiskeys, rekey, NamedDimsArray
Reexport.@reexport import SparseArrays: sparse, spzeros, droptol!, sparsevec, spdiagm, findnz

# Module for SymPy symbol workspace to avoid polluting MacroModelling namespace
module SymPyWorkspace
    # Import SpecialFunctions
    using ..SpecialFunctions: erfcinv, erfc
    
    # Define density-related functions directly in the workspace
    # These need to be available for symbolic expressions
    function norminvcdf(p::T)::T where T
        -erfcinv(2*p) * 1.4142135623730951
    end
    norminv(p) = norminvcdf(p)
    qnorm(p) = norminvcdf(p)

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

    Max = max
    Min = min
end

# Reserved names that cannot be used as variables, shocks, or parameters
# These are functions and operators available in SymPyWorkspace
const SYMPYWORKSPACE_RESERVED_NAMES = Set([
    # Mathematical functions
    :exp, :exp2, :exp10, :log, :log2, :log10, :sin, :cos, :tan, :asin, :atan, :asinh, :acosh, :atanh, :sqrt, :abs, :min, :max,
    :sum, :prod, :length, :abs2,
    # Special functions
    :erfcinv, :erfc,
    # Density functions
    :norminvcdf, :norminv, :qnorm,
    :normlogpdf, :normpdf, :normcdf,
    :pnorm, :dnorm,
    # Aliases
    :Max, :Min,
    # Core types
    :Expr, :Symbol
])

# Type definitions
const Symbol_input = Union{Symbol,Vector{Symbol},Matrix{Symbol},Tuple{Symbol,Vararg{Symbol}}}
const String_input = Union{S,Vector{S},Matrix{S},Tuple{S,Vararg{S}}} where S <: AbstractString
const ParameterType = Union{Nothing,
                            Pair{Symbol, Float64},
                            Pair{S, Float64},
                            Tuple{Pair{Symbol, Float64}, Vararg{Pair{Symbol, Float64}}},
                            Tuple{Pair{S, Float64}, Vararg{Pair{S, Float64}}},
                            Vector{Pair{Symbol, Float64}},
                            Vector{Pair{S, Float64}},
                            Pair{Symbol, Int},
                            Pair{S, Int},
                            Tuple{Pair{Symbol, Int}, Vararg{Pair{Symbol, Int}}},
                            Tuple{Pair{S, Int}, Vararg{Pair{S, Int}}},
                            Vector{Pair{Symbol, Int}},
                            Vector{Pair{S, Int}},
                            Pair{Symbol, Real},
                            Pair{S, Real},
                            Tuple{Pair{Symbol, Real}, Vararg{Pair{Symbol, Real}}},
                            Tuple{Pair{S, Real}, Vararg{Pair{S, Real}}},
                            Vector{Pair{Symbol, Real}},
                            Vector{Pair{S, Real}},
                            Dict{S, Float64},
                            Dict{Symbol, Float64},
                            Tuple{Int, Vararg{Int}},
                            Matrix{Int},
                            Tuple{Float64, Vararg{Float64}},
                            Matrix{Float64},
                            Tuple{Real, Vararg{Real}},
                            Matrix{Real},
                            Vector{Float64} } where S <: AbstractString

# Type for steady state function argument
# Accepts a function, `nothing` (explicitly clear)
const SteadyStateFunctionType = Union{Nothing, Function, Missing}

using DispatchDoctor
# @stable default_mode = "disable" begin

# Imports
include("default_options.jl")
include("common_docstrings.jl")
include("structures.jl")
include("solver_parameters.jl")
include("options_and_caches.jl")
include("nsss_solver.jl")
include("macros.jl")
include("get_functions.jl")
include("dynare.jl")
include("inspect.jl")
include("moments.jl")
include("./algorithms/fast_lapack_wrappers.jl")
include("perturbation.jl")

include("./algorithms/sylvester.jl")
include("./algorithms/lyapunov.jl")
include("./algorithms/nonlinear_solver.jl")
include("./algorithms/quadratic_matrix_equation.jl")

include("./filter/find_shocks.jl")
include("./filter/inversion.jl")
include("./filter/kalman.jl")


# end # dispatch_doctor


export @model, @parameters, solve!

export plot_irfs, plot_irf, plot_IRF, plot_simulations, plot_solution, plot_simulation, plot_girf #, plot
export plot_conditional_forecast, plot_conditional_variance_decomposition, plot_forecast_error_variance_decomposition, plot_fevd, plot_model_estimates, plot_shock_decomposition
export plotlyjs_backend, gr_backend
export plot_irfs!, plot_irf!, plot_IRF!, plot_girf!, plot_simulations!, plot_simulation!, plot_conditional_forecast!, plot_model_estimates!, plot_solution!

export Normal, Beta, Cauchy, Gamma, InverseGamma

export get_irfs, get_irf, get_IRF, simulate, get_simulation, get_simulations, get_girf
export get_conditional_forecast
export get_solution, get_first_order_solution, get_perturbation_solution, get_second_order_solution, get_third_order_solution
export get_steady_state, get_SS, get_ss, get_non_stochastic_steady_state, get_stochastic_steady_state, get_SSS, steady_state, SS, SSS, ss, sss
export get_non_stochastic_steady_state_residuals, get_residuals, check_residuals
export get_moments, get_statistics, get_covariance, get_standard_deviation, get_variance, get_var, get_std, get_stdev, get_cov, var, std, stdev, cov, get_mean #, mean
export get_autocorrelation, get_correlation, get_variance_decomposition, get_corr, get_autocorr, get_var_decomp, corr, autocorr
export get_fevd, fevd, get_forecast_error_variance_decomposition, get_conditional_variance_decomposition
export calculate_jacobian, calculate_hessian, calculate_third_order_derivatives
export calculate_first_order_solution, calculate_second_order_solution, calculate_third_order_solution #, calculate_jacobian_manual, calculate_jacobian_sparse, calculate_jacobian_threaded
export get_shock_decomposition, get_model_estimates, get_estimated_shocks, get_estimated_variables, get_estimated_variable_standard_deviations, get_loglikelihood
export Tolerances

export translate_mod_file, translate_dynare_file, import_model, import_dynare
export write_mod_file, write_dynare_file, write_to_dynare_file, write_to_dynare, export_dynare, export_to_dynare, export_mod_file, export_model

export get_equations, get_steady_state_equations, get_dynamic_equations, get_calibration_equations, get_parameters, get_calibrated_parameters, get_parameters_in_equations, get_parameters_defined_by_parameters, get_parameters_defining_parameters, get_calibration_equation_parameters, get_variables, get_nonnegativity_auxiliary_variables, get_dynamic_auxiliary_variables, get_shocks, get_state_variables, get_jump_variables, get_missing_parameters, has_missing_parameters, get_solution_counts, print_solution_counts
# Internal
export irf, girf

# StatsPlotsExt

function plot_irfs  end
function plot_irf   end
function plot_IRF   end
function plot_girf  end
function plot_simulations   end
function plot_simulation    end
function plot_conditional_forecast  end
function plot_model_estimates   end
function plot_shock_decomposition   end
function plot_solution  end
function plot_conditional_variance_decomposition    end
function plot_forecast_error_variance_decomposition end
function plot_fevd  end
function plotlyjs_backend   end
function gr_backend end

function plot_irfs!  end
function plot_irf!   end
function plot_IRF!   end
function plot_girf!  end
function plot_simulations!   end
function plot_simulation!    end
function plot_conditional_forecast!  end
function plot_model_estimates!   end
function plot_solution!  end

# TuringExt

function Normal  end
function Beta   end
function Cauchy   end
function Gamma  end
function InverseGamma  end

# Remove comment for debugging
# export block_solver, remove_redundant_SS_vars!, write_parameters_input!, parse_variables_input_to_index, undo_transformer , transformer, calculate_third_order_stochastic_steady_state, calculate_second_order_stochastic_steady_state, filter_and_smooth
# export create_symbols_eqs!, write_steady_state_solver_function!, write_functions_mapping!, solve!, parse_algorithm_to_state_update, block_solver, block_solver_AD, calculate_covariance, calculate_jacobian, calculate_first_order_solution, expand_steady_state, get_symbols, calculate_covariance_AD, parse_shocks_input_to_index

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
@static if isdefined(Symbolics, Symbol("@register_derivative"))
    Symbolics.@register_derivative norminvcdf(p) 1 1 / normpdf(norminvcdf(p))
    # norminv and qnorm are aliases of norminvcdf, so they share the same rule:
    Symbolics.@register_derivative norminv(p) 1 1 / normpdf(norminvcdf(p))
    Symbolics.@register_derivative qnorm(p) 1 1 / normpdf(norminvcdf(p))

    # ── normlogpdf ──
    # d/dz (normlogpdf(z)) = −z
    Symbolics.@register_derivative normlogpdf(z) 1 -z

    # ── normpdf & dnorm ──
    # normpdf(z) = (1/√(2π)) e^(−z²/2) ⇒ derivative = −z * normpdf(z)
    Symbolics.@register_derivative normpdf(z) 1 -z * normpdf(z)
    # alias:
    Symbolics.@register_derivative dnorm(z) 1 -z * normpdf(z)

    # ── normcdf & pnorm ──
    # d/dz (normcdf(z)) = normpdf(z)
    Symbolics.@register_derivative normcdf(z) 1 normpdf(z)
    # alias:
    Symbolics.@register_derivative pnorm(z) 1 normpdf(z)
else
    function Symbolics.derivative(::typeof(norminvcdf), args::NTuple{1,Any}, ::Val{1})
        p = args[1]
        1 / normpdf(norminvcdf(p))
    end
    Symbolics.derivative(::typeof(norminv), args::NTuple{1,Any}, ::Val{1}) =
        Symbolics.derivative(norminvcdf, args, Val{1}())
    Symbolics.derivative(::typeof(qnorm),  args::NTuple{1,Any}, ::Val{1}) =
        Symbolics.derivative(norminvcdf, args, Val{1}())

    # ── normlogpdf ──
    function Symbolics.derivative(::typeof(normlogpdf), args::NTuple{1,Any}, ::Val{1})
        z = args[1]
        -z
    end

    # ── normpdf & dnorm ──
    function Symbolics.derivative(::typeof(normpdf), args::NTuple{1,Any}, ::Val{1})
        z = args[1]
        -z * normpdf(z)
    end
    Symbolics.derivative(::typeof(dnorm), args::NTuple{1,Any}, ::Val{1}) =
        Symbolics.derivative(normpdf, args, Val{1}())

    # ── normcdf & pnorm ──
    function Symbolics.derivative(::typeof(normcdf), args::NTuple{1,Any}, ::Val{1})
        z = args[1]
        normpdf(z)
    end
    Symbolics.derivative(::typeof(pnorm), args::NTuple{1,Any}, ::Val{1}) =
        Symbolics.derivative(normcdf, args, Val{1}())
end

@stable default_mode = "disable" begin


Base.show(io::IO, 𝓂::ℳ) = println(io, 
                "Model:        ", 𝓂.model_name, 
                "\nVariables", 
                "\n Total:       ", 𝓂.constants.post_model_macro.nVars,
                "\n  Auxiliary:  ", length(𝓂.constants.post_model_macro.exo_present) + length(𝓂.constants.post_model_macro.aux),
                "\n States:      ", 𝓂.constants.post_model_macro.nPast_not_future_and_mixed,
                "\n  Auxiliary:  ",  length(intersect(𝓂.constants.post_model_macro.past_not_future_and_mixed, 𝓂.constants.post_model_macro.aux_present)),
                "\n Jumpers:     ", 𝓂.constants.post_model_macro.nFuture_not_past_and_mixed, # 𝓂.constants.post_model_macro.mixed, 
                "\n  Auxiliary:  ", length(intersect(𝓂.constants.post_model_macro.future_not_past_and_mixed, union(𝓂.constants.post_model_macro.aux_present, 𝓂.constants.post_model_macro.aux_future))),
                "\nShocks:       ", 𝓂.constants.post_model_macro.nExo,
                "\nParameters:   ", length(𝓂.constants.post_model_macro.parameters_in_equations),
                if isempty(𝓂.constants.post_complete_parameters.missing_parameters)
                    ""
                else
                    "\n Missing:     " * repr(length(𝓂.constants.post_complete_parameters.missing_parameters))
                end,
                if 𝓂.equations.calibration == Expr[]
                    ""
                else
                    "\nCalibration\nequations:    " * repr(length(𝓂.equations.calibration))
                end,
                # "\n¹: including auxiliary variables"
                # "\nVariable bounds (upper,lower,any): ",sum(𝓂.upper_bounds .< Inf),", ",sum(𝓂.lower_bounds .> -Inf),", ",length(𝓂.bounds),
                )

check_for_dynamic_variables(ex::Int) = false
check_for_dynamic_variables(ex::Float64) = false
check_for_dynamic_variables(ex::Symbol) = occursin(r"₍₁₎|₍₀₎|₍₋₁₎",string(ex))

# end # dispatch_doctor

function compare_args_and_kwargs(dicts::Vector{S}) where S <: Dict
    N = length(dicts)
    @assert N ≥ 2 "Need at least two dictionaries to compare"

    diffs = Dict{Symbol,Any}()

    # assume all dictionaries share the same set of keys
    for k in keys(dicts[1])
        if k in [:plot_data, :plot_type]
            # skip keys that are not relevant for comparison
            continue
        end

        vals = [d[k] for d in dicts]

        if all(v -> v isa Dict, vals)
            # recurse into nested dictionaries
            nested = compare_args_and_kwargs(vals)
            if !isempty(nested)
                diffs[k] = nested
            end

        elseif all(v -> v isa KeyedArray, vals)
            # compare by length and elementwise equality
            base = vals[1]
            identical = all(v -> length(v) == length(base) && all(collect(v) .== collect(base)), vals[2:end])
            if !identical
                diffs[k] = vals
            end

        elseif all(v -> v isa AbstractArray, vals)
            # compare by length and elementwise equality
            base = vals[1]
            identical = all(v -> length(v) == length(base) && all(v .== base), vals[2:end])
            if !identical
                diffs[k] = vals
            end

        else
            # scalar or other types
            identical = all(v -> v == vals[1], vals[2:end])
            if !identical
                diffs[k] = vals
            end
        end
    end

    return diffs
end


function mul_reverse_AD!(   C::Matrix{S},
                            A::AbstractMatrix{M},
                            B::AbstractMatrix{N}) where {S <: Real, M <: Real, N <: Real}
    ℒ.mul!(C,A,B)
end




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

function normalize_filtering_options(filter::Symbol,
                                      smooth::Bool,
                                      algorithm::Symbol,
                                      shock_decomposition::Bool,
                                      warmup_iterations::Int;
                                      maxlog::Int = DEFAULT_MAXLOG)
    @assert filter ∈ [:kalman, :inversion] "Currently only the kalman filter (:kalman) for linear models and the inversion filter (:inversion) for linear and nonlinear models are supported."

    pruning = algorithm ∈ (:pruned_second_order, :pruned_third_order)

    if shock_decomposition && algorithm ∈ (:second_order, :third_order)
        @info "Shock decomposition is not available for $(algorithm) solutions, but is available for first order, pruned second order, and pruned third order solutions. Setting `shock_decomposition = false`." maxlog = maxlog
        shock_decomposition = false
    end

    if algorithm != :first_order && filter != :inversion
        @info "Higher order solution algorithms only support the inversion filter. Setting `filter = :inversion`." maxlog = maxlog
        filter = :inversion
    end

    if filter != :kalman && smooth
        @info "Only the Kalman filter supports smoothing. Setting `smooth = false`." maxlog = maxlog
        smooth = false
    end

    if warmup_iterations > 0
        if filter == :kalman
            @info "`warmup_iterations` is not a valid argument for the Kalman filter. Ignoring input for `warmup_iterations`." maxlog = maxlog
            warmup_iterations = 0
        elseif algorithm != :first_order
            @info "Warmup iterations are currently only available for first order solutions in combination with the inversion filter. Ignoring input for `warmup_iterations`." maxlog = maxlog
            warmup_iterations = 0
        end
    end

    return filter, smooth, algorithm, shock_decomposition, pruning, warmup_iterations
end


function adjust_generalised_irf_flag(generalised_irf::Bool,
                                    generalised_irf_warmup_iterations::Int,
                                    generalised_irf_draws::Int,
                                    algorithm::Symbol,
                                    occasionally_binding_constraints::Bool,
                                    shocks::Union{Symbol_input, String_input, Matrix{Float64}, KeyedArray{Float64}};
                                    maxlog::Int = DEFAULT_MAXLOG)
    if generalised_irf
        if algorithm == :first_order && !occasionally_binding_constraints
            @info "Generalised IRFs coincide with normal IRFs for first-order solutions of models without/inactive occasionally binding constraints (OBC). Use `ignore_obc = false` for models with OBCs or a higher-order algorithm (e.g. `algorithm = :pruned_second_order`) to compute generalised IRFs that differ from normal IRFs. Setting `generalised_irf = false`." maxlog = maxlog
            generalised_irf = false
        elseif shocks == :none
            @info "Cannot compute generalised IRFs for model without shocks. Setting `generalised_irf = false`." maxlog = maxlog
            generalised_irf = false
        end
    end

    if !generalised_irf
        if generalised_irf_warmup_iterations != 100
        @info "`generalised_irf_warmup_iterations` is ignored because `generalised_irf = false`." maxlog = maxlog
        elseif generalised_irf_draws != 50
            @info "`generalised_irf_draws` is ignored because `generalised_irf = false`." maxlog = maxlog
        end
    end

    return generalised_irf
end

end # dispatch_doctor

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

function process_shocks_input(shocks::Union{Symbol_input, String_input, Matrix{Float64}, KeyedArray{Float64}},
                                negative_shock::Bool,
                                shock_size::Real,
                                periods::Int,
                                𝓂::ℳ; 
                                maxlog::Int = DEFAULT_MAXLOG)
    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks
    
    shocks = 𝓂.constants.post_model_macro.nExo == 0 ? :none : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == 𝓂.constants.post_model_macro.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        periods_extended = periods + size(shocks)[2]
        
        shock_history = zeros(𝓂.constants.post_model_macro.nExo, periods_extended)

        shock_history[:,1:size(shocks)[2]] = shocks
        
        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shocks_axis = collect(axiskeys(shocks,1))

        shocks_symbols = shocks_axis isa String_input ? shocks_axis .|> Meta.parse .|> replace_indices : shocks_axis

        shock_input = map(x->Symbol(replace(string(x), "₍ₓ₎" => "")), shocks_symbols)

        @assert length(setdiff(shock_input, 𝓂.constants.post_model_macro.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."

        periods_extended = periods + size(shocks)[2]
        
        shock_history = zeros(𝓂.constants.post_model_macro.nExo, periods_extended)
        
        shock_history[indexin(shock_input,𝓂.constants.post_model_macro.exo), 1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa Expr
        error("Expressions are not a valid input for shocks. Please provide a Symbol, Vector of Symbols, Matrix of Float64, KeyedArray of Float64, or :none.")
    elseif (typeof(shocks) <: Symbol_input) || (typeof(shocks) <: String_input)
        shock_history = zeros(𝓂.constants.post_model_macro.nExo, periods)

        periods_extended = periods
        
        shock_idx = parse_shocks_input_to_index(shocks, 𝓂.constants)
    end

    if shocks isa KeyedArray{Float64} || shocks isa Matrix{Float64} || shocks == :none
        if negative_shock != DEFAULT_NEGATIVE_SHOCK
            @info "`negative_shock = $negative_shock` has no effect when providing a custom shock matrix. Setting `negative_shock = $DEFAULT_NEGATIVE_SHOCK`." maxlog = maxlog

            negative_shock = DEFAULT_NEGATIVE_SHOCK
        end

        if shock_size != DEFAULT_SHOCK_SIZE
            @info "`shock_size = $shock_size` has no effect when providing a custom shock matrix. Setting `shock_size = $DEFAULT_SHOCK_SIZE`." maxlog = maxlog

            shock_size = DEFAULT_SHOCK_SIZE
        end
    end

    return shocks, negative_shock, shock_size, periods_extended, shock_idx, shock_history
end

@stable default_mode = "disable" begin

function process_ignore_obc_flag(shocks,
                                 ignore_obc::Bool,
                                 𝓂::ℳ; 
                                 maxlog::Int = DEFAULT_MAXLOG)
    stochastic_model = length(𝓂.constants.post_model_macro.exo) > 0
    obc_model = length(𝓂.equations.obc_violation) > 0

    obc_shocks_included = false

    if stochastic_model && obc_model
        if shocks isa Matrix{Float64}
            obc_indices = contains.(string.(𝓂.constants.post_model_macro.exo), "ᵒᵇᶜ")
            if any(obc_indices)
                obc_shocks_included = sum(abs2, shocks[obc_indices, :]) > 1e-10
            end
        elseif shocks isa KeyedArray{Float64}
            shock_axis = collect(axiskeys(shocks, 1))
            shock_axis = shock_axis isa Vector{String} ? shock_axis .|> Meta.parse .|> replace_indices : shock_axis

            obc_shocks = 𝓂.constants.post_model_macro.exo[contains.(string.(𝓂.constants.post_model_macro.exo), "ᵒᵇᶜ")]
            relevant_shocks = intersect(obc_shocks, shock_axis)

            if !isempty(relevant_shocks)
                obc_shocks_included = sum(abs2, shocks(relevant_shocks, :)) > 1e-10
            end
        else
            shock_idx = parse_shocks_input_to_index(shocks, 𝓂.constants)

            selected_shocks = if (shock_idx isa Vector) || (shock_idx isa UnitRange)
                length(shock_idx) > 0 ? 𝓂.constants.post_model_macro.exo[shock_idx] : Symbol[]
            else
                [𝓂.constants.post_model_macro.exo[shock_idx]]
            end

            obc_shocks = 𝓂.constants.post_model_macro.exo[contains.(string.(𝓂.constants.post_model_macro.exo), "ᵒᵇᶜ")]
            obc_shocks_included = !isempty(intersect(selected_shocks, obc_shocks))
        end
    end

    ignore_obc_flag = ignore_obc

    if ignore_obc_flag && !obc_model
        @info "`ignore_obc = true` has no effect because $(𝓂.model_name) has no occasionally binding constraints. Setting `ignore_obc = false`." maxlog = maxlog
        ignore_obc_flag = false
    end

    if ignore_obc_flag && obc_shocks_included
        @warn "`ignore_obc = true` cannot be applied because shocks affecting occasionally binding constraints are included. Enforcing the constraints instead and setting `ignore_obc = false`." maxlog = maxlog
        ignore_obc_flag = false
    end

    occasionally_binding_constraints = obc_model && !ignore_obc_flag

    return ignore_obc_flag, occasionally_binding_constraints, obc_shocks_included
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
        sym_value = SPyPyC.symbols(string(symbs), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($symbs = $sym_value))
    end

    eq = Core.eval(SymPyWorkspace, transformed_expr)

    if avoid_solve || count_ops(Meta.parse(string(eq))) > 15
        soll = nothing
    else
        soll = solve_symbolically(eq, Core.eval(SymPyWorkspace, :minmax__P))
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
        # jac .= 𝒜.jacobian(𝒷(), xx -> 𝓂.functions.obc_violation(xx, p), X)[1]'
        jac .= 𝒟.jacobian(xx -> 𝓂.functions.obc_violation(xx, p), backend, X)'
    end

    res .= 𝓂.functions.obc_violation(X, p)

	return nothing
end

function obc_objective_optim_fun(X::Vector{S}, grad::Vector{S})::S where S
    if length(grad) > 0
        grad .= 2 .* X
    end
    
    sum(abs2, X)
end

function set_up_obc_violation_function!(𝓂)
    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎$")))

    sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))

    # write indices in auxiliary objects
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎")))

    dyn_var_present = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_var_present_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

    SS_and_pars_names = ms.SS_and_pars_names

    dyn_var_present_idx = indexin(dyn_var_present   , SS_and_pars_names)

    alll = []
    for (i,var) in enumerate(present_varss)
        if !(match(r"^χᵒᵇᶜ", string(var)) === nothing)
            push!(alll,:($var = Y[$(dyn_var_present_idx[i]),1:max(periods, 1)]))
        end
    end

    calc_obc_violation = :(function calculate_obc_violation(x, p)
        state, state_update, reference_steady_state, 𝓂, algorithm, periods, shock_values = p

        T = 𝓂.constants.post_model_macro

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

        $(𝓂.equations.obc_violation...)

        return vcat(constraint_values...)
    end)

    𝓂.functions.obc_violation = @RuntimeGeneratedFunction(calc_obc_violation)

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
    for (i,eq) in enumerate(𝓂.equations.dynamic)
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
    while length(𝓂.caches.solver_cache) > 1
        pop!(𝓂.caches.solver_cache)
    end

    𝓂.caches.first_order_solution_matrix = zeros(0,0)
    𝓂.caches.first_order_obc_solution_matrix = zeros(0,0)
    𝓂.caches.qme_solution = zeros(0,0)
    𝓂.caches.second_order_solution = spzeros(0,0)
    𝓂.caches.third_order_solution = spzeros(0,0)

    𝓂.caches.second_order_stochastic_steady_state = Float64[]
    𝓂.caches.pruned_second_order_stochastic_steady_state = Float64[]
    𝓂.caches.third_order_stochastic_steady_state = Float64[]
    𝓂.caches.pruned_third_order_stochastic_steady_state = Float64[]

    𝓂.caches.valid_for.first_order_solution = Float64[]
    𝓂.caches.valid_for.second_order_solution = Float64[]
    𝓂.caches.valid_for.pruned_second_order_solution = Float64[]
    𝓂.caches.valid_for.third_order_solution = Float64[]
    𝓂.caches.valid_for.pruned_third_order_solution = Float64[]

    return nothing
end


const CACHE_VALIDITY_FIELDS = (
    :non_stochastic_steady_state,
    :first_order_solution,
    :second_order_solution,
    :pruned_second_order_solution,
    :third_order_solution,
    :pruned_third_order_solution,
)


@inline function cache_valid_for_parameters(valid_for::Vector{Float64}, parameters::AbstractVector{<:Real})::Bool
    length(valid_for) == length(parameters) || return false
    @inbounds for i in eachindex(parameters)
        if valid_for[i] != parameters[i]
            return false
        end
    end
    return true
end


"""
    set_custom_steady_state_function!(𝓂::ℳ, f::SteadyStateFunctionType)

*Internal function* - Set a custom function to calculate the steady state of the model.

This function is not exported. Users should instead pass the `steady_state_function` argument to functions like:
- `get_irf(𝓂, steady_state_function = f)`
- `get_steady_state(𝓂, steady_state_function = f)`
- `simulate(𝓂, steady_state_function = f)`

This function allows users to provide their own steady state solver, which can be useful when:
- The default numerical solver has difficulty finding the steady state
- An analytical solution for the steady state is known
- A more efficient custom solver is available

# Arguments
- `𝓂`: Model object
- `f`: A function that accepts either `(parameters)` or `(out, parameters)` and provides steady state values in the same order as `get_NSSS_and_parameters`: variables first, then calibrated parameters (if any).

# Keyword Arguments
- `verbose` [Default: `false`, Type: `Bool`]: Print information about the variable and parameter ordering.

# Details
The custom function `f` can have either signature:
```julia
f(parameters::AbstractVector{<:Real}) -> AbstractVector{<:Real}
f!(out::AbstractVector{<:Real}, parameters::AbstractVector{<:Real}) -> Union{Nothing, AbstractVector{<:Real}}
```
When both signatures are applicable, the in-place signature is used.

Where:
- Input: Parameter values in the declaration order (as defined in `@parameters`). Parameter order is available from `get_parameters(𝓂)`.
- Output: Steady state values in the same order as `get_NSSS_and_parameters`: variables in `sort(union(𝓂.constants.post_model_macro.var, 𝓂.constants.post_model_macro.exo_past, 𝓂.constants.post_model_macro.exo_future))`, followed by calibrated parameters in `𝓂.equations.calibration_parameters` (if any). For in-place functions, `out` is filled in this order.

# Examples
```julia
using MacroModelling

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

# Define a custom steady state function
# get_variables(RBC) returns [:c, :k, :q, :z] (sorted alphabetically)
# get_parameters(RBC) returns [:std_z, :ρ, :δ, :α, :β] (in declaration order)
# Return values must match the order used by get_NSSS_and_parameters:
# variables in sort(union(RBC.var, RBC.exo_past, RBC.exo_future)), then any calibrated parameters.
function my_steady_state(params)
    std_z, ρ, δ, α, β = params
    
    # Analytical steady state
    k_ss = ((1/β - 1 + δ) / α)^(1/(α - 1))
    q_ss = k_ss^α
    c_ss = q_ss - δ * k_ss
    z_ss = 0.0
    
    return [c_ss, k_ss, q_ss, z_ss]  # Order matches get_NSSS_and_parameters(RBC)
end

# Use with get_irf, get_steady_state, or simulate
get_irf(RBC, steady_state_function = my_steady_state)
```

# Returns
- `nothing`

See also: [`get_variables`](@ref), [`get_parameters`](@ref), [`get_steady_state`](@ref), [`get_irf`](@ref), [`simulate`](@ref)
"""
function set_custom_steady_state_function!(𝓂::ℳ, f::SteadyStateFunctionType)
    if f === nothing
        𝓂.functions.NSSS_custom = nothing
    elseif f isa Function && f !== 𝓂.functions.NSSS_custom
        𝓂.functions.NSSS_custom = f
    end

    return nothing
end



"""
    infer_step(x_axis)

Infer the step for an axis.

For dates, if the last two points share the same day-of-month, the step is
inferred in whole months (e.g. Month(1), Month(3), …). Otherwise the raw
difference is used. For non time types, uses the plain difference.
"""
function infer_step(x_axis::AbstractVector{T}) where {T<:Number}
    x_axis[end] - x_axis[end-1]
end

function infer_step(x_axis::AbstractVector{T}) where {T<:Dates.TimeType}
    d1 = x_axis[end-1]
    d2 = x_axis[end]

    # try to infer a monthly step if aligned by day-of-month
    if Dates.day(d1) == Dates.day(d2)
        m1 = 12 * Dates.year(d1) + Dates.month(d1)
        m2 = 12 * Dates.year(d2) + Dates.month(d2)
        mstep = m2 - m1
        if mstep != 0
            return Dates.Month(mstep)
        end
    end

    # fall back to the raw difference (in days, milliseconds, …)
    return d2 - d1
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

# Helper to convert dense matrix to sparse using I,J,V format (avoids Julia 1.12 SparseArrays bug)
function dense_to_sparse(A::DenseMatrix{S}, tol::R) where {S <: Real, R <: AbstractFloat}
    m, n = size(A)
    I = Int[]
    J = Int[]
    V = S[]
    @inbounds for j in 1:n
        for i in 1:m
            v = A[i,j]
            if abs(v) > tol
                push!(I, i)
                push!(J, j)
                push!(V, v)
            end
        end
    end
    return sparse(I, J, V, m, n)
end

function choose_matrix_format(A::DenseMatrix{S}; 
                                density_threshold::Float64 = .1, 
                                min_length::Int = 1000,
                                tol::R = 1e-14,
                                multithreaded::Bool = true)::Union{Matrix{S}, SparseMatrixCSC{S, Int}, ThreadedSparseArrays.ThreadedSparseMatrixCSC{S, Int, SparseMatrixCSC{S, Int}}} where {R <: AbstractFloat, S <: Real}
    if sum(abs.(A) .> tol) / length(A) < density_threshold && length(A) > min_length
        # Use dense_to_sparse to avoid Julia 1.12 SparseArrays bug in SparseMatrixCSC(::Matrix)
        a = dense_to_sparse(A, tol)
        if multithreaded
            return ThreadedSparseArrays.ThreadedSparseMatrixCSC(a)
        else
            return a
        end
    else
        return convert(Matrix, A)
    end
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

    estimated_nnz = 0
    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{T}()
    X = zeros(T, 0, 0)

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

function sparse_preallocated!(Ŝ::Matrix{T}; ℂ::higher_order_workspace{T,F,H} = Higher_order_workspace()) where {T <: Real, F <: AbstractFloat, H <: Real}
    if !(eltype(ℂ.tmp_sparse_prealloc6[3]) == T)
        ℂ.tmp_sparse_prealloc6 = Higher_order_workspace(T = T, S = F)
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


function compressed_kron²(a::AbstractMatrix{T};
                    rowmask::Vector{Int} = Int[],
                    colmask::Vector{Int} = Int[],
                    tol::AbstractFloat = eps(),
                    sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{T}} = (Int[], Int[], T[], Int[], Int[], Int[], T[])) where T <: Real

    a_is_adjoint = typeof(a) <: ℒ.Adjoint{T,Matrix{T}}

    if a_is_adjoint
        â = copy(a')
        a = sparse(a')

        rmask = colmask
        colmask = rowmask
        rowmask = rmask
    elseif typeof(a) <: DenseMatrix{T}
        â = copy(a)
        a = sparse(a)
    else
        â = convert(Matrix, a)  # Convert to dense matrix for faster access
    end

    # Get the number of rows and columns
    n_rows, n_cols = size(a)

    # Calculate the number of unique pair indices for rows and columns
    m2_rows = n_rows * (n_rows + 1) ÷ 2    # For rows: i ≤ j
    m2_cols = n_cols * (n_cols + 1) ÷ 2    # For columns: i ≤ j

    if rowmask == Int[0] || colmask == Int[0]
        if a_is_adjoint
            return spzeros(T, m2_cols, m2_rows)
        else
            return spzeros(T, m2_rows, m2_cols)
        end
    end

    # Initialize arrays to collect indices and values
    lennz = nnz(a)

    m2_c = length(colmask) > 0 ? length(colmask) : m2_cols
    m2_r = length(rowmask) > 0 ? length(rowmask) : m2_rows

    m2_exp = (length(colmask) > 0 || length(rowmask) > 0) ? 2 : 3

    if length(sparse_preallocation[1]) == 0
        estimated_nnz = floor(Int, max(m2_r * m2_c * (lennz / length(a)) ^ m2_exp, 10000))

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

    k = 0

    # Find unique non-zero row and column indices
    rowinds, colinds, _ = findnz(a)
    ui = unique(rowinds)
    uj = unique(colinds)

    norowmask = length(rowmask) == 0
    nocolmask = length(colmask) == 0

    for i1 in ui
        for j1 in ui
            if j1 ≤ i1

                row = (i1 - 1) * i1 ÷ 2 + j1

                if norowmask || row in rowmask
                    for i2 in uj
                        for j2 in uj
                            if j2 ≤ i2

                                col = (i2 - 1) * i2 ÷ 2 + j2

                                if nocolmask || col in colmask
                                    @inbounds aii = â[i1, i2]
                                    @inbounds aij = â[i1, j2]
                                    @inbounds aji = â[j1, i2]
                                    @inbounds ajj = â[j1, j2]

                                    # Sum over both permutations of (i2, j2)
                                    val = aii * ajj + aij * aji

                                    if abs(val) > tol
                                        divisor = i1 == j1 ? 2 : 1

                                        k += 1

                                        if k > estimated_nnz
                                            estimated_nnz += Int(ceil(max(1000, estimated_nnz * .1)))
                                            estimated_nnz = min(m2_cols * m2_rows, estimated_nnz)
                                            resize!(I, estimated_nnz)
                                            resize!(J, estimated_nnz)
                                            resize!(V, estimated_nnz)
                                        end

                                        I[k] = row
                                        J[k] = col
                                        V[k] = val / divisor
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    resize!(I, k)
    resize!(J, k)
    resize!(V, k)

    # Create the sparse matrix from the collected indices and values
    if a_is_adjoint
        klasttouch = sparse_preallocation[4]
        csrrowptr  = sparse_preallocation[5]
        csrcolval  = sparse_preallocation[6]
        csrnzval   = sparse_preallocation[7]

        resize!(klasttouch, m2_rows)
        resize!(csrrowptr, m2_cols + 1)
        resize!(csrcolval, length(J))
        resize!(csrnzval, length(J))

        out = sparse!(J, I, V, m2_cols, m2_rows, +, klasttouch, csrrowptr, csrcolval, csrnzval, J, I, V)
    else
        klasttouch = sparse_preallocation[4]
        csrrowptr  = sparse_preallocation[5]
        csrcolval  = sparse_preallocation[6]
        csrnzval   = sparse_preallocation[7]

        resize!(klasttouch, m2_cols)
        resize!(csrrowptr, m2_rows + 1)
        resize!(csrcolval, length(I))
        resize!(csrnzval, length(I))

        out = sparse!(I, J, V, m2_rows, m2_cols, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
    end

    return out
end


# function kron³(A::AbstractSparseMatrix{T}, M₃::third_order) where T <: Real
#     rows, cols, vals = findnz(A)

#     # Dictionary to accumulate sums of values for each coordinate
#     result_dict = Dict{Tuple{Int, Int}, T}()

#     # Using a single iteration over non-zero elements
#     nvals = length(vals)

#     lk = ReentrantLock()

#     Polyester.@batch for i in 1:nvals
#     # for i in 1:nvals
#         for j in 1:nvals
#             for k in 1:nvals
#                 r1, c1, v1 = rows[i], cols[i], vals[i]
#                 r2, c2, v2 = rows[j], cols[j], vals[j]
#                 r3, c3, v3 = rows[k], cols[k], vals[k]
                
#                 sorted_cols = [c1, c2, c3]
#                 sorted_rows = [r1, r2, r3] # a lot of time spent here
#                 sort!(sorted_rows, rev = true) # a lot of time spent here
                
#                 if haskey(M₃.𝐈₃, sorted_cols) # && haskey(M₃.𝐈₃, sorted_rows) # a lot of time spent here
#                     row_idx = M₃.𝐈₃[sorted_rows]
#                     col_idx = M₃.𝐈₃[sorted_cols]

#                     key = (row_idx, col_idx)

#                     # begin
#                     #     lock(lk)
#                     #     try
#                             if haskey(result_dict, key)
#                                 result_dict[key] += v1 * v2 * v3
#                             else
#                                 result_dict[key] = v1 * v2 * v3
#                             end
#                     #     finally
#                     #         unlock(lk)
#                     #     end
#                     # end
#                 end
#             end
#         end
#     end

#     # Extract indices and values from the dictionary
#     result_rows = Int[]
#     result_cols = Int[]
#     result_vals = T[]

#     for (ks, valu) in result_dict
#         push!(result_rows, ks[1])
#         push!(result_cols, ks[2])
#         push!(result_vals, valu)
#     end
    
#     # Create the sparse matrix from the collected indices and values
#     return sparse!(result_rows, result_cols, result_vals, size(M₃.𝐂₃, 2), size(M₃.𝐔₃, 1))
# end

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
            # Check if v[i].second and v[j].second are equal or if one is subset of the other
            if v[i].second == v[j].second
                # Exact match: combine first elements and remove duplicate
                v[i] = v[i].first ∪ v[j].first => v[i].second
                deleteat!(v, j)
                subset_found = true
                break
            elseif all(elem -> elem in v[j].second, v[i].second) || all(elem -> elem in v[i].second, v[j].second)
                # One is subset of the other: combine the first elements and assign to the one with the larger second element
                if length(v[i].second) > length(v[j].second)
                    v[i] = v[i].first ∪ v[j].first => v[i].second
                    deleteat!(v, j)
                else
                    v[j] = v[i].first ∪ v[j].first => v[j].second
                    deleteat!(v, i)
                end
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
                                    constants::constants,
                                    variables::Union{Symbol_input,String_input};
                                    covariance::Union{Symbol_input,String_input} = Symbol[],
                                    tol::AbstractFloat = eps())

    T = constants.post_model_macro
    

    orders = Pair{Vector{Symbol}, Vector{Symbol}}[]

    nˢ = T.nPast_not_future_and_mixed
    
    if variables == :full_covar
        return [T.var => T.past_not_future_and_mixed]
    else
        var_idx = MacroModelling.parse_variables_input_to_index(variables, constants) |> sort
        observables = T.var[var_idx]
    end

    # Precompute state indices to avoid repeated indexin calls
    state_idx_in_var = indexin(T.past_not_future_and_mixed, T.var) .|> Int
    𝐒₁_states = 𝐒₁[state_idx_in_var, 1:nˢ]

    for obs in observables
        obs_in_var_idx = indexin([obs],T.var) .|> Int
        dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0

        # Iterative propagation without redundant allocations
        while true
            new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states) .> tol)
            if new_deps == dependencies_in_states
                break
            end
            dependencies_in_states = new_deps
        end

        dependencies = T.past_not_future_and_mixed[dependencies_in_states]

        push!(orders,[obs] => sort(dependencies))
    end
    
    # If covariance variables are specified, compute dependencies and add entries for those pairs
    if !(covariance == Symbol[])
        covar_var_idx = MacroModelling.parse_variables_input_to_index(covariance, constants) |> sort
        covariance_vars = T.var[covar_var_idx]
        
        # Compute dependencies for covariance variables (if not already computed)
        for covar_var in covariance_vars
            # Check if this variable's dependencies are already computed
            if isnothing(findfirst(x -> covar_var in x.first, orders))
                obs_in_var_idx = indexin([covar_var], T.var) .|> Int
                dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0

                # Iterative propagation without redundant allocations
                while true
                    new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states) .> tol)
                    if new_deps == dependencies_in_states
                        break
                    end
                    dependencies_in_states = new_deps
                end

                dependencies = T.past_not_future_and_mixed[dependencies_in_states]
                push!(orders,[covar_var] => sort(dependencies))
            end
        end
        
        # Build lookup dictionary for faster searches
        var_to_idx = Dict{Symbol, Int}()
        for (idx, order) in enumerate(orders)
            for var in order.first
                var_to_idx[var] = idx
            end
        end
        
        # Add entries for all pairs of covariance variables
        for i in 1:length(covariance_vars)
            for j in (i+1):length(covariance_vars)
                # Find dependencies for both variables using lookup dictionary
                idx_i = var_to_idx[covariance_vars[i]]
                idx_j = var_to_idx[covariance_vars[j]]
                
                deps_i = orders[idx_i].second
                deps_j = orders[idx_j].second
                # Union of dependencies for covariance computation
                combined_deps = sort(union(deps_i, deps_j))
                push!(orders, [covariance_vars[i], covariance_vars[j]] => combined_deps)
            end
        end
    end

    sort!(orders, by = x -> length(x[2]), rev = true)

    return combine_pairs(orders)
end


function determine_efficient_order(𝐒₁::Matrix{<: Real},
                                    𝐒₂::AbstractMatrix{<: Real},
                                    constants::constants,
                                    variables::Union{Symbol_input,String_input};
                                    covariance::Union{Symbol_input,String_input} = Symbol[],
                                    tol::AbstractFloat = eps())

    T = constants.post_model_macro
    

    orders = Pair{Vector{Symbol}, Vector{Symbol}}[]

    nˢ = T.nPast_not_future_and_mixed
    nᵉ = T.nExo
    
    if variables == :full_covar
        return [T.var => T.past_not_future_and_mixed]
    else
        var_idx = MacroModelling.parse_variables_input_to_index(variables, constants) |> sort
        observables = T.var[var_idx]
    end

    # Build selector for state variables in the augmented state vector [states; 1; shocks]
    s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
    
    # Kronecker product indices for state-state interactions
    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    # Precompute state indices and matrix slices to avoid repeated operations
    state_idx_in_var = indexin(T.past_not_future_and_mixed, T.var) .|> Int
    𝐒₁_states = 𝐒₁[state_idx_in_var, 1:nˢ]
    𝐒₂_states = nnz(𝐒₂) > 0 ? 𝐒₂[state_idx_in_var, kron_s_s] : nothing

    for obs in observables
        obs_in_var_idx = indexin([obs],T.var) .|> Int
        
        # First order dependencies
        dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0
        
        # Second order dependencies from quadratic terms (s ⊗ s)
        if nnz(𝐒₂) > 0
            s_s_to_y₂ = 𝐒₂[obs_in_var_idx, kron_s_s]
            
            # Check which state variable pairs have influence
            # Vectorized approach: reshape to nˢ×nˢ and check column/row sums
            s_s_matrix = reshape(vec(sum(abs, s_s_to_y₂, dims=1) .> tol), nˢ, nˢ)
            dependencies_in_states = dependencies_in_states .| vec(sum(s_s_matrix, dims=2) .> 0) .| vec(sum(s_s_matrix, dims=1) .> 0)
        end

        # Propagate dependencies through the system (iterative closure)
        # considering both first and second order propagation
        while true
            prev_dependencies = dependencies_in_states
            
            # First order propagation
            new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states) .> tol)
            
            # Second order propagation: if state i and state j are dependencies,
            # their product can affect states
            if !isnothing(𝐒₂_states)
                # Generate selector vector for columns where both states are dependencies
                selector = vec(ℒ.kron(prev_dependencies, prev_dependencies))
                if any(selector)
                    # Check which states are affected by the selected products
                    affected = vec(sum(abs, 𝐒₂_states[:, selector], dims=2) .> tol)
                    new_deps = new_deps .| affected
                end
            end
            
            if new_deps == dependencies_in_states
                break
            end
            dependencies_in_states = new_deps
        end

        dependencies = T.past_not_future_and_mixed[dependencies_in_states]

        push!(orders,[obs] => sort(dependencies))
    end
    
    # If covariance variables are specified, compute dependencies and add entries for those pairs
    if !(covariance == Symbol[])
        covar_var_idx = MacroModelling.parse_variables_input_to_index(covariance, constants) |> sort
        covariance_vars = T.var[covar_var_idx]
        
        # Compute dependencies for covariance variables (if not already computed)
        for covar_var in covariance_vars
            # Check if this variable's dependencies are already computed
            if isnothing(findfirst(x -> covar_var in x.first, orders))
                obs_in_var_idx = indexin([covar_var], T.var) .|> Int
                
                # First order dependencies
                dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0
                
                # Second order dependencies from quadratic terms (s ⊗ s)
                if nnz(𝐒₂) > 0
                    s_s_to_y₂ = 𝐒₂[obs_in_var_idx, kron_s_s]
                    # Vectorized approach: reshape to nˢ×nˢ and check column/row sums
                    s_s_matrix = reshape(vec(sum(abs, s_s_to_y₂, dims=1) .> tol), nˢ, nˢ)
                    dependencies_in_states = dependencies_in_states .| vec(sum(s_s_matrix, dims=2) .> 0) .| vec(sum(s_s_matrix, dims=1) .> 0)
                end

                # Propagate dependencies through the system
                # Precompute matrix slices
                𝐒₁_states_local = 𝐒₁[state_idx_in_var, 1:nˢ]
                𝐒₂_states_local = nnz(𝐒₂) > 0 ? 𝐒₂[state_idx_in_var, kron_s_s] : nothing
                
                while true
                    prev_dependencies = dependencies_in_states
                    
                    # First order propagation
                    new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states_local) .> tol)
                    
                    # Second order propagation
                    if !isnothing(𝐒₂_states_local)
                        # Generate selector vector for columns where both states are dependencies
                        selector = vec(ℒ.kron(prev_dependencies, prev_dependencies))
                        if any(selector)
                            affected = vec(sum(abs, 𝐒₂_states_local[:, selector], dims=2) .> tol)
                            new_deps = new_deps .| affected
                        end
                    end
                    
                    if new_deps == dependencies_in_states
                        break
                    end
                    dependencies_in_states = new_deps
                end

                dependencies = T.past_not_future_and_mixed[dependencies_in_states]
                push!(orders,[covar_var] => sort(dependencies))
            end
        end
        
        # Add entries for all pairs of covariance variables
        for i in 1:length(covariance_vars)
            for j in (i+1):length(covariance_vars)
                # Find dependencies for both variables (they should exist now)
                idx_i = findfirst(x -> covariance_vars[i] in x.first, orders)
                idx_j = findfirst(x -> covariance_vars[j] in x.first, orders)
                
                deps_i = orders[idx_i].second
                deps_j = orders[idx_j].second
                # Union of dependencies for covariance computation
                combined_deps = sort(union(deps_i, deps_j))
                push!(orders, [covariance_vars[i], covariance_vars[j]] => combined_deps)
            end
        end
    end

    sort!(orders, by = x -> length(x[2]), rev = true)

    return combine_pairs(orders)
end


function determine_efficient_order(𝐒₁::Matrix{<: Real},
                                    𝐒₂::AbstractSparseMatrix{<: Real},
                                    𝐒₃::AbstractSparseMatrix{<: Real},
                                    constants::constants,
                                    variables::Union{Symbol_input,String_input};
                                    covariance::Union{Symbol_input,String_input} = Symbol[],
                                    tol::AbstractFloat = eps())

    T = constants.post_model_macro
    

    orders = Pair{Vector{Symbol}, Vector{Symbol}}[]

    nˢ = T.nPast_not_future_and_mixed
    nᵉ = T.nExo
    
    if variables == :full_covar
        return [T.var => T.past_not_future_and_mixed]
    else
        var_idx = MacroModelling.parse_variables_input_to_index(variables, constants) |> sort
        observables = T.var[var_idx]
    end

    # Build selectors for state variables in the augmented state vector [states; 1; shocks]
    s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
    
    # Kronecker product indices for interactions
    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_s_s_s = ℒ.kron(kron_s_s, s_in_s⁺)
    
    # Precompute state indices and matrix slices
    state_idx_in_var = indexin(T.past_not_future_and_mixed, T.var) .|> Int
    𝐒₁_states = 𝐒₁[state_idx_in_var, 1:nˢ]
    has_S₂ = nnz(𝐒₂) > 0
    has_S₃ = nnz(𝐒₃) > 0
    𝐒₂_states = has_S₂ ? 𝐒₂[state_idx_in_var, kron_s_s] : nothing
    𝐒₃_states = has_S₃ ? 𝐒₃[state_idx_in_var, kron_s_s_s] : nothing

    function compute_dependencies(obs_in_var_idx::Vector{Int})
        # First order dependencies
        dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0

        # Second order dependencies from quadratic terms (s ⊗ s)
        if has_S₂
            s_s_to_y₂ = 𝐒₂[obs_in_var_idx, kron_s_s]
            s_s_matrix = reshape(vec(sum(abs, s_s_to_y₂, dims=1) .> tol), nˢ, nˢ)
            dependencies_in_states = dependencies_in_states .| vec(sum(s_s_matrix, dims=2) .> 0) .| vec(sum(s_s_matrix, dims=1) .> 0)
        end

        # Third order dependencies from cubic terms (s ⊗ s ⊗ s)
        if has_S₃
            s_s_s_to_y₃ = 𝐒₃[obs_in_var_idx, kron_s_s_s]
            s_s_s_tensor = reshape(vec(sum(abs, s_s_s_to_y₃, dims=1) .> tol), nˢ, nˢ, nˢ)
            dependencies_in_states = dependencies_in_states .| vec(sum(s_s_s_tensor, dims=(2,3)) .> 0) .|
                                                             vec(sum(s_s_s_tensor, dims=(1,3)) .> 0) .|
                                                             vec(sum(s_s_s_tensor, dims=(1,2)) .> 0)
        end

        # Propagate dependencies through the system (iterative closure)
        while true
            prev_dependencies = dependencies_in_states

            # First order propagation
            new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states) .> tol)

            # Second order propagation
            if !isnothing(𝐒₂_states)
                selector = vec(ℒ.kron(prev_dependencies, prev_dependencies))
                if any(selector)
                    affected = vec(sum(abs, 𝐒₂_states[:, selector], dims=2) .> tol)
                    new_deps = new_deps .| affected
                end
            end

            # Third order propagation
            if !isnothing(𝐒₃_states)
                selector = vec(ℒ.kron(ℒ.kron(prev_dependencies, prev_dependencies), prev_dependencies))
                if any(selector)
                    affected = vec(sum(abs, 𝐒₃_states[:, selector], dims=2) .> tol)
                    new_deps = new_deps .| affected
                end
            end

            if new_deps == dependencies_in_states
                break
            end
            dependencies_in_states = new_deps
        end

        return T.past_not_future_and_mixed[dependencies_in_states]
    end

    for obs in observables
        obs_in_var_idx = indexin([obs],T.var) .|> Int
        dependencies = compute_dependencies(obs_in_var_idx)

        push!(orders,[obs] => sort(dependencies))
    end
    
    # If covariance variables are specified, compute dependencies and add entries for those pairs
    if !(covariance == Symbol[])
        covar_var_idx = MacroModelling.parse_variables_input_to_index(covariance, constants) |> sort
        covariance_vars = T.var[covar_var_idx]
        
        # Compute dependencies for covariance variables (if not already computed)
        for covar_var in covariance_vars
            # Check if this variable's dependencies are already computed
            if isnothing(findfirst(x -> covar_var in x.first, orders))
                obs_in_var_idx = indexin([covar_var], T.var) .|> Int
                dependencies = compute_dependencies(obs_in_var_idx)
                push!(orders,[covar_var] => sort(dependencies))
            end
        end
        
        # Add entries for all pairs of covariance variables
        for i in 1:length(covariance_vars)
            for j in (i+1):length(covariance_vars)
                # Find dependencies for both variables (they should exist now)
                idx_i = findfirst(x -> covariance_vars[i] in x.first, orders)
                idx_j = findfirst(x -> covariance_vars[j] in x.first, orders)
                
                deps_i = orders[idx_i].second
                deps_j = orders[idx_j].second
                # Union of dependencies for covariance computation
                combined_deps = sort(union(deps_i, deps_j))
                push!(orders, [covariance_vars[i], covariance_vars[j]] => combined_deps)
            end
        end
    end

    sort!(orders, by = x -> length(x[2]), rev = true)

    return combine_pairs(orders)
end


function get_and_check_observables(T::post_model_macro, data::KeyedArray{Float64})::Vector{Symbol}
    @assert size(data,1) <= T.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    observables = collect(axiskeys(data,1))

    @assert observables isa Vector{String} || observables isa Vector{Symbol}  "Make sure that the data has variables names as rows. They can be either Strings or Symbols."

    observables_symbols = observables isa String_input ? observables .|> Meta.parse .|> replace_indices : observables

    @assert length(setdiff(observables_symbols, T.var)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(observables_symbols, T.var))

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
    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    full_NSSS = ms.full_NSSS_display

    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, 
                                    stochastic = algorithm != :first_order,
                                    return_variables_only = true, 
                                    derivatives = false, 
                                    verbose = opts.verbose,
                                    tol = opts.tol,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³])

    reference_steady_state = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : relevant_SS(s) for s in full_NSSS]

    relevant_NSSS = get_steady_state(𝓂, algorithm = :first_order, 
                                    stochastic = false, 
                                    return_variables_only = true, 
                                    derivatives = false, 
                                    verbose = opts.verbose,
                                    tol = opts.tol,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³])

    NSSS = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : relevant_NSSS(s) for s in full_NSSS]

    SSS_delta = NSSS - reference_steady_state

    return reference_steady_state, NSSS, SSS_delta
end

# compatibility with SymPy
Max = max
Min = min

function simplify(ex::Expr)::Union{Expr,Symbol,Int}
    ex_ss = convert_to_ss_equation(ex)

    for x in get_symbols(ex_ss)
        sym_value = SPyPyC.symbols(string(x), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($x = $sym_value))
    end

    parsed = ex_ss |> x -> Core.eval(SymPyWorkspace, x) |> string |> Meta.parse

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
                        :($(x.args[3]) * $(x.args[2])) : # avoid 2X syntax. doesn't work with sympy
                    x :
                x :
            unblock(x) : 
        x,
    eq)
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

replace_indices_special(x::Symbol) = x

replace_indices(x::String) = Symbol(replace(x, "{" => "◖", "}" => "◗"))

replace_indices_in_symbol(x::Symbol) = replace(string(x), "◖" => "{", "◗" => "}")


"""
    apply_custom_name(symbol::Symbol, custom_names::Dict{Symbol, String})

Apply custom name from dictionary if available, otherwise use default name.
"""
function apply_custom_name(symbol::R, custom_names::AbstractDict{S, T})::R where {R <: Union{Symbol, String}, S, T}
    # First, check for an exact match with the original symbol
    if haskey(custom_names, symbol)
        return R(custom_names[symbol])
    end
    
    # Handle cross-type check for exact match (String vs Symbol)
    if symbol isa Symbol && haskey(custom_names, String(replace_indices_in_symbol(symbol)))
        return R(custom_names[String(replace_indices_in_symbol(symbol))])
    elseif symbol isa String && haskey(custom_names, Symbol(symbol))
        return R(custom_names[Symbol(symbol)])
    end

    # If no exact match, strip lag operators and compare base names.
    s_str = string(symbol)
    lag_regex = r"^(.*)(ᴸ⁽.*⁾)$"
    m = match(lag_regex, s_str)

    base_symbol_str, lag_part = if m !== nothing
        (m.captures[1], m.captures[2])
    else
        (s_str, "")
    end

    for (key, value) in custom_names
        key_str = string(key)
        key_m = match(lag_regex, key_str)
        
        base_key_str = if key_m !== nothing
            key_m.captures[1]
        else
            key_str
        end

        if base_key_str == base_symbol_str
            return R(string(value) * lag_part)
        end
    end

    return symbol
end

function normalize_superscript(x::Symbol)
    return normalize_superscript(string(x))
end

function normalize_superscript(x::AbstractString)
    sub_map = Dict(
        '₀' => '0', '₁' => '1', '₂' => '2', '₃' => '3', '₄' => '4',
        '₅' => '5', '₆' => '6', '₇' => '7', '₈' => '8', '₉' => '9',
        '₊' => '+', '₋' => '-', '₌' => '=', '₍' => '(', '₎' => ')',
        'ₐ' => 'a', 'ₑ' => 'e', 'ₕ' => 'h', 'ᵢ' => 'i', 'ⱼ' => 'j',
        'ₖ' => 'k', 'ₗ' => 'l', 'ₘ' => 'm', 'ₙ' => 'n', 'ₒ' => 'o',
        'ₚ' => 'p', 'ᵣ' => 'r', 'ₛ' => 's', 'ₜ' => 't', 'ᵤ' => 'u',
        'ᵥ' => 'v', 'ₓ' => 'x'
    )
    super_map = Dict(
        '⁰' => '0', '¹' => '1', '²' => '2', '³' => '3', '⁴' => '4',
        '⁵' => '5', '⁶' => '6', '⁷' => '7', '⁸' => '8', '⁹' => '9',
        '⁺' => '+', '⁻' => '-', '⁼' => '=', '⁽' => '(', '⁾' => ')',
        'ᵃ' => 'a', 'ᵇ' => 'b', 'ᶜ' => 'c', 'ᵈ' => 'd', 'ᵉ' => 'e',
        'ᶠ' => 'f', 'ᵍ' => 'g', 'ʰ' => 'h', 'ᶦ' => 'i', 'ʲ' => 'j',
        'ᵏ' => 'k', 'ˡ' => 'l', 'ᵐ' => 'm', 'ⁿ' => 'n', 'ᵒ' => 'o',
        'ᵖ' => 'p', 'ʳ' => 'r', 'ˢ' => 's', 'ᵗ' => 't', 'ᵘ' => 'u',
        'ᵛ' => 'v', 'ʷ' => 'w', 'ˣ' => 'x', 'ʸ' => 'y', 'ᶻ' => 'z'
    )

    buf = IOBuffer()
    for c in x
        if haskey(sub_map, c)
            write(buf, sub_map[c])
        elseif haskey(super_map, c)
            write(buf, super_map[c])
        else
            write(buf, c)
        end
    end
    return String(take!(buf))
end

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

function replace_indices_special(exxpr::Expr)::Union{Expr,Symbol}
    postwalk(x -> begin
        x isa Symbol ?
            replace_indices(string(x)) :
        x isa Expr ?
            x.head == :curly ?
                Symbol(string(x.args[1]) * "◖" * string(x.args[2]) * "◗") :
            x.head == :call ?
                x.args[1] == :(*) ?
                    Symbol(string(x.args[2]), string(x.args[3])) :
                x :
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
                    end
                end
                # Handle parameters from par_calib_list - expand indexed ones, keep non-indexed
                for p in par_calib_list[u]
                    if p ∈ indexed_names
                        push!(expanded_par_var, Symbol(string(p) * "◖" * string(i) * "◗"))
                    else
                        push!(expanded_par_var, p)
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
        else#if par ∈ expanded_list ## breaks parameters defined in parameter block
            push!(expanded_inputs, par)
            push!(expanded_values, compressed_values[i])
        end
    end
    return expanded_inputs, expanded_values
end


function expand_steady_state(SS_and_pars::Vector{M}, ms::post_complete_parameters) where M
    X = ms.steady_state_expand_matrix
    return X * SS_and_pars
end



function create_symbols_eqs!(𝓂::ℳ)::symbolics
    # create symbols in SymPyWorkspace to avoid polluting MacroModelling namespace
    symbols_in_dynamic_equations = reduce(union, get_symbols.(𝓂.equations.dynamic))

    symbols_in_dynamic_equations_wo_subscripts = Symbol.(replace.(string.(symbols_in_dynamic_equations), r"₍₋?(₀|₁|ₛₛ|ₓ)₎$"=>""))

    symbols_in_ss_equations = reduce(union,get_symbols.(𝓂.equations.steady_state_aux))

    symbols_in_equation = union(𝓂.constants.post_model_macro.parameters_in_equations, 
                                𝓂.constants.post_complete_parameters.parameters, 
                                𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters,
                                symbols_in_dynamic_equations,
                                symbols_in_dynamic_equations_wo_subscripts,
                                symbols_in_ss_equations) #, 𝓂.dynamic_variables_future)

    symbols_pos = []
    symbols_neg = []
    symbols_none = []

    for symb in symbols_in_equation
        if haskey(𝓂.constants.post_parameters_macro.bounds, symb)
            if 𝓂.constants.post_parameters_macro.bounds[symb][1] >= 0
                push!(symbols_pos, symb)
            elseif 𝓂.constants.post_parameters_macro.bounds[symb][2] <= 0
                push!(symbols_neg, symb)
            else 
                push!(symbols_none, symb)
            end
        else
            push!(symbols_none, symb)
        end
    end

    # Create symbols in SymPyWorkspace instead of MacroModelling namespace
    for pos in symbols_pos
        sym_value = SPyPyC.symbols(string(pos), real = true, finite = true, positive = true)
        Core.eval(SymPyWorkspace, :($pos = $sym_value))
    end

    for neg in symbols_neg
        sym_value = SPyPyC.symbols(string(neg), real = true, finite = true, negative = true)
        Core.eval(SymPyWorkspace, :($neg = $sym_value))
    end

    for none in symbols_none
        sym_value = SPyPyC.symbols(string(none), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($none = $sym_value))
    end

    symbolics(
                map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.equations.steady_state_aux),
                # map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.dyn_equations_future),

                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_shift_var_present_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_shift_var_past_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_shift_var_future_list),

                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_shift2_var_past_list),

                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.dyn_var_present_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.dyn_var_past_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.dyn_var_future_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_ss_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.dyn_exo_list),

                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_exo_future_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_exo_present_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_exo_past_list),

                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.dyn_future_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.dyn_present_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.dyn_past_list),

                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.var_present_list_aux_SS),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.var_past_list_aux_SS),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.var_future_list_aux_SS),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.ss_list_aux_SS),

                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.var_list_aux_SS),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dynamic_variables_list),
                # map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dynamic_variables_future_list),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_model_macro.par_list_aux_SS),

                map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.equations.calibration),
                map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.equations.calibration_parameters),
                # map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.constants.post_complete_parameters.parameters),

                # Set(Core.eval(SymPyWorkspace, :([$(𝓂.constants.post_model_macro.var_present...)]))),
                # Set(Core.eval(SymPyWorkspace, :([$(𝓂.constants.post_model_macro.var_past...)]))),
                # Set(Core.eval(SymPyWorkspace, :([$(𝓂.constants.post_model_macro.var_future...)]))),
                Set(Core.eval(SymPyWorkspace, :([$(𝓂.constants.post_model_macro.vars_in_ss_equations...)]))),

                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_parameters_macro.ss_calib_list),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.constants.post_parameters_macro.par_calib_list),

                [Set() for _ in 1:length(𝓂.equations.steady_state_aux)],
                # [Set() for _ in 1:length(𝓂.calibration_equations)],
                # [Set() for _ in 1:length(𝓂.equations.steady_state_aux)],
                # [Set() for _ in 1:length(𝓂.calibration_equations)]
                )
end



function remove_redundant_SS_vars!(𝓂::ℳ, Symbolics::symbolics; avoid_solve::Bool = false)
    ss_equations = Symbolics.ss_equations

    # check variables which appear in two time periods. they might be redundant in steady state
    redundant_vars = intersect.(
        union.(
            intersect.(Symbolics.var_future_list_aux_SS, Symbolics.var_present_list_aux_SS),
            intersect.(Symbolics.var_future_list_aux_SS, Symbolics.var_past_list_aux_SS),
            intersect.(Symbolics.var_present_list_aux_SS, Symbolics.var_past_list_aux_SS),
            intersect.(Symbolics.ss_list_aux_SS, Symbolics.var_present_list_aux_SS),
            intersect.(Symbolics.ss_list_aux_SS, Symbolics.var_past_list_aux_SS),
            intersect.(Symbolics.ss_list_aux_SS, Symbolics.var_future_list_aux_SS)
        ),
    Symbolics.var_list_aux_SS)

    redundant_idx = getindex(1:length(redundant_vars), (length.(redundant_vars) .> 0) .& (length.(Symbolics.var_list_aux_SS) .> 1))
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



function write_ss_check_function!(𝓂::ℳ;
                                    cse = true,
                                    skipzeros = true, 
                                    density_threshold::Float64 = .1,
                                    nnz_parallel_threshold::Int = 1000000,
                                    min_length::Int = 10000)
    unknowns = union(setdiff(𝓂.constants.post_model_macro.vars_in_ss_equations, 𝓂.constants.post_model_macro.➕_vars), 𝓂.equations.calibration_parameters)

    ss_equations = vcat(𝓂.equations.steady_state, 𝓂.equations.calibration)



    np = length(𝓂.constants.post_complete_parameters.parameters)
    nu = length(unknowns)
    # nc = length(𝓂.calibration_equations_no_var)

    Symbolics.@variables 𝔓[1:np] 𝔘[1:nu]# ℭ[1:nc]

    parameter_dict = Dict{Symbol, Symbol}()
    back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
    calib_vars = Symbol[]
    calib_expr = []


    for (i,v) in enumerate(𝓂.constants.post_complete_parameters.parameters)
        push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
    end

    for (i,v) in enumerate(unknowns)
        push!(parameter_dict, v => :($(Symbol("𝔘_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔘_$i"))), @__MODULE__) => 𝔘[i])
    end

    for (i,v) in enumerate(𝓂.equations.calibration_no_var)
        push!(calib_vars, v.args[1])
        push!(calib_expr, v.args[2])
        # push!(parameter_dict, v.args[1] => :($(Symbol("ℭ_$i"))))
        # push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("ℭ_$i"))), @__MODULE__) => ℭ[i])
    end

    calib_replacements = Dict{Symbol, Union{Expr, Symbol, Number}}()
    for (i,x) in enumerate(calib_vars)
        replacement = Dict{Symbol, Union{Expr, Symbol, Number}}(x => calib_expr[i])
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

    if lennz > nnz_parallel_threshold
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


    𝓂.functions.NSSS_check = func_exprs


    # SS_and_pars = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.constants.post_model_macro.parameters_in_equations,𝓂.constants.post_model_macro.➕_vars))))), 𝓂.calibration_equations_parameters))

    # eqs = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

    # nx = length(𝓂.parameter_values)

    # np = length(SS_and_pars)

    nϵˢ = length(ss_equations)

    # nc = length(𝓂.calibration_equations_no_var)

    # Symbolics.@variables 𝔛¹[1:nx] 𝔓¹[1:np]

    # ϵˢ = zeros(Symbolics.Num, nϵˢ)

    # calib_vals = zeros(Symbolics.Num, nc)

    # 𝓂.SS_calib_func(calib_vals, 𝔓)

    # 𝓂.functions.NSSS_check(ϵˢ, 𝔓, 𝔘, calib_vals)

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

    if lennz > nnz_parallel_threshold
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

    𝓂.caches.∂equations_∂parameters = buffer
    𝓂.functions.NSSS_∂equations_∂parameters = func_exprs



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

    if lennz > nnz_parallel_threshold
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

    𝓂.caches.∂equations_∂SS_and_pars = buffer
    𝓂.functions.NSSS_∂equations_∂SS_and_pars = func_exprs

    return nothing
end







function solve_steady_state!(𝓂::ℳ, 
                            opts::CalculationOptions,
                            ss_solver_parameters_algorithm::Symbol,
                            ss_solver_parameters_maxtime::Real;
                            silent::Bool = false)::Tuple{Vector{Float64}, Float64, Bool}
    """
    Internal function to solve and constants the steady state.
    Returns: (SS_and_pars, solution_error, found_solution)
    """
    start_time = time()
    
    if 𝓂.constants.post_parameters_macro.precompile
        return Float64[], 0.0, false
    end
    
    if !(𝓂.functions.NSSS_custom isa Function)
        if !silent 
            print("Find non-stochastic steady state:\t\t\t\t\t") 
        end
    end
    
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts, cold_start = true)
    
    found_solution = true
    
    if !(𝓂.functions.NSSS_custom isa Function)
        select_fastest_SS_solver_parameters!(𝓂, tol = opts.tol)
        
        if solution_error > opts.tol.NSSS_acceptance_tol
            found_solution = find_SS_solver_parameters!(Val(ss_solver_parameters_algorithm), 𝓂, tol = opts.tol, verbosity = 0, maxtime = ss_solver_parameters_maxtime, maxiter = 1000000000)
            
            if found_solution
                SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts, cold_start = true)
            end
        end
    end
    
    if !(𝓂.functions.NSSS_custom isa Function)
        if !silent 
            println(round(time() - start_time, digits = 3), " seconds") 
        end
    end
    
    if !found_solution
        @warn "Could not find non-stochastic steady state. Consider setting bounds on variables or calibrated parameters in the `@parameters` section (e.g. `k > 10`)."
    end
    
    𝓂.caches.non_stochastic_steady_state = SS_and_pars
    
    if found_solution
        𝓂.caches.valid_for.non_stochastic_steady_state = eltype(𝓂.parameter_values) <: ℱ.Dual ? Float64.(ℱ.value.(𝓂.parameter_values)) : Float64.(𝓂.parameter_values)
    end
    
    return SS_and_pars, solution_error, found_solution
end

# Centralised helper to write symbolic derivatives and map functions
function write_symbolic_derivatives!(𝓂::ℳ; perturbation_order::Int = 1, silent::Bool = false)
    start_time = time()

    if !silent
        if perturbation_order == 1
            print("Take symbolic derivatives up to first order:\t\t\t\t")
        elseif perturbation_order == 2
            print("Take symbolic derivatives up to second order:\t\t\t\t")
        elseif perturbation_order == 3
            print("Take symbolic derivatives up to third order:\t\t\t\t")
        end
    end

    write_auxiliary_indices!(𝓂)
    
    write_functions_mapping!(𝓂, perturbation_order)

    if !silent
        println(round(time() - start_time, digits = 3), " seconds")
    end

    return nothing
end


function reverse_diff_friendly_push!(x,y)
    push!(x,y)
end

function calculate_SS_solver_runtime_and_loglikelihood(pars::Vector{Float64}, 𝓂::ℳ; tol::Tolerances = Tolerances())::Float64
    log_lik = 0.0
    log_lik -= -sum(pars[1:19])                                 # logpdf of a gamma dist with mean and variance 1
    σ = 5
    log_lik -= -log(σ * sqrt(2 * π)) - (pars[20]^2 / (2 * σ^2)) # logpdf of a normal dist with mean = 0 and variance = 5^2

    pars[1:2] = sort(pars[1:2], rev = true)

    par_inputs = solver_parameters(pars..., 1, 0.0, 2)

    while length(𝓂.caches.solver_cache) > 1
        pop!(𝓂.caches.solver_cache)
    end

    runtime = @elapsed outmodel = try solve_nsss_wrapper(𝓂.parameter_values, 𝓂, tol, false, true, [par_inputs]) catch end

    runtime = outmodel isa Tuple{Vector{Float64}, Tuple{Float64, Int64}} ? 
                    (outmodel[2][1] > tol.NSSS_acceptance_tol) || !isfinite(outmodel[2][1]) ? 
                        10 : 
                    runtime : 
                10

    return log_lik / 1e4 + runtime * 1e3
end

"""
    find_SS_solver_parameters!(::Val{:ESCH}, 𝓂::ℳ; maxtime::Real = 120, maxiter::Int = 2500000, tol::Tolerances = Tolerances(), verbosity = 0)

Find optimal steady state solver parameters using NLopt's ESCH algorithm.

This function optimizes solver parameters to minimize runtime while maintaining solver accuracy.
It uses the ESCH global optimization algorithm from the NLopt package.

# Arguments
- `𝓂`: Model structure
- `maxtime`: Maximum time in seconds for optimization
- `maxiter`: Maximum number of iterations
- `tol`: Tolerance structure
- `verbosity`: Verbosity level for output
"""
function find_SS_solver_parameters!(::Val{:ESCH}, 𝓂::ℳ; maxtime::Real = 120, maxiter::Int = 2500000, tol::Tolerances = Tolerances(), verbosity = 0)
    pars = rand(20) .+ 1
    pars[20] -= 1

    lbs = fill(eps(), length(pars))
    lbs[20] = -20

    ubs = fill(100.0, length(pars))
    
    opt = NLopt.Opt(NLopt.:GN_ESCH, length(pars))

    opt.min_objective = (x,p) -> calculate_SS_solver_runtime_and_loglikelihood(x, 𝓂, tol = tol)

    NLopt.lower_bounds!(opt, lbs)
    NLopt.upper_bounds!(opt, ubs)

    opt.xtol_abs = eps(Float32)
    opt.ftol_abs = eps(Float32)

    # opt.maxeval = maxiter
    opt.maxtime = maxtime

    (minf,x,ret) = NLopt.optimize(opt, pars)

    verbosity > 0 && @info "NLopt return code: $ret"

    pars = x

    par_inputs = solver_parameters(pars..., 1, 0.0, 2)

    SS_and_pars, (solution_error, iters) = solve_nsss_wrapper(𝓂.parameter_values, 𝓂, tol, false, true, [par_inputs])

    if solution_error < tol.NSSS_acceptance_tol
        push!(DEFAULT_SOLVER_PARAMETERS, par_inputs)
        𝓂.constants.post_complete_parameters = update_post_complete_parameters(
            𝓂.constants.post_complete_parameters;
            nsss_fastest_solver_parameter_idx = length(DEFAULT_SOLVER_PARAMETERS),
        )
        return true
    else 
        return false
    end
end


function select_fastest_SS_solver_parameters!(𝓂::ℳ;
                                                tol::Tolerances = Tolerances(),
                                                n_samples::Int = 100)::Nothing
    @assert n_samples > 1 "n_samples must be greater than 1."
    @assert n_samples ÷ 2 >= 1 "n_samples must be at least 2."

    best_idx = 1
    best_score = Inf

    solved = false

    solved_NSSS = 𝓂.caches.solver_cache[end]

    for (i_param, p) in enumerate(DEFAULT_SOLVER_PARAMETERS)
        times = Vector{Float64}(undef, n_samples)
        valid = true
        
        for i in 1:n_samples
            start_time = time()

            while length(𝓂.caches.solver_cache) > 1
                pop!(𝓂.caches.solver_cache)
            end

            SS_and_pars, (solution_error, iters) = solve_nsss_wrapper(𝓂.parameter_values, 𝓂, tol, false, true, [p])

            elapsed_time = time() - start_time

            times[i] = elapsed_time

            if solution_error > tol.NSSS_acceptance_tol
                valid = false
                break
            end
        end

        if valid
            sort!(times)
            score = times[n_samples ÷ 2]

            if !isfinite(best_score) || score < best_score
                best_score = score
                best_idx = i_param
            end

            solved = true
        end
    end

    while length(𝓂.caches.solver_cache) > 1
        pop!(𝓂.caches.solver_cache)
    end

    push!(𝓂.caches.solver_cache, solved_NSSS)

    if solved
        𝓂.constants.post_complete_parameters = update_post_complete_parameters(
            𝓂.constants.post_complete_parameters;
            nsss_fastest_solver_parameter_idx = best_idx,
        )
    end

    return nothing
end

function update_init_buf!(init_buf::AbstractVector{T}, lbs, ubs, n_guess, ssv_val, sv_val, guess, use_ssv::Bool) where {T}
    @inbounds for i in 1:n_guess
        if use_ssv
            v = clamp(ssv_val, lbs[i], ubs[i])
            init_buf[i] = ubs[i] <= one(T) ? T(0.1) : v
        else
            g = guess[i]
            v = g < T(1e12) ? g : sv_val
            init_buf[i] = clamp(v, lbs[i], ubs[i])
        end
    end
end

function update_sol_values!(sol_values::AbstractVector{T}, sol_new::AbstractVector{T}, lbs::AbstractVector{T}, ubs::AbstractVector{T}, n_guess::Int) where {T}
    @inbounds for i in 1:n_guess
        sol_values[i] = clamp(sol_new[i], lbs[i], ubs[i])
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
    ftol = tol.NSSS_ftol
    n_guess = length(guess)
    init_buf = SS_solve_block.ss_problem.workspace.best_previous_guess
    use_ssv = separate_starting_value isa Float64
    ssv_val = use_ssv ? T(separate_starting_value) : zero(T)
    sv_val = T(solver_params.starting_value)
    update_init_buf!(init_buf, lbs, ubs, n_guess, ssv_val, sv_val, guess, use_ssv)

    if !extended_problem
        lb_core = SS_solve_block.ss_problem.workspace.l_bounds
        ub_core = SS_solve_block.ss_problem.workspace.u_bounds
        copyto!(lb_core, 1, lbs, 1, n_guess)
        copyto!(ub_core, 1, ubs, 1, n_guess)
    end

    optimizer_init = if extended_problem
        ext_init = SS_solve_block.extended_ss_problem.workspace.best_previous_guess
        @inbounds begin
            for i in 1:n_guess
                ext_init[i] = init_buf[i]
            end
            for i in 1:length(closest_parameters_and_solved_vars)
                ext_init[n_guess + i] = closest_parameters_and_solved_vars[i]
            end
        end
        ext_init
    else
        init_buf
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
                                        optimizer_init,
                                        parameters_and_solved_vars,
                                        extended_problem ? lbs : SS_solve_block.ss_problem.workspace.l_bounds,
                                        extended_problem ? ubs : SS_solve_block.ss_problem.workspace.u_bounds,
                                        solver_params,
                                        tol = tol   )

    sol_minimum = info[4] # isnan(sum(abs, info[4])) ? Inf : ℒ.norm(info[4])
    
    rel_sol_minimum = info[3]
    
    sol_values = SS_solve_block.ss_problem.workspace.best_current_guess
    if isnothing(sol_new_tmp)
        update_sol_values!(sol_values, init_buf, lbs, ubs, n_guess)
    else
        update_sol_values!(sol_values, sol_new_tmp, lbs, ubs, n_guess)
    end

    total_iters[1] += info[1]
    total_iters[2] += info[2]

    if sol_minimum < ftol && verbose
        extended_problem_str = extended_problem ? "(extended problem) " : ""

        if separate_starting_value isa Bool
            starting_value_str = ""
        else
            starting_value_str = "and starting point: $separate_starting_value"
        end

        has_small_guess = false
        all_small_guess = true
        @inbounds for i in eachindex(guess)
            is_small = guess[i] < T(1e12)
            has_small_guess |= is_small
            all_small_guess &= is_small
        end

        if all_small_guess && separate_starting_value isa Bool
            any_guess_str = "previous solution, "
        elseif has_small_guess && separate_starting_value isa Bool
            any_guess_str = "provided guess, "
        else
            any_guess_str = ""
        end

        SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.workspace.func_buffer, sol_values, parameters_and_solved_vars)
        max_resid = maximum(abs, SS_solve_block.ss_problem.workspace.func_buffer)

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
                        preferred_solver_parameter_idx::Int,
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

    SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.workspace.func_buffer, guess, parameters_and_solved_vars)

    res = SS_solve_block.ss_problem.workspace.func_buffer

    sol_minimum  = ℒ.norm(res)

    if !cold_start
        if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol
            # ∇ = 𝒟.jacobian(x->(ss_solve_blocks(parameters_and_solved_vars, x)), backend, guess)

            # ∇̂ = ℒ.lu!(∇, check = false)

            SS_solve_block.ss_problem.jac(SS_solve_block.ss_problem.workspace.jac_buffer, guess, parameters_and_solved_vars)

            ∇ = SS_solve_block.ss_problem.workspace.jac_buffer

            sol_cache = SS_solve_block.ss_problem.workspace.lu_buffer
            # sol_cache.A = sol_cache.alg isa 𝒮.FastLUFactorization ? copy(∇) : ∇
            sol_cache.A = ∇
            # copy!(sol_cache.A, ∇)
            sol_cache.b = res
            sol = 𝒮.solve!(sol_cache)

            if 𝒮.SciMLBase.successful_retcode(sol.retcode) || sol.retcode == 𝒮.SciMLBase.ReturnCode.Default
                guess_update = sol_cache.u
                if has_nonfinite(guess_update)
                    rel_sol_minimum = 1.0
                else
                    new_guess = guess - guess_update
                    rel_sol_minimum = ℒ.norm(guess_update) / max(ℒ.norm(new_guess), sol_minimum)
                end
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
    n_solver_parameters = length(parameters)
    @assert n_solver_parameters > 0 "At least one steady-state solver parameter set is required."

    SS_optimizer = levenberg_marquardt
    ext_candidates = (true, false)
    algo_candidates = (newton, levenberg_marquardt)

    if cold_start
        guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only
        start_vals = fail_fast_solvers_only ? (false,) : (false, T(1.206), T(1.5), T(0.7688), T(2.0), T(0.897))
        for g in guesses
            for i in 1:n_solver_parameters
                p = parameters[i == 1 ? preferred_solver_parameter_idx : (i <= preferred_solver_parameter_idx ? i - 1 : i)]
                for ext in ext_candidates # try first the system where values and parameters can vary, next try the system where only values can vary
                    for s in start_vals
                        if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol# || rel_sol_minimum > rtol
                            if solved_yet continue end

                            sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                            # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                                g, 
                                                                p,
                                                                ext,
                                                                s)
                                                                
                            if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                                solved_yet = true
                            end
                        end
                    end
                end
            end
        end
    else !cold_start

        start_vals = Vector{Union{Bool, T}}(undef, 7)
        start_vals[1] = false
        start_vals[3] = T(1.206)
        start_vals[4] = T(1.5)
        start_vals[5] = T(0.7688)
        start_vals[6] = T(2.0)
        start_vals[7] = T(0.897)

        s_candidates = fail_fast_solvers_only ? @view(start_vals[1:1]) : start_vals
        n_parameter_iters = fail_fast_solvers_only ? 1 : n_solver_parameters
        fail_fast_parameter_idx = n_solver_parameters == 1 ? 1 : (n_solver_parameters <= preferred_solver_parameter_idx ? n_solver_parameters - 1 : n_solver_parameters)

        for i in 1:n_parameter_iters
            p = parameters[fail_fast_solvers_only ? fail_fast_parameter_idx : (i == 1 ? preferred_solver_parameter_idx : (i <= preferred_solver_parameter_idx ? i - 1 : i))]
            start_vals[2] = T(p.starting_value)
            for s in s_candidates
                for algo in algo_candidates
                    if sol_minimum > tol.NSSS_acceptance_tol || !isfinite(sol_minimum) # || rel_sol_minimum > rtol
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


function _prepare_stochastic_steady_state_base_terms(parameters::Vector{M},
                                                     𝓂::ℳ;
                                                     opts::CalculationOptions = merge_calculation_options(),
                                                     estimation::Bool = false) where M
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts, estimation = estimation)

    if solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error)
        return (false,
            zeros(M, T.nVars),
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            constants)
    end

    ms = ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, ms)

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                         constants,
                                                         𝓂.workspaces,
                                                         𝓂.caches;
                                                         opts = opts,
                                                         initial_guess = 𝓂.caches.qme_solution)

    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        if opts.verbose println("1st order solution not found") end
        return (false,
            all_SS,
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            constants)
    end

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                  initial_guess = 𝓂.caches.second_order_solution,
                                                  opts = opts)

    update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)
    𝐒₂ = sparse(𝐒₂ * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{M, Int}

    if !solved2
        if opts.verbose println("2nd order solution not found") end
        return (false,
            all_SS,
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            constants)
    end

    𝐒₁ = [𝐒₁[:,1:T.nPast_not_future_and_mixed] zeros(T.nVars) 𝐒₁[:,T.nPast_not_future_and_mixed+1:end]]

    aug_state₁ = sparse([zeros(T.nPast_not_future_and_mixed); 1; zeros(T.nExo)])
    tmp = (T.I_nPast - 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed])
    tmp̄ = ℒ.lu(tmp, check = false)

    if !ℒ.issuccess(tmp̄)
        if opts.verbose println("SSS not found") end
        return (false,
            all_SS,
            SS_and_pars,
            solution_error,
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0,0),
            spzeros(M,0,0),
            zeros(M,0),
            constants)
    end

    SSSstates = collect(tmp \ (𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[T.past_not_future_and_mixed_idx])

        return (true,
            all_SS,
            SS_and_pars,
            solution_error,
            ∇₁,
            ∇₂,
            𝐒₁,
            𝐒₂,
            SSSstates,
            constants)
end

function calculate_stochastic_steady_state(::Val{:second_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false) where M
    common = _prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂, SSSstates, _ = common

    if !ok
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    so = 𝓂.constants.second_order
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    A = 𝐒₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
    B̂ = 𝐒₂[:,kron_s⁺_s⁺]

    SSSstates, converged = solve_stochastic_steady_state_newton(Val(:second_order), 𝐒₁, 𝐒₂, collect(SSSstates), 𝓂)

    if !converged
        if opts.verbose println("SSS not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    state = A * SSSstates + B̂ * ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1)) / 2
    return all_SS + Vector{M}(state), converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
end

function calculate_stochastic_steady_state(::Val{:pruned_second_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false) where M
    common = _prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂, SSSstates, _ = common

    if !ok
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0)
    end

    state = 𝐒₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * SSSstates +
            𝐒₂ * ℒ.kron(sparse([zeros(𝓂.constants.post_model_macro.nPast_not_future_and_mixed); 1; zeros(𝓂.constants.post_model_macro.nExo)]), sparse([zeros(𝓂.constants.post_model_macro.nPast_not_future_and_mixed); 1; zeros(𝓂.constants.post_model_macro.nExo)])) / 2

    return all_SS + Vector{M}(state), true, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂
end



function solve_stochastic_steady_state_newton(::Val{:second_order}, 
                                              𝐒₁::Matrix{R}, 
                                              𝐒₂::AbstractSparseMatrix{R}, 
                                              x::Vector{R},
                                              𝓂::ℳ;
                                              tol::AbstractFloat = 1e-14) where R <: AbstractFloat
    # @timeit_debug timer "Setup matrices" begin

    # Get cached computational constants
    constants = initialise_constants!(𝓂)
    so = constants.second_order
    T = constants.post_model_macro
    s_in_s⁺ = so.s_in_s⁺
    s_in_s = so.s_in_s
    I_nPast = T.I_nPast
    
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    
    kron_s⁺_s = so.kron_s⁺_s
    
    A = 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
    B = 𝐒₂[T.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[T.past_not_future_and_mixed_idx,kron_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    
    # Pre-allocate augmented state vector [x; 1]
    x_aug = Vector{R}(undef, length(x) + 1)
    x_aug[end] = one(R)

    # end # timeit_debug
      
    # @timeit_debug timer "Iterations" begin

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, length(x))

        ∂x = (A + B * ℒ.kron(x_aug, I_nPast) - I_nPast)

        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end

        x̂ = A * x + B̂ * ℒ.kron(x_aug, x_aug) / 2

        Δx = ∂x̂ \ (x̂ - x)
        
        if i > 3 && isapprox(x̂, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    # end # timeit_debug

    copyto!(x_aug, 1, x, 1, length(x))
    return x, isapprox(A * x + B̂ * ℒ.kron(x_aug, x_aug) / 2, x, rtol = tol)
end





function calculate_stochastic_steady_state(::Val{:third_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false) where M <: Real
    common = _prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂, SSSstates, _ = common

    if !ok
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    𝐒₁_raw = [𝐒₁[:, 1:nPast] 𝐒₁[:, nPast+2:end]]

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁_raw, 𝐒₂,
                                                 𝓂.constants,
                                                 𝓂.workspaces,
                                                 𝓂.caches;
                                                 initial_guess = 𝓂.caches.third_order_solution,
                                                 opts = opts)

    update_perturbation_counter!(𝓂.counters, solved3, estimation = estimation, order = 3)

    if !solved3
        if opts.verbose println("3rd order solution not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    if length(𝓂.workspaces.third_order.Ŝ) == 0 || !(eltype(𝐒₃) == eltype(𝓂.workspaces.third_order.Ŝ))
        𝓂.workspaces.third_order.Ŝ = 𝐒₃ * 𝓂.constants.third_order.𝐔₃
    else
        mul_reverse_AD!(𝓂.workspaces.third_order.Ŝ, 𝐒₃, 𝓂.constants.third_order.𝐔₃)
    end

    Ŝ = 𝓂.workspaces.third_order.Ŝ
    𝐒₃̂ = sparse_preallocated!(Ŝ, ℂ = 𝓂.workspaces.third_order)::SparseMatrixCSC{M, Int}

    so = 𝓂.constants.second_order
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    kron_s⁺_s⁺_s⁺ = so.kron_s⁺_s⁺_s⁺

    A = 𝐒₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
    B̂ = 𝐒₂[:,kron_s⁺_s⁺]
    Ĉ = 𝐒₃̂[:,kron_s⁺_s⁺_s⁺]

    SSSstates, converged = solve_stochastic_steady_state_newton(Val(:third_order), 𝐒₁, 𝐒₂, 𝐒₃̂, collect(SSSstates), 𝓂)

    if !converged
        if opts.verbose println("SSS not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    state = A * SSSstates + B̂ * ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1)) / 2 + Ĉ * ℒ.kron(vcat(SSSstates,1), ℒ.kron(vcat(SSSstates,1), vcat(SSSstates,1))) / 6

    return all_SS + Vector{M}(state), converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃̂
end

function calculate_stochastic_steady_state(::Val{:pruned_third_order},
                                           parameters::Vector{M},
                                           𝓂::ℳ;
                                           opts::CalculationOptions = merge_calculation_options(),
                                           estimation::Bool = false) where M <: Real
    common = _prepare_stochastic_steady_state_base_terms(parameters, 𝓂, opts = opts, estimation = estimation)
    ok, all_SS, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂, SSSstates, _ = common

    if !ok
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    𝐒₁_raw = [𝐒₁[:, 1:nPast] 𝐒₁[:, nPast+2:end]]

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁_raw, 𝐒₂,
                                                 𝓂.constants,
                                                 𝓂.workspaces,
                                                 𝓂.caches;
                                                 initial_guess = 𝓂.caches.third_order_solution,
                                                 opts = opts)

    update_perturbation_counter!(𝓂.counters, solved3, estimation = estimation, order = 3)

    if !solved3
        if opts.verbose println("3rd order solution not found") end
        return all_SS, false, SS_and_pars, solution_error, zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0), zeros(M,0,0), spzeros(M,0,0), spzeros(M,0,0)
    end

    if length(𝓂.workspaces.third_order.Ŝ) == 0 || !(eltype(𝐒₃) == eltype(𝓂.workspaces.third_order.Ŝ))
        𝓂.workspaces.third_order.Ŝ = 𝐒₃ * 𝓂.constants.third_order.𝐔₃
    else
        mul_reverse_AD!(𝓂.workspaces.third_order.Ŝ, 𝐒₃, 𝓂.constants.third_order.𝐔₃)
    end

    Ŝ = 𝓂.workspaces.third_order.Ŝ
    𝐒₃̂ = sparse_preallocated!(Ŝ, ℂ = 𝓂.workspaces.third_order)::SparseMatrixCSC{M, Int}

    aug_state₁ = sparse([zeros(𝓂.constants.post_model_macro.nPast_not_future_and_mixed); 1; zeros(𝓂.constants.post_model_macro.nExo)])
    state = 𝐒₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * SSSstates + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2

    return all_SS + Vector{M}(state), true, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃̂
end


function solve_stochastic_steady_state_newton(::Val{:third_order}, 
                                              𝐒₁::Matrix{Float64}, 
                                              𝐒₂::AbstractSparseMatrix{Float64}, 
                                              𝐒₃::AbstractSparseMatrix{Float64},
                                              x::Vector{Float64},
                                              𝓂::ℳ;
                                              tol::AbstractFloat = 1e-14)
    # Get cached computational constants
    so = ensure_computational_constants!(𝓂.constants)
    T = 𝓂.constants.post_model_macro
    s_in_s⁺ = so.s_in_s⁺
    s_in_s = so.s_in_s
    I_nPast = T.I_nPast
    
    kron_s⁺_s⁺ = so.kron_s⁺_s⁺
    
    kron_s⁺_s = so.kron_s⁺_s
    
    kron_s⁺_s⁺_s⁺ = so.kron_s⁺_s⁺_s⁺
    
    kron_s_s⁺_s⁺ = so.kron_s_s⁺_s⁺
    
    A = 𝐒₁[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6

    # Pre-allocate augmented state vector [x; 1]
    x_aug = Vector{Float64}(undef, length(x) + 1)
    x_aug[end] = 1.0

    for i in 1:max_iters
        copyto!(x_aug, 1, x, 1, length(x))
        kron_x_aug = ℒ.kron(x_aug, x_aug)
        kron_x_kron = ℒ.kron(x_aug, kron_x_aug)

        ∂x = (A + B * ℒ.kron(x_aug, I_nPast) + C * ℒ.kron(kron_x_aug, I_nPast) / 2 - I_nPast)
        
        ∂x̂ = ℒ.lu!(∂x, check = false)
        
        if !ℒ.issuccess(∂x̂)
            return x, false
        end
        
        Δx = ∂x̂ \ (A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6 - x)

        if i > 5 && isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)
            break
        end
        
        # x += Δx
        ℒ.axpy!(-1, Δx, x)
    end

    copyto!(x_aug, 1, x, 1, length(x))
    kron_x_aug = ℒ.kron(x_aug, x_aug)
    kron_x_kron = ℒ.kron(x_aug, kron_x_aug)
    return x, isapprox(A * x + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x, rtol = tol)
end



function steady_state_symbolic_mode_flags(ss_symbolic_mode::Symbol, precompile::Bool = false)
    precompile && (ss_symbolic_mode = :none)
    ss_symbolic_mode == :none && return true, false
    ss_symbolic_mode == :single_equation && return false, false
    ss_symbolic_mode == :full && return false, true
    error("Invalid ss_symbolic_mode $(ss_symbolic_mode). Expected :none, :single_equation, or :full.")
end

function set_up_steady_state_solver!(𝓂::ℳ; verbose::Bool, silent::Bool, ss_symbolic_mode::Symbol = :single_equation)
    avoid_solve, symbolic_enabled = steady_state_symbolic_mode_flags(ss_symbolic_mode, 𝓂.constants.post_parameters_macro.precompile)

    if !𝓂.constants.post_parameters_macro.precompile
        start_time = time()

        if !silent print("Remove redundant variables in non-stochastic steady state problem:\t") end

        symbolics = create_symbols_eqs!(𝓂)

        remove_redundant_SS_vars!(𝓂, symbolics, avoid_solve = avoid_solve)

        if !silent println(round(time() - start_time, digits = 3), " seconds") end

        start_time = time()

        if !silent print("Set up non-stochastic steady state problem:\t\t\t\t") end

        write_ss_check_function!(𝓂)

        write_steady_state_solver_function!(𝓂, symbolic_enabled, symbolics, verbose = verbose, avoid_solve = avoid_solve)

        𝓂.equations.obc_violation = write_obc_violation_equations(𝓂)
        
        set_up_obc_violation_function!(𝓂)

        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    else
        start_time = time()

        if !silent print("Set up non-stochastic steady state problem:\t\t\t\t") end

        write_ss_check_function!(𝓂)

        write_steady_state_solver_function!(𝓂, false, nothing, verbose = verbose, avoid_solve = avoid_solve)

        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    end

    return nothing
end

function solve!(𝓂::ℳ; 
                parameters::ParameterType = nothing, 
                steady_state_function::SteadyStateFunctionType = missing,
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
    
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    
    # Handle steady_state_function argument
    set_custom_steady_state_function!(𝓂, steady_state_function)
    
    # @timeit_debug timer "Write parameter inputs" begin

    write_parameters_input!(𝓂, parameters, verbose = opts.verbose)
    
    if 𝓂.functions.functions_written &&
        isnothing(𝓂.functions.NSSS_custom) &&
        𝓂.constants.nsss_solver.n_steps == 0

        set_up_steady_state_solver!(𝓂,
                                    verbose = opts.verbose,
                                    silent = silent,
                                    ss_symbolic_mode = 𝓂.constants.post_parameters_macro.ss_symbolic_mode)
    end
    
    if !𝓂.functions.functions_written
        verbose = opts.verbose
        
        perturbation_order = 1

        set_up_steady_state_solver!(𝓂,
                        verbose = verbose,
                        silent = silent,
                        ss_symbolic_mode = 𝓂.constants.post_parameters_macro.ss_symbolic_mode)
    
        SS_and_pars, solution_error, found_solution = solve_steady_state!(𝓂,
                                                                           opts,
                                                                           𝓂.constants.post_parameters_macro.ss_solver_parameters_algorithm,
                                                                           𝓂.constants.post_parameters_macro.ss_solver_parameters_maxtime,
                                                                           silent = silent)
            
        write_symbolic_derivatives!(𝓂; perturbation_order = perturbation_order, silent = silent)

        𝓂.functions.functions_written = true
    end

    # Check for missing parameters after processing input
    if !isempty(𝓂.constants.post_complete_parameters.missing_parameters)
        error("Cannot solve model: missing parameter values for $(𝓂.constants.post_complete_parameters.missing_parameters). Provide them via the `parameters` keyword argument (e.g., `parameters = [:α => 0.3, :β => 0.99]`).")
    end

    # end # timeit_debug

    if 𝓂.constants.second_order.𝛔 == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0) && 
        algorithm ∈ [:second_order, :pruned_second_order]
        start_time = time()
        if !silent print("Take symbolic derivatives up to second order:\t\t\t\t") end
        write_functions_mapping!(𝓂, 2)
        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    elseif 𝓂.constants.third_order.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0) && algorithm ∈ [:third_order, :pruned_third_order]
        start_time = time()
        if !silent print("Take symbolic derivatives up to third order:\t\t\t\t") end
        write_functions_mapping!(𝓂, 3)
        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    end

    if dynamics
        first_order_needs_recalc = !cache_valid_for_parameters(𝓂.caches.valid_for.first_order_solution, 𝓂.parameter_values) || isempty(𝓂.caches.first_order_solution_matrix)
        second_order_needs_recalc = !cache_valid_for_parameters(𝓂.caches.valid_for.second_order_solution, 𝓂.parameter_values) || size(𝓂.caches.second_order_solution, 2) == 0
        pruned_second_order_needs_recalc = !cache_valid_for_parameters(𝓂.caches.valid_for.pruned_second_order_solution, 𝓂.parameter_values) || isempty(𝓂.caches.pruned_second_order_stochastic_steady_state)
        third_order_needs_recalc = !cache_valid_for_parameters(𝓂.caches.valid_for.third_order_solution, 𝓂.parameter_values) || size(𝓂.caches.third_order_solution, 2) == 0
        pruned_third_order_needs_recalc = !cache_valid_for_parameters(𝓂.caches.valid_for.pruned_third_order_solution, 𝓂.parameter_values) || isempty(𝓂.caches.pruned_third_order_stochastic_steady_state)

        obc_not_solved = isempty(𝓂.caches.first_order_obc_solution_matrix)

        if  ((:first_order         == algorithm) && (first_order_needs_recalc || (obc && obc_not_solved))) ||
            ((:second_order        == algorithm) && (second_order_needs_recalc || (obc && obc_not_solved))) ||
            ((:pruned_second_order == algorithm) && (pruned_second_order_needs_recalc || (obc && obc_not_solved))) ||
            ((:third_order         == algorithm) && (third_order_needs_recalc || (obc && obc_not_solved))) ||
            ((:pruned_third_order  == algorithm) && (pruned_third_order_needs_recalc || (obc && obc_not_solved)))

            # @timeit_debug timer "Solve for NSSS (if necessary)" begin

            SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

            # end # timeit_debug

            @assert solution_error < opts.tol.NSSS_acceptance_tol "Could not find non-stochastic steady state."
            
            # @timeit_debug timer "Calculate Jacobian" begin

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)# |> Matrix
            
            # end # timeit_debug

            # @timeit_debug timer "Calculate first order solution" begin

            S₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                                constants,
                                                                𝓂.workspaces,
                                                                𝓂.caches;
                                                                opts = opts,
                                                                initial_guess = 𝓂.caches.qme_solution)

            update_perturbation_counter!(𝓂.counters, solved, order = 1)

            # end # timeit_debug

            @assert solved "Could not find stable first order solution."

            if obc
                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1, verbose = false)

                ∇̂₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)# |> Matrix
            
                Ŝ₁, qme_sol, solved = calculate_first_order_solution(∇̂₁,
                                                                    constants,
                                                                    𝓂.workspaces,
                                                                    𝓂.caches;
                                                                    opts = opts,
                                                                    initial_guess = 𝓂.caches.qme_solution)
                
                update_perturbation_counter!(𝓂.counters, solved, order = 1)

                write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 0, verbose = false)

                𝓂.caches.first_order_obc_solution_matrix = Ŝ₁
            else
                𝓂.caches.first_order_obc_solution_matrix = zeros(0,0)
            end
            
            𝓂.caches.first_order_solution_matrix = S₁
            𝓂.caches.non_stochastic_steady_state = SS_and_pars
        end

        if  ((:second_order  == algorithm) && second_order_needs_recalc) ||
            ((:third_order  == algorithm) && third_order_needs_recalc)
            

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_stochastic_steady_state(Val(:second_order), 𝓂.parameter_values, 𝓂, opts = opts) # , timer = timer)
            
            if !converged  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

            𝓂.caches.second_order_stochastic_steady_state = stochastic_steady_state

            𝓂.caches.valid_for.second_order_solution = eltype(𝓂.parameter_values) <: ℱ.Dual ? Float64.(ℱ.value.(𝓂.parameter_values)) : Float64.(𝓂.parameter_values)
        end
        
        if  ((:pruned_second_order  == algorithm) && pruned_second_order_needs_recalc) ||
            ((:pruned_third_order  == algorithm) && pruned_third_order_needs_recalc)

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_stochastic_steady_state(Val(:pruned_second_order), 𝓂.parameter_values, 𝓂, opts = opts) # , timer = timer)

            if !converged  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

            𝓂.caches.pruned_second_order_stochastic_steady_state = stochastic_steady_state

            𝓂.caches.valid_for.pruned_second_order_solution = eltype(𝓂.parameter_values) <: ℱ.Dual ? Float64.(ℱ.value.(𝓂.parameter_values)) : Float64.(𝓂.parameter_values)
        end
        
        if  ((:third_order  == algorithm) && third_order_needs_recalc)
            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_stochastic_steady_state(Val(:third_order), 𝓂.parameter_values, 𝓂, opts = opts)

            if !converged  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

            𝓂.caches.third_order_stochastic_steady_state = stochastic_steady_state

            𝓂.caches.valid_for.third_order_solution = eltype(𝓂.parameter_values) <: ℱ.Dual ? Float64.(ℱ.value.(𝓂.parameter_values)) : Float64.(𝓂.parameter_values)
        end

        if ((:pruned_third_order  == algorithm) && pruned_third_order_needs_recalc)

            stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_stochastic_steady_state(Val(:pruned_third_order), 𝓂.parameter_values, 𝓂, opts = opts)

            if !converged  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

            𝓂.caches.pruned_third_order_stochastic_steady_state = stochastic_steady_state

            𝓂.caches.valid_for.pruned_third_order_solution = eltype(𝓂.parameter_values) <: ℱ.Dual ? Float64.(ℱ.value.(𝓂.parameter_values)) : Float64.(𝓂.parameter_values)
        end

    end
    
    return nothing
end




function create_second_order_auxiliary_matrices(constants::constants)
    T = constants.post_model_macro
    

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

    so = constants.second_order
    so.𝛔 = 𝛔
    so.𝛔c₂ = 𝐔₂ * 𝛔 * 𝐂₂
    so.𝛔𝐂₂ = 𝛔 * 𝐂₂
    so.𝐂₂ = 𝐂₂
    so.𝐔₂ = 𝐔₂
    so.𝐔∇₂ = 𝐔∇₂
    return so
end



function add_sparse_entries!(P, perm)
    n = size(P, 1)
    for i in 1:n
        P[perm[i], i] += 1.0
    end
end


function create_third_order_auxiliary_matrices(constants::constants, ∇₃_col_indices::Vector{Int})
    T = constants.post_model_macro
    

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

    to = constants.third_order
    to.𝐂₃ = 𝐂₃
    to.𝐔₃ = 𝐔₃
    to.𝐈₃ = 𝐈₃
    to.𝐂∇₃ = 𝐂∇₃
    to.𝐔∇₃ = 𝐔∇₃
    to.𝐏 = 𝐏
    to.𝐏₁ₗ = 𝐏₁ₗ
    to.𝐏₁ᵣ = 𝐏₁ᵣ
    to.𝐏₁ₗ̂ = 𝐏₁ₗ̂
    to.𝐏₂ₗ̂ = 𝐏₂ₗ̂
    to.𝐏₁ₗ̄ = 𝐏₁ₗ̄
    to.𝐏₂ₗ̄ = 𝐏₂ₗ̄
    to.𝐏₁ᵣ̃ = 𝐏₁ᵣ̃
    to.𝐏₂ᵣ̃ = 𝐏₂ᵣ̃
    to.𝐒𝐏 = 𝐒𝐏
    return to
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
    
    nx = BigInt(length(𝔙)::Int)
    # np = length(𝔓)::BigInt
    nϵ = length(dyn_equations)::Int

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
                X_ncols_n = nx^n # Use BigInt for the power calculation, cast to Int

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
                        power_of_nx = nx^(n-1) # Start with nx^(n-1) for v1 term
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
                        # For compressed output, only include entries where variable indices
                        # are in non-increasing order (v_n <= v_{n-1} <= ... <= v_1).
                        # This matches the compression rule used for the X-matrix.
                        # Unsorted tuples represent the same derivative (by symmetry of
                        # mixed partials) but the compressed column formula maps them to
                        # WRONG positions, corrupting the Jacobian.
                        is_compressed_P = true
                        for k_rule = 1:(n-1)
                            if var_indices_full[n-k_rule+1] > var_indices_full[n-k_rule]
                                is_compressed_P = false
                                break
                            end
                        end

                        if !is_compressed_P
                            k_temp_P += 1
                            continue
                        end

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
                        power_of_nx = nx^(n-1)
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


function write_functions_mapping!(𝓂::ℳ, max_perturbation_order::Int; 
                                    density_threshold::Float64 = .1, 
                                    min_length::Int = 1000,
                                    nnz_parallel_threshold::Int = 1000000,
                                    # parallel = Symbolics.SerialForm(),
                                    # parallel = Symbolics.ShardedForm(1500,4),
                                    cse = true,
                                    skipzeros = true)

    future_varss  = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₁₎$")))
    present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎$")))
    past_varss    = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₋₁₎$")))
    shock_varss   = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍ₓ₎$")))
    ss_varss      = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍ₛₛ₎$")))

    sort!(future_varss  ,by = x->replace(string(x),r"₍₁₎$"=>"")) #sort by name without time index because otherwise eps_zᴸ⁽⁻¹⁾₍₋₁₎ comes before eps_z₍₋₁₎
    sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))
    sort!(past_varss    ,by = x->replace(string(x),r"₍₋₁₎$"=>""))
    sort!(shock_varss   ,by = x->replace(string(x),r"₍ₓ₎$"=>""))
    sort!(ss_varss      ,by = x->replace(string(x),r"₍ₛₛ₎$"=>""))

    dyn_future_list = collect(reduce(union, 𝓂.constants.post_model_macro.dyn_future_list))
    dyn_present_list = collect(reduce(union, 𝓂.constants.post_model_macro.dyn_present_list))
    dyn_past_list = collect(reduce(union, 𝓂.constants.post_model_macro.dyn_past_list))
    dyn_exo_list = collect(reduce(union,𝓂.constants.post_model_macro.dyn_exo_list))
    dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.constants.post_model_macro.dyn_ss_list))) .* "₍ₛₛ₎")

    future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
    present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
    past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
    exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
    stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))

    vars_raw = vcat(dyn_future_list[indexin(sort(future),future)],
                    dyn_present_list[indexin(sort(present),present)],
                    dyn_past_list[indexin(sort(past),past)],
                    dyn_exo_list[indexin(sort(exo),exo)])

    dyn_var_future_idx = 𝓂.constants.post_complete_parameters.dyn_var_future_idx
    dyn_var_present_idx = 𝓂.constants.post_complete_parameters.dyn_var_present_idx
    dyn_var_past_idx = 𝓂.constants.post_complete_parameters.dyn_var_past_idx
    dyn_ss_idx = 𝓂.constants.post_complete_parameters.dyn_ss_idx

    dyn_var_idxs = vcat(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx)

    pars_ext = vcat(𝓂.constants.post_complete_parameters.parameters, 𝓂.equations.calibration_parameters)
    parameters_and_SS = vcat(pars_ext, dyn_ss_list[indexin(sort(stst),stst)])

    np = length(parameters_and_SS)
    nv = length(vars_raw)
    nc = length(𝓂.equations.calibration)
    nps = length(𝓂.constants.post_complete_parameters.parameters)
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


    for v in 𝓂.equations.calibration_no_var
        push!(calib_vars, v.args[1])
        push!(calib_expr, v.args[2])
    end


    calib_replacements = Dict{Symbol, Union{Expr, Symbol, Number}}()
    for (i,x) in enumerate(calib_vars)
        replacement = Dict{Symbol, Union{Expr, Symbol, Number}}(x => calib_expr[i])
        for ii in i+1:length(calib_vars)
            calib_expr[ii] = replace_symbols(calib_expr[ii], replacement)
        end
        push!(calib_replacements, x => calib_expr[i])
    end


    dyn_equations = 𝓂.equations.dynamic |> 
        x -> replace_symbols.(x, Ref(calib_replacements)) |> 
        x -> replace_symbols.(x, Ref(parameter_dict)) |> 
        x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
        x -> Symbolics.substitute.(x, Ref(back_to_array_dict))

    derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs)


    ∇₁_dyn = derivatives[1][1]

    lennz = nnz(∇₁_dyn)

    jacobian_dense_by_heuristic = (lennz / length(∇₁_dyn) > density_threshold) || (length(∇₁_dyn) < min_length)
    # NOTE: Keep Jacobian generation and cache buffer dense for allocation/perf profiling consistency.
    # Re-enable `jacobian_dense_by_heuristic` directly to restore sparse Jacobian path switching.
    force_dense_jacobian = true

    if force_dense_jacobian || jacobian_dense_by_heuristic
        derivatives_mat = convert(Matrix, ∇₁_dyn)
        buffer = zeros(Float64, size(∇₁_dyn))
    else
        derivatives_mat = ∇₁_dyn
        buffer = similar(∇₁_dyn, Float64)
        buffer.nzval .= 0
    end
    
    if lennz > nnz_parallel_threshold
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

    𝓂.caches.jacobian = buffer


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

    if lennz > nnz_parallel_threshold
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

    𝓂.caches.jacobian_parameters = buffer_parameters
 

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

    if lennz > nnz_parallel_threshold
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

    𝓂.caches.jacobian_SS_and_pars = buffer_SS_and_pars
    
    # Create jacobian_functions struct with all three functions
    𝓂.functions.jacobian = jacobian_functions(func_exprs, func_∇₁_parameters, func_∇₁_SS_and_pars)




    # if max_perturbation_order >= 1
    #     SS_and_pars = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.constants.post_model_macro.parameters_in_equations,𝓂.constants.post_model_macro.➕_vars))))), 𝓂.calibration_equations_parameters))

    #     eqs = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

    #     nx = length(𝓂.parameter_values)

    #     np = length(SS_and_pars)

    #     nϵˢ = length(eqs)

    #     nc = length(𝓂.calibration_equations_no_var)

    #     Symbolics.@variables 𝔛¹[1:nx] 𝔓¹[1:np]

    #     ϵˢ = zeros(Symbolics.Num, nϵˢ)
    
    #     calib_vals = zeros(Symbolics.Num, nc)

    #     𝓂.SS_calib_func(calib_vals, 𝔛¹)
    
    #     𝓂.functions.NSSS_check(ϵˢ, 𝔛¹, 𝔓¹, calib_vals)
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

    #     if lennz > nnz_parallel_threshold
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

    #     𝓂.caches.∂equations_∂parameters = buffer
    #     𝓂.functions.NSSS_∂equations_∂parameters = func_exprs



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

    #     if lennz > nnz_parallel_threshold
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

    #     𝓂.caches.∂equations_∂SS_and_pars = buffer
    #     𝓂.functions.NSSS_∂equations_∂SS_and_pars = func_exprs
    # end
        
    if max_perturbation_order >= 2
    # second order
        derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs; max_perturbation_order = 2, output_compressed = false)

        if 𝓂.constants.second_order.𝛔 == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            𝓂.constants.second_order = create_second_order_auxiliary_matrices(𝓂.constants)

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

            if lennz > nnz_parallel_threshold
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

            𝓂.caches.hessian = buffer


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

            if lennz > nnz_parallel_threshold
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

            𝓂.caches.hessian_parameters = buffer_parameters
        

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

            if lennz > nnz_parallel_threshold
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

            𝓂.caches.hessian_SS_and_pars = buffer_SS_and_pars
            
            # Create hessian_functions struct with all three functions
            𝓂.functions.hessian = hessian_functions(func_exprs, func_∇₂_parameters, func_∇₂_SS_and_pars)
        end
    end

    if max_perturbation_order == 3
        derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs; max_perturbation_order = max_perturbation_order, output_compressed = true)
    # third order
        if 𝓂.constants.third_order.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            I,J,V = findnz(derivatives[3][1])
            𝓂.constants.third_order = create_third_order_auxiliary_matrices(𝓂.constants, unique(J))
        
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

            if lennz > nnz_parallel_threshold
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

            𝓂.caches.third_order_derivatives = buffer


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

            if lennz > nnz_parallel_threshold
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

            𝓂.caches.third_order_derivatives_parameters = buffer_parameters
        

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

            if lennz > nnz_parallel_threshold
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

            𝓂.caches.third_order_derivatives_SS_and_pars = buffer_SS_and_pars
            
            # Create third_order_derivatives_functions struct with all three functions
            𝓂.functions.third_order_derivatives = third_order_derivatives_functions(func_exprs, func_∇₃_parameters, func_∇₃_SS_and_pars)
        end
    end

    return nothing
end


function write_auxiliary_indices!(𝓂::ℳ)
    # write indices in auxiliary objects
    dyn_var_future_list  = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₁₎")))
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎")))
    dyn_var_past_list    = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₋₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₋₁₎")))
    dyn_exo_list         = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍ₓ₎")))
    dyn_ss_list          = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₛₛ₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍ₛₛ₎")))

    dyn_var_future  = Symbol.(string.(sort(collect(reduce(union,dyn_var_future_list)))))
    dyn_var_present = Symbol.(string.(sort(collect(reduce(union,dyn_var_present_list)))))
    dyn_var_past    = Symbol.(string.(sort(collect(reduce(union,dyn_var_past_list)))))
    dyn_exo         = Symbol.(string.(sort(collect(reduce(union,dyn_exo_list)))))
    dyn_ss          = Symbol.(string.(sort(collect(reduce(union,dyn_ss_list)))))

    SS_and_pars_names = vcat(Symbol.(string.(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future)))), 𝓂.equations.calibration_parameters)

    dyn_var_future_idx  = indexin(dyn_var_future    , SS_and_pars_names)
    dyn_var_present_idx = indexin(dyn_var_present   , SS_and_pars_names)
    dyn_var_past_idx    = indexin(dyn_var_past      , SS_and_pars_names)
    dyn_ss_idx          = indexin(dyn_ss            , SS_and_pars_names)

    shocks_ss = zeros(length(dyn_exo))

    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        dyn_var_future_idx = dyn_var_future_idx,
        dyn_var_present_idx = dyn_var_present_idx,
        dyn_var_past_idx = dyn_var_past_idx,
        dyn_ss_idx = dyn_ss_idx,
        shocks_ss = shocks_ss,
    )

    return nothing
end

write_parameters_input!(𝓂::ℳ, parameters::Nothing; verbose::Bool = true) = return parameters
write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,Float64}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Pair{S,Float64}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}(parameters[1] |> Meta.parse |> replace_indices => parameters[2]), verbose = verbose)



write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Float64},Vararg{Pair{Symbol,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict(parameters), verbose = verbose)
# write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Union{Symbol,AbstractString},Union{Float64,Int}},Vararg{Pair{Union{Symbol,AbstractString},Union{Float64,Int}}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
# write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Int},Vararg{Pair{AbstractString,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, Dict(parameters), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{S,Float64},Vararg{Pair{S,Float64}}}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters])
, verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{Symbol, Float64}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol, Float64}([replace_indices(string(i[1])) => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{S, Float64}}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol, Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Dict{S,Float64}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}((keys(parameters) .|> Meta.parse .|> replace_indices) .=> values(parameters)), verbose = verbose)


write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,Int}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([replace_indices(string(parameters[1])) => parameters[2]]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Pair{S,Int}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}((parameters[1] |> Meta.parse |> replace_indices) => parameters[2]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Int},Vararg{Pair{Symbol,Int}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([replace_indices(string(i[1])) => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{S,Int},Vararg{Pair{S,Int}}}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, Dict{Symbol,Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{Symbol, Int}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([replace_indices(string(i[1])) => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{S, Int}}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Dict{S,Int}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}((keys(parameters) .|> Meta.parse .|> replace_indices) .=> values(parameters)), verbose = verbose)


write_parameters_input!(𝓂::ℳ, parameters::Pair{Symbol,Real}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([replace_indices(string(parameters[1])) => parameters[2]]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Pair{S,Real}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}((parameters[1] |> Meta.parse |> replace_indices) => parameters[2]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Dict{S,Real}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}((keys(parameters) .|> Meta.parse .|> replace_indices) .=> values(parameters)), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{Symbol,Real},Vararg{Pair{Symbol,Float64}}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([replace_indices(string(i[1])) => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Tuple{Pair{S,Real},Vararg{Pair{S,Float64}}}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{Symbol, Real}}; verbose::Bool = true) = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([replace_indices(string(i[1])) => i[2] for i in parameters]), verbose = verbose)
write_parameters_input!(𝓂::ℳ, parameters::Vector{Pair{S, Real}}; verbose::Bool = true) where S <: AbstractString = write_parameters_input!(𝓂::ℳ, OrderedDict{Symbol,Float64}([i[1] |> Meta.parse |> replace_indices => i[2] for i in parameters]), verbose = verbose)



function write_parameters_input!(𝓂::ℳ, parameters::D; verbose::Bool = true) where D <: AbstractDict{Symbol,Float64}
    # Handle missing parameters - add them if they are in the missing_parameters list
    p = 𝓂.constants.post_complete_parameters
    missing_params_provided = intersect(collect(keys(parameters)), p.missing_parameters)
    
    if !isempty(missing_params_provided)
        
        # Remove the provided missing params from the missing list
        remaining_missing = setdiff(p.missing_parameters, missing_params_provided)
        
        # If all missing parameters are now provided, print a message
        if !isempty(remaining_missing)
            @info "Remaining missing parameters: ", remaining_missing
        end

        # Amend parameter order by provided missing params
        # declared_params = parameters that were never missing (have non-NaN values)
        # We identify them as parameters that are not in the union of missing_params_provided and still-missing params
        all_missing = union(missing_params_provided, remaining_missing)
        declared_params = setdiff(p.parameters, all_missing)
        
        # Get the current parameter values for declared params
        declared_param_indices = indexin(declared_params, p.parameters)
        declared_values = 𝓂.parameter_values[declared_param_indices]
        
        # Get values for the newly provided missing params (currently NaN in parameter_values)
        # We'll set them later after the bounds check
        missing_values = fill(NaN, length(missing_params_provided))
        
        # Get values for the remaining missing params (still NaN)
        remaining_missing_values = fill(NaN, length(remaining_missing))
        
        # Reorder both parameters and parameter_values arrays
        new_parameters = vcat(declared_params, collect(missing_params_provided), remaining_missing)
        𝓂.constants.post_complete_parameters = update_post_complete_parameters(
            p;
            parameters = new_parameters,
            missing_parameters = remaining_missing,
        )
        𝓂.parameter_values = vcat(declared_values, missing_values, remaining_missing_values)
        
        # Clear NSSS solver cache because parameter order/count changed.
        # It will be rebuilt during the next NSSS setup.
        while length(𝓂.caches.solver_cache) > 0
            pop!(𝓂.caches.solver_cache)
        end
    end
    
    # Handle remaining parameters (not missing ones)
    p = 𝓂.constants.post_complete_parameters
    if length(setdiff(collect(keys(parameters)), p.parameters))>0
        @warn("Parameters not part of the model are ignored: $(setdiff(collect(keys(parameters)),p.parameters))")
        for kk in setdiff(collect(keys(parameters)), p.parameters)
            delete!(parameters,kk)
        end
    end

    bounds_broken = false

    for (par,val) in parameters
        if haskey(𝓂.constants.post_parameters_macro.bounds,par)
            if val > 𝓂.constants.post_parameters_macro.bounds[par][2]
                @warn("Calibration is out of bounds for $par < $(𝓂.constants.post_parameters_macro.bounds[par][2])\t parameter value: $val")
                bounds_broken = true
                continue
            end
            if val < 𝓂.constants.post_parameters_macro.bounds[par][1]
                @warn("Calibration is out of bounds for $par > $(𝓂.constants.post_parameters_macro.bounds[par][1])\t parameter value: $val")
                bounds_broken = true
                continue
            end
        end
    end

    if bounds_broken
        @warn("Parameters unchanged.")
    else
        ntrsct_idx = map(x-> getindex(1:length(𝓂.parameter_values), p.parameters .== x)[1], collect(keys(parameters)))
        # ntrsct_idx = indexin(collect(keys(parameters)), p.parameters)
        
        if !all(𝓂.parameter_values[ntrsct_idx] .== collect(values(parameters))) && !(p.parameters[ntrsct_idx] == [:activeᵒᵇᶜshocks])
            if verbose println("Parameter changes: ") end
        end
            
        for i in 1:length(parameters)
            if 𝓂.parameter_values[ntrsct_idx[i]] != collect(values(parameters))[i]
                if verbose println("\t",p.parameters[ntrsct_idx[i]],"\tfrom ",𝓂.parameter_values[ntrsct_idx[i]],"\tto ",collect(values(parameters))[i]) end

                𝓂.parameter_values[ntrsct_idx[i]] = collect(values(parameters))[i]
            end
        end
    end

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
        @warn "Model has $(length(𝓂.parameter_values)) parameters. $(length(parameters)) were provided. The following will be ignored: $(join(parameters[length(𝓂.parameter_values)+1:end], " "))"

        parameters = parameters[1:length(𝓂.parameter_values)]
    end

    bounds_broken = false
    parameters_dict = Dict(𝓂.constants.post_complete_parameters.parameters .=> parameters)

    for (par, val) in parameters_dict
        if haskey(𝓂.constants.post_parameters_macro.bounds,par)
            if val > 𝓂.constants.post_parameters_macro.bounds[par][2]
                @warn("Calibration is out of bounds for $par < $(𝓂.constants.post_parameters_macro.bounds[par][2])\t parameter value: $val")
                bounds_broken = true
                continue
            end
            if val < 𝓂.constants.post_parameters_macro.bounds[par][1]
                @warn("Calibration is out of bounds for $par > $(𝓂.constants.post_parameters_macro.bounds[par][1])\t parameter value: $val")
                bounds_broken = true
                continue
            end
        end
    end

    if bounds_broken
        @warn("Parameters unchanged.")
    else
        if !all(parameters .== 𝓂.parameter_values[1:length(parameters)])
            match_idx = []
            for (i, v) in enumerate(parameters)
                if v != 𝓂.parameter_values[i]
                    push!(match_idx,i)
                end
            end
            
            changed_vals = parameters[match_idx]
            changed_pars = 𝓂.constants.post_complete_parameters.parameters[match_idx]

            if verbose 
                println("Parameter changes: ")
                for (i,m) in enumerate(match_idx)
                    println("\t",changed_pars[i],"\tfrom ",𝓂.parameter_values[m],"\tto ",changed_vals[i])
                end
            end

            𝓂.parameter_values[match_idx] = parameters[match_idx]
        end
    end

    if 𝓂.caches.valid_for.non_stochastic_steady_state != 𝓂.parameter_values && verbose
        println("New parameters changed the steady state.")
    end

    return nothing
end


# function create_timings_for_estimation!(𝓂::ℳ, observables::Vector{Symbol})
#     dyn_equations = 𝓂.dyn_equations

#     vars_to_exclude = setdiff(𝓂.constants.post_model_macro.present_only, observables)

#     # Mapping variables to their equation index
#     variable_to_equation = Dict{Symbol, Vector{Int}}()
#     for var in vars_to_exclude
#         for (eq_idx, vars_set) in enumerate(𝓂.dyn_var_present_list)
#         # for var in vars_set
#             if var in vars_set
#                 if haskey(variable_to_equation, var)
#                     push!(variable_to_equation[var],eq_idx)
#                 else
#                     variable_to_equation[var] = [eq_idx]
#                 end
#             end
#         end
#     end

#     # cols_to_exclude = indexin(𝓂.constants.post_model_macro.var, setdiff(𝓂.constants.post_model_macro.present_only, observables))
#     cols_to_exclude = indexin(setdiff(𝓂.constants.post_model_macro.present_only, observables), 𝓂.constants.post_model_macro.var)

#     present_idx = 𝓂.constants.post_model_macro.nFuture_not_past_and_mixed .+ (setdiff(range(1, 𝓂.constants.post_model_macro.nVars), cols_to_exclude))

#     dyn_var_future_list  = deepcopy(𝓂.dyn_var_future_list)
#     dyn_var_present_list = deepcopy(𝓂.dyn_var_present_list)
#     dyn_var_past_list    = deepcopy(𝓂.dyn_var_past_list)
#     dyn_exo_list         = deepcopy(𝓂.dyn_exo_list)
#     dyn_ss_list          = deepcopy(𝓂.dyn_ss_list)

#     rows_to_exclude = Int[]

#     for vidx in values(variable_to_equation)
#         for v in vidx
#             if v ∉ rows_to_exclude
#                 push!(rows_to_exclude, v)

#                 for vv in vidx
#                     dyn_var_future_list[vv] = union(dyn_var_future_list[vv], dyn_var_future_list[v])
#                     dyn_var_present_list[vv] = union(dyn_var_present_list[vv], dyn_var_present_list[v])
#                     dyn_var_past_list[vv] = union(dyn_var_past_list[vv], dyn_var_past_list[v])
#                     dyn_exo_list[vv] = union(dyn_exo_list[vv], dyn_exo_list[v])
#                     dyn_ss_list[vv] = union(dyn_ss_list[vv], dyn_ss_list[v])
#                 end

#                 break
#             end
#         end
#     end

#     rows_to_include = setdiff(1:𝓂.constants.post_model_macro.nVars, rows_to_exclude)

#     all_symbols = setdiff(reduce(union,collect.(get_symbols.(dyn_equations)))[rows_to_include], vars_to_exclude)
#     parameters_in_equations = sort(setdiff(all_symbols, match_pattern(all_symbols,r"₎$")))
    
#     dyn_var_future  =  sort(setdiff(collect(reduce(union,dyn_var_future_list[rows_to_include])), vars_to_exclude))
#     dyn_var_present =  sort(setdiff(collect(reduce(union,dyn_var_present_list[rows_to_include])), vars_to_exclude))
#     dyn_var_past    =  sort(setdiff(collect(reduce(union,dyn_var_past_list[rows_to_include])), vars_to_exclude))
#     dyn_var_ss      =  sort(setdiff(collect(reduce(union,dyn_ss_list[rows_to_include])), vars_to_exclude))

#     all_dyn_vars        = union(dyn_var_future, dyn_var_present, dyn_var_past)

#     @assert length(setdiff(dyn_var_ss, all_dyn_vars)) == 0 "The following variables are (and cannot be) defined only in steady state (`[ss]`): $(setdiff(dyn_var_ss, all_dyn_vars))"

#     all_vars = union(all_dyn_vars, dyn_var_ss)

#     present_only              = sort(setdiff(dyn_var_present,union(dyn_var_past,dyn_var_future)))
#     future_not_past           = sort(setdiff(dyn_var_future, dyn_var_past))
#     past_not_future           = sort(setdiff(dyn_var_past, dyn_var_future))
#     mixed                     = sort(setdiff(dyn_var_present, union(present_only, future_not_past, past_not_future)))
#     future_not_past_and_mixed = sort(union(future_not_past,mixed))
#     past_not_future_and_mixed = sort(union(past_not_future,mixed))
#     present_but_not_only      = sort(setdiff(dyn_var_present,present_only))
#     mixed_in_past             = sort(intersect(dyn_var_past, mixed))
#     not_mixed_in_past         = sort(setdiff(dyn_var_past,mixed_in_past))
#     mixed_in_future           = sort(intersect(dyn_var_future, mixed))
#     exo                       = sort(collect(reduce(union,dyn_exo_list)))
#     var                       = sort(dyn_var_present)
#     aux_tmp                   = sort(filter(x->occursin(r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾",string(x)), dyn_var_present))
#     aux                       = aux_tmp[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∉ exo, aux_tmp)]
#     exo_future                = dyn_var_future[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_future)]
#     exo_present               = dyn_var_present[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_present)]
#     exo_past                  = dyn_var_past[map(x->Symbol(replace(string(x),r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")) ∈ exo, dyn_var_past)]

#     nPresent_only              = length(present_only)
#     nMixed                     = length(mixed)
#     nFuture_not_past_and_mixed = length(future_not_past_and_mixed)
#     nPast_not_future_and_mixed = length(past_not_future_and_mixed)
#     nPresent_but_not_only      = length(present_but_not_only)
#     nVars                      = length(all_vars)
#     nExo                       = length(collect(exo))

#     present_only_idx              = indexin(present_only,var)
#     present_but_not_only_idx      = indexin(present_but_not_only,var)
#     future_not_past_and_mixed_idx = indexin(future_not_past_and_mixed,var)
#     past_not_future_and_mixed_idx = indexin(past_not_future_and_mixed,var)
#     mixed_in_future_idx           = indexin(mixed_in_future,dyn_var_future)
#     mixed_in_past_idx             = indexin(mixed_in_past,dyn_var_past)
#     not_mixed_in_past_idx         = indexin(not_mixed_in_past,dyn_var_past)
#     past_not_future_idx           = indexin(past_not_future,var)

#     reorder       = indexin(var, [present_only; past_not_future; future_not_past_and_mixed])
#     dynamic_order = indexin(present_but_not_only, [past_not_future; future_not_past_and_mixed])

#     @assert length(intersect(union(var,exo),parameters_in_equations)) == 0 "Parameters and variables cannot have the same name. This is the case for: " * repr(sort([intersect(union(var,exo),parameters_in_equations)...]))

#     T = timings(present_only,
#                 future_not_past,
#                 past_not_future,
#                 mixed,
#                 future_not_past_and_mixed,
#                 past_not_future_and_mixed,
#                 present_but_not_only,
#                 mixed_in_past,
#                 not_mixed_in_past,
#                 mixed_in_future,
#                 exo,
#                 var,
#                 aux,
#                 exo_present,

#                 nPresent_only,
#                 nMixed,
#                 nFuture_not_past_and_mixed,
#                 nPast_not_future_and_mixed,
#                 nPresent_but_not_only,
#                 nVars,
#                 nExo,

#                 present_only_idx,
#                 present_but_not_only_idx,
#                 future_not_past_and_mixed_idx,
#                 not_mixed_in_past_idx,
#                 past_not_future_and_mixed_idx,
#                 mixed_in_past_idx,
#                 mixed_in_future_idx,
#                 past_not_future_idx,

#                 reorder,
#                 dynamic_order)

#     push!(𝓂.estimation_helper, observables => T)

#     return nothing
# end



function calculate_jacobian(parameters::Vector{M},
                            SS_and_pars::Vector{N},
                            caches_obj::caches,
                            jacobian_funcs::jacobian_functions)::Matrix{M} where {M,N}
    if eltype(caches_obj.jacobian) != M
        if caches_obj.jacobian isa SparseMatrixCSC
            jac_buffer = similar(caches_obj.jacobian,M)
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(M, size(caches_obj.jacobian))
        end
    else
        jac_buffer = caches_obj.jacobian
    end
    
    jacobian_funcs.f(jac_buffer, parameters, SS_and_pars)

    if M === Float64
        caches_obj.jacobian = jac_buffer
    end
    
    return jac_buffer
end

function calculate_hessian(parameters::Vector{M}, 
                            SS_and_pars::Vector{N}, 
                            caches_obj::caches,
                            hessian_funcs::hessian_functions)::SparseMatrixCSC{M, Int} where {M,N}
    if eltype(caches_obj.hessian) != M
        if caches_obj.hessian isa SparseMatrixCSC
            hes_buffer = similar(caches_obj.hessian,M)
            hes_buffer.nzval .= 0
        else
            hes_buffer = zeros(M, size(caches_obj.hessian))
        end
    else
        hes_buffer = caches_obj.hessian
    end

    hessian_funcs.f(hes_buffer, parameters, SS_and_pars)

    if M === Float64
        caches_obj.hessian = hes_buffer
    end
    
    return hes_buffer
end


function calculate_third_order_derivatives(parameters::Vector{M}, 
                                            SS_and_pars::Vector{N}, 
                                            caches_obj::caches,
                                            third_order_derivatives_funcs::third_order_derivatives_functions)::SparseMatrixCSC{M, Int} where {M,N}
    if eltype(caches_obj.third_order_derivatives) != M
        if caches_obj.third_order_derivatives isa SparseMatrixCSC
            third_buffer = similar(caches_obj.third_order_derivatives,M)
            third_buffer.nzval .= 0
        else
            third_buffer = zeros(M, size(caches_obj.third_order_derivatives))
        end
    else
        third_buffer = caches_obj.third_order_derivatives
    end

    third_order_derivatives_funcs.f(third_buffer, parameters, SS_and_pars)

    if M === Float64
        caches_obj.third_order_derivatives = third_buffer
    end
    
    return third_buffer
end



function compute_irf_responses(𝓂::ℳ,
                                state_update::Function,
                                initial_state::Union{Vector{Vector{Float64}},Vector{Float64}},
                                level::Vector{Float64};
                                periods::Int,
                                shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}},
                                variables::Union{Symbol_input,String_input},
                                shock_size::Real,
                                negative_shock::Bool,
                                generalised_irf::Bool,
                                generalised_irf_warmup_iterations::Int,
                                generalised_irf_draws::Int,
                                enforce_obc::Bool,
                                algorithm::Symbol)

    if enforce_obc
        function obc_state_update(present_states, present_shocks::Vector{R}, state_update::Function) where R <: Float64
            unconditional_forecast_horizon = 𝓂.constants.post_model_macro.max_obc_horizon

            reference_ss = 𝓂.caches.non_stochastic_steady_state

            obc_shock_idx = contains.(string.(𝓂.constants.post_model_macro.exo),"ᵒᵇᶜ")

            periods_per_shock = 𝓂.constants.post_model_macro.max_obc_horizon + 1

            num_shocks = sum(obc_shock_idx) ÷ periods_per_shock

            p = (present_states, state_update, reference_ss, 𝓂, algorithm, unconditional_forecast_horizon, present_shocks)

            constraints_violated = any(𝓂.functions.obc_violation(zeros(num_shocks*periods_per_shock), p) .> eps(Float32))

            if constraints_violated
                opt = NLopt.Opt(NLopt.:LD_SLSQP, num_shocks*periods_per_shock)

                opt.min_objective = obc_objective_optim_fun

                opt.xtol_abs = eps(Float32)
                opt.ftol_abs = eps(Float32)
                opt.maxeval = 500

                upper_bounds = fill(eps(), 1 + 2*(max(num_shocks*periods_per_shock-1, 1)))

                NLopt.inequality_constraint!(opt, (res, x, jac) -> obc_constraint_optim_fun(res, x, jac, p), upper_bounds)

                (minf,x,ret) = NLopt.optimize(opt, zeros(num_shocks*periods_per_shock))

                present_shocks[contains.(string.(𝓂.constants.post_model_macro.exo),"ᵒᵇᶜ")] .= x

                constraints_violated = any(𝓂.functions.obc_violation(x, p) .> eps(Float32))

                solved = !constraints_violated
            else
                solved = true
            end

            present_states = state_update(present_states, present_shocks)

            return present_states, present_shocks, solved
        end

        if generalised_irf
            return girf(state_update,
                        obc_state_update,
                        initial_state,
                        level,
                        𝓂.constants;
                        periods = periods,
                        shocks = shocks,
                        shock_size = shock_size,
                        variables = variables,
                        negative_shock = negative_shock,
                        warmup_periods = generalised_irf_warmup_iterations,
                        draws = generalised_irf_draws)
        else
            return irf(state_update,
                        obc_state_update,
                        initial_state,
                        level,
                        𝓂.constants;
                        periods = periods,
                        shocks = shocks,
                        shock_size = shock_size,
                        variables = variables,
                        negative_shock = negative_shock)
        end
    else
        if generalised_irf
            return girf(state_update,
                        initial_state,
                        level,
                        𝓂.constants;
                        periods = periods,
                        shocks = shocks,
                        shock_size = shock_size,
                        variables = variables,
                        negative_shock = negative_shock,
                        warmup_periods = generalised_irf_warmup_iterations,
                        draws = generalised_irf_draws)
        else
            return irf(state_update,
                        initial_state,
                        level,
                        𝓂.constants;
                        periods = periods,
                        shocks = shocks,
                        shock_size = shock_size,
                        variables = variables,
                        negative_shock = negative_shock)
        end
    end
end


function irf(state_update::Function, 
    obc_state_update::Function,
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64},
    constants::constants; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}
    T = constants.post_model_macro

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

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."
        
        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,constants)
    end

    var_idx = parse_variables_input_to_index(variables, constants) |> sort

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
    constants::constants; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}
    T = constants.post_model_macro

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

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."
        
        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,constants)
    end

    var_idx = parse_variables_input_to_index(variables, constants) |> sort

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
    constants::constants; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false, 
    warmup_periods::Int = 100, 
    draws::Int = 50)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}
    T = constants.post_model_macro

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model (model has $(T.nExo) shocks)."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."

        shock_history = zeros(T.nExo, periods + 1)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks == :simulate
        shock_history = randn(T.nExo,periods) * shock_size

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,constants)
    end

    var_idx = parse_variables_input_to_index(variables, constants) |> sort

    Y = zeros(T.nVars, periods + 1, length(shock_idx))

    for (i,ii) in enumerate(shock_idx)
        initial_state_copy = deepcopy(initial_state)

        accepted_draws = 0

        for draw in 1:draws
            ok = true

            initial_state_copy² = deepcopy(initial_state_copy)

            for i in 1:warmup_periods
                initial_state_copy² = state_update(initial_state_copy², randn(T.nExo))
                if any(!isfinite, [x for v in initial_state_copy² for x in v])
                    # @warn "No solution in warmup period: $i"
                    ok = false
                    break
                end
            end
            
            if !ok continue end

            Y₁ = zeros(T.nVars, periods + 1)
            Y₂ = zeros(T.nVars, periods + 1)

            baseline_noise = randn(T.nExo)

            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -shock_size : shock_size
            end

            if pruning
                initial_state_copy² = state_update(initial_state_copy², baseline_noise)
                
                if any(!isfinite, [x for v in initial_state_copy² for x in v]) continue end

                initial_state₁ = deepcopy(initial_state_copy²)
                initial_state₂ = deepcopy(initial_state_copy²)

                Y₁[:,1] = initial_state_copy² |> sum
                Y₂[:,1] = initial_state_copy² |> sum
            else
                Y₁[:,1] = state_update(initial_state_copy², baseline_noise)
                
                if any(!isfinite, Y₁[:,1]) continue end

                Y₂[:,1] = state_update(initial_state_copy², baseline_noise)
                
                if any(!isfinite, Y₂[:,1]) continue end
            end

            for t in 1:periods
                baseline_noise = randn(T.nExo)

                if pruning
                    initial_state₁ = state_update(initial_state₁, baseline_noise)
                
                    if any(!isfinite, [x for v in initial_state₁ for x in v])
                        ok = false
                        break
                    end

                    initial_state₂ = state_update(initial_state₂, baseline_noise + shock_history[:,t])
                
                    if any(!isfinite, [x for v in initial_state₂ for x in v])
                        ok = false
                        break
                    end

                    Y₁[:,t+1] = initial_state₁ |> sum
                    Y₂[:,t+1] = initial_state₂ |> sum
                else
                    Y₁[:,t+1] = state_update(Y₁[:,t],baseline_noise)

                    if any(!isfinite, Y₁[:,t+1])
                        ok = false
                        break
                    end

                    Y₂[:,t+1] = state_update(Y₂[:,t],baseline_noise + shock_history[:,t])

                    if any(!isfinite, Y₂[:,t+1])
                        ok = false
                        break
                    end
                end
            end

            if !ok continue end

            Y[:,:,i] += Y₂ - Y₁

            accepted_draws += 1
        end
        
        if accepted_draws == 0
            @warn "No draws accepted. Results are empty."
        elseif accepted_draws < draws
            # average over accepted draws, if desired
            @info "$accepted_draws of $draws draws accepted for shock: $(shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input, String_input} ? T.exo[ii] : :Shock_matrix)"
            Y[:, :, i] ./= accepted_draws
        else
            Y[:, :, i] ./= accepted_draws
        end
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


function girf(state_update::Function,
    obc_state_update::Function,
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64}, 
    constants::constants; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false, 
    warmup_periods::Int = 100, 
    draws::Int = 50)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}
    T = constants.post_model_macro

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

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."

        shock_history = zeros(T.nExo, periods + 1)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks == :simulate
        shock_history = randn(T.nExo,periods) * shock_size
        
        shock_history[contains.(string.(T.exo),"ᵒᵇᶜ"),:] .= 0

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,constants)
    end

    var_idx = parse_variables_input_to_index(variables, constants) |> sort

    Y = zeros(T.nVars, periods + 1, length(shock_idx))

    for (i,ii) in enumerate(shock_idx)
        initial_state_copy = deepcopy(initial_state)

        accepted_draws = 0

        for draw in 1:draws
            ok = true

            initial_state_copy² = deepcopy(initial_state_copy)

            warmup_shocks = randn(T.nExo)
            warmup_shocks[contains.(string.(T.exo), "ᵒᵇᶜ")] .= 0

            # --- warmup ---
            for i_w in 1:warmup_periods
                initial_state_copy², _, solved = obc_state_update(initial_state_copy², warmup_shocks, state_update)
                if !solved
                    # @warn "No solution in warmup period: $i_w"
                    ok = false
                    break
                end
            end
            
            if !ok continue end

            Y₁ = zeros(T.nVars, periods + 1)
            Y₂ = zeros(T.nVars, periods + 1)

            baseline_noise = randn(T.nExo)
            baseline_noise[contains.(string.(T.exo), "ᵒᵇᶜ")] .= 0

            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input, String_input}
                shock_history = zeros(T.nExo, periods)
                shock_history[ii, 1] = negative_shock ? -shock_size : shock_size
            end

            # --- period 1 ---
            if pruning
                initial_state_copy², _, solved = obc_state_update(initial_state_copy², baseline_noise, state_update)
                if !solved continue end

                initial_state₁ = deepcopy(initial_state_copy²)
                initial_state₂ = deepcopy(initial_state_copy²)

                Y₁[:, 1] = initial_state_copy² |> sum
                Y₂[:, 1] = initial_state_copy² |> sum
            else
                Y₁[:, 1], _, solved = obc_state_update(initial_state_copy², baseline_noise, state_update)
                if !solved continue end

                Y₂[:, 1], _, solved = obc_state_update(initial_state_copy², baseline_noise, state_update)
                if !solved continue end
            end

            # --- remaining periods ---
            for t in 1:periods
                baseline_noise = randn(T.nExo)
                baseline_noise[contains.(string.(T.exo), "ᵒᵇᶜ")] .= 0

                if pruning
                    initial_state₁, _, solved = obc_state_update(initial_state₁, baseline_noise, state_update)
                    if !solved
                        # @warn "No solution in period: $t"
                        ok = false
                        break
                    end

                    initial_state₂, _, solved = obc_state_update(initial_state₂, baseline_noise + shock_history[:, t], state_update)
                    if !solved
                        # @warn "No solution in period: $t"
                        ok = false
                        break
                    end

                    Y₁[:, t + 1] = initial_state₁ |> sum
                    Y₂[:, t + 1] = initial_state₂ |> sum
                else
                    Y₁[:, t + 1], _, solved = obc_state_update(Y₁[:, t], baseline_noise, state_update)
                    if !solved
                        # @warn "No solution in period: $t"
                        ok = false
                        break
                    end

                    Y₂[:, t + 1], _, solved = obc_state_update(Y₂[:, t], baseline_noise + shock_history[:, t], state_update)
                    if !solved
                        # @warn "No solution in period: $t"
                        ok = false
                        break
                    end
                end
            end

            if !ok continue end

            # Note: replace `i` if your outer scope uses another index
            Y[:, :, i] .+= (Y₂ .- Y₁)
            accepted_draws += 1
        end

        if accepted_draws == 0
            @warn "No draws accepted. Results are empty."
        elseif accepted_draws < draws
            # average over accepted draws, if desired
            @info "$accepted_draws of $draws draws accepted for shock: $(shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input, String_input} ? T.exo[ii] : :Shock_matrix)"
            Y[:, :, i] ./= accepted_draws
        else
            Y[:, :, i] ./= accepted_draws
        end
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


function parse_variables_input_to_index(variables::Union{Symbol_input, String_input, Vector{Vector{Symbol}}, Vector{Tuple{Symbol,Vararg{Symbol}}}, Vector{Vector{Symbol}}, Tuple{Tuple{Symbol,Vararg{Symbol}}, Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}}, 𝓂::ℳ)::Union{UnitRange{Int}, Vector{Int}}
    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    if variables == :all_excluding_auxiliary_and_obc
        return ms.vars_idx_excluding_aux_obc
    elseif variables == :all_excluding_obc
        return ms.vars_idx_excluding_obc
    end

    return parse_variables_input_to_index(variables, 𝓂.constants)
end

function parse_variables_input_to_index(variables::Union{Symbol_input, String_input, Vector{Vector{Symbol}}, Vector{Tuple{Symbol,Vararg{Symbol}}}, Vector{Vector{Symbol}}, Tuple{Tuple{Symbol,Vararg{Symbol}}, Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}}, constants::constants)::Union{UnitRange{Int}, Vector{Int}}
    T = constants.post_model_macro
    

    # Handle nested vector conversion separately
    if variables isa Vector{Vector{String}}
        variables = [group .|> Meta.parse .|> replace_indices for group in variables]
    elseif variables isa String_input
        variables = variables .|> Meta.parse .|> replace_indices
    end

    if variables == :all_excluding_auxiliary_and_obc
        return Int.(indexin(setdiff(T.var[.!contains.(string.(T.var),"ᵒᵇᶜ")],union(T.aux, T.exo_present)),sort(union(T.var,T.aux,T.exo_present))))
        # return indexin(setdiff(setdiff(T.var,T.exo_present),T.aux),sort(union(T.var,T.aux,T.exo_present)))
    elseif variables == :all_excluding_obc
        return Int.(indexin(T.var[.!contains.(string.(T.var),"ᵒᵇᶜ")],sort(union(T.var,T.aux,T.exo_present))))
    elseif variables == :all
        return 1:length(union(T.var,T.aux,T.exo_present))
    elseif variables isa Matrix{Symbol}
        if length(setdiff(variables,T.var)) > 0
            @warn "The following variables are not part of the model: " * join(string.(setdiff(variables,T.var)),", ") * ". Use `get_variables(𝓂)` to list valid names."
            return Int[]
        end
        return getindex(1:length(T.var),convert(Vector{Bool},vec(sum(variables .== T.var,dims= 2))))
    elseif variables isa Vector{Vector{Symbol}}
        # For grouped inputs, return union of all variables
        all_vars = reduce(vcat, variables)
        if length(setdiff(all_vars,T.var)) > 0
            @warn "The following variables are not part of the model: " * join(string.(setdiff(all_vars,T.var)),", ") * ". Use `get_variables(𝓂)` to list valid names."
            return Int[]
        end
        return Int.(indexin(unique(all_vars), T.var))
    elseif variables isa Vector{Tuple{Symbol,Vararg{Symbol}}}
        # For grouped inputs with tuples, return union of all variables
        all_vars = reduce(vcat, [collect(group) for group in variables])
        if length(setdiff(all_vars,T.var)) > 0
            @warn "The following variables are not part of the model: " * join(string.(setdiff(all_vars,T.var)),", ") * ". Use `get_variables(𝓂)` to list valid names."
            return Int[]
        end
        return Int.(indexin(unique(all_vars), T.var))
    elseif variables isa Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}}
        # For grouped inputs with tuple of tuples, return union of all variables
        all_vars = reduce(vcat, [collect(group) for group in variables])
        if length(setdiff(all_vars,T.var)) > 0
            @warn "The following variables are not part of the model: " * join(string.(setdiff(all_vars,T.var)),", ") * ". Use `get_variables(𝓂)` to list valid names."
            return Int[]
        end
        return Int.(indexin(unique(all_vars), T.var))
    elseif variables isa Vector{Symbol}
        if length(setdiff(variables,T.var)) > 0
            @warn "The following variables are not part of the model: " * join(string.(setdiff(variables,T.var)),", ") * ". Use `get_variables(𝓂)` to list valid names."
            return Int[]
        end
        return Int.(indexin(variables, T.var))
    elseif variables isa Tuple{Symbol,Vararg{Symbol}}
        if length(setdiff(variables,T.var)) > 0
            @warn "The following variables are not part of the model: " * join(string.(setdiff(Symbol.(collect(variables)),T.var)),", ") * ". Use `get_variables(𝓂)` to list valid names."
            return Int[]
        end
        return Int.(indexin(variables, T.var))
    elseif variables isa Symbol
        if length(setdiff([variables],T.var)) > 0
            @warn "The following variable is not part of the model: $(setdiff([variables],T.var)[1]). Use `get_variables(𝓂)` to list valid names."
            return Int[]
        end
        return Int.(indexin([variables], T.var))
    else
        @warn "Invalid `variables` argument. Provide a Symbol, Tuple, Vector, Matrix, or one of the documented selectors such as `:all`."
        return Int[]
    end
end



# Helper function to check if input is grouped covariance format
function is_grouped_covariance_input(variables::Union{Symbol_input,String_input, Vector{Vector{Symbol}},Vector{Tuple{Symbol,Vararg{Symbol}}},Vector{Vector{Symbol}},Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}})::Bool
    # Check if it's a nested structure (vector of vectors, vector of tuples, or tuple of tuples)
    return variables isa Vector{Vector{Symbol}} || variables isa Vector{Vector{String}} ||
           variables isa Vector{Tuple{Symbol,Vararg{Symbol}}} || variables isa Vector{Tuple{String,Vararg{String}}} ||
           variables isa Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}} || 
           variables isa Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}
end

# Function to parse grouped covariance input into groups of indices
function parse_covariance_groups(variables::Union{Symbol_input,String_input, Vector{Vector{Symbol}},Vector{Tuple{Symbol,Vararg{Symbol}}},Vector{Vector{Symbol}},Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}}, constants::constants)::Vector{Vector{Int}}
    T = constants.post_model_macro
    

    # Convert String_input to Symbol_input for nested structures
    if variables isa Vector{Vector{String}}
        variables = [group .|> Meta.parse .|> replace_indices for group in variables]
    elseif variables isa Vector{Tuple{String,Vararg{String}}}
        variables = [Tuple(group .|> Meta.parse .|> replace_indices) for group in variables]
    elseif variables isa Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}
        variables = Tuple(Tuple(group .|> Meta.parse .|> replace_indices) for group in variables)
    end
    
    if !is_grouped_covariance_input(variables)
        # Not grouped, return single group
        idx = parse_variables_input_to_index(variables, constants)
        return [collect(idx)]
    end
    
    # Parse each group (convert tuples to vectors for uniform handling)
    groups = Vector{Vector{Int}}()
    for group in variables
        group_vec = group isa Tuple ? collect(group) : group
        if length(setdiff(group_vec, T.var)) > 0
            @warn "The following variables are not part of the model: " * join(string.(setdiff(group_vec,T.var)),", ") * ". Use `get_variables(𝓂)` to list valid names."
            push!(groups, Int[])
        else
            push!(groups, Int.(indexin(group_vec, T.var)))
        end
    end
    
    return groups
end




function parse_shocks_input_to_index(shocks::Expr, constants::constants)
    parsed = replace_indices(shocks)
    if parsed isa Symbol
        return parse_shocks_input_to_index(parsed, constants)
    end
    @warn "Invalid `shocks` argument. Provide a Symbol, Tuple, Vector, Matrix, or one of the documented selectors such as `:all`."
    return Int[]
end

function parse_shocks_input_to_index(shocks::BitVector, constants::constants)
    T = constants.post_model_macro
    if length(shocks) != T.nExo
        @warn "Invalid `shocks` argument. BitVector length does not match number of shocks."
        return Int[]
    end
    return getindex(1:T.nExo, shocks)
end

function parse_shocks_input_to_index(shocks::BitMatrix, constants::constants)
    T = constants.post_model_macro
    if size(shocks, 1) != T.nExo
        @warn "Invalid `shocks` argument. BitMatrix row count does not match number of shocks."
        return Int[]
    end
    return getindex(1:T.nExo, vec(sum(shocks, dims = 2) .> 0))
end

function parse_shocks_input_to_index(shocks::Union{Symbol_input, String_input}, constants::constants)
    T = constants.post_model_macro
    

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
            @warn "The following shocks are not part of the model: " * join(string.(setdiff(shocks,T.exo)),", ") * ". Use `get_shocks(𝓂)` to list valid shock names."
            shock_idx = Int64[]
        else
            shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(shocks .== T.exo,dims= 2))))
        end
    elseif shocks isa Vector{Symbol}
        if length(setdiff(shocks,T.exo)) > 0
            @warn "The following shocks are not part of the model: " * join(string.(setdiff(shocks,T.exo)),", ") * ". Use `get_shocks(𝓂)` to list valid shock names."
            shock_idx = Int64[]
        else
            shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(shocks,1,length(shocks)) .== T.exo, dims= 2))))
        end
    elseif shocks isa Tuple{Symbol, Vararg{Symbol}}
        if length(setdiff(shocks,T.exo)) > 0
            @warn "The following shocks are not part of the model: " * join(string.(setdiff(Symbol.(collect(shocks)),T.exo)),", ") * ". Use `get_shocks(𝓂)` to list valid shock names."
            shock_idx = Int64[]
        else
            shock_idx = getindex(1:T.nExo,convert(Vector{Bool},vec(sum(reshape(collect(shocks),1,length(shocks)) .== T.exo,dims= 2))))
        end
    elseif shocks isa Symbol
        if length(setdiff([shocks],T.exo)) > 0
            @warn "The following shock is not part of the model: " * join(string(setdiff([shocks],T.exo)[1]),", ") * ". Use `get_shocks(𝓂)` to list valid shock names."
            # TODO: mention shocks part of the model
            shock_idx = Int64[]
        else
            shock_idx = getindex(1:T.nExo,shocks .== T.exo)
        end
    else
        @warn "Invalid `shocks` argument. Provide a Symbol, Tuple, Vector, Matrix, or one of the documented selectors such as `:all`."
        shock_idx = Int64[]
    end
    return shock_idx
end



# end # dispatch_doctor

# function Stateupdate(::Val{:first_order}, states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     return [P.first_order.solution_matrix * [states[1][T.past_not_future_and_mixed_idx]; shocks]]
# end

# function Stateupdate(::Val{:second_order}, states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     aug_state₁ = [states[1][T.past_not_future_and_mixed_idx]; shocks]

#     aug_state = [states[1][T.past_not_future_and_mixed_idx]; 1; shocks]

#     𝐒₁ = P.first_order.solution_matrix
#     𝐒₂ = P.second_order_solution * P.second_order_auxiliary_matrices.𝐔₂

#     return [𝐒₁ * aug_state₁ + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2]
# end

# function Stateupdate(::Val{:pruned_second_order}, pruned_states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     aug_state₁ = [pruned_states[1][T.past_not_future_and_mixed_idx]; 1; shocks]

#     aug_state₁̃ = [pruned_states[1][T.past_not_future_and_mixed_idx]; shocks]
#     aug_state₂̃ = [pruned_states[2][T.past_not_future_and_mixed_idx]; zero(shocks)]
    
#     𝐒₁ = P.first_order.solution_matrix
#     𝐒₂ = P.second_order_solution * P.second_order_auxiliary_matrices.𝐔₂

#     return [𝐒₁ * aug_state₁̃, 𝐒₁ * aug_state₂̃ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2]
# end

# function Stateupdate(::Val{:third_order}, states::Vector{Vector{S}}, shocks::Vector{R}, T::timings, P::perturbation) where {S <: Real, R <: Real}
#     aug_state₁ = [states[1][T.past_not_future_and_mixed_idx]; shocks]

#     aug_state = [states[1][T.past_not_future_and_mixed_idx]; 1; shocks]

#     𝐒₁ = P.first_order.solution_matrix
#     𝐒₂ = P.second_order_solution * P.second_order_auxiliary_matrices.𝐔₂
#     𝐒₃ = P.third_order_solution * P.third_order_auxiliary_matrices.𝐔₃

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
#     𝐒₂ = P.second_order_solution * P.second_order_auxiliary_matrices.𝐔₂
#     𝐒₃ = P.third_order_solution * P.third_order_auxiliary_matrices.𝐔₃
    
#     kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

#     return [𝐒₁ * aug_state₁̃, 𝐒₁ * aug_state₂̃ + 𝐒₂ * kron_aug_state₁ / 2, 𝐒₁ * aug_state₃̃ + 𝐒₂ * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒₃ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
# end

end # dispatch_doctor

noop_state_update(::Float64, ::Float64) = nothing

function parse_algorithm_to_state_update(algorithm::Symbol, 𝓂::ℳ, occasionally_binding_constraints::Bool)::Tuple{Function, Bool}
    state_update::Function = noop_state_update
    pruning::Bool = algorithm ∈ [:pruned_second_order, :pruned_third_order]

    past_idx = 𝓂.constants.post_model_macro.past_not_future_and_mixed_idx
    nPast = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    nVars = 𝓂.constants.post_model_macro.nVars

    if occasionally_binding_constraints
        Ŝ₁ = 𝓂.caches.first_order_obc_solution_matrix

        if algorithm == :first_order
            state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                aug_state = [state[past_idx]; shock]
                return Ŝ₁ * aug_state
            end
        elseif algorithm ∈ [:second_order, :third_order]
            𝐒₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂
            Ŝ₁̂ = [Ŝ₁[:,1:nPast] zeros(nVars) Ŝ₁[:,nPast+1:end]]

            if algorithm == :second_order
                state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[past_idx]; 1; shock]
                    return Ŝ₁̂ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                end
            else  # :third_order
                𝐒₃ = 𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃
                state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[past_idx]; 1; shock]
                    return Ŝ₁̂ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
                end
            end
        elseif algorithm == :pruned_second_order
            𝐒₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂
            Ŝ₁̂ = [Ŝ₁[:,1:nPast] zeros(nVars) Ŝ₁[:,nPast+1:end]]

            state_update = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
                aug_state₁ = [pruned_states[1][past_idx]; 1; shock]
                aug_state₂ = [pruned_states[2][past_idx]; 0; zero(shock)]
                return [Ŝ₁̂ * aug_state₁, Ŝ₁̂ * aug_state₂ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2]
            end
        elseif algorithm == :pruned_third_order
            𝐒₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂
            𝐒₃ = 𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃
            Ŝ₁̂ = [Ŝ₁[:,1:nPast] zeros(nVars) Ŝ₁[:,nPast+1:end]]

            state_update = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
                aug_state₁ = [pruned_states[1][past_idx]; 1; shock]
                aug_state₁̂ = [pruned_states[1][past_idx]; 0; shock]
                aug_state₂ = [pruned_states[2][past_idx]; 0; zero(shock)]
                aug_state₃ = [pruned_states[3][past_idx]; 0; zero(shock)]
                kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)
                return [Ŝ₁̂ * aug_state₁, Ŝ₁̂ * aug_state₂ + 𝐒₂ * kron_aug_state₁ / 2, Ŝ₁̂ * aug_state₃ + 𝐒₂ * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒₃ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
            end
        end
    else
        if algorithm == :first_order
            S₁ = 𝓂.caches.first_order_solution_matrix
            state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                aug_state = [state[past_idx]; shock]
                return S₁ * aug_state
            end
        elseif algorithm ∈ [:second_order, :third_order]
            S₁ = 𝓂.caches.first_order_solution_matrix
            𝐒₁ = [S₁[:,1:nPast] zeros(nVars) S₁[:,nPast+1:end]]
            𝐒₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂

            if algorithm == :second_order
                state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[past_idx]; 1; shock]
                    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                end
            else  # :third_order
                𝐒₃ = 𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃
                state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[past_idx]; 1; shock]
                    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
                end
            end
        elseif algorithm == :pruned_second_order
            S₁ = 𝓂.caches.first_order_solution_matrix
            𝐒₁ = [S₁[:,1:nPast] zeros(nVars) S₁[:,nPast+1:end]]
            𝐒₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂

            state_update = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
                aug_state₁ = [pruned_states[1][past_idx]; 1; shock]
                aug_state₂ = [pruned_states[2][past_idx]; 0; zero(shock)]
                return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2]
            end
        elseif algorithm == :pruned_third_order
            S₁ = 𝓂.caches.first_order_solution_matrix
            𝐒₁ = [S₁[:,1:nPast] zeros(nVars) S₁[:,nPast+1:end]]
            𝐒₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂
            𝐒₃ = 𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃

            state_update = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
                aug_state₁ = [pruned_states[1][past_idx]; 1; shock]
                aug_state₁̂ = [pruned_states[1][past_idx]; 0; shock]
                aug_state₂ = [pruned_states[2][past_idx]; 0; zero(shock)]
                aug_state₃ = [pruned_states[3][past_idx]; 0; zero(shock)]
                kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)
                return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * kron_aug_state₁ / 2, 𝐒₁ * aug_state₃ + 𝐒₂ * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒₃ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
            end
        end
    end

    return (state_update, pruning)
end


@stable default_mode = "disable" begin

function get_custom_steady_state_buffer!(𝓂::ℳ, expected_length::Int)
    buffer = 𝓂.workspaces.custom_steady_state_buffer

    if length(buffer) != expected_length
        buffer = Vector{Float64}(undef, expected_length)
        𝓂.workspaces.custom_steady_state_buffer = buffer
    end

    return buffer
end

function evaluate_custom_steady_state_function(𝓂::ℳ,
                                                parameter_values::AbstractVector{S},
                                                expected_length::Int,
                                                expected_parameter_length::Int)::Vector{S} where {S <: Real}
    if length(parameter_values) != expected_parameter_length
        throw(ArgumentError("Custom steady state function expected $expected_parameter_length parameters, got $(length(parameter_values))."))
    end

    has_inplace = hasmethod(𝓂.functions.NSSS_custom, Tuple{typeof(parameter_values), typeof(parameter_values)})

    if has_inplace
        get_custom_steady_state_buffer!(𝓂, expected_length)
        
        output = Vector{S}(undef, expected_length)
        try 
            𝓂.functions.NSSS_custom(output, parameter_values)
        catch
        end
        return output
    elseif applicable(𝓂.functions.NSSS_custom, parameter_values)
        raw_result = try
            𝓂.functions.NSSS_custom(parameter_values)
        catch
            nothing
        end
        
        if raw_result === nothing
            return Vector{S}(fill(NaN, expected_length))
        end
        
        if !(raw_result isa AbstractVector)
            throw(ArgumentError("Custom steady state function returned $(typeof(raw_result)); expected an AbstractVector."))
        end
        
        if length(raw_result) != expected_length
            throw(ArgumentError("Custom steady state function returned $(length(raw_result)) values, expected $expected_length."))
        end
        
        return Vector{S}(raw_result)
    else
        throw(ArgumentError("Custom steady state function must accept either (parameters) or (out, parameters)."))
    end
end

# @stable default_mode = "disable" begin

function find_variables_to_exclude(𝓂::ℳ, observables::Vector{Symbol})
    # reduce system
    vars_to_exclude = setdiff(𝓂.constants.post_model_macro.present_only, observables)

    # Mapping variables to their equation index
    variable_to_equation = Dict{Symbol, Vector{Int}}()
    for var in vars_to_exclude
        for (eq_idx, vars_set) in enumerate(𝓂.constants.post_model_macro.dyn_var_present_list)
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

"""
    update_perturbation_counter!(counters::SolveCounters, solved::Bool; estimation::Bool = false, order::Int = 1)

Updates the perturbation solve counters based on whether the solve was successful and the perturbation order.
Always increments the total counter, and increments the failed counter if the solve failed.
"""
function update_perturbation_counter!(counters::SolveCounters, solved::Bool; estimation::Bool = false, order::Int = 1)
    if order == 1
        if estimation
            counters.first_order_solves_total_estimation += 1
            if !solved
                counters.first_order_solves_failed_estimation += 1
            end
        else
            counters.first_order_solves_total += 1
            if !solved
                counters.first_order_solves_failed += 1
            end
        end
    elseif order == 2
        if estimation
            counters.second_order_solves_total_estimation += 1
            if !solved
                counters.second_order_solves_failed_estimation += 1
            end
        else
            counters.second_order_solves_total += 1
            if !solved
                counters.second_order_solves_failed += 1
            end
        end
    elseif order == 3
        if estimation
            counters.third_order_solves_total_estimation += 1
            if !solved
                counters.third_order_solves_failed_estimation += 1
            end
        else
            counters.third_order_solves_total += 1
            if !solved
                counters.third_order_solves_failed += 1
            end
        end
    end
end

"""
    update_ss_counter!(counters::SolveCounters, solved::Bool; estimation::Bool = false)

Updates the steady state solve counters based on whether the solve was successful.
Always increments the total counter, and increments the failed counter if the solve failed.
"""
function update_ss_counter!(counters::SolveCounters, solved::Bool; estimation::Bool = false)
    if estimation
        counters.ss_solves_total_estimation += 1
        if !solved
            counters.ss_solves_failed_estimation += 1
        end
    else
        counters.ss_solves_total += 1
        if !solved
            counters.ss_solves_failed += 1
        end
    end
end

function get_NSSS_and_parameters(𝓂::ℳ, 
                                    parameter_values::Vector{S}; 
                                    opts::CalculationOptions = merge_calculation_options(),
                                    cold_start::Bool = false,
                                    estimation::Bool = false)::Tuple{Vector{S}, Tuple{S, Int}} where S <: Real
                                    # timer::TimerOutput = TimerOutput(),
    # @timeit_debug timer "Calculate NSSS" begin
    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    
    # Use custom steady state function if available, otherwise use default solver
    if 𝓂.functions.NSSS_custom isa Function
        vars_in_ss_equations = ms.vars_in_ss_equations
        expected_length = length(vars_in_ss_equations) + length(𝓂.equations.calibration_parameters)

        SS_and_pars_tmp = evaluate_custom_steady_state_function(
            𝓂,
            parameter_values,
            expected_length,
            length(𝓂.constants.post_complete_parameters.parameters),
        )

        residual = zeros(length(𝓂.equations.steady_state) + length(𝓂.equations.calibration))
        
        𝓂.functions.NSSS_check(residual, parameter_values, SS_and_pars_tmp)
        
        solution_error = ℒ.norm(residual)

        iters = 0

        # if !isfinite(solution_error) || solution_error > opts.tol.NSSS_acceptance_tol
        #     throw(ArgumentError("Custom steady state function failed steady state check: residual $solution_error > $(opts.tol.NSSS_acceptance_tol). Parameters: $(parameter_values). Steady state and parameters returned: $(SS_and_pars_tmp)."))
        # end
        X = ms.custom_ss_expand_matrix
        SS_and_pars = X * SS_and_pars_tmp
    else
        fastest_idx = 𝓂.constants.post_complete_parameters.nsss_fastest_solver_parameter_idx
        preferred_solver_parameter_idx = fastest_idx < 1 || fastest_idx > length(DEFAULT_SOLVER_PARAMETERS) ? 1 : fastest_idx
        SS_and_pars, (solution_error, iters) = solve_nsss_wrapper(parameter_values, 𝓂, opts.tol, opts.verbose, cold_start, DEFAULT_SOLVER_PARAMETERS, preferred_solver_parameter_idx = preferred_solver_parameter_idx)
    end

    # Update counters
    solved = !(solution_error > opts.tol.NSSS_acceptance_tol || isnan(solution_error))
    update_ss_counter!(𝓂.counters, solved, estimation = estimation)
    
    if !solved
        if opts.verbose 
            println("Failed to find NSSS") 
        end
        # return (SS_and_pars, (10.0, iters))#, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # end # timeit_debug
    return SS_and_pars, (solution_error, iters)
end


function check_bounds(parameter_values::Vector{S}, 𝓂::ℳ)::Bool where S <: Real
    if !all(isfinite,parameter_values) return true end

    if length(𝓂.constants.post_parameters_macro.bounds) > 0 
        for (k,v) in 𝓂.constants.post_parameters_macro.bounds
            if k ∈ 𝓂.constants.post_complete_parameters.parameters
                if min(max(parameter_values[indexin([k], 𝓂.constants.post_complete_parameters.parameters)][1], v[1]), v[2]) != parameter_values[indexin([k], 𝓂.constants.post_complete_parameters.parameters)][1]
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
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false) where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_stochastic_steady_state(Val(:second_order), parameter_values, 𝓂, opts = opts, estimation = estimation) # timer = timer, 
    
    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        if opts.verbose println("Could not find 2nd order stochastic steady state") end
        return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], collect(sss), converged
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = collect(sss) - all_SS

    return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:pruned_second_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{S}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_stochastic_steady_state(Val(:pruned_second_order), parameter_values, 𝓂, opts = opts, estimation = estimation) # timer = timer, 

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        if opts.verbose println("Could not find 2nd order stochastic steady state") end
        return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], [zeros(𝓂.constants.post_model_macro.nVars), zeros(𝓂.constants.post_model_macro.nVars)], converged
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = [zeros(𝓂.constants.post_model_macro.nVars), collect(sss) - all_SS]

    return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:third_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{S}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_stochastic_steady_state(Val(:third_order), parameter_values, 𝓂, opts = opts, estimation = estimation) # timer = timer,  

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        if opts.verbose println("Could not find 3rd order stochastic steady state") end
        return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], collect(sss), converged
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = collect(sss) - all_SS

    return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged
end



function get_relevant_steady_state_and_state_update(::Val{:pruned_third_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{S}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_stochastic_steady_state(Val(:pruned_third_order), parameter_values, 𝓂, opts = opts, estimation = estimation) # timer = timer, 

    if !converged || solution_error > opts.tol.NSSS_acceptance_tol
        if opts.verbose println("Could not find 3rd order stochastic steady state") end
        return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], [zeros(𝓂.constants.post_model_macro.nVars), zeros(𝓂.constants.post_model_macro.nVars), zeros(𝓂.constants.post_model_macro.nVars)], converged
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = [zeros(𝓂.constants.post_model_macro.nVars), collect(sss) - all_SS, zeros(𝓂.constants.post_model_macro.nVars)]

    return 𝓂.constants, SS_and_pars, [𝐒₁, 𝐒₂, 𝐒₃], state, converged
end


function get_relevant_steady_state_and_state_update(::Val{:first_order}, 
                                                    parameter_values::Vector{S}, 
                                                    𝓂::ℳ; 
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    estimation::Bool = false)::Tuple{constants, Vector{S}, Union{Matrix{S},Vector{AbstractMatrix{S}}}, Vector{Vector{Float64}}, Bool} where S <: Real
                                                    # timer::TimerOutput = TimerOutput(), 
    # Initialize constants at entry point
    constants_obj = initialise_constants!(𝓂)

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameter_values, opts = opts, estimation = estimation) # timer = timer,

    state = zeros(𝓂.constants.post_model_macro.nVars)

    if solution_error > opts.tol.NSSS_acceptance_tol # || isnan(solution_error) if it's NaN the first condition is false anyway
        # println("NSSS not found")
        return 𝓂.constants, SS_and_pars, zeros(S, 0, 0), [state], solution_error < opts.tol.NSSS_acceptance_tol
    end

    ∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian) # , timer = timer)# |> Matrix

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants_obj,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution)


    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        # println("NSSS not found")
        return 𝓂.constants, SS_and_pars, zeros(S, 0, 0), [state], solved
    end

    return 𝓂.constants, SS_and_pars, 𝐒₁, [state], solved
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

# Include ForwardDiff Dual specializations for forward-mode AD
# Must be at the end of the module because they depend on function definitions
include("./custom_autodiff_rules/forwarddiff.jl")

# Include rrule definitions for reverse-mode AD (Zygote/ChainRulesCore)
# Must be at the end of the module because rrules depend on function definitions
include("./custom_autodiff_rules/rrules.jl")

end
