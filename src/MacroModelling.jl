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
# import ForwardDiff as ℱ  # moved to ForwardDiffExt
# import Diffractor: DiffractorForwardBackend
# 𝒷 = 𝒜.ForwardDiffBackend
# 𝒷 = Diffractor.DiffractorForwardBackend

import LoopVectorization: @turbo
# import Polyester
import NLopt
# import Zygote
import SparseArrays
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
import MacroTools: unblock, postwalk, prewalk, @capture, flatten, rmlines

# import SpeedMapping: speedmapping
import Suppressor: @suppress
import REPL
import Unicode
import MatrixEquations # good overview: https://cscproxy.mpi-magdeburg.mpg.de/mpcsc/benner/talks/Benner-Melbourne2019.pdf
# import NLboxsolve: nlboxsolve
# using NamedArrays
import AxisKeys

import Random
import Random: AbstractRNG

import ChainRulesCore: rrule, NoTangent, @thunk, ProjectTo, unthunk, AbstractZero
# import RecursiveFactorization as RF

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

# Imports
include("default_options.jl")
include("common_docstrings.jl")
include("structures.jl")
include("./steady_state/solver_parameters.jl")
include("options_and_caches.jl")
include("./steady_state/nsss_solver.jl")
include("occasionally_binding_constraints.jl")
include("./parser/macros.jl")
include("./parser/equation_processing.jl")
include("./parser/model_setup.jl")
include("./parser/equation_modification.jl")
include("get_functions.jl")
include("dynare.jl")
include("inspect.jl")
include("aumann_shapley.jl")
include("moments.jl")
include("./algorithms/fast_lapack_wrappers.jl")
include("./perturbation/derivatives.jl")
include("./perturbation/solution.jl")
include("./steady_state/stochastic_steady_state.jl")
include("impulse_response_function.jl")

include("./algorithms/preconditioner.jl")
include("./algorithms/sylvester.jl")
include("./algorithms/lyapunov.jl")
include("./algorithms/nonlinear_solver.jl")
include("./algorithms/quadratic_matrix_equation.jl")

include("./filter/find_shocks.jl")
include("./filter/decomposition.jl")
include("./filter/inversion.jl")
include("./filter/kalman.jl")
include("./filter/particle.jl")


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
export get_shock_decomposition, get_model_estimates, get_estimated_shocks, get_estimated_variables, get_estimated_variable_standard_deviations, get_loglikelihood, get_loglikelihood
export Tolerances, SolverTolerances, NsssTolerances, AdTolerances, FirstOrderTolerances, HigherOrderTolerances

export translate_mod_file, translate_dynare_file, import_model, import_dynare
export write_mod_file, write_dynare_file, write_to_dynare_file, write_to_dynare, export_dynare, export_to_dynare, export_mod_file, export_model

export get_equations, get_steady_state_equations, get_dynamic_equations, get_calibration_equations, get_parameters, get_calibrated_parameters, get_parameters_in_equations, get_parameters_defined_by_parameters, get_parameters_defining_parameters, get_calibration_equation_parameters, get_variables, get_nonnegativity_auxiliary_variables, get_dynamic_auxiliary_variables, get_shocks, get_state_variables, get_jump_variables, get_missing_parameters, has_missing_parameters, get_solution_counts, print_solution_counts
export write_julia_model_file, replace_equations!, replace_calibration_equations!
export update_equations!, update_calibration_equations!, add_equation!, add_calibration_equation!, remove_equation!, remove_calibration_equation!, get_revision_history
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

# end # dispatch_doctor





# Generic primal-value extraction — identity for plain reals.
# ForwardDiffExt extends this for ForwardDiff.Dual numbers.
primal(x::Real) = x


function normalize_filtering_options(filter::Symbol,
                                      smooth::Bool,
                                      algorithm::Symbol,
                                      shock_decomposition::Bool,
                                      warmup_iterations::Int;
                                      maxlog::Int = DEFAULT_MAXLOG)
    # `:particle` is a convenience alias for the bootstrap particle filter.
    filter = get(PARTICLE_FILTER_ALIASES, filter, filter)

    @assert filter ∈ SUPPORTED_FILTERS "Unsupported `filter = :$(filter)`. Choose the Kalman filter (`:kalman`, linear models), the inversion filter (`:inversion`, linear and nonlinear models), or one of the particle filters (`:bootstrap_particle`, `:auxiliary_particle`, `:tempered_particle`; linear and nonlinear models). `:particle` is accepted as an alias for `:bootstrap_particle`."

    is_particle = filter ∈ PARTICLE_FILTERS

    pruning = algorithm ∈ (:pruned_second_order, :pruned_third_order)

    if shock_decomposition && algorithm ∈ (:second_order, :third_order)
        @info "Shock decomposition is not available for $(algorithm) solutions, but is available for first order, pruned second order, and pruned third order solutions. Setting `shock_decomposition = false`." maxlog = maxlog
        shock_decomposition = false
    end

    # Higher-order solutions are handled by the inversion filter by default, but
    # the particle filters are explicitly valid at every order too.
    if algorithm != :first_order && filter != :inversion && !is_particle
        @info "Higher order solution algorithms only support the inversion and particle filters. Setting `filter = :inversion`." maxlog = maxlog
        filter = :inversion
        is_particle = false
    end

    # Smoothing is available for the Kalman filter (Durbin-Koopman smoother) and
    # for the particle filters (fixed-interval smoothing along the filter's
    # genealogy).
    #
    # For the inversion filter there is nothing left to smooth. Given x₀ it solves
    # yₜ = g(xₜ₋₁, εₜ)[observables] for εₜ exactly, so xₜ is a *deterministic*
    # function of y₁..ₜ — the filtering distribution is a point mass. Conditioning
    # on future data cannot sharpen a point mass, hence p(xₜ|y₁..T) = p(xₜ|y₁..ₜ)
    # and a backward pass recovers exactly the shocks the forward pass already
    # found. The filtered estimate *is* the smoothed estimate; `smooth` is a no-op
    # rather than an unsupported option.
    #
    # Two caveats, neither of which a smoothing recursion would fix. The initial
    # state x₀ is not identified by the data and is fixed at the (stochastic)
    # steady state; refining it is a fixed-point problem over x₀, not a backward
    # recursion. And with more shocks than observables the per-period solve picks
    # the minimum-norm εₜ (at higher order, the root whose basin contains the
    # origin — see `find_shocks`), which is a per-period choice a smoother could in
    # principle redistribute across time; doing so would be a different estimator,
    # not the inversion filter's smoother.
    if filter == :inversion && smooth
        @info "The inversion filter identifies the state exactly, so its smoothed and filtered estimates coincide. Setting `smooth = false`." maxlog = maxlog
        smooth = false
    end

    if warmup_iterations > 0
        if filter == :kalman || is_particle
            @info "`warmup_iterations` is not a valid argument for the $(filter == :kalman ? "Kalman" : "particle") filter. Ignoring input for `warmup_iterations`." maxlog = maxlog
            warmup_iterations = 0
        end
    end

    return filter, smooth, algorithm, shock_decomposition, pruning, warmup_iterations
end


function normalize_presample_periods(presample_periods::Int,
                                     data_length::Integer;
                                     maxlog::Int = DEFAULT_MAXLOG)
    @assert presample_periods >= 0 "`presample_periods` must be non-negative."
    @assert data_length >= 0 "`data_length` must be non-negative."

    normalized_presample_periods = min(presample_periods, Int(data_length))

    if normalized_presample_periods != presample_periods
        @info "`presample_periods = $(presample_periods)` exceeds the available data length ($(data_length)). Setting `presample_periods = $(normalized_presample_periods)`." maxlog = maxlog
    end

    return normalized_presample_periods
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


@unstable function process_shocks_input(shocks::Union{Symbol_input, String_input, Matrix{Float64}, KeyedArray{Float64}},
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
    else
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





function invalidate_cache_validity!(𝓂::ℳ)
    𝓂.caches.valid_for.non_stochastic_steady_state = Float64[]
    𝓂.caches.valid_for.jacobian = Float64[]
    𝓂.caches.valid_for.hessian = Float64[]
    𝓂.caches.valid_for.third_order_derivatives = Float64[]
    𝓂.caches.valid_for.first_order_solution = Float64[]
    𝓂.caches.valid_for.first_order_obc_solution = Float64[]
    𝓂.caches.valid_for.second_order_solution = Float64[]
    𝓂.caches.valid_for.pruned_second_order_solution = Float64[]
    𝓂.caches.valid_for.second_order_stochastic_steady_state = Float64[]
    𝓂.caches.valid_for.pruned_second_order_stochastic_steady_state = Float64[]
    𝓂.caches.valid_for.third_order_solution = Float64[]
    𝓂.caches.valid_for.pruned_third_order_solution = Float64[]
    𝓂.caches.valid_for.third_order_stochastic_steady_state = Float64[]
    𝓂.caches.valid_for.pruned_third_order_stochastic_steady_state = Float64[]
    𝓂.caches.valid_for.covariance_first_order = Float64[]
    𝓂.caches.valid_for.covariance_second_order = Float64[]
    𝓂.caches.valid_for.covariance_third_order = Float64[]
    𝓂.caches.valid_for.covariance_third_order_obs_key = Int[]
    𝓂.caches.valid_for.covariance_third_order_autocorr = Float64[]
    𝓂.caches.valid_for.covariance_third_order_autocorr_obs_key = Int[]
    𝓂.caches.valid_for.covariance_third_order_autocorr_periods = Int[]
    return nothing
end


function reset_nsss_solver_cache!(𝓂::ℳ)
    empty!(𝓂.caches.solver)

    c = 𝓂.constants.nsss_solver
    ms = 𝓂.constants.post_complete_parameters
    seed = Vector{Vector{Float64}}()

    for step_idx in 1:c.n_steps
        if c.step_types[step_idx] == NUMERICAL_STEP
            wr = c.write_ranges[step_idx]
            nbr = c.numerical_bounds_ranges[step_idx]
            guess_len = min(length(wr), length(nbr))
            guesses = Vector{Float64}(undef, guess_len)

            for i in 1:guess_len
                sol_idx = c.write_indices[wr[i]]
                sol_name = sol_idx <= length(ms.nsss_sol_names) ? ms.nsss_sol_names[sol_idx] : Symbol("")
                guesses[i] = get(𝓂.constants.post_parameters_macro.guess, sol_name, Inf)
            end

            push!(seed, guesses)
            push!(seed, Float64[Inf])
        end
    end

    push!(seed, fill(Inf, length(ms.parameters)))
    push!(𝓂.caches.solver, seed)

    return nothing
end


function clear_solution_caches!(𝓂::ℳ, algorithm::Symbol)
    reset_nsss_solver_cache!(𝓂)

    𝓂.caches.first_order_solution_matrix = zeros(0,0)
    𝓂.caches.first_order_obc_solution_matrix = zeros(0,0)
    𝓂.caches.qme_solution = zeros(0,0)
    𝓂.caches.has_unit_roots = false
    𝓂.caches.second_order_solution = spzeros(0,0)
    𝓂.caches.third_order_solution = spzeros(0,0)

    𝓂.caches.second_order_stochastic_steady_state = Float64[]
    𝓂.caches.pruned_second_order_stochastic_steady_state = Float64[]
    𝓂.caches.third_order_stochastic_steady_state = Float64[]
    𝓂.caches.pruned_third_order_stochastic_steady_state = Float64[]

    resize!(𝓂.caches.non_stochastic_steady_state, 0)

    invalidate_cache_validity!(𝓂)

    return nothing
end


const CACHE_VALIDITY_FIELDS = (
    :non_stochastic_steady_state,
    :jacobian,
    :hessian,
    :third_order_derivatives,
    :first_order_solution,
    :first_order_obc_solution,
    :second_order_solution,
    :pruned_second_order_solution,
    :second_order_stochastic_steady_state,
    :pruned_second_order_stochastic_steady_state,
    :third_order_solution,
    :pruned_third_order_solution,
    :third_order_stochastic_steady_state,
    :pruned_third_order_stochastic_steady_state,
    :covariance_first_order,
    :covariance_second_order,
    :covariance_third_order,
)


function cache_valid_for_parameters(valid_for::Vector{Float64}, parameters::AbstractVector{<:Real})::Bool
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



# Helper to convert dense matrix to sparse, avoiding the Julia 1.12 SparseArrays bug in
# `SparseMatrixCSC(::Matrix)`. Builds CSC arrays directly in column-major order:
# one counting pass + one fill pass with exact allocations and no COO→CSC sort step,
# so this is at least as fast as the stdlib path it replaces.
function dense_to_sparse(A::DenseMatrix{S}, tol::R) where {S <: Real, R <: AbstractFloat}
    m, n = size(A)
    nnz_count = 0
    @inbounds for v in A
        abs(v) > tol && (nnz_count += 1)
    end
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, nnz_count)
    nzval  = Vector{S}(undef, nnz_count)
    k = 0
    colptr[1] = 1
    @inbounds for j in 1:n
        for i in 1:m
            v = A[i, j]
            if abs(v) > tol
                k += 1
                rowval[k] = i
                nzval[k]  = v
            end
        end
        colptr[j + 1] = k + 1
    end
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

# No-tolerance variant matching `sparse(::Matrix)` semantics (drops only structural zeros via
# `iszero`, so AD-active values with zero primal but nonzero partials are preserved).
function dense_to_sparse(A::DenseMatrix{S}) where {S}
    m, n = size(A)
    nnz_count = 0
    @inbounds for v in A
        iszero(v) || (nnz_count += 1)
    end
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, nnz_count)
    nzval  = Vector{S}(undef, nnz_count)
    k = 0
    colptr[1] = 1
    @inbounds for j in 1:n
        for i in 1:m
            v = A[i, j]
            if !iszero(v)
                k += 1
                rowval[k] = i
                nzval[k]  = v
            end
        end
        colptr[j + 1] = k + 1
    end
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

# Passthrough for already-sparse inputs: lets callers safely funnel mixed Matrix/SparseMatrixCSC
# values (e.g., return types of `calculate_second_order_solution`) through the same code path.
dense_to_sparse(A::AbstractSparseMatrix) = A

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


@unstable function choose_matrix_format(A::ℒ.Adjoint{S, M}; 
                                density_threshold::Float64 = .1, 
                                min_length::Int = 1000,
                                tol::R = 1e-14,
                                multithreaded::Bool = true)::Union{Matrix{S}, SparseMatrixCSC{S, Int}, ThreadedSparseArrays.ThreadedSparseMatrixCSC{S, Int, SparseMatrixCSC{S, Int}}} where {R <: AbstractFloat, S <: Real, M <: AbstractMatrix{S}}
    if A.parent isa AbstractSparseMatrix || A.parent isa ThreadedSparseArrays.ThreadedSparseMatrixCSC
        # Materialise sparse adjoints as SparseMatrixCSC to avoid unsupported
        # ThreadedSparseMatrixCSC(::Adjoint{<:ThreadedSparseMatrixCSC}) conversion.
        return choose_matrix_format(sparse(A),
                                    density_threshold = density_threshold,
                                    min_length = min_length,
                                    multithreaded = multithreaded,
                                    tol = tol)
    else
        return choose_matrix_format(Matrix(A),
                                    density_threshold = density_threshold,
                                    min_length = min_length,
                                    multithreaded = multithreaded,
                                    tol = tol)
    end
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

@unstable function choose_matrix_format(A::DenseMatrix{S}; 
                                density_threshold::Float64 = .1, 
                                min_length::Int = 1000,
                                tol::R = 1e-14,
                                multithreaded::Bool = true)::Union{Matrix{S}, SparseMatrixCSC{S, Int}, ThreadedSparseArrays.ThreadedSparseMatrixCSC{S, Int, SparseMatrixCSC{S, Int}}} where {R <: AbstractFloat, S <: Real}
    if count(x -> abs(x) > tol, A) / length(A) < density_threshold && length(A) > min_length
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

@unstable function choose_matrix_format(A::AbstractSparseMatrix{S}; 
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


function sparse_preallocated!(Ŝ::Matrix{T}; ℂ::higher_order_workspace{T,F,H} = Higher_order_workspace()) where {T <: Real, F <: AbstractFloat, H <: Real}
    if !(eltype(ℂ.tmp_sparse_prealloc6[3]) == T)
        ℂ.tmp_sparse_prealloc6 = Higher_order_workspace(T, F)
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




# Dead code: compressed_kron (2-arg) — never called anywhere; rrule also dead
#=
# 2-arg overload: compressed_kron(A, σ)
# Computes  𝐔∇₃ * kron(A, σ) * 𝐂₃
# directly in compressed (sorted-triple) space without forming any n³×n³ intermediates.
#
# A is nᵣ × nᶜ (may be rectangular),  σ is nᵣ² × nᶜ².
# Output is m₃ᵣ × m₃ᶜ sparse where m₃ᵣ = nᵣ(nᵣ+1)(nᵣ+2)/6, m₃ᶜ = nᶜ(nᶜ+1)(nᶜ+2)/6.
#
# kron(A,σ) at row (i,j,k) col (a,b,c) equals A[i,a]*σ[(j-1)*nᵣ+k, (b-1)*nᶜ+c].
# 𝐔∇₃ sums all row triples that sort to (i₁≥j₁≥k₁); 𝐂₃ selects the sorted column (α≥β≥γ).
function compressed_kron(A::AbstractMatrix{TA},
                         σ::AbstractMatrix{Tσ};
                         tol::AbstractFloat = eps(),
                         sparse_preallocation::Tuple{Vector{Int}, Vector{Int}, Vector{<:Real}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{<:Real}} = (Int[], Int[], Float64[], Int[], Int[], Int[], Float64[])) where {TA <: Real, Tσ <: Real}

    T = promote_type(TA, Tσ)

    nᵣ, nᶜ = size(A)
    size(σ) == (nᵣ^2, nᶜ^2) || throw(DimensionMismatch("σ must be $(nᵣ^2)×$(nᶜ^2), got $(size(σ))"))

    m₃ᵣ = nᵣ * (nᵣ + 1) * (nᵣ + 2) ÷ 6
    m₃ᶜ = nᶜ * (nᶜ + 1) * (nᶜ + 2) ÷ 6

    # Convert to sparse for CSC iteration
    As = A isa SparseMatrixCSC ? A : sparse(A)
    σs = σ isa SparseMatrixCSC ? σ : sparse(σ)

    rv_A = SparseArrays.rowvals(As)
    nzv_A = nonzeros(As)
    rv_σ = SparseArrays.rowvals(σs)
    nzv_σ = nonzeros(σs)

    # --- sparse buffer management ---
    spI = sparse_preallocation[1]
    spJ = sparse_preallocation[2]
    spV_untyped = sparse_preallocation[3]
    spV = if eltype(spV_untyped) == T
        spV_untyped
    else
        Vector{T}(undef, length(spV_untyped))
    end

    lennz_A = nnz(As)
    lennz_σ = nnz(σs)
    len_A = length(A)
    len_σ = length(σ)

    avg_density = sqrt((lennz_A / max(len_A, 1)) * (lennz_σ / max(len_σ, 1)))

    if length(spI) == 0
        estimated_nnz = floor(Int, max(m₃ᵣ * m₃ᶜ * avg_density ^ 3, 10000))
        resize!(spI, estimated_nnz)
        resize!(spJ, estimated_nnz)
        resize!(spV, estimated_nnz)
    else
        estimated_nnz = length(spV)
        resize!(spI, estimated_nnz)
        resize!(spJ, estimated_nnz)
        resize!(spV, estimated_nnz)
    end

    II = spI
    JJ = spJ
    VV = spV

    cnt = 0

    # Iterate sorted column triples (α ≥ β ≥ γ) where α indexes A's columns
    # and (β, γ) index σ's columns via σ_col = (β-1)*nᶜ + γ.
    for α in 1:nᶜ
        rng_A = SparseArrays.nzrange(As, α)
        isempty(rng_A) && continue

        for β in 1:α
            for γ in 1:β
                σ_col = (β - 1) * nᶜ + γ
                rng_σ = SparseArrays.nzrange(σs, σ_col)
                isempty(rng_σ) && continue

                col = (α - 1) * α * (α + 1) ÷ 6 + (β - 1) * β ÷ 2 + γ

                @inbounds for pA in rng_A
                    i = rv_A[pA]
                    a_val = nzv_A[pA]

                    for pσ in rng_σ
                        s = rv_σ[pσ]
                        σ_val = nzv_σ[pσ]

                        val = a_val * σ_val
                        abs(val) > tol || continue

                        # Decompose σ row: s = (j-1)*nᵣ + k
                        j = (s - 1) ÷ nᵣ + 1
                        k = (s - 1) % nᵣ + 1

                        # Sort row triple (i, j, k) → (i₁ ≥ j₁ ≥ k₁)
                        i₁ = i; j₁ = j; k₁ = k
                        if i₁ < j₁; i₁, j₁ = j₁, i₁; end
                        if j₁ < k₁; j₁, k₁ = k₁, j₁; end
                        if i₁ < j₁; i₁, j₁ = j₁, i₁; end

                        row = (i₁ - 1) * i₁ * (i₁ + 1) ÷ 6 + (j₁ - 1) * j₁ ÷ 2 + k₁

                        cnt += 1

                        if cnt > estimated_nnz
                            estimated_nnz += Int(ceil(max(1000, estimated_nnz * 0.1)))
                            estimated_nnz = min(m₃ᵣ * m₃ᶜ, estimated_nnz)
                            resize!(II, estimated_nnz)
                            resize!(JJ, estimated_nnz)
                            resize!(VV, estimated_nnz)
                        end

                        II[cnt] = row
                        JJ[cnt] = col
                        VV[cnt] = val
                    end
                end
            end
        end
    end

    resize!(II, cnt)
    resize!(JJ, cnt)
    resize!(VV, cnt)

    # Sparse assembly with preallocation buffers
    klasttouch = sparse_preallocation[4]
    csrrowptr  = sparse_preallocation[5]
    csrcolval  = sparse_preallocation[6]
    csrnzval_untyped = sparse_preallocation[7]
    csrnzval = if eltype(csrnzval_untyped) == T
        csrnzval_untyped
    else
        Vector{T}(undef, length(csrnzval_untyped))
    end

    resize!(klasttouch, m₃ᶜ)
    resize!(csrrowptr, m₃ᵣ + 1)
    resize!(csrcolval, length(II))
    resize!(csrnzval, length(II))

    out = if cnt >= m₃ᶜ + 1
        sparse!(II, JJ, VV, m₃ᵣ, m₃ᶜ, +, klasttouch, csrrowptr, csrcolval, csrnzval, II, JJ, VV)
    else
        SparseArrays.sparse(II, JJ, VV, m₃ᵣ, m₃ᶜ)
    end

    if tol > 0
        droptol!(out, tol)
    end

    return out
end
=#




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

# Dead code: A_mult_kron_power_3_B — never called anywhere
# function A_mult_kron_power_3_B(A::AbstractSparseMatrix{R},
#                                 B::Union{ℒ.Adjoint{T,Matrix{T}},DenseMatrix{T}}; 
#                                 tol::AbstractFloat = eps()) where {R <: Real, T <: Real}
#     n_row = size(B,1)
#     n_col = size(B,2)
#
#     vals = T[]
#     rows = Int[]
#     cols = Int[]
#
#     Ar, Ac, Av = findnz(A)
#
#     for row in unique(Ar)
#         idx_mat, vals_mat = A[row,:] |> findnz
#
#         for col in 1:size(B,2)^3
#             col_1, col_3 = divrem((col - 1) % (n_col^2), n_col) .+ 1
#             col_2 = ((col - 1) ÷ (n_col^2)) + 1
#
#             mult_val = 0.0
#
#             for (i,idx) in enumerate(idx_mat)
#                 i_1, i_3 = divrem((idx - 1) % (n_row^2), n_row) .+ 1
#                 i_2 = ((idx - 1) ÷ (n_row^2)) + 1
#                 @inbounds mult_val += vals_mat[i] * B[i_1,col_1] * B[i_2,col_2] * B[i_3,col_3]
#             end
#
#             if abs(mult_val) > tol
#                 push!(vals,mult_val)
#                 push!(rows,row)
#                 push!(cols,col)
#             end
#         end
#     end
#
#     sparse(rows,cols,vals,size(A,1),size(B,2)^3)
# end



function get_and_check_observables(T::post_model_macro, data::KeyedArray)::Vector{Symbol}
    @assert size(data,1) <= T.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    observables = collect(axiskeys(data,1))

    @assert observables isa Vector{String} || observables isa Vector{Symbol}  "Make sure that the data has variables names as rows. They can be either Strings or Symbols."

    observables_symbols = observables isa String_input ? observables .|> Meta.parse .|> replace_indices : observables

    @assert length(setdiff(observables_symbols, T.var)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(observables_symbols, T.var))

    sort!(observables_symbols)
    
    return observables_symbols
end

# Dead code: bivariate_moment, product_moments, multiplicate, generateSumVectors — never called anywhere
# function bivariate_moment(moment::Vector{Int}, rho::Int)::Int
#     if (moment[1] + moment[2]) % 2 == 1
#         return 0
#     end
#
#     result = 1
#     coefficient = 1
#     odd_value = 2 * (moment[1] % 2)
#
#     for j = 1:min(moment[1] ÷ 2, moment[2] ÷ 2)
#         coefficient *= 2 * (moment[1] ÷ 2 + 1 - j) * (moment[2] ÷ 2 + 1 - j) * rho^2 / (j * (2 * j - 1 + odd_value))
#         result += coefficient
#     end
#
#     if odd_value == 2
#         result *= rho
#     end
#
#     result *= prod(1:2:moment[1]) * prod(1:2:moment[2])
#
#     return result
# end
#
#
# function product_moments(V, ii, nu)::Int
#     s = sum(nu)
#
#     if s == 0
#         return 1
#     elseif isodd(s)
#         return 0
#     end
#
#     mask = .!(nu .== 0)
#     nu = nu[mask]
#     ii = ii[mask]
#     V = V[ii, ii]
#
#     m, s2 = length(ii), s / 2
#
#     if m == 1
#         return (V^s2 * prod(1:2:s-1))[1]
#     elseif m == 2
#         if V[1,1]==0 || V[2,2]==0
#             return 0
#         end
#         rho = V[1, 2] / sqrt(V[1, 1] * V[2, 2])
#         return (V[1, 1]^(nu[1] / 2) * V[2, 2]^(nu[2] / 2) * bivariate_moment(nu, Int(rho)))[1]
#     end
#
#     inu = sortperm(nu, rev=true)
#
#     sort!(nu, rev=true)
#
#     V = V[inu, inu]
#
#     x = zeros(Int, 1, m)
#     V = V / 2
#     nu2 = nu' / 2
#     p = 2
#     q = nu2 * V * nu2'
#     y = 0
#
#     for _ in 1:round(Int, prod(nu .+ 1) / 2)
#         y += p * q^s2
#         for j in 1:m
#             if x[j] < nu[j]
#                 x[j] += 1
#                 p = -round(p * (nu[j] + 1 - x[j]) / x[j])
#                 q -= (2 * (nu2 - x) * V[:, j] .+ V[j, j])[1]
#                 break
#             else
#                 x[j] = 0
#                 p = isodd(nu[j]) ? -p : p
#                 q += (2 * nu[j] * (nu2 - x) * V[:, j] .- nu[j]^2 * V[j, j])[1]
#             end
#         end
#     end
#
#     return y / prod(1:s2)
# end
#
#
# function multiplicate(p::Int, order::Int)
#     # precompute p powers
#     pⁿ = [p^i for i in 0:order-1]
#
#     DP = spzeros(Bool, p^order, prod(p - 1 .+ (1:order)) ÷ factorial(order))
#
#     binom_p_ord = binomial(p + order - 1, order)
#
#     # Initialize index and binomial arrays
#     indexes = ones(Int, order)  # Vector to hold current indexes
#     binomials = zeros(Int, order)  # Vector to hold binomial values
#
#     # Helper function to handle the nested loops
#     function loop(level::Int)
#         for i=1:p
#             indexes[level] = i
#             binomials[level] = binomial(p + level - 1 - i, level)
#
#             if level < order  # If not at innermost loop yet, continue nesting
#                 loop(level + 1)
#             else  # At innermost loop, perform calculation
#                 n = sum((indexes[k] - 1) * pⁿ[k] for k in 1:order)
#                 m = binom_p_ord - sum(binomials[k] for k in 1:order)
#                 DP[n+1, m] = 1  # Arrays are 1-indexed in Julia
#             end
#         end
#     end
#
#     loop(1)  # Start the recursive loop
#
#     return DP
# end
#
#
# function generateSumVectors(vectorLength::Int, totalSum::Int)::Union{Vector{Int}, Vector{ℒ.Adjoint{Int, Vector{Int}}}}
#     # Base case: if vectorLength is 1, return totalSum
#     if vectorLength == 1
#         return [totalSum]
#     end
#
#     # Recursive case: generate all possible vectors for smaller values of vectorLength and totalSum
#     return [[currentInt; smallerVector...]' for currentInt in totalSum:-1:0 for smallerVector in generateSumVectors(vectorLength-1, totalSum-currentInt)]
# end







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
    SS_and_pars = copy(SS_and_pars) # decouple from workspace output_buffer before select_fastest overwrites it
    
    found_solution = true
    
    if !(𝓂.functions.NSSS_custom isa Function)
        select_fastest_SS_solver_parameters!(𝓂, tol = opts.tol)
        
        if solution_error > opts.tol.nsss.acceptance_tol
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
    
    return SS_and_pars, solution_error, found_solution
end

# Centralised helper to write symbolic derivatives and map functions


function calculate_SS_solver_runtime_and_loglikelihood(pars::Vector{Float64}, 𝓂::ℳ; tol::Tolerances = Tolerances())::Float64
    log_lik = 0.0
    log_lik -= -sum(pars[1:19])                                 # logpdf of a gamma dist with mean and variance 1
    σ = 5
    log_lik -= -log(σ * sqrt(2 * π)) - (pars[20]^2 / (2 * σ^2)) # logpdf of a normal dist with mean = 0 and variance = 5^2

    pars[1:2] = sort(pars[1:2], rev = true)

    par_inputs = solver_parameters(pars..., 1, 0.0, 2)

    reset_nsss_solver_cache!(𝓂)

    runtime = @elapsed outmodel = try solve_nsss_wrapper(𝓂.parameter_values, 𝓂, tol, false, true, [par_inputs]) catch end

    runtime = outmodel isa Tuple{Vector{Float64}, Tuple{Float64, Int64}} ? 
                    (outmodel[2][1] > tol.nsss.acceptance_tol) || !isfinite(outmodel[2][1]) ? 
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

    if solution_error < tol.nsss.acceptance_tol
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

    solved_NSSS = 𝓂.caches.solver[end]

    for (i_param, p) in enumerate(DEFAULT_SOLVER_PARAMETERS)
        times = Vector{Float64}(undef, n_samples)
        valid = true
        
        for i in 1:n_samples
            start_time = time()

            reset_nsss_solver_cache!(𝓂)

            SS_and_pars, (solution_error, iters) = solve_nsss_wrapper(𝓂.parameter_values, 𝓂, tol, false, true, [p])

            elapsed_time = time() - start_time

            times[i] = elapsed_time

            if solution_error > tol.nsss.acceptance_tol
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

    empty!(𝓂.caches.solver)
    push!(𝓂.caches.solver, solved_NSSS)

    if solved
        𝓂.constants.post_complete_parameters = update_post_complete_parameters(
            𝓂.constants.post_complete_parameters;
            nsss_fastest_solver_parameter_idx = best_idx,
        )
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
        if algorithm == :first_order
            SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

            @assert solution_error < opts.tol.nsss.acceptance_tol "Could not find non-stochastic steady state."

            ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)

            S₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                                constants,
                                                                𝓂.workspaces,
                                                                𝓂.caches;
                                                                opts = opts,
                                                                initial_guess = 𝓂.caches.qme_solution,
                                                                parameter_values = 𝓂.parameter_values)

            update_perturbation_counter!(𝓂.counters, solved, order = 1)

            @assert solved "Could not find stable first order solution."

        elseif algorithm == :second_order
            sss_result = calculate_stochastic_steady_state(Val(:second_order), 𝓂.parameter_values, 𝓂, opts = opts)
            if !sss_result[2]  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

        elseif algorithm == :pruned_second_order
            sss_result = calculate_stochastic_steady_state(Val(:pruned_second_order), 𝓂.parameter_values, 𝓂, opts = opts)
            if !sss_result[2]  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

        elseif algorithm == :third_order
            calculate_stochastic_steady_state(Val(:second_order), 𝓂.parameter_values, 𝓂, opts = opts)
            sss_result = calculate_stochastic_steady_state(Val(:third_order), 𝓂.parameter_values, 𝓂, opts = opts)
            if !sss_result[2]  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end

        elseif algorithm == :pruned_third_order
            calculate_stochastic_steady_state(Val(:pruned_second_order), 𝓂.parameter_values, 𝓂, opts = opts)
            sss_result = calculate_stochastic_steady_state(Val(:pruned_third_order), 𝓂.parameter_values, 𝓂, opts = opts)
            if !sss_result[2]  @warn "Solution does not have a stochastic steady state. Try reducing shock sizes by multiplying them with a number < 1." end
        end

        if obc
            calculate_first_order_obc_solution!(𝓂, constants, opts)
        end

    end
    
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
        while length(𝓂.caches.solver) > 0
            pop!(𝓂.caches.solver)
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






@unstable function parse_variables_input_to_index(variables::Union{Symbol_input, String_input, Vector{Vector{Symbol}}, Vector{Tuple{Symbol,Vararg{Symbol}}}, Vector{Vector{Symbol}}, Tuple{Tuple{Symbol,Vararg{Symbol}}, Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}}, 𝓂::ℳ)::Union{UnitRange{Int}, Vector{Int}}
    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    if variables == :all_excluding_auxiliary_and_obc
        return ms.vars_idx_excluding_aux_obc
    elseif variables == :all_excluding_obc
        return ms.vars_idx_excluding_obc
    end

    return parse_variables_input_to_index(variables, 𝓂.constants)
end

@unstable function parse_variables_input_to_index(variables::Union{Symbol_input, String_input, Vector{Vector{Symbol}}, Vector{Tuple{Symbol,Vararg{Symbol}}}, Vector{Vector{Symbol}}, Tuple{Tuple{Symbol,Vararg{Symbol}}, Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}}, constants::constants)::Union{UnitRange{Int}, Vector{Int}}
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


function symmetrise_covariance_upper(covariance::AbstractMatrix{T}) where T <: Real
    covariance_upper = ℒ.triu(covariance)
    return covariance_upper + covariance_upper' - ℒ.Diagonal(ℒ.diag(covariance_upper))
end


function covariance_to_correlation(covariance::AbstractMatrix{T}) where T <: Real
    covariance_symmetric = symmetrise_covariance_upper(covariance)
    diag_covariance = convert(Vector{T}, ℒ.diag(covariance_symmetric))
    max_diag = maximum(d -> d > 0 ? d : zero(T), diag_covariance; init = zero(T))
    degenerate_tol = max(eps(T), eps(T) * max_diag)
    std_corr = Vector{T}(undef, length(diag_covariance))

    @inbounds for i in eachindex(diag_covariance)
        diag_entry = diag_covariance[i]
        std_corr[i] = diag_entry > degenerate_tol ? sqrt(diag_entry) : convert(T, NaN)
    end

    correlation = covariance_symmetric ./ (std_corr * std_corr')

    # Clamp machine-precision noise to zero for clean display output.
    # Skipped for AD element types (e.g. ForwardDiff.Dual) where
    # replacing with zero(T) would destroy derivative partials.
    if T <: AbstractFloat
        noise_tol = eps(T)^(T(2)/T(3))
        n = size(correlation, 1)
        @inbounds for j in 1:n, i in 1:n
            if i != j
                c = correlation[i, j]
                if !isnan(c) && abs(c) < noise_tol
                    correlation[i, j] = zero(T)
                end
            end
        end
    end

    return correlation, covariance_symmetric, diag_covariance, std_corr
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

@unstable function parse_shocks_input_to_index(shocks::Union{Symbol_input, String_input}, constants::constants)
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


noop_state_update(state::AbstractVector{<:Real}, ::AbstractVector{<:Real}) = state
noop_state_update(state::AbstractVector{<:AbstractVector{<:Real}}, ::AbstractVector{<:Real}) = state

function initialize_pruned_state(state::AbstractVector{T}, n_states::Int) where T <: Real
    return [Vector{T}(state), zeros(T, n_states)]
end

function initialize_pruned_state(state::AbstractVector{T}, n_states::Int, ::Val{3}) where T <: Real
    return [Vector{T}(state), zeros(T, n_states), zeros(T, n_states)]
end

function pruned_second_order_state_update(pruned_states::AbstractVector{<:AbstractVector{T}}, shock::AbstractVector{S}, past_idx, n_states::Int, 𝐒₁, 𝐒₂) where {T <: Real, S <: Real}
    aug_state₁ = [pruned_states[1][past_idx]; 1; shock]
    aug_state₂ = [pruned_states[2][past_idx]; 0; zero(shock)]
    return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * compressed_kron²_power(aug_state₁) / 2]
end

function pruned_second_order_state_update(state::AbstractVector{T}, shock::AbstractVector{S}, past_idx, n_states::Int, 𝐒₁, 𝐒₂) where {T <: Real, S <: Real}
    return pruned_second_order_state_update(initialize_pruned_state(state, n_states), shock, past_idx, n_states, 𝐒₁, 𝐒₂)
end

function pruned_third_order_state_update(pruned_states::AbstractVector{<:AbstractVector{T}}, shock::AbstractVector{S}, past_idx, n_states::Int, 𝐒₁, 𝐒₂, 𝐒₃) where {T <: Real, S <: Real}
    aug_state₁ = [pruned_states[1][past_idx]; 1; shock]
    aug_state₁̂ = [pruned_states[1][past_idx]; 0; shock]
    aug_state₂ = [pruned_states[2][past_idx]; 0; zero(shock)]
    aug_state₃ = [pruned_states[3][past_idx]; 0; zero(shock)]
    kron_aug_state₁ = compressed_kron²_power(aug_state₁)
    return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * kron_aug_state₁ / 2, 𝐒₁ * aug_state₃ + 𝐒₂ * compressed_kron²(aug_state₁̂, aug_state₂) + 𝐒₃ * compressed_kron³_power(aug_state₁) / 6]
end

function pruned_third_order_state_update(state::AbstractVector{T}, shock::AbstractVector{S}, past_idx, n_states::Int, 𝐒₁, 𝐒₂, 𝐒₃) where {T <: Real, S <: Real}
    return pruned_third_order_state_update(initialize_pruned_state(state, n_states, Val(3)), shock, past_idx, n_states, 𝐒₁, 𝐒₂, 𝐒₃)
end

@unstable function parse_algorithm_to_state_update(algorithm::Symbol, 𝓂::ℳ, occasionally_binding_constraints::Bool)::Tuple{Function, Bool}
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
            𝐒₂ = 𝓂.caches.second_order_solution
            Ŝ₁̂ = [Ŝ₁[:,1:nPast] zeros(nVars) Ŝ₁[:,nPast+1:end]]

            if algorithm == :second_order
                state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[past_idx]; 1; shock]
                return Ŝ₁̂ * aug_state + 𝐒₂ * compressed_kron²_power(aug_state) / 2
                end
            else  # :third_order
                𝐒₃ = 𝓂.caches.third_order_solution
                state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[past_idx]; 1; shock]
                    return Ŝ₁̂ * aug_state + 𝐒₂ * compressed_kron²_power(aug_state) / 2 + 𝐒₃ * compressed_kron³_power(aug_state) / 6
                end
            end
        elseif algorithm == :pruned_second_order
            𝐒₂ = 𝓂.caches.second_order_solution
            Ŝ₁̂ = [Ŝ₁[:,1:nPast] zeros(nVars) Ŝ₁[:,nPast+1:end]]
            state_update = (state, shock) -> pruned_second_order_state_update(state, shock, past_idx, nVars, Ŝ₁̂, 𝐒₂)
        elseif algorithm == :pruned_third_order
            𝐒₂ = 𝓂.caches.second_order_solution
            𝐒₃ = 𝓂.caches.third_order_solution
            Ŝ₁̂ = [Ŝ₁[:,1:nPast] zeros(nVars) Ŝ₁[:,nPast+1:end]]
            state_update = (state, shock) -> pruned_third_order_state_update(state, shock, past_idx, nVars, Ŝ₁̂, 𝐒₂, 𝐒₃)
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
            𝐒₂ = 𝓂.caches.second_order_solution

            if algorithm == :second_order
                state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[past_idx]; 1; shock]
                return 𝐒₁ * aug_state + 𝐒₂ * compressed_kron²_power(aug_state) / 2
                end
            else  # :third_order
                𝐒₃ = 𝓂.caches.third_order_solution
                state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S}
                    aug_state = [state[past_idx]; 1; shock]
                    return 𝐒₁ * aug_state + 𝐒₂ * compressed_kron²_power(aug_state) / 2 + 𝐒₃ * compressed_kron³_power(aug_state) / 6
                end
            end
        elseif algorithm == :pruned_second_order
            S₁ = 𝓂.caches.first_order_solution_matrix
            𝐒₁ = [S₁[:,1:nPast] zeros(nVars) S₁[:,nPast+1:end]]
            𝐒₂ = 𝓂.caches.second_order_solution
            state_update = (state, shock) -> pruned_second_order_state_update(state, shock, past_idx, nVars, 𝐒₁, 𝐒₂)
        elseif algorithm == :pruned_third_order
            S₁ = 𝓂.caches.first_order_solution_matrix
            𝐒₁ = [S₁[:,1:nPast] zeros(nVars) S₁[:,nPast+1:end]]
            𝐒₂ = 𝓂.caches.second_order_solution
            𝐒₃ = 𝓂.caches.third_order_solution
            state_update = (state, shock) -> pruned_third_order_state_update(state, shock, past_idx, nVars, 𝐒₁, 𝐒₂, 𝐒₃)
        end
    end

    return (state_update, pruning)
end



function get_custom_steady_state_workspace!(𝓂::ℳ, expected_length::Int)
    buffer = 𝓂.workspaces.custom_steady_state

    if length(buffer) != expected_length
        buffer = Vector{Float64}(undef, expected_length)
        𝓂.workspaces.custom_steady_state = buffer
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
        get_custom_steady_state_workspace!(𝓂, expected_length)
        
        output = Vector{S}(undef, expected_length)
        try 
            𝓂.functions.NSSS_custom(output, parameter_values)
        catch
            fill!(output, S(NaN))
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

# Dead code: find_variables_to_exclude — never called anywhere
# function find_variables_to_exclude(𝓂::ℳ, observables::Vector{Symbol})
#     # reduce system
#     vars_to_exclude = setdiff(𝓂.constants.post_model_macro.present_only, observables)
#
#     # Mapping variables to their equation index
#     variable_to_equation = Dict{Symbol, Vector{Int}}()
#     for var in vars_to_exclude
#         for (eq_idx, vars_set) in enumerate(𝓂.constants.post_model_macro.dyn_var_present_list)
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
#
#     return variable_to_equation
# end


# Dead code: create_broadcaster — never called anywhere
# function create_broadcaster(indices::Vector{Int}, n::Int)
#     broadcaster = spzeros(n, length(indices))
#     for (i, vid) in enumerate(indices)
#         broadcaster[vid,i] = 1.0
#     end
#     return broadcaster
# end

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
    return nothing
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
    return nothing
end

function get_NSSS_and_parameters(𝓂::ℳ, 
                                    parameter_values::Vector{S}; 
                                    opts::CalculationOptions = merge_calculation_options(),
                                    cold_start::Bool = false,
                                    estimation::Bool = false,
                                    caching::Bool = true)::Tuple{Vector{S}, Tuple{S, Int}} where S <: Real
                                    # timer::TimerOutput = TimerOutput(),

    # @timeit_debug timer "Calculate NSSS" begin
    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    
    # Cache hit: return cached NSSS if valid for current parameters
    if caching && S === Float64 && cache_valid_for_parameters(𝓂.caches.valid_for.non_stochastic_steady_state, parameter_values) && !isempty(𝓂.caches.non_stochastic_steady_state)
        return (copy(𝓂.caches.non_stochastic_steady_state), (zero(S), 0))::Tuple{Vector{S}, Tuple{S, Int}}
    end

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

        residual = 𝓂.workspaces.nsss_solver.check_residual
        fill!(residual, 0.0)
        
        𝓂.functions.NSSS_check(residual, parameter_values, SS_and_pars_tmp)
        
        solution_error = ℒ.norm(residual)

        iters = 0

        # if !isfinite(solution_error) || solution_error > opts.tol.nsss.acceptance_tol
        #     throw(ArgumentError("Custom steady state function failed steady state check: residual $solution_error > $(opts.tol.nsss.acceptance_tol). Parameters: $(parameter_values). Steady state and parameters returned: $(SS_and_pars_tmp)."))
        # end
        X = ms.custom_ss_expand_matrix
        SS_and_pars = X * SS_and_pars_tmp
    else
        fastest_idx = 𝓂.constants.post_complete_parameters.nsss_fastest_solver_parameter_idx
        preferred_solver_parameter_idx = fastest_idx < 1 || fastest_idx > length(DEFAULT_SOLVER_PARAMETERS) ? 1 : fastest_idx
        SS_and_pars, (solution_error, iters) = solve_nsss_wrapper(parameter_values, 𝓂, opts.tol, opts.verbose, cold_start, DEFAULT_SOLVER_PARAMETERS, preferred_solver_parameter_idx = preferred_solver_parameter_idx)
    end

    # Update counters
    solved = !(solution_error > opts.tol.nsss.acceptance_tol || isnan(solution_error))
    update_ss_counter!(𝓂.counters, solved, estimation = estimation)
    
    if !solved
        if opts.verbose 
            println("Failed to find NSSS") 
        end
        # return (SS_and_pars, (10.0, iters))#, x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    # end # timeit_debug

    # Cache write: store NSSS result and stamp
    if caching
        cache_ss = 𝓂.caches.non_stochastic_steady_state
        if length(cache_ss) != length(SS_and_pars)
            resize!(cache_ss, length(SS_and_pars))
        end
        copyto!(cache_ss, SS_and_pars)
        if solved
            𝓂.caches.valid_for.non_stochastic_steady_state = Float64.(primal.(parameter_values))
        else
            𝓂.caches.valid_for.non_stochastic_steady_state = Float64[]
        end
    end

    return SS_and_pars, (solution_error, iters)
end




end # dispatch_doctor

@setup_workload begin
    @compile_workload begin
        @model RBC_for_precompile precompile = true begin
            1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
            z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
        end

        @parameters RBC_for_precompile silent = true precompile = true begin
            δ = 0.02
            α = 0.5
        end

        # Warm the standard first-order workflow used by a simple IRF script.
        get_steady_state(
            RBC_for_precompile;
            # derivatives = false,
            stochastic = false,
            return_variables_only = true,
            silent = true,
        )
        get_irf(
            RBC_for_precompile;
            algorithm = :first_order,
            periods = 40,
            variables = :all,
            shocks = :all,
            verbose = false,
        )
        get_moments(
            RBC_for_precompile;
            algorithm = :first_order,
            variables = :all,
            non_stochastic_steady_state = true,
            mean = true,
            standard_deviation = true,
            variance = true,
            covariance = true,
            correlation = true,
            # derivatives = false,
            silent = true,
            verbose = false,
        )
    end
end
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

# ForwardDiff Dual specializations moved to ext/ForwardDiffExt.jl

# Include rrule definitions for reverse-mode AD (Zygote/ChainRulesCore)
# Must be at the end of the module because rrules depend on function definitions
include("./rrules.jl")

end
