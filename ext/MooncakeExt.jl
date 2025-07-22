module MooncakeExt

import Mooncake
import MacroModelling
import SparseArrays
import LinearAlgebra as ℒ
import ChainRulesCore

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, typeof(MacroModelling.mul_reverse_AD!), Matrix{S}, AbstractMatrix{M}, AbstractMatrix{N}} where {S <: Real, M <: Real, N <: Real}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, MacroModelling.higher_order_caches{S,F}, typeof(MacroModelling.sparse_preallocated!), Matrix{S}} where {S <: Real, F <: AbstractFloat} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, AbstractFloat, typeof(MacroModelling.calculate_second_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, SparseArrays.AbstractSparseMatrix{Float64}, Vector{Float64}, MacroModelling.ℳ} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, AbstractFloat, typeof(MacroModelling.calculate_third_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, SparseArrays.AbstractSparseMatrix{Float64}, SparseArrays.AbstractSparseMatrix{Float64}, Vector{Float64}, MacroModelling.ℳ} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, MacroModelling.CalculationOptions, typeof(MacroModelling.get_NSSS_and_parameters), MacroModelling.ℳ, Vector{S}} where S <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, typeof(MacroModelling.calculate_jacobian), Vector{M}, Vector{N}, MacroModelling.ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, typeof(MacroModelling.calculate_hessian), Vector{M}, Vector{N}, MacroModelling.ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, typeof(MacroModelling.calculate_third_order_derivatives), Vector{M}, Vector{N}, MacroModelling.ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, MacroModelling.timings, MacroModelling.CalculationOptions, AbstractMatrix{R}, typeof(MacroModelling.calculate_first_order_solution), Matrix{R}} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, MacroModelling.timings, AbstractMatrix{R}, MacroModelling.CalculationOptions, typeof(MacroModelling.calculate_second_order_solution), AbstractMatrix{R}, SparseArrays.SparseMatrixCSC{R}, AbstractMatrix{R}, MacroModelling.second_order_auxiliary_matrices, MacroModelling.caches} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, MacroModelling.timings, AbstractMatrix{R}, MacroModelling.CalculationOptions, typeof(MacroModelling.calculate_third_order_solution), AbstractMatrix{R}, SparseArrays.SparseMatrixCSC{R}, SparseArrays.SparseMatrixCSC{R}, AbstractMatrix{R}, SparseArrays.SparseMatrixCSC{R}, MacroModelling.second_order_auxiliary_matrices, MacroModelling.third_order_auxiliary_matrices, MacroModelling.caches} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, Int, Float64, typeof(MacroModelling.find_shocks), Val{:LagrangeNewton}, Vector{Float64}, Vector{Float64}, AbstractMatrix{Float64}, ℒ.Diagonal{Bool, Vector{Bool}}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, Vector{Float64}} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, Int, Float64, typeof(MacroModelling.find_shocks), Val{:LagrangeNewton}, Vector{Float64}, Vector{Float64}, Vector{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, ℒ.Diagonal{Bool, Vector{Bool}}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, Vector{Float64}} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, Int, Int, MacroModelling.CalculationOptions, Symbol, typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:first_order}, Vector{Vector{Float64}}, Matrix{Float64}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, Int, Int, MacroModelling.CalculationOptions, Symbol, typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:pruned_second_order}, Vector{Vector{Float64}}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, Int, Int, MacroModelling.CalculationOptions, Symbol, typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:second_order}, Vector{Float64}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, Int, Int, MacroModelling.CalculationOptions, Symbol, typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:pruned_third_order}, Vector{Vector{Float64}}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, Int, Int, MacroModelling.CalculationOptions, Symbol, typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:third_order}, Vector{Float64}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, Symbol, AbstractFloat, AbstractFloat, Bool, typeof(MacroModelling.solve_lyapunov_equation), AbstractMatrix{R}, AbstractMatrix{R}} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{Any, AbstractMatrix{<:AbstractFloat}, Symbol, AbstractFloat, AbstractFloat, MacroModelling.sylvester_caches, Bool,
    ::typeof(MacroModelling.solve_sylvester_equation), AbstractMatrix{R}, AbstractMatrix{R}, AbstractMatrix{R}} where R <: AbstractFloat true


function ChainRulesCore.rrule(func_ir::Any,
    ℂ::MacroModelling.higher_order_caches{T,F},
    ::typeof(MacroModelling.sparse_preallocated!),
    Ŝ::Matrix{T}) where {T<:Real,F<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.sparse_preallocated!, Ŝ; ℂ=ℂ)
end

function ChainRulesCore.rrule(func_ir::Any,
    ::typeof(MacroModelling.mul_reverse_AD!),
    C::Matrix{S},
    A::AbstractMatrix{M},
    B::AbstractMatrix{N}) where {S<:Real,M<:Real,N<:Real}
    ChainRulesCore.rrule(MacroModelling.mul_reverse_AD!, C, A, B)
end

function ChainRulesCore.rrule(func_ir::Any,
    ::typeof(MacroModelling.calculate_jacobian),
    parameters::Vector{M},
    SS_and_pars::Vector{N},
    m::MacroModelling.ℳ) where {M<:AbstractFloat,N<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.calculate_jacobian, parameters, SS_and_pars, m)
end

function ChainRulesCore.rrule(func_ir::Any,
    ::typeof(MacroModelling.calculate_hessian),
    parameters::Vector{M},
    SS_and_pars::Vector{N},
    m::MacroModelling.ℳ) where {M<:AbstractFloat,N<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.calculate_hessian, parameters, SS_and_pars, m)
end

function ChainRulesCore.rrule(func_ir::Any,
    ::typeof(MacroModelling.calculate_third_order_derivatives),
    parameters::Vector{M},
    SS_and_pars::Vector{N},
    m::MacroModelling.ℳ) where {M<:AbstractFloat,N<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.calculate_third_order_derivatives, parameters, SS_and_pars, m)
end

function ChainRulesCore.rrule(func_ir::Any,
    tol::AbstractFloat,
    ::typeof(MacroModelling.calculate_second_order_stochastic_steady_state),
    ::Val{:newton},
    S1::Matrix{Float64},
    S2::SparseArrays.AbstractSparseMatrix{Float64},
    x::Vector{Float64},
    m::MacroModelling.ℳ)
    ChainRulesCore.rrule(MacroModelling.calculate_second_order_stochastic_steady_state,
        Val(:newton), S1, S2, x, m; tol=tol)
end

function ChainRulesCore.rrule(func_ir::Any,
    tol::AbstractFloat,
    ::typeof(MacroModelling.calculate_third_order_stochastic_steady_state),
    ::Val{:newton},
    S1::Matrix{Float64},
    S2::SparseArrays.AbstractSparseMatrix{Float64},
    S3::SparseArrays.AbstractSparseMatrix{Float64},
    x::Vector{Float64},
    m::MacroModelling.ℳ)
    ChainRulesCore.rrule(MacroModelling.calculate_third_order_stochastic_steady_state,
        Val(:newton), S1, S2, S3, x, m; tol=tol)
end

function ChainRulesCore.rrule(func_ir::Any,
    opts::MacroModelling.CalculationOptions,
    ::typeof(MacroModelling.get_NSSS_and_parameters),
    m::MacroModelling.ℳ,
    x::Vector{S}) where {S<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.get_NSSS_and_parameters, m, x; opts=opts)
end

function ChainRulesCore.rrule(func_ir::Any,
    T::MacroModelling.timings,
    opts::MacroModelling.CalculationOptions,
    initial_guess::AbstractMatrix{R},
    ::typeof(MacroModelling.calculate_first_order_solution),
    ∇₁::Matrix{R}) where {R<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.calculate_first_order_solution,
        ∇₁; T=T, opts=opts, initial_guess=initial_guess)
end

function ChainRulesCore.rrule(func_ir::Any,
    T::MacroModelling.timings,
    initial_guess::AbstractMatrix{R},
    opts::MacroModelling.CalculationOptions,
    ::typeof(MacroModelling.calculate_second_order_solution),
    ∇₁::AbstractMatrix{R},
    ∇₂::SparseArrays.SparseMatrixCSC{R},
    𝑺₁::AbstractMatrix{R},
    M₂::MacroModelling.second_order_auxiliary_matrices,
    ℂC::MacroModelling.caches) where {R<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.calculate_second_order_solution,
        ∇₁, ∇₂, 𝑺₁, M₂, ℂC; T=T, initial_guess=initial_guess, opts=opts)
end

function ChainRulesCore.rrule(func_ir::Any,
    T::MacroModelling.timings,
    initial_guess::AbstractMatrix{R},
    opts::MacroModelling.CalculationOptions,
    ::typeof(MacroModelling.calculate_third_order_solution),
    ∇₁::AbstractMatrix{R},
    ∇₂::SparseArrays.SparseMatrixCSC{R},
    ∇₃::SparseArrays.SparseMatrixCSC{R},
    𝑺₁::AbstractMatrix{R},
    𝐒₂::SparseArrays.SparseMatrixCSC{R},
    M₂::MacroModelling.second_order_auxiliary_matrices,
    M₃::MacroModelling.third_order_auxiliary_matrices,
    ℂC::MacroModelling.caches) where {R<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.calculate_third_order_solution,
        ∇₁, ∇₂, ∇₃, 𝑺₁, 𝐒₂, M₂, M₃, ℂC;
        T=T, initial_guess=initial_guess, opts=opts)
end

function ChainRulesCore.rrule(func_ir::Any,
    alg::Symbol,
    tol::AbstractFloat,
    acc_tol::AbstractFloat,
    verbose::Bool,
    ::typeof(MacroModelling.solve_lyapunov_equation),
    A::AbstractMatrix{R},
    C::AbstractMatrix{R}) where {R<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.solve_lyapunov_equation,
        A, C;
        lyapunov_algorithm=alg,
        tol=tol,
        acceptance_tol=acc_tol,
        verbose=verbose)
end

function ChainRulesCore.rrule(func_ir::Any,
    initial_guess::AbstractMatrix{<:AbstractFloat},
    syl_alg::Symbol,
    acc_tol::AbstractFloat,
    tol::AbstractFloat,
    𝕊ℂ::MacroModelling.sylvester_caches,
    verbose::Bool,
    ::typeof(MacroModelling.solve_sylvester_equation),
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    C::AbstractMatrix{R}) where {R<:AbstractFloat}
    ChainRulesCore.rrule(MacroModelling.solve_sylvester_equation,
        A, B, C;
        initial_guess=initial_guess,
        sylvester_algorithm=syl_alg,
        acceptance_tol=acc_tol,
        tol=tol,
        𝕊ℂ=𝕊ℂ,
        verbose=verbose)
end

function ChainRulesCore.rrule(func_ir::Any,
    max_iter::Int,
    tol::Float64,
    ::typeof(MacroModelling.find_shocks),
    ::Val{:LagrangeNewton},
    initial_guess::Vector{Float64},
    kron_buffer::Vector{Float64},
    kron_buffer2::AbstractMatrix{Float64},
    J::ℒ.Diagonal{Bool,Vector{Bool}},
    S_i::AbstractMatrix{Float64},
    S_i2e::AbstractMatrix{Float64},
    shock_independent::Vector{Float64})
    ChainRulesCore.rrule(MacroModelling.find_shocks,
        Val(:LagrangeNewton),
        initial_guess,
        kron_buffer,
        kron_buffer2,
        J,
        S_i,
        S_i2e,
        shock_independent;
        max_iter=max_iter,
        tol=tol)
end

function ChainRulesCore.rrule(func_ir::Any,
    max_iter::Int,
    tol::Float64,
    ::typeof(MacroModelling.find_shocks),
    ::Val{:LagrangeNewton},
    initial_guess::Vector{Float64},
    kron_buffer::Vector{Float64},
    kron_buffer2::Vector{Float64},
    kron_buffer3::AbstractMatrix{Float64},
    kron_buffer4::AbstractMatrix{Float64},
    kron_buffer5::AbstractMatrix{Float64},
    J::ℒ.Diagonal{Bool,Vector{Bool}},
    S_i::AbstractMatrix{Float64},
    S_i2e::AbstractMatrix{Float64},
    S_i3e::AbstractMatrix{Float64},
    shock_independent::Vector{Float64})
    ChainRulesCore.rrule(MacroModelling.find_shocks,
        Val(:LagrangeNewton),
        initial_guess,
        kron_buffer,
        kron_buffer2,
        kron_buffer3,
        kron_buffer4,
        kron_buffer5,
        J,
        S_i,
        S_i2e,
        S_i3e,
        shock_independent;
        max_iter=max_iter,
        tol=tol)
end

function ChainRulesCore.rrule(func_ir::Any,
    warm_iters::Int,
    presample::Int,
    opts::MacroModelling.CalculationOptions,
    filt_alg::Symbol,
    ::typeof(MacroModelling.calculate_inversion_filter_loglikelihood),
    alg::Val{A},
    state,
    S,
    data,
    observables,
    T::MacroModelling.timings) where A
    ChainRulesCore.rrule(MacroModelling.calculate_inversion_filter_loglikelihood,
        alg, state, S, data, observables, T;
        warmup_iterations=warm_iters,
        presample_periods=presample,
        opts=opts,
        filter_algorithm=filt_alg)
end

end # module
