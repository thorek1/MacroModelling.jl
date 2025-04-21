module MooncakeExt

import Mooncake: @from_rrule, DefaultCtx
import MacroModelling: mul_reverse_AD!, sparse_preallocated!, calculate_second_order_stochastic_steady_state, calculate_third_order_stochastic_steady_state, calculate_jacobian, calculate_hessian, calculate_third_order_derivatives, get_NSSS_and_parameters, calculate_first_order_solution, calculate_second_order_solution, solve_lyapunov_equation, solve_sylvester_equation, find_shocks, calculate_inversion_filter_loglikelihood, ℳ, second_order_auxilliary_matrices, third_order_auxilliary_matrices, caches, timings
import SparseArrays: SparseMatrixCSC, SparseVector, AbstractSparseArray, AbstractSparseMatrix
import LinearAlgebra as ℒ

@from_rrule DefaultCtx Tuple{typeof(mul_reverse_AD!), Matrix{S}, AbstractMatrix{M}, AbstractMatrix{N}}  where {S <: Real, M <: Real, N <: Real}

@from_rrule DefaultCtx Tuple{typeof(sparse_preallocated!), Matrix{T}} where {T <: Real} true

@from_rrule DefaultCtx Tuple{typeof(calculate_second_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, AbstractSparseMatrix{Float64}, Vector{Float64}, ℳ} true

@from_rrule DefaultCtx Tuple{typeof(calculate_third_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, AbstractSparseMatrix{Float64}, AbstractSparseMatrix{Float64}, Vector{Float64}, ℳ} true

@from_rrule DefaultCtx Tuple{typeof(calculate_jacobian), Vector{M}, Vector{N}, ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

@from_rrule DefaultCtx Tuple{typeof(calculate_hessian), Vector{M}, Vector{N}, ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

@from_rrule DefaultCtx Tuple{typeof(calculate_third_order_derivatives), Vector{M}, Vector{N}, ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

@from_rrule DefaultCtx Tuple{typeof(get_NSSS_and_parameters), ℳ, Vector{Float64}} true

@from_rrule DefaultCtx Tuple{typeof(calculate_first_order_solution), Matrix{R}} where R <: AbstractFloat true

@from_rrule DefaultCtx Tuple{typeof(calculate_second_order_solution), AbstractMatrix{R}, SparseMatrixCSC{R}, AbstractMatrix{R}, second_order_auxilliary_matrices, caches} where R <: AbstractFloat true

@from_rrule DefaultCtx Tuple{typeof(calculate_second_order_solution), AbstractMatrix{R}, SparseMatrixCSC{R}, SparseMatrixCSC{R}, AbstractMatrix{R}, SparseMatrixCSC{R}, second_order_auxilliary_matrices, third_order_auxilliary_matrices, caches} where R <: AbstractFloat true

@from_rrule DefaultCtx Tuple{typeof(solve_lyapunov_equation), AbstractMatrix{R}, AbstractMatrix{R}} where R <: AbstractFloat true

@from_rrule DefaultCtx Tuple{typeof(solve_sylvester_equation), AbstractMatrix{R}, AbstractMatrix{R}, AbstractMatrix{R}} where R <: AbstractFloat true

@from_rrule DefaultCtx Tuple{typeof(find_shocks), Val{:LagrangeNewton}, Vector{Float64}, Vector{Float64}, AbstractMatrix{Float64}, ℒ.Diagonal{Bool, Vector{Bool}}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, Vector{Float64}} true

@from_rrule DefaultCtx Tuple{typeof(find_shocks), Val{:LagrangeNewton}, Vector{Float64}, Vector{Float64}, Vector{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, 
ℒ.Diagonal{Bool, Vector{Bool}}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, Vector{Float64}} true

@from_rrule DefaultCtx Tuple{typeof(calculate_inversion_filter_loglikelihood), Val{:first_order}, Vector{Vector{Float64}}, Matrix{Float64}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, timings} true

@from_rrule DefaultCtx Tuple{typeof(calculate_inversion_filter_loglikelihood), Val{:pruned_second_order},Vector{Vector{Float64}}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, timings} true

@from_rrule DefaultCtx Tuple{typeof(calculate_inversion_filter_loglikelihood), Val{:second_order},Vector{Float64}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, timings} true

@from_rrule DefaultCtx Tuple{typeof(calculate_inversion_filter_loglikelihood), Val{:pruned_third_order},Vector{Vector{Float64}}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, timings} true

@from_rrule DefaultCtx Tuple{typeof(calculate_inversion_filter_loglikelihood), Val{:third_order},Vector{Float64}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, timings} true

end # module