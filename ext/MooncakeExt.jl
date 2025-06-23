module MooncakeExt

import Mooncake
import MacroModelling
import SparseArrays
import LinearAlgebra as ℒ

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.mul_reverse_AD!), Matrix{S}, AbstractMatrix{M}, AbstractMatrix{N}}  where {S <: Real, M <: Real, N <: Real}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.sparse_preallocated!), Matrix{T}} where {T <: Real} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_second_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, SparseArrays.AbstractSparseMatrix{Float64}, Vector{Float64}, MacroModelling.ℳ} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_third_order_stochastic_steady_state), Val{:newton}, Matrix{Float64}, SparseArrays.AbstractSparseMatrix{Float64}, SparseArrays.AbstractSparseMatrix{Float64}, Vector{Float64}, MacroModelling.ℳ} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_jacobian), Vector{M}, Vector{N}, MacroModelling.ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_hessian), Vector{M}, Vector{N}, MacroModelling.ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_third_order_derivatives), Vector{M}, Vector{N}, MacroModelling.ℳ} where {M <: AbstractFloat, N <: AbstractFloat}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.get_NSSS_and_parameters), MacroModelling.ℳ, Vector{S} where S <: AbstractFloat} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_first_order_solution), Matrix{R}} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_second_order_solution), AbstractMatrix{R}, SparseArrays.SparseMatrixCSC{R}, AbstractMatrix{R}, MacroModelling.second_order_auxilliary_matrices, MacroModelling.caches} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_second_order_solution), AbstractMatrix{R}, SparseArrays.SparseMatrixCSC{R}, SparseArrays.SparseMatrixCSC{R}, AbstractMatrix{R}, SparseArrays.SparseMatrixCSC{R}, MacroModelling.second_order_auxilliary_matrices, MacroModelling.third_order_auxilliary_matrices, MacroModelling.caches} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.solve_lyapunov_equation), AbstractMatrix{R}, AbstractMatrix{R}} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.solve_sylvester_equation), AbstractMatrix{R}, AbstractMatrix{R}, AbstractMatrix{R}} where R <: AbstractFloat true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.find_shocks), Val{:LagrangeNewton}, Vector{Float64}, Vector{Float64}, AbstractMatrix{Float64}, ℒ.Diagonal{Bool, Vector{Bool}}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, Vector{Float64}} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.find_shocks), Val{:LagrangeNewton}, Vector{Float64}, Vector{Float64}, Vector{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, 
ℒ.Diagonal{Bool, Vector{Bool}}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, Vector{Float64}} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:first_order}, Vector{Vector{Float64}}, Matrix{Float64}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:pruned_second_order},Vector{Vector{Float64}}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:second_order},Vector{Float64}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:pruned_third_order},Vector{Vector{Float64}}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(MacroModelling.calculate_inversion_filter_loglikelihood), Val{:third_order},Vector{Float64}, Vector{AbstractMatrix{Float64}}, Matrix{Float64}, Union{Vector{String}, Vector{Symbol}}, MacroModelling.timings} true

end # module
