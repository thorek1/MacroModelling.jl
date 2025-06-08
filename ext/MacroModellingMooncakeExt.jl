module MacroModellingMooncakeExt

using MacroModelling
import Mooncake
import Mooncake: @from_rrule, DefaultCtx

@from_rrule DefaultCtx Tuple{typeof(calculate_first_order_solution), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(calculate_second_order_solution), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(calculate_third_order_solution), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(solve_sylvester_equation), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(solve_lyapunov_equation), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(find_shocks), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(calculate_inversion_filter_loglikelihood), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(run_kalman_iterations), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(mul_reverse_AD!), Vararg{Any}} false
@from_rrule DefaultCtx Tuple{typeof(sparse_preallocated!), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(calculate_second_order_stochastic_steady_state), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(calculate_third_order_stochastic_steady_state), Vararg{Any}} true
@from_rrule DefaultCtx Tuple{typeof(calculate_jacobian), Vararg{Any}} false
@from_rrule DefaultCtx Tuple{typeof(calculate_hessian), Vararg{Any}} false
@from_rrule DefaultCtx Tuple{typeof(calculate_third_order_derivatives), Vararg{Any}} false
@from_rrule DefaultCtx Tuple{typeof(get_NSSS_and_parameters), Vararg{Any}} true

end
