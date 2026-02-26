# Third-Order Stochastic Steady State rrule Implementation

## Summary

Implemented the outer rrule for `calculate_third_order_stochastic_steady_state(parameters::Vector{M}, 𝓂::ℳ; ...)` that manually chains pullbacks from component functions without using automatic differentiation internally.

## Implementation Details

### Forward Pass

The rrule captures pullback functions from each component during the forward pass:

1. **Steady State**: `rrule(get_NSSS_and_parameters, ...)` → stores `NSSS_back`
2. **Jacobian**: `rrule(calculate_jacobian, ...)` → stores `∇₁_back`
3. **First-order Solution**: `rrule(calculate_first_order_solution, ...)` → stores `S1_back`
4. **Hessian**: `rrule(calculate_hessian, ...)` → stores `∇₂_back`
5. **Second-order Solution**: `rrule(calculate_second_order_solution, ...)` → stores `S2_back`
6. **Third-order Derivatives**: `rrule(calculate_third_order_derivatives, ...)` → stores `∇₃_back`
7. **Third-order Solution**: `rrule(calculate_third_order_solution, ...)` → stores `S3_back`
8. **Inner SSS Solver**: `rrule(calculate_third_order_stochastic_steady_state, Val(:newton), ...)` → stores `SSS_back`

### Backward Pass

The pullback function chains gradients in reverse order:

1. Start with gradient w.r.t. output stochastic steady state: `∂sss`
2. Backpropagate through inner SSS solver (if not pruning)
3. Backpropagate through matrix transformations (S2*U2, S3*U3, sparse operations)
4. Backpropagate through third-order solution → get `∂∇₃, ∂∇₂, ∂∇₁, ∂𝐒₂, ∂𝐒₁`
5. Backpropagate through second-order solution → add to `∂∇₂, ∂∇₁, ∂𝐒₁`
6. Backpropagate through first-order solution → add to `∂∇₁`
7. Backpropagate through derivative calculations → get `∂parameters` and `∂SS_and_pars`
8. Backpropagate through steady state calculation → add to `∂parameters`
9. Accumulate all `∂parameters` contributions

### Key Features

- **No AD inside rrule**: Manually chains pullbacks, doesn't use `Zygote.gradient` or similar
- **Handles failures gracefully**: Returns zero gradients when NSSS or solutions fail
- **Pruning support**: Simplified backprop when pruning = true
- **Type stability**: Maintains type information throughout
- **Gradient accumulation**: Properly combines gradients from multiple sources

### Comparison with Inner rrule

The inner rrule (lines 214-283) handles the iterative Newton solver for the stochastic steady state equation:
- Uses implicit differentiation
- Solves linear system for sensitivity
- Only differentiates w.r.t. 𝐒₁, 𝐒₂, 𝐒₃

The outer rrule (lines 295-502) handles the full computation:
- Chains through NSSS calculation
- Chains through all perturbation order solutions
- Differentiates w.r.t. model parameters

## Testing

Test script: `/tmp/test_third_order_gradient.jl`
- Uses Caldara et al (2012) model
- Computes gradient of loglikelihood w.r.t. parameters
- Verifies gradient is finite

Expected behavior:
- Gradient computation should succeed without errors
- All gradient values should be finite
- Results should match automatic differentiation through component rrules

## Files Modified

- `src/custom_autodiff_rules/zygote.jl`: Added outer rrule (217 lines)
