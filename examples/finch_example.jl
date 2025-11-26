"""
Example script demonstrating the usage of Finch-based higher order solution functions.

This script shows how the new Finch-based functions can be integrated into a model solution workflow.
"""

# Note: This is a conceptual example showing how to use the new functions.
# In practice, these would be called internally by MacroModelling.jl

using MacroModelling

# Define a simple RBC model
@model RBC_Finch_Example begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC_Finch_Example begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# Standard usage - this will use the default implementation
println("Solving model with standard implementation...")
sol_standard = get_solution(RBC_Finch_Example, algorithm = :second_order)

println("First order solution matrix size: ", size(sol_standard[2]))
println("Second order solution matrix size: ", size(sol_standard[3]))

# The Finch-based functions are now available in the package and can be used
# when implementing custom solution algorithms or benchmarking

println("\nFinch-based functions are now available:")
println("  - calculate_second_order_solution_finch")
println("  - calculate_third_order_solution_finch")
println("  - mat_mult_kron_finch")
println("  - compressed_kron³_finch")

# Future integration example (pseudo-code):
# To integrate these into the main solution workflow, you would:
# 1. Add an option to select the implementation method
# 2. Dispatch to the appropriate function based on user preference
#
# Example:
# @parameters RBC_Finch_Example begin
#     ...
#     solution_method = :finch  # or :standard
# end
#
# Then in the solve! function:
# if opts.solution_method == :finch
#     𝐒₂, solved = calculate_second_order_solution_finch(...)
# else
#     𝐒₂, solved = calculate_second_order_solution(...)
# end

println("\n✓ Example completed successfully!")
println("\nThe Finch-based implementations provide an alternative way to compute")
println("higher-order perturbation solutions that may be more efficient for")
println("certain problem sizes and sparsity patterns.")
