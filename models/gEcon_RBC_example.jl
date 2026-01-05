# Example: Converting a gEcon model to MacroModelling.jl
# This file demonstrates how to use the derive_focs function to convert
# an optimization-based model specification to the standard FOC form.

# The original gEcon model is:
#
# block CONSUMER {
#     definitions { u[] = (C[]^mu * (1 - L_s[])^(1 - mu))^(1 - eta) / (1 - eta); };
#     controls { K_s[], C[], L_s[], I[]; };
#     objective { U[] = u[] + beta * E[][U[1]]; };
#     constraints {
#         I[] + C[] = pi[] + r[] * K_s[-1] + W[] * L_s[];
#         K_s[] = (1 - delta) * K_s[-1] + I[];
#     };
# };
#
# block FIRM {
#     controls { K_d[], L_d[], Y[]; };
#     objective { pi[] = Y[] - L_d[] * W[] - r[] * K_d[]; };
#     constraints { Y[] = Z[] * K_d[]^alpha * L_d[]^(1 - alpha); };
# };
#
# block EQUILIBRIUM {
#     identities { K_d[] = K_s[-1]; L_d[] = L_s[]; };
# };
#
# block EXOG {
#     identities { Z[] = exp(phi * log(Z[-1]) + epsilon_Z[]); };
#     shocks { epsilon_Z[]; };
# };

using MacroModelling

println("=== Converting gEcon RBC Model to MacroModelling.jl ===\n")

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Use derive_focs to derive FOCs from optimization problems
# ──────────────────────────────────────────────────────────────────────────────

println("--- Step 1: Deriving Consumer FOCs ---")

# For the consumer block, we use log utility for simplicity
# (The CES utility in gEcon would require more complex substitutions)
# u[0] = μ * log(C[0]) + (1 - μ) * log(1 - L_s[0])

consumer_controls = [:K_s, :C, :L_s, :I]
consumer_objective = :(U[0] = μ * log(C[0]) + (1 - μ) * log(1 - L_s[0]) + β * U[1])
consumer_constraints = [
    :(I[0] + C[0] = π[0] + r[0] * K_s[-1] + W[0] * L_s[0]),
    :(K_s[0] = (1 - δ) * K_s[-1] + I[0])
]

consumer_focs, consumer_mults = derive_focs(
    controls = consumer_controls,
    objective = consumer_objective,
    constraints = consumer_constraints,
    discount_factor = :β,
    block_name = "c"  # Short name for cleaner multiplier names
)

println("Consumer FOCs derived:")
for (i, foc) in enumerate(consumer_focs)
    println("  $i. $foc")
end
println("  Lagrange multipliers: $consumer_mults")

println("\n--- Step 2: Deriving Firm FOCs ---")

firm_controls = [:K_d, :L_d, :Y]
firm_objective = :(π[0] = Y[0] - L_d[0] * W[0] - r[0] * K_d[0])
firm_constraints = [
    :(Y[0] = Z[0] * K_d[0]^α * L_d[0]^(1 - α))
]

firm_focs, firm_mults = derive_focs(
    controls = firm_controls,
    objective = firm_objective,
    constraints = firm_constraints,
    discount_factor = :β,
    block_name = "f"
)

println("Firm FOCs derived:")
for (i, foc) in enumerate(firm_focs)
    println("  $i. $foc")
end
println("  Lagrange multipliers: $firm_mults")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Write the model using standard @model macro
# ──────────────────────────────────────────────────────────────────────────────

println("\n--- Step 3: Building the Full Model ---")

# The FOCs tell us:
# Consumer:
#   1. K_s FOC: -λ_c_2[0] + β * λ_c_1[1] * r[1] + β * λ_c_2[1] * (1-δ) = 0
#   2. C FOC: μ/C[0] - λ_c_1[0] = 0 → μ/C[0] = λ_c_1[0]
#   3. L_s FOC: -(1-μ)/(1-L_s[0]) + λ_c_1[0]*W[0] = 0
#   4. I FOC: -λ_c_1[0] + λ_c_2[0] = 0 → λ_c_1[0] = λ_c_2[0]
#
# Since λ_c_1 = λ_c_2 = λ (shadow price of wealth), we can simplify.
# Let λ = μ/C (from C FOC). Then:
#   - L_s FOC: (1-μ)/(1-L_s) = (μ/C)*W → labor-leisure tradeoff
#   - K_s FOC: 1 = β * (r[1] + (1-δ)) * (C[0]/C[1]) → Euler equation
#
# Firm (static): Since λ_f_1 = 1 from Y FOC:
#   - K_d FOC: r = α * Y/K_d
#   - L_d FOC: W = (1-α) * Y/L_d

@model RBC_gEcon begin
    # Consumer Euler equation (from K_s FOC, eliminating λ)
    # Original: μ/C[0] = β * (μ/C[1]) * (r[1] + (1-δ))
    1 / C[0] = β * (1 / C[1]) * (r[1] + (1 - δ))
    
    # Labor-leisure tradeoff (from L_s FOC)
    # (1-μ)/(1-L_s) = (μ/C)*W
    (1 - μ) * C[0] / (μ * (1 - L_s[0])) = W[0]
    
    # Consumer budget constraint
    C[0] + I[0] = π[0] + r[0] * K_s[-1] + W[0] * L_s[0]
    
    # Capital accumulation
    K_s[0] = (1 - δ) * K_s[-1] + I[0]
    
    # Firm FOCs: factor prices = marginal products
    r[0] = α * Y[0] / K_d[0]
    W[0] = (1 - α) * Y[0] / L_d[0]
    
    # Production function
    Y[0] = Z[0] * K_d[0]^α * L_d[0]^(1 - α)
    
    # Profit (zero in equilibrium with CRS)
    π[0] = Y[0] - W[0] * L_d[0] - r[0] * K_d[0]
    
    # Market clearing
    K_d[0] = K_s[-1]
    L_d[0] = L_s[0]
    
    # Technology shock
    Z[0] = exp(ϕ * log(Z[-1]) + σ_Z * ε_Z[x])
end

@parameters RBC_gEcon begin
    δ = 0.025    # Depreciation rate
    β = 0.99     # Discount factor
    μ = 0.3      # Consumption share in utility
    α = 0.36     # Capital share in production
    ϕ = 0.95     # AR(1) coefficient for technology
    σ_Z = 0.01   # Standard deviation of technology shock
end

println("\nModel created successfully!")
println(RBC_gEcon)

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Verify the model works
# ──────────────────────────────────────────────────────────────────────────────

println("\n--- Step 4: Verifying the Model ---")

ss = get_SS(RBC_gEcon)
println("\nSteady State:")
println(ss)

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Compute IRFs
# ──────────────────────────────────────────────────────────────────────────────

println("\n--- Step 5: Computing IRFs ---")

irfs = get_irf(RBC_gEcon)

println("\nIRF to technology shock (first 5 periods):")
println(irfs)

println("\n=== gEcon Model Conversion Complete ===")
