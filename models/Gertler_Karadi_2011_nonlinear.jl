using MacroModelling

@model Gertler_Karadi_2011_nonlinear begin
    # [4] Marginal utility of consumption with external habit.
    ϱ[0] = (C[0] - h * C[-1])^(-σ) - β * h * (C[1] - h * C[0])^(-σ)

    # [4'] Household Euler equation for the riskless asset.
    β * R[0] * Λ[1] = 1.0

    # [--] Stochastic discount factor.
    Λ[0] = ϱ[0] / ϱ[-1]

    # [3] Labor market equilibrium.
    χ * L[0]^φ_l = ϱ[0] * Pₘ[0] * (1 - α) * Y[0] / L[0]

    # [11] Marginal value of bankers' capital (ν_t).
    ν[0] = (1 - θ) * β * Λ[1] * (Rᵏ[1] - R[0]) + β * Λ[1] * θ * x[1] * ν[1]

    # [11] Marginal value of bankers' net worth (η_t).
    η[0] = (1 - θ) + β * Λ[1] * θ * z[1] * η[1]

    # [13] Incentive-constraint leverage relation (φ_t = η_t/(λ - ν_t)).
    φ[0] = η[0] / (λ - ν[0])

    # [14] Gross growth rate of bankers' capital (z_{t-1,t} = N_t/N_{t-1}).
    z[0] = (Rᵏ[0] - R[-1]) * φ[-1] + R[-1]

    # [--] Gross growth rate of assets (x_{t-1,t} = Q_t S_t/(Q_{t-1} S_{t-1})).
    x[0] = φ[0] / φ[-1] * z[0]

    # [15] Aggregate intermediary balance sheet.
    Q[0] * K[0] = φ[0] * N[0]

    # [16] Aggregate bankers' net worth (survivors + entrants).
    N[0] = Nᵉ[0] + Nⁿ[0]

    # [17] Existing bankers' net worth accumulation.
    Nᵉ[0] = θ * z[0] * N[-1] * exp(-σ_Ne * ε_Ne[x])

    # [18] Entering bankers' net worth.
    Nⁿ[0] = ω * Q[0] * ξ[0] * K[-1]

    # [25] Gross return on capital.
    Rᵏ[0] = (Pₘ[0] * α * Ym[0] / K[-1] + ξ[0] * (Q[0] - δ_rate[0])) / Q[-1]

    # [22] Intermediate goods production.
    Ym[0] = A[0] * (ξ[0] * U[0] * K[-1])^α * L[0]^(1 - α)

    # [27] Capital producer's optimal investment condition (Tobin's Q).
    Q[0] = 1 + η_i / 2 * ((In[0] + I[ss]) / (In[-1] + I[ss]) - 1)^2 +
        η_i * ((In[0] + I[ss]) / (In[-1] + I[ss]) - 1) * (In[0] + I[ss]) / (In[-1] + I[ss]) -
        β * Λ[1] * η_i * ((In[1] + I[ss]) / (In[0] + I[ss]) - 1) * ((In[1] + I[ss]) / (In[0] + I[ss]))^2

    # [--] Utilization-dependent depreciation.
    δ_rate[0] = δ + b_u / (1 + ζ_u) * (U[0]^(1 + ζ_u) - 1)

    # [23] Optimal capacity utilization.
    Pₘ[0] * α * Ym[0] / U[0] = b_u * U[0]^ζ_u * ξ[0] * K[-1]

    # [26] Net investment.
    In[0] = I[0] - δ_rate[0] * ξ[0] * K[-1]

    # [35] Capital accumulation.
    K[0] = ξ[0] * K[-1] + In[0]

    # [--] Government consumption.
    G[0] = g_y * Y[ss] * exp(ĝ[0])

    # [34] Aggregate resource constraint.
    Y[0] = C[0] + G[0] + I[0] + η_i / 2 * ((In[0] + I[ss]) / (In[-1] + I[ss]) - 1)^2 * (In[0] + I[ss])

    # [--] Wholesale and retail output relation.
    Ym[0] = Y[0] * D[0]

    # [--] Calvo price dispersion.
    D[0] = γ_p * D[-1] * π[-1]^(-γ_p_index * ϵ_p) * π[0]^ϵ_p +
        (1 - γ_p) * ((1 - γ_p * π[-1]^(γ_p_index * (1 - ϵ_p)) * π[0]^(ϵ_p - 1)) / (1 - γ_p))^(-ϵ_p / (1 - ϵ_p))

    # [--] Markup definition.
    X[0] = 1 / Pₘ[0]

    # [31] Optimal reset-price numerator recursion (F_t).
    F[0] = Y[0] * Pₘ[0] + β * γ_p * Λ[1] * π[1]^ϵ_p * π[0]^(-ϵ_p * γ_p_index) * F[1]

    # [31] Optimal reset-price denominator recursion (Z_t).
    Z[0] = Y[0] + β * γ_p * Λ[1] * π[1]^(ϵ_p - 1) * π[0]^(γ_p_index * (1 - ϵ_p)) * Z[1]

    # [31]–[32] Optimal reset-price inflation.
    π_star[0] = ϵ_p / (ϵ_p - 1) * F[0] / Z[0] * π[0]

    # [33] Aggregate price index.
    π[0]^(1 - ϵ_p) = γ_p * π[-1]^(γ_p_index * (1 - ϵ_p)) + (1 - γ_p) * π_star[0]^(1 - ϵ_p)

    # [38] Fisher equation.
    i[0] = R[0] * π[1]

    # [37] Interest-rate rule (Taylor rule with smoothing).
    i[0] = i[-1]^ρ_i * ((1 / β) * π[0]^κ_pi * (X[0] / (ϵ_p / (ϵ_p - 1)))^κ_y)^(1 - ρ_i) * exp(σ_i * ε_i[x])

    # [--] TFP process.
    A[0] = A[-1]^ρ_A * exp(-σ_A * ε_A[x])

    # [--] Capital quality process.
    ξ[0] = ξ[-1]^ρ_ξ * exp(-σ_ξ * ε_ξ[x])

    # [--] Government spending process.
    ĝ[0] = ρ_g * ĝ[-1] - σ_g * ε_g[x]

    # [--] Effective capital convenience variable.
    Keff[0] = ξ[0] * K[-1]

    # [--] Wage convenience variable.
    w[0] = Pₘ[0] * (1 - α) * Y[0] / L[0]

    # [--] Marginal value product of capital convenience variable.
    VMPK[0] = Pₘ[0] * α * Y[0] / (ξ[0] * K[-1])

    # [--] Premium convenience variable (Rᵏ_{t+1}/R_{t+1}).
    prem[0] = Rᵏ[1] / R[0]
end

@parameters Gertler_Karadi_2011_nonlinear begin
    β = 0.99
    σ = 1.0
    h = 0.815
    χ = 3.410808502195193
    φ_l = 0.276
    ζ_u = 7.2
    θ = 0.97155955
    α = 0.33
    δ = 0.025
    g_y = 0.2
    η_i = 1.728
    ϵ_p = 4.167
    γ_p = 0.779
    γ_p_index = 0.241
    ρ_i = 0.8
    κ_pi = 1.5
    κ_y = -0.5 / 4
    ρ_ξ = 0.66
    σ_ξ = 0.05
    ρ_A = 0.95
    σ_A = 0.01
    ρ_g = 0.95
    σ_g = 0.01
    σ_Ne = 0.01
    σ_i = 0.01
    λ = 0.38149498593241726
    ω = 0.0022277804179292593
    b_u = 0.037601010101010155
end
