using MacroModelling

# function GK2011_intermediary_leverage(spread, β, θ, λ)
#     aa = λ * β * θ * spread
#     bb = -(1 - θ) * (λ - β * spread)
#     cc = 1 - θ
#     discriminant = bb^2 - 4 * aa * cc

#     if discriminant < 0
#         return NaN
#     end

#     return (-bb - sqrt(discriminant)) / (2 * aa)
# end

# function GK2011_spread_gap(spread, β, θ, λ, ω)
#     R = 1 / β
#     φ = GK2011_intermediary_leverage(spread, β, θ, λ)
#     z = spread * φ + R

#     return φ * ω / (1 - θ * z) - 1
# end

# function GK2011_solve_spread(β, θ, λ, ω)
#     lower = 1e-8
#     lower_gap = GK2011_spread_gap(lower, β, θ, λ, ω)

#     upper = lower
#     upper_gap = lower_gap
#     for trial in exp.(range(log(lower * 1.01), log(0.05), length = 400))
#         trial_gap = GK2011_spread_gap(trial, β, θ, λ, ω)
#         if isfinite(trial_gap)
#             upper = trial
#             upper_gap = trial_gap
#             if lower_gap * upper_gap <= 0
#                 break
#             end
#         end
#     end

#     if lower_gap * upper_gap > 0 || !isfinite(upper_gap)
#         error("Could not bracket the GK2011 steady-state intermediary spread.")
#     end

#     midpoint = (lower + upper) / 2
#     for iteration in 1:200
#         midpoint = (lower + upper) / 2
#         midpoint_gap = GK2011_spread_gap(midpoint, β, θ, λ, ω)

#         if abs(midpoint_gap) < 1e-13
#             return midpoint
#         elseif lower_gap * midpoint_gap <= 0
#             upper = midpoint
#             upper_gap = midpoint_gap
#         else
#             lower = midpoint
#             lower_gap = midpoint_gap
#         end
#     end

#     return midpoint
# end

# function GK2011_nonlinear_steady_state!(ss, parameters)
#     β, σ, h, χ, φ_l, ζ_u, θ, α, δ, g_y, η_i, ϵ_p, γ_p, γ_p_index, ρ_i,
#     κ_pi, ρ_ξ, std_ξ, ρ_A, std_A, ρ_g, std_g, std_Ne, std_i, λ, ω, b_u,
#     κ_y = parameters

#     Pm = (ϵ_p - 1) / ϵ_p
#     X = 1 / Pm
#     R = 1 / β
#     spread = GK2011_solve_spread(β, θ, λ, ω)
#     Rk = R + spread
#     φ = GK2011_intermediary_leverage(spread, β, θ, λ)
#     z = spread * φ + R
#     x = z
#     ν = ((1 - θ) * β * spread) / (1 - β * θ * x)
#     η = (1 - θ) / (1 - β * θ * z)

#     K_to_L = (Pm * α / (Rk - 1 + δ))^(1 / (1 - α))
#     consumption_to_labor = (1 - g_y) * K_to_L^α - δ * K_to_L
#     labor_constant =
#         (1 - β * h) * ((1 - h) * consumption_to_labor)^(-σ) *
#         Pm * (1 - α) * K_to_L^α
#     L = (labor_constant / χ)^(1 / (φ_l + σ))
#     K = K_to_L * L
#     Y = K^α * L^(1 - α)
#     Ym = Y
#     I = δ * K
#     G = g_y * Y
#     C = Y - I - G
#     varrho = (1 - β * h) * ((1 - h) * C)^(-σ)
#     Λ = 1.0
#     Q = 1.0
#     Ξ = 1.0
#     U = 1.0
#     δ_rate = δ
#     In = 0.0
#     N = K / φ
#     Ne = θ * z * N
#     Nn = ω * K
#     Keff = K
#     w = Pm * (1 - α) * Y / L
#     VMPK = Pm * α * Y / K
#     D = 1.0
#     F_price = Y * Pm / (1 - β * γ_p)
#     Z_price = Y / (1 - β * γ_p)
#     Pi = 1.0
#     Pi_star = 1.0
#     i_nom = R
#     prem = Rk / R
#     A = 1.0
#     g_gap = 0.0

#     if length(ss) != 38
#         resize!(ss, 38)
#     end

#     ss[1] = A
#     ss[2] = C
#     ss[3] = D
#     ss[4] = F_price
#     ss[5] = G
#     ss[6] = I
#     ss[7] = In
#     ss[8] = K
#     ss[9] = Keff
#     ss[10] = L
#     ss[11] = Λ
#     ss[12] = N
#     ss[13] = Ne
#     ss[14] = Nn
#     ss[15] = Pi
#     ss[16] = Pi_star
#     ss[17] = Pm
#     ss[18] = Q
#     ss[19] = R
#     ss[20] = Rk
#     ss[21] = U
#     ss[22] = VMPK
#     ss[23] = X
#     ss[24] = Ξ
#     ss[25] = Y
#     ss[26] = Ym
#     ss[27] = Z_price
#     ss[28] = δ_rate
#     ss[29] = η
#     ss[30] = g_gap
#     ss[31] = i_nom
#     ss[32] = ν
#     ss[33] = φ
#     ss[34] = prem
#     ss[35] = varrho
#     ss[36] = w
#     ss[37] = x
#     ss[38] = z

#     return ss
# end

# function GK2011_nonlinear_steady_state(parameters)
#     ss = Vector{Float64}(undef, 38)
#     return GK2011_nonlinear_steady_state!(ss, parameters)
# end


@model Gertler_Karadi_2011_nonlinear begin
    # GK replication eq. 1: marginal utility of consumption with external habit.
    varrho[0] = (C[0] - h * C[-1])^(-σ) - β * h * (C[1] - h * C[0])^(-σ)

    # GK replication eq. 2: household Euler equation for the riskless asset.
    β * R[0] * Lambda[1] = 1.0

    # GK replication eq. 3: stochastic discount factor.
    Lambda[0] = varrho[0] / varrho[-1]

    # GK replication eq. 4: labor market equilibrium.
    χ * L[0]^φ_l = varrho[0] * Pm[0] * (1 - α) * Y[0] / L[0]

    # GK replication eq. 5: marginal value of bankers' capital.
    nu[0] = (1 - θ) * β * Lambda[1] * (Rk[1] - R[0]) + β * Lambda[1] * θ * x[1] * nu[1]

    # GK replication eq. 6: marginal value of bankers' net worth.
    eta[0] = (1 - θ) + β * Lambda[1] * θ * z[1] * eta[1]

    # GK replication eq. 7: incentive-constraint leverage relation.
    phi[0] = eta[0] / (λ - nu[0])

    # GK replication eq. 8: gross growth rate of bankers' capital.
    z[0] = (Rk[0] - R[-1]) * phi[-1] + R[-1]

    # GK replication eq. 9: gross growth rate of bankers' net worth.
    x[0] = phi[0] / phi[-1] * z[0]

    # GK replication eq. 10: aggregate intermediary balance sheet.
    Q[0] * K[0] = phi[0] * N[0]

    # GK replication eq. 11: aggregate bankers' net worth.
    N[0] = Ne[0] + Nn[0]

    # GK replication eq. 12: existing bankers' net worth accumulation.
    Ne[0] = θ * z[0] * N[-1] * exp(-std_Ne * eps_Ne[x])

    # GK replication eq. 13: entering bankers' net worth.
    Nn[0] = ω * Q[0] * Xi[0] * K[-1]

    # GK replication eq. 14: gross return on capital.
    Rk[0] = (Pm[0] * α * Ym[0] / K[-1] + Xi[0] * (Q[0] - delta_rate[0])) / Q[-1]

    # GK replication eq. 15: intermediate goods production.
    Ym[0] = A[0] * (Xi[0] * U[0] * K[-1])^α * L[0]^(1 - α)

    # GK replication eq. 16: capital producer's optimal investment condition.
    Q[0] = 1 + η_i / 2 * ((In[0] + I[ss]) / (In[-1] + I[ss]) - 1)^2 +
        η_i * ((In[0] + I[ss]) / (In[-1] + I[ss]) - 1) * (In[0] + I[ss]) / (In[-1] + I[ss]) -
        β * Lambda[1] * η_i * ((In[1] + I[ss]) / (In[0] + I[ss]) - 1) * ((In[1] + I[ss]) / (In[0] + I[ss]))^2

    # GK replication eq. 17: utilization-dependent depreciation.
    delta_rate[0] = δ + b_u / (1 + ζ_u) * (U[0]^(1 + ζ_u) - 1)

    # GK replication eq. 18: optimal capacity utilization.
    Pm[0] * α * Ym[0] / U[0] = b_u * U[0]^ζ_u * Xi[0] * K[-1]

    # GK replication eq. 19: net investment.
    In[0] = I[0] - delta_rate[0] * Xi[0] * K[-1]

    # GK replication eq. 20: capital accumulation.
    K[0] = Xi[0] * K[-1] + In[0]

    # GK replication eq. 21: government consumption.
    G[0] = g_y * Y[ss] * exp(g_gap[0])

    # GK replication eq. 22: aggregate resource constraint.
    Y[0] = C[0] + G[0] + I[0] + η_i / 2 * ((In[0] + I[ss]) / (In[-1] + I[ss]) - 1)^2 * (In[0] + I[ss])

    # GK replication eq. 23: wholesale and retail output relation.
    Ym[0] = Y[0] * D[0]

    # GK replication eq. 24: Calvo price dispersion.
    D[0] = γ_p * D[-1] * Pi[-1]^(-γ_p_index * ϵ_p) * Pi[0]^ϵ_p +
        (1 - γ_p) * ((1 - γ_p * Pi[-1]^(γ_p_index * (1 - γ_p)) * Pi[0]^(γ_p - 1)) / (1 - γ_p))^(-ϵ_p / (1 - γ_p))

    # GK replication eq. 25: markup definition.
    X[0] = 1 / Pm[0]

    # GK replication eq. 26: optimal reset-price numerator recursion.
    F_price[0] = Y[0] * Pm[0] + β * γ_p * Lambda[1] * Pi[1]^ϵ_p * Pi[0]^(-ϵ_p * γ_p_index) * F_price[1]

    # GK replication eq. 27: optimal reset-price denominator recursion.
    Z_price[0] = Y[0] + β * γ_p * Lambda[1] * Pi[1]^(ϵ_p - 1) * Pi[0]^(γ_p_index * (1 - ϵ_p)) * Z_price[1]

    # GK replication eq. 28: optimal reset-price inflation.
    Pi_star[0] = ϵ_p / (ϵ_p - 1) * F_price[0] / Z_price[0] * Pi[0]

    # GK replication eq. 29: aggregate price index.
    Pi[0]^(1 - ϵ_p) = γ_p * Pi[-1]^(γ_p_index * (1 - ϵ_p)) + (1 - γ_p) * Pi_star[0]^(1 - ϵ_p)

    # GK replication eq. 30: Fisher equation.
    i_nom[0] = R[0] * Pi[1]

    # GK replication eq. 31: interest-rate rule.
    i_nom[0] = i_nom[-1]^ρ_i * ((1 / β) * Pi[0]^κ_pi * (X[0] / (ϵ_p / (ϵ_p - 1)))^κ_y)^(1 - ρ_i) * exp(std_i * eps_i[x])

    # GK replication eq. 32: TFP process.
    A[0] = A[-1]^ρ_A * exp(-std_A * eps_A[x])

    # GK replication eq. 33: capital quality process.
    Xi[0] = Xi[-1]^ρ_ξ * exp(-std_ξ * eps_ξ[x])

    # GK replication eq. 34: government spending process.
    g_gap[0] = ρ_g * g_gap[-1] - std_g * eps_g[x]

    # GK replication eq. 35: effective capital convenience variable.
    Keff[0] = Xi[0] * K[-1]

    # GK replication eq. 36: wage convenience variable.
    w[0] = Pm[0] * (1 - α) * Y[0] / L[0]

    # GK replication eq. 37: marginal value product of capital convenience variable.
    VMPK[0] = Pm[0] * α * Y[0] / (Xi[0] * K[-1])

    # GK replication eq. 39: premium convenience variable.
    prem[0] = Rk[1] / R[0]
end


# @parameters Gertler_Karadi_2011_nonlinear steady_state_function = GK2011_nonlinear_steady_state! begin
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
    ρ_i = 0.0
    κ_pi = 1.5
    κ_y = -0.5 / 4
    ρ_ξ = 0.66
    std_ξ = 0.05
    ρ_A = 0.95
    std_A = 0.01
    ρ_g = 0.95
    std_g = 0.01
    std_Ne = 0.01
    std_i = 0.01
    λ = 0.38149498593241726
    ω = 0.0022277804179292593
    b_u = 0.037601010101010155

    β > 0
    σ > 0
    0 <= h < 1
    χ > 0
    φ_l > 0
    ζ_u > 0
    0 < θ < 1
    0 < α < 1
    0 < δ < 1
    0 <= g_y < 1
    η_i >= 0
    ϵ_p > 1
    0 <= γ_p < 1
    0 <= γ_p_index < 1
    0 <= ρ_i < 1
    κ_pi > 1
    0 <= ρ_ξ < 1
    std_ξ >= 0
    0 <= ρ_A < 1
    std_A >= 0
    0 <= ρ_g < 1
    std_g >= 0
    std_Ne >= 0
    std_i >= 0
    λ > 0
    ω > 0
    b_u > 0
end
