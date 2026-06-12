using MacroModelling

@model Bernanke_Gertler_Gilchrist_1999_nonlinear begin
    # Household Euler equation: intertemporal optimality for deposits.
    C[0]^(-1) = β * C[1]^(-1) * R[0] / Pi[1]

    # Household intratemporal condition: consumption-leisure tradeoff.
    ζ * C[0] / (1 - H[0]) = W[0]

    # Labor aggregation with entrepreneurial labor share Ω.
    L[0] = H[0]^(1 - Ω)

    # Final goods technology: Cobb-Douglas production.
    Y[0] = A[0] * K[-1]^α * L[0]^(1 - α)

    # Household wage from marginal product of household labor.
    W[0] = (1 - α) * (1 - Ω) * Y[0] / (X[0] * H[0])

    # Entrepreneurial labor income from marginal product of entrepreneurial labor.
    W_e[0] = (1 - α) * Ω * Y[0] / X[0]

    # Gross return on capital before external finance costs.
    Rk[0] = (α * Y[0] / (X[0] * K[-1]) + (1 - δ) * Q[0]) / Q[-1]

    # Capital accumulation with convex investment adjustment costs.
    K[0] = (1 - δ) * K[-1] + (I[0] / K[-1] - ψ_i / 2 * (I[0] / K[-1] - δ)^2) * K[-1]

    # Tobin's Q implied by the marginal adjustment cost of investment.
    Q[0] = 1 / (1 - ψ_i * (I[0] / K[-1] - δ))

    # Aggregate resource constraint including entrepreneurial consumption and monitoring costs.
    Y[0] = C[0] + Ce[0] + I[0] + Gov[0] + monitoring_cost[0]

    # Monetary policy rule used in the paper's quantitative model.
    R[0] = (Pi_ss / β)^(1 - ρ_R) * R[-1]^ρ_R * (Pi[0] / Pi_ss)^((1 - ρ_R) * ϕ_pi) * (Y[0] / Y[ss])^((1 - ρ_R) * ϕ_y) * exp(std_R * eps_R[x])

    # Calvo price index.
    1.0 = θ * Pi[0]^(ϵ_p - 1) + (1 - θ) * Pi_star[0]^(1 - ϵ_p)

    # Optimal reset price from the Calvo price-setting FOC.
    Pi_star[0] = ϵ_p / (ϵ_p - 1) * price_aux_1[0] / price_aux_2[0]

    # Calvo numerator recursion for the optimal reset price.
    price_aux_1[0] = Y[0] / X[0] / C[0] + β * θ * Pi[1]^ϵ_p * price_aux_1[1]

    # Calvo denominator recursion for the optimal reset price.
    price_aux_2[0] = Y[0] / C[0] + β * θ * Pi[1]^(ϵ_p - 1) * price_aux_2[1]

    # Aggregate technology process.
    log(A[0]) = ρ_A * log(A[-1]) + std_A * eps_A[x]

    # Government spending share with persistent spending disturbance.
    Gov[0] = Gov_y * Y[0] * exp(gov_gap[0])

    # Government spending disturbance.
    gov_gap[0] = ρ_G * gov_gap[-1] + std_G * eps_G[x]

    # Money demand equation included in the baseline quantitative model.
    M_real[0] = b_m * C[0] / (R[0] - 1)

    # Entrepreneurial net worth before consumption, net of lender monitoring share Γ.
    V[0] = (1 - Gamma[0]) * Rk[0] * Q[-1] * K[-1]

    # Entrepreneurial consumption rule with survival probability γ_e.
    Ce[0] = (1 - γ_e) * V[0]

    # Entrepreneurial net worth: retained entrepreneurial wealth plus labor income.
    N[0] = γ_e * V[0] + W_e[0]

    # Balance sheet identity defining leverage.
    leverage[0] * N[0] = Q[0] * K[0]

    # Appendix B lognormal distribution: default probability F(ωbar).
    F[0] = normcdf((log(omega_bar[0]) + σ_ω^2 / 2) / σ_ω)

    # Appendix B lognormal distribution: partial expectation G(ωbar).
    G[0] = normcdf((log(omega_bar[0]) - σ_ω^2 / 2) / σ_ω)

    # Appendix B contract object Γ(ωbar) = G(ωbar) + ωbar * (1 - F(ωbar)).
    Gamma[0] = G[0] + omega_bar[0] * (1 - F[0])

    # Lender participation condition solved for the external finance premium.
    finance_premium[0] = (1 - 1 / leverage[0]) / (Gamma[0] - μ * G[0])

    # Optimal contract FOC linking leverage and the default threshold.
    leverage[0] - 1 = (1 - F[0]) * (Gamma[0] - μ * G[0]) / ((1 - F[0] - μ * omega_bar[0] * normpdf((log(omega_bar[0]) + σ_ω^2 / 2) / σ_ω) / σ_ω) * (1 - Gamma[0]))

    # External finance premium wedge: expected capital return over the safe real rate.
    Rk[1] = finance_premium[0] * R[0] / Pi[1]

    # Aggregate monitoring costs paid in default states.
    monitoring_cost[0] = μ * G[0] * Rk[0] * Q[-1] * K[-1]

    # Observable/log-output convenience variable.
    log_y[0] = log(Y[0])

    # Annualized external finance premium convenience variable.
    premium_ann[0] = 400 * log(finance_premium[0])
end

@parameters Bernanke_Gertler_Gilchrist_1999_nonlinear begin
    β = 0.99
    α = 0.35
    δ = 0.025
    Ω = 0.015384615384615385
    ζ = 2.0
    ψ_i = 4.0
    μ = 0.12
    σ_ω = 0.28
    γ_e = 0.9728
    Pi_ss = 1.0
    θ = 0.75
    ϵ_p = 11.0
    ρ_R = 0.9
    ϕ_pi = 1.1
    ϕ_y = 0.0
    ρ_A = 0.95
    std_A = 0.01
    Gov_y = 0.20
    ρ_G = 0.95
    std_G = 0.01
    std_R = 0.0025
    b_m = 0.01
end
