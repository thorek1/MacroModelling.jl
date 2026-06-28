using MacroModelling

@model Bernanke_Gertler_Gilchrist_1999 begin
    # [B.3] Household Euler equation: intertemporal optimality for deposits.
    C[0]^(-1) = β * C[1]^(-1) * R[0] / π[1]

    # [B.4] Household intratemporal condition: consumption-leisure tradeoff.
    ζ * C[0] / (1 - H[0]) = W[0]

    # [4.6] Labor aggregation with entrepreneurial labor share Ω (Hᵉ normalized to 1).
    L[0] = H[0]^(1 - Ω) * Hᵉ^Ω

    # [4.1] Final goods technology: Cobb-Douglas production.
    Y[0] = A[0] * K[-1]^α * L[0]^(1 - α)

    # [4.11] Household labor demand: wage equals marginal product.
    W[0] = (1 - α) * (1 - Ω) * Y[0] / (X[0] * H[0])

    # [4.12] Entrepreneurial labor income from marginal product.
    Wᵉ[0] = (1 - α) * Ω * Y[0] / X[0]

    # [4.4] Gross return on capital (Rᵏ = R^k in the paper).
    Rᵏ[0] = (α * Y[0] / (X[0] * K[-1]) + (1 - δ) * Q[0]) / Q[-1]

    # [4.2] Capital accumulation with convex investment adjustment costs.
    K[0] = (1 - δ) * K[-1] + (I[0] / K[-1] - ψⁱ / 2 * (I[0] / K[-1] - δ)^2) * K[-1]

    # [4.3] Tobin's Q implied by the marginal adjustment cost of investment.
    Q[0] = 1 / (1 - ψⁱ * (I[0] / K[-1] - δ))

    # [B.8] Aggregate resource constraint including entrepreneurial consumption and monitoring costs.
    Y[0] = C[0] + Cᵉ[0] + I[0] + Gov[0] + monitoring_cost[0]

    # [4.25] Monetary policy rule used in the paper's quantitative model.
    R[0] = (πˢˢ / β)^(1 - ρʳ) * R[-1]^ρʳ * (π[0] / πˢˢ)^((1 - ρʳ) * ϕ_pi) * (Y[0] / Y[ss])^((1 - ρʳ) * ϕʸ) * exp(σʳ * εʳ[x])

    # [B.12] Calvo price index.
    1.0 = θ * π[0]^(ϵᵖ - 1) + (1 - θ) * πstar[0]^(1 - ϵᵖ)

    # [B.11] Optimal reset price from the Calvo price-setting FOC.
    πstar[0] = ϵᵖ / (ϵᵖ - 1) * price_aux_1[0] / price_aux_2[0]

    # [B.11a] Calvo numerator recursion.
    price_aux_1[0] = Y[0] / X[0] / C[0] + β * θ * π[1]^ϵᵖ * price_aux_1[1]

    # [B.11b] Calvo denominator recursion.
    price_aux_2[0] = Y[0] / C[0] + β * θ * π[1]^(ϵᵖ - 1) * price_aux_2[1]

    # [4.27] Aggregate technology process.
    log(A[0]) = ρᵃ * log(A[-1]) + σᵃ * εᵃ[x]

    # [4.26] Government spending level (level form from the log-linearized process).
    Gov[0] = Govʸ * Y[0] * exp(gov_gap[0])

    # [4.26] Government spending AR(1) disturbance.
    gov_gap[0] = ρᵍ * gov_gap[-1] + σᵍ * εᵍ[x]

    # [B.5] Money demand equation.
    M_real[0] = χ * C[0] * R[0] / (R[0] - 1)

    # [4.8] Entrepreneurial equity (net of lender repayment and monitoring).
    V[0] = (1 - Γ[0]) * Rᵏ[0] * Q[-1] * K[-1]

    # [4.8] Entrepreneurial consumption: dying entrepreneurs consume their equity.
    Cᵉ[0] = (1 - γᵉ) * V[0]

    # [4.7] Entrepreneurial net worth: retained equity plus labor income.
    N[0] = γᵉ * V[0] + Wᵉ[0]

    # [3.2] Balance sheet identity defining leverage.
    leverage[0] * N[0] = Q[0] * K[0]

    # [A.2] Lognormal distribution: default probability F(ω̄).
    F[0] = normcdf((log(ω̄[0]) + σ_ω^2 / 2) / σ_ω)

    # [A.2] Lognormal distribution: partial expectation G(ω̄).
    G[0] = normcdf((log(ω̄[0]) - σ_ω^2 / 2) / σ_ω)

    # [A.2] Contract object Γ(ω̄) = G(ω̄) + ω̄(1 - F(ω̄)).
    Γ[0] = G[0] + ω̄[0] * (1 - F[0])

    # [3.5] Lender participation condition (inverted for external finance premium s).
    s[0] = (1 - 1 / leverage[0]) / (Γ[0] - μ * G[0])

    # [A.1]–[A.2] Optimal contract FOC linking leverage and the default threshold.
    leverage[0] - 1 = (1 - F[0]) * (Γ[0] - μ * G[0]) / ((1 - F[0] - μ * ω̄[0] * normpdf((log(ω̄[0]) + σ_ω^2 / 2) / σ_ω) / σ_ω) * (1 - Γ[0]))

    # [4.5] External finance premium wedge (ex-post).
    Rᵏ[1] = s[0] * R[0] / π[1]

    # [B.8] Aggregate monitoring costs paid in default states.
    monitoring_cost[0] = μ * G[0] * Rᵏ[0] * Q[-1] * K[-1]

    # Observable/log-output convenience variable.
    log_y[0] = log(Y[0])

    # Annualized external finance premium convenience variable.
    premium_ann[0] = 400 * log(s[0])
end

@parameters Bernanke_Gertler_Gilchrist_1999 begin
    β = 0.99
    α = 0.35
    δ = 0.025
    Ω = 0.015384615384615385
    Hᵉ = 1.0
    ζ = 2.0
    ψⁱ = 4.0
    μ = 0.12
    σ_ω = 0.28
    γᵉ = 0.9728
    πˢˢ = 1.0
    θ = 0.75
    ϵᵖ = 11.0
    ρʳ = 0.9
    ϕ_pi = 1.1
    ϕʸ = 0.0
    ρᵃ = 0.95
    σᵃ = 0.01
    Govʸ = 0.20
    ρᵍ = 0.95
    σᵍ = 0.01
    σʳ = 0.0025
    χ = 0.0099
end
