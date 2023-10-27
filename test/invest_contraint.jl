using MacroModelling

@model RBC begin
    c[0]^(-γ) + λ[0] = β * (c[1]^(-γ) * ((1 - δ) + α * exp(z[1]) * k[0]^(α - 1)) - (1 - δ) * λ[0])
    c[0] + i[0] = exp(z[0]) * k[-1]^α
    k[0] = (1 - δ) * k[-1] + exp(zⁱ[0]) * i[0]
    λ[0] * (i[0] - ϕ * i[ss]) = 0
    z[0] = ρ * z[-1] + std_z * eps_z[x]
    zⁱ[0] = ρⁱ * zⁱ[-1] + std_zⁱ * eps_zⁱ[x]

    i[0] ≥ ϕ * i[ss] | λ[0], eps_zⁱ > 0
    # 0 = max(i[0] - ϕ * i[ss], λ[0]) | eps_zⁱ > 0

    # î[0] = min(i[0], ϕ * i[ss]) | eps_zⁱ > 0
    # bind[0] = i[0] - ϕ * i[ss]
end

@parameters RBC begin
    std_z = 0.01
    std_zⁱ= 0.01
    ρ = 0.2
    ρⁱ= 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
    ϕ = 1.01
    γ = 1
    i > 0
end

SS(RBC)

plot_irf(RBC)

# assume that lagrange multiplier is always < 0
:(i[0] ≥ ϕ * i[ss] |  λ[0], eps_zⁱ > 0) |> dump