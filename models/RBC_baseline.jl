# King, R.G., Plosser, C.I., and Rebelo, S.T. (1988). Production, growth and business cycles:
# I. The basic neoclassical model. Journal of Monetary Economics, 21(2–3), 195–232.
# https://doi.org/10.1016/0304-3932(88)90030-X

using MacroModelling

@model RBC_baseline begin
	c[0] ^ (-σ) = β * c[1] ^ (-σ) * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)

	ψ * c[0] ^ σ / (1 - l[0]) = w[0]

	k[0] = (1 - δ) * k[-1] + i[0]

	y[0] = c[0] + i[0] + g[0]

	y[0] = z[0] * k[-1] ^ α * l[0] ^ (1 - α)

	w[0] = y[0] * (1 - α) / l[0]

	r[0] = y[0] * α * 4 / k[-1]

	z[0] = (1 - ρᶻ) + ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

	g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

end


@parameters RBC_baseline begin
	σᶻ = 0.066

	σᵍ = .104

	σ = 1

	α = 1/3

	i_y = 0.25

	k_y = 10.4

	ρᶻ = 0.97

	ρᵍ = 0.989

	g_y = 0.2038

	ḡ | ḡ = g_y * y[ss]

    δ = i_y / k_y

    β = 1 / (α / k_y + (1 - δ))

	ψ | l[ss] = 1/3
end
