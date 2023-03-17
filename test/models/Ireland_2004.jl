@model Ireland_2004 begin
	a[0] = ρᵃ * a[-1] + σᵃ * ϵᵃ[x]

	e[0] = ρᵉ * e[-1] + σᵉ * ϵᵉ[x]

	x[0] = αˣ * x[-1] + (1 - αˣ) * x[1] - (r̂[0] - π̂[1]) + a[0] * (1 - ω) * (1 - ρᵃ)

	π̂[0] = β * (αᵖ * π̂[-1] + π̂[1] * (1 - αᵖ)) + x[0] * ψ - e[0]

	x[0] = ŷ[0] - a[0] * ω

	ĝ[0] = σᶻ * ϵᶻ[x] + ŷ[0] - ŷ[-1]

	r̂[0] - r̂[-1] = π̂[0] * ρᵖ + ĝ[0] * ρᵍ + x[0] * ρˣ + σʳ * ϵʳ[x]
end


@parameters Ireland_2004 verbose = true begin
	β = 0.99

	ψ = 0.1

	ω = 0.0581

	αˣ = 0.00001

	αᵖ = 0.00001

	ρᵖ = 0.3866

	ρᵍ = 0.3960

	ρˣ = 0.1654

	ρᵃ = 0.9048

	ρᵉ = 0.9907

	σʳ = 0.0028

	σᵃ = 0.0302

	σᵉ = 0.0002

	σᶻ = 0.0089
end