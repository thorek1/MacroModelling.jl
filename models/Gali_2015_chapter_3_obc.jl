@model Gali_2015_chapter_3_obc begin
	W_real[0] = C[0] ^ σ * N[0] ^ φ

	Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]

	R[0] = 1 / Q[0]

	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

	R[0] = Pi[1] * realinterest[0]

	R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))

	C[0] = Y[0]

	log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]

	log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]

	nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]

	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])

	1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)

	S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]

	Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)

	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]

	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]

	log_y[0] = log(Y[0])

	log_W_real[0] = log(W_real[0])

	log_N[0] = log(N[0])

	pi_ann[0] = 4 * log(Pi[0])

	i_ann[0] = 4 * log(R[0])

	r_real_ann[0] = 4 * log(realinterest[0])

	M_real[0] = Y[0] / R[0] ^ η

end


@parameters Gali_2015_chapter_3_obc begin
    R̄ = 1.0

	σ = 1

	φ = 5

	ϕᵖⁱ = 1.5
	
	ϕʸ = 0.125

	θ = 0.75

	ρ_ν = 0.5

	ρ_z = 0.5

	ρ_a = 0.9

	β = 0.99

	η = 3.77

	α = 0.25

	ϵ = 9

	τ = 0

    std_a = .01

    std_z = .05

    std_nu = .0025

    # R > 1.000001
end

using StatsPlots

plot_irf(Gali_2015_chapter_3_obc, shocks = :eps_z, ignore_obc = true)
plot_simulations(Gali_2015_chapter_3_obc)
plot_simulations(Gali_2015_chapter_3_obc, variables = :all)

get_solution(Gali_2015_chapter_3_obc)
get_solution(Gali_2015_chapter_3_obc, algorithm = :riccati, parameters = (:activeᵒᵇᶜshocks => 1.0, :R̄ => 0.0))
get_solution(Gali_2015_chapter_3_obc, parameters = (:R̄ => 1.0))

get_parameters(Gali_2015_chapter_3_obc)

SS(Gali_2015_chapter_3_obc)
SS(Gali_2015_chapter_3_obc, parameters = (:R̄ => 1.0))
SS(Gali_2015_chapter_3_obc, parameters = (:R̄ => 0.0))
Gali_2015_chapter_3_obc.ss_aux_equations
Gali_2015_chapter_3_obc.dyn_equations

𝓂 = Gali_2015_chapter_3_obc

SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, true, false, 𝓂.solver_parameters)

∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix

∇₁l = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix

𝓂.model_jacobian