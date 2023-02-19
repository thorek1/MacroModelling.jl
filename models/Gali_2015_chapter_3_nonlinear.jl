@model Gali_2015_chapter_3_nonlinear begin
	W_real[0] = C[0] ^ siggma * N[0] ^ varphi

	Q[0] = betta * (C[1] / C[0]) ^ (-siggma) * Z[1] / Z[0] / Pi[1]

	R[0] = 1 / Q[0]

	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - alppha)

	R[0] = Pi[1] * realinterest[0]

	R[0] = 1 / betta * Pi[0] ^ phi_pi * (Y[0] / Y[ss]) ^ phi_y * exp(nu[0])

	C[0] = Y[0]

	log(A[0]) = rho_a * log(A[-1]) + std_a * eps_a[x]

	log(Z[0]) = rho_z * log(Z[-1]) - std_z * eps_z[x]

	nu[0] = rho_nu * nu[-1] + std_nu * eps_nu[x]

	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - alppha) / N[0])

	1 = theta * Pi[0] ^ (epsilon - 1) + (1 - theta) * Pi_star[0] ^ (1 - epsilon)

	S[0] = (1 - theta) * Pi_star[0] ^ (( - epsilon) / (1 - alppha)) + theta * Pi[0] ^ (epsilon / (1 - alppha)) * S[-1]

	Pi_star[0] ^ (1 + epsilon * alppha / (1 - alppha)) = epsilon * x_aux_1[0] / x_aux_2[0] * (1 - tau) / (epsilon - 1)

	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-siggma) + betta * theta * Pi[1] ^ (epsilon + alppha * epsilon / (1 - alppha)) * x_aux_1[1]

	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-siggma) + betta * theta * Pi[1] ^ (epsilon - 1) * x_aux_2[1]

	log_y[0] = log(Y[0])

	log_W_real[0] = log(W_real[0])

	log_N[0] = log(N[0])

	pi_ann[0] = 4 * log(Pi[0])

	i_ann[0] = 4 * log(R[0])

	r_real_ann[0] = 4 * log(realinterest[0])

	M_real[0] = Y[0] / R[0] ^ eta

end


@parameters Gali_2015_chapter_3_nonlinear begin
	siggma = 1

	varphi = 5

	phi_pi = 1.5

	phi_y = 0.125

	theta = 0.75

	rho_nu = 0.5

	rho_z = 0.5

	rho_a = 0.9

	betta = 0.99

	eta = 3.77

	alppha = 0.25

	epsilon = 9

	tau = 0

    std_a = .01

    std_z = .05

    std_nu = .0025

end

