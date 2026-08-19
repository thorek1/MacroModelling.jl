using MacroModelling

@model Gali_2015_chapter_3_nonlinear begin
	W_real[0] = C[0] ^ sigma * N[0] ^ varphi

	Q[0] = beta * (C[1] / C[0]) ^ (-sigma) * Z[1] / Z[0] / Pi[1]

	R[0] = 1 / Q[0]

	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - alpha)

	R[0] = Pi[1] * realinterest[0]

	R[0] = 1 / beta * Pi[0] ^ phi__p__i * (Y[0] / Y[ss]) ^ phi__y * exp(nu[0])

	C[0] = Y[0]

	log(A[0]) = rho__a * log(A[-1]) + std_a * eps_a[x]

	log(Z[0]) = rho__z * log(Z[-1]) - std_z * eps_z[x]

	nu[0] = rho__nu * nu[-1] + std_nu * eps_nu[x]

	MC[0] = W_real[0] / ((1 - alpha) * Y[0] * S[0] / N[0])

	1 = theta * Pi[0] ^ (epsilon - 1) + (1 - theta) * Pi_star[0] ^ (1 - epsilon)

	S[0] = (1 - theta) * Pi_star[0] ^ (( - epsilon) / (1 - alpha)) + theta * Pi[0] ^ (epsilon / (1 - alpha)) * S[-1]

	Pi_star[0] ^ (1 + alpha * epsilon / (1 - alpha)) = epsilon * x_aux_1[0] / x_aux_2[0] * (1 - tau) / (epsilon - 1)

	x_aux_1[0] = Z[0] * Y[0] * MC[0] * C[0] ^ (-sigma) + beta * theta * Pi[1] ^ (epsilon + alpha * epsilon / (1 - alpha)) * x_aux_1[1]

	x_aux_2[0] = C[0] ^ (-sigma) * Z[0] * Y[0] + beta * theta * Pi[1] ^ (epsilon - 1) * x_aux_2[1]

	log_y[0] = log(Y[0])

	log_W_real[0] = log(W_real[0])

	log_N[0] = log(N[0])

	pi_ann[0] = 4 * log(Pi[0])

	i_ann[0] = 4 * log(R[0])

	r_real_ann[0] = 4 * log(realinterest[0])

	M_real[0] = Y[0] / R[0] ^ eta

end


@parameters Gali_2015_chapter_3_nonlinear begin
	sigma = 1.0

	varphi = 5.0

	phi__p__i = 1.5

	phi__y = 0.125

	theta = 0.75

	rho__nu = 0.5

	rho__z = 0.5

	rho__a = 0.9

	beta = 0.99

	eta = 3.77

	alpha = 0.25

	epsilon = 9.0

	tau = 0.0

	std_a = 0.01

	std_z = 0.05

	std_nu = 0.0025

end

