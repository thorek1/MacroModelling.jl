@model Ascari_Sbordone_2014 begin
	1 / y[0] ^ sigma = beta * (1 + i[0]) / (pi[1] * y[1] ^ sigma)

	w[0] = y[0] ^ sigma * d_n * exp(zeta[0]) * N[0] ^ phi_par

	p_star[0] = ((1 - theta * pi[-1] ^ ((1 - epsilon) * var_rho) * pi[0] ^ (epsilon - 1)) / (1 - theta)) ^ (1 / (1 - epsilon))

	p_star[0] ^ (1 + epsilon * alpha / (1 - alpha)) = epsilon / ((epsilon - 1) * (1 - alpha)) * psi[0] / phi[0]

	psi[0] = w[0] * exp(A[0]) ^ (( - 1) / (1 - alpha)) * y[0] ^ (1 / (1 - alpha) - sigma) + beta * theta * pi[0] ^ (epsilon * ( - var_rho) / (1 - alpha)) * pi[1] ^ (epsilon / (1 - alpha)) * psi[1]

	phi[0] = y[0] ^ (1 - sigma) + beta * theta * pi[0] ^ ((1 - epsilon) * var_rho) * pi[1] ^ (epsilon - 1) * phi[1]

	N[0] = s[0] * (y[0] / exp(A[0])) ^ (1 / (1 - alpha))

	s[0] = (1 - theta) * p_star[0] ^ (( - epsilon) / (1 - alpha)) + theta * pi[-1] ^ (var_rho * ( - epsilon) / (1 - alpha)) * pi[0] ^ (epsilon / (1 - alpha)) * s[-1]

	(1 + i[0]) / (1 + i_bar) = ((1 + i[-1]) / (1 + i_bar)) ^ rho_i * ((pi[0]/ Pi_bar) ^ phi_pi * (y[0] / Y_bar) ^ phi_y) ^ (1 - rho_i) * exp(v[0])

	MC_real[0] = w[0] * 1 / (1 - alpha) * exp(A[0]) ^ (1 / (alpha - 1)) * y[0] ^ (alpha / (1 - alpha))

	real_interest[0] = (1 + i[0]) / pi[1]

	Utility[0] = log(y[0]) - d_n * exp(zeta[0]) * N[0] ^ (1 + phi_par) / (1 + phi_par) + beta * Utility[1]

	v[0] = rho_v * v[-1] + σᵥ * e_v[x]

	A[0] = rho_a * A[-1] + σₐ * e_a[x]

	zeta[0] = rho_zeta * zeta[-1] + σ_zeta * e_zeta[x]

	A_tilde[0] = exp(A[0]) / s[0]

	Average_markup[0] = 1 / MC_real[0]

	Marginal_markup[0] = p_star[0] / MC_real[0]

	price_adjustment_gap[0] = 1 / p_star[0]
end


@parameters Ascari_Sbordone_2014 verbose = true begin
    Pi_bar = (1+trend_inflation/100)^(1/4)

	N[ss] = 1/3 | d_n
    
    (1/3) ^ (1 - alpha) =  y[ss] | Y_bar

    i_bar = Pi_bar / beta - 1
    
	beta = 0.99

	trend_inflation = 0

	alpha = 0

	theta = 0.75

	epsilon = 10

	sigma = 1

	rho_v = 0

	rho_a = 0

	rho_zeta = 0

	phi_par = 1

	phi_pi = 2

	phi_y = 0.125

	rho_i = 0.8

	var_rho = 0

	σ_zeta = .01

	σₐ = .01

	σᵥ = .01
end
