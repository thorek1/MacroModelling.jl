@model Aguiar_Gopinath_2007 begin
	y[0] = (exp(g[0]) * l[0]) ^ alpha * exp(z[0]) * k[-1] ^ (1 - alpha)

	z[0] = rho_z * z[-1] + σᶻ * eps_z[x]

	g[0] = (1 - rho_g) * mu_g + rho_g * g[-1] + σᵍ * eps_g[x]

	l_m[0] = 1 - l[0] # works only with this; fix the nonnegativity constraint

	u[0] = (c[0] ^ gamma * l_m[0] ^ (1 - gamma)) ^ (1 - sigma) / (1 - sigma)

	uc[0] = (1 - sigma) * u[0] * gamma / c[0]

	ul[0] = (1 - sigma) * u[0] * ( - (1 - gamma)) / l_m[0]

	c[0] + k[0] * exp(g[0]) = y[0] + (1 - delta) * k[-1] - k[-1] * phi / 2 * (k[0] * exp(g[0]) / k[-1] - exp(mu_g)) ^ 2 - b[-1] + b[0] * exp(g[0]) * q[0]

	1 / q[0] = 1 + r_star + psi * (exp(b[0] - b_star) - 1)

	exp(g[0]) * uc[0] * (1 + phi * (k[0] * exp(g[0]) / k[-1] - exp(mu_g))) = beta * exp(g[0] * gamma * (1 - sigma)) * uc[1] * (1 - delta + (1 - alpha) * y[1] / k[0] - phi / 2 * (k[1] * exp(g[1]) * ( - (2 * (k[1] * exp(g[1]) / k[0] - exp(mu_g)))) / k[0] + (k[1] * exp(g[1]) / k[0] - exp(mu_g)) ^ 2))

	ul[0] + y[0] * alpha * uc[0] / l[0] = 0

	q[0] * exp(g[0]) * uc[0] = beta * exp(g[0] * gamma * (1 - sigma)) * uc[1]

	invest[0] = k[-1] * phi / 2 * (k[0] * exp(g[0]) / k[-1] - exp(mu_g)) ^ 2 + k[0] * exp(g[0]) - (1 - delta) * k[-1]

	c_y[0] = c[0] / y[0]

	i_y[0] = invest[0] / y[0]

	nx[0] = (b[-1] - b[0] * exp(g[0]) * q[0]) / y[0]

	delta_y[0] = g[-1] + log(y[0]) - log(y[-1])

end


@parameters Aguiar_Gopinath_2007 begin
	beta = 1 / 1.02

	gamma = 0.36

	b_share = 0.1

	b_share * y[ss] =  b_star | b_star

	1 + r_star = 1 / q[ss] | r_star

	psi = 0.001

	alpha = 0.68

	sigma = 2

	delta = 0.05

	phi = 4

	mu_g = log(1.0066)

	rho_z = 0.95

	rho_g = 0.01

	σᶻ = .01

	σᵍ = .0005
end