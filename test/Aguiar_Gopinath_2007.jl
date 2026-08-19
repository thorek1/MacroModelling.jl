using MacroModelling

@model Aguiar_Gopinath_2007 begin
	y[0] = (exp(g[0]) * l[0]) ^ alpha * exp(z[0]) * k[-1] ^ (1 - alpha)

	z[0] = rho_z * z[-1] + sigma__z * eps_z[x]

	g[0] = (1 - rho_g) * mu_g + rho_g * g[-1] + sigma__g * eps_g[x]

	u[0] = (c[0] ^ gamma * (1 - l[0]) ^ (1 - gamma)) ^ (1 - sigma) / (1 - sigma)

	uc[0] = gamma * u[0] * (1 - sigma) / c[0]

	ul[0] = u[0] * (1 - sigma) * ( - (1 - gamma)) / (1 - l[0])

	c[0] + exp(g[0]) * k[0] = y[0] + k[-1] * (1 - delta) - k[-1] * phi / 2 * (exp(g[0]) * k[0] / k[-1] - exp(mu_g)) ^ 2 - b[-1] + exp(g[0]) * b[0] * q[0]

	1 / q[0] = 1 + r_star + psi * (exp(b[0] - b_star) - 1)

	exp(g[0]) * uc[0] * (1 + phi * (exp(g[0]) * k[0] / k[-1] - exp(mu_g))) = beta * exp((1 - sigma) * g[0] * gamma) * uc[1] * (1 - delta + (1 - alpha) * y[1] / k[0] - phi / 2 * (k[1] * exp(g[1]) * ( - (2 * (k[1] * exp(g[1]) / k[0] - exp(mu_g)))) / k[0] + (k[1] * exp(g[1]) / k[0] - exp(mu_g)) ^ 2))

	ul[0] + uc[0] * y[0] * alpha / l[0] = 0

	uc[0] * exp(g[0]) * q[0] = beta * exp((1 - sigma) * g[0] * gamma) * uc[1]

	invest[0] = exp(g[0]) * k[0] + k[-1] * phi / 2 * (exp(g[0]) * k[0] / k[-1] - exp(mu_g)) ^ 2 - k[-1] * (1 - delta)

	c_y[0] = c[0] / y[0]

	i_y[0] = invest[0] / y[0]

	nx[0] = (b[-1] - exp(g[0]) * b[0] * q[0]) / y[0]

	delta_y[0] = g[-1] + log(y[0]) - log(y[-1])

end


@parameters Aguiar_Gopinath_2007 begin
	gamma = 0.36

	b_share	=	0.1

	psi = 0.001

	alpha = 0.68

	sigma = 2.0

	delta = 0.05

	phi = 4.0

	rho_z = 0.95

	rho_g = 0.01

	sigma__z = 0.01

	sigma__g = 0.0005

	beta = 0.9803921568627451

	mu_g = 0.006578315360122507

	b_star = 0.0645176839232937

	r_star = 0.029166381484582272

end

