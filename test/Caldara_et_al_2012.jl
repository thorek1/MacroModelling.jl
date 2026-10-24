using MacroModelling

@model Caldara_et_al_2012 begin
	V[0] = ((1 - beta) * (c[0] ^ nu * (1 - l[0]) ^ (1 - nu)) ^ (1 - 1 / psi) + beta * V[1] ^ (1 - 1 / psi)) ^ (1 / (1 - 1 / psi))

	exp(s[0]) = V[1] ^ (1 - gamma)

	1 = beta * c[0] * (1 + zeta * exp(z[1]) * k[0] ^ (zeta - 1) * l[1] ^ (1 - zeta) - delta) * (((1 - l[1]) / (1 - l[0])) ^ (1 - nu) * (c[1] / c[0]) ^ nu) ^ (1 - 1 / psi) / c[1]

	R_k[0] = zeta * exp(z[1]) * k[0] ^ (zeta - 1) * l[1] ^ (1 - zeta) - delta

	SDF_plus__1[0] = (((1 - l[1]) / (1 - l[0])) ^ (1 - nu) * (c[1] / c[0]) ^ nu) ^ (1 - 1 / psi) * beta * c[0] / c[1]

	1 + R_f[0] = 1 / SDF_plus__1[0]

	c[0] * (1 - nu) / nu / (1 - l[0]) = (1 - zeta) * exp(z[0]) * k[-1] ^ zeta * l[0] ^ (-zeta)

	c[0] + i[0] = exp(z[0]) * k[-1] ^ zeta * l[0] ^ (1 - zeta)

	k[0] = i[0] + k[-1] * (1 - delta)

	z[0] = lambda * z[-1] + sigma[0] * epsilon__z[x]

	y[0] = exp(z[0]) * k[-1] ^ zeta * l[0] ^ (1 - zeta)

	log(sigma[0]) = (1 - rho) * log(sigma_bar) + rho * log(sigma[-1]) + eta * omega[x]

end


@parameters Caldara_et_al_2012 begin
	beta = 0.991

	zeta = 0.3

	delta = 0.0196

	lambda = 0.95

	psi = 0.5

	gamma = 40.0

	sigma_bar = 0.021

	eta = 0.1

	rho = 0.9

	nu = 0.36218431417051217

end

