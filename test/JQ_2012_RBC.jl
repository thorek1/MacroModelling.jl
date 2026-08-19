using MacroModelling

@model JQ_2012_RBC begin
	w[0] / c[0] ^ sigma = alpha / (1 - n[0])

	c[0] ^ (-sigma) = beta * (R[0] - tau) / (1 - tau) * c[1] ^ (-sigma)

	w[0] * n[0] + b[-1] - b[0] / R[0] + d[0] = c[0]

	(1 - theta) * z[0] * k[-1] ^ theta * n[0] ^ (-theta) = w[0] / (1 - mu[0] * (1 + 2 * kappa * (d[0] - d[ss])))

	(1 + 2 * kappa * (d[0] - d[ss])) * beta * (c[0] / c[1]) ^ sigma / (1 + 2 * kappa * (d[1] - d[ss])) * (1 - delta + theta * (1 - (1 + 2 * kappa * (d[1] - d[ss])) * mu[1]) * z[1] * k[0] ^ (theta - 1) * n[1] ^ (1 - theta)) + mu[0] * (1 + 2 * kappa * (d[0] - d[ss])) * xi[0] = 1

	R[0] * beta * (c[0] / c[1]) ^ sigma * (1 + 2 * kappa * (d[0] - d[ss])) / (1 + 2 * kappa * (d[1] - d[ss])) + (1 - tau) * R[0] * mu[0] * (1 + 2 * kappa * (d[0] - d[ss])) * xi[0] / (R[0] - tau) = 1

	b[0] / R[0] + k[-1] * (1 - delta) + z[0] * k[-1] ^ theta * n[0] ^ (1 - theta) - w[0] * n[0] - b[-1] - k[0] = d[0] + kappa * (d[0] - d[ss]) ^ 2

	xi[0] * (k[0] - (1 - tau) * b[0] / (R[0] - tau)) = z[0] * k[-1] ^ theta * n[0] ^ (1 - theta)

	log(z[0] / zbar) = A_1_1 * log(z[-1] / zbar) + A_1__2 * log(xi[-1] / xi_bar) + sigma__z * epsilon__z[x]

	log(xi[0] / xi_bar) = log(z[-1] / zbar) * A_2__1 + log(xi[-1] / xi_bar) * A_2_2 + sigma__x__i * epsilon__x__i[x]

	y[0] = z[0] * k[-1] ^ theta * n[0] ^ (1 - theta)

	k[0] = k[-1] * (1 - delta) + i[0]

	v[0] = d[0] + c[0] * beta / c[1] * v[1]

	1 + r[0] = (R[0] - tau) / (1 - tau)

end


@parameters JQ_2012_RBC begin
	BY_ratio	=	3.36

	nbar	=	0.3

	zbar = 1.0

	beta = 0.9825

	sigma = 1.0

	theta = 0.36

	delta = 0.025

	tau = 0.35

	kappa = 0.146

	A_1_1 = 0.9457

	A_1__2 = (-0.0091)

	A_2__1 = 0.0321

	A_2_2 = 0.9703

	sigma__z = 0.0045

	sigma__x__i = 0.0098

	xi_bar = 0.16337753022030044

	alpha = 1.8834086344418162

end

