using MacroModelling

@model JQ_2012 begin
	w[0] / c[0] ^ siggma - alppha / (1 - n[0]) = 0

	c[0] ^ (-siggma) = betta * (R[0] - tau) / (1 - tau) * c[1] ^ (-siggma)

	w[0] * n[0] + b[-1] - b[0] / R[0] + d[0] - c[0] = 0

	(1 - theta) * z[0] * k[-1] ^ theta * n[0] ^ (-theta) = w[0] * 1 / (1 - mu[0] * (1 + 2 * kappa * (d[0] - d[ss])))

	betta * (c[0] / c[1]) ^ siggma * (1 + 2 * kappa * (d[0] - d[ss])) / (1 + 2 * kappa * (d[1] - d[ss])) * (1 - delta + theta * (1 - (1 + 2 * kappa * (d[1] - d[ss])) * mu[1]) * z[1] * k[0] ^ (theta - 1) * n[1] ^ (1 - theta)) + (1 + 2 * kappa * (d[0] - d[ss])) * mu[0] * xi[0] = 1

	(1 + 2 * kappa * (d[0] - d[ss])) / (1 + 2 * kappa * (d[1] - d[ss])) * (c[0] / c[1]) ^ siggma * betta * R[0] + (1 + 2 * kappa * (d[0] - d[ss])) * mu[0] * xi[0] * R[0] * (1 - tau) / (R[0] - tau) = 1

	b[0] / R[0] + k[-1] * (1 - delta) + z[0] * k[-1] ^ theta * n[0] ^ (1 - theta) - w[0] * n[0] - b[-1] - k[0] - (d[0] + kappa * (d[0] - d[ss]) ^ 2) = 0

	xi[0] * (k[0] - b[0] * (1 - tau) / (R[0] - tau)) = z[0] * k[-1] ^ theta * n[0] ^ (1 - theta)

	log(z[0] / z[ss]) = A11 * log(z[-1] / z[ss]) + A12 * log(xi[-1] / xi_bar) + σᶻ * eps_z[x]

	log(xi[0] / xi_bar) = log(z[-1] / z[ss]) * A21 + log(xi[-1] / xi_bar) * A22 + σˣⁱ * eps_xi[x]

	y[0] = z[0] * k[-1] ^ theta * n[0] ^ (1 - theta)

	invest[0] = k[0] - k[-1] * (1 - delta)

	v[0] = d[0] + c[0] * betta / c[1] * v[1]

	r[0] = (R[0] - tau) / (1 - tau) - 1

	# yhat[0] = log(y[0]) - log(y[ss])

	# chat[0] = log(c[0]) - log(c[ss])

	# ihat[0] = log(invest[0]) - log(invest[ss])

	# nhat[0] = log(n[0]) - log(n[ss])

	# muhat[0] = log(mu[0]) - log(mu[ss])

	# byhat[0] = (b[-1] / (1 + r[-1]) - b[0] / (1 + r[0])) / y[0]

	# dyhat[0] = d[0] / y[0]

	# vyhat[0] = log(v[0] / (k[-1] - b[-1])) - log((v[0] / (k[-1] - b[-1])))

end


@parameters JQ_2012 begin
	σᶻ = sqrt(0.00002)

	σˣⁱ = sqrt(0.000096)

	n[ss] = .3 | alppha

	BY_ratio = 3.36

	b[ss] / (1 + r[ss]) / y[ss] = BY_ratio | xi_bar

	# xi_bar = .163

	z[ss] = 1 | betta

	# betta = 0.9825

	siggma = 1

	# theta = 0.36

	# delta = 0.025

	k[ss] = 10.0795 | delta
	tau = 0.35
	r[ss] = 0.0178 | theta
	# v[ss] = 6.56 | theta
	# w[ss] = 2.1825 | tau
	# w[ss] = 2.1825 | tau
	# y[ss] = 1.0632 | siggma

	kappa = 0.146

	sigma_xi = 0.0098

	sigma_z = 0.0045

	covariance_z_xi =0

	A11 = 0.9457

	A12 = -0.0091

	A21 = 0.0321

	A22 = 0.9703

	.1633 < xi < .1634

end
