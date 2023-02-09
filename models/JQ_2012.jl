@model JQ_2012 begin
	1 = P[0] * betta * gamma[1] * (c[1] - h * c[0]) ^ (-siggma) / (gamma[0] * (c[0] - h * c[-1]) ^ (-siggma)) * (1 + r[0]) / P[1]

	y[0] * 1 / eta[0] * theta / u[0] * (1 - mu[0] * (1 + (d[0] - (d[0])) * 2 * kappa)) - k[-1] * ((1 - xi_bar * (mu[0])) / betta - (1 - delta)) * u[0] ^ psi - (1 + (d[0] - (d[0])) * 2 * kappa) * theta * (1 - eta[0]) / eta[0] / u[0] * chi[0] = 0

	mu[0] * xi[0] + betta * gamma[1] * (c[1] - h * c[0]) ^ (-siggma) / (gamma[0] * (c[0] - h * c[-1]) ^ (-siggma)) * ((1 - delta) * Q[1] + (theta * 1 / eta[1] * y[1] / k[0] - ((1 - xi_bar * (mu[0])) / betta - (1 - delta)) * (u[1] ^ (1 + psi) - 1) / (1 + psi)) / (1 + 2 * kappa * (d[1] - (d[0]))) - theta * 1 / eta[1] * y[1] / k[0] * mu[1] - theta * (1 - eta[1]) / eta[1] / k[0] * chi[1]) - Q[0] = 0

	(zeta[0] * (1 - varrho * (invest[0] / invest[-1] - 1) ^ 2) + invest[0] * zeta[0] * (invest[0] / invest[-1] - 1) * varrho * ( - 2) * 1 / invest[-1]) * Q[0] + invest[1] * zeta[1] * varrho * ( - 2) * (invest[1] / invest[0] - 1) * ( - invest[1]) / invest[0] ^ 2 * betta * gamma[1] * (c[1] - h * c[0]) ^ (-siggma) / (gamma[0] * (c[0] - h * c[-1]) ^ (-siggma)) * Q[1] - 1 / (1 + (d[0] - (d[0])) * 2 * kappa) = 0

	k[0] = invest[0] * zeta[0] * (1 - varrho * (invest[0] / invest[-1] - 1) ^ 2) + (1 - delta) * k[-1]

	betta * omega * wopthat[1] + chat[-1] * ( - (epsilon * (upsilon_bar - 1) * (1 - betta * omega) / (upsilon_bar + epsilon * (upsilon_bar - 1)) * h * siggma / (1 - h))) + chat[0] * epsilon * (upsilon_bar - 1) * (1 - betta * omega) / (upsilon_bar + epsilon * (upsilon_bar - 1)) * siggma / (1 - h) + epsilon * (upsilon_bar - 1) * (1 - betta * omega) / (upsilon_bar + epsilon * (upsilon_bar - 1)) * (log(P[0]) - log((P[0]))) + epsilon * (upsilon_bar - 1) * (1 - betta * omega) / (upsilon_bar + epsilon * (upsilon_bar - 1)) * upsilonhat[0] + (log(n[0]) - log((n[0]))) * epsilon * (upsilon_bar - 1) * (1 - betta * omega) / (upsilon_bar + epsilon * (upsilon_bar - 1)) / epsilon + What[0] * upsilon_bar * epsilon * (upsilon_bar - 1) * (1 - betta * omega) / (upsilon_bar + epsilon * (upsilon_bar - 1)) / (epsilon * (upsilon_bar - 1)) - wopthat[0] = 0

	W[0] = (omega * W[-1] ^ (1 / (1 - upsilon[0])) + (1 - omega) * w_opt[0] ^ (1 / (1 - upsilon[0]))) ^ (1 - upsilon[0])

	y[0] * 1 / eta[0] * (1 - theta) / n[0] * (1 - mu[0] * (1 + (d[0] - (d[0])) * 2 * kappa)) - W[0] / P[0] - (1 - theta) * (1 - eta[0]) / eta[0] / n[0] * (1 + (d[0] - (d[0])) * 2 * kappa) * chi[0] = 0

	P[0] * betta * gamma[1] * (c[1] - h * c[0]) ^ (-siggma) / (gamma[0] * (c[0] - h * c[-1]) ^ (-siggma)) * R[0] * (1 + (d[0] - (d[0])) * 2 * kappa) / (1 + 2 * kappa * (d[1] - (d[0]))) / P[1] + R[0] * (1 + (d[0] - (d[0])) * 2 * kappa) * mu[0] * xi[0] / (1 + r[0]) = 1

	P[0] * (y[0] * phi * (P[0] / P[-1] - 1) / P[-1] + y[1] * phi * (P[1] / P[0] - 1) * ( - P[1]) / P[0] ^ 2 * betta * gamma[1] * (c[1] - h * c[0]) ^ (-siggma) / (gamma[0] * (c[0] - h * c[-1]) ^ (-siggma)) * (1 + (d[0] - (d[0])) * 2 * kappa) / (1 + 2 * kappa * (d[1] - (d[0])))) - (1 + (d[0] - (d[0])) * 2 * kappa) * chi[0] = 0

	V[0] = d[0] + betta * gamma[1] * (c[1] - h * c[0]) ^ (-siggma) / (gamma[0] * (c[0] - h * c[-1]) ^ (-siggma)) * V[1]

	xi[0] * (k[0] - b[0] / (P[0] * (1 + r[0]))) = y[0]

	b[0] / R[0] + P[0] * (y[0] - k[-1] * ((1 - xi_bar * (mu[0])) / betta - (1 - delta)) * (u[0] ^ (1 + psi) - 1) / (1 + psi)) - b[-1] - n[0] * W[0] - P[0] * y[0] * phi / 2 * (P[0] / P[-1] - 1) ^ 2 - (d[0] + kappa * (d[0] - (d[0])) ^ 2) * P[0] - invest[0] * P[0] = 0

	b[-1] + n[0] * W[0] + d[0] * P[0] - b[0] / (1 + r[0]) - c[0] * P[0] - T[0] = 0

	P[0] * G[0] + b[0] * (1 / R[0] - 1 / (1 + r[0])) - T[0] = 0

	rho_R * (r[-1] - (r[0])) + (1 - rho_R) * nu_1 * (P[0] - P[-1]) + ((1 - rho_R) * nu_2 + nu_3) * (y[0] - (y[0])) - nu_3 * (y[-1] - (y[0])) + var_sigma[0] - (r[0] - (r[0])) = 0

	y[0] = exp(z[0]) * (u[0] * k[-1]) ^ theta * n[0] ^ (1 - theta)

	byhat[0] = (b[-1] / (1 + r[-1]) - b[0] / (1 + r[0])) / (y[0] * P[0])

	r[0] = (R[0] - tau) / (1 - tau) - 1

	yhat[0] = log(y[0]) - log((y[0]))

	chat[0] = log(c[0]) - log((c[0]))

	ihat[0] = log(invest[0]) - log((invest[0]))

	nhat[0] = log(n[0]) - log((n[0]))

	muhat[0] = log(mu[0]) - log((mu[0]))

	upsilonhat[0] = log(upsilon[0]) - log(upsilon_bar)

	wopthat[0] = log(w_opt[0]) - log((w_opt[0]))

	What[0] = log(W[0]) - log((W[0]))

	dyhat[0] = d[0] / y[0]

	vyhat[0] = log(V[0] / (k[-1] - b[-1])) - log((V[0]) / ((k[0]) - (b[0])))

	z[0] = rho_z * z[-1] + eps_z[x]

	log(zeta[0]) = rho_zeta * log(zeta[-1]) + eps_zeta[x]

	log(gamma[0]) = rho_gamma * log(gamma[-1]) + eps_gamma[x]

	log(eta[0] / eta_bar) = rho_eta * log(eta[-1] / eta_bar) + eps_eta[x]

	log(upsilon[0] / upsilon_bar) = rho_upsilon * log(upsilon[-1] / upsilon_bar) + eps_upsilon[x]

	log(G[0] / G_bar) = rho_G * log(G[-1] / G_bar) + rho_gz * (z[0] - z[-1]) + eps_G[x]

	var_sigma[0] = rho_varsigma * var_sigma[-1] + eps_varsigma[x]

	log(xi[0] / xi_bar) = rho_xi * log(xi[-1] / xi_bar) + eps_xi[x]

	# y_obs[0] = log(y[0]) - log(y[-1])

	# c_obs[0] = log(c[0]) - log(c[-1])

	# invest_obs[0] = log(invest[0]) - log(invest[-1])

	# pi_obs[0] = log(P[0]) - log(P[-1])

	# r_obs[0] = r[0] - (r[0])

	# n_obs[0] = log(n[0]) - log(n[-1])

	# W_obs[0] = log(W[0] / P[0]) - log(W[-1] / P[-1])

	# debt_repurchase_obs[0] = byhat[0]

end


@parameters JQ_2012 begin
	betta = 0.9825

	tau = 0.35

	alppha = 16.736

	theta = 0.36

	delta = 0.025

	# xi_bar = 0.199

	BY_ratio = 3.36

	b[ss] / (1 + r[ss]) / y[ss] = BY_ratio | xi_bar

	# G[ss] = 0.179 | G_bar

	G[ss] = 0.18 * y[ss] | G_bar

	# GY_ratio = 0.18

	siggma = 1.09

	epsilon = 1.761

	h = 0.608

	omega = 0.278

	phi = 0.031

	varrho = 0.021

	psi = 0.815

	kappa = 0.426

	eta[ss] = 1.137 | eta_bar

	upsilon[ss] = 1.025 | upsilon_bar

	rho_z = 0.902

	rho_zeta = 0.922

	rho_gamma = 0.794

	rho_eta = 0.906

	rho_upsilon = 0.627

	rho_G = 0.955

	rho_varsigma = 0.203

	rho_xi = 0.969

	rho_gz = 0.509

	rho_R = 0.745

	nu_1 = 2.410

	nu_2 = 0

	nu_3 = 0.121

end

