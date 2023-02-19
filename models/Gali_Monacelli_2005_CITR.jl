@model Gali_Monacelli_2005_CITR begin
	x[0] = x[1] - sigma_a ^ (-1) * (r[0] - pih[1] - rnat[0])

	pih[0] = pih[1] * beta + x[0] * kappa_a

	rnat[0] = ( - sigma_a) * Gamma * (1 - rhoa) * a[0] + sigma_a * alpha * (Theta + Psi) * (ystar[1] - ystar[0])

	ynat[0] = Gamma * a[0] + ystar[0] * alpha * Psi

	x[0] = y[0] - ynat[0]

	y[0] = ystar[0] + sigma_a ^ (-1) * s[0]

	pi[0] = pih[0] + alpha * (s[0] - s[-1])

	s[0] = s[-1] + deprec_rate[0] - pih[0]

	y[0] = a[0] + n[0]

	nx[0] = s[0] * alpha * (omega / sigma - 1)

	y[0] = c[0] + s[0] * alpha * omega / sigma

	real_wage[0] = sigma * c[0] + n[0] * phi

	a[0] = rhoa * a[-1] + eps_a[x]

	ystar[0] = rhoy * ystar[-1] + eps_star[x]

	r[0] = pi[0] * phi_pi

end


@parameters Gali_Monacelli_2005_CITR begin
	sigma = 1

	eta = 1

	gamma = 1

	phi = 3

	theta = 0.75

	beta = 0.99

	alpha = 0.4

	phi_pi = 1.5

	rhoa = 0.9

	rhoy = 0.86

	rho = beta^(-1)-1

	omega = sigma*gamma+(1-alpha)*(sigma*eta-1)

	sigma_a = sigma/(1-alpha+alpha*omega)

	Theta = (1-alpha)*(sigma*eta-1)+sigma*gamma-1

	lambda = (1-beta*theta)*(1-theta)/theta

	kappa_a = lambda*(sigma_a+phi)

	Gamma = (1+phi)/(sigma_a+phi)

	Psi = sigma_a*(-Theta)/(sigma_a+phi)

end