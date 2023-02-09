using MacroModelling

@model GM_2005 begin
	x[0] = x[1] - (r[0] - pih[1] - rnat[0]) * (sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1)))) ^ (-1)

	pih[0] = beta * pih[1] + x[0] * (1 - beta * theta) * (1 - theta) / theta * (phi + sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1))))

	rnat[0] = a[0] * (1 - rhoa) * (1 + phi) / (phi + sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1)))) * ( - (sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1))))) + (ystar[1] - ystar[0]) * alpha * sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1))) * ((1 - alpha) * (sigma * eta - 1) + sigma * gamma - 1 + sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1))) * ( - ((1 - alpha) * (sigma * eta - 1) + sigma * gamma - 1)) / (phi + sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1)))))

	ynat[0] = a[0] * (1 + phi) / (phi + sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1)))) + ystar[0] * alpha * sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1))) * ( - ((1 - alpha) * (sigma * eta - 1) + sigma * gamma - 1)) / (phi + sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1))))

	x[0] = y[0] - ynat[0]

	y[0] = ystar[0] + s[0] * (sigma / (1 - alpha + alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1)))) ^ (-1)

	pi[0] = pih[0] + alpha * (s[0] - s[-1])

	s[0] = s[-1] + e[0] - e[-1] - pih[0]

	y[0] = a[0] + n[0]

	nx[0] = s[0] * alpha * ((sigma * gamma + (1 - alpha) * (sigma * eta - 1)) / sigma - 1)

	y[0] = c[0] + s[0] * alpha * (sigma * gamma + (1 - alpha) * (sigma * eta - 1)) / sigma

	real_wage[0] = sigma * c[0] + phi * n[0]

	a[0] = rhoa * a[-1] + eps_a[x]

	ystar[0] = rhoy * ystar[-1] + eps_star[x]

	r[0] = pi[0] * phi_pi

	pi[0] = p[0] - p[-1]

	pih[0] = ph[0] - ph[-1]

	deprec_rate[0] = e[0] - e[-1]

end


@parameters GM_2005 begin

	sigma = 1

	eta = 1

	gamma = 1

	phi = 3

	epsilon = 6

	theta = 0.75

	beta = 0.99

	alpha = 0.4

	phi_pi = 1.5

	rhoa = 0.9

	rhoy = 0.86

end

