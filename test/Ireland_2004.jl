using MacroModelling

@model Ireland_2004 begin
	a[0] = rho__a * a[-1] + sigma__a * epsilon__a[x]

	e[0] = rho__e * e[-1] + sigma__e * epsilon__e[x]

	x[0] = alpha__x * x[-1] + (1 - alpha__x) * x[1] - (rhat[0] - pi_hat[1]) + a[0] * (1 - omega) * (1 - rho__a)

	pi_hat[0] = beta * (alpha__p * pi_hat[-1] + pi_hat[1] * (1 - alpha__p)) + x[0] * psi - e[0]

	x[0] = yhat[0] - a[0] * omega

	ghat[0] = yhat[0] + sigma__z * epsilon__z[x] - yhat[-1]

	rhat[0] - rhat[-1] = pi_hat[0] * rho__p + ghat[0] * rho__g + x[0] * rho__x + sigma__r * epsilon__r[x]

end


@parameters Ireland_2004 begin
	beta = 0.99

	psi = 0.1

	omega = 0.0581

	alpha__x = 1.0e-5

	alpha__p = 1.0e-5

	rho__p = 0.3866

	rho__g = 0.396

	rho__x = 0.1654

	rho__a = 0.9048

	rho__e = 0.9907

	sigma__r = 0.0028

	sigma__a = 0.0302

	sigma__e = 0.0002

	sigma__z = 0.0089

end

