using MacroModelling

@model AnSchorfheide_Gaussian begin
	1 = exp(( - tau) * c[1] + tau * c[0] + R[0] - z[1] - p[1])

	(exp(tau * c[0]) - 1) * (1 - nu) / nu / (tau * (1 - nu) / nu / kap / exp(pist / 400) ^ 2) / exp(pist / 400) ^ 2 = (exp(p[0]) - 1) * (0.5 / nu + exp(p[0]) * (1 - 0.5 / nu)) - exp(p[1] + ( - tau) * c[1] + tau * c[0] + y[1] - y[0]) * (exp(p[1]) - 1) * 1 / exp(rrst / 400)

	# exp(c[0] - y[0]) = 1 - (exp(p[0]) - 1) ^ 2 * tau * (1 - nu) / nu / kap * 1 / cyst / 2

	# R[0] = rhor * R[-1] + p[0] * (1 - rhor) * psi1 + (1 - rhor) * psi2 * (y[0]) + sig_r * e_r[x]

	exp(c[0] - y[0]) = exp(( - g[0])) - (exp(p[0]) - 1) ^ 2 * tau * (1 - nu) / nu / kap * 1 / cyst / 2

	R[0] = rhor * R[-1] + p[0] * (1 - rhor) * psi1 + (1 - rhor) * psi2 * (y[0] - g[0]) + sig_r * e_r[x]

	g[0] = rhog * g[-1] + sig_g * e_g[x]

	z[0] = rhoz * z[-1] + sig_z * e_z[x]

	# dy[0] = y[0] - y[-1]

	# YGR[0] = gamst + 100 * (z[0] + dy[0])

	# INFL[0] = pist + 400 * p[0]

	# INT[0] = pist + rrst + gamst * 4 + 400 * R[0]

end


@parameters AnSchorfheide_Gaussian begin
	tau = 2.0000

	nu = 0.1000

	kap = 0.3300

	cyst = 0.8500

	psi1 = 1.5000

	psi2 = 0.1250

	rhor = 0.7500

	rhog = 0.9500

	rhoz = 0.9000

	rrst = 1.0000

	pist = 3.2000

	gamst = 0.5500

	sig_r = 0.002

	sig_g = 0.006

	sig_z = 0.003

end

