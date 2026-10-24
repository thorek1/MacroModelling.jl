using MacroModelling

@model Ghironi_Melitz_2005 begin
	1 = Nd[0] * rho_tilde_d[0] ^ (1 - theta) + Nxbar[0] * rho_tilde_xbar[0] ^ (1 - theta)

	1 = Ndbar[0] * rho_tilde_dbar[0] ^ (1 - theta) + Nx[0] * rho_tilde_x[0] ^ (1 - theta)

	rho_tilde_d[0] = theta / (theta - 1) * w[0] / (Z[0] * ztilde_d)

	rho_tilde_dbar[0] = theta / (theta - 1) * wbar[0] / (Zbar[0] * ztilde_dbar)

	rho_tilde_x[0] = w[0] * theta / (theta - 1) * tau / (Z[0] * ztilde_x[0]) / Q[0]

	rho_tilde_xbar[0] = wbar[0] * tau * theta * Q[0] / (theta - 1) / (Zbar[0] * ztilde_xbar[0])

	dtilde[0] = dtilde_d[0] + Nx[0] / Nd[0] * dtilde_x[0]

	dtilde_bar[0] = dtilde_dbar[0] + Nxbar[0] / Ndbar[0] * dtilde_xbar[0]

	dtilde_d[0] = rho_tilde_d[0] ^ (1 - theta) / theta * C[0]

	dtilde_dbar[0] = rho_tilde_dbar[0] ^ (1 - theta) / theta * Cbar[0]

	vtilde[0] = w[0] * fe / Z[0]

	vtilde_bar[0] = wbar[0] * febar / Zbar[0]

	dtilde_x[0] = (theta - 1) * w[0] * fx / Z[0] / (k - (theta - 1))

	dtilde_xbar[0] = wbar[0] * (theta - 1) / (k - (theta - 1)) * fxbar / Zbar[0]

	Nx[0] / Nd[0] = (zmin / ztilde_x[0]) ^ k * (k / (k - (theta - 1))) ^ (k / (theta - 1))

	Nxbar[0] / Ndbar[0] = (k / (k - (theta - 1))) ^ (k / (theta - 1)) * (zminbar / ztilde_xbar[0]) ^ k

	Nd[0] = (1 - delta) * (Nd[-1] + Ne[-1])

	Ndbar[0] = (1 - delta) * (Ndbar[-1] + Nebar[-1])

	C[0] ^ (-gamma) = beta * (1 + r[0]) * C[1] ^ (-gamma)

	Cbar[0] ^ (-gamma) = beta * (1 + rbar[0]) * Cbar[1] ^ (-gamma)

	vtilde[0] = (1 - delta) * beta * (C[1] / C[0]) ^ (-gamma) * (vtilde[1] + dtilde[1])

	vtilde_bar[0] = (1 - delta) * beta * (Cbar[1] / Cbar[0]) ^ (-gamma) * (vtilde_bar[1] + dtilde_bar[1])

	C[0] = w[0] * L + Nd[0] * dtilde[0] - vtilde[0] * Ne[0]

	Cbar[0] = wbar[0] * Lbar + Ndbar[0] * dtilde_bar[0] - vtilde_bar[0] * Nebar[0]

	Q[0] = Nxbar[0] * rho_tilde_xbar[0] ^ (1 - theta) * C[0] / (Nx[0] * rho_tilde_x[0] ^ (1 - theta) * Cbar[0])

	Qtilde[0] = ((Ndbar[0] / (Ndbar[0] + Nx[0]) * TOL[0] ^ (1 - theta) + Nx[0] / (Ndbar[0] + Nx[0]) * (ztilde_d * tau / ztilde_x[0]) ^ (1 - theta)) / (Nd[0] / (Nd[0] + Nxbar[0]) + Nxbar[0] / (Nd[0] + Nxbar[0]) * (ztilde_dbar * tau * TOL[0] / ztilde_xbar[0]) ^ (1 - theta))) ^ (1 / (1 - theta))

	Qtilde[0] = Q[0] * ((Nd[0] + Nxbar[0]) / (Ndbar[0] + Nx[0])) ^ (( - 1) / (theta - 1))

	Z[0] = (1 - rho_Z) * 1.0 + rho_Z * Z[-1] + sigma__z * epsilon__z[x]

	Zbar[0] = 1.0 * (1 - rho_Zbar) + rho_Zbar * Zbar[-1] + sigma__z_bar * epsilon__z_bar[x]

	ztilde_x[0] = (theta * fx * (w[0] / Z[0]) ^ theta * (1 + (theta - 1) / (k - (theta - 1))) * Q[0] ^ (-theta) * tau ^ (theta - 1) * (theta / (theta - 1)) ^ (theta - 1) * Cbar[0] ^ (-1)) ^ (1 / (theta - 1))

	ztilde_xbar[0] = (fxbar * (1 + (theta - 1) / (k - (theta - 1))) * tau ^ (theta - 1) * theta * (theta / (theta - 1)) ^ (theta - 1) * (wbar[0] / Zbar[0]) ^ theta * Q[0] ^ theta * C[0] ^ (-1)) ^ (1 / (theta - 1))

	zx[0] = ztilde_x[0] / (k / (k - (theta - 1))) ^ (1 / (theta - 1))

	zxbar[0] = ztilde_xbar[0] / (k / (k - (theta - 1))) ^ (1 / (theta - 1))

end


@parameters Ghironi_Melitz_2005 begin
	sigma__z = 0.01

	sigma__z_bar = 0.01

	beta = 0.99

	gamma = 2.0

	delta = 0.025

	theta = 3.8

	k = 3.4

	tau = 1.3

	zmin = 1.0

	zminbar = 1.0

	fe = 1.0

	febar = 1.0

	L = 1.0

	Lbar = 1.0

	rho_Z = 0.9

	rho_Zbar = 0.9

	fx_share	=	0.235

	fx = fx_share*(1-beta*(1-delta))/(beta*(1-delta))*fe

	fxbar = fx_share*(1-beta*(1-delta))/(beta*(1-delta))*febar

	ztilde_d = (k/(k-(theta-1)))^(1/(theta-1))*zmin

	ztilde_dbar = (k/(k-(theta-1)))^(1/(theta-1))*zminbar

end

