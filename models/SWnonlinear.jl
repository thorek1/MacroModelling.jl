@model SWnonlinear begin
	y[0] = c[0] + inve[0] + y[ss] * gy[0] + afunc[0] * kp[-1] / (1 + ctrend / 100)

	y[0] * (pdot[0] + curvp * (1 - cfc) / cfc) / (1 + curvp * (1 - cfc) / cfc) = a[0] * k[0] ^ calfa * lab[0] ^ (1 - calfa) - (cfc - 1) * y[ss]

	k[0] = kp[-1] * zcap[0] / (1 + ctrend / 100)

	kp[0] = inve[0] * qs[0] * (1 - Sfunc[0]) + kp[-1] * (1 - ctou) / (1 + ctrend / 100)

	pdot[0] = (1 - cprobp) * (Pratio[0] / dp[0]) ^ (( - cfc) * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)) + pdot[-1] * cprobp * (dp[-1] / dp[0] * pinf[-1] ^ cindp * pinf[ss] ^ (1 - cindp) / pinf[0]) ^ (( - cfc) * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1))

	wdot[0] = (1 - cprobw) * (wnew[0] / dw[0]) ^ (( - clandaw) * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1)) + wdot[-1] * cprobw * (dw[-1] / dw[0] * pinf[-1] ^ cindw * pinf[ss] ^ (1 - cindw) / pinf[0]) ^ (( - clandaw) * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1))

	1 = (1 - cprobp) * (Pratio[0] / dp[0]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1)) + cprobp * (dp[-1] / dp[0] * pinf[-1] ^ cindp * pinf[ss] ^ (1 - cindp) / pinf[0]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1))

	1 = (1 - cprobw) * (wnew[0] / dw[0]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1)) + cprobw * (dw[-1] / dw[0] * pinf[-1] ^ cindw * pinf[ss] ^ (1 - cindw) / pinf[0]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1))

	1 = dp[0] * (1 + pdotl[0] * curvp * (1 - cfc) / cfc) / (1 + curvp * (1 - cfc) / cfc)

	w[0] = dw[0] * (1 + curvw * (1 - clandaw) / clandaw * wdotl[0]) / (1 + curvw * (1 - clandaw) / clandaw)

	pdotl[0] = (1 - cprobp) * Pratio[0] / dp[0] + cprobp * dp[-1] / dp[0] * pinf[-1] ^ cindp * pinf[ss] ^ (1 - cindp) / pinf[0] * pdotl[-1]

	wdotl[0] = (1 - cprobw) * wnew[0] / dw[0] + cprobw * dw[-1] / dw[0] * pinf[-1] ^ cindw * pinf[ss] ^ (1 - cindw) / pinf[0] * wdotl[-1]

	xi[0] = exp((csigma - 1) / (1 + csigl) * (lab[0] * (curvw * (1 - clandaw) / clandaw + wdot[0]) / (1 + curvw * (1 - clandaw) / clandaw)) ^ (1 + csigl)) * (c[0] - c[-1] * chabb / (1 + ctrend / 100)) ^ (-csigma)

	1 = qs[0] * pk[0] * (1 - Sfunc[0] - (1 + ctrend / 100) * inve[0] * SfuncD[0] / inve[-1]) + SfuncD[1] * xi[1] / xi[0] * qsaux[0] * pk[1] * ((1 + ctrend / 100) * inve[1] / inve[0]) ^ 2 * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	xi[0] = xi[1] * b[0] * r[0] * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) / pinf[1]

	rk[0] = afuncD[0]

	pk[0] = (rk[1] * zcap[1] - afunc[1] + (1 - ctou) * pk[1]) * xi[1] * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) / xi[0]

	k[0] = lab[0] * w[0] * calfa / (1 - calfa) / rk[0]

	mc[0] = w[0] ^ (1 - calfa) * rk[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))

	wnew[0] * gamw1[0] * (1 + curvw * (1 - clandaw)) / (1 + curvw * (1 - clandaw) / clandaw) = clandaw * gamw2[0] + gamw3[0] * curvw * (1 - clandaw) / clandaw * (clandaw - 1) / (1 + curvw * (1 - clandaw) / clandaw) * wnew[0] ^ (1 + clandaw * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1))

	gamw1[0] = lab[0] * dw[0] ^ (clandaw * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1)) + gamw1[1] * (pinf[ss] ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1)) * xi[1] / xi[0] * (1 + ctrend / 100) * cprobw * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	gamw2[0] = (c[0] - c[-1] * chabb / (1 + ctrend / 100)) * lab[0] * sw[0] * dw[0] ^ (clandaw * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1)) * (lab[0] * (curvw * (1 - clandaw) / clandaw + wdot[0]) / (1 + curvw * (1 - clandaw) / clandaw)) ^ csigl + gamw2[1] * (pinf[ss] ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - clandaw) * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1)) * xi[1] / xi[0] * (1 + ctrend / 100) * cprobw * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	gamw3[0] = lab[0] + gamw3[1] * pinf[ss] ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1] * xi[1] / xi[0] * (1 + ctrend / 100) * cprobw * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	Pratio[0] * gam1[0] * (1 + curvp * (1 - cfc)) / (1 + curvp * (1 - cfc) / cfc) = cfc * gam2[0] + gam3[0] * (cfc - 1) * curvp * (1 - cfc) / cfc / (1 + curvp * (1 - cfc) / cfc) * Pratio[0] ^ (1 + cfc * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1))

	gam1[0] = y[0] * dp[0] ^ (cfc * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)) + gam1[1] * xi[1] / xi[0] * (1 + ctrend / 100) * cprobp * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) * (pinf[ss] ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1))

	gam2[0] = y[0] * mc[0] * spinf[0] * dp[0] ^ (cfc * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)) + gam2[1] * xi[1] / xi[0] * (1 + ctrend / 100) * cprobp * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) * (pinf[ss] ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1]) ^ (( - cfc) * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1))

	gam3[0] = y[0] + gam3[1] * pinf[ss] ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1] * xi[1] / xi[0] * (1 + ctrend / 100) * cprobp * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	qsaux[0] = qs[1]

	r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / (constepinf/100)) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0]

	# r[0] = max(1.00025,r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / pinf[ss]) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])

	afunc[0] = rk[ss] * 1 / (czcap / (1 - czcap)) * (exp(czcap / (1 - czcap) * (zcap[0] - 1)) - 1)

	afuncD[0] = rk[ss] * exp(czcap / (1 - czcap) * (zcap[0] - 1))

	Sfunc[0] = csadjcost / 2 * ((1 + ctrend / 100) * inve[0] / inve[-1] - (1 + ctrend / 100)) ^ 2

	SfuncD[0] = csadjcost * ((1 + ctrend / 100) * inve[0] / inve[-1] - (1 + ctrend / 100))

	a[0] = 1 - crhoa + crhoa * a[-1] + ea[x] / 100

	b[0] = 1 - crhob + crhob * b[-1] + eb[x] * ( - (((1 - chabb / (1 + ctrend / 100)) / (csigma * (1 + chabb / (1 + ctrend / 100)))) ^ (-1))) / 100

	gy[0] - cg = crhog * (gy[-1] - cg) + egy[x] / 100 + ea[x] * cgy / 100

	qs[0] = 1 - crhoqs + crhoqs * qs[-1] + eqs[x] * csadjcost * (1 + ctrend / 100) ^ 2 * (1 + 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (1 - csigma)) / 100

	ms[0] = 1 - crhoms + crhoms * ms[-1] + ems[x] / 100

	spinf[0] = 1 - crhopinf + crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]

	epinfma[0] = epinf[x] * 1 / (1 / (1 + cindp * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) * (1 - cprobp) * (1 - cprobp * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) / cprobp / (1 + curvp * (cfc - 1))) / 100

	sw[0] = 1 - crhow + crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]

	ewma[0] = ew[x] * 1 / (1 / (1 + curvw * (clandaw - 1)) * (1 - cprobw) * (1 - cprobw * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) / (cprobw * (1 + (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)))) / 100

	yflex[0] = cflex[0] + inveflex[0] + gy[0] * yflex[ss] + afuncflex[0] * kpflex[-1] / (1 + ctrend / 100)

	yflex[0] = a[0] * kflex[0] ^ calfa * labflex[0] ^ (1 - calfa) - (cfc - 1) * yflex[ss]

	kflex[0] = kpflex[-1] * zcapflex[0] / (1 + ctrend / 100)

	kpflex[0] = inveflex[0] * qs[0] * (1 - Sfuncflex[0]) + kpflex[-1] * (1 - ctou) / (1 + ctrend / 100)

	xiflex[0] = exp((csigma - 1) / (1 + csigl) * labflex[0] ^ (1 + csigl)) * (cflex[0] - cflex[-1] * chabb / (1 + ctrend / 100)) ^ (-csigma)

	1 = qs[0] * pkflex[0] * (1 - Sfuncflex[0] - (1 + ctrend / 100) * inveflex[0] * SfuncDflex[0] / inveflex[-1]) + SfuncDflex[1] * qsaux[0] * xiflex[1] / xiflex[0] * pkflex[1] * ((1 + ctrend / 100) * inveflex[1] / inveflex[0]) ^ 2 * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	xiflex[0] = xiflex[1] * b[0] * rrflex[0] * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	rkflex[0] = afuncDflex[0]

	pkflex[0] = (rkflex[1] * zcapflex[1] - afuncflex[1] + (1 - ctou) * pkflex[1]) * xiflex[1] * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) / xiflex[0]

	kflex[0] = labflex[0] * calfa / (1 - calfa) * wflex[0] / rkflex[0]

	mcflex = wflex[0] ^ (1 - calfa) * rkflex[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))

	wflex[0] * (1 + curvw * (1 - clandaw)) / (1 + curvw * (1 - clandaw) / clandaw) = sw[ss] * (labflex[0] ^ csigl * clandaw * (cflex[0] - cflex[-1] * chabb / (1 + ctrend / 100)) + wflex[0] * curvw * (1 - clandaw) / clandaw * (clandaw - 1) / (1 + curvw * (1 - clandaw) / clandaw))

	# (1 + curvp * (1 - cfc)) / (1 + curvp * (1 - cfc) / cfc) = spinf[ss] * cfc * mcflex[0] + spinf[ss] * (cfc - 1) * curvp * (1 - cfc) / cfc / (1 + curvp * (1 - cfc) / cfc)

	afuncflex[0] = rkflex[ss] * 1 / (czcap / (1 - czcap)) * (exp(czcap / (1 - czcap) * (zcapflex[0] - 1)) - 1)

	afuncDflex[0] = rkflex[ss] * exp(czcap / (1 - czcap) * (zcapflex[0] - 1))

	Sfuncflex[0] = csadjcost / 2 * ((1 + ctrend / 100) * inveflex[0] / inveflex[-1] - (1 + ctrend / 100)) ^ 2

	SfuncDflex[0] = csadjcost * ((1 + ctrend / 100) * inveflex[0] / inveflex[-1] - (1 + ctrend / 100))

	# ygap[0] = 100 * log(y[0] / yflex[0])

	# dy[0] = ctrend + 100 * (y[0] / y[-1] - 1)

	# dc[0] = ctrend + 100 * (c[0] / c[-1] - 1)

	# dinve[0] = ctrend + 100 * (inve[0] / inve[-1] - 1)

	# pinfobs[0] = 100 * (pinf[0] - pinf[ss]) + constepinf

	# robs[0] = 100 * (r[0] - 1)

	# dwobs[0] = ctrend + 100 * (w[0] / w[-1] - 1)

	# labobs[0] = 100 * (lab[0] / lab[ss] - 1)

end


@parameters SWnonlinear begin
	mcflex = mc[ss] | mcflex
	# ctou = 0.025

	# clandaw = 1.5

	# clandap = 1.61        

	# czcap = 0.54

	# cg = 0.18

	# curvp = 10

	# curvw = 10

	# csadjcost = 5.48

	# csigma = 1.39

	# chabb = 0.71

	# cprobw = 0.73

	# csigl = 1.92

	# cprobp = 0.65

	# cindw = 0.59

	# cindp = 0.22

	# cfc = 1.61

	# constepinf = 0.8099999999999996

	# constebeta = 0.16

	# ctrend = 0.4299999999999972

	# calfa = .19

	# crpi = 2.03

	# crr = 0.81

	# cry = 0.08

	# crdy = 0.22

	# crhoa = 0.95

	# crhob = 0.18

	# crhog = 0.97

	# cgy = 0.52

	# crhoqs = 0.71

	# crhoms = 0.2

	# crhopinf = 0.90

	# cmap = 0.74

	# crhow = 0.97

	# cmaw = 0.88

	# constelab=0

	ctou = 0.025

	# # czcap = 0.54

	cg = 0.18

	# # cfc = 1.61

	# # curvp = 10

	clandaw = 1.5

	curvw = 10

	# csadjcost = 5.48

	# csigma = 1.39

	# chabb = 0.71

	# cprobw = 0.73

	# csigl = 1.92

	# cprobp = 0.65

	# cindw = 0.59

	# cindp = 0.22

	# constepinf = 0.8099999999999996

	# constebeta = 0.16

	# ctrend = 0.4299999999999972

	# calfa = .19

	# crpi = 2.03

	# crr = 0.81

	# cry = 0.08

	# crdy = 0.22

	# crhoa = 0.95

	# crhob = 0.18

	# crhog = 0.97

	# cgy = 0.52

	# crhoqs = 0.71

	# crhoms = 0.2

	# crhopinf = 0.90

	# cmap = 0.74

	# crhow = 0.97

	# cmaw = 0.88

	crhoa = 0.95827

	crhob = 0.22137

	crhog = 0.97391

	crhoqs = 0.70524

	crhoms = 0.11421

	crhopinf = 0.83954

	crhow = 0.9745

	cmap = 0.69414

	cmaw = 0.93617

	csadjcost = 5.5811

	csigma = 1.4103

	chabb = 0.68049

	cprobw = 0.80501

	csigl = 2.2061

	cindw = 0.56351

	cindp = 0.24165

	czcap = 0.49552

	cfc = 1.3443

	crpi = 1.931

	crr = 0.82512

	cry = 0.097844

	crdy = 0.25114

	constepinf = 0.8731

	constebeta = 0.12575

	ctrend = 0.4419

	cgy = 0.53817

	calfa = 0.18003

	curvp = 64.5595

	cprobp = 0.667
	# gamw1 > 6
	# gamw2 > 6
	# gamw3 > 6
end
