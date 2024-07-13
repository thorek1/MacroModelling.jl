@model Smets_Wouters_2007_linear begin
	a[0] = calfa * rkf[0] + (1 - calfa) * wf[0]

	zcapf[0] = rkf[0] * 1 / (czcap / (1 - czcap))

	rkf[0] = wf[0] + labf[0] - kf[0]

	kf[0] = zcapf[0] + kpf[-1]

	invef[0] = qs[0] + 1 / (1 + cgamma * cbetabar) * (pkf[0] * 1 / (csadjcost * cgamma ^ 2) + invef[-1] + invef[1] * cgamma * cbetabar)

	pkf[0] = b[0] * (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) - rrf[0] + rkf[1] * (crk / (crk + (1 - ctou))) + pkf[1] * ((1 - ctou) / (crk + (1 - ctou)))

	cf[0] = b[0] + cf[-1] * chabb / cgamma / (1 + chabb / cgamma) + cf[1] * 1 / (1 + chabb / cgamma) + (labf[0] - labf[1]) * ((csigma - 1) * cwhlc / (csigma  *(1 + chabb / cgamma))) - rrf[0] * (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))

	yf[0] = g[0] + cf[0] * ccy + invef[0] * ciy + zcapf[0] * crkky

	yf[0] = cfc * (a[0] + calfa * kf[0] + (1 - calfa) * labf[0])

	wf[0] = labf[0] * csigl + cf[0] * 1 / (1 - chabb / cgamma) - cf[-1] * chabb / cgamma / (1 - chabb / cgamma)

	kpf[0] = kpf[-1] * (1 - cikbar) + invef[0] * cikbar + qs[0] * csadjcost * cgamma ^ 2 * cikbar

	mc[0] = calfa * rk[0] + (1 - calfa) * w[0] - a[0]

	zcap[0] = 1 / (czcap / (1 - czcap)) * rk[0]

	rk[0] = w[0] + lab[0] - k[0]

	k[0] = zcap[0] + kp[-1]

	inve[0] = qs[0] + 1 / (1 + cgamma * cbetabar) * (pk[0] * 1 / (csadjcost * cgamma ^ 2) + inve[-1] + inve[1] * cgamma * cbetabar)

	pk[0] = pinf[1] - r[0] + b[0] * 1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) + rk[1] * (crk / (crk + (1 - ctou))) + pk[1] * ((1 - ctou) / (crk + (1 - ctou)))

	c[0] = b[0] + c[-1] * chabb / cgamma / (1 + chabb / cgamma) + c[1] * 1 / (1 + chabb / cgamma) + 
	(lab[0] - lab[1]) * ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) - (r[0] - pinf[1]) * (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))

	y[0] = g[0] + c[0] * ccy + inve[0] * ciy + zcap[0] * crkky

	y[0] = cfc * (a[0] + calfa * k[0] + (1 - calfa) * lab[0])

	pinf[0] = spinf[0] + 1 / (1 + cindp * cgamma * cbetabar) * (cindp * pinf[-1] + pinf[1] * cgamma * cbetabar + mc[0] * (1 - cprobp) * (1 - cprobp * cgamma * cbetabar) / cprobp / (1 + (cfc - 1) * curvp))

	w[0] = sw[0] + w[-1] * 1 / (1 + cgamma * cbetabar) + w[1] * cgamma * cbetabar / (1 + cgamma * cbetabar) + pinf[-1] * cindw / (1 + cgamma * cbetabar) - pinf[0] * (1 + cindw * cgamma * cbetabar) / (1 + cgamma * cbetabar) + pinf[1] * cgamma * cbetabar / (1 + cgamma * cbetabar) + (csigl * lab[0] + c[0] * 1 / (1 - chabb / cgamma) - c[-1] * chabb / cgamma / (1 - chabb / cgamma) - w[0]) * 1 / (1 + (clandaw - 1) * curvw) * (1 - cprobw) * (1 - cprobw * cgamma * cbetabar) / (cprobw * (1 + cgamma * cbetabar))

	r[0] = pinf[0] * crpi * (1 - crr) + (1 - crr) * cry * (y[0] - yf[0]) + crdy * (y[0] - yf[0] - y[-1] + yf[-1]) + crr * r[-1] + ms[0]

	a[0] = crhoa * a[-1] + z_ea * ea[x]

	b[0] = crhob * b[-1] + z_eb * eb[x]

	g[0] = crhog * g[-1] + z_eg * eg[x] + z_ea * ea[x] * cgy

	qs[0] = crhoqs * qs[-1] + z_eqs * eqs[x]

	ms[0] = crhoms * ms[-1] + z_em * em[x]

	spinf[0] = crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]

	epinfma[0] = z_epinf * epinf[x]

	sw[0] = crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]

	ewma[0] = z_ew * ew[x]

	kp[0] = kp[-1] * (1 - cikbar) + inve[0] * cikbar + qs[0] * csadjcost * cgamma ^ 2 * cikbar

	dy[0] = ctrend + y[0] - y[-1]

	dc[0] = ctrend + c[0] - c[-1]

	dinve[0] = ctrend + inve[0] - inve[-1]

	pinfobs[0] = constepinf + pinf[0]

	robs[0] = r[0] + conster

	dwobs[0] = ctrend + w[0] - w[-1]

	labobs[0] = lab[0] + constelab

end


@parameters Smets_Wouters_2007_linear begin
	ctou = .025

	clandaw = 1.5

	cg = 0.18

	curvp = 10

	curvw = 10

	calfa = .24

	csigma = 1.5

	cfc = 1.5

	cgy = 0.51

	csadjcost = 6.0144

	chabb = 0.6361

	cprobw = 0.8087

	csigl = 1.9423

	cprobp = 0.6

	cindw = 0.3243

	cindp = 0.47

	czcap = 0.2696

	crpi = 1.488

	crr = 0.8762

	cry = 0.0593

	crdy = 0.2347

	crhoa = 0.9977

	crhob = 0.5799

	crhog = 0.9957

	crhoqs = 0.7165

	crhoms = 0

	crhopinf = 0

	crhow = 0

	cmap = 0

	cmaw = 0

	constelab = 0

	constepinf = 0.7

	constebeta = 0.7420

	ctrend = 0.3982

	z_ea	= 0.4618

	z_eb	= 1.8513

	z_eg	= 0.6090

	z_em	= 0.2397

	z_ew	= 0.2089

	z_eqs	= 0.6017

	z_epinf	= 0.1455

	cpie 	= 1 + constepinf / 100         							# gross inflation rate
	
	cgamma 	= 1 + ctrend / 100          							# gross growth rate
	
	cbeta 	= 1 / (1 + constebeta / 100)    						# discount factor
	
	clandap = cfc                									# fixed cost share/gross price markup
	
	cbetabar= cbeta * cgamma ^ (-csigma)   							# growth-adjusted discount factor in Euler equation
	
	cr 		= cpie / cbetabar  										# steady state gross real interest rate
	
	crk 	= 1 / cbetabar - (1 - ctou) 							# steady state rental rate
	
	cw 		= (calfa ^ calfa * (1 - calfa) ^ (1 - calfa) / (clandap * crk ^ calfa)) ^ (1 / (1 - calfa))	# steady state real wage
	
	cikbar 	= 1 - (1 - ctou) / cgamma								# (1-k_1) in equation LOM capital, equation (8)
	
	cik 	= cikbar * cgamma										# i_k: investment-capital ratio
	
	clk 	= (1 - calfa) / calfa * crk / cw						# labor to capital ratio
	
	cky 	= cfc * clk ^ (calfa - 1)								# k_y: steady state output ratio
	
	ciy 	= cik * cky												# investment-output ratio
	
	ccy 	= 1 - cg - cik * cky									# consumption-output ratio
	
	crkky 	= crk * cky												# z_y=R_{*}^k*k_y
	
	cwhlc 	= (1 / clandaw) * (1 - calfa) / calfa * crk * cky / ccy	# W^{h}_{*}*L_{*}/C_{*} used in c_2 in equation (2)
	
	conster = (cr - 1) * 100										# steady state federal funds rate ($\bar r$)

end
