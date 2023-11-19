@model Guerrieri_Iacoviello_2017 begin
	c[0] + c1[0] + ik[0] = y[0]

	uc[0] = BETA * r[0] / dp[1] * uc[1]

	uc[0] * w[0] / xw[0] = az[0] * n[0] ^ ETA

	uc[0] * q[0] = uh[0] + uc[1] * BETA * q[1]

	c1[0] + q[0] * (h1[0] - h1[-1]) + r[-1] * b[-1] / dp[0] = w1[0] * n1[0] + b[0] + INDTR * log(ap[0])

	uc1[0] * (1 - lm[0]) = BETA1 * (r[0] / dp[1] - RHOD * lm[1] / dp[1]) * uc1[1]

	w1[0] * uc1[0] / xw1[0] = az[0] * n1[0] ^ ETA

	q[0] * uc1[0] = uh1[0] + uc1[1] * q[1] * BETA1 + q[0] * uc1[0] * lm[0] * (1 - RHOD) * M

	y[0] = n[0] ^ ((1 - ALPHA) * (1 - SIGMA)) * n1[0] ^ ((1 - ALPHA) * SIGMA) * k[-1] ^ ALPHA

	y[0] * (1 - ALPHA) * (1 - SIGMA) = n[0] * w[0] * xp[0]

	y[0] * (1 - ALPHA) * SIGMA = n1[0] * w1[0] * xp[0]

	log(dp[0] / PIBAR) - LAGP * log(dp[-1] / PIBAR) = BETA * (log(dp[1] / PIBAR) - log(dp[0] / PIBAR) * LAGP) - (1 - TETAP) * (1 - BETA * TETAP) / TETAP * log(xp[0] / XP_SS) + log(ap[0]) * (1 - INDTR)

	log(dw[0] / PIBAR) - LAGW * log(dw[-1] / PIBAR) = BETA * (log(dw[1] / PIBAR) - log(dw[0] / PIBAR) * LAGW) - (1 - TETAW) * (1 - BETA * TETAW) / TETAW * log(xw[0] / XW_SS) + log(aw[0])

	log(dw1[0] / PIBAR) - LAGW * log(dw1[-1] / PIBAR) = log(aw[0]) + BETA * (log(dw1[1] / PIBAR) - LAGW * log(dw1[0] / PIBAR)) - (1 - TETAW) * (1 - BETA * TETAW) / TETAW * log(xw1[0] / XW_SS)

	log(rnot[0]) = TAYLOR_R * log(r[-1]) + (1 - TAYLOR_R) * TAYLOR_P * (log(dp[0] / PIBAR) * 0.25 + 0.25 * log(dp[-1] / PIBAR) + 0.25 * log(dp[-2] / PIBAR) + 0.25 * log(dp[-3] / PIBAR)) + (1 - TAYLOR_R) * TAYLOR_Y * log(y[0] / lly) + (1 - TAYLOR_R) * TAYLOR_Q / 4 * log(q[0] / q[-1]) + (1 - TAYLOR_R) * log(PIBAR / BETA) + log(arr[0])

	uc[0] = (1 - EC) / (1 - BETA * EC) * (az[0] / (c[0] - EC * c[-1]) - BETA * EC * az[1] / (c[1] - c[0] * EC))

	uc1[0] = (1 - EC) / (1 - BETA1 * EC) * (az[0] / (c1[0] - EC * c1[-1]) - az[1] * BETA1 * EC / (c1[1] - c1[0] * EC))

	uh[0] = (1 - EH) / (1 - BETA * EH) * JEI * (az[0] * aj[0] / (1 - h1[0] - EH * (1 - h1[-1])) - az[1] * BETA * EH * aj[1] / (1 - h1[1] - EH * (1 - h1[0])))

	uh1[0] = JEI * (1 - EH) / (1 - BETA1 * EH) * (az[0] * aj[0] / (h1[0] - h1[-1] * EH) - aj[1] * az[1] * BETA1 * EH / (h1[1] - h1[0] * EH))

	uc[0] * qk[0] * (1 - PHIK * (ik[0] - ik[-1]) / llik) = uc[0] - PHIK * BETA * uc[1] * qk[1] * (ik[1] - ik[0]) / llik

	uc[0] * qk[0] / ak[0] = BETA * uc[1] * (rk[1] + qk[1] * (1 - DK) / ak[1])

	k[0] / ak[0] = ik[0] + k[-1] * (1 - DK) / ak[0]

	y[0] * ALPHA = k[-1] * xp[0] * rk[0]

	dw[0] = w[0] * dp[0] / w[-1]

	dw1[0] = dp[0] * w1[0] / w1[-1]

	log(aj[0]) = RHO_J * log(aj[-1]) + z_j[0]

	z_j[0] = RHO_J2 * z_j[-1] + eps_j[x]

	log(ak[0]) = RHO_K * log(ak[-1]) + STD_K * eps_k[x]

	log(ap[0]) = RHO_P * log(ap[-1]) + STD_P * eps_p[x]

	log(aw[0]) = RHO_W * log(aw[-1]) + STD_W * eps_w[x]

	log(arr[0]) = RHO_R * log(arr[-1]) + STD_R * eps_r[x]

	log(az[0]) = RHO_Z * log(az[-1]) + STD_Z * eps_z[x]

	0 = min(bnot[0] - b[0], lm[0])
	# bnot[0] = b[0]

	bnot[0] = h1[0] * q[0] * (1 - RHOD) * M + b[-1] * RHOD / dp[0]

	maxlev[0] = b[0] - bnot[0]

	r[0] = max(RBAR, rnot[0])
	# r[0] = rnot[0]

end


@parameters Guerrieri_Iacoviello_2017 begin
	RBAR = 1

	BETA = 0.995

	BETA1 = 0.9921849949330452

	EC = 0.6841688730310923

	EH = 0.8798650668795864

	ETA = 1

	JEI = 0.04

	M = 0.9

	ALPHA = 0.3

	PHIK = 4.120924218703865

	DK = 0.025

	LAGP = 0

	LAGW = 0

	PIBAR = 1.005

	INDTR = 0

	SIGMA = 0.5012798413194606

	TAYLOR_P = 1.719559906725518

	TAYLOR_Q = 0

	TAYLOR_R = 0.5508743735338286

	TAYLOR_Y = 0.09436959071018983

	TETAP = 0.9182319022631061

	TETAW = 0.9162909334165672

	XP_SS = 1.2

	XW_SS = 1.2

	RHO_J = 0.983469150669198

	RHO_K = 0.7859395713107814

	RHO_P = 0

	RHO_R = 0.623204934949152

	RHO_W = 0

	RHO_Z = 0.7555575007590176

	STD_J = 0.07366860797541266

	STD_K = 0.03601489154765812

	STD_P = 0.002964296803248907

	STD_R = 0.001315097718876929

	STD_W = 0.00996414482032244

	STD_Z = 0.01633680112129254

	RHO_J2 = 0

	RHOD = 0.6945068431131589

	ITAYLOR_W = 0

	llr = 1 / BETA

	llrk = llr - (1-DK)

	llxp = XP_SS

	llxw = XW_SS

	llxw1 = XW_SS

	lllm = (1 - BETA1/BETA) / (1 - BETA1*RHOD/PIBAR)

	QHTOC = JEI/(1-BETA)

	QH1TOC1 = JEI/(1-BETA1-lllm*M*(1-RHOD))

	KTOY = ALPHA/(llxp*llrk)

	BTOQH1 = M*(1-RHOD)/(1-RHOD/PIBAR)

	C1TOY = (1-ALPHA)*SIGMA/(1+(1/BETA-1)*BTOQH1*QH1TOC1)*(1/llxp)

	CTOY = (1-C1TOY-DK*KTOY)

	lln = ((1-SIGMA)*(1-ALPHA)/(llxp*llxw*CTOY))^(1/(1+ETA))

	lln1 = (SIGMA*(1-ALPHA)/(llxp*llxw1*C1TOY))^(1/(1+ETA))

	lly = KTOY^(ALPHA/(1-ALPHA))*lln^(1-SIGMA)*lln1^SIGMA

	llctot = lly-DK*KTOY*lly

	llik = KTOY*DK*lly

	llk = KTOY*lly 

	llq = QHTOC*CTOY*lly + QH1TOC1*C1TOY*lly

end
