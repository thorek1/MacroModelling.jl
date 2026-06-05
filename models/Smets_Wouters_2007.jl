@model Smets_Wouters_2007 begin
	y[0] = c[0] + inve[0] + y[ss] * gy[0] + afunc[0] * kp[-1] / cgamma

	y[0] * (pdot[0] + curvP) / (1 + curvP) = a[0] * k[0] ^ calfa * lab[0] ^ (1 - calfa) - (cfc - 1) * y[ss]

	k[0] = kp[-1] * zcap[0] / cgamma

	kp[0] = inve[0] * qs[0] * (1 - Sfunc[0]) + kp[-1] * (1 - ctou) / cgamma

	pdot[0] = (1 - cprobp) * (Pratio[0] / dp[0]) ^ (( - cfc) * (1 + curvP) / (cfc - 1)) + pdot[-1] * cprobp * (dp[-1] / dp[0] * pinf[-1] ^ cindp * cpie ^ (1 - cindp) / pinf[0]) ^ (( - cfc) * (1 + curvP) / (cfc - 1))

	wdot[0] = (1 - cprobw) * (wnew[0] / dw[0]) ^ (( - clandaw) * (1 + curvW) / (clandaw - 1)) + wdot[-1] * cprobw * (dw[-1] / dw[0] * pinf[-1] ^ cindw * cpie ^ (1 - cindw) / pinf[0]) ^ (( - clandaw) * (1 + curvW) / (clandaw - 1))

	1 = (1 - cprobp) * (Pratio[0] / dp[0]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1)) + cprobp * (dp[-1] / dp[0] * pinf[-1] ^ cindp * cpie ^ (1 - cindp) / pinf[0]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1))

	1 = (1 - cprobw) * (wnew[0] / dw[0]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1)) + cprobw * (dw[-1] / dw[0] * pinf[-1] ^ cindw * cpie ^ (1 - cindw) / pinf[0]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1))

	1 = dp[0] * (1 + pdotl[0] * curvP) / (1 + curvP)

	w[0] = dw[0] * (1 + curvW * wdotl[0]) / (1 + curvW)

	pdotl[0] = (1 - cprobp) * Pratio[0] / dp[0] + cprobp * dp[-1] / dp[0] * pinf[-1] ^ cindp * cpie ^ (1 - cindp) / pinf[0] * pdotl[-1]

	wdotl[0] = (1 - cprobw) * wnew[0] / dw[0] + cprobw * dw[-1] / dw[0] * pinf[-1] ^ cindw * cpie ^ (1 - cindw) / pinf[0] * wdotl[-1]

	xi[0] = exp((csigma - 1) / (1 + csigl) * (lab[0] * (curvW + wdot[0]) / (1 + curvW)) ^ (1 + csigl)) * (c[0] - c[-1] * chabb / cgamma) ^ (-csigma)

	1 = qs[0] * pk[0] * (1 - Sfunc[0] - cgamma * inve[0] * SfuncD[0] / inve[-1]) + SfuncD[1] * xi[1] / xi[0] * qsaux[0] * pk[1] * (cgamma * inve[1] / inve[0]) ^ 2 * cbetabar

	xi[0] = xi[1] * b[0] * r[0] * cbetabar / pinf[1]

	rk[0] = afuncD[0]

	pk[0] = (rk[1] * zcap[1] - afunc[1] + (1 - ctou) * pk[1]) * xi[1] * cbetabar / xi[0]

	k[0] = lab[0] * w[0] * calfa / (1 - calfa) / rk[0]

	mc[0] = w[0] ^ (1 - calfa) * rk[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))

	wnew[0] * gamw1[0] * (1 + curvw * (1 - clandaw)) / (1 + curvW) = clandaw * gamw2[0] + gamw3[0] * curvW * (clandaw - 1) / (1 + curvW) * wnew[0] ^ (1 + clandaw * (1 + curvW) / (clandaw - 1))

	gamw1[0] = lab[0] * dw[0] ^ (clandaw * (1 + curvW) / (clandaw - 1)) + gamw1[1] * (cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1)) * xi[1] / xi[0] * cgamma * cprobw * cbetabar

	gamw2[0] = (c[0] - c[-1] * chabb / cgamma) * lab[0] * sw[0] * dw[0] ^ (clandaw * (1 + curvW) / (clandaw - 1)) * (lab[0] * (curvW + wdot[0]) / (1 + curvW)) ^ csigl + gamw2[1] * (cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - clandaw) * (1 + curvW) / (clandaw - 1)) * xi[1] / xi[0] * cgamma * cprobw * cbetabar

	gamw3[0] = lab[0] + gamw3[1] * cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1] * xi[1] / xi[0] * cgamma * cprobw * cbetabar

	Pratio[0] * gam1[0] * (1 + curvp * (1 - cfc)) / (1 + curvP) = cfc * gam2[0] + gam3[0] * (cfc - 1) * curvP / (1 + curvP) * Pratio[0] ^ (1 + cfc * (1 + curvP) / (cfc - 1))

	gam1[0] = y[0] * dp[0] ^ (cfc * (1 + curvP) / (cfc - 1)) + gam1[1] * xi[1] / xi[0] * cgamma * cprobp * cbetabar * (cpie ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1))

	gam2[0] = y[0] * mc[0] * spinf[0] * dp[0] ^ (cfc * (1 + curvP) / (cfc - 1)) + gam2[1] * xi[1] / xi[0] * cgamma * cprobp * cbetabar * (cpie ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1]) ^ (( - cfc) * (1 + curvP) / (cfc - 1))

	gam3[0] = y[0] + gam3[1] * cpie ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1] * xi[1] / xi[0] * cgamma * cprobp * cbetabar

	qsaux[0] = qs[1]

	# r[0] = max(1.00025,r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / pinfss) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])
	
	r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0]

	afunc[0] = rk[ss] * 1 / cZcap * (exp(cZcap * (zcap[0] - 1)) - 1)

	afuncD[0] = rk[ss] * exp(cZcap * (zcap[0] - 1))

	Sfunc[0] = csadjcost / 2 * (cgamma * inve[0] / inve[-1] - cgamma) ^ 2

	SfuncD[0] = csadjcost * (cgamma * inve[0] / inve[-1] - cgamma)

	a[0] = 1 - crhoa + crhoa * a[-1] + z_ea / 100 * ea[x]

	b[0] = 1 - crhob + crhob * b[-1] +  z_eb / 100 * SCALE1_eb * eb[x]

	gy[0] - cg = crhog * (gy[-1] - cg) + z_eg / 100 * eg[x] + z_ea / 100 * ea[x] * cgy

	qs[0] = 1 - crhoqs + crhoqs * qs[-1] + z_eqs / 100 * SCALE1_eqs * eqs[x]

	ms[0] = 1 - crhoms + crhoms * ms[-1] + z_em / 100 * em[x]

	spinf[0] = 1 - crhopinf + crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]

	epinfma[0] = z_epinf / 100 * SCALE1_epinf * epinf[x]

	sw[0] = 1 - crhow + crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]

	ewma[0] = z_ew / 100 * SCALE1_ew * ew[x]

	yflex[0] = cflex[0] + inveflex[0] + gy[0] * yflex[ss] + afuncflex[0] * kpflex[-1] / cgamma

	yflex[0] = a[0] * kflex[0] ^ calfa * labflex[0] ^ (1 - calfa) - (cfc - 1) * yflex[ss]

	kflex[0] = kpflex[-1] * zcapflex[0] / cgamma

	kpflex[0] = inveflex[0] * qs[0] * (1 - Sfuncflex[0]) + kpflex[-1] * (1 - ctou) / cgamma

	xiflex[0] = exp((csigma - 1) / (1 + csigl) * labflex[0] ^ (1 + csigl)) * (cflex[0] - cflex[-1] * chabb / cgamma) ^ (-csigma)

	1 = qs[0] * pkflex[0] * (1 - Sfuncflex[0] - cgamma * inveflex[0] * SfuncDflex[0] / inveflex[-1]) + SfuncDflex[1] * qsaux[0] * xiflex[1] / xiflex[0] * pkflex[1] * (cgamma * inveflex[1] / inveflex[0]) ^ 2 * cbetabar

	xiflex[0] = xiflex[1] * b[0] * rrflex[0] * cbetabar

	rkflex[0] = afuncDflex[0]

	pkflex[0] = (rkflex[1] * zcapflex[1] - afuncflex[1] + (1 - ctou) * pkflex[1]) * xiflex[1] * cbetabar / xiflex[0]

	kflex[0] = labflex[0] * calfa / (1 - calfa) * wflex[0] / rkflex[0]

	mcflex = wflex[0] ^ (1 - calfa) * rkflex[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))

	wflex[0] * (1 + curvw * (1 - clandaw)) / (1 + curvW) = sw[ss] * (labflex[0] ^ csigl * clandaw * (cflex[0] - cflex[-1] * chabb / cgamma) + wflex[0] * curvW * (clandaw - 1) / (1 + curvW))

	# (1 + curvp * (1 - cfc)) / (1 + curvP) = spinf[ss] * cfc * mcflex + spinf[ss] * (cfc - 1) * curvP / (1 + curvP)

	afuncflex[0] = rkflex[ss] * 1 / cZcap * (exp(cZcap * (zcapflex[0] - 1)) - 1)

	afuncDflex[0] = rkflex[ss] * exp(cZcap * (zcapflex[0] - 1))

	Sfuncflex[0] = csadjcost / 2 * (cgamma * inveflex[0] / inveflex[-1] - cgamma) ^ 2

	SfuncDflex[0] = csadjcost * (cgamma * inveflex[0] / inveflex[-1] - cgamma)

	ygap[0] = 100 * log(y[0] / yflex[0])

	dy[0] = ctrend + 100 * (y[0] / y[-1] - 1)

	dc[0] = ctrend + 100 * (c[0] / c[-1] - 1)

	dinve[0] = ctrend + 100 * (inve[0] / inve[-1] - 1)

	pinfobs[0] = 100 * (pinf[0] - 1)

	robs[0] = 100 * (r[0] - 1)

	dwobs[0] = ctrend + 100 * (w[0] / w[-1] - 1)

	labobs[0] = constelab + 100 * (lab[0] / lab[ss] - 1)

end


@parameters Smets_Wouters_2007 begin
	SCALE1_eb = -((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) ^ (-1)

	SCALE1_eqs = (cgamma ^ 2 * csadjcost) * (1 + cbeta * cgamma ^ (1 - csigma))

	SCALE1_epinf = 1 / ((1 / (1 + cbetabar * cgamma * cindp)) * ((1 - cprobp) * (1 - cbetabar * cgamma * cprobp) / cprobp) / ((cfc - 1) * curvp + 1))

	SCALE1_ew = 1 / ((1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / ((1 + cbetabar * cgamma) * cprobw) * (1 / ((clandaw - 1) * curvw + 1)))

	cgamma 	= 1 + ctrend / 100          							# gross growth rate
	
	cbeta 	= 1 / (1 + constebeta / 100)    						# discount factor

	cZcap   = czcap / (1 - czcap)
    
    curvP = curvp * (1 - clandap) / clandap

    curvW = curvw * (1 - clandaw) / clandaw

	clandap = cfc                									# fixed cost share/gross price markup
	
	cbetabar= cbeta * cgamma ^ (-csigma)   							# growth-adjusted discount factor in Euler equation

    mcflex = mc[ss] | mcflex
	
    pinf[ss] = 1 + constepinf / 100 | cpie

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

	z_ea	= 0.4618 # technology shock

	z_eb	= 1.8513 # risk-premium shock

	z_eg	= 0.6090 # government shock

	z_em	= 0.2397 # interest rate shock

	z_ew	= 0.2089 # wage mark-up shock

	z_eqs	= 0.6017 # investment-specific shock

	z_epinf	= 0.1455 # price mark-up shock

	1e-6 > ygap > -1e-6
end



"""
    SW07_nonlinear_steady_state!(out, parameters)

Non-allocating custom non-stochastic steady state solver for the `SW07_nonlinear` model.

The analytical steady state is computed directly in level space (ratios and products) rather
than in logs, so no `log` is taken of intermediate expressions. This avoids the implicit
positivity domain constraints that `log` imposes on intermediate terms (e.g. the
consumption-to-output ratio `1 - inve/y - gy`), which is useful when the solver or automatic
differentiation evaluates the function at parameter values that would otherwise produce a
`DomainError`.

The function fills `out` in place with the steady state values in the order expected by
`get_NSSS_and_parameters`: the variables in `sort(union(var, exo_past, exo_future))` followed
by the calibrated parameters (`mcflex`, `pinfss`).

`parameters` is the vector of parameter values in declaration order (as returned by
`get_parameters(SW07_nonlinear)`).

# Domain / validity conditions

Even in level space, the fractional/real powers and divisions in the closed form restrict the
admissible parameter region. A real, strictly positive (economically valid) steady state
requires:

- `cgamma = 1 + ctrend/100 > 0`         (base of `cgamma^csigma`, `cgamma^(1-csigma)`)
- `cbeta = 1/(1 + constebeta/100) > 0`
- `0 < calfa < 1`                        (bases `calfa^calfa`, `(1-calfa)^(1-calfa)`; exponent `1/(1-calfa)`)
- `cfc > 0`                              (so `mc = 1/cfc > 0`, base of `w`)
- `rk = cbeta^(-1)*cgamma^csigma - (1-ctou) > 0`   (base of `rk^(-calfa)`), i.e.
  `(1+constebeta/100)*(1+ctrend/100)^csigma > 1 - ctou`
- `chabb < cgamma`                       (so `1 - chabb/cgamma > 0`, base of `lab` and `xi`)
- `csigl != -1` (and `csigl > -1` so `1/(1+csigl) > 0`)
- `clandaw != 1` and `clandaw != 0`      (exponent of `w` in `gamw1`)
- `c_y = 1 - inve/y - gy > 0`            (feasibility; base of `lab`), i.e.
  `cg + (ctou + ctrend/100) * k_y < 1`
- `denw = 1 - cbeta*cgamma^(1-csigma)*cprobw > 0`   (so `gamw1, gamw2, gamw3 > 0`)
- `denp = 1 - cbeta*cgamma^(1-csigma)*cprobp > 0`   (so `gam1, gam2, gam3 > 0`)

Under these conditions every level variable is positive. The constant/observable entries
(`a, b, ms, qs, sw, spinf, pdot, wdot, dp, pdotl, wdotl, pk, zcap, qsaux, Pratio = 1`;
`afunc, Sfunc, SfuncD, epinfma, ewma, labobs, ygap = 0`; `dy, dc, dinve, dwobs = ctrend`;
`pinfobs = constepinf`; `robs = 100*(r-1)`) carry no range restriction. The remaining
variables are forced positive by the power chain: `rk, w, k_lab, k_y, lab, k, kp, y, c, inve,
wnew, dw, xi, r, pinf, gam1, gam2, gam3, gamw1, gamw2, gamw3` (the signs of the `gam*`/`gamw*`
block follow `denp`/`denw`).

Boundary (parametrised) limits where the steady state degenerates: `calfa -> 1`, `rk -> 0`,
`cfc -> 0`, `chabb -> cgamma`, `c_y -> 0` (i.e. `gy + inve/y -> 1`), `csigl -> -1`,
`clandaw -> 1`, `denw -> 0` (`cprobw -> 1/(cbeta*cgamma^(1-csigma))`), and `denp -> 0`
(`cprobp -> 1/(cbeta*cgamma^(1-csigma))`) all send one or more variables to `0` or `Inf`.

# Example
```julia
get_steady_state(SW07_nonlinear, steady_state_function = SW07_nonlinear_steady_state!)
```
"""
function SW07_nonlinear_steady_state!(out::AbstractVector, parameters::AbstractVector)
    ctou, cg, clandaw, curvw, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf,
    crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cindw, cindp,
    czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, ctrend, cgy,
    calfa, curvp, cprobp = parameters

    cgamma  = 1 + ctrend / 100
    cbeta   = 1 / (1 + constebeta / 100)
    clandap = cfc

    rk   = cbeta^(-1) * cgamma^csigma - (1 - ctou)
    mc   = 1 / clandap
    w    = (mc * calfa^calfa * (1 - calfa)^(1 - calfa) * rk^(-calfa))^(1 / (1 - calfa))

    inve_kp = 1 - (1 - ctou) / cgamma           # inve / kp
    inve_k  = cgamma * inve_kp                   # inve / k
    k_lab   = calfa / (1 - calfa) * w / rk       # k / lab
    k_y     = k_lab^(1 - calfa) * cfc            # k / y
    gy      = cg
    c_y     = 1 - inve_k * k_y - gy              # c / y

    lab  = (w * k_y / (clandaw * k_lab * c_y * (1 - chabb / cgamma)))^(1 / (1 + csigl))
    k    = k_lab * lab
    y    = k / k_y
    c    = c_y * y
    inve = inve_k * k
    kp   = cgamma * k
    r    = (1 + constepinf / 100) / (cbeta * cgamma^(-csigma))
    pinf = 1 + constepinf / 100
    xi   = (c * (1 - chabb / cgamma))^(-csigma) * exp((csigma - 1) / (1 + csigl) * lab^(1 + csigl))

    disc  = cbeta * cgamma^(1 - csigma)
    denw  = 1 - disc * cprobw
    denp  = 1 - disc * cprobp
    gamw1 = lab * w^(clandaw / (clandaw - 1) - curvw) / denw
    gamw2 = gamw1 * (1 - chabb / cgamma) * c * lab^csigl
    gamw3 = lab / denw
    gam1  = y / denp
    gam2  = gam1 * mc
    gam3  = gam1

    wnew = w
    pk   = 1.0
    afuncD = rk

    out[1]  = 1.0           # Pratio
    out[2]  = 0.0           # Sfunc
    out[3]  = 0.0           # SfuncD
    out[4]  = 0.0           # SfuncDflex
    out[5]  = 0.0           # Sfuncflex
    out[6]  = 1.0           # a
    out[7]  = 0.0           # afunc
    out[8]  = afuncD        # afuncD
    out[9]  = afuncD        # afuncDflex
    out[10] = 0.0           # afuncflex
    out[11] = 1.0           # b
    out[12] = c             # c
    out[13] = c             # cflex
    out[14] = ctrend        # dc
    out[15] = ctrend        # dinve
    out[16] = 1.0           # dp
    out[17] = wnew          # dw
    out[18] = ctrend        # dwobs
    out[19] = ctrend        # dy
    out[20] = 0.0           # epinfma
    out[21] = 0.0           # ewma
    out[22] = gam1          # gam1
    out[23] = gam2          # gam2
    out[24] = gam3          # gam3
    out[25] = gamw1         # gamw1
    out[26] = gamw2         # gamw2
    out[27] = gamw3         # gamw3
    out[28] = gy            # gy
    out[29] = inve          # inve
    out[30] = inve          # inveflex
    out[31] = k             # k
    out[32] = k             # kflex
    out[33] = kp            # kp
    out[34] = kp            # kpflex
    out[35] = lab           # lab
    out[36] = lab           # labflex
    out[37] = 0.0           # labobs
    out[38] = mc            # mc
    out[39] = 1.0           # ms
    out[40] = 1.0           # pdot
    out[41] = 1.0           # pdotl
    out[42] = pinf          # pinf
    out[43] = constepinf    # pinfobs
    out[44] = pk            # pk
    out[45] = pk            # pkflex
    out[46] = 1.0           # qs
    out[47] = 1.0           # qsaux
    out[48] = r             # r
    out[49] = rk            # rk
    out[50] = rk            # rkflex
    out[51] = 100 * (r - 1) # robs
    out[52] = r / pinf      # rrflex
    out[53] = 1.0           # spinf
    out[54] = 1.0           # sw
    out[55] = w             # w
    out[56] = 1.0           # wdot
    out[57] = 1.0           # wdotl
    out[58] = w             # wflex
    out[59] = wnew          # wnew
    out[60] = xi            # xi
    out[61] = xi            # xiflex
    out[62] = y             # y
    out[63] = y             # yflex
    out[64] = 0.0           # ygap
    out[65] = 1.0           # zcap
    out[66] = 1.0           # zcapflex
    out[67] = mc            # mcflex (calibrated parameter)
    out[68] = 1 + constepinf / 100  # pinfss (calibrated parameter)

    return nothing
end
