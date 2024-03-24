@model SW07_nonlinear begin
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

    r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / pinfss) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0]

    # r[0] = max(1.00025,r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / pinf[ss]) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])

    afunc[0] = rk[ss] * 1 / (czcap / (1 - czcap)) * (exp(czcap / (1 - czcap) * (zcap[0] - 1)) - 1)

    afuncD[0] = rk[ss] * exp(czcap / (1 - czcap) * (zcap[0] - 1))

    Sfunc[0] = csadjcost / 2 * ((1 + ctrend / 100) * inve[0] / inve[-1] - (1 + ctrend / 100)) ^ 2

    SfuncD[0] = csadjcost * ((1 + ctrend / 100) * inve[0] / inve[-1] - (1 + ctrend / 100))

    a[0] = 1 - crhoa + crhoa * a[-1] + z_ea * ea[x] / 100

    b[0] = 1 - crhob + crhob * b[-1] + z_eb * eb[x] * ( - (((1 - chabb / (1 + ctrend / 100)) / (csigma * (1 + chabb / (1 + ctrend / 100)))) ^ (-1))) / 100

    gy[0] - cg = crhog * (gy[-1] - cg) + z_eg * egy[x] / 100 + z_ea * ea[x] * cgy / 100

    qs[0] = 1 - crhoqs + crhoqs * qs[-1] + z_eqs * eqs[x] * csadjcost * (1 + ctrend / 100) ^ 2 * (1 + 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (1 - csigma)) / 100

    ms[0] = 1 - crhoms + crhoms * ms[-1] + z_em * ems[x] / 100

    spinf[0] = 1 - crhopinf + crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]

    epinfma[0] = z_epinf * epinf[x] * 1 / (1 / (1 + cindp * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) * (1 - cprobp) * (1 - cprobp * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) / cprobp / (1 + curvp * (cfc - 1))) / 100

    sw[0] = 1 - crhow + crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]

    ewma[0] = z_ew * ew[x] * 1 / (1 / (1 + curvw * (clandaw - 1)) * (1 - cprobw) * (1 - cprobw * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) / (cprobw * (1 + (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)))) / 100

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

    ygap[0] = 100 * log(y[0] / yflex[0])

    dy[0] = ctrend + 100 * (y[0] / y[-1] - 1)

    dc[0] = ctrend + 100 * (c[0] / c[-1] - 1)

    dinve[0] = ctrend + 100 * (inve[0] / inve[-1] - 1)

    pinfobs[0] = 100 * (pinf[0] - pinf[ss]) + constepinf

    robs[0] = 100 * (r[0] - 1)

    dwobs[0] = ctrend + 100 * (w[0] / w[-1] - 1)

    labobs[0] = 100 * (lab[0] / lab[ss] - 1)

end


@parameters SW07_nonlinear begin
    ctou=.025
    clandaw=1.5
    cg=0.18
    curvp=10
    curvw=10
    
    calfa=.24
    csigma=1.5
    cfc=1.5
    cgy=0.51
    
    csadjcost= 6.0144
    chabb=    0.6361    
    cprobw=   0.8087
    csigl=    1.9423
    cprobp=   0.6
    cindw=    0.3243
    cindp=    0.47
    czcap=    0.2696
    crpi=     1.488
    crr=      0.8762
    cry=      0.0593
    crdy=     0.2347
    
    crhoa=    0.9977
    crhob=    0.5799
    crhog=    0.9957
    # crhols=   0.9928
    crhoqs=   0.7165
    # crhoas=1 
    crhoms=0
    crhopinf=0
    crhow=0
    cmap = 0
    cmaw  = 0
    
    clandap=cfc
    cbetabar=cbeta*cgamma^(-csigma)
    cr=cpie/(cbeta*cgamma^(-csigma))
    crk=(cbeta^(-1))*(cgamma^csigma) - (1-ctou)
    cw = (calfa^calfa*(1-calfa)^(1-calfa)/(clandap*crk^calfa))^(1/(1-calfa))
    cikbar=(1-(1-ctou)/cgamma)
    cik=(1-(1-ctou)/cgamma)*cgamma
    clk=((1-calfa)/calfa)*(crk/cw)
    cky=cfc*(clk)^(calfa-1)
    ciy=cik*cky
    ccy=1-cg-cik*cky
    crkky=crk*cky
    cwhlc=(1/clandaw)*(1-calfa)/calfa*crk*cky/ccy
    cwly=1-crk*cky
    
    conster=(cr-1)*100
    ctrend=(1.004-1)*100
    constepinf=(1.005-1)*100

    cpie=1+constepinf/100
    cgamma=1+ctrend/100 

    cbeta=1/(1+constebeta/100)
    constebeta = 100 / .9995 - 100

    constelab=0

    z_ea = 0.4618
    z_eb = 1.8513
    z_eg = 0.6090
    z_eqs = 0.6017
    z_em = 0.2397
    z_epinf = 0.1455
    z_ew = 0.2089

    mcflex = mc[ss] | mcflex
    pinf[ss] = 1 + constepinf / 100 | pinfss
end
