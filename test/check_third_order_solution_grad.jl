using MacroModelling
import DifferentiationInterface
import ADTypes
import Mooncake
import LinearAlgebra as ℒ
using BenchmarkTools
using Random
include("../models/Smets_Wouters_2007.jl")

# using MacroModelling

@model Smets_Wouters_2007_SV_EPZ begin
    # Epstein Zin
    # U[0] = exp((csigma - 1) / (1 + csigl) * (lab[0] * (curvW + wdot[0]) / (1 + curvW)) ^ (1 + csigl)) * ((c[0] - c[-1] * chabb / cgamma) / (1 - csigma)) ^ (1 - csigma)

    U[0] = (1 - cbetabar) * exp((csigma - 1) / (1 + csigl) * (lab[0] * (curvW + wdot[0]) / (1 + curvW)) ^ (1 + csigl)) / exp((csigma - 1) / (1 + csigl) * (lab[ss] * (curvW + wdot[ss]) / (1 + curvW)) ^ (1 + csigl)) * ((c[0] - c[-1] * chabb / cgamma) / ((c[ss] - c[ss] * chabb / cgamma))) ^ (1 - csigma)


    # (V[0]) = 1 - cbetabar + cbetabar * ((V[1]) ^ (1 - αᴱ)) ^ (1 / (1 - αᴱ))
  
    # (V[0]) = U[0] + cbetabar * V[1]
 
    (V[0]) = U[0] + cbetabar * ((V[1]) ^ (1 - αᴱ)) ^ (1 / (1 - αᴱ))
 
    # V[0] = U[0] - cbetabar * ((-V[1]) ^ (1 - αᴱ)) ^ (1 / (1 - αᴱ))
 
 
    # E_t_V_tp1_1_minus_alpha[0] = V[1] ^ (1 - αᴱ)
 
    E_t_V_tp1_1_minus_sigma[0] = (V[1]) ^ (1 - csigma)
 
    # E_t_V_tp1_1_minus_alpha[0] = (-V[1]) ^ (1 - αᴱ)
 
 
    # M[0] = cbetabar * (exp(V[0]) / E_t_V_tp1_1_minus_alpha[-1] ^ (1 / (1 - αᴱ)) ) ^ (-αᴱ)
 
    M[0] = cbetabar * ((V[1]) / E_t_V_tp1_1_minus_sigma[0] ^ (1 / (1 - αᴱ)) ) ^ (-αᴱ)
 
    # M[0] = cbetabar * ((V[1]) / E_t_V_tp1_1_minus_alpha[0] ^ (1 / (1 - αᴱ)) ) ^ (-αᴱ)
 
    # M[0] = cbetabar * (-V[1] / E_t_V_tp1_1_minus_alpha[0] ^ (1 / (1 - αᴱ)) ) ^ (-αᴱ)
 
 
    # Stochastic volatility
    log(z_ea[0])    = (1 - rho_z_ea) * log(z̄_ea) + rho_z_ea * log(z_ea[-1]) + z_z_ea * e_z_ea[x]
    log(z_eb[0])    = (1 - rho_z_eb) * log(z̄_eb) + rho_z_eb * log(z_eb[-1]) + z_z_eb * e_z_eb[x]
    log(z_eg[0])    = (1 - rho_z_eg) * log(z̄_eg) + rho_z_eg * log(z_eg[-1]) + z_z_eg * e_z_eg[x]
    log(z_em[0])    = (1 - rho_z_em) * log(z̄_em) + rho_z_em * log(z_em[-1]) + z_z_em * e_z_em[x]
    log(z_ew[0])    = (1 - rho_z_ew) * log(z̄_ew) + rho_z_ew * log(z_ew[-1]) + z_z_ew * e_z_ew[x]
    log(z_eqs[0])   = (1 - rho_z_eqs) * log(z̄_eqs) + rho_z_eqs * log(z_eqs[-1]) + z_z_eqs * e_z_eqs[x]
    log(z_epinf[0]) = (1 - rho_z_epinf) * log(z̄_epinf) + rho_z_epinf * log(z_epinf[-1]) + z_z_epinf * e_z_epinf[x]
 
 
 
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
 
    1 = qs[0] * pk[0] * (1 - Sfunc[0] - cgamma * inve[0] * SfuncD[0] / inve[-1]) + SfuncD[1] * xi[1] / xi[0] * qsaux[0] * pk[1] * (cgamma * inve[1] / inve[0]) ^ 2 * M[0]
 
    xi[0] = xi[1] * b[0] * r[0] * M[0] / pinf[1]
 
    rk[0] = afuncD[0]
 
    pk[0] = (rk[1] * zcap[1] - afunc[1] + (1 - ctou) * pk[1]) * xi[1] * M[0] / xi[0]
 
    k[0] = lab[0] * w[0] * calfa / (1 - calfa) / rk[0]
 
    mc[0] = w[0] ^ (1 - calfa) * rk[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))
 
    wnew[0] * gamw1[0] * (1 + curvw * (1 - clandaw)) / (1 + curvW) = clandaw * gamw2[0] + gamw3[0] * curvW * (clandaw - 1) / (1 + curvW) * wnew[0] ^ (1 + clandaw * (1 + curvW) / (clandaw - 1))
 
    gamw1[0] = lab[0] * dw[0] ^ (clandaw * (1 + curvW) / (clandaw - 1)) + gamw1[1] * (cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1)) * xi[1] / xi[0] * cgamma * cprobw * M[0]
 
    gamw2[0] = (c[0] - c[-1] * chabb / cgamma) * lab[0] * sw[0] * dw[0] ^ (clandaw * (1 + curvW) / (clandaw - 1)) * (lab[0] * (curvW + wdot[0]) / (1 + curvW)) ^ csigl + gamw2[1] * (cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - clandaw) * (1 + curvW) / (clandaw - 1)) * xi[1] / xi[0] * cgamma * cprobw * M[0]
 
    gamw3[0] = lab[0] + gamw3[1] * cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1] * xi[1] / xi[0] * cgamma * cprobw * M[0]
 
    Pratio[0] * gam1[0] * (1 + curvp * (1 - cfc)) / (1 + curvP) = cfc * gam2[0] + gam3[0] * (cfc - 1) * curvP / (1 + curvP) * Pratio[0] ^ (1 + cfc * (1 + curvP) / (cfc - 1))
 
    gam1[0] = y[0] * dp[0] ^ (cfc * (1 + curvP) / (cfc - 1)) + gam1[1] * xi[1] / xi[0] * cgamma * cprobp * M[0] * (cpie ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1))
 
    gam2[0] = y[0] * mc[0] * spinf[0] * dp[0] ^ (cfc * (1 + curvP) / (cfc - 1)) + gam2[1] * xi[1] / xi[0] * cgamma * cprobp * M[0] * (cpie ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1]) ^ (( - cfc) * (1 + curvP) / (cfc - 1))
 
    gam3[0] = y[0] + gam3[1] * cpie ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1] * xi[1] / xi[0] * cgamma * cprobp * M[0]
 
    qsaux[0] = qs[1]
 
    # r[0] = max(1.00025,r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / pinfss) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])
   
    r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0]
 
    afunc[0] = rk[ss] * 1 / cZcap * (exp(cZcap * (zcap[0] - 1)) - 1)
 
    afuncD[0] = rk[ss] * exp(cZcap * (zcap[0] - 1))
 
    Sfunc[0] = csadjcost / 2 * (cgamma * inve[0] / inve[-1] - cgamma) ^ 2
 
    SfuncD[0] = csadjcost * (cgamma * inve[0] / inve[-1] - cgamma)
 
    a[0] = 1 - crhoa + crhoa * a[-1] + z_ea[0] / 100 * ea[x]
 
    b[0] = 1 - crhob + crhob * b[-1] +  z_eb[0] / 100 * SCALE1_eb * eb[x]
 
    gy[0] - cg = crhog * (gy[-1] - cg) + z_eg[0] / 100 * eg[x] + z_ea[0] / 100 * ea[x] * cgy
 
    qs[0] = 1 - crhoqs + crhoqs * qs[-1] + z_eqs[0] / 100 * SCALE1_eqs * eqs[x]
 
    ms[0] = 1 - crhoms + crhoms * ms[-1] + z_em[0] / 100 * em[x]
 
    spinf[0] = 1 - crhopinf + crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]
 
    epinfma[0] = z_epinf[0] / 100 * SCALE1_epinf * epinf[x]
 
    sw[0] = 1 - crhow + crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]
 
    ewma[0] = z_ew[0] / 100 * SCALE1_ew * ew[x]
 
    yflex[0] = cflex[0] + inveflex[0] + gy[0] * yflex[ss] + afuncflex[0] * kpflex[-1] / cgamma
 
    yflex[0] = a[0] * kflex[0] ^ calfa * labflex[0] ^ (1 - calfa) - (cfc - 1) * yflex[ss]
 
    kflex[0] = kpflex[-1] * zcapflex[0] / cgamma
 
    kpflex[0] = inveflex[0] * qs[0] * (1 - Sfuncflex[0]) + kpflex[-1] * (1 - ctou) / cgamma
 
    xiflex[0] = exp((csigma - 1) / (1 + csigl) * labflex[0] ^ (1 + csigl)) * (cflex[0] - cflex[-1] * chabb / cgamma) ^ (-csigma)
 
    1 = qs[0] * pkflex[0] * (1 - Sfuncflex[0] - cgamma * inveflex[0] * SfuncDflex[0] / inveflex[-1]) + SfuncDflex[1] * qsaux[0] * xiflex[1] / xiflex[0] * pkflex[1] * (cgamma * inveflex[1] / inveflex[0]) ^ 2 * M[0]
 
    xiflex[0] = xiflex[1] * b[0] * rrflex[0] * M[0]
 
    rkflex[0] = afuncDflex[0]
 
    pkflex[0] = (rkflex[1] * zcapflex[1] - afuncflex[1] + (1 - ctou) * pkflex[1]) * xiflex[1] * M[0] / xiflex[0]
 
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
    
    1 + spreadobs[0] / 40000 = b[0] + constespread + z_spreadobs * e_spreadobs[x]
end

@parameters Smets_Wouters_2007_SV_EPZ begin
    # Epstein-Zin
    αᴱ = 0
 
    #### do NOT change parameters here in the parameter declaration - everything needs to be recompiled if you change it here (slow) - change it in the function calls instead (fast - see below)
   
    # Stochastic volatility
    ## steady state volatility
    z̄_ea       = 0.4618
    z̄_eb       = 1.8513
    z̄_eg       = 0.6090
    z̄_em       = 0.2397
    z̄_ew       = 0.2089
    z̄_eqs      = 0.6017
    z̄_epinf    = 0.1455
 
    #### do NOT change parameters here in the parameter declaration - everything needs to be recompiled if you change it here (slow) - change it in the function calls instead (fast - see below)
 
    ## stochastic volatility shock size
    z_z_ea      = 1
    z_z_eb      = 1
    z_z_eg      = 1
    z_z_em      = 1
    z_z_ew      = 1
    z_z_eqs     = 1
    z_z_epinf   = 1
 
    #### do NOT change parameters here in the parameter declaration - everything needs to be recompiled if you change it here (slow) - change it in the function calls instead (fast - see below)
 
    ## stochastic volatility AR(1) coefficient
    rho_z_ea    = 0.75
    rho_z_eb    = 0.75
    rho_z_eg    = 0.75
    rho_z_em    = 0.75
    rho_z_ew    = 0.75
    rho_z_eqs   = 0.75
    rho_z_epinf = 0.75
 
    #### do NOT change parameters here in the parameter declaration - everything needs to be recompiled if you change it here (slow) - change it in the function calls instead (fast - see below)
 
    SCALE1_eb = -((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) ^ (-1)
 
    SCALE1_eqs = (cgamma ^ 2 * csadjcost) * (1 + cbeta * cgamma ^ (1 - csigma))
 
    SCALE1_epinf = 1 / ((1 / (1 + cbetabar * cgamma * cindp)) * ((1 - cprobp) * (1 - cbetabar * cgamma * cprobp) / cprobp) / ((cfc - 1) * curvp + 1))
 
    SCALE1_ew = 1 / ((1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / ((1 + cbetabar * cgamma) * cprobw) * (1 / ((clandaw - 1) * curvw + 1)))
 
    cgamma  = 1 + ctrend / 100                                      # gross growth rate
   
    cbeta   = 1 / (1 + constebeta / 100)                            # discount factor
 
    cZcap   = czcap / (1 - czcap)
   
    curvP = curvp * (1 - clandap) / clandap
 
    curvW = curvw * (1 - clandaw) / clandaw
 
    clandap = cfc                                                   # fixed cost share/gross price markup
   
    cbetabar= cbeta * cgamma ^ (-csigma)                            # growth-adjusted discount factor in Euler equation
 
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
 
    constespread = 0
    
    z_spreadobs = 0.02
    # z_ea  = 0.4618 # technology shock
 
    # z_eb  = 1.8513 # risk-premium shock
 
    # z_eg  = 0.6090 # government shock
 
    # z_em  = 0.2397 # interest rate shock
 
    # z_ew  = 0.2089 # wage mark-up shock
 
    # z_eqs = 0.6017 # investment-specific shock
 
    # z_epinf   = 0.1455 # price mark-up shock
 
    1e-6 > ygap > -1e-6
end

# ── Objective: norm of the 3rd order solution matrix ───────────────────────
# get_solution(model, parameters; algorithm = :third_order) returns
# (SS, [S1, S2, S3], success).  We take ‖S3‖_F and differentiate wrt parameters.

function third_order_norm(params)
    SS, sol_matrices, success = get_solution(
        Smets_Wouters_2007_SV_EPZ, params;
        algorithm = :third_order,
        verbose = true,
    )
    if !success
        return Inf
    end
    S3 = sol_matrices[3]
    return ℒ.norm(S3)
end

params0 = copy(Smets_Wouters_2007_SV_EPZ.parameter_values)

println("=== Forward pass ===")
val = third_order_norm(params0)
println("‖S3‖_F = $val")

println("\n=== Reverse-mode gradient (Mooncake via DI) ===")
backend = ADTypes.AutoMooncake(config = nothing)
grad = DifferentiationInterface.gradient(third_order_norm, backend, params0)
@profview grad = DifferentiationInterface.gradient(third_order_norm, backend, params0)

println("\n=== Results ===")
println("gradient = $grad")
println("norm of gradient = $(ℒ.norm(grad))")


Random.seed!(1234)
@benchmark grad = DifferentiationInterface.gradient(third_order_norm, backend, params0 .+ randn(length(params0)) * 1e-6)