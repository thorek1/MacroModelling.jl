@model FS2000 begin
    dA[0] = exp(gam + z_e_a  *  e_a[x])

    log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  e_m[x]

    - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0

    W[0] = l[0] / n[0]

    - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0

    R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]

    1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0

    c[0] + k[0] = exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  *  e_a[x])) * k[-1]

    P[0] * c[0] = m[0]

    m[0] - 1 + d[0] = l[0]

    e[0] = exp(z_e_a  *  e_a[x])

    y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x]))

    gy_obs[0] = dA[0] * y[0] / y[-1]

    gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]

    log_gy_obs[0] = log(gy_obs[0])

    log_gp_obs[0] = log(gp_obs[0])
end


@parameters FS2000 begin
    alp     = 0.356
    bet     = 0.993
    gam     = 0.0085
    mst     = 1.0002
    rho     = 0.129
    psi     = 0.65
    del     = 0.01
    z_e_a   = 0.035449
    z_e_m   = 0.008862
end

# Translated from: https://archives.dynare.org/documentation/examples.html
# be aware that dynare dynamics differ if c[2] or P[2] (not sure which one) are not declared explicitly as an auxiliary variable (c_lead(0) = c(+1)). The system in dynare has one less variable and the higher order solutions are different for the stochastic vol term.


# Custom steady state function for FS2000 model
# Variable order: P, R, W, c, d, dA, e, gp_obs, gy_obs, k, l, log_gp_obs, log_gy_obs, m, n, y
# Parameter order: alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m
function FS2000_custom_steady_state_function(parameters)
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = parameters

    dA = exp(gam)
    gst = 1/dA
    m = mst
    
    khst = ( (1-gst*bet*(1-del)) / (alp*gst^alp*bet) )^(1/(alp-1))
    xist = ( ((khst*gst)^alp - (1-gst*(1-del))*khst)/mst )^(-1)
    nust = psi*mst^2/( (1-alp)*(1-psi)*bet*gst^alp*khst^alp )
    n  = xist/(nust+xist)
    P  = xist + nust
    k  = khst*n

    l  = psi*mst*n/( (1-psi)*(1-n) )
    c  = mst/P
    d  = l - mst + 1
    y  = k^alp*n^(1-alp)*gst^alp
    R  = mst/bet
    W  = l/n
    ist  = y-c
    q  = 1 - d

    e = 1
    
    gp_obs = m/dA
    gy_obs = dA

    log_gp_obs = log(gp_obs)
    log_gy_obs = log(gy_obs)

    # Return in order: P, R, W, c, d, dA, e, gp_obs, gy_obs, k, l, log_gp_obs, log_gy_obs, m, n, y
    return [P, R, W, c, d, dA, e, gp_obs, gy_obs, k, l, log_gp_obs, log_gy_obs, m, n, y]
end
