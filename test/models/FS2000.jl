# Translated from: https://archives.dynare.org/documentation/examples.html
# be aware that dynare dynamics differ if c[2] or P[2] (not sure which one) are not declared explicitly as an auxilliary variable (c_lead(0) = c(+1);). The system in dynare has one less variable and the higher order solutions are different for the stochastic vol term.

@model m begin
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

end


@parameters m verbose = true begin  
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


# estimated_params;
# alp, beta_pdf, 0.356, 0.02; 
# bet, beta_pdf, 0.993, 0.002;
# gam, normal_pdf, 0.0085, 0.003;
# mst, normal_pdf, 1.0002, 0.007;
# rho, beta_pdf, 0.129, 0.223;
# psi, beta_pdf, 0.65, 0.05;
# del, beta_pdf, 0.01, 0.005;
# stderr z_e_a * e_a[x], inv_gamma_pdf, 0.035449, inf;
# stderr z_e_m * e_m[x], inv_gamma_pdf, 0.008862, inf;
# end;

# varobs gp_obs gy_obs;

# // computes only the posterior mode for demonstration. 
# //For full Metropolis simulation set mh_replic=20000. It will take several hours
# estimation(datafile=fsdat,nobs=192,loglinear,mh_replic=0,mh_nblocks=5,mh_jscale=0.8);
