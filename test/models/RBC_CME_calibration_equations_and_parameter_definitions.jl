@model m begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end


@parameters m verbose = true begin
    alpha | k[ss] / (4 * y[ss]) = cap_share
    cap_share = 1.66
    # alpha = .157

    beta | R[ss] = R_ss
    R_ss = 1.0035
    # beta = .999

    # delta | c[ss]/y[ss] = 1 - I_K_ratio
    # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
    # I_K_ratio = .15
    delta = .0226

    Pibar | Pi[ss] = Pi_ss
    Pi_ss = R_ss - Pi_real
    Pi_real = 1/1000
    # Pibar = 1.0008

    phi_pi = 1.5
    rhoz = 9 / 10
    std_eps = .0068
    rho_z_delta = rhoz
    std_z_delta = .005
end