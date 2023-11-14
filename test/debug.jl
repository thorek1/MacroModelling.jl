using MacroModelling


@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
    # A[0]=exp(z[0])
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
    # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
end


@parameters RBC_CME symbolic = true verbose = true begin
    # alpha | k[ss] / (4 * y[ss]) = cap_share
    # cap_share = 1.66
    alpha = .157

    # beta | R[ss] = R_ss
    # R_ss = 1.0035
    beta = .999

    # delta | c[ss]/y[ss] = 1 - I_K_ratio
    # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
    # I_K_ratio = .15
    delta = .0226

    # Pibar | Pi[ss] = Pi_ss
    # Pi_ss = 1.0025
    Pibar = 1.0008

    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005

    # cap_share > 0
    # R_ss > 0
    # Pi_ss > 0
    # I_K_ratio > 0 

    # 0 < alpha < 1 
    0 < beta < 1
    # 0 < delta < 1
    0 < Pibar
    # 0 <= rhoz < 1
    phi_pi > 0

    # 0 < A < 1
    # 0 < k < 50
    0 < Pi
    0 < R
end

import SymPyPythonCall as SPPC
symbols_pos = [:Pibar, :phi_pi, :beta]

for pos in symbols_pos
    eval(:($pos = SPPC.symbols($(string(pos)), real = true, finite = true, positive = true)))
end

vars_to_solve = Set(eval(:([$(symbols_pos...)])))

eqs_to_solve = Pibar^(phi_pi/(phi_pi-1))/beta

soll = SPPC.solve(eqs_to_solve,collect(vars_to_solve)[2])

soll_free_syms = SPPC.free_symbols.(soll[1])

intersect(vars_to_solve,soll_free_syms)

x.atoms().|> ↓
xx = (free_symbols(x)) .|> ↓