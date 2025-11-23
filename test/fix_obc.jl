using Revise
using MacroModelling

@model borrowing_constraint begin
    Y[0] + B[0] = C[0] + R * B[-1]

    log(Y[0]) = ρ * log(Y[-1]) + σ * ε[x]

    C[0]^(-γ) = β * R * C[1]^(-γ) + λ[0]

    0 = max(B[0] - m * Y[0], -λ[0])
end

@parameters borrowing_constraint begin
    R = 1.05
    β = 0.945
    ρ = 0.9
    σ = 0.05
    m = 1
    γ = 1
end

SS(borrowing_constraint)

get_solution(borrowing_constraint)

𝓂 = borrowing_constraint

𝓂.model_jacobian

parameters = 𝓂.parameter_values

SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, true, false, 𝓂.solver_parameters)
    
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> sparse
    


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