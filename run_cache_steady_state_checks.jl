using MacroModelling

include("models/RBC_baseline.jl")

reference_ss, nsss, sss_delta = MacroModelling.get_relevant_steady_states(RBC_baseline, :first_order)
SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(RBC_baseline, RBC_baseline.parameter_values)
all_ss = MacroModelling.expand_steady_state(SS_and_pars, RBC_baseline)

idx_excluding_obc = MacroModelling.parse_variables_input_to_index(:all_excluding_obc, RBC_baseline)
idx_excluding_aux_obc = MacroModelling.parse_variables_input_to_index(:all_excluding_auxiliary_and_obc, RBC_baseline)

@model RBC_switch begin
    1 / c[0] = (beta / c[1]) * (alpha * exp(z[1]) * k[0]^(alpha - 1) + (1 - delta))
    c[0] + k[0] = (1 - delta) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^alpha
    z[0] = rho * z[-1] + std_z * eps_z[x]
end

@parameters RBC_switch begin
    std_z = 0.01
    rho = 0.2
    delta = 0.02
    alpha = 0.5
    beta = 0.95
end

function rbc_steady_state(params)
    std_z, rho, delta, alpha, beta = params

    k_ss = ((1 / beta - 1 + delta) / alpha)^(1 / (alpha - 1))
    q_ss = k_ss^alpha
    c_ss = q_ss - delta * k_ss
    z_ss = 0.0

    return [c_ss, k_ss, q_ss, z_ss]
end

ss = get_steady_state(RBC_switch, steady_state_function = rbc_steady_state, derivatives = false)

println(reference_ss[1])
println(nsss[1])
println(ss)
println(length(all_ss))
println(length(idx_excluding_obc))
println(length(idx_excluding_aux_obc))
