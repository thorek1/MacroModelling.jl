# This script contains the Julia code from the plotting.md documentation.
# It is modified to save all plots referenced in the markdown file
# to the docs/assets/ directory, allowing the documentation to be regenerated.

## Setup
using MacroModelling
import StatsPlots
using AxisKeys
import Random; Random.seed!(10) # For reproducibility of :simulate

# Load a model
@model Gali_2015_chapter_3_nonlinear begin
	W_real[0] = C[0] ^ σ * N[0] ^ φ
	Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]
	R[0] = 1 / Q[0]
	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)
	R[0] = Pi[1] * realinterest[0]
	R[0] = 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0])
	C[0] = Y[0]
	log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]
	log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]
	nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]
	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])
	1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)
	S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]
	Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)
	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]
	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]
	log_y[0] = log(Y[0])
	log_W_real[0] = log(W_real[0])
	log_N[0] = log(N[0])
	pi_ann[0] = 4 * log(Pi[0])
	i_ann[0] = 4 * log(R[0])
	r_real_ann[0] = 4 * log(realinterest[0])
	M_real[0] = Y[0] / R[0] ^ η
end

@parameters Gali_2015_chapter_3_nonlinear begin
	σ = 1
	φ = 5
	ϕᵖⁱ = 1.5
	ϕʸ = 0.125
	θ = 0.75
	ρ_ν = 0.5
	ρ_z = 0.5
	ρ_a = 0.9
	β = 0.99
	η = 3.77
	α = 0.25
	ϵ = 9
	τ = 0
    std_a = .01
    std_z = .05
    std_nu = .0025
end

## Impulse response functions (IRF)
plot_irf(Gali_2015_chapter_3_nonlinear, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :default_irf)

### Algorithm 
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :second_order_irf)

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :first_order_irf)

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :compare_orders_irf)

# The following plot is built on the previous one
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :pruned_third_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :multiple_orders_irf)

### Initial state 
init_state = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true)
get_state_variables(Gali_2015_chapter_3_nonlinear)
init_state(:nu,:,:) .= 0.1
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state), save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :custom_init_irf)

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, initial_state = vec(init_state), save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :no_shock_init_irf)

# This plot is built on the previous one
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, initial_state = vec(init_state))
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, plot_type = :stack, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :stacked_init_irf)

init_state_2nd = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true, algorithm = :second_order)
init_state_2nd(:nu,:,:) .= 0.1
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_2nd), algorithm = :second_order)

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state))
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_2nd), algorithm = :second_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :multi_sol_init_irf)

init_state_pruned_3rd_in_diff = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true) - get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, algorithm = :pruned_third_order, levels = true)
init_states_pruned_3rd_vec = [zero(vec(init_state_pruned_3rd_in_diff)), vec(init_state_pruned_3rd_in_diff), zero(vec(init_state_pruned_3rd_in_diff))]
init_states_pruned_3rd_vec[1][18] = 0.1
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = init_states_pruned_3rd_vec, algorithm = :pruned_third_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :pruned_3rd_vec_irf)

init_state_pruned_3rd = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true, algorithm = :pruned_third_order)
init_state_pruned_3rd(:nu,:,:) .= 0.1
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_pruned_3rd), algorithm = :pruned_third_order)

# This plot builds on the previous one
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_pruned_3rd), algorithm = :pruned_third_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_2nd), algorithm = :second_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state), save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :all_sol_init_irf)

### Shocks
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :single_shock_irf)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = "eps_a")
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = [:eps_a, :eps_z], save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :multi_shocks_irf)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = (:eps_a, :eps_z))
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = [:eps_a :eps_z])
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :all_excluding_obc, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :all_ex_obc_irf)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :all)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :simulate, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :simulated_irf)

init_state = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true)
get_state_variables(Gali_2015_chapter_3_nonlinear)
init_state(:nu,:,:) .= 0.1
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, initial_state = vec(init_state), save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :deterministic_irf)

shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shocks[1])
for (i,s) in enumerate(shocks[2:end])
    if i == length(shocks[2:end])
        plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = s, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :compare_shocks_irf)
    else
        plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = s)
    end
end

n_periods = 3
shock_keyedarray = KeyedArray(zeros(length(shocks), n_periods), Shocks = shocks, Periods = 1:n_periods)
shock_keyedarray("eps_a",[1]) .= 1
shock_keyedarray("eps_z",[2]) .= -1/2
shock_keyedarray("eps_nu",[3]) .= 1/3
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shock_keyedarray, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :shock_series_irf)

shock_matrix = zeros(length(shocks), n_periods)
shock_matrix[1,1] = 1
shock_matrix[3,2] = -1/2
shock_matrix[2,3] = 1/3
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shock_matrix)

shock_matrix_1 = zeros(length(shocks), n_periods)
shock_matrix_1[1,1] = 1
shock_matrix_1[3,2] = -1/2
shock_matrix_1[2,3] = 1/3   
shock_matrix_2 = zeros(length(shocks), n_periods * 2)
shock_matrix_2[1,4] = -1
shock_matrix_2[3,5] = 1/2
shock_matrix_2[2,6] = -1/3
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shock_matrix_1)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = shock_matrix_2, plot_type = :stack, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :stacked_matrices_irf)

### Periods
plot_irf(Gali_2015_chapter_3_nonlinear, periods = 10, shocks = :eps_a, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :ten_periods_irf)

plot_irf(Gali_2015_chapter_3_nonlinear, periods = 10, shocks = :eps_a)
shock_matrix_periods = zeros(length(shocks), 15)
shock_matrix_periods[1,1] = .1
shock_matrix_periods[3,5] = -1/2
shock_matrix_periods[2,15] = 1/3
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = shock_matrix_periods, periods = 20, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :mixed_periods_irf)

### shock_size 
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, shock_size = -2, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :shock_size_irf)

### negative_shock
plot_irf(Gali_2015_chapter_3_nonlinear, negative_shock = true, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :negative_shock_irf)

### variables
plot_irf(Gali_2015_chapter_3_nonlinear, variables = [:Y, :Pi], save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :var_select_irf)
plot_irf(Gali_2015_chapter_3_nonlinear, variables = (:Y, :Pi))
plot_irf(Gali_2015_chapter_3_nonlinear, variables = [:Y :Pi])
plot_irf(Gali_2015_chapter_3_nonlinear, variables = ["Y", "Pi"])
plot_irf(Gali_2015_chapter_3_nonlinear, variables = :Y)
plot_irf(Gali_2015_chapter_3_nonlinear, variables = "Y")
plot_irf(Gali_2015_chapter_3_nonlinear, variables = :all_excluding_auxiliary_and_obc)

@model FS2000 begin
    dA[0] = exp(gam + z_e_a  * e_a[x])
    log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  * e_m[x]
    - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0
    W[0] = l[0] / n[0]
    - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0
    R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  * e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]
    1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  * e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0
    c[0] + k[0] = exp( - alp * (gam + z_e_a  * e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  * e_a[x])) * k[-1]
    P[0] * c[0] = m[0]
    m[0] - 1 + d[0] = l[0]
    e[0] = exp(z_e_a  * e_a[x])
    y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  * e_a[x]))
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

plot_irf(FS2000, variables = :all_excluding_obc, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :with_aux_vars_irf)

@model Gali_2015_chapter_3_obc begin
	W_real[0] = C[0] ^ σ * N[0] ^ φ
	Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]
	R[0] = 1 / Q[0]
	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)
	R[0] = Pi[1] * realinterest[0]
	R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))
	C[0] = Y[0]
	log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]
	log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]
	nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]
	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])
	1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)
	S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]
	Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)
	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]
	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]
	log_y[0] = log(Y[0])
	log_W_real[0] = log(W_real[0])
	log_N[0] = log(N[0])
	pi_ann[0] = 4 * log(Pi[0])
	i_ann[0] = 4 * log(R[0])
	r_real_ann[0] = 4 * log(realinterest[0])
	M_real[0] = Y[0] / R[0] ^ η
end

@parameters Gali_2015_chapter_3_obc begin
    R̄ = 1.0
	σ = 1
	φ = 5
	ϕᵖⁱ = 1.5
	ϕʸ = 0.125
	θ = 0.75
	ρ_ν = 0.5
	ρ_z = 0.5
	ρ_a = 0.9
	β = 0.99
	η = 3.77
	α = 0.25
	ϵ = 9
	τ = 0
    std_a = .01
    std_z = .05
    std_nu = .0025
    R > 1.0001
end

plot_irf(Gali_2015_chapter_3_obc, variables = :all, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :with_obc_vars_irf)

# The following call generates the `obc_binding_irf` image referenced in the markdown.
# The code for this specific plot is not explicitly shown but implied.
plot_irf(Gali_2015_chapter_3_obc, shocks = :eps_z, shock_size = 3, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :obc_binding_irf)

get_equations(Gali_2015_chapter_3_obc)

### parameters
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.95, shocks = :eps_a, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :beta_095_irf)
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.99, shocks = :eps_a)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.95, shocks = :eps_a, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :compare_beta_irf)

# This plot builds on the previous one
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.99, shocks = :eps_a)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.95, shocks = :eps_a)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.97, :τ => 0.5), shocks = :eps_a, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :multi_params_irf)

plot_irf(Gali_2015_chapter_3_nonlinear, parameters = [:β => 0.98, :τ => 0.25], shocks = :eps_a)

params = get_parameters(Gali_2015_chapter_3_nonlinear, values = true)
param_vals = [p[2] for p in params]
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = param_vals, shocks = :eps_a)

### ignore_obc
plot_irf(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, ignore_obc = true, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :obc_irf)
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, ignore_obc = true, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :compare_obc_irf)

### generalised_irf
plot_irf(Gali_2015_chapter_3_obc, generalised_irf = true, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :obc_girf_irf)
plot_irf(Gali_2015_chapter_3_obc, generalised_irf = true, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3)
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :obc_girf_compare_irf)

# This plot builds on the previous one
plot_irf(Gali_2015_chapter_3_obc, generalised_irf = true, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3)
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3)
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, ignore_obc = true, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :obc_all_compare_irf)

plot_irf(Gali_2015_chapter_3_nonlinear, generalised_irf = true, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :girf_2nd_irf)
plot_irf(Gali_2015_chapter_3_nonlinear, generalised_irf = true, shocks = :eps_a,  algorithm = :pruned_second_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :pruned_second_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :girf_compare_irf)

### generalised_irf_warmup_iterations and generalised_irf_draws
plot_irf(Gali_2015_chapter_3_nonlinear, generalised_irf = true, shocks = :eps_a,  algorithm = :pruned_second_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, generalised_irf = true, generalised_irf_draws = 1000, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :girf_1000_irf)

# This plot builds on the previous one
plot_irf(Gali_2015_chapter_3_nonlinear, generalised_irf = true, shocks = :eps_a,  algorithm = :pruned_second_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, generalised_irf = true, generalised_irf_draws = 1000, shocks = :eps_a,  algorithm = :pruned_second_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, generalised_irf = true, generalised_irf_draws = 5000, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :girf_5000_irf)

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :pruned_second_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, generalised_irf = true, generalised_irf_draws = 5000, generalised_irf_warmup_iterations = 500, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :girf_5000_500_irf)

### label
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.99, :τ => 0.0), shocks = :eps_a, label = "Std. params")
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.95, :τ => 0.5), shocks = :eps_a, label = "Alt. params", save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :custom_labels_irf)

plot_irf(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.99, :τ => 0.0), shocks = :eps_a, label = :standard)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.95, :τ => 0.5), shocks = :eps_a, label = :alternative)

plot_irf(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.99, :τ => 0.0), shocks = :eps_a, label = 0.99)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.95, :τ => 0.5), shocks = :eps_a, label = 0.95, save_plots = true, save_plots_format = :svg)

### plot_attributes
ec_color_palette = ["#FFD724", "#353B73", "#2F9AFB", "#B8AAA2", "#E75118", "#6DC7A9", "#F09874", "#907800"]
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shocks[1])
for (i,s) in enumerate(shocks[2:end])
    if i == length(shocks[2:end])
        plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = s, plot_attributes = Dict(:palette => ec_color_palette), plot_type = :stack, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :custom_colors_irf)
    else
        plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = s, plot_attributes = Dict(:palette => ec_color_palette), plot_type = :stack)
    end
end

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, plot_attributes = Dict(:fontfamily => "computer modern"), save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :custom_font_irf)

### plots_per_page
plot_irf(Gali_2015_chapter_3_nonlinear, variables = [:Y, :Pi, :R, :C, :N, :W_real, :MC, :i_ann, :A], shocks = :eps_a, plots_per_page = 2, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :two_per_page_irf)

### show_plots
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, show_plots = false)

### save_plots, save_plots_format, save_plots_path, save_pots_name
plot_irf(Gali_2015_chapter_3_nonlinear, save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :impulse_response)

### verbose
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, verbose = true)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, parameters = :β => 0.955, verbose = true)

### tol
using MacroModelling: Tolerances
custom_tol = Tolerances(qme_acceptance_tol = 1e-12, sylvester_acceptance_tol = 1e-12)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, tol = custom_tol, algorithm = :second_order, parameters = :β => 0.9555,verbose = true)

### quadratic_matrix_equation_algorithm
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, quadratic_matrix_equation_algorithm = :doubling, parameters = :β => 0.95555, verbose = true)

### sylvester_algorithm
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order, sylvester_algorithm = :bartels_stewart, verbose = true)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :third_order, sylvester_algorithm = (:doubling, :bicgstab), verbose = true)




# The `rename_dictionary` argument (default: `Dict()`, type: `AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}}`) maps variable or shock symbols to custom display names in plots. This is particularly useful when comparing models with different variable naming conventions, allowing them to be displayed with consistent labels.

# For example, to rename variables for clearer display:

# ```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    rename_dictionary = Dict(:Y => "Output", :Pi => "Inflation", :R => "Interest Rate"), 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_irf)
# ```

# This feature is especially valuable when overlaying IRFs from different models. Consider comparing FS2000 (which uses lowercase variable names like `c`) with Gali_2015_chapter_3_nonlinear (which uses uppercase like `C`). The `rename_dictionary` allows harmonizing these names when plotting them together:

# ```julia
# First model (FS2000) with lowercase variable names
plot_irf(FS2000,
    shocks = :e_m,
    rename_dictionary = Dict(:c => "Consumption", :y => "Output", :R => "Interest Rate"), 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_irf)

# Overlay second model (Gali_2015_chapter_3_nonlinear) with different naming, mapped to same display names
plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_nu,
    rename_dictionary = Dict(:C => "Consumption", :Y => "Output", :R => "Interest Rate"), 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_irf)
# ```

# Both models now appear in the plot with consistent, readable labels, making comparison straightforward.

# The `rename_dictionary` also works with shocks. For example, Gali_2015_chapter_3_nonlinear has shocks `eps_a` and `nu`, while FS2000 has `e_a` and `e_m`. To compare these with consistent labels:

# ```julia
# Gali model with shocks eps_a and nu
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = [:eps_a, :eps_nu],
    rename_dictionary = Dict(:eps_a => "Technology Shock", :eps_nu => "Monetary Policy Shock"), 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_shock_irf)

# FS2000 model with shocks e_a and e_m  
plot_irf!(FS2000,
    shocks = [:e_a, :e_m],
    rename_dictionary = Dict(:e_a => "Technology Shock", :e_m => "Monetary Policy Shock"), 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_shock_irf)
# ```

# The `rename_dictionary` accepts flexible type combinations for keys and values—both `Symbol` and `String` types work interchangeably:

# ```julia
# All of these are valid and equivalent:
Dict(:Y => "Output")              # Symbol key, String value
Dict("Y" => "Output")             # String key, String value
Dict(:Y => :Output)               # Symbol key, Symbol value
Dict("Y" => :Output)              # String key, Symbol value
# ```

# This flexibility is particularly useful for models like Backus_Kehoe_Kydland_1992, which uses both internal symbol representations and more accessible string names with special characters:

# ```julia
# Define the Backus model (abbreviated for clarity)
@model Backus_Kehoe_Kydland_1992 begin
    for co in [H, F]
        Y{co}[0] = ((LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co} * Z{co}[-1]^(-nu{co}))^(-1/nu{co})

        K{co}[0] = (1-delta{co})*K{co}[-1] + S{co}[0]

        X{co}[0] = for lag in (-4+1):0 phi{co} * S{co}[lag] end

        A{co}[0] = (1-eta{co}) * A{co}[-1] + N{co}[0]

        L{co}[0] = 1 - alpha{co} * N{co}[0] - (1-alpha{co})*eta{co} * A{co}[-1]

        U{co}[0] = (C{co}[0]^mu{co}*L{co}[0]^(1-mu{co}))^gamma{co}

        psi{co} * mu{co} / C{co}[0]*U{co}[0] = LGM[0]

        psi{co} * (1-mu{co}) / L{co}[0] * U{co}[0] * (-alpha{co}) = - LGM[0] * (1-theta{co}) / N{co}[0] * (LAMBDA{co}[0] * K{co}[-4]^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co})*Y{co}[0]^(1+nu{co})

        for lag in 0:(4-1)  
            beta{co}^lag * LGM[lag]*phi{co}
        end +
        for lag in 1:4
            -beta{co}^lag * LGM[lag] * phi{co} * (1-delta{co})
        end = beta{co}^4 * LGM[+4] * theta{co} / K{co}[0] * (LAMBDA{co}[+4] * K{co}[0]^theta{co} * N{co}[+4]^(1-theta{co})) ^ (-nu{co})* Y{co}[+4]^(1+nu{co})

        LGM[0] = beta{co} * LGM[+1] * (1+sigma{co} * Z{co}[0]^(-nu{co}-1)*Y{co}[+1]^(1+nu{co}))

        NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1]))/Y{co}[0]
    end

    (LAMBDA{H}[0]-1) = rho{H}{H}*(LAMBDA{H}[-1]-1) + rho{H}{F}*(LAMBDA{F}[-1]-1) + Z_E{H} * E{H}[x]

    (LAMBDA{F}[0]-1) = rho{F}{F}*(LAMBDA{F}[-1]-1) + rho{F}{H}*(LAMBDA{H}[-1]-1) + Z_E{F} * E{F}[x]

    for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end
end

@parameters Backus_Kehoe_Kydland_1992 begin
    K_ss = 11
    K[ss] = K_ss | beta
    
    mu      =    0.34
    gamma   =    -1.0
    alpha   =    1
    eta     =    0.5
    theta   =    0.36
    nu      =    3
    sigma   =    0.01
    delta   =    0.025
    phi     =    1/4
    psi     =    0.5

    Z_E = 0.00852
    
    rho{H}{H} = 0.906
    rho{F}{F} = rho{H}{H}
    rho{H}{F} = 0.088
    rho{F}{H} = rho{H}{F}
end

# Backus model example showing String to String mapping
plot_irf(Backus_Kehoe_Kydland_1992,
    shocks = "E{H}",
    rename_dictionary = Dict("C{H}" => "Home Consumption", 
                             "C{F}" => "Foreign Consumption",
                             "Y{H}" => "Home Output",
                             "Y{F}" => "Foreign Output"), 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_irf)
# ```

# This flexibility allows natural usage regardless of whether variables are referenced as symbols or strings in the code. Variables or shocks not included in the dictionary retain their default names. The renaming applies to all plot elements including legends, axis labels, and tables.

# ```julia
# First shock in the scenario
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a)

# Add second shock to show cumulative effect
plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_nu,
    plot_type = :stack, 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :stack)
# ```

# The `:stack` visualization shows how each shock contributes to the total response, with the second shock's effect layered on top of the first.

#### Using `:compare` for Parameter Comparisons

# When comparing IRFs across different parameter values, `:compare` displays the responses as separate lines:

# ```julia
# Baseline parameterization
plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.99,),
    shocks = :eps_a)

# Alternative parameterization for comparison
plot_irf!(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.95,),
    shocks = :eps_a,
    plot_type = :compare, 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :compare)
# ```



# Policy Functions

# The `plot_solution` function visualizes the solution of the model (mapping of past states to present variables) around the relevant steady state (e.g. higher order perturbation algorithms are centred around the stochastic steady state).

# The relevant steady state is plotted along with the mapping from the chosen past state to one present variable per plot. All other (non-chosen) states remain in the relevant steady state.

# In the case of pruned higher order solutions there are as many (latent) state vectors as the perturbation order. The first and third order baseline state vectors are the non-stochastic steady state and the second order baseline state vector is the stochastic steady state. Deviations for the chosen state are only added to the first order baseline state. The plot shows the mapping from `σ` standard deviations (first order) added to the first order non-stochastic steady state and the present variables. Note that there is no unique mapping from the "pruned" states and the "actual" reported state. Hence, the plots shown are just one realisation of infinitely many possible mappings.

# If the model contains occasionally binding constraints and `ignore_obc = false` they are enforced using shocks.

## Basic Usage

# First, define and load a model:


@model Gali_2015_chapter_3_nonlinear begin
	W_real[0] = C[0] ^ σ * N[0] ^ φ
	Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]
	R[0] = 1 / Q[0]
	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)
	R[0] = Pi[1] * realinterest[0]
	R[0] = 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0])
	C[0] = Y[0]
	log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]
	log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]
	nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]
	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])
	1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)
	S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]
	Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)
	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]
	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]
	log_y[0] = log(Y[0])
	log_W_real[0] = log(W_real[0])
	log_N[0] = log(N[0])
	pi_ann[0] = 4 * log(Pi[0])
	i_ann[0] = 4 * log(R[0])
	r_real_ann[0] = 4 * log(realinterest[0])
	M_real[0] = Y[0] / R[0] ^ η
end

@parameters Gali_2015_chapter_3_nonlinear begin
	σ = 1
	φ = 5
	ϕᵖⁱ = 1.5
	ϕʸ = 0.125
	θ = 0.75
	ρ_ν = 0.5
	ρ_z = 0.5
	ρ_a = 0.9
	β = 0.99
	η = 3.77
	α = 0.25
	ϵ = 9
	τ = 0
    std_a = .01
    std_z = .05
    std_nu = .0025
end


plot_solution(Gali_2015_chapter_3_nonlinear, :A,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets")

# ![Gali 2015 solution](../assets/solution__Gali_2015_chapter_3_nonlinear__A__1.png)

# The function plots each endogenous variable in period `t` against the state variable `A` in `t-1`. Each subplot shows how the variable changes on the y-axis as `A` varies within the specified range over the x-axis. The relevant steady state is indicated by a circle of the same color as the line. The title of each subplot indicates the variable name and the title of the overall plot indicates the model name, and page number (if multiple pages are needed). The legend below the plots indicate the solution algorithm used and the nature of the steady state (stochastic or non-stochastic).

## Function Arguments

plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :Pi],
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :selection)


# ![Gali 2015 solution - selected variables (Y, Pi)](../assets/selection__Gali_2015_chapter_3_nonlinear__A__1.png)



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


plot_solution(FS2000, :k,
	variables = :all_excluding_obc,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :aux)


# ![FS2000 solution - including auxiliary variables](../assets/aux__FS2000__k__1.png)


@model Gali_2015_chapter_3_obc begin
	W_real[0] = C[0] ^ σ * N[0] ^ φ
	Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]
	R[0] = 1 / Q[0]
	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)
	R[0] = Pi[1] * realinterest[0]
	R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))
	C[0] = Y[0]
	log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]
	log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]
	nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]
	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])
	1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)
	S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]
	Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)
	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]
	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]
	log_y[0] = log(Y[0])
	log_W_real[0] = log(W_real[0])
	log_N[0] = log(N[0])
	pi_ann[0] = 4 * log(Pi[0])
	i_ann[0] = 4 * log(R[0])
	r_real_ann[0] = 4 * log(realinterest[0])
	M_real[0] = Y[0] / R[0] ^ η
end

@parameters Gali_2015_chapter_3_obc begin
    R̄ = 1.0
	σ = 1
	φ = 5
	ϕᵖⁱ = 1.5
	ϕʸ = 0.125
	θ = 0.75
	ρ_ν = 0.5
	ρ_z = 0.5
	ρ_a = 0.9
	β = 0.99
	η = 3.77
	α = 0.25
	ϵ = 9
	τ = 0
    std_a = .01
    std_z = .05
    std_nu = .0025
    R > 1.0001
end


# Plotting the IRF for all variables including OBC-related ones reveals the OBC-related auxiliary variables:


plot_solution(Gali_2015_chapter_3_obc, :A,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :obc_variables)

# ![Gali 2015 OBC solution - eps_z shock with OBC variables](../assets/.png)

# The OBC-related variables appear in the last subplot.
# Note that with the `eps_z` shock, the interest rate `R` hits the effective lower bound in period 1:

# ![Gali 2015 OBC solution - eps_z shock hitting lower bound](../assets/.png)



### Solution Algorithm


# Plot first-order policy function
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :first_order)

# Overlay second-order to compare
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :second_order,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :mult_algorithms)


# ![Gali 2015 solution - multiple solution methods](../assets/compare_orders_solution__Gali_2015_chapter_3_nonlinear__2.png)


plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :pruned_third_order,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :mult_algorithms_third_order)


# ![Gali 2015 IRF - eps_a shock (multiple orders)](../assets/multiple_orders_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)


plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    σ = 5,
    algorithm = :pruned_third_order,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :range)


# ![Gali 2015 IRF - eps_a shock (multiple orders)](../assets/multiple_orders_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)


plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = :β => 0.95,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :params)



# ![Gali 2015 solution - different parameter values](../assets/different_parameters_solution__Gali_2015_chapter_3_nonlinear__1.png)


# Plot with default parameters
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = :β => 0.99)

# Overlay with different discount factor to compare
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    parameters = :β => 0.95,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :params_compare)


# ![Gali 2015 IRF - eps_a shock comparing β values](../assets/compare_beta_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)


plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    parameters = (:β => 0.97, :τ => 0.5),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :mult_params_compare)



# ![Gali 2015 IRF - eps_a shock with multiple parameter changes](../assets/multi_params_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)


plot_solution(Gali_2015_chapter_3_obc, :A)


# ![Gali 2015 OBC IRF - eps_z shock with OBC](../assets/obc_irf__Gali_2015_chapter_3_obc__eps_z__1.png)


plot_solution!(Gali_2015_chapter_3_obc, :A,
    ignore_obc = true,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :obc_ignore)


# ![Gali 2015 OBC IRF - eps_z shock comparing with and without OBC](../assets/compare_obc_irf__Gali_2015_chapter_3_obc__eps_z__1.png)

plot_solution(Gali_2015_chapter_3_obc, :A,
    algorithm = :pruned_second_order,
    parameters = :β => 0.99,
    label = "2nd Order with OBC"
    )

# Add solution without OBC
plot_solution!(Gali_2015_chapter_3_obc, :A,
    algorithm = :pruned_second_order,
    ignore_obc = true,
    label = "2nd Order without OBC"
    )

# Add different parameter setting
plot_solution!(Gali_2015_chapter_3_obc, :A,
    algorithm = :pruned_second_order,
    parameters = :β => 0.9925,
    label = "2nd Order with OBC and β=0.9925",
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :labels
    )


ec_color_palette =
[
	"#FFD724", 	# "Sunflower Yellow"
	"#353B73", 	# "Navy Blue"
	"#2F9AFB", 	# "Sky Blue"
	"#B8AAA2", 	# "Taupe Grey"
	"#E75118", 	# "Vermilion"
	"#6DC7A9", 	# "Mint Green"
	"#F09874", 	# "Coral"
	"#907800"  	# "Olive"
]
    
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    plot_attributes = Dict(:palette => ec_color_palette))
    
for a in [:second_order, :third_order]
    plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
        algorithm = a,
        plot_attributes = Dict(:palette => ec_color_palette),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :attr
)
end

# ![Gali 2015 IRF - all shocks with custom color palette](../assets/custom_colors_irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__2.png)


plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    plot_attributes = Dict(:fontfamily => "computer modern"),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :attr_font)


plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    plot_attributes = Dict(:linestyle => :dashdot, :linewidth => 2),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :attr_line)

# ![Gali 2015 IRF - eps_a shock with custom font](../assets/custom_font_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)


plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :Pi, :R, :C],
    plots_per_page = 2,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :two_per_page)


# ![Gali 2015 IRF - eps_a shock (2 plots per page)](../assets/two_per_page_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)


plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    rename_dictionary = Dict(:Y => "Output", :Pi => "Inflation", :R => "Interest Rate"),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict)


# ![Gali 2015 IRF - eps_z shock rename dictionary](../assets/rename_dict_irf__Gali_2015_chapter_3_nonlinear__eps_z__1.png)



@model Caldara_et_al_2012 begin
	V[0] = ((1 - β) * (c[0] ^ ν * (1 - l[0]) ^ (1 - ν)) ^ (1 - 1 / ψ) + β * V[1] ^ (1 - 1 / ψ)) ^ (1 / (1 - 1 / ψ))
	exp(s[0]) = V[1] ^ (1 - γ)
	1 = (1 + ζ * exp(z[1]) * k[0] ^ (ζ - 1) * l[1] ^ (1 - ζ) - δ) * c[0] * β * (((1 - l[1]) / (1 - l[0])) ^ (1 - ν) * (c[1] / c[0]) ^ ν) ^ (1 - 1 / ψ) / c[1]
	Rᵏ[0] = ζ * exp(z[1]) * k[0] ^ (ζ - 1) * l[1] ^ (1 - ζ) - δ
	SDF⁺¹[0] = c[0] * β * (((1 - l[1]) / (1 - l[0])) ^ (1 - ν) * (c[1] / c[0]) ^ ν) ^ (1 - 1 / ψ) / c[1]
	1 + Rᶠ[0] = 1 / SDF⁺¹[0]
	(1 - ν) / ν * c[0] / (1 - l[0]) = (1 - ζ) * exp(z[0]) * k[-1] ^ ζ * l[0] ^ (-ζ)
	c[0] + i[0] = exp(z[0]) * k[-1] ^ ζ * l[0] ^ (1 - ζ)
	k[0] = i[0] + k[-1] * (1 - δ)
	z[0] = λ * z[-1] + σ[0] * ϵᶻ[x]
	y[0] = exp(z[0]) * k[-1] ^ ζ * l[0] ^ (1 - ζ)
	log(σ[0]) = (1 - ρ) * log(σ̄) + ρ * log(σ[-1]) + η * ω[x]
    gross_r[0] = 1 + Rᶠ[0]
end

@parameters Caldara_et_al_2012 begin
	β = 0.991
	l[ss] = 1/3 | ν
	ζ = 0.3
	δ = 0.0196
	λ = 0.95
	ψ = 0.5
	γ = 40
	σ̄ = 0.021
	η = 0.1
	ρ = 0.9
end


# First model (FS2000) with lowercase variable names
plot_solution(FS2000, :k,
    variables = [:c, :y, :R],
    rename_dictionary = Dict(:c => "Consumption", :y => "Output", :R => "Interest Rate"))

# Overlay second model (Caldara_et_al_2012) with different naming, mapped to same display names
plot_solution!(Caldara_et_al_2012, :k,
    variables = [:c, :y, :gross_r],
    rename_dictionary = Dict(:c => "Consumption", :y => "Output", :gross_r => "Interest Rate"),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_combine)


# ![FS2000 and Gali 2015 IRF - multiple models with rename dictionary](../assets/rename_dict_irf__multiple_models__multiple_shocks__1.png)


# Define the Backus model (abbreviated for clarity)
@model Backus_Kehoe_Kydland_1992 begin
    for co in [H, F]
        Y{co}[0] = ((LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co} * Z{co}[-1]^(-nu{co}))^(-1/nu{co})
        K{co}[0] = (1-delta{co})*K{co}[-1] + S{co}[0]
        X{co}[0] = for lag in (-4+1):0 phi{co} * S{co}[lag] end
        A{co}[0] = (1-eta{co}) * A{co}[-1] + N{co}[0]
        L{co}[0] = 1 - alpha{co} * N{co}[0] - (1-alpha{co})*eta{co} * A{co}[-1]
        U{co}[0] = (C{co}[0]^mu{co}*L{co}[0]^(1-mu{co}))^gamma{co}
        psi{co} * mu{co} / C{co}[0]*U{co}[0] = LGM[0]
        psi{co} * (1-mu{co}) / L{co}[0] * U{co}[0] * (-alpha{co}) = - LGM[0] * (1-theta{co}) / N{co}[0] * (LAMBDA{co}[0] * K{co}[-4]^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co})*Y{co}[0]^(1+nu{co})

        for lag in 0:(4-1)  
            beta{co}^lag * LGM[lag]*phi{co}
        end +
        for lag in 1:4
            -beta{co}^lag * LGM[lag] * phi{co} * (1-delta{co})
        end = beta{co}^4 * LGM[+4] * theta{co} / K{co}[0] * (LAMBDA{co}[+4] * K{co}[0]^theta{co} * N{co}[+4]^(1-theta{co})) ^ (-nu{co})* Y{co}[+4]^(1+nu{co})

        LGM[0] = beta{co} * LGM[+1] * (1+sigma{co} * Z{co}[0]^(-nu{co}-1)*Y{co}[+1]^(1+nu{co}))
        NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1]))/Y{co}[0]
    end

    (LAMBDA{H}[0]-1) = rho{H}{H}*(LAMBDA{H}[-1]-1) + rho{H}{F}*(LAMBDA{F}[-1]-1) + Z_E{H} * E{H}[x]
    (LAMBDA{F}[0]-1) = rho{F}{F}*(LAMBDA{F}[-1]-1) + rho{F}{H}*(LAMBDA{H}[-1]-1) + Z_E{F} * E{F}[x]

    for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end
end

@parameters Backus_Kehoe_Kydland_1992 begin
    K_ss = 11
    K[ss] = K_ss | beta
    
    mu      =    0.34
    gamma   =    -1.0
    alpha   =    1
    eta     =    0.5
    theta   =    0.36
    nu      =    3
    sigma   =    0.01
    delta   =    0.025
    phi     =    1/4
    psi     =    0.5

    Z_E = 0.00852
    
    rho{H}{H} = 0.906
    rho{F}{F} = rho{H}{H}
    rho{H}{F} = 0.088
    rho{F}{H} = rho{H}{F}
end

# Backus model example showing String to String mapping
plot_solution(Backus_Kehoe_Kydland_1992, "K{H}",
    rename_dictionary = Dict("C{H}" => "Home Consumption", 
                             "C{F}" => "Foreign Consumption",
                             "Y{H}" => "Home Output",
                             "Y{F}" => "Foreign Output"),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_string)


# ![Backus IRF - E{H} shock with rename dictionary](../assets/rename_dict_irf__Backus_Kehoe_Kydland_1992__E{H}__1.png)


# Conditional Variance Decomposition
@model Smets_Wouters_2007_linear begin
    a[0] = calfa * rkf[0] + (1 - calfa) * wf[0]
    zcapf[0] = rkf[0] * 1 / (czcap / (1 - czcap))
    rkf[0] = wf[0] + labf[0] - kf[0]
    kf[0] = zcapf[0] + kpf[-1]
    invef[0] = qs[0] + 1 / (1 + cgamma * cbetabar) * (pkf[0] * 1 / (csadjcost * cgamma ^ 2) + invef[-1] + invef[1] * cgamma * cbetabar)
    pkf[0] = b[0] * (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) - rrf[0] + rkf[1] * (crk / (crk + (1 - ctou))) + pkf[1] * ((1 - ctou) / (crk + (1 - ctou)))
    cf[0] = b[0] + cf[-1] * chabb / cgamma / (1 + chabb / cgamma) + cf[1] * 1 / (1 + chabb / cgamma) + (labf[0] - labf[1]) * ((csigma - 1) * cwhlc / (csigma  *(1 + chabb / cgamma))) - rrf[0] * (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))
    yf[0] = g[0] + cf[0] * ccy + invef[0] * ciy + zcapf[0] * crkky
    yf[0] = cfc * (a[0] + calfa * kf[0] + (1 - calfa) * labf[0])
    wf[0] = labf[0] * csigl + cf[0] * 1 / (1 - chabb / cgamma) - cf[-1] * chabb / cgamma / (1 - chabb / cgamma)
    kpf[0] = kpf[-1] * (1 - cikbar) + invef[0] * cikbar + qs[0] * csadjcost * cgamma ^ 2 * cikbar
    mc[0] = calfa * rk[0] + (1 - calfa) * w[0] - a[0]
    zcap[0] = 1 / (czcap / (1 - czcap)) * rk[0]
    rk[0] = w[0] + lab[0] - k[0]
    k[0] = zcap[0] + kp[-1]
    inve[0] = qs[0] + 1 / (1 + cgamma * cbetabar) * (pk[0] * 1 / (csadjcost * cgamma ^ 2) + inve[-1] + inve[1] * cgamma * cbetabar)
    pk[0] = pinf[1] - r[0] + b[0] * 1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) + rk[1] * (crk / (crk + (1 - ctou))) + pk[1] * ((1 - ctou) / (crk + (1 - ctou)))
    c[0] = b[0] + c[-1] * chabb / cgamma / (1 + chabb / cgamma) + c[1] * 1 / (1 + chabb / cgamma) + 
    (lab[0] - lab[1]) * ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) - (r[0] - pinf[1]) * (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))
    y[0] = g[0] + c[0] * ccy + inve[0] * ciy + zcap[0] * crkky
    y[0] = cfc * (a[0] + calfa * k[0] + (1 - calfa) * lab[0])
    pinf[0] = spinf[0] + 1 / (1 + cindp * cgamma * cbetabar) * (cindp * pinf[-1] + pinf[1] * cgamma * cbetabar + mc[0] * (1 - cprobp) * (1 - cprobp * cgamma * cbetabar) / cprobp / (1 + (cfc - 1) * curvp))
    w[0] = sw[0] + w[-1] * 1 / (1 + cgamma * cbetabar) + w[1] * cgamma * cbetabar / (1 + cgamma * cbetabar) + pinf[-1] * cindw / (1 + cgamma * cbetabar) - pinf[0] * (1 + cindw * cgamma * cbetabar) / (1 + cgamma * cbetabar) + pinf[1] * cgamma * cbetabar / (1 + cgamma * cbetabar) + (csigl * lab[0] + c[0] * 1 / (1 - chabb / cgamma) - c[-1] * chabb / cgamma / (1 - chabb / cgamma) - w[0]) * 1 / (1 + (clandaw - 1) * curvw) * (1 - cprobw) * (1 - cprobw * cgamma * cbetabar) / (cprobw * (1 + cgamma * cbetabar))
    r[0] = pinf[0] * crpi * (1 - crr) + (1 - crr) * cry * (y[0] - yf[0]) + crdy * (y[0] - yf[0] - y[-1] + yf[-1]) + crr * r[-1] + ms[0]
    a[0] = crhoa * a[-1] + z_ea * ea[x]
    b[0] = crhob * b[-1] + z_eb * eb[x]
    g[0] = crhog * g[-1] + z_eg * eg[x] + z_ea * ea[x] * cgy
    qs[0] = crhoqs * qs[-1] + z_eqs * eqs[x]
    ms[0] = crhoms * ms[-1] + z_em * em[x]
    spinf[0] = crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]
    epinfma[0] = z_epinf * epinf[x]
    sw[0] = crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]
    ewma[0] = z_ew * ew[x]
    kp[0] = kp[-1] * (1 - cikbar) + inve[0] * cikbar + qs[0] * csadjcost * cgamma ^ 2 * cikbar
    dy[0] = ctrend + y[0] - y[-1]
    dc[0] = ctrend + c[0] - c[-1]
    dinve[0] = ctrend + inve[0] - inve[-1]
    pinfobs[0] = constepinf + pinf[0]
    robs[0] = r[0] + conster
    dwobs[0] = ctrend + w[0] - w[-1]
    labobs[0] = lab[0] + constelab
end

@parameters Smets_Wouters_2007_linear begin
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
    z_ea	= 0.4618
    z_eb	= 1.8513
    z_eg	= 0.6090
    z_em	= 0.2397
    z_ew	= 0.2089
    z_eqs	= 0.6017
    z_epinf	= 0.1455
    cpie 	= 1 + constepinf / 100         							# gross inflation rate
    cgamma 	= 1 + ctrend / 100          							# gross growth rate
    cbeta 	= 1 / (1 + constebeta / 100)    						# discount factor
    clandap = cfc                									# fixed cost share/gross price markup
    cbetabar= cbeta * cgamma ^ (-csigma)   							# growth-adjusted discount factor in Euler equation
    cr 		= cpie / cbetabar  										# steady state gross real interest rate
    crk 	= 1 / cbetabar - (1 - ctou) 							# steady state rental rate
    cw 		= (calfa ^ calfa * (1 - calfa) ^ (1 - calfa) / (clandap * crk ^ calfa)) ^ (1 / (1 - calfa))	# steady state real wage
    cikbar 	= 1 - (1 - ctou) / cgamma								# (1-k_1) in equation LOM capital, equation (8)
    cik 	= cikbar * cgamma										# i_k: investment-capital ratio
    clk 	= (1 - calfa) / calfa * crk / cw						# labor to capital ratio
    cky 	= cfc * clk ^ (calfa - 1)								# k_y: steady state output ratio
    ciy 	= cik * cky												# investment-output ratio
    ccy 	= 1 - cg - cik * cky									# consumption-output ratio
    crkky 	= crk * cky												# z_y=R_{*}^k*k_y
    cwhlc 	= (1 / clandaw) * (1 - calfa) / calfa * crk * cky / ccy	# W^{h}_{*}*L_{*}/C_{*} used in c_2 in equation (2)
    conster = (cr - 1) * 100										# steady state federal funds rate ($\bar r$)
end


plot_conditional_variance_decomposition(Smets_Wouters_2007_linear,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets")


plot_fevd(Smets_Wouters_2007_linear, periods = 12,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :short_period)


plot_fevd(Smets_Wouters_2007_linear,
    variables = [:inve, :c, :y, :pinf, :w, :lab],
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :var_select)



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

plot_fevd(FS2000,
	variables = :all_excluding_obc,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :aux)


plot_fevd(Smets_Wouters_2007_linear,
    parameters = :z_eg => 1,
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :param_change)


plot_fevd(Smets_Wouters_2007_linear,
    parameters = (:z_eg => 1.5, :crpi => 1.75),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :param_change_2)


ec_color_palette =
[
	"#FFD724", 	# "Sunflower Yellow"
	"#353B73", 	# "Navy Blue"
	"#2F9AFB", 	# "Sky Blue"
	"#B8AAA2", 	# "Taupe Grey"
	"#E75118", 	# "Vermilion"
	"#6DC7A9", 	# "Mint Green"
	"#F09874", 	# "Coral"
	"#907800"  	# "Olive"
]


plot_fevd(Smets_Wouters_2007_linear,
    plot_attributes = Dict(:palette => ec_color_palette),
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :color_palette)



plot_fevd(Smets_Wouters_2007_linear,
    plot_attributes = Dict(:fontfamily => "computer modern"),
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :font_family)


plot_fevd(Smets_Wouters_2007_linear,
    plot_attributes = Dict(:fillalpha => .5),
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :fill_alpha)



plot_fevd(Smets_Wouters_2007_linear,
    variables = [:inve, :c, :y, :pinf, :w, :lab],
    plots_per_page = 4,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :four_per_page)



plot_fevd(Smets_Wouters_2007_linear,
    rename_dictionary = Dict(:y => "Output", :pinfobs => "Inflation", :robs => "Interest Rate", :inve => "Investment", :c => "Consumption", :w => "Wages", :lab => "Labor"),
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_fevd)

@model Backus_Kehoe_Kydland_1992 begin
    for co in [H, F]
        Y{co}[0] = ((LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co} * Z{co}[-1]^(-nu{co}))^(-1/nu{co})
        K{co}[0] = (1-delta{co})*K{co}[-1] + S{co}[0]
        X{co}[0] = for lag in (-4+1):0 phi{co} * S{co}[lag] end
        A{co}[0] = (1-eta{co}) * A{co}[-1] + N{co}[0]
        L{co}[0] = 1 - alpha{co} * N{co}[0] - (1-alpha{co})*eta{co} * A{co}[-1]
        U{co}[0] = (C{co}[0]^mu{co}*L{co}[0]^(1-mu{co}))^gamma{co}
        psi{co} * mu{co} / C{co}[0]*U{co}[0] = LGM[0]
        psi{co} * (1-mu{co}) / L{co}[0] * U{co}[0] * (-alpha{co}) = - LGM[0] * (1-theta{co}) / N{co}[0] * (LAMBDA{co}[0] * K{co}[-4]^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co})*Y{co}[0]^(1+nu{co})

        for lag in 0:(4-1)  
            beta{co}^lag * LGM[lag]*phi{co}
        end +
        for lag in 1:4
            -beta{co}^lag * LGM[lag] * phi{co} * (1-delta{co})
        end = beta{co}^4 * LGM[+4] * theta{co} / K{co}[0] * (LAMBDA{co}[+4] * K{co}[0]^theta{co} * N{co}[+4]^(1-theta{co})) ^ (-nu{co})* Y{co}[+4]^(1+nu{co})

        LGM[0] = beta{co} * LGM[+1] * (1+sigma{co} * Z{co}[0]^(-nu{co}-1)*Y{co}[+1]^(1+nu{co}))
        NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1]))/Y{co}[0]
    end

    (LAMBDA{H}[0]-1) = rho{H}{H}*(LAMBDA{H}[-1]-1) + rho{H}{F}*(LAMBDA{F}[-1]-1) + Z_E{H} * E{H}[x]
    (LAMBDA{F}[0]-1) = rho{F}{F}*(LAMBDA{F}[-1]-1) + rho{F}{H}*(LAMBDA{H}[-1]-1) + Z_E{F} * E{F}[x]

    for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end
end

@parameters Backus_Kehoe_Kydland_1992 begin
    K_ss = 11
    K[ss] = K_ss | beta
    
    mu      =    0.34
    gamma   =    -1.0
    alpha   =    1
    eta     =    0.5
    theta   =    0.36
    nu      =    3
    sigma   =    0.01
    delta   =    0.025
    phi     =    1/4
    psi     =    0.5

    Z_E = 0.00852
    
    rho{H}{H} = 0.906
    rho{F}{F} = rho{H}{H}
    rho{H}{F} = 0.088
    rho{F}{H} = rho{H}{F}
end

# Backus model example showing String to String mapping
plot_fevd(Backus_Kehoe_Kydland_1992,
    rename_dictionary = Dict("K{H}" => "Capital (Home)", 
                             "K{F}" => "Capital (Foreign)",
                             "Y{H}" => "Output (Home)",
                             "Y{F}" => "Output (Foreign)"),
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :rename_dict_string)
