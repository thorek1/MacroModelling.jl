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

# The `plot_solution` function visualizes policy functions by plotting the relationship between a state variable and endogenous variables. This shows how variables respond to changes in a state variable around the steady state, revealing the model's decision rules.

## Basic Usage

# First, define and load a model:

# ```julia
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
# ```

# Calling `plot_solution` requires specifying a state variable. By default, it plots **all endogenous variables** as functions of the specified state over a range of ±2 standard deviations:

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :c, 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :solution)
# ```

# The function plots each endogenous variable against the state variable `A`. Each subplot shows how the variable changes as `A` varies within the specified range. The steady state is indicated by horizontal and vertical reference lines.

## Function Arguments

### State Variable (Required)

# The `state` argument (type: `Union{Symbol, String}`) specifies which state variable to vary. This must be a state variable from the model (variables with lagged values).

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :nu)  # Using Symbol
plot_solution(Gali_2015_chapter_3_nonlinear, "nu", 
	save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :solution) # Using String
# ```

### Variables to Plot

# The `variables` argument (default: `:all_excluding_obc`, type: `Union{Symbol, String, Vector{Symbol}, Vector{String}}`) determines which endogenous variables to display.

# Available options: `:all`, `:all_excluding_obc`, `:all_excluding_aux`, `:all_excluding_aux_and_obc`, or specify variables explicitly using symbols, strings, or vectors.

# Select specific variables:

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C, :Pi])
# ```

# Plot all variables including auxiliary variables:

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = :all)
# ```

### Solution Algorithm

# The `algorithm` argument (default: `:first_order`, type: `Symbol`) specifies which algorithm to solve for the dynamics of the model. Available algorithms: `:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, `:pruned_third_order`.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :second_order)
# ```

# At higher orders, policy functions become nonlinear, showing how the response varies across different states.

### State Variable Range

# The `σ` argument (default: `2`, type: `Union{Int64, Float64}`) specifies the range of the state variable as a multiple of its standard deviation. The state variable varies from `-σ * std(state)` to `+σ * std(state)`.

# Plot over a wider range (±3 standard deviations):

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    σ = 3)
# ```

# Plot over a narrower range (±1 standard deviation):

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    σ = 1)
# ```

### Alternative Parameters

# The `parameters` argument (default: `nothing`, type: `Union{Nothing, Vector{Float64}, Vector{Int64}}`) allows plotting with different parameter values without modifying the model.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = [1, 5, 1.5, 0.125, 0.75, 0.5, 0.5, 0.9, 0.99, 3.77, 0.25, 9, 0.5, 0.01, 0.05, 0.0025])
# ```

# The parameter vector must match the model's parameter order and length.

### Occasionally Binding Constraints

# The `ignore_obc` argument (default: `false`, type: `Bool`) determines whether to ignore occasionally binding constraints when solving the model.

# ```julia
plot_solution(model_with_obc, :state,
    ignore_obc = true)
# ```

### Plot Labels

# The `label` argument (default: `""`, type: `Union{Real, String, Symbol}`) adds custom labels to the plot legend. This is useful when comparing multiple solutions using `plot_solution!` to overlay plots:

# ```julia
# Plot first-order solution
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C, :Pi],
    label = "First Order")

# Add second-order solution
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C, :Pi],
    algorithm = :second_order,
    label = "Second Order")

# Add third-order solution
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C, :Pi],
    algorithm = :third_order,
    label = "Third Order")
# ```

# This allows direct comparison of how policy functions differ across solution methods, revealing the importance of nonlinearities in the model.

### Display Control

# The `show_plots` argument (default: `true`, type: `Bool`) controls whether plots are displayed in the plotting pane.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    show_plots = false)  # Generate plots without displaying
# ```

### Saving Plots

# The `save_plots` argument (default: `false`, type: `Bool`) determines whether to save plots to disk.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    save_plots = true)
# ```

#### Save Plot Format

# The `save_plots_format` argument (default: `:pdf`, type: `Symbol`) specifies the file format for saved plots. Common formats: `:pdf`, `:png`, `:svg`.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    save_plots = true,
    save_plots_format = :png)
# ```

#### Save Plot Name

# The `save_plots_name` argument (default: `"solution"`, type: `Union{String, Symbol}`) specifies the prefix for saved plot filenames. The filename format is: `prefix__ModelName__state__page.format`.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    save_plots = true,
    save_plots_name = "policy_A")
# Creates: policy_A__Gali_2015_chapter_3_nonlinear__A__1.pdf
# ```

#### Save Plot Path

# The `save_plots_path` argument (default: `"."`, type: `String`) specifies the directory where plots are saved.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    save_plots = true,
    save_plots_path = "plots/policy_functions")
# ```

### Plots Per Page

# The `plots_per_page` argument (default: `6`, type: `Int`) controls how many subplots appear on each page. Useful for managing large numbers of variables.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    plots_per_page = 9)  # 3x3 grid
# ```

### Variable and Shock Renaming

# The `rename_dictionary` argument (default: `Dict()`, type: `AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}}`) allows renaming variables and shocks in plot labels for clearer display.

# Basic renaming for readable labels:

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    rename_dictionary = Dict(:Y => "Output", :C => "Consumption", :Pi => "Inflation"))
# ```

# This feature is particularly useful when comparing models with different variable naming conventions. For example, when overlaying policy functions from FS2000 (which uses lowercase `c` for consumption) and Gali_2015_chapter_3_nonlinear (which uses uppercase `C`):

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:C, :Y],
    rename_dictionary = Dict(:C => "Consumption", :Y => "Output"))

plot_solution!(FS2000, :e_a,
    variables = [:c, :y],
    rename_dictionary = Dict(:c => "Consumption", :y => "Output"))
# ```

# The `rename_dictionary` accepts flexible type combinations for keys and values. The following are all equivalent:

# ```julia
# Symbol keys, String values
rename_dictionary = Dict(:Y => "Output")

# String keys, String values
rename_dictionary = Dict("Y" => "Output")

# Symbol keys, Symbol values
rename_dictionary = Dict(:Y => :Output)

# String keys, Symbol values
rename_dictionary = Dict("Y" => :Output)
# ```

# For models with special characters in variable names (like the Backus_Kehoe_Kydland_1992 model which uses symbols like `Symbol("C{H}")`):

# ```julia
plot_solution(Backus_Kehoe_Kydland_1992, :K,
    rename_dictionary = Dict(
        Symbol("C{H}") => "Home Consumption",
        Symbol("C{F}") => "Foreign Consumption"))
# ```

# The renaming applies to all plot elements: legends, axis labels, and tables.

### Custom Plot Attributes

# The `plot_attributes` argument (default: `Dict()`, type: `Dict`) allows passing additional styling attributes to the plotting backend.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    plot_attributes = Dict(
        :linewidth => 3,
        :linestyle => :dash,
        :color => :red))
# ```

### Verbosity

# The `verbose` argument (default: `true`, type: `Bool`) controls whether to print progress messages during computation.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    verbose = false)  # Suppress output
# ```

### Numerical Tolerance

# The `tol` argument (default: `Tolerances()`, type: `Tolerances`) specifies numerical tolerance settings for the solver.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    tol = Tolerances(tol = 1e-12))
# ```

### Quadratic Matrix Equation Solver

# The `quadratic_matrix_equation_algorithm` argument (default: `:bicgstab`, type: `Symbol`) specifies which algorithm to use for solving quadratic matrix equations in higher-order solutions.

# Available algorithms: `:bicgstab`, `:gmres`, `:dqgmres`.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :second_order,
    quadratic_matrix_equation_algorithm = :gmres)
# ```

### Sylvester Equation Solver

# The `sylvester_algorithm` argument (default: depends on model size, type: `Union{Symbol, Vector{Symbol}, Tuple{Symbol, Vararg{Symbol}}}`) specifies which algorithm to use for solving Sylvester equations.

# Available algorithms: `:doubling`, `:bartels_stewart`, `:bicgstab`, `:gmres`, `:dqgmres`.

# For second-order solutions, specify a single algorithm:

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :second_order,
    sylvester_algorithm = :bartels_stewart)
# ```

# For third-order solutions, different algorithms can be specified for the second- and third-order Sylvester equations using a `Tuple`:

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :third_order,
    sylvester_algorithm = (:doubling, :bicgstab))
# ```

### Lyapunov Equation Solver

# The `lyapunov_algorithm` argument (default: `:doubling`, type: `Symbol`) specifies which algorithm to use for solving Lyapunov equations.

# Available algorithms: `:doubling`, `:bartels_stewart`, `:bicgstab`, `:gmres`, `:dqgmres`.

# ```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    lyapunov_algorithm = :bartels_stewart)
# ```
