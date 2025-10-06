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
plot_irf(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3)
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