# This script contains the Julia code from the plotting.md documentation.
# It is modified to save all plots referenced in the markdown file
# to the docs/assets/ directory, allowing the documentation to be regenerated.

## Setup
# using Revise
using MacroModelling
import StatsPlots
using AxisKeys
import Random; Random.seed!(10) # For reproducibility of :simulate


## README
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

plot_irf(RBC, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        # save_plots_name = :readme_irf
        )

## OBC
@model Gali_2015_chapter_3_obc begin
    W_real[0] = C[0] ^ σ * N[0] ^ φ

    Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]

    R[0] = 1 / Q[0]

    Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

    R[0] = Pi[1] * realinterest[0]

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

    R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))

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

    R > 1.000001
end


Random.seed!(20)
plot_simulations(Gali_2015_chapter_3_obc, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :sim_obc
        )


# ![Simulation_elb](../assets/Gali_2015_chapter_3_obc__simulation__1.png)

Random.seed!(20)
plot_simulations(Gali_2015_chapter_3_obc, parameters = :R̄ => 0.99, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :sim_obc_elb
        )


# ![Simulation_elb2](../assets/Gali_2015_chapter_3_obc__simulation__2.png)


Random.seed!(20)
plot_simulations(Gali_2015_chapter_3_obc, ignore_obc = true, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :sim_ignore_obc
        )


# ![Simulation_no_elb](../assets/Gali_2015_chapter_3_obc__simulation__no.png)

Random.seed!(20)
plot_irf(Gali_2015_chapter_3_obc, shocks = :eps_z, parameters = :R̄ => 1.0, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :obc_irf_higher_bound
        )


# ![IRF_elb](../assets/Gali_2015_chapter_3_obc__eps_z.png)


shcks = zeros(1,15)
shcks[5] =  3.0
shcks[10] = 2.0
shcks[15] = 1.0

sks = KeyedArray(shcks;  Shocks = [:eps_z], Periods = 1:15)  # KeyedArray is provided by the `AxisKeys` package

plot_irf(Gali_2015_chapter_3_obc, 
        shocks = sks, 
        periods = 10, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :obc_irf
        )


# ![Shock_series_elb](../assets/Gali_2015_chapter_3_obc__shock_matrix__1.png)

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

plot_irf(borrowing_constraint, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :obc_irf
        )


# ![Positive_shock](../assets/borrowing_constraint__ε_pos.png)

plot_irf(borrowing_constraint, 
        negative_shock = true, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :obc_neg_irf
        )


# ![Negative_shock](../assets/borrowing_constraint__ε_neg.png)

shcks = zeros(1,30)
shcks[10] =  .6
shcks[30] = -.6

sks = KeyedArray(shcks;  Shocks = [:ε], Periods = 1:30)  # KeyedArray is provided by the `AxisKeys` package

plot_irf(borrowing_constraint, 
        shocks = sks, 
        periods = 50, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :obc_shocks_irf
        )


# ![Simulation](../assets/borrowing_constraint__obc.png)

plot_irf(borrowing_constraint, 
        shocks = sks, 
        periods = 50, 
        ignore_obc = true, 
        save_plots = true, 
        save_plots_path = "./docs/src/assets", 
        save_plots_format = :png, 
        save_plots_name = :obc_shocks_irf_no_obc
        )


# ![Simulation](../assets/borrowing_constraint__no_obc.png)


## Impulse response functions (IRF)
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

plot_irf(Gali_2015_chapter_3_nonlinear, 
            save_plots = true, 
            save_plots_path = "./docs/src/assets", 
            save_plots_format = :png, 
            save_plots_name = :default_irf)


# Plot with baseline parameters
plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = :β => 0.99,
    shocks = :eps_a)

# Add with different algorithm AND parameters
plot_irf!(Gali_2015_chapter_3_nonlinear,
    parameters = :β => 0.95,
    shocks = :eps_a,
    algorithm = :second_order, 
            save_plots = true, 
            save_plots_path = "./docs/src/assets", 
            save_plots_format = :png, 
            save_plots_name = :compare_beta_and_orders)


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
plot_irf(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, save_plots = true, save_plots_path = "./docs/src/assets", save_plots_format = :png, save_plots_name = :obc_irf)
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
Random.seed!(10)

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

# Plot with baseline parameters
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = :β => 0.99)

# Add with different algorithm AND parameters
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    parameters = :β => 0.95,
    algorithm = :second_order,
    save_plots = true, 
    save_plots_format = :png, 
    save_plots_path = "./docs/src/assets", 
    save_plots_name = :compare_beta_and_orders_solution)


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
Random.seed!(10)
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


# Conditional Forecasting
Random.seed!(10)
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


conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),Variables = [:Y], Periods = [1])
conditions[1,1] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,  
                            conditions,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets")

# Set up conditions
conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),
                    Variables = [:Y], 
                    Periods = 1:1)
conditions_ka[1,1] = 1.0

# Plot conditional forecast with baseline parameters
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         parameters = :β => 0.99)

# Add conditional forecast with different discount factor
plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                          conditions_ka,
                          parameters = :β => 0.95,
    save_plots = true, 
    save_plots_format = :png, 
    save_plots_path = "./docs/src/assets", 
    save_plots_name = :cnd_fcst_one_diff)


# Plot with baseline settings
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         parameters = :β => 0.99)

# Add with different algorithm AND parameters
plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                          conditions_ka,
                          parameters = :β => 0.95,
                          algorithm = :second_order,
    save_plots = true, 
    save_plots_format = :png, 
    save_plots_path = "./docs/src/assets", 
    save_plots_name = :cnd_fcst_two_diff)


                          
conditions = Matrix{Union{Nothing,Float64}}(undef,23,8)
conditions[12,1] = 1.0
conditions[12,2] = 1.1
conditions[12,3] = 1.2
conditions[12,4] = 1.3
conditions[12,5] = 1.4
conditions[12,6] = 1.5
conditions[12,7] = 1.6
conditions[12,8] = 1.7

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,  
                            conditions,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_10_periods)

    
conditions_sp = spzeros(23,8)
conditions_sp[12,1] = 1.0
conditions_sp[12,2] = 1.1
conditions_sp[12,3] = 1.2
conditions_sp[12,4] = 1.3
conditions_sp[12,5] = 1.4
conditions_sp[12,6] = 1.5
conditions_sp[12,7] = 1.6
conditions_sp[12,8] = 1.7

conditions_sp[9,8] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,  
                            conditions_sp,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_2_vars)


    
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,  
                            conditions)

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,  
                            conditions_sp,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_plot_overlay)


    
# conditions_diff = spzeros(23,8)
# conditions_diff[9,8] = 1.0

# plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,  
#                             conditions)

# plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,  
#                             conditions_diff,
#                             plot_type = :stack,
#     save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_plot_stack)



conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:R, :Y, :MC], Periods = 1:3)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0
conditions_ka[3,3] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_plot_ka)
                         

    
shocks = Matrix{Union{Nothing,Float64}}(undef,3,8)
shocks[2,:] .= 0
shocks[3,:] .= 0

conditions = Matrix{Union{Nothing,Float64}}(undef,23,8)
conditions[12,1] = 1.0
conditions[12,2] = 1.1
conditions[12,3] = 1.2
conditions[12,4] = 1.3
conditions[12,5] = 1.4
conditions[12,6] = 1.5
conditions[12,7] = 1.6
conditions[12,8] = 1.7

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions,
                         shocks = shocks,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_shocks)
                         

    
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions,
                         shocks = shocks)

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_shocks_compare)


    
shocks_sp = spzeros(3,3)
shocks_sp[1,1] = 0.1
shocks_sp[2,2] = 0.1
shocks_sp[3,3] = 0.1

conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:R, :Y, :MC], Periods = 1:3)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0
conditions_ka[3,3] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         shocks = shocks_sp,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_shocks_sp)
    

shocks_ka = KeyedArray(Matrix{Float64}(undef,1,3),Variables = [:eps_a], Periods = 1:3)
shocks_ka .= 0.0

conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:R, :Y, :MC], Periods = 1:3)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0
conditions_ka[3,3] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         shocks = shocks_ka,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_shocks_ka)
    

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         shocks = shocks_ka)

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         shocks = shocks_sp,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_shocks_compare_sp_ka)

    
conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:R, :Y, :MC], Periods = 1:3)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0
conditions_ka[3,3] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         algorithm = :second_order,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_second_order)


    
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka)

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         algorithm = :second_order,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_second_order_combine)


plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         algorithm = :pruned_third_order,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_higher_order_combine)
    
    

init_state = get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    levels = true)
    
init_state(:nu,:,:) .= 0.1

conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,13),Variables = [:R, :Y, :MC], Periods = 1:13)
conditions_ka[1,11] = 1.0
conditions_ka[2,12] = 1.0
conditions_ka[3,13] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         initial_state = vec(init_state),
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_init_state)
    

init_state_pruned_3rd_in_diff = get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    levels = true) - get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    algorithm = :pruned_third_order,
    levels = true)
    
init_states_pruned_3rd_vec = [
    zero(vec(init_state_pruned_3rd_in_diff)),
    vec(init_state_pruned_3rd_in_diff),
    zero(vec(init_state_pruned_3rd_in_diff)),
]

init_states_pruned_3rd_vec[1][18] = 0.1

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         initial_state = init_states_pruned_3rd_vec,
                         algorithm = :pruned_third_order,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_init_state_3rd_order)

    

init_state_pruned_3rd = get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    levels = true,
    algorithm = :pruned_third_order)

init_state_pruned_3rd(:nu, :,  :) .= 0.1

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         initial_state = vec(init_state_pruned_3rd),
                         algorithm = :pruned_third_order)
                         
plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         initial_state = vec(init_state),
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_init_state_compare_orders)
    

conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,13),Variables = [:R, :Y, :MC], Periods = 1:13)
conditions_ka[1,11] = 1.0
conditions_ka[2,12] = 1.0
conditions_ka[3,13] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         periods = 10,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_periods)
    

shocks_ka = KeyedArray(Matrix{Float64}(undef,1,20),Variables = [:eps_a], Periods = 1:20)
shocks_ka .= 0.0

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         shocks = shocks_ka,
                         periods = 30,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_30_periods)
    


conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:R, :Y, :MC], Periods = 1:3)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0
conditions_ka[3,3] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         variables = [:Y, :Pi],
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_vars)
    
    
    
    
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

conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:P, :R, :c], Periods = 1:3)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0
conditions_ka[3,3] = 1.0

plot_conditional_forecast(FS2000,
                         conditions_ka,
                         variables = :all_excluding_obc,
    save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_all_excluding_obc)
    
    
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



conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:C, :R, :Y], Periods = 1:3)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0
conditions_ka[3,3] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_obc,
                         conditions_ka,
                         variables = :all,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_all)
                         

conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:R, :Y, :MC], Periods = 1:3)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0
conditions_ka[3,3] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         parameters = :β => 0.95,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_beta_95)
                         

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         parameters = :β => 0.99)

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         parameters = :β => 0.95,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_compare_beta)
                         

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions_ka,
                         parameters = (:β => 0.97, :τ => 0.5),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_multi_params)
                         

conditions_in_dev_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),Variables = [:Y], Periods = 1:1)
conditions_in_dev_ka[1,1] = -0.05

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditions_in_dev_ka,
                         conditions_in_levels = false,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_no_levels)
                         

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                         conditions_in_dev_ka,
                         conditions_in_levels = false,
                         algorithm = :second_order,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_no_levels_second_order)

                         
conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),Variables = [:Y], Periods = 1:1)
conditions_ka[1,1] = 1.0

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
    conditions_ka,
    parameters = (:β => 0.99, :τ => 0.0),
    label = "Std. params")

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
    conditions_ka,
    parameters = (:β => 0.95, :τ => 0.5),
    label = "Alt. params",
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_label)

                         

conditions_ka1 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,10),Variables = [:Y], Periods = 1:10)
conditions_ka1 .= 0.1

# First shock in the scenario
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
    conditions_ka1,
    conditions_in_levels = false)

conditions_ka2 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,10),Variables = [:Y,:R], Periods = 1:10)
conditions_ka2[1,:] .= -0.1
conditions_ka2[2,:] .= 0.0

# Add second shock to show cumulative effect
plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
    conditions_ka2,
    conditions_in_levels = false,
    plot_type = :stack,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_stack)

                         

conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),Variables = [:R], Periods = 1:1)
conditions_ka .= 1.0

# Baseline parameterization
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
    conditions_ka,
    parameters = :β => 0.99)

# Alternative parameterization for comparison
plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
    conditions_ka,
    parameters = :β => 0.95,
    plot_type = :compare,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_compare)

                         

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


conditions_ka1 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,10),Variables = [:Y], Periods = 1:10)
conditions_ka1 .= 0.1

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
    conditions_ka1,
    conditions_in_levels = false)


conditions_ka2 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,10),Variables = [:Y,:R], Periods = 1:10)
conditions_ka2[1,:] .= -0.1
conditions_ka2[2,:] .= 0.0

# Add second shock to show cumulative effect
plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
    conditions_ka2,
    conditions_in_levels = false,
    plot_attributes = Dict(:palette => ec_color_palette),
    plot_type = :stack,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_color)

                         

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
    conditions_ka,
    plot_attributes = Dict(:fontfamily => "computer modern"),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_font)

                         

conditions_ka1 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,10),Variables = [:Y], Periods = 1:10)
conditions_ka1 .= 0.1

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
    conditions_ka1,
    conditions_in_levels = false,
    variables = [:Y, :Pi, :R, :C, :N, :W_real, :MC, :i_ann, :A],
    plots_per_page = 2,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_2_per_page)


conditions_ka1 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,10),Variables = [:Y], Periods = 1:10)
conditions_ka1 .= 0.1

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
    conditions_ka1,
    conditions_in_levels = false,
    rename_dictionary = Dict(:Y => "Output", :Pi => "Inflation", :R => "Interest Rate"),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_rename_dict)



conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:P, :R, :c], Periods = 1:3)
conditions_ka[1,1] = 1.01
conditions_ka[2,2] = 1.02
conditions_ka[3,3] = 1.03

plot_conditional_forecast(FS2000,
                         conditions_ka,
                         rename_dictionary = Dict(:c => "Consumption", :y => "Output", :R => "Interest Rate"))

conditions_ka1 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,10),Variables = [:Y], Periods = 1:10)
conditions_ka1 .= 0.01

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
    conditions_ka1,
    conditions_in_levels = false,
    rename_dictionary = Dict(:C => "Consumption", :Y => "Output", :R => "Interest Rate"),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_rename_dict2)



conditions_ka1 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,10),Variables = [:Y], Periods = 1:10)
conditions_ka1 .= 0.01

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                            conditions_ka1,
                            conditions_in_levels = false,
                            rename_dictionary = Dict(:eps_a => "Technology Shock", :eps_nu => "Monetary Policy Shock"))

conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,3,3),Variables = [:P, :R, :c], Periods = 1:3)
conditions_ka[1,1] = 1.01
conditions_ka[2,2] = 1.02
conditions_ka[3,3] = 1.03

plot_conditional_forecast!(FS2000,
                         conditions_ka,
                         rename_dictionary = Dict(:e_a => "Technology Shock", :e_m => "Monetary Policy Shock"),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_rename_dict_shocks)



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


conditions_ka = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,2),Variables = ["C{H}", "C{F}"], Periods = 1:2)
conditions_ka[1,1] = 1.0
conditions_ka[2,2] = 1.0

plot_conditional_forecast(Backus_Kehoe_Kydland_1992,
    conditions_ka,
    rename_dictionary = Dict("C{H}" => "Home Consumption", 
                             "C{F}" => "Foreign Consumption",
                             "Y{H}" => "Home Output",
                             "Y{F}" => "Foreign Output"),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :cnd_fcst_rename_dict_string)


# Model Estimates
Random.seed!(10)

#### add aux functions (plot_shock_decomposition) ####


using MacroModelling, StatsPlots, CSV, DataFrames, AxisKeys, Dates


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

dat = CSV.read("test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

plot_model_estimates(FS2000, data,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates)


# ![FS2000 model estimates](../assets/estimates__FS2000__3.png)

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)

# Plot with baseline parameters
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                    sim_data,
                    parameters = :β => 0.99)

# Add with different algorithm AND parameters
plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     parameters = :β => 0.95,
                     algorithm = :second_order,
                     save_plots = true, 
                     save_plots_format = :png, 
                     save_plots_path = "./docs/src/assets", 
                     save_plots_name = :estimates_compare_beta_and_orders)


## Data (Required)

dat = CSV.read("test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)', Variable = Symbol.("log_".*names(dat)), Time = 1:size(dat)[1])
data = log.(data)


function quarterly_dates(start_date::Date, len::Int)
    dates = Vector{Date}(undef, len)
    current_date = start_date
    for i in 1:len
        dates[i] = current_date
        current_date = current_date + Dates.Month(3)
    end
    return dates
end

data_rekey = rekey(data, :Time => quarterly_dates(Date(1960, 1, 1), size(data,2)))

plot_model_estimates(FS2000, data_rekey,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_rekey)

sim_data = simulate(FS2000)([:log_gy_obs,:log_gp_obs],:,:simulate)

plot_model_estimates!(FS2000, sim_data,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_multiple_data)


## Data in levels
sim = simulate(FS2000, levels = false)
plot_model_estimates(FS2000, sim([:y,:R],:,:simulate), data_in_levels = false,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_levels_false)

## Filter
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

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data, filter = :kalman)

plot_model_estimates!(Gali_2015_chapter_3_nonlinear, sim_data, filter = :inversion,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_filters)

## Smooth

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data, smooth = true)

plot_model_estimates!(Gali_2015_chapter_3_nonlinear, sim_data, smooth = false,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_smooth)

plot_model_estimates!(Gali_2015_chapter_3_nonlinear, sim_data, filter = :inversion, smooth = false,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_smooth_inversion)


## Presample periods

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data, presample_periods = 20,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_presample)


## Forecast periods

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_forecast_default)

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data, forecast_periods = 24,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_forecast_24)

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data, forecast_periods = 0,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_forecast_0)

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data, parameters = :β => 0.99)
plot_model_estimates!(Gali_2015_chapter_3_nonlinear, sim_data, parameters = :β => 0.95, forecast_periods = 18,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_forecast_compare)


## Shock decomposition

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data, shock_decomposition = true,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_shock_decomp_true)

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear, sim_data, shock_decomposition = false,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_shock_decomp)

## Shocks

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     shocks = [:eps_a, :eps_nu],
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_selected_shocks)

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     shocks = :none,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_no_shocks)


## Solution Algorithm

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     algorithm = :second_order,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_second_order)


# ![Gali 2015 conditional forecast - second order](../assets/cnd_fcst_second_order__Gali_2015_chapter_3_nonlinear__1.png)

plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data)

plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     algorithm = :second_order,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_first_and_second_order)


# ![Gali 2015 conditional forecast - first and second order](../assets/cnd_fcst_second_order_combine__Gali_2015_chapter_3_nonlinear__2.png)

plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     algorithm = :pruned_third_order,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_multiple_orders)


# ![Gali 2015 conditional forecast - multiple orders](../assets/cnd_fcst_higher_order_combine__Gali_2015_chapter_3_nonlinear__1.png)

## Variables to Plot

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     variables = [:Y, :Pi],
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_vars)


# ![Gali 2015 conditional forecast - selected variables (Y, Pi)](../assets/cnd_fcst_vars__Gali_2015_chapter_3_nonlinear__1.png)

plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     variables = :all_excluding_auxiliary_and_obc)
                     
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     variables = :all_excluding_obc)
                     
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

sim_data_FS2000 = simulate(FS2000)([:y],:,:simulate)
plot_model_estimates(FS2000,
                     sim_data_FS2000,
                     variables = :all_excluding_obc,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_all_excluding_obc)


# ![FS2000 conditional forecast - e_a shock with auxiliary variables](../assets/cnd_fcst_all_excluding_obc__FS2000__1.png)

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

sim_data_Gali_obc = simulate(Gali_2015_chapter_3_obc)([:R],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_obc,
                     sim_data_Gali_obc,
                     variables = :all,
                     save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_all)


# ![Gali 2015 OBC conditional forecast - with OBC variables](../assets/cnd_fcst_all__Gali_2015_chapter_3_obc__3.png)

## Parameter Values

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     parameters = :β => 0.95,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_beta_95)


# ![Gali 2015 conditional forecast - `β = 0.95`](../assets/cnd_fcst_beta_95__Gali_2015_chapter_3_nonlinear__1.png)

plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     parameters = :β => 0.99)

plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     parameters = :β => 0.95,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_beta_95_vs_99)


# ![Gali 2015 conditional forecast - comparing β values](../assets/cnd_fcst_compare_beta__Gali_2015_chapter_3_nonlinear__2.png)

plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     parameters = (:β => 0.97, :τ => 0.5),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_multi_params)


# ![Gali 2015 conditional forecast - multiple parameter changes](../assets/cnd_fcst_multi_params__Gali_2015_chapter_3_nonlinear__2.png)

plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     parameters = [:β => 0.98, :τ => 0.25],
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_multi_params_2)

# params = get_parameters(Gali_2015_chapter_3_nonlinear, values = true)
# 16-element Vector{Pair{String, Float64}}:
#       "σ" => 1.0
#       "φ" => 5.0
#     "ϕᵖⁱ" => 1.5
#      "ϕʸ" => 0.125
#       "θ" => 0.75
#     "ρ_ν" => 0.5
#     "ρ_z" => 0.5
#     "ρ_a" => 0.9
#       "β" => 0.95
#       "η" => 3.77
#       "α" => 0.25
#       "ϵ" => 9.0
#       "τ" => 0.5
#   "std_a" => 0.01
#   "std_z" => 0.05
#  "std_nu" => 0.0025

# param_vals = [p[2] for p in params]
# 16-element Vector{Float64}:
#  1.0
#  5.0
#  1.5
#  0.125
#  0.75
#  0.5
#  0.5
#  0.9
#  0.95
#  3.77
#  0.25
#  9.0
#  0.5
#  0.01
#  0.05
#  0.0025

# plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
#                      sim_data,
#                      parameters = param_vals,
#                          save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_multi_params_3)


## Plot Labels

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     parameters = (:β => 0.99, :τ => 0.0),
                     label = "Std. params")

plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
                      sim_data,
                      parameters = (:β => 0.95, :τ => 0.5),
                      label = "Alt. params",
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_labels)


# ![Gali 2015 conditional forecast - custom labels](../assets/cnd_fcst_label__Gali_2015_chapter_3_nonlinear__2.png)

# plot_model_estimates(Gali_2015_chapter_3_nonlinear,
#                      sim_data,
#                      parameters = (:β => 0.99, :τ => 0.0),
#                      label = :standard)

# plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
#     sim_data,
#     parameters = (:β => 0.95, :τ => 0.5),
#     label = :alternative,
#                          save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_labels_symbol)

# plot_model_estimates(Gali_2015_chapter_3_nonlinear,
#     sim_data,
#     parameters = (:β => 0.99, :τ => 0.0),
#     label = 0.99)

# plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
#     sim_data,
#     parameters = (:β => 0.95, :τ => 0.5),
#     label = 0.95,
#                          save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_labels_value)


## Plot Attributes

ec_color_palette =
[
    "#FFD724",  # "Sunflower Yellow"
    "#353B73",  # "Navy Blue"
    "#2F9AFB",  # "Sky Blue"
    "#B8AAA2",  # "Taupe Grey"
    "#E75118",  # "Vermilion"
    "#6DC7A9",  # "Mint Green"
    "#F09874",  # "Coral"
    "#907800"   # "Olive"
]
    
sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     plot_attributes = Dict(:palette => ec_color_palette),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_color)


# ![Gali 2015 conditional forecast - custom color palette](../assets/cnd_fcst_color__Gali_2015_chapter_3_nonlinear__2.png)

plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     plot_attributes = Dict(:fontfamily => "computer modern"),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_font)


# ![Gali 2015 conditional forecast - custom font](../assets/cnd_fcst_font__Gali_2015_chapter_3_nonlinear__1.png)


## Plots Per Page

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                     sim_data,
                     variables = [:Y, :Pi, :R, :C, :N, :W_real, :MC, :i_ann, :A],
                     plots_per_page = 2,
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_2_per_page)


# ![Gali 2015 conditional forecast - 2 plots per page](../assets/cnd_fcst_2_per_page__Gali_2015_chapter_3_nonlinear__3.png)

## Variable and Shock Renaming (rename dictionary)

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates(Gali_2015_chapter_3_nonlinear,
    sim_data,
    rename_dictionary = Dict(:Y => "Output", :Pi => "Inflation", :R => "Interest Rate"),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_rename_dict)

# ![Gali 2015 conditional forecast - rename dictionary](../assets/cnd_fcst_rename_dict__Gali_2015_chapter_3_nonlinear__1.png)

sim_data_FS2000 = simulate(FS2000)([:y],:,:simulate)
plot_model_estimates(FS2000,
                         sim_data_FS2000,
                         rename_dictionary = Dict(
                            :c => "Consumption", 
                            :y => "Output", 
                            :R => "Interest Rate"
                         ))

sim_data = simulate(Gali_2015_chapter_3_nonlinear)([:Y],:,:simulate)
plot_model_estimates!(Gali_2015_chapter_3_nonlinear,
    sim_data,
    rename_dictionary = Dict(
        :C => "Consumption", 
        :Y => "Output", 
        :R => "Interest Rate"
        ),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_rename_dict_multiple_models)

# ![FS2000 and Gali 2015 conditional forecast - multiple models with rename dictionary](../assets/cnd_fcst_rename_dict2__multiple_models__2.png)

plot_model_estimates(Gali_2015_chapter_3_nonlinear,
                            sim_data,
                            rename_dictionary = Dict(
                                :eps_a => "Technology Shock", 
                                :eps_nu => "Monetary Policy Shock"
                                ))

plot_model_estimates!(FS2000,
                         sim_data_FS2000,
                         rename_dictionary = Dict(
                            :e_a => "Technology Shock", 
                            :e_m => "Monetary Policy Shock"
                            ),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_rename_dict_multiple_models_shocks)


# ![FS2000 and Gali 2015 conditional forecast - multiple models with shock rename dictionary](../assets/cnd_fcst_rename_dict_shocks__multiple_models__7.png)

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



sim_data = simulate(Backus_Kehoe_Kydland_1992)(["Y{H}"],:,:simulate)
plot_model_estimates(Backus_Kehoe_Kydland_1992,
    sim_data,
    rename_dictionary = Dict("C{H}" => "Home Consumption", 
                             "C{F}" => "Foreign Consumption",
                             "Y{H}" => "Home Output",
                             "Y{F}" => "Foreign Output"),
                         save_plots = true, save_plots_format = :png, save_plots_path = "./docs/src/assets", save_plots_name = :estimates_rename_dict_string)

# ![Backus, Kehoe, Kydland 1992 conditional forecast - E{H} shock with rename dictionary](../assets/cnd_fcst_rename_dict_string__Backus_Kehoe_Kydland_1992__1.png)



# Estimation

using Random
Random.seed!(3)

using MacroModelling

using StatsPlots
using CSV, DataFrames, AxisKeys
import DynamicPPL
import Turing
import Turing: NUTS, sample, logpdf
import ADTypes: AutoZygote
# import Zygote
import MCMCChains: Chains

using HDF5

using MCMCChainsStorage

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

# load data
dat = CSV.read("docs/src/assets/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)


prior_distributions = [
    Beta(0.356, 0.02, μσ = true),           # alp
    Beta(0.993, 0.002, μσ = true),          # bet
    Normal(0.0085, 0.003),                  # gam
    Normal(1.0002, 0.007),                  # mst
    Beta(0.129, 0.223, μσ = true),          # rho
    Beta(0.65, 0.05, μσ = true),            # psi
    Beta(0.01, 0.005, μσ = true),           # del
    InverseGamma(0.035449, Inf, μσ = true), # z_e_a
    InverseGamma(0.008862, Inf, μσ = true)  # z_e_m
]

Turing.@model function FS2000_loglikelihood_function(prior_distributions, data, m; verbose = false)
    parameters ~ Turing.arraydist(prior_distributions)

    # if DynamicPPL.leafcontext(DynamicPPL.__context__) !== DynamicPPL.PriorContext() 
        Turing.@addlogprob! get_loglikelihood(m, 
                                data, 
                                parameters)
    # end
end


FS2000_loglikelihood = FS2000_loglikelihood_function(prior_distributions, data, FS2000)

# n_samples = 100

# chain_NUTS = sample(FS2000_loglikelihood, NUTS(), n_samples, progress = false, initial_params = FS2000.parameter_values)

# h5open("docs/src/assets/chain_NUTS.h5", "w") do f
#   write(f, chain_NUTS)
# end

chain_NUTS = h5open("docs/src/assets/chain_NUTS.h5", "r") do f read(f, Chains) end


chain_NUTS_rn = replacenames(chain_NUTS, Dict(["parameters[$i]" for i in 1:length(FS2000.parameters)] .=> FS2000.parameters))

chain_NUTS = replacenames(chain_NUTS, Dict(FS2000.parameters .=> ["parameters[$i]" for i in 1:length(FS2000.parameters)]))

# ensure output directory exists and save the chain plot as PNG
p = plot(chain_NUTS_rn)
savefig(p, joinpath("./docs/src/assets", "FS2000_chain_NUTS.png"))

# ![NUTS chain](../assets/FS2000_chain_NUTS.png)


using ComponentArrays, MCMCChains
import DynamicPPL: logjoint

parameter_mean = mean(chain_NUTS)

pars = ComponentArray([parameter_mean.nt[2]], Axis(:parameters));

logjoint(FS2000_loglikelihood, pars)

function calculate_log_probability(par1, par2, pars_syms, orig_pars, model)
    orig_pars[1][pars_syms] = [par1, par2]
    logjoint(model, orig_pars)
end

granularity = 32;

par1 = :del;
par2 = :gam;

paridx1 = indexin([par1], FS2000.parameters)[1];
paridx2 = indexin([par2], FS2000.parameters)[1];

par_range1 = collect(range(minimum(chain_NUTS[Symbol("parameters[$paridx1]")]), stop = maximum(chain_NUTS[Symbol("parameters[$paridx1]")]), length = granularity));
par_range2 = collect(range(minimum(chain_NUTS[Symbol("parameters[$paridx2]")]), stop = maximum(chain_NUTS[Symbol("parameters[$paridx2]")]), length = granularity));

p = surface(par_range1, par_range2, 
            (x,y) -> calculate_log_probability(x, y, [paridx1, paridx2], pars, FS2000_loglikelihood),
            camera=(30, 65),
            colorbar=false,
            color=:inferno);

joint_loglikelihood = [logjoint(FS2000_loglikelihood, ComponentArray([reduce(hcat, get(chain_NUTS, :parameters)[1])[s,:]], Axis(:parameters))) for s in 1:length(chain_NUTS)];

scatter3d!(vec(collect(chain_NUTS[Symbol("parameters[$paridx1]")])),
            vec(collect(chain_NUTS[Symbol("parameters[$paridx2]")])),
            joint_loglikelihood,
            mc = :viridis, 
            marker_z = collect(1:length(chain_NUTS)), 
            msw = 0,
            legend = false, 
            colorbar = false, 
            xlabel = string(par1),
            ylabel = string(par2),
            zlabel = "Log probability",
            alpha = 0.5);

p
savefig(p, joinpath("./docs/src/assets", "FS2000_posterior_surface.png"))


# ![Posterior surface](../assets/FS2000_posterior_surface.png)


modeFS2000 = Turing.maximum_a_posteriori(FS2000_loglikelihood, 
                                        # adtype = AutoZygote(), 
                                        initial_params = FS2000.parameter_values)

get_estimated_shocks(FS2000, data, parameters = collect(modeFS2000.values))

plot_model_estimates(FS2000, data,
                        save_plots = true, 
                        save_plots_format = :png, 
                        save_plots_path = "./docs/src/assets", 
                        save_plots_name = :estimation_tutorial)


## RBC tutorial

using SparseArrays
using AxisKeys
import StatsPlots
using MacroModelling

Random.seed!(10)

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
end

@parameters RBC begin
    σᶻ= 0.01
    ρᶻ= 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

## Plot impulse response functions (IRFs)

plot_irf(RBC,
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_irf)

# ![RBC IRF](../assets/irf__RBC__eps_z__1.png)


## Explore other parameter values

plot_irf(RBC, parameters = :α => 0.3,
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_irf_alpha_0_3)

# ![IRF plot](../assets/irf__RBC_new__eps_z__1.png)

## Plot model simulation

plot_simulations(RBC,
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_sim)

# ![Simulate RBC](../assets/irf__RBC_sim__eps_z__1.png)

## Plot specific series of shocks

shock_series = zeros(1,4)
shock_series[1,2] = 1
shock_series[1,4] = -1
plot_irf(RBC, shocks = shock_series,
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_shock_matrix)

# ![Series of shocks RBC](../assets/irf__RBC__shock_matrix__1.png)

get_steady_state(RBC,parameters = :β => .951)

get_standard_deviation(RBC, parameters = (:α => 0.5, :β => .95))

## Model solution
plot_solution(RBC, :k,
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_solution)

# ![RBC solution](../assets/solution__RBC__1.png)

## Conditional forecasts


conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,4),Variables = [:c], Periods = 1:4)
conditions[1:4] .= [-.01,0,.01,.02];

shocks = spzeros(1,5)
shocks[1,5] = -1;


plot_conditional_forecast(RBC, conditions, shocks = shocks, conditions_in_levels = false,
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_cond_fcst)


# ![RBC conditional forecast](../assets/conditional_fcst__RBC__conditional_forecast__1.png)


## SW03 tutorial
# Work with a complex model - Smets and Wouters (2003)
using Random
using MacroModelling
import StatsPlots
using AxisKeys

Random.seed!(42)

@model Smets_Wouters_2003 begin
    -q[0] + beta * ((1 - tau) * q[1] + epsilon_b[1] * (r_k[1] * z[1] - psi^-1 * r_k[ss] * (-1 + exp(psi * (-1 + z[1])))) * (C[1] - h * C[0])^(-sigma_c))
    -q_f[0] + beta * ((1 - tau) * q_f[1] + epsilon_b[1] * (r_k_f[1] * z_f[1] - psi^-1 * r_k_f[ss] * (-1 + exp(psi * (-1 + z_f[1])))) * (C_f[1] - h * C_f[0])^(-sigma_c))
    -r_k[0] + alpha * epsilon_a[0] * mc[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^(-1 + alpha)
    -r_k_f[0] + alpha * epsilon_a[0] * mc_f[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^(-1 + alpha)
    -G[0] + T[0]
    -G[0] + G_bar * epsilon_G[0]
    -G_f[0] + T_f[0]
    -G_f[0] + G_bar * epsilon_G[0]
    -L[0] + nu_w[0]^-1 * L_s[0]
    -L_s_f[0] + L_f[0] * (W_i_f[0] * W_f[0]^-1)^(lambda_w^-1 * (-1 - lambda_w))
    L_s_f[0] - L_f[0]
    L_s_f[0] + lambda_w^-1 * L_f[0] * W_f[0]^-1 * (-1 - lambda_w) * (-W_disutil_f[0] + W_i_f[0]) * (W_i_f[0] * W_f[0]^-1)^(-1 + lambda_w^-1 * (-1 - lambda_w))
    Pi_ws_f[0] - L_s_f[0] * (-W_disutil_f[0] + W_i_f[0])
    Pi_ps_f[0] - Y_f[0] * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p))
    -Q[0] + epsilon_b[0]^-1 * q[0] * (C[0] - h * C[-1])^(sigma_c)
    -Q_f[0] + epsilon_b[0]^-1 * q_f[0] * (C_f[0] - h * C_f[-1])^(sigma_c)
    -W[0] + epsilon_a[0] * mc[0] * (1 - alpha) * L[0]^(-alpha) * (K[-1] * z[0])^alpha
    -W_f[0] + epsilon_a[0] * mc_f[0] * (1 - alpha) * L_f[0]^(-alpha) * (K_f[-1] * z_f[0])^alpha
    -Y_f[0] + Y_s_f[0]
    Y_s[0] - nu_p[0] * Y[0]
    -Y_s_f[0] + Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p))
    beta * epsilon_b[1] * (C_f[1] - h * C_f[0])^(-sigma_c) - epsilon_b[0] * R_f[0]^-1 * (C_f[0] - h * C_f[-1])^(-sigma_c)
    beta * epsilon_b[1] * pi[1]^-1 * (C[1] - h * C[0])^(-sigma_c) - epsilon_b[0] * R[0]^-1 * (C[0] - h * C[-1])^(-sigma_c)
    Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) - lambda_p^-1 * Y_f[0] * (1 + lambda_p) * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-1 - lambda_p^-1 * (1 + lambda_p))
    epsilon_b[0] * W_disutil_f[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) - omega * epsilon_b[0] * epsilon_L[0] * L_s_f[0]^sigma_l
    -1 + xi_p * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1) + (1 - xi_p) * pi_star[0]^(-lambda_p^-1)
    -1 + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1) + xi_w * (W[-1] * W[0]^-1)^(-lambda_w^-1) * (pi[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1)
    -Phi - Y_s[0] + epsilon_a[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^alpha
    -Phi - Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_a[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^alpha
    std_eta_b * eta_b[x] - log(epsilon_b[0]) + rho_b * log(epsilon_b[-1])
    -std_eta_L * eta_L[x] - log(epsilon_L[0]) + rho_L * log(epsilon_L[-1])
    std_eta_I * eta_I[x] - log(epsilon_I[0]) + rho_I * log(epsilon_I[-1])
    std_eta_w * eta_w[x] - f_1[0] + f_2[0]
    std_eta_a * eta_a[x] - log(epsilon_a[0]) + rho_a * log(epsilon_a[-1])
    std_eta_p * eta_p[x] - g_1[0] + g_2[0] * (1 + lambda_p)
    std_eta_G * eta_G[x] - log(epsilon_G[0]) + rho_G * log(epsilon_G[-1])
    -f_1[0] + beta * xi_w * f_1[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1) + epsilon_b[0] * w_star[0] * L[0] * (1 + lambda_w)^-1 * (C[0] - h * C[-1])^(-sigma_c) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w))
    -f_2[0] + beta * xi_w * f_2[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) + omega * epsilon_b[0] * epsilon_L[0] * (L[0] * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)))^(1 + sigma_l)
    -g_1[0] + beta * xi_p * pi_star[0] * g_1[1] * pi_star[1]^-1 * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1) + epsilon_b[0] * pi_star[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c)
    -g_2[0] + beta * xi_p * g_2[1] * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_b[0] * mc[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c)
    -nu_w[0] + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)) + xi_w * nu_w[-1] * (W[-1] * pi[0]^-1 * W[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w))
    -nu_p[0] + (1 - xi_p) * pi_star[0]^(-lambda_p^-1 * (1 + lambda_p)) + xi_p * nu_p[-1] * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p))
    -K[0] + K[-1] * (1 - tau) + I[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2)
    -K_f[0] + K_f[-1] * (1 - tau) + I_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2)
    U[0] - beta * U[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C[0] - h * C[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s[0]^(1 + sigma_l))
    U_f[0] - beta * U_f[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C_f[0] - h * C_f[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s_f[0]^(1 + sigma_l))
    -epsilon_b[0] * (C[0] - h * C[-1])^(-sigma_c) + q[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2 - varphi * I[-1]^-1 * epsilon_I[0] * I[0] * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])) + beta * varphi * I[0]^-2 * epsilon_I[1] * q[1] * I[1]^2 * (-1 + I[0]^-1 * epsilon_I[1] * I[1])
    -epsilon_b[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) + q_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2 - varphi * I_f[-1]^-1 * epsilon_I[0] * I_f[0] * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])) + beta * varphi * I_f[0]^-2 * epsilon_I[1] * q_f[1] * I_f[1]^2 * (-1 + I_f[0]^-1 * epsilon_I[1] * I_f[1])
    std_eta_pi * eta_pi[x] - log(pi_obj[0]) + rho_pi_bar * log(pi_obj[-1]) + log(calibr_pi_obj) * (1 - rho_pi_bar)
    -C[0] - I[0] - T[0] + Y[0] - psi^-1 * r_k[ss] * K[-1] * (-1 + exp(psi * (-1 + z[0])))
    -calibr_pi + std_eta_R * eta_R[x] - log(R[ss]^-1 * R[0]) + r_Delta_pi * (-log(pi[ss]^-1 * pi[-1]) + log(pi[ss]^-1 * pi[0])) + r_Delta_y * (-log(Y[ss]^-1 * Y[-1]) + log(Y[ss]^-1 * Y[0]) + log(Y_f[ss]^-1 * Y_f[-1]) - log(Y_f[ss]^-1 * Y_f[0])) + rho * log(R[ss]^-1 * R[-1]) + (1 - rho) * (log(pi_obj[0]) + r_pi * (-log(pi_obj[0]) + log(pi[ss]^-1 * pi[-1])) + r_Y * (log(Y[ss]^-1 * Y[0]) - log(Y_f[ss]^-1 * Y_f[0])))
    -C_f[0] - I_f[0] + Pi_ws_f[0] - T_f[0] + Y_f[0] + L_s_f[0] * W_disutil_f[0] - L_f[0] * W_f[0] - psi^-1 * r_k_f[ss] * K_f[-1] * (-1 + exp(psi * (-1 + z_f[0])))
    epsilon_b[0] * (K[-1] * r_k[0] - r_k[ss] * K[-1] * exp(psi * (-1 + z[0]))) * (C[0] - h * C[-1])^(-sigma_c)
    epsilon_b[0] * (K_f[-1] * r_k_f[0] - r_k_f[ss] * K_f[-1] * exp(psi * (-1 + z_f[0]))) * (C_f[0] - h * C_f[-1])^(-sigma_c)
end

@parameters Smets_Wouters_2003 begin  
    lambda_p = .368
    G_bar = .362
    lambda_w = 0.5
    Phi = .819

    alpha = 0.3
    beta = 0.99
    gamma_w = 0.763
    gamma_p = 0.469
    h = 0.573
    omega = 1
    psi = 0.169

    r_pi = 1.684
    r_Y = 0.099
    r_Delta_pi = 0.14
    r_Delta_y = 0.159

    sigma_c = 1.353
    sigma_l = 2.4
    tau = 0.025
    varphi = 6.771
    xi_w = 0.737
    xi_p = 0.908

    rho = 0.961
    rho_b = 0.855
    rho_L = 0.889
    rho_I = 0.927
    rho_a = 0.823
    rho_G = 0.949
    rho_pi_bar = 0.924

    std_eta_b = 0.336
    std_eta_L = 3.52
    std_eta_I = 0.085
    std_eta_a = 0.598
    std_eta_w = 0.6853261
    std_eta_p = 0.7896512
    std_eta_G = 0.325
    std_eta_R = 0.081
    std_eta_pi = 0.017

    calibr_pi_obj | 1 = pi_obj[ss]
    calibr_pi | pi[ss] = pi_obj[ss]
end

## Plot impulse response functions (IRFs)

plot_irf(Smets_Wouters_2003,
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_irf)

# ![RBC IRF](../assets/irf__SW03__eta_R__1.png)

## Explore other parameter values

plot_irf(Smets_Wouters_2003, 
         parameters = :alpha => 0.305, 
         variables = [:U,:Y,:I,:R,:C], 
         shocks = :eta_R, 
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_irf_alpha_0_305)

# ![IRF plot](../assets/irf__SW03_new__eta_R__1.png)

## Plot model simulation

plot_simulations(Smets_Wouters_2003, variables = [:U,:Y,:I,:R,:C],
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_sim)

# ![Simulate Smets_Wouters_2003](../assets/irf__SW03__simulation__1.png)

## Plot specific series of shocks

shock_series = KeyedArray(zeros(2,12), Shocks = [:eta_b, :eta_w], Periods = 1:12)
shock_series[1,2] = 1
shock_series[2,12] = -1
plot_irf(Smets_Wouters_2003, shocks = shock_series, variables = [:W,:r_k,:w_star,:R],
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_shock_matrix)


# ![Series of shocks RBC](../assets/irf__SW03__shock_matrix__1.png)

get_steady_state(Smets_Wouters_2003, 
                 parameter_derivatives = [:alpha,:G_bar], 
                 parameters = :beta => .991)

get_standard_deviation(Smets_Wouters_2003, 
                       parameter_derivatives = [:alpha,:beta], 
                       parameters = (:alpha => 0.3, :beta => .99))
### Plot conditional variance decomposition

plot_conditional_variance_decomposition(Smets_Wouters_2003, variables = [:U,:Y,:I,:R,:C],
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_fevd)

# ![FEVD Smets_Wouters_2003](../assets/fevd__SW03__1.png)

## Model solution

plot_solution(Smets_Wouters_2003, :pi, variables = [:C,:I,:K,:L,:W,:R],
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_solution)

# ![Smets_Wouters_2003 solution](../assets/solution__SW03__1.png)

## Conditional forecasts

conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,4),Variables = [:Y, :pi], Periods = 1:4)
conditions[1,1:4] .= [-.01,0,.01,.02];
conditions[2,1:4] .= [.01,0,-.01,-.02];

shocks = Matrix{Union{Nothing,Float64}}(undef,9,5)
shocks[[1:3...,5,9],1:2] .= 0;
shocks[9,5] = -1;

plot_conditional_forecast(Smets_Wouters_2003,conditions, shocks = shocks, plots_per_page = 6,variables = [:Y,:pi,:W],conditions_in_levels = false,
        save_plots = true, 
        save_plots_format = :png, 
        save_plots_path = "./docs/src/assets", 
        save_plots_name = :tutorial_cond_fcst)


# ![Smets_Wouters_2003 conditional forecast 1](../assets/conditional_fcst__SW03__conditional_forecast__1.png)

# ![Smets_Wouters_2003 conditional forecast 2](../assets/conditional_fcst__SW03__conditional_forecast__2.png)
