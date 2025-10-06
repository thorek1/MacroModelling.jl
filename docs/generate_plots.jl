# Script to generate plots for the plotting documentation
# Run this script from the docs directory to generate all plots referenced in plotting.md

using MacroModelling
import StatsPlots

# Create assets directory if it doesn't exist
assets_dir = joinpath(@__DIR__, "assets")
if !isdir(assets_dir)
    mkdir(assets_dir)
end

# Define the model
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

println("Model defined successfully")

# Generate plots with save_plots_name to create unique names
println("Generating plot 1: Basic IRF for eps_a")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1",
         show_plots = false)

println("Generating plot 2: Second order solution for eps_a")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a, 
         algorithm = :second_order,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__2_second_order",
         show_plots = false)

println("Generating plot 3: First order again for comparison")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__2",
         show_plots = false)

println("Generating plot 4: First and second order comparison")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a,
         show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          shocks = :eps_a, 
          algorithm = :second_order,
          save_plots = true, 
          save_plots_format = :png,
          save_plots_path = assets_dir,
          save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_first_and_second_order",
          show_plots = false)

println("Generating plot 5: Multiple orders including pruned third")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a,
         show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          shocks = :eps_a, 
          algorithm = :second_order,
          show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          shocks = :eps_a, 
          algorithm = :pruned_third_order,
          save_plots = true, 
          save_plots_format = :png,
          save_plots_path = assets_dir,
          save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_multiple_orders",
          show_plots = false)

println("Generating plot 6: IRF with initial state")
init_state = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true)
init_state(:nu,:,:) .= 0.1
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a, 
         initial_state = vec(init_state),
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_init_state",
         show_plots = false)

println("Generating plot 7: IRF with no shock but initial state")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :none, 
         initial_state = vec(init_state),
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__no_shock__1_init_state",
         show_plots = false)

println("Generating plot 8: Stacked IRF")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :none, 
         initial_state = vec(init_state),
         show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          shocks = :eps_a, 
          plot_type = :stack,
          save_plots = true, 
          save_plots_format = :png,
          save_plots_path = assets_dir,
          save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1",
          show_plots = false)

println("Generating plot 9: Single shock eps_a")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__2",
         show_plots = false)

println("Generating plot 10: Multiple shocks eps_a and eps_z")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = [:eps_a, :eps_z],
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__3",
         show_plots = false)

# Note: Multiple shock plots will create separate files for each shock
# So we need to handle eps_z separately
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_z,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_z__3",
         show_plots = false)

println("Generating plot 11: Simulated shocks")
import Random
Random.seed!(10)
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :simulate,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__simulation__1",
         show_plots = false)

println("Generating plot 12: Comparing all shocks")
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = shocks[1],
         show_plots = false)
for s in shocks[2:end]
    plot_irf!(Gali_2015_chapter_3_nonlinear, 
              shocks = s,
              show_plots = false)
end
# Save the comparison plot
StatsPlots.savefig(joinpath(assets_dir, "irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1_linear.png"))

println("Generating plot 13: Shock matrix")
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
n_periods = 3
shock_keyedarray = KeyedArray(zeros(length(shocks), n_periods), Shocks = shocks, Periods = 1:n_periods)
shock_keyedarray("eps_a",[1]) .= 1
shock_keyedarray("eps_z",[2]) .= -1/2
shock_keyedarray("eps_nu",[3]) .= 1/3
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = shock_keyedarray,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__shock_matrix__2",
         show_plots = false)

println("Generating plot 14: 10 periods")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         periods = 10, 
         shocks = :eps_a,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_10_periods",
         show_plots = false)

println("Generating plot 15: Shock size -2")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a, 
         shock_size = -2,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_shock_size",
         show_plots = false)

println("Generating plot 16: Negative shock")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         negative_shock = true,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_z__1_neg_shock",
         show_plots = false)

println("Generating plot 17: Variable selection")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         variables = [:Y, :Pi],
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_var_select",
         show_plots = false)

println("Generating plot 18: Beta = 0.95")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         parameters = :β => 0.95, 
         shocks = :eps_a,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_beta_0_95",
         show_plots = false)

println("Generating plot 19: Compare beta values")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         parameters = :β => 0.99, 
         shocks = :eps_a,
         show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          parameters = :β => 0.95, 
          shocks = :eps_a,
          save_plots = true, 
          save_plots_format = :png,
          save_plots_path = assets_dir,
          save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__2_compare_beta",
          show_plots = false)

println("Generating plot 20: Beta and tau comparison")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         parameters = :β => 0.99, 
         shocks = :eps_a,
         show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          parameters = :β => 0.95, 
          shocks = :eps_a,
          show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          parameters = (:β => 0.97, :τ => 0.5), 
          shocks = :eps_a,
          save_plots = true, 
          save_plots_format = :png,
          save_plots_path = assets_dir,
          save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__2_beta_tau",
          show_plots = false)

println("Generating plot 21: Custom labels")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         parameters = (:β => 0.99, :τ => 0.0), 
         shocks = :eps_a, 
         label = "Std. params",
         show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          parameters = (:β => 0.95, :τ => 0.5), 
          shocks = :eps_a, 
          label = "Alt. params",
          save_plots = true, 
          save_plots_format = :png,
          save_plots_path = assets_dir,
          save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__2_custom_labels",
          show_plots = false)

println("Generating plot 22: Custom color palette")
ec_color_palette = ["#FFD724", "#353B73", "#2F9AFB", "#B8AAA2", "#E75118", "#6DC7A9", "#F09874", "#907800"]
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = shocks[1],
         show_plots = false)
for s in shocks[2:end]
    plot_irf!(Gali_2015_chapter_3_nonlinear, 
              shocks = s, 
              plot_attributes = Dict(:palette => ec_color_palette), 
              plot_type = :stack,
              show_plots = false)
end
StatsPlots.savefig(joinpath(assets_dir, "irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__2_ec_colors.png"))

println("Generating plot 23: Custom font")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         shocks = :eps_a, 
         plot_attributes = Dict(:fontfamily => "computer modern"),
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_cm_font",
         show_plots = false)

println("Generating plot 24: Plots per page")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         variables = [:Y, :Pi, :R, :C, :N, :W_real, :MC, :i_ann, :A], 
         shocks = :eps_a, 
         plots_per_page = 2,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_9_vars_2_per_page",
         show_plots = false)

# Define OBC model for remaining plots
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

println("Generating plot 25: OBC model with ignore_obc")
plot_irf(Gali_2015_chapter_3_obc, 
         shocks = :eps_z, 
         variables = [:Y,:R,:Pi,:C], 
         shock_size = 3,
         show_plots = false)
plot_irf!(Gali_2015_chapter_3_obc, 
          shocks = :eps_z, 
          variables = [:Y,:R,:Pi,:C], 
          shock_size = 3, 
          ignore_obc = true,
          save_plots = true, 
          save_plots_format = :png,
          save_plots_path = assets_dir,
          save_plots_name = "irf__Gali_2015_chapter_3_obc__eps_z__1_ignore_obc",
          show_plots = false)

println("Generating plot 26: GIRF for OBC model")
plot_irf(Gali_2015_chapter_3_obc, 
         generalised_irf = true, 
         shocks = :eps_z, 
         variables = [:Y,:R,:Pi,:C], 
         shock_size = 3,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_obc__eps_z__1_girf",
         show_plots = false)

println("Generating plot 27: GIRF with different draw counts")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         generalised_irf = true, 
         shocks = :eps_a,  
         algorithm = :pruned_second_order,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__1_girf",
         show_plots = false)

println("Generating plot 28: GIRF with 1000 draws")
plot_irf(Gali_2015_chapter_3_nonlinear, 
         generalised_irf = true, 
         shocks = :eps_a,  
         algorithm = :pruned_second_order,
         show_plots = false)
plot_irf!(Gali_2015_chapter_3_nonlinear, 
          generalised_irf = true, 
          generalised_irf_draws = 1000, 
          shocks = :eps_a,  
          algorithm = :pruned_second_order,
          save_plots = true, 
          save_plots_format = :png,
          save_plots_path = assets_dir,
          save_plots_name = "irf__Gali_2015_chapter_3_nonlinear__eps_a__2_girf_1000_draws",
          show_plots = false)

println("Generating additional reference plots for OBC model")
plot_irf(Gali_2015_chapter_3_obc, 
         variables = :all,
         shocks = :eps_z,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_obc__eps_z__3",
         show_plots = false)

plot_irf(Gali_2015_chapter_3_obc, 
         shocks = :eps_z,
         save_plots = true, 
         save_plots_format = :png,
         save_plots_path = assets_dir,
         save_plots_name = "irf__Gali_2015_chapter_3_obc__eps_z__2",
         show_plots = false)

println("All plots generated successfully!")
println("Plots saved to: ", assets_dir)
