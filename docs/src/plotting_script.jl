# # Plotting

# MacroModelling.jl integrates a comprehensive plotting toolkit based on [StatsPlots.jl](https://github.com/JuliaPlots/StatsPlots.jl). The plotting API is exported together with the modelling macros, so once you define a model you can immediately visualise impulse responses, simulations, conditional forecasts, model estimates, variance decompositions, and policy functions. All plotting functions live in the `StatsPlotsExt` extension, which is loaded automatically when StatsPlots is imported or used.

# ## Setup

# Load the packages once per session:
# import Pkg
# Pkg.offline(true)
# Pkg.add(["Revise", "StatsPlots"])

using Revise
using MacroModelling
import StatsPlots

# Load a model:

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



# ## Impulse response functions (IRF)
# A call to `plot_irf` computes IRFs for **every exogenous shock** and **every endogenous variable**, using the model’s default solution method (first-order perturbation) and a **one-standard-deviation positive** shock.

plot_irf(Gali_2015_chapter_3_nonlinear, save_plots = true, save_plots_format = :png)

# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

# The plot shows every endogenous variable affected by each exogenous shock and annotates the title with the model name, shock identifier, sign of the impulse (positive by default), and the page indicator (e.g. `(1/3)`). Each subplot overlays the steady state as a horizontal reference line (non‑stochastic for first‑order solutions, stochastic otherwise) and, when the variable is strictly positive, adds a secondary axis with percentage deviations.

# ### Algorithm 
# [Default: :first_order, Type: Symbol]: algorithm to solve for the dynamics of the model. Available algorithms: :first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order
# You can plot IRFs for different solution algorithms. Here we use a second-order perturbation solution:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order, save_plots = true, save_plots_format = :png)

# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_second_order.png)
# The most notable difference is that at second order we observe dynamics for S, which is constant at first order (under certainty equivalence). Furthermore, the steady state levels changed due to the stochastic steady state incorporating precautionary behaviour (see horizontal lines).
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, save_plots = true, save_plots_format = :png)

# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

# We can compare the two solution methods side by side by plotting them on the same graph:

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, save_plots = true, save_plots_format = :png)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order, save_plots = true, save_plots_format = :png)

# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_first_and_second_order.png)
# In the plots now see both solution methods overlaid. The first-order solution is shown in blue, the second-order solution in orange, as indicated in the legend below the plot. Note that the steady state levels can be different for the two solution methods. For variables where the relevant steady state (non-stochastic steady state for first order and stochastic steady state for higher order) is the same (e.g. A) we see the level on the left axis and percentage deviations on the right axis. For variables where the steady state differs between the two solution methods (e.g. C) we only see absolute level deviations (abs. Δ) on the left axis. Furthermore, the relevant steady state level is mentioned in a table below the plot for reference (rounded so that you can spot the difference to the nearest comparable steady state).

# We can add more solution methods to the same plot:
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :pruned_third_order, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_multiple_orders.png)

# Note that the pruned third-order solution includes the effect of time varying risk and flips the sign for the reaction of MC and N. The additional solution is added to the plot as another colored line and another entry in the legend and a new entry in the table below highlighting the relevant steady states.


# ### Initial state 
# [Default: [0.0], Type: Union{Vector{Vector{Float64}},Vector{Float64}}]: The initial state defines the starting point for the model. In the case of pruned solution algorithms the initial state can be given as multiple state vectors (Vector{Vector{Float64}}). In this case the initial state must be given in deviations from the non-stochastic steady state. In all other cases the initial state must be given in levels. If a pruned solution algorithm is selected and initial_state is a Vector{Float64} then it impacts the first order initial state vector only. The state includes all variables as well as exogenous variables in leads or lags if present. get_irf(𝓂, shocks = :none, variables = :all, periods = 1) returns a KeyedArray with all variables. The KeyedArray type is provided by the AxisKeys package.

# The initial state defines the starting point for the IRF. The initial state needs to contain all variables of the model as well as any leads or lags if present. One way to get the correct ordering and number of variables is to call get_irf(𝓂, shocks = :none, variables = :all, periods = 1) which returns a KeyedArray with all variables in the correct order. The KeyedArray type is provided by the AxisKeys package. For 

init_state = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true)

# Only state variables will have an impact on the IRF. You can check which variables are state variables using:


get_state_variables(Gali_2015_chapter_3_nonlinear)
# Now lets modify the initial state and set nu to 0.1:
init_state(:nu,:,:) .= 0.1


# Now we can input the modified initial state into the plot_irf function as a vector:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state), save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_init_state.png)

# Note that we also defined the shock eps_a to see how the model reacts to a shock to A. For more details on the shocks input see the corresponding section.
# You can see the difference in the IRF compared to the IRF starting from the non-stochastic steady state. By setting nu to a higher level we essentially mix the effect of a shock to nu with a shock to A. Since here we are working with the linear solution we can disentangle the two effects by stacking the two components. Let's start with the IRF from the initial state as defined above:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, initial_state = vec(init_state), save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__no_shock__1_init_state.png)
# and then we stack the IRF from a shock to A on top of it:
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, plot_type = :stack, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1.png)

# Note how the two components are shown with a label attached to it that is explained in the table below. The blue line refers to the first input: without a shock and a non-zero initial state and the red line corresponds to the second input which start from the relevant steady state and shocks eps_a. Both components add up to the solid line that is the same as in the case of combining the eps_a shock with the initial state.

# We can do the same for higher order solutions. Lets start with the second order solution. First we get the initial state in levels from the second order solution:
init_state_2nd = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true, algorithm = :second_order)

# Then we set nu to 0.1:
init_state_2nd(:nu,:,:) .= 0.1

# and plot the IRF for eps_a starting from this initial state:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_2nd), algorithm = :second_order, save_plots = true, save_plots_format = :png)

# while here can as well stack the two components, they will not add up linearly since we are working with a non-linear solution. Instead we can compare the IRF from the initial state across the two solution methods:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state), save_plots = true, save_plots_format = :png)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_2nd), algorithm = :second_order, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_multi_sol.png)

# The plot shows two lines in the legend which are mapped to the relevant input differences in the table below. The first line corresponds to the initial state used for the first order solution as well as the IRF using the first order solution and the second line corresponds to the initial state used for the second order solution and using the second order solution. Note that the steady states are different across the two solution methods and thereby also the initial states except for nu which we set to 0.1 in both cases. Note as well a second table below the first one that shows the relevant steady states for both solution methods. The relevant steady state of A is the same across both solution methods and in the corresponding subplot we see the level on the left axis and percentage deviations on the right axis. For all other variables the relevant steady state differs across solution methods and we only see absolute level deviations (abs. Δ) on the left axis and the relevant steady states in the table at the bottom.

# For pruned solution methods the initial state can also be given as multiple state vectors (Vector{Vector{Float64}}). If a vector of vectors is provided the values must be in difference from the non-stochastic steady state. In case only one vector is provided, the values have to be in levels, and the impact of the initial state is assumed to have the full nonlinear effect in the first period. Providing a vector of vectors allows to set the pruned higher order auxilliary state vectors. This can be useful in some cases but do note that those higher order auxilliary state vector have only a linear impact on the dynamics. Let's start by assembling the vector of vectors:

init_state_pruned_3rd_in_diff = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true) - get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, algorithm = :pruned_third_order, levels = true)
# The first and third order dynamics do not have a risk impact on the steady state, so they are zero. The second order steady state has the risk adjustment. Let's assemble the vectors for the third order case:

init_states_pruned_3rd_vec = [zero(vec(init_state_pruned_3rd_in_diff)), vec(init_state_pruned_3rd_in_diff), zero(vec(init_state_pruned_3rd_in_diff))]

# Then we set nu to 0.1 in the first order terms. Inspecting init_state_pruned_3rd_in_diff we see that nu is the 18th variable in the vector:
init_states_pruned_3rd_vec[1][18] = 0.1


plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = init_states_pruned_3rd_vec, algorithm = :pruned_third_order, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_pruned_3rd_vec_vec.png)

# Equivalently we can use a simple vector as input for the initial state. In this case the values must be in levels and the impact of the initial state is assumed to have the full nonlinear effect in the first period:
init_state_pruned_3rd = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true, algorithm = :pruned_third_order)

init_state_pruned_3rd(:nu,:,:) .= 0.1

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_pruned_3rd), algorithm = :pruned_third_order, save_plots = true, save_plots_format = :png)

# Lets compare this now with the second order and first order version starting from their respective relevant steady states.

plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state_2nd), algorithm = :second_order, save_plots = true, save_plots_format = :png)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state), save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_multi_sol_w_init.png)
# Also here we see that the pruned third order solution changes the dynamics while the relevant steady states are the same as for the second order solution.


# ### Shocks
# shocks for which to calculate the IRFs. Inputs can be a shock name passed on as either a Symbol or String (e.g. :y, or "y"), or Tuple, Matrix or Vector of String or Symbol. :simulate triggers random draws of all shocks (excluding occasionally binding constraints (obc) related shocks). :all_excluding_obc will contain all shocks but not the obc related ones. :all will contain also the obc related shocks. A series of shocks can be passed on using either a Matrix{Float64}, or a KeyedArray{Float64} as input with shocks (Symbol or String) in rows and periods in columns. The KeyedArray type is provided by the AxisKeys package. The period of the simulation will correspond to the length of the input in the period dimension + the number of periods defined in periods. If the series of shocks is input as a KeyedArray{Float64} make sure to name the rows with valid shock names of type Symbol. Any shocks not part of the model will trigger a warning. :none in combination with an initial_state can be used for deterministic simulations.

# We can call individual shocks by name:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, save_plots = true, save_plots_format = :png)

# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

# The same works if we input the shock name as a string:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = "eps_a", save_plots = true, save_plots_format = :png)

# or multiple shocks at once (as strings or symbols):
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = [:eps_a, :eps_z], save_plots = true, save_plots_format = :png)


# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__3.png)

# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_z__3.png)

# This also works if we input multiple shocks as a Tuple:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = (:eps_a, :eps_z), save_plots = true, save_plots_format = :png)
# or a matrix:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = [:eps_a :eps_z], save_plots = true, save_plots_format = :png)


# Then there are some predefined options:
# - `:all_excluding_obc` (default) plots all shocks not used to enforce occasionally binding constraints (OBC).
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :all_excluding_obc, save_plots = true, save_plots_format = :png)

# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_nu__1.png)

# - `:all` plots all shocks including the OBC related ones.
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :all, save_plots = true, save_plots_format = :png)

# - `:simulate` triggers random draws of all shocks (excluding obc related shocks). You can set the seed to get reproducible results (e.g. `import Random; Random.seed!(10)`).
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :simulate, save_plots = true, save_plots_format = :png)

# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__simulation__1.png)

# - `:none` can be used in combination with an initial_state for deterministic simulations. See the section on initial_state for more details. Let's start by getting the initial state in levels:

init_state = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true)

# Only state variables will have an impact on the IRF. You can check which variables are state variables using:

get_state_variables(Gali_2015_chapter_3_nonlinear)
# Now lets modify the initial state and set nu to 0.1:
init_state(:nu,:,:) .= 0.1


# Now we can input the modified initial state into the plot_irf function as a vector and set shocks to :none:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, initial_state = vec(init_state), save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__no_shock__1.png)
# Note how this is similar to a shock to eps_nu but instead we set nu 0.1 in the initial state and then let the model evolve deterministically from there. In the title the reference to the shock disappeared as we set it to :none.

# We can also compare shocks:
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shocks[1], save_plots = true, save_plots_format = :png)

for s in shocks[2:end]
    plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = s, save_plots = true, save_plots_format = :png)
end
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1_linear.png)

# Now we see all three shocks overlaid in the same plot. The legend below the plot indicates which color corresponds to which shock and in the title we now see that all shocks are positive and we have multiple shocks in the plot.

# A series of shocks can be passed on using either a Matrix{Float64}, or a KeyedArray{Float64} as input with shocks (Symbol or String) in rows and periods in columns. Let's start with a KeyedArray:
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
n_periods = 3
shock_keyedarray = KeyedArray(zeros(length(shocks), n_periods), Shocks = shocks, Periods = 1:n_periods)
# and then we set a one standard deviation shock to eps_a in period 1, a negative 1/2 standard deviation shock to eps_z in period 2 and a 1/3 standard deviation shock to eps_nu in period 3:
shock_keyedarray("eps_a",[1]) .= 1
shock_keyedarray("eps_z",[2]) .= -1/2
shock_keyedarray("eps_nu",[3]) .= 1/3

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shock_keyedarray, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__shock_matrix__2.png)
# In the title it is now mentioned that the input is a series of shocks and the values of the shock processes Z and nu move with the shifted timing and note that the impact of the eps_z shock has a - in front of it in the model definition, which is why they both move in the same direction. Note also that the number of periods is prolonged by the number of periods in the shock input. Here we defined 3 periods of shocks and the default number of periods is 40, so we see 43 periods in total.

# The same can be done with a Matrix:
shock_matrix = zeros(length(shocks), n_periods)
shock_matrix[1,1] = 1
shock_matrix[3,2] = -1/2
shock_matrix[2,3] = 1/3

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shock_matrix, save_plots = true, save_plots_format = :png)

# In certain circumstances a shock matrix might correspond to a certain scenario and if we are working with linear solutions we can stack the IRF for different scenarios or components of scenarios. Let's say we have two scenarios defined by two different shock matrices:
shock_matrix_1 = zeros(length(shocks), n_periods)
shock_matrix_1[1,1] = 1
shock_matrix_1[3,2] = -1/2
shock_matrix_1[2,3] = 1/3   

shock_matrix_2 = zeros(length(shocks), n_periods * 2)
shock_matrix_2[1,4] = -1
shock_matrix_2[3,5] = 1/2
shock_matrix_2[2,6] = -1/3
# We can plot them on top of each other using the :stack option:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shock_matrix_1, save_plots = true, save_plots_format = :png)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = shock_matrix_2, plot_type = :stack, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__shock_matrix__2_mult_mats.png)

# The blue bars correspond to the first shock matrix and the red to the second shock matrix and they are labeled accordingly in the legend below the plot. The solid line corresponds to the sum of both components. Now we see 46 periods as the second shock matrix has 6 periods and the first one 3 periods and the default number of periods is 40.




# ### Periods
# number of periods for which to calculate the output. In case a matrix of shocks was provided, periods defines how many periods after the series of shocks the output continues.
# You set the number of periods to 10 like this (for the eps_a shock):
plot_irf(Gali_2015_chapter_3_nonlinear, periods = 10, shocks = :eps_a, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_10_periods.png)
# The x-axis adjust automatically and now only shows 10 periods.

# Let's take a shock matrix with 15 period length as input and set the periods argument to 20 and compare it to the previous plot with 10 periods:
shock_matrix = zeros(length(shocks), 15)
shock_matrix[1,1] = .1
shock_matrix[3,5] = -1/2
shock_matrix[2,15] = 1/3

plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = shock_matrix, periods = 20, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1_mixed_periods.png)
# The x-axis adjusted to 35 periods and we see the first plot ending after 10 periods and the second plot ending after 35 periods. The legend below the plot indicates which color corresponds to which shock and in the title we now see that we have multiple shocks in the plot.


# ### shock_size 
# affects the size of shocks as long as they are not set to :none or a shock matrix.
# [Default: 1.0, Type: Real]: size of the shocks in standard deviations. Only affects shocks that are not passed on as a matrix or KeyedArray or set to :none. A negative value will flip the sign of the shock.
# You can set the size of the shock using the shock_size argument. Here we set it to -2 standard deviations:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, shock_size = -2, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_shock_size.png)

# Note how the sign of the shock flipped and the size of the reaction increased.



# ### negative_shock
# calculate IRFs for a negative shock.
# [Default: false, Type: Bool]: if true, calculates IRFs for a negative shock. Only affects shocks that are not passed on as a matrix or KeyedArray or set to :none.

# You can also set negative_shock to true to get the IRF for a negative one standard deviation shock:
plot_irf(Gali_2015_chapter_3_nonlinear, negative_shock = true, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_z__1_neg_shock.png)



# ### variables
# [Default: :all_excluding_obc]: variables for which to show the results. Inputs can be a variable name passed on as either a Symbol or String (e.g. :y or "y"), or Tuple, Matrix or Vector of String or Symbol. Any variables not part of the model will trigger a warning. :all_excluding_auxiliary_and_obc contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). :all_excluding_obc contains all shocks less those related to auxiliary variables. :all will contain all variables.

# You can select specific variables to plot. Here we select only output (Y) and inflation (Pi) using a Vector of Symbols:
plot_irf(Gali_2015_chapter_3_nonlinear, variables = [:Y, :Pi], save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_var_select.png)
# The plot now only shows the two selected variables (sorted alphabetically) in a plot with two subplots for each shock.
# The same can be done using a Tuple:
plot_irf(Gali_2015_chapter_3_nonlinear, variables = (:Y, :Pi), save_plots = true, save_plots_format = :png)
# a Matrix:
plot_irf(Gali_2015_chapter_3_nonlinear, variables = [:Y :Pi], save_plots = true, save_plots_format = :png)
# or providing the variable names as strings:
plot_irf(Gali_2015_chapter_3_nonlinear, variables = ["Y", "Pi"], save_plots = true, save_plots_format = :png)
# or a single variable as a Symbol:
plot_irf(Gali_2015_chapter_3_nonlinear, variables = :Y, save_plots = true, save_plots_format = :png)
# or as a string:
plot_irf(Gali_2015_chapter_3_nonlinear, variables = "Y", save_plots = true, save_plots_format = :png)

# Then there are some predefined options:
# - `:all_excluding_auxiliary_and_obc` (default) plots all variables except auxiliary variables and those used to enforce occasionally binding constraints (OBC).
plot_irf(Gali_2015_chapter_3_nonlinear, variables = :all_excluding_auxiliary_and_obc, save_plots = true, save_plots_format = :png)
# - `:all_excluding_obc` plots all variables except those used to enforce occasionally binding constraints (OBC).
# In order to see the auxilliary variables let's use a model that has auxilliary variables defined. We can use the FS2000 model:
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
# both c and P appear in t+2 and will thereby add auxilliary variables to the model. If we now plot the IRF for all variables excluding obc related ones we see the auxilliary variables as well:
plot_irf(FS2000, variables = :all_excluding_obc, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__FS2000__e_a__1_aux.png.png)
# c and P appear twice, once as the variable itself and once as an auxilliary variable with the L(1) superscript, indicating that it is the value of the variable in t+1 as it is expected to be in t.

# - `:all` plots all variables including auxiliary variables and those used to enforce occasionally binding constraints (OBC). Therefore let's use the Gali_2015_chapter_3 model with an effective lower bound (note the max statement in the Taylor rule):
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

# if we now plot the IRF for all variables including obc related ones we see the obc related auxilliary variables as well:
plot_irf(Gali_2015_chapter_3_obc, variables = :all, save_plots = true, save_plots_format = :png)
# ![RBC_baseline IRF](../assets/irf__Gali_2015_chapter_3_obc__eps_z__3.png)
# Here you see the obc related variables in the last subplot.
# Note that given the eps_z shock the interest rate R hits the effective lower bound in period 1 and stays there for that period:
# ![RBC_baseline IRF](../assets/irf__Gali_2015_chapter_3_obc__eps_z__2.png)
# The effective lower bound is enforced using shocks to the equation containing the max statement. For details of the construction of the occasionally binding constraint see the documentation. For this specific model you can also look at the equations the parser wrote in order to enforce the obc:
get_equations(Gali_2015_chapter_3_obc)



# ### parameters
# If nothing is provided, the solution is calculated for the parameters defined previously. Acceptable inputs are a Vector of parameter values, a Vector or Tuple of Pairs of the parameter Symbol or String and value. If the new parameter values differ from the previously defined the solution will be recalculated.

# Let's start by changing the discount factor β from 0.99 to 0.95:
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.95, shocks = :eps_a, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_beta_0_95.png)
# The steady states and dynamics changed as a result of changing the discount factor. As it is a bit more difficult to see what changed between the previous IRF with β = 0.99 and the current one with β = 0.95, we can overlay the two IRFs. Since parameter changes are permanent we first must first set β = 0.99 again and then overlay the IRF with β = 0.95 on top of it:
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.99, shocks = :eps_a, save_plots = true, save_plots_format = :png)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.95, shocks = :eps_a, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_compare_beta.png)
# The legend below the plot indicates which color corresponds to which value of β and the table underneath the plot shows the relevant steady states for both values of β. Note that the steady states differ across the two values of β and also the dynamics, even when the steady state is still the same (e.g. for Y).

# We can also change multiple parameters at once and compare it to the previous plots. Here we change β to 0.97 and τ to 0.5 using a Tuple of Pairs and define the variables with Symbols:
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.97, :τ => 0.5), shocks = :eps_a, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_beta_tau.png)
# Since the calls to the plot function now differ in more than one input argument, the legend below the plot indicates which color corresponds to which combination of inputs and the table underneath the plot shows the relevant steady states for all three combinations of inputs.

# We can also use a Vector of Pairs:
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = [:β => 0.98, :τ => 0.25], shocks = :eps_a, save_plots = true, save_plots_format = :png)

# or simply a Vector of parameter values in the order they were defined in the model. We can get them by using:
params = get_parameters(Gali_2015_chapter_3_nonlinear, values = true)
param_vals = [p[2] for p in params]

plot_irf(Gali_2015_chapter_3_nonlinear, parameters = param_vals, shocks = :eps_a, save_plots = true, save_plots_format = :png)

# ### ignore_obc
# [Default: false, Type: Bool]: if true, ignores occasionally binding constraints (obc) even if they are part of the model. This can be useful for comparing the dynamics of a model with obc to the same model without obc.
# If the model has obc defined, we can ignore them using the ignore_obc argument. Here we compare the IRF of the Gali_2015_chapter_3_obc model with and without obc. Let's start by looking at the IRF for a 3 standard deviation eps_z shock with the obc enforced. The the shock size section and the variabels section for more details on the input arguments. By default obc is enforced so we can call:
plot_irf(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, save_plots = true, save_plots_format = :png)
# Then we can overlay the IRF ignoring the obc:
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, ignore_obc = true, save_plots = true, save_plots_format = :png)
# ![RBC_baseline IRF](../assets/irf__Gali_2015_chapter_3_obc__eps_z__1_ignore_obc.png)
# The legend below the plot indicates which color corresponds to which value of ignore_obc. Note how the interest rate R hits the effective lower bound in period 1 to 3 when obc is enforced (blue line) but not when obc is ignored (orange line). Also note how the dynamics of the other variables change as a result of enforcing the obc. The recession is deeper and longer when the obc is enforced. The length of the lower bound period depends on the size of the shock. 


# ### generalised_irf
# [Default: false, Type: Bool]: if true, calculates generalised IRFs (GIRFs) instead of standard IRFs. GIRFs are calculated by simulating the model with and without the shock and taking the difference. This is repeated for a number of draws and the average is taken. GIRFs can be used for models with non-linearities and/or state-dependent dynamics such as higher order solutions or models with occasionally binding constraints (obc).

# Lets look at the IRF of the Gali_2015_chapter_3_obc model for a 3 standard deviation eps_z shock with and without using generalised_irf. We start by looking at GIRF:
plot_irf(Gali_2015_chapter_3_obc, generalised_irf = true, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, save_plots = true, save_plots_format = :png)
# ![RBC_baseline IRF](../assets/irf__Gali_2015_chapter_3_obc__eps_z__1_girf.png)
# and then we overlay the standard IRF:
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, save_plots = true, save_plots_format = :png)
# ![RBC_baseline IRF](../assets/irf__Gali_2015_chapter_3_obc__eps_z__1_girf.png)
# The legend below the plot indicates which color corresponds to which value of generalised_irf. Note how the interest rate R hits the effective lower bound in period 1 to 3 when using the standard IRF (orange line). This suggest that for the GIRF the accepted draws covers many cases where the OBC is not binding. We can confirm this by also overlaying the IRF ignoring the OBC.
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, ignore_obc = true, save_plots = true, save_plots_format = :png)
# ![RBC_baseline IRF](../assets/irf__Gali_2015_chapter_3_obc__eps_z__1_girf_ignore_obc.png)
# We see that the IRF ignoring the obc sees R falling more, suggesting that the GIRF draws indeed covers cases where the OBC is  binding. The recession is deeper and longer when the obc is enforced. The length of the lower bound period depends on the size of the shock.

# Another use case for GIRFs is to look at the IRF of a model with a higher order solution. Let's look at the IRF of the Gali_2015_chapter_3_nonlinear model solved with pruned second order perturbation for a 1 standard deviation eps_a shock with and without using generalised_irf. We start by looking at GIRF:
plot_irf(Gali_2015_chapter_3_nonlinear, generalised_irf = true, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_girf.png)
# Some lines are very jittery highlighting the state-dependent nature of the GIRF and the dominant effec tof randomness (e.g. N or MC).

# Now lets overlay the standard IRF for the pruned second order solution:
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :pruned_second_order, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_girf_compare.png)

# The comparison of the IRFs for S reveals that the reaction of S is highly state dependent and can go either way depending on the state of the economy when the shock hits. The same is true for W_real, while the other variables are less state dependent and the GIRF and standard IRF are more similar.

# ### generalised_irf_warmup_iterations and generalised_irf_draws
# The number of draws and warmup iterations can be adjusted using the generalised_irf_draws and generalised_irf_warmup_iterations arguments. Increasing the number of draws will increase the accuracy of the GIRF at the cost of increased computation time. The warmup iterations are used to ensure that the starting points of the individual draws are exploring the state space sufficiently and are representative of the model's ergodic distribution.

# Lets start with the GIRF that had the wiggly lines above:
plot_irf(Gali_2015_chapter_3_nonlinear, generalised_irf = true, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_format = :png)

# and then we overlay the GIRF with 1000 draws:
plot_irf!(Gali_2015_chapter_3_nonlinear, generalised_irf = true, generalised_irf_draws = 1000, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_format = :png)
# here we see that the lines are less wiggly as the number of draws increased:
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_girf_1000_draws.png)

# and then we overlay the GIRF with 5000 draws:
plot_irf!(Gali_2015_chapter_3_nonlinear, generalised_irf = true, generalised_irf_draws = 5000, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_format = :png)
# lines are even less wiggly as the number of draws increased further:
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_girf_5000_draws.png)

# In order to fully cover the ergodic distribution of the model it can be useful to increase the number of warmup iterations as well. Here we overlay the standard IRF for the pruned second order solution with the GIRF with 5000 draws and 500 warmup iterations:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :pruned_second_order, save_plots = true, save_plots_format = :png)

plot_irf!(Gali_2015_chapter_3_nonlinear, generalised_irf = true, generalised_irf_draws = 5000, generalised_irf_warmup_iterations = 500, shocks = :eps_a,  algorithm = :pruned_second_order, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_girf_5000_draws_500_warmup.png)
# With this amount of draws and wamrup itereations the difference between the GIRF and standard IRF is very small. This suggest that there is little state-dependence in the model with a second order pruned solution for a 1 standard deviation eps_a shock and the initial insight from the GIRF with 100 draws and 50 warmup iterations was mainly driven by randomness.


# ### label
# Labels for the plots are shown when you use the plot_irf! function to overlay multiple IRFs. By default the label is just a running number but this argument can be used to provide custom labels. Acceptable inputs are a String, Symbol, or a Real.

# Using labels can be useful when the inputs differs in complex ways (shock matrices or multiple input changes) and you want to provide a more descriptive label. 
# Let's for example compare the IRF of the Gali_2015_chapter_3_nonlinear model for a 1 standard deviation eps_a shock with β = 0.99 and τ = 0 to the IRF with β = 0.95 and τ = 0.5 using custom labels String input:
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.99, :τ => 0.0), shocks = :eps_a, label = "Std. params", save_plots = true, save_plots_format = :png)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.95, :τ => 0.5), shocks = :eps_a, label = "Alt. params", save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_custom_labels.png)
# The plot now has the name of the labels in the legend below the plot instead of just 1 and 2. Furthermore, the tables highlighting the relevant input differences and relevant steady states also have the labels in the first column instead of just 1 and 2.

# The same can be achieved using Symbols as inputs (though they are a bit less expressive):
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.99, :τ => 0.0), shocks = :eps_a, label = :standard, save_plots = true, save_plots_format = :png)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.95, :τ => 0.5), shocks = :eps_a, label = :alternative, save_plots = true, save_plots_format = :png)

# or with Real inputs:
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.99, :τ => 0.0), shocks = :eps_a, label = 0.99, save_plots = true, save_plots_format = :png)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.95, :τ => 0.5), shocks = :eps_a, label = 0.95, save_plots = true, save_plots_format = :svg)


# ### plot_attributes
# [Default: Dict()]: dictionary of attributes passed on to the plotting function. See the Plots.jl documentation for details.

# You can also change the color palette using the plot_attributes argument. Here we define a custom color palette (inspired by the color scheme used in the European Commissions economic reports) and use it to plot the IRF of all shocks defined in the Gali_2015_chapter_3_nonlinear model and stack them on top of each other:
# First we define the custom color palette using hex color codes:
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


# Then we get all shocks defined in the model:
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)

# and then we plot the IRF for the first shock:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shocks[1], save_plots = true, save_plots_format = :png)

# and then we overlay the IRF for the remaining shocks using the custom color palette by passing on a dictionnary:
for s in shocks[2:end]
    plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = s, plot_attributes = Dict(:palette => ec_color_palette), plot_type = :stack, save_plots = true, save_plots_format = :png)
end
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__2_ec_colors.png)
# The colors of the shocks now follow the custom color palette.

# We can also change other attributes such as the font family (see [here](https://github.com/JuliaPlots/Plots.jl/blob/v1.41.1/src/backends/gr.jl#L61) for options):
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, plot_attributes = Dict(:fontfamily => "computer modern"), save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_cm_font.png)
# All text in the plot is now in the computer modern font. Do note that the rendering of the fonts inherits the constraints of the plotting backend (GR in this case) - e.g. the superscript + is not rendered properly for this font.


# ### plots_per_page
# [Default: 6, Type: Int]: number of subplots per page. If the number of variables to plot exceeds this number, multiple pages will be created.
# Lets select 9 variables to plot and set plots_per_page to 4:
plot_irf(Gali_2015_chapter_3_nonlinear, variables = [:Y, :Pi, :R, :C, :N, :W_real, :MC, :i_ann, :A], shocks = :eps_a, plots_per_page = 2, save_plots = true, save_plots_format = :png)
# ![RBC IRF](../assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_9_vars_2_per_page.png)
# The first page shows the first two variables (sorted alphabetically) in a plot with two subplots for each shock. The title indicates that this is page 1 of 5.

# ### show_plots
# [Default: true, Type: Bool]: if true, shows the plots otherwise they are just returned as an object. 
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, show_plots = false)

# ### save_plots, save_plots_format, save_plots_path, save_pots_name
# [Default: false, Type: Bool]: if true, saves the plots to disk otherwise they are just shown and returned as an object. The plots are saved in the format specified by the save_plots_format argument and in the path specified by the save_plots_path argument (the fodlers will be created if they dont exist already). Each plot is saved as a separate file with a name that indicates the model name, shocks, and a running number if there are multiple plots. The default path is the current working directory (pwd()) and the default format is :pdf. Acceptable formats are those supported by the Plots.jl package ([input formats compatible with GR](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats)).

# Here we save the IRFs for all variables and all shocks of the Gali_2015_chapter_3_nonlinear model as a svg file in a directory one level up in the folder hierarchy in a new folder called `plots` with the filename prefix: `:impulse_response`:
plot_irf(Gali_2015_chapter_3_nonlinear, save_plots = true, save_plots_format = :png, save_plots_path = "./../plots", save_plots_name = :impulse_response)

# The plots appear in the specified folder with the specified prefix. Each plot is saved in a separate file. The naming reflects the model used, the shock shown and the running index per shocks if the number of variables exceeds the number of plots per page.


# ### verbose
# [Default: false, Type: Bool]: if true, enables verbose output related to the solution of the model
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, verbose = true)

# The code outputs information about the solution of the steady state blocks.
# If we change the parameters he also needs to redo the first order solution:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, parameters = :β => 0.955, verbose = true)



# ### tol
# [Default: Tolerances(), Type: Tolerances]: define various tolerances for the algorithm used to solve the model. See documentation of Tolerances for more details: ?Tolerances
# You can adjust the tolerances used in the numerical solvers. The Tolerances object allows you to set tolerances for the non-stochastic steady state solver (NSSS), Sylvester equations, Lyapunov equation, and quadratic matrix equation (qme). For example, to set tighter tolerances:
custom_tol = Tolerances(qme_tol = 1e-16, lyapunov_tol = 1e-16)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, tol = custom_tol, save_plots = true, save_plots_format = :png)

# This can be useful when you need higher precision in the solution or when the default tolerances are not sufficient for convergence.


# ### quadratic_matrix_equation_algorithm
# [Default: :schur, Type: Symbol]: algorithm to solve quadratic matrix equation (A * X ^ 2 + B * X + C = 0). Available algorithms: :schur, :doubling
# The quadratic matrix equation solver is used internally when solving the model. You can choose between different algorithms. The :schur algorithm is generally faster and more reliable, while :doubling can be more precise in some cases:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, quadratic_matrix_equation_algorithm = :doubling, save_plots = true, save_plots_format = :png)

# For most use cases, the default :schur algorithm is recommended.


# ### sylvester_algorithm
# [Default: selector that uses :doubling for smaller problems and switches to :bicgstab for larger problems, Type: Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}}]: algorithm to solve the Sylvester equation (A * X * B + C = X). Available algorithms: :doubling, :bartels_stewart, :bicgstab, :dqgmres, :gmres. Input argument can contain up to two elements in a Vector or Tuple. The first (second) element corresponds to the second (third) order perturbation solutions' Sylvester equation. If only one element is provided it corresponds to the second order perturbation solutions' Sylvester equation.
# You can specify which algorithm to use for solving Sylvester equations. For a second-order solution, you might want to use the :bartels_stewart algorithm:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order, sylvester_algorithm = :bartels_stewart, save_plots = true, save_plots_format = :png)

# For third-order solutions, you can specify different algorithms for the second and third order Sylvester equations using a Tuple:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :third_order, sylvester_algorithm = (:doubling, :bicgstab), save_plots = true, save_plots_format = :png)

# The choice of algorithm can affect both speed and precision, with :doubling and :bartels_stewart generally being faster but :bicgstab, :dqgmres, and :gmres being better for large sparse problems.


# ### lyapunov_algorithm
# [Default: :doubling, Type: Symbol]: algorithm to solve Lyapunov equation (A * X * A' + C = X). Available algorithms: :doubling, :bartels_stewart, :bicgstab, :gmres
# The Lyapunov equation solver is used when computing variance-covariance matrices. You can choose between different algorithms. The :doubling algorithm is fast and precise for most cases:
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, lyapunov_algorithm = :bartels_stewart, save_plots = true, save_plots_format = :png)

# For large sparse problems, iterative methods like :bicgstab or :gmres might be more efficient, though they may be less precise.


# ### changing more than one input and using ! function
