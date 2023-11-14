using MacroModelling

@model GI2017 begin
	c[0] + c1[0] + ik[0] = y[0]

	uc[0] = BETA * r[0] / dp[1] * uc[1]

	uc[0] * w[0] / xw[0] = az[0] * n[0] ^ ETA

	uc[0] * q[0] = uh[0] + uc[1] * BETA * q[1]

	c1[0] + q[0] * (h1[0] - h1[-1]) + r[-1] * b[-1] / dp[0] = w1[0] * n1[0] + b[0] + INDTR * log(ap[0])

	uc1[0] * (1 - lm[0]) = BETA1 * (r[0] / dp[1] - RHOD * lm[1] / dp[1]) * uc1[1]

	w1[0] * uc1[0] / xw1[0] = az[0] * n1[0] ^ ETA

	q[0] * uc1[0] = uh1[0] + uc1[1] * q[1] * BETA1 + q[0] * uc1[0] * lm[0] * (1 - RHOD) * M

	y[0] = n[0] ^ ((1 - ALPHA) * (1 - SIGMA)) * n1[0] ^ ((1 - ALPHA) * SIGMA) * k[-1] ^ ALPHA

	y[0] * (1 - ALPHA) * (1 - SIGMA) = n[0] * w[0] * xp[0]

	y[0] * (1 - ALPHA) * SIGMA = n1[0] * w1[0] * xp[0]

	log(dp[0] / PIBAR) - LAGP * log(dp[-1] / PIBAR) = BETA * (log(dp[1] / PIBAR) - log(dp[0] / PIBAR) * LAGP) - (1 - TETAP) * (1 - BETA * TETAP) / TETAP * log(xp[0] / XP_SS) + log(ap[0]) * (1 - INDTR)

	log(dw[0] / PIBAR) - LAGW * log(dw[-1] / PIBAR) = BETA * (log(dw[1] / PIBAR) - log(dw[0] / PIBAR) * LAGW) - (1 - TETAW) * (1 - BETA * TETAW) / TETAW * log(xw[0] / XW_SS) + log(aw[0])

	log(dw1[0] / PIBAR) - LAGW * log(dw1[-1] / PIBAR) = log(aw[0]) + BETA * (log(dw1[1] / PIBAR) - LAGW * log(dw1[0] / PIBAR)) - (1 - TETAW) * (1 - BETA * TETAW) / TETAW * log(xw1[0] / XW_SS)

	log(rnot[0]) = TAYLOR_R * log(r[-1]) + (1 - TAYLOR_R) * TAYLOR_P * (log(dp[0] / PIBAR) * 0.25 + 0.25 * log(dp[-1] / PIBAR) + 0.25 * log(dp[-2] / PIBAR) + 0.25 * log(dp[-3] / PIBAR)) + (1 - TAYLOR_R) * TAYLOR_Y * log(y[0] / lly) + (1 - TAYLOR_R) * TAYLOR_Q / 4 * log(q[0] / q[-1]) + (1 - TAYLOR_R) * log(PIBAR / BETA) + log(arr[0])

	uc[0] = (1 - EC) / (1 - BETA * EC) * (az[0] / (c[0] - EC * c[-1]) - BETA * EC * az[1] / (c[1] - c[0] * EC))

	uc1[0] = (1 - EC) / (1 - BETA1 * EC) * (az[0] / (c1[0] - EC * c1[-1]) - az[1] * BETA1 * EC / (c1[1] - c1[0] * EC))

	uh[0] = (1 - EH) / (1 - BETA * EH) * JEI * (az[0] * aj[0] / (1 - h1[0] - EH * (1 - h1[-1])) - az[1] * BETA * EH * aj[1] / (1 - h1[1] - EH * (1 - h1[0])))

	uh1[0] = JEI * (1 - EH) / (1 - BETA1 * EH) * (az[0] * aj[0] / (h1[0] - h1[-1] * EH) - aj[1] * az[1] * BETA1 * EH / (h1[1] - h1[0] * EH))

	uc[0] * qk[0] * (1 - PHIK * (ik[0] - ik[-1]) / llik) = uc[0] - PHIK * BETA * uc[1] * qk[1] * (ik[1] - ik[0]) / llik

	uc[0] * qk[0] / ak[0] = BETA * uc[1] * (rk[1] + qk[1] * (1 - DK) / ak[1])

	k[0] / ak[0] = ik[0] + k[-1] * (1 - DK) / ak[0]

	y[0] * ALPHA = k[-1] * xp[0] * rk[0]

	dw[0] = w[0] * dp[0] / w[-1]

	dw1[0] = dp[0] * w1[0] / w1[-1]

	log(aj[0]) = RHO_J * log(aj[-1]) + z_j[0]

	z_j[0] = RHO_J2 * z_j[-1] + eps_j[x]

	log(ak[0]) = RHO_K * log(ak[-1]) + STD_K * eps_k[x]

	log(ap[0]) = RHO_P * log(ap[-1]) + STD_P * eps_p[x]

	log(aw[0]) = RHO_W * log(aw[-1]) + STD_W * eps_w[x]

	log(arr[0]) = RHO_R * log(arr[-1]) + STD_R * eps_r[x]

	log(az[0]) = RHO_Z * log(az[-1]) + STD_Z * eps_z[x]

	0 = min(bnot[0] - b[0], lm[0])
	# bnot[0] = b[0]

	bnot[0] = h1[0] * q[0] * (1 - RHOD) * M + b[-1] * RHOD / dp[0]

	maxlev[0] = b[0] - bnot[0]

	r[0] = max(RBAR, rnot[0])
	# r[0] = rnot[0]

end


@parameters GI2017 begin
	RBAR = 1

	BETA = 0.995

	BETA1 = 0.9921849949330452

	EC = 0.6841688730310923

	EH = 0.8798650668795864

	ETA = 1

	JEI = 0.04

	M = 0.9

	ALPHA = 0.3

	PHIK = 4.120924218703865

	DK = 0.025

	LAGP = 0

	LAGW = 0

	PIBAR = 1.005

	INDTR = 0

	SIGMA = 0.5012798413194606

	TAYLOR_P = 1.719559906725518

	TAYLOR_Q = 0

	TAYLOR_R = 0.5508743735338286

	TAYLOR_Y = 0.09436959071018983

	TETAP = 0.9182319022631061

	TETAW = 0.9162909334165672

	XP_SS = 1.2

	XW_SS = 1.2

	RHO_J = 0.983469150669198

	RHO_K = 0.7859395713107814

	RHO_P = 0

	RHO_R = 0.623204934949152

	RHO_W = 0

	RHO_Z = 0.7555575007590176

	STD_J = 0.07366860797541266

	STD_K = 0.03601489154765812

	STD_P = 0.002964296803248907

	STD_R = 0.001315097718876929

	STD_W = 0.00996414482032244

	STD_Z = 0.01633680112129254

	RHO_J2 = 0

	RHOD = 0.6945068431131589

	ITAYLOR_W = 0

	llr = 1 / BETA

	llrk = llr - (1-DK)

	llxp = XP_SS

	llxw = XW_SS

	llxw1 = XW_SS

	lllm = (1 - BETA1/BETA) / (1 - BETA1*RHOD/PIBAR)

	QHTOC = JEI/(1-BETA)

	QH1TOC1 = JEI/(1-BETA1-lllm*M*(1-RHOD))

	KTOY = ALPHA/(llxp*llrk)

	BTOQH1 = M*(1-RHOD)/(1-RHOD/PIBAR)

	C1TOY = (1-ALPHA)*SIGMA/(1+(1/BETA-1)*BTOQH1*QH1TOC1)*(1/llxp)

	CTOY = (1-C1TOY-DK*KTOY)

	lln = ((1-SIGMA)*(1-ALPHA)/(llxp*llxw*CTOY))^(1/(1+ETA))

	lln1 = (SIGMA*(1-ALPHA)/(llxp*llxw1*C1TOY))^(1/(1+ETA))

	lly = KTOY^(ALPHA/(1-ALPHA))*lln^(1-SIGMA)*lln1^SIGMA

	llctot = lly-DK*KTOY*lly

	llik = KTOY*DK*lly

	llk = KTOY*lly 

	llq = QHTOC*CTOY*lly + QH1TOC1*C1TOY*lly

	k > 16.5
	q > 12
end

# SS(GI2017)

# min.([1,2], [3,1.8])
# -1 0
# 1 0
# 1 2
# -1 2
# 2 1
# 2 -1
# 1 -2

GI2017.obc_violation_function

import StatsPlots

plot_irf(GI2017, shocks = :eps_z, parameters = :STD_Z => .2, variables = :all, negative_shock = true)


plot_irf(GI2017, shocks = :eps_z)
plot_irf(GI2017, variables = :all)
plot_irf(GI2017, variables = :all, negative_shock = true)

plot_irf(GI2017, shocks = :eps_z, parameters = :STD_Z => .2, variables = :all)

plot_irf(GI2017, shocks = :eps_z, parameters = :STD_Z => .2, variables = :all, negative_shock = true)
plot_irf(GI2017, shocks = :eps_z, parameters = :STD_Z => .2, variables = :all, negative_shock = true, ignore_obc = true)

plot_irf(GI2017, shocks = :eps_z, negative_shock = true, ignore_obc = true)

plot_irf(GI2017, ignore_obc = true, shocks = :eps_z, negative_shock = true, parameters = :STD_Z => .2, variables = :all)


plot_irf(GI2017, shocks = :eps_z, negative_shock = true, variables = :all)





get_solution(GI2017, parameters = :STD_Z => .2)

import MacroModelling: parse_algorithm_to_state_update
import JuMP, NLopt, MadNLP
import LinearAlgebra as ℒ

𝓂 = GI2017
T = 𝓂.timings
algorithm = :first_order

unconditional_forecast_horizon = 40

obc_shock_idx = contains.(string.(T.exo),"ᵒᵇᶜ")
periods_per_shock = sum(obc_shock_idx)÷length(𝓂.obc_violation_equations)
num_shocks = length(𝓂.obc_violation_equations)


present_shocks = -Float64.(:eps_z .== T.exo)
state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂)

past_states = zeros(T.nVars)
past_shocks = zeros(T.nExo)

reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())



state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂)
periods = 40

Y = zeros(T.nVars,periods)
# Y = zeros(Real,T.nVars,periods,1)
# T.exo
obc_shocks = [i[1] for i in 𝓂.obc_shock_bounds]

obc_shock_idx = contains.(string.(T.exo),"ᵒᵇᶜ")

shocks = zeros(T.nExo,periods)
shocks[:,1] = -Float64.(:eps_z .== T.exo)
shock_values = shocks[:,1]

shocks[obc_shock_idx,:] .= 0

reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())
# shock_history[16,1]

past_initial_state = zeros(T.nVars)
past_shocks = zeros(T.nExo)


periods_per_shock = sum(obc_shock_idx)÷length(𝓂.obc_violation_equations)
num_shocks = length(𝓂.obc_violation_equations)



precision_factor = 1.0
past_states = past_initial_state
past_shocks = past_shocks
present_shocks = shock_values
unconditional_forecast_horizon = 40

state_update = 𝓂.solution.perturbation.first_order.state_update

reference_steady_state = 𝓂.solution.non_stochastic_steady_state

obc_shock_idx = contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")

obc_inequalities_idx = findall(x->contains(string(x), "Χᵒᵇᶜ") , 𝓂.var)

periods_per_shock = sum(obc_shock_idx)÷length(obc_inequalities_idx)

num_shocks = length(obc_inequalities_idx)


function obc_state_update(past_states::Vector{R}, past_shocks::Vector{R}, present_shocks::Vector{R}, 𝓂) where R <: Float64
	unconditional_forecast_horizon = 40

	state_update = 𝓂.solution.perturbation.first_order.state_update

	reference_steady_state = 𝓂.solution.non_stochastic_steady_state

	obc_shock_idx = contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")

	periods_per_shock = 𝓂.max_obc_shift + 1
	
	num_shocks = sum(obc_shock_idx)÷periods_per_shock

	constraints_violated = any(JuMP.value.(𝓂.obc_violation_function(zeros(num_shocks*periods_per_shock), past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1]) .> 1e-12)
	
	if constraints_violated
		# Find shocks fulfilling constraint
		# model = JuMP.Model(MadNLP.Optimizer)
		model = JuMP.Model(NLopt.Optimizer)
		# JuMP.set_attribute(model, "algorithm", :LD_SLSQP)
		JuMP.set_attribute(model, "algorithm", :AUGLAG)
		JuMP.set_attribute(model, "local_optimizer", :LD_LBFGS)
		# JuMP.set_attribute(model, "algorithm", :LD_MMA)

		JuMP.set_silent(model)

		# JuMP.set_attribute(model, "tol", 1e-12)

		# Create the variables over the full set of indices first.
		JuMP.@variable(model, x[1:num_shocks*periods_per_shock])
		
		# Now loop through obc_shock_bounds to set the bounds on these variables.
		# maxmin_indicators = 𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[2]
		# for (idx, v) in enumerate(maxmin_indicators)
		#     idxs = (idx - 1) * periods_per_shock + 1:idx * periods_per_shock
		#     if v
		# # #         if 𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[2][idx]
		#         JuMP.set_upper_bound.(x[idxs], 0)
		#             # JuMP.set_lower_bound.(x[idxs], 0)
		#     else
		#             # JuMP.set_upper_bound.(x[idxs], 0)
		#         JuMP.set_lower_bound.(x[idxs], 0)
		#     end
		# # #     # else
		#         # if 𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[2][idx]
		#         #     JuMP.set_lower_bound.(x[idxs], 0)
		#         # else
		#         #     JuMP.set_upper_bound.(x[idxs], 0)
		#         # end
		# # #     # end
		# end

		JuMP.@objective(model, Min, x' * ℒ.I * x)

		JuMP.@constraint(model, 𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1] .<= 0)

		JuMP.optimize!(model)
		
		solved = JuMP.termination_status(model) ∈ [JuMP.OPTIMAL,JuMP.LOCALLY_SOLVED]

		# precision = JuMP.objective_value(model)

		# if precision > eps(Float32) @warn "Bounds enforced up to reduced precision: $precision" end # I need the dual value (constraints). this relates to the shock size

		present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")] .= JuMP.value.(x)
	else
		solved = true
	end

	present_states = state_update(past_states,JuMP.value.(past_shocks))

	return present_states, present_shocks, solved
end



shock_values = shocks[:,1]
obc_state_update(past_initial_state, past_shocks, shock_values, 𝓂)
past_states, past_shocks, solved  = obc_state_update(past_initial_state, past_shocks, shock_values, 𝓂)
shocks[:,1] = past_shocks
if !solved @warn "No solution at iteration 1" end

for i in 2:periods
    shock_values = shocks[:,i]
    past_states, past_shocks, solved  = obc_state_update(past_states, past_shocks, shock_values, 𝓂)
    Y[:,i-1] = past_states
    shocks[:,i] = past_shocks
    if !solved 
        @warn "No solution at iteration $i" 
        break 
    end
end









obc_state_update(past_states, past_shocks, shock_values, 𝓂)

past_states::Vector{R}, past_shocks::Vector{R}, present_shocks::Vector{R}

i = 11

present_shocks = shocks[:,i]

unconditional_forecast_horizon = 40

state_update = 𝓂.solution.perturbation.first_order.state_update

reference_steady_state = 𝓂.solution.non_stochastic_steady_state

obc_shock_idx = contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")

periods_per_shock = 𝓂.max_obc_shift + 1

num_shocks = sum(obc_shock_idx)÷periods_per_shock

constraints_violated = any(JuMP.value.(𝓂.obc_violation_function(zeros(num_shocks*periods_per_shock), past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1]) .> 1e-12)

# if constraints_violated
	# Find shocks fulfilling constraint
	# model = JuMP.Model(MadNLP.Optimizer)
	model = JuMP.Model(NLopt.Optimizer)
	# JuMP.set_attribute(model, "algorithm", :LD_SLSQP)
	JuMP.set_attribute(model, "algorithm", :AUGLAG)
	JuMP.set_attribute(model, "local_optimizer", :LD_LBFGS)
	# JuMP.set_attribute(model, "algorithm", :LD_MMA)

	JuMP.set_silent(model)

	# JuMP.set_attribute(model, "tol", 1e-12)

	# Create the variables over the full set of indices first.
	JuMP.@variable(model, x[1:num_shocks*periods_per_shock])
	
	# Now loop through obc_shock_bounds to set the bounds on these variables.
	# maxmin_indicators = 𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[2]
	# for (idx, v) in enumerate(maxmin_indicators)
	#     idxs = (idx - 1) * periods_per_shock + 1:idx * periods_per_shock
	#     if v
	# #         if 𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[2][idx]
	#         JuMP.set_upper_bound.(x[idxs], 0)
	# #             JuMP.set_lower_bound.(x[idxs], 0)
	#     else
	# #             JuMP.set_upper_bound.(x[idxs], 0)
	#         JuMP.set_lower_bound.(x[idxs], 0)
	#     end
	# #     # else
	# #     #     if 𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[2][idx]
	# #     #         JuMP.set_lower_bound.(x[idxs], 0)
	# #     #     else
	# #     #         JuMP.set_upper_bound.(x[idxs], 0)
	# #     #     end
	# #     # end
	# end

	JuMP.@objective(model, Min, x' * ℒ.I * x)

	JuMP.@constraint(model, 𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1] .<= 0)

	JuMP.optimize!(model)
	
	solved = JuMP.termination_status(model) ∈ [JuMP.OPTIMAL,JuMP.LOCALLY_SOLVED]

	# precision = JuMP.objective_value(model)

	# if precision > eps(Float32) @warn "Bounds enforced up to reduced precision: $precision" end # I need the dual value (constraints). this relates to the shock size

	present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")] .= JuMP.value.(x)
# else
# 	solved = true
# end

present_states = state_update(past_states,JuMP.value.(past_shocks))




import StatsPlots
sum(abs,shocks,dims = 1)
plot_irf(GI2017, shocks = shocks)