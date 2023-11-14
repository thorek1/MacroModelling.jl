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

get_solution(GI2017, parameters = :STD_Z => .2)
# import StatsPlots

# plot_irf(GI2017, ignore_obc = true, shocks = :eps_z, negative_shock = true, parameters = :STD_Z => .2, variables = :all)

# check obc violation function
import MacroModelling: parse_algorithm_to_state_update
import JuMP, NLopt

𝓂 = GI2017
T = 𝓂.timings
algorithm = :first_order

unconditional_forecast_horizon = 40

obc_shock_idx = contains.(string.(T.exo),"ᵒᵇᶜ")
periods_per_shock = sum(obc_shock_idx)÷length(𝓂.obc_violation_equations)
num_shocks = length(𝓂.obc_violation_equations)

shocks = zeros(T.nExo,unconditional_forecast_horizon)
shocks[:,1] = -Float64.(:eps_z .== T.exo)
shock_values = shocks[:,1]


present_shocks = -Float64.(:eps_z .== T.exo)
state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂)

past_states = zeros(T.nVars)
past_shocks = zeros(T.nExo)

reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

# GI2017 = nothing
# import MathOptInterface as MOI
import Ipopt
import LinearAlgebra as ℒ
import JuMP, NLopt

# 𝓂 = testmax
# 𝓂 = borrcon
# 𝓂 = RBC
𝓂 = GI2017
algorithm = :first_order
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
MacroModelling.solve!(𝓂, parameters = :activeᵒᵇᶜshocks => 1, verbose = false, dynamics = true, algorithm = algorithm)
state_update = 𝓂.solution.perturbation.first_order.state_update

reference_steady_state = 𝓂.solution.non_stochastic_steady_state

obc_shock_idx = contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")

obc_inequalities_idx = findall(x->contains(string(x), "Χᵒᵇᶜ") , 𝓂.var)

periods_per_shock = sum(obc_shock_idx)÷length(obc_inequalities_idx)

num_shocks = length(obc_inequalities_idx)

using BenchmarkTools, MadNLP, Clarabel, COSMO#, Optimization, OptimizationNLopt


function obc_violation_function(x, past_initial_state, past_shocks, state_update, reference_steady_state, 𝓂, algorithm, periods, shock_values)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:326 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:334 =#
    T = 𝓂.timings
    # println(x[1])
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:336 =#
    Y = zeros(JuMP.AffExpr, T.nVars, periods + 2)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:338 =#
    # shock_values = typeof(x[1]).(shock_values)
    shock_values[contains.(string.(T.exo), "ᵒᵇᶜ")] .= x
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:340 =#
    zero_shock = zero(shock_values)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:342 =#
    Y[:, 1] = state_update(past_initial_state, past_shocks)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:344 =#
    Y[:, 2] = state_update(Y[:, 1], shock_values)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:346 =#
    for t = 2:periods + 1
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:347 =#
        Y[:, t + 1] = state_update(Y[:, t], zero_shock)
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:348 =#
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:350 =#
    Y .+= reference_steady_state[1:T.nVars]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:352 =#
    aj₍₁₎ = Y[1, 3:end]
    ak₍₁₎ = Y[2, 3:end]
    az₍₁₎ = Y[6, 3:end]
    c₍₁₎ = Y[9, 3:end]
    c1₍₁₎ = Y[10, 3:end]
    dp₍₁₎ = Y[11, 3:end]
    dw₍₁₎ = Y[14, 3:end]
    dw1₍₁₎ = Y[15, 3:end]
    h1₍₁₎ = Y[16, 3:end]
    ik₍₁₎ = Y[17, 3:end]
    lm₍₁₎ = Y[19, 3:end]
    q₍₁₎ = Y[23, 3:end]
    qk₍₁₎ = Y[24, 3:end]
    rk₍₁₎ = Y[26, 3:end]
    uc₍₁₎ = Y[28, 3:end]
    uc1₍₁₎ = Y[29, 3:end]
    aj₍₀₎ = Y[1, 2:end - 1]
    ak₍₀₎ = Y[2, 2:end - 1]
    ap₍₀₎ = Y[3, 2:end - 1]
    arr₍₀₎ = Y[4, 2:end - 1]
    aw₍₀₎ = Y[5, 2:end - 1]
    az₍₀₎ = Y[6, 2:end - 1]
    b₍₀₎ = Y[7, 2:end - 1]
    bnot₍₀₎ = Y[8, 2:end - 1]
    c₍₀₎ = Y[9, 2:end - 1]
    c1₍₀₎ = Y[10, 2:end - 1]
    dp₍₀₎ = Y[11, 2:end - 1]
    dpᴸ⁽⁻²⁾₍₀₎ = Y[11, 2:end - 1]
    dpᴸ⁽⁻¹⁾₍₀₎ = Y[11, 2:end - 1]
    dw₍₀₎ = Y[14, 2:end - 1]
    dw1₍₀₎ = Y[15, 2:end - 1]
    h1₍₀₎ = Y[16, 2:end - 1]
    ik₍₀₎ = Y[17, 2:end - 1]
    k₍₀₎ = Y[18, 2:end - 1]
    lm₍₀₎ = Y[19, 2:end - 1]
    maxlev₍₀₎ = Y[20, 2:end - 1]
    n₍₀₎ = Y[21, 2:end - 1]
    n1₍₀₎ = Y[22, 2:end - 1]
    q₍₀₎ = Y[23, 2:end - 1]
    qk₍₀₎ = Y[24, 2:end - 1]
    r₍₀₎ = Y[25, 2:end - 1]
    rk₍₀₎ = Y[26, 2:end - 1]
    rnot₍₀₎ = Y[27, 2:end - 1]
    uc₍₀₎ = Y[28, 2:end - 1]
    uc1₍₀₎ = Y[29, 2:end - 1]
    uh₍₀₎ = Y[30, 2:end - 1]
    uh1₍₀₎ = Y[31, 2:end - 1]
    w₍₀₎ = Y[32, 2:end - 1]
    w1₍₀₎ = Y[33, 2:end - 1]
    xp₍₀₎ = Y[34, 2:end - 1]
    xw₍₀₎ = Y[35, 2:end - 1]
    xw1₍₀₎ = Y[36, 2:end - 1]
    y₍₀₎ = Y[37, 2:end - 1]
    z_j₍₀₎ = Y[38, 2:end - 1]
    Χᵒᵇᶜ⁺ꜝ²ꜝ₍₀₎ = Y[39, 2:end - 1]
    Χᵒᵇᶜ⁻ꜝ¹ꜝ₍₀₎ = Y[40, 2:end - 1]
    χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎ = Y[41, 2:end - 1]
    χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎ = Y[42, 2:end - 1]
    χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎ = Y[43, 2:end - 1]
    χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎ = Y[44, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝ₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²²⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²³⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²¹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁴⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁵⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁶⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁷⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁸⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³²⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³³⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³¹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁴⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁵⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁶⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁷⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁸⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹²⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹³⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹¹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁴⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁵⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁶⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁷⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁸⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁴⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁴⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁵⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁶⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁷⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁸⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝ₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²²⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²³⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²¹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁴⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁵⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁶⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁷⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁸⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³²⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³³⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³¹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁴⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁵⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁶⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁷⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁸⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹²⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹³⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹¹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁴⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁵⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁶⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁷⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁸⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁴⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁵⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁶⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁷⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁸⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁹⁾₍₀₎ = Y[87, 2:end - 1]
    aj₍₋₁₎ = Y[1, 1:end - 2]
    ak₍₋₁₎ = Y[2, 1:end - 2]
    ap₍₋₁₎ = Y[3, 1:end - 2]
    arr₍₋₁₎ = Y[4, 1:end - 2]
    aw₍₋₁₎ = Y[5, 1:end - 2]
    az₍₋₁₎ = Y[6, 1:end - 2]
    b₍₋₁₎ = Y[7, 1:end - 2]
    c₍₋₁₎ = Y[9, 1:end - 2]
    c1₍₋₁₎ = Y[10, 1:end - 2]
    dp₍₋₁₎ = Y[11, 1:end - 2]
    dpᴸ⁽⁻²⁾₍₋₁₎ = Y[11, 1:end - 2]
    dpᴸ⁽⁻¹⁾₍₋₁₎ = Y[11, 1:end - 2]
    dw₍₋₁₎ = Y[14, 1:end - 2]
    dw1₍₋₁₎ = Y[15, 1:end - 2]
    h1₍₋₁₎ = Y[16, 1:end - 2]
    ik₍₋₁₎ = Y[17, 1:end - 2]
    k₍₋₁₎ = Y[18, 1:end - 2]
    q₍₋₁₎ = Y[23, 1:end - 2]
    r₍₋₁₎ = Y[25, 1:end - 2]
    w₍₋₁₎ = Y[32, 1:end - 2]
    w1₍₋₁₎ = Y[33, 1:end - 2]
    z_j₍₋₁₎ = Y[38, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²²⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²³⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²¹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁰⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁴⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁵⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁶⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁷⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁸⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³²⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³³⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³¹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁰⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁴⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁵⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁶⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁷⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁸⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹²⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹³⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹¹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁰⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁴⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁵⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁶⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁷⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁸⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁰⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁴⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁵⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁶⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁷⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁸⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²²⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²³⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²¹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁰⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁴⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁵⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁶⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁷⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁸⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³²⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³³⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³¹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁰⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁴⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁵⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁶⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁷⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁸⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹²⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹³⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹¹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁰⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁴⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁵⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁶⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁷⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁸⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁰⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁴⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁵⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁶⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁷⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁸⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁹⁾₍₋₁₎ = Y[87, 1:end - 2]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:353 =#
    RBAR = 𝓂.parameter_values[1]
    BETA = 𝓂.parameter_values[2]
    BETA1 = 𝓂.parameter_values[3]
    EC = 𝓂.parameter_values[4]
    EH = 𝓂.parameter_values[5]
    ETA = 𝓂.parameter_values[6]
    JEI = 𝓂.parameter_values[7]
    M = 𝓂.parameter_values[8]
    ALPHA = 𝓂.parameter_values[9]
    PHIK = 𝓂.parameter_values[10]
    DK = 𝓂.parameter_values[11]
    LAGP = 𝓂.parameter_values[12]
    LAGW = 𝓂.parameter_values[13]
    PIBAR = 𝓂.parameter_values[14]
    INDTR = 𝓂.parameter_values[15]
    SIGMA = 𝓂.parameter_values[16]
    TAYLOR_P = 𝓂.parameter_values[17]
    TAYLOR_Q = 𝓂.parameter_values[18]
    TAYLOR_R = 𝓂.parameter_values[19]
    TAYLOR_Y = 𝓂.parameter_values[20]
    TETAP = 𝓂.parameter_values[21]
    TETAW = 𝓂.parameter_values[22]
    XP_SS = 𝓂.parameter_values[23]
    XW_SS = 𝓂.parameter_values[24]
    RHO_J = 𝓂.parameter_values[25]
    RHO_K = 𝓂.parameter_values[26]
    RHO_P = 𝓂.parameter_values[27]
    RHO_R = 𝓂.parameter_values[28]
    RHO_W = 𝓂.parameter_values[29]
    RHO_Z = 𝓂.parameter_values[30]
    STD_J = 𝓂.parameter_values[31]
    STD_K = 𝓂.parameter_values[32]
    STD_P = 𝓂.parameter_values[33]
    STD_R = 𝓂.parameter_values[34]
    STD_W = 𝓂.parameter_values[35]
    STD_Z = 𝓂.parameter_values[36]
    RHO_J2 = 𝓂.parameter_values[37]
    RHOD = 𝓂.parameter_values[38]
    ITAYLOR_W = 𝓂.parameter_values[39]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:354 =#
    llr = 1 / BETA
    llrk = llr - (1 - DK)
    llxp = XP_SS
    llxw = XW_SS
    llxw1 = XW_SS
    lllm = (1 - BETA1 / BETA) / (1 - (BETA1 * RHOD) / PIBAR)
    QHTOC = JEI / (1 - BETA)
    QH1TOC1 = JEI / ((1 - BETA1) - lllm * M * (1 - RHOD))
    KTOY = ALPHA / (llxp * llrk)
    BTOQH1 = (M * (1 - RHOD)) / (1 - RHOD / PIBAR)
    C1TOY = (((1 - ALPHA) * SIGMA) / (1 + (1 / BETA - 1) * BTOQH1 * QH1TOC1)) * (1 / llxp)
    CTOY = (1 - C1TOY) - DK * KTOY
    lln = (((1 - SIGMA) * (1 - ALPHA)) / (llxp * llxw * CTOY)) ^ (1 / (1 + ETA))
    lln1 = ((SIGMA * (1 - ALPHA)) / (llxp * llxw1 * C1TOY)) ^ (1 / (1 + ETA))
    lly = KTOY ^ (ALPHA / (1 - ALPHA)) * lln ^ (1 - SIGMA) * lln1 ^ SIGMA
    llctot = lly - DK * KTOY * lly
    llik = KTOY * DK * lly
    llk = KTOY * lly
    llq = QHTOC * CTOY * lly + QH1TOC1 * C1TOY * lly
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:355 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:356 =#
    Χᵒᵇᶜ⁺ꜝ²ꜝ = reference_steady_state[39]
    χᵒᵇᶜ⁺ꜝ²ꜝʳ = reference_steady_state[41]
    χᵒᵇᶜ⁻ꜝ¹ꜝʳ = reference_steady_state[43]
    χᵒᵇᶜ⁻ꜝ¹ꜝˡ = reference_steady_state[44]
    χᵒᵇᶜ⁺ꜝ²ꜝˡ = reference_steady_state[42]
    Χᵒᵇᶜ⁻ꜝ¹ꜝ = reference_steady_state[40]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:358 =#
    constraint_values = Vector[]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:359 =#
    shock_sign_indicators = Bool[]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:361 =#
    begin
        push!(constraint_values, [sum(χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎ .* χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎)])
        push!(constraint_values, [sum(χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎ .* χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎)])

        push!(constraint_values, (-χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎))
        push!(constraint_values, (-χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎))
        # push!(constraint_values, (min.(χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎, χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎)))
        push!(shock_sign_indicators, false)
    end
    begin
        push!(constraint_values, (χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎))
        push!(constraint_values, (χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎))
        # push!(constraint_values, (max.(χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎, χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎)))
        push!(shock_sign_indicators, true)
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:363 =#
    # retval = [sum(χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎ .* χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎), sum(χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎ .* χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎), -χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎..., -χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎..., χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎..., χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎...]
    # return (retval, shock_sign_indicators)
    return (vcat(constraint_values...), shock_sign_indicators)
end




# using ForwardDiff

x0 = zeros(82)
# ForwardDiff.gradient(x->mcp_optim(x,[]), zeros(82))
obc_violation_function(x0, past_states, past_shocks, state_update, reference_steady_state, 𝓂, algorithm, unconditional_forecast_horizon, (present_shocks))[1] .|> abs .|> JuMP.value |> sum


    import MadNLP

    using BenchmarkTools


    model = JuMP.Model(COSMO.Optimizer)
    # model = JuMP.Model(Clarabel.Optimizer)
    model = JuMP.Model(MadNLP.Optimizer)
    model = JuMP.Model(Ipopt.Optimizer)
using Hypatia, StatusSwitchingQP, SCS, ProxSDP, HiGHS#, EAGO
using ECOS, Alpine
# @benchmark begin
        # model = JuMP.Model(MadNLP.Optimizer)
        model = JuMP.Model(Ipopt.Optimizer)
        # model = JuMP.Model(Hypatia.Optimizer) # posdef
        # model = JuMP.Model(EAGO.Optimizer) # crash
        # model = JuMP.Model(StatusSwitchingQP.Optimizer)#constraint not supported
        # model = JuMP.Model(SCS.Optimizer)#constraint not supported
        # model = JuMP.Model(ProxSDP.Optimizer)#constraint not supported
        # model = JuMP.Model(ECOS.Optimizer)#constraint not supported
        model = JuMP.Model(Alpine.Optimizer)#constraint not supported
        # model = JuMP.Model(HiGHS.Optimizer)#constraint not supported
# model = JuMP.Model(NLopt.Optimizer)

# JuMP.set_attribute(model, "algorithm", :LD_SLSQP)
# JuMP.set_attribute(model, "algorithm", :AUGLAG)
# JuMP.set_attribute(model, "local_optimizer", :LD_LBFGS) #doesnt handle quadexpr
# JuMP.set_attribute(model, "local_optimizer", :LN_NELDERMEAD) 
# JuMP.set_attribute(model, "local_optimizer", :LD_VAR2)

# JuMP.set_attribute(model, "maxiters", 3)
# JuMP.set_attribute(model, "iters", 3)
# JuMP.set_attribute(model, "algorithm", :LD_MMA)
# JuMP.set_attribute(model, "algorithm", :LN_COBYLA)

# set_attribute(model, "algorithm", :LN_COBYLA) #too long
# JuMP.set_silent(model)

# JuMP.set_attribute(model, "iter", 1e5)

# Create the variables over the full set of indices first.
JuMP.@variable(model, x[1:num_shocks*periods_per_shock])

JuMP.@objective(model, Min, x' * ℒ.I * x)

# JuMP.@constraint(model, obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, algorithm, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1] .<= 0)

JuMP.@constraint(model, obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, algorithm, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1][3:end] .<= 0)
JuMP.@constraint(model, obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, algorithm, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1][1:2] == 0)

JuMP.optimize!(model)

JuMP.termination_status(model) 
# end
# end


solved = JuMP.termination_status(model) ∈ [JuMP.OPTIMAL,JuMP.LOCALLY_SOLVED]
JuMP.value.(x)

oouutt = obc_violation_function(JuMP.value.(x), past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1] .|> JuMP.value
oouutt[oouutt .> 0] |> sum

sum(abs2,JuMP.value.(x))

𝓂.obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1] .|> JuMP.value |> sum

𝓂.obc_violation_function(JuMP.value.(x), past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1][3*periods+1:4*periods]

viols = JuMP.value.(𝓂.obc_violation_function(JuMP.value.(x), past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, JuMP.AffExpr.(present_shocks))[1])

viols[viols .> 0] |> maximum
present_states = state_update(past_states,JuMP.value.(past_shocks))
present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")] .= JuMP.value.(x)

# present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")] .= 0
# 𝓂.var


xx = JuMP.value.(x)
# xx .-=  4
# xx[4] = 40
# xx[abs.(xx) .< 1e-5].= 0



















function obc_violation_function_optim(x, past_initial_state, past_shocks, state_update, reference_steady_state, 𝓂, periods, shock_values)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:326 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:334 =#
    T = 𝓂.timings
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:336 =#
    Y = zeros(typeof(x[1]), T.nVars, periods + 2)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:338 =#
    shock_values = typeof(x[1]).(shock_values)
    shock_values[contains.(string.(T.exo), "ᵒᵇᶜ")] .= x
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:340 =#
    zero_shock = zero(shock_values)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:342 =#
    Y[:, 1] = state_update(past_initial_state, past_shocks)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:344 =#
    Y[:, 2] = state_update(Y[:, 1], shock_values)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:346 =#
    for t = 2:periods + 1
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:347 =#
        Y[:, t + 1] = state_update(Y[:, t], zero_shock)
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:348 =#
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:350 =#
    Y .+= reference_steady_state[1:T.nVars]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:352 =#
    aj₍₁₎ = Y[1, 3:end]
    ak₍₁₎ = Y[2, 3:end]
    az₍₁₎ = Y[6, 3:end]
    c₍₁₎ = Y[9, 3:end]
    c1₍₁₎ = Y[10, 3:end]
    dp₍₁₎ = Y[11, 3:end]
    dw₍₁₎ = Y[14, 3:end]
    dw1₍₁₎ = Y[15, 3:end]
    h1₍₁₎ = Y[16, 3:end]
    ik₍₁₎ = Y[17, 3:end]
    lm₍₁₎ = Y[19, 3:end]
    q₍₁₎ = Y[23, 3:end]
    qk₍₁₎ = Y[24, 3:end]
    rk₍₁₎ = Y[26, 3:end]
    uc₍₁₎ = Y[28, 3:end]
    uc1₍₁₎ = Y[29, 3:end]
    aj₍₀₎ = Y[1, 2:end - 1]
    ak₍₀₎ = Y[2, 2:end - 1]
    ap₍₀₎ = Y[3, 2:end - 1]
    arr₍₀₎ = Y[4, 2:end - 1]
    aw₍₀₎ = Y[5, 2:end - 1]
    az₍₀₎ = Y[6, 2:end - 1]
    b₍₀₎ = Y[7, 2:end - 1]
    bnot₍₀₎ = Y[8, 2:end - 1]
    c₍₀₎ = Y[9, 2:end - 1]
    c1₍₀₎ = Y[10, 2:end - 1]
    dp₍₀₎ = Y[11, 2:end - 1]
    dpᴸ⁽⁻²⁾₍₀₎ = Y[11, 2:end - 1]
    dpᴸ⁽⁻¹⁾₍₀₎ = Y[11, 2:end - 1]
    dw₍₀₎ = Y[14, 2:end - 1]
    dw1₍₀₎ = Y[15, 2:end - 1]
    h1₍₀₎ = Y[16, 2:end - 1]
    ik₍₀₎ = Y[17, 2:end - 1]
    k₍₀₎ = Y[18, 2:end - 1]
    lm₍₀₎ = Y[19, 2:end - 1]
    maxlev₍₀₎ = Y[20, 2:end - 1]
    n₍₀₎ = Y[21, 2:end - 1]
    n1₍₀₎ = Y[22, 2:end - 1]
    q₍₀₎ = Y[23, 2:end - 1]
    qk₍₀₎ = Y[24, 2:end - 1]
    r₍₀₎ = Y[25, 2:end - 1]
    rk₍₀₎ = Y[26, 2:end - 1]
    rnot₍₀₎ = Y[27, 2:end - 1]
    uc₍₀₎ = Y[28, 2:end - 1]
    uc1₍₀₎ = Y[29, 2:end - 1]
    uh₍₀₎ = Y[30, 2:end - 1]
    uh1₍₀₎ = Y[31, 2:end - 1]
    w₍₀₎ = Y[32, 2:end - 1]
    w1₍₀₎ = Y[33, 2:end - 1]
    xp₍₀₎ = Y[34, 2:end - 1]
    xw₍₀₎ = Y[35, 2:end - 1]
    xw1₍₀₎ = Y[36, 2:end - 1]
    y₍₀₎ = Y[37, 2:end - 1]
    z_j₍₀₎ = Y[38, 2:end - 1]
    Χᵒᵇᶜ⁺ꜝ²ꜝ₍₀₎ = Y[39, 2:end - 1]
    Χᵒᵇᶜ⁻ꜝ¹ꜝ₍₀₎ = Y[40, 2:end - 1]
    χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎ = Y[41, 2:end - 1]
    χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎ = Y[42, 2:end - 1]
    χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎ = Y[43, 2:end - 1]
    χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎ = Y[44, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝ₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²²⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²³⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²¹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁴⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁵⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁶⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁷⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁸⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³²⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³³⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³¹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁴⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁵⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁶⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁷⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁸⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹²⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹³⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹¹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁴⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁵⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁶⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁷⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁸⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁴⁰⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁴⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁵⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁶⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁷⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁸⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁹⁾₍₀₎ = Y[45, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝ₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²²⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²³⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²¹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁴⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁵⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁶⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁷⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁸⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³²⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³³⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³¹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁴⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁵⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁶⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁷⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁸⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹²⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹³⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹¹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁴⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁵⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁶⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁷⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁸⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁴⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁵⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁶⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁷⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁸⁾₍₀₎ = Y[87, 2:end - 1]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁹⁾₍₀₎ = Y[87, 2:end - 1]
    aj₍₋₁₎ = Y[1, 1:end - 2]
    ak₍₋₁₎ = Y[2, 1:end - 2]
    ap₍₋₁₎ = Y[3, 1:end - 2]
    arr₍₋₁₎ = Y[4, 1:end - 2]
    aw₍₋₁₎ = Y[5, 1:end - 2]
    az₍₋₁₎ = Y[6, 1:end - 2]
    b₍₋₁₎ = Y[7, 1:end - 2]
    c₍₋₁₎ = Y[9, 1:end - 2]
    c1₍₋₁₎ = Y[10, 1:end - 2]
    dp₍₋₁₎ = Y[11, 1:end - 2]
    dpᴸ⁽⁻²⁾₍₋₁₎ = Y[11, 1:end - 2]
    dpᴸ⁽⁻¹⁾₍₋₁₎ = Y[11, 1:end - 2]
    dw₍₋₁₎ = Y[14, 1:end - 2]
    dw1₍₋₁₎ = Y[15, 1:end - 2]
    h1₍₋₁₎ = Y[16, 1:end - 2]
    ik₍₋₁₎ = Y[17, 1:end - 2]
    k₍₋₁₎ = Y[18, 1:end - 2]
    q₍₋₁₎ = Y[23, 1:end - 2]
    r₍₋₁₎ = Y[25, 1:end - 2]
    w₍₋₁₎ = Y[32, 1:end - 2]
    w1₍₋₁₎ = Y[33, 1:end - 2]
    z_j₍₋₁₎ = Y[38, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²²⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²³⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²¹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁰⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁴⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁵⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁶⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁷⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁸⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻²⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³²⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³³⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³¹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁰⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁴⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁵⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁶⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁷⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁸⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻³⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹²⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹³⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹¹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁰⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁴⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁵⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁶⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁷⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁸⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁰⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁴⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁵⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁶⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁷⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁸⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁹⁾₍₋₁₎ = Y[45, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²²⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²³⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²¹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁰⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁴⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁵⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁶⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁷⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁸⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻²⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³²⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³³⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³¹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁰⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁴⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁵⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁶⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁷⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁸⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻³⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹²⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹³⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹¹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁰⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁴⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁵⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁶⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁷⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁸⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻¹⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁰⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁴⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁵⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁶⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁷⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁸⁾₍₋₁₎ = Y[87, 1:end - 2]
    ϵᵒᵇᶜ⁻ꜝ¹ꜝᴸ⁽⁻⁹⁾₍₋₁₎ = Y[87, 1:end - 2]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:353 =#
    RBAR = 𝓂.parameter_values[1]
    BETA = 𝓂.parameter_values[2]
    BETA1 = 𝓂.parameter_values[3]
    EC = 𝓂.parameter_values[4]
    EH = 𝓂.parameter_values[5]
    ETA = 𝓂.parameter_values[6]
    JEI = 𝓂.parameter_values[7]
    M = 𝓂.parameter_values[8]
    ALPHA = 𝓂.parameter_values[9]
    PHIK = 𝓂.parameter_values[10]
    DK = 𝓂.parameter_values[11]
    LAGP = 𝓂.parameter_values[12]
    LAGW = 𝓂.parameter_values[13]
    PIBAR = 𝓂.parameter_values[14]
    INDTR = 𝓂.parameter_values[15]
    SIGMA = 𝓂.parameter_values[16]
    TAYLOR_P = 𝓂.parameter_values[17]
    TAYLOR_Q = 𝓂.parameter_values[18]
    TAYLOR_R = 𝓂.parameter_values[19]
    TAYLOR_Y = 𝓂.parameter_values[20]
    TETAP = 𝓂.parameter_values[21]
    TETAW = 𝓂.parameter_values[22]
    XP_SS = 𝓂.parameter_values[23]
    XW_SS = 𝓂.parameter_values[24]
    RHO_J = 𝓂.parameter_values[25]
    RHO_K = 𝓂.parameter_values[26]
    RHO_P = 𝓂.parameter_values[27]
    RHO_R = 𝓂.parameter_values[28]
    RHO_W = 𝓂.parameter_values[29]
    RHO_Z = 𝓂.parameter_values[30]
    STD_J = 𝓂.parameter_values[31]
    STD_K = 𝓂.parameter_values[32]
    STD_P = 𝓂.parameter_values[33]
    STD_R = 𝓂.parameter_values[34]
    STD_W = 𝓂.parameter_values[35]
    STD_Z = 𝓂.parameter_values[36]
    RHO_J2 = 𝓂.parameter_values[37]
    RHOD = 𝓂.parameter_values[38]
    ITAYLOR_W = 𝓂.parameter_values[39]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:354 =#
    llr = 1 / BETA
    llrk = llr - (1 - DK)
    llxp = XP_SS
    llxw = XW_SS
    llxw1 = XW_SS
    lllm = (1 - BETA1 / BETA) / (1 - (BETA1 * RHOD) / PIBAR)
    QHTOC = JEI / (1 - BETA)
    QH1TOC1 = JEI / ((1 - BETA1) - lllm * M * (1 - RHOD))
    KTOY = ALPHA / (llxp * llrk)
    BTOQH1 = (M * (1 - RHOD)) / (1 - RHOD / PIBAR)
    C1TOY = (((1 - ALPHA) * SIGMA) / (1 + (1 / BETA - 1) * BTOQH1 * QH1TOC1)) * (1 / llxp)
    CTOY = (1 - C1TOY) - DK * KTOY
    lln = (((1 - SIGMA) * (1 - ALPHA)) / (llxp * llxw * CTOY)) ^ (1 / (1 + ETA))
    lln1 = ((SIGMA * (1 - ALPHA)) / (llxp * llxw1 * C1TOY)) ^ (1 / (1 + ETA))
    lly = KTOY ^ (ALPHA / (1 - ALPHA)) * lln ^ (1 - SIGMA) * lln1 ^ SIGMA
    llctot = lly - DK * KTOY * lly
    llik = KTOY * DK * lly
    llk = KTOY * lly
    llq = QHTOC * CTOY * lly + QH1TOC1 * C1TOY * lly
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:355 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:356 =#
    Χᵒᵇᶜ⁺ꜝ²ꜝ = reference_steady_state[39]
    χᵒᵇᶜ⁺ꜝ²ꜝʳ = reference_steady_state[41]
    χᵒᵇᶜ⁻ꜝ¹ꜝʳ = reference_steady_state[43]
    χᵒᵇᶜ⁻ꜝ¹ꜝˡ = reference_steady_state[44]
    χᵒᵇᶜ⁺ꜝ²ꜝˡ = reference_steady_state[42]
    Χᵒᵇᶜ⁻ꜝ¹ꜝ = reference_steady_state[40]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:358 =#
    constraint_values = Vector[]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:359 =#
    shock_sign_indicators = Bool[]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:361 =#
    begin
        # push!(constraint_values, [sum(χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎ .* χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎)])
        # push!(constraint_values, (-χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎))
        # push!(constraint_values, (-χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎))
        push!(constraint_values, (min.(χᵒᵇᶜ⁻ꜝ¹ꜝˡ₍₀₎, χᵒᵇᶜ⁻ꜝ¹ꜝʳ₍₀₎)))
        push!(shock_sign_indicators, false)
    end
    begin
        # push!(constraint_values, [sum(χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎ .* χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎)])
        # push!(constraint_values, (χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎))
        # push!(constraint_values, (χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎))
        push!(constraint_values, (max.(χᵒᵇᶜ⁺ꜝ²ꜝˡ₍₀₎, χᵒᵇᶜ⁺ꜝ²ꜝʳ₍₀₎)))
        push!(shock_sign_indicators, true)
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:363 =#
    return (vcat(constraint_values...), shock_sign_indicators)
end




using Optimization, OptimizationNLopt, ForwardDiff #, OptimizationOptimJL, OptimizationMOI, Ipopt


mcp_optim(x,p) = sum(abs2, x) + 1e8*sum(abs2, obc_violation_function_optim(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, (present_shocks))[1])

f = OptimizationFunction(mcp_optim, Optimization.AutoForwardDiff())

x0 = zeros(82)

prob = OptimizationProblem(f, x0, [])
soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 300)
# @profview sol = solve(prob, NLopt.LD_LBFGS())

(mcp_optim(sol,[]) - sum(abs2, sol)) / 1e8






mcp_cons([0.0],sol,[])







mcp_optim(x,p) = sum(abs2, x)
mcp_cons(res,x,p) = (res .= [sum(abs2, obc_violation_function(x, past_states, past_shocks, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, (present_shocks))[1])])

# mcp_optim(x0,[])
# mcp_cons(0.0,x0,[])

f = OptimizationFunction(mcp_optim, Optimization.AutoForwardDiff(), cons = mcp_cons)

x0 = zeros(82)

prob = OptimizationProblem(f, x0, [], lcons = [-Inf], ucons = [0.0])

sol = solve(prob, Ipopt.Optimizer())

sol = solve(prob, IPNewton())

sol = solve(prob, NLopt.LD_SLSQP())

# sol = solve(prob, LBFGS(), maxiters = 1000)
using BenchmarkTools

sol = solve(prob, LBFGS())
sol = solve(prob, NLopt.AUGLAG(), local_method = NLopt.LD_LBFGS())
mcp_optim(sol,[]) - sum(abs2,sol)

@benchmark sol = solve(prob, NLopt.LD_LBFGS())
@benchmark sol = solve(prob, NLopt.LD_SLSQP())
@benchmark sol = solve(prob, NLopt.LD_TNEWTON_PRECOND_RESTART())
mcp_optim(sol,[])
mcp_optim(x0,[])


prob = OptimizationProblem(f, sol, [])
sol = solve(prob, NLopt.LN_SBPLX(), maxiters = 100000)
sum(abs2,sol)
sol[1:41]
sol[42:end]
# @benchmark begin
