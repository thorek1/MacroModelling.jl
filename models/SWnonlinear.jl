# using MacroModelling

@model SWnonlinear begin
	y[0] = c[0] + inve[0] + y[ss] * gy[0] + afunc[0] * kp[-1] / (1 + ctrend / 100)

	y[0] * (pdot[0] + curvp * (1 - cfc) / cfc) / (1 + curvp * (1 - cfc) / cfc) = a[0] * k[0] ^ calfa * lab[0] ^ (1 - calfa) - (cfc - 1) * y[ss]

	k[0] = kp[-1] * zcap[0] / (1 + ctrend / 100)

	kp[0] = inve[0] * qs[0] * (1 - Sfunc[0]) + kp[-1] * (1 - ctou) / (1 + ctrend / 100)

	pdot[0] = (1 - cprobp) * (Pratio[0] / dp[0]) ^ (( - cfc) * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)) + pdot[-1] * cprobp * (dp[-1] / dp[0] * pinf[-1] ^ cindp * pinf[ss] ^ (1 - cindp) / pinf[0]) ^ (( - cfc) * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1))

	wdot[0] = (1 - cprobw) * (wnew[0] / dw[0]) ^ (( - clandaw) * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1)) + wdot[-1] * cprobw * (dw[-1] / dw[0] * pinf[-1] ^ cindw * pinf[ss] ^ (1 - cindw) / pinf[0]) ^ (( - clandaw) * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1))

	1 = (1 - cprobp) * (Pratio[0] / dp[0]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1)) + cprobp * (dp[-1] / dp[0] * pinf[-1] ^ cindp * pinf[ss] ^ (1 - cindp) / pinf[0]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1))

	1 = (1 - cprobw) * (wnew[0] / dw[0]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1)) + cprobw * (dw[-1] / dw[0] * pinf[-1] ^ cindw * pinf[ss] ^ (1 - cindw) / pinf[0]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1))

	1 = dp[0] * (1 + pdotl[0] * curvp * (1 - cfc) / cfc) / (1 + curvp * (1 - cfc) / cfc)

	w[0] = dw[0] * (1 + curvw * (1 - clandaw) / clandaw * wdotl[0]) / (1 + curvw * (1 - clandaw) / clandaw)

	pdotl[0] = (1 - cprobp) * Pratio[0] / dp[0] + cprobp * dp[-1] / dp[0] * pinf[-1] ^ cindp * pinf[ss] ^ (1 - cindp) / pinf[0] * pdotl[-1]

	wdotl[0] = (1 - cprobw) * wnew[0] / dw[0] + cprobw * dw[-1] / dw[0] * pinf[-1] ^ cindw * pinf[ss] ^ (1 - cindw) / pinf[0] * wdotl[-1]

	xi[0] = exp((csigma - 1) / (1 + csigl) * (lab[0] * (curvw * (1 - clandaw) / clandaw + wdot[0]) / (1 + curvw * (1 - clandaw) / clandaw)) ^ (1 + csigl)) * (c[0] - c[-1] * chabb / (1 + ctrend / 100)) ^ (-csigma)

	1 = qs[0] * pk[0] * (1 - Sfunc[0] - (1 + ctrend / 100) * inve[0] * SfuncD[0] / inve[-1]) + SfuncD[1] * xi[1] / xi[0] * qsaux[0] * pk[1] * ((1 + ctrend / 100) * inve[1] / inve[0]) ^ 2 * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	xi[0] = xi[1] * b[0] * r[0] * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) / pinf[1]

	rk[0] = afuncD[0]

	pk[0] = (rk[1] * zcap[1] - afunc[1] + (1 - ctou) * pk[1]) * xi[1] * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) / xi[0]

	k[0] = lab[0] * w[0] * calfa / (1 - calfa) / rk[0]

	mc[0] = w[0] ^ (1 - calfa) * rk[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))

	wnew[0] * gamw1[0] * (1 + curvw * (1 - clandaw)) / (1 + curvw * (1 - clandaw) / clandaw) = clandaw * gamw2[0] + gamw3[0] * curvw * (1 - clandaw) / clandaw * (clandaw - 1) / (1 + curvw * (1 - clandaw) / clandaw) * wnew[0] ^ (1 + clandaw * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1))

	gamw1[0] = lab[0] * dw[0] ^ (clandaw * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1)) + gamw1[1] * (pinf[ss] ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1)) * xi[1] / xi[0] * (1 + ctrend / 100) * cprobw * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	gamw2[0] = (c[0] - c[-1] * chabb / (1 + ctrend / 100)) * lab[0] * sw[0] * dw[0] ^ (clandaw * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1)) * (lab[0] * (curvw * (1 - clandaw) / clandaw + wdot[0]) / (1 + curvw * (1 - clandaw) / clandaw)) ^ csigl + gamw2[1] * (pinf[ss] ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - clandaw) * (1 + curvw * (1 - clandaw) / clandaw) / (clandaw - 1)) * xi[1] / xi[0] * (1 + ctrend / 100) * cprobw * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	gamw3[0] = lab[0] + gamw3[1] * pinf[ss] ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1] * xi[1] / xi[0] * (1 + ctrend / 100) * cprobw * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	Pratio[0] * gam1[0] * (1 + curvp * (1 - cfc)) / (1 + curvp * (1 - cfc) / cfc) = cfc * gam2[0] + gam3[0] * (cfc - 1) * curvp * (1 - cfc) / cfc / (1 + curvp * (1 - cfc) / cfc) * Pratio[0] ^ (1 + cfc * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1))

	gam1[0] = y[0] * dp[0] ^ (cfc * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)) + gam1[1] * xi[1] / xi[0] * (1 + ctrend / 100) * cprobp * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) * (pinf[ss] ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1]) ^ (( - (1 + curvp * (1 - cfc))) / (cfc - 1))

	gam2[0] = y[0] * mc[0] * spinf[0] * dp[0] ^ (cfc * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)) + gam2[1] * xi[1] / xi[0] * (1 + ctrend / 100) * cprobp * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) * (pinf[ss] ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1]) ^ (( - cfc) * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1))

	gam3[0] = y[0] + gam3[1] * pinf[ss] ^ (1 - cindp) * pinf[0] ^ cindp / pinf[1] * xi[1] / xi[0] * (1 + ctrend / 100) * cprobp * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	qsaux[0] = qs[1]

	r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / (1 + constepinf / 100)) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0]

	# r[0] = max(1.00025,r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / pinf[ss]) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])

	afunc[0] = rk[ss] * 1 / (czcap / (1 - czcap)) * (exp(czcap / (1 - czcap) * (zcap[0] - 1)) - 1)

	afuncD[0] = rk[ss] * exp(czcap / (1 - czcap) * (zcap[0] - 1))

	Sfunc[0] = csadjcost / 2 * ((1 + ctrend / 100) * inve[0] / inve[-1] - (1 + ctrend / 100)) ^ 2

	SfuncD[0] = csadjcost * ((1 + ctrend / 100) * inve[0] / inve[-1] - (1 + ctrend / 100))

	a[0] = 1 - crhoa + crhoa * a[-1] + ea[x] / 100

	b[0] = 1 - crhob + crhob * b[-1] + eb[x] * ( - (((1 - chabb / (1 + ctrend / 100)) / (csigma * (1 + chabb / (1 + ctrend / 100)))) ^ (-1))) / 100

	gy[0] - cg = crhog * (gy[-1] - cg) + egy[x] / 100 + ea[x] * cgy / 100

	qs[0] = 1 - crhoqs + crhoqs * qs[-1] + eqs[x] * csadjcost * (1 + ctrend / 100) ^ 2 * (1 + 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (1 - csigma)) / 100

	ms[0] = 1 - crhoms + crhoms * ms[-1] + ems[x] / 100

	spinf[0] = 1 - crhopinf + crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]

	epinfma[0] = epinf[x] * 1 / (1 / (1 + cindp * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) * (1 - cprobp) * (1 - cprobp * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) / cprobp / (1 + curvp * (cfc - 1))) / 100

	sw[0] = 1 - crhow + crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]

	ewma[0] = ew[x] * 1 / (1 / (1 + curvw * (clandaw - 1)) * (1 - cprobw) * (1 - cprobw * (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)) / (cprobw * (1 + (1 + ctrend / 100) * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)))) / 100

	yflex[0] = cflex[0] + inveflex[0] + gy[0] * yflex[ss] + afuncflex[0] * kpflex[-1] / (1 + ctrend / 100)

	yflex[0] = a[0] * kflex[0] ^ calfa * labflex[0] ^ (1 - calfa) - (cfc - 1) * yflex[ss]

	kflex[0] = kpflex[-1] * zcapflex[0] / (1 + ctrend / 100)

	kpflex[0] = inveflex[0] * qs[0] * (1 - Sfuncflex[0]) + kpflex[-1] * (1 - ctou) / (1 + ctrend / 100)

	xiflex[0] = exp((csigma - 1) / (1 + csigl) * labflex[0] ^ (1 + csigl)) * (cflex[0] - cflex[-1] * chabb / (1 + ctrend / 100)) ^ (-csigma)

	1 = qs[0] * pkflex[0] * (1 - Sfuncflex[0] - (1 + ctrend / 100) * inveflex[0] * SfuncDflex[0] / inveflex[-1]) + SfuncDflex[1] * qsaux[0] * xiflex[1] / xiflex[0] * pkflex[1] * ((1 + ctrend / 100) * inveflex[1] / inveflex[0]) ^ 2 * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	xiflex[0] = xiflex[1] * b[0] * rrflex[0] * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma)

	rkflex[0] = afuncDflex[0]

	pkflex[0] = (rkflex[1] * zcapflex[1] - afuncflex[1] + (1 - ctou) * pkflex[1]) * xiflex[1] * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) / xiflex[0]

	kflex[0] = labflex[0] * calfa / (1 - calfa) * wflex[0] / rkflex[0]

	mcflex = wflex[0] ^ (1 - calfa) * rkflex[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))

	wflex[0] * (1 + curvw * (1 - clandaw)) / (1 + curvw * (1 - clandaw) / clandaw) = sw[ss] * (labflex[0] ^ csigl * clandaw * (cflex[0] - cflex[-1] * chabb / (1 + ctrend / 100)) + wflex[0] * curvw * (1 - clandaw) / clandaw * (clandaw - 1) / (1 + curvw * (1 - clandaw) / clandaw))

	# (1 + curvp * (1 - cfc)) / (1 + curvp * (1 - cfc) / cfc) = spinf[ss] * cfc * mcflex[0] + spinf[ss] * (cfc - 1) * curvp * (1 - cfc) / cfc / (1 + curvp * (1 - cfc) / cfc)

	afuncflex[0] = rkflex[ss] * 1 / (czcap / (1 - czcap)) * (exp(czcap / (1 - czcap) * (zcapflex[0] - 1)) - 1)

	afuncDflex[0] = rkflex[ss] * exp(czcap / (1 - czcap) * (zcapflex[0] - 1))

	Sfuncflex[0] = csadjcost / 2 * ((1 + ctrend / 100) * inveflex[0] / inveflex[-1] - (1 + ctrend / 100)) ^ 2

	SfuncDflex[0] = csadjcost * ((1 + ctrend / 100) * inveflex[0] / inveflex[-1] - (1 + ctrend / 100))

	# ygap[0] = 100 * log(y[0] / yflex[0])

	# dy[0] = ctrend + 100 * (y[0] / y[-1] - 1)

	# dc[0] = ctrend + 100 * (c[0] / c[-1] - 1)

	# dinve[0] = ctrend + 100 * (inve[0] / inve[-1] - 1)

	# pinfobs[0] = 100 * (pinf[0] - pinf[ss]) + constepinf

	# robs[0] = 100 * (r[0] - 1)

	# dwobs[0] = ctrend + 100 * (w[0] / w[-1] - 1)

	# labobs[0] = 100 * (lab[0] / lab[ss] - 1)

end


@parameters SWnonlinear begin
	# mcflex = mc[ss] | mcflex
	mcflex = 1 / cfc

	ctou = 0.025

	cg = 0.18

	clandaw = 1.5

	curvw = 10

	crhoa = 0.95827

	crhob = 0.22137

	crhog = 0.97391

	crhoqs = 0.70524

	crhoms = 0.11421

	crhopinf = 0.83954

	crhow = 0.9745

	cmap = 0.69414

	cmaw = 0.93617

	csadjcost = 5.5811

	csigma = 1.4103

	chabb = 0.68049

	cprobw = 0.80501

	csigl = 2.2061

	cindw = 0.56351

	cindp = 0.24165

	czcap = 0.49552

	cfc = 1.3443

	crpi = 1.931

	crr = 0.82512

	cry = 0.097844

	crdy = 0.25114

	constepinf = 0.8731

	constebeta = 0.12575

	ctrend = 0.4419

	cgy = 0.53817

	calfa = 0.18003

	curvp = 64.5595

	cprobp = 0.667

	# gamw1 > .1 # 6
	# gamw2 > .1 # 6
	# gamw3 > .1 # 6

	# gam1 >  3.5 # this is the only one that is necessary
	# gam2 >  2
	# gam3 > .1 # 4

	# k > .1 # 7
	# kp > .1
	# mc > .1
	# c > .1 # 7
	# w > .1 # 7
	# y > .1 # 7
	# inve > .1 # 7
	# lab > .1 # 7
end

# SS(SWnonlinear)

# SWnonlinear.SS_solve_func






# c          		 =	0.896367
# y          		 =	1.36422
# lab        		 =	1.34391
# inve       		 =	0.222297
# kp         		 =	7.58964
# k          		 =	7.55624
# zcap       		 =	1
# Pratio     		 =	1
# wnew       		 =	0.832362
# pinf       		 =	1.00873
# pdot       		 =	1
# wdot       		 =	1
# dp         		 =	1
# dw         		 =	0.832362
# pdotl      		 =	1
# wdotl      		 =	1
# rk         		 =	0.0325031
# w          		 =	0.832362
# mc         		 =	0.743882
# xi         		 =	8.00755
# r          		 =	1.0163
# pk         		 =	1
# gam1       		 =	4.07181
# gam2       		 =	3.02894
# gam3       		 =	4.07181
# gamw1      		 =	24.5877
# gamw2      		 =	13.6439
# gamw3      		 =	6.8062
# qsaux      		 =	1
# afunc      		 =	0
# afuncD     		 =	0.0325031
# Sfunc      		 =	0
# SfuncD     		 =	0
# a          		 =	1
# b          		 =	1
# gy         		 =	0.18
# qs         		 =	1
# ms         		 =	1
# spinf      		 =	1
# epinfma    		 =	0
# sw         		 =	1
# ewma       		 =	0
# cflex      		 =	0.896367
# yflex      		 =	1.36422
# labflex    		 =	1.34391
# inveflex   		 =	0.222297
# kpflex     		 =	7.58964
# kflex      		 =	7.55624
# zcapflex   		 =	1
# rkflex     		 =	0.0325031
# wflex      		 =	0.832362
# mcflex     		 =	0.743882
# xiflex     		 =	8.00755
# rrflex     		 =	1.0075
# pkflex     		 =	1
# afuncflex  		 =	0
# afuncDflex 		 =	0.0325031
# Sfuncflex  		 =	0
# SfuncDflex 		 =	0
# dy         		 =	0.4419
# dc         		 =	0.4419
# dinve      		 =	0.4419
# pinfobs    		 =	0.8731
# robs       		 =	1.62996
# dwobs      		 =	0.4419
# labobs     		 =	0
# ygap       		 =	0




# 𝓂 = SWnonlinear
# parameters = 𝓂.parameter_values
# import MacroModelling: block_solver, ℐ, 𝒷
# cold_start = true
# verbose = true
# solver_parameters = 𝓂.solver_parameters


# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3346 =#
# closest_solution = 𝓂.NSSS_solver_cache[end]
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3347 =#


# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3357 =#
# ctou = parameters[1]
# cg = parameters[2]
# clandaw = parameters[3]
# curvw = parameters[4]
# csigma = parameters[15]
# chabb = parameters[16]
# cprobw = parameters[17]
# csigl = parameters[18]
# czcap = parameters[21]
# cfc = parameters[22]
# crpi = parameters[23]
# cry = parameters[25]
# constepinf = parameters[27]
# constebeta = parameters[28]
# ctrend = parameters[29]
# calfa = parameters[31]
# curvp = parameters[32]
# cprobp = parameters[33]
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3358 =#
# calfa = min(max(calfa, 2.220446049250313e-16), 1.0e12)
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3359 =#
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3360 =#
# NSSS_solver_cache_tmp = []
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3361 =#
# solution_error = 0.0
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3362 =#
# iters = 0
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3363 =#
# qs = 1
# ➕₂ = max(eps(), 1)
# ➕₁ = max(eps(), 1)
# params_and_solved_vars = [cfc, curvp, cprobp, ➕₁]
# lbs = [2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16]
# ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
# inits = [max.(lbs[1:length(closest_solution[1])], min.(ubs[1:length(closest_solution[1])], closest_solution[1])), closest_solution[2]]
# block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[1]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())
# solution = block_solver_AD(params_and_solved_vars, 1, 𝓂.ss_solve_blocks[1], inits, lbs, ubs, solver_parameters, cold_start, verbose)
# iters += (solution[2])[2]
# solution_error += (solution[2])[1]
# sol = solution[1]



# Pratio = sol[1]
# dp = sol[2]
# pdotl = sol[3]


# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
# 		sol
# 	else
# 		ℱ.value.(sol)
# 	end]
# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
# 		params_and_solved_vars
# 	else
# 		ℱ.value.(params_and_solved_vars)
# 	end]
# ➕₁₁ = max(eps(), 1)
# ➕₄ = max(eps(), 1)
# ➕₃ = max(eps(), 1)
# ➕₈ = min(max(2.220446049250313e-16, ctrend / 100 + 1), 1.0e12)
# solution_error += abs(➕₈ - (ctrend / 100 + 1))
# ewma = 0
# sw = 1
# ➕₁₀ = max(eps(), 1)
# wdot = 1
# qsaux = 1
# SfuncD = 0
# Sfunc = 0
# pk = 1
# params_and_solved_vars = [ctou, csigma, czcap, constebeta, ➕₈]
# lbs = [-1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16]
# ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 700.0, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
# inits = [max.(lbs[1:length(closest_solution[3])], min.(ubs[1:length(closest_solution[3])], closest_solution[3])), closest_solution[4]]
# block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[2]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())
# solution = block_solver_AD(params_and_solved_vars, 2, 𝓂.ss_solve_blocks[2], inits, lbs, ubs, solver_parameters, cold_start, verbose)
# iters += (solution[2])[2]
# solution_error += (solution[2])[1]
# sol = solution[1]



# afunc = sol[1]
# afuncD = sol[2]
# rk = sol[3]
# zcap = sol[4]
# ➕₁₄ = sol[5]

# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
# 		sol
# 	else
# 		ℱ.value.(sol)
# 	end]
# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
# 		params_and_solved_vars
# 	else
# 		ℱ.value.(params_and_solved_vars)
# 	end]
# pdot = 1
# a = 1
# gy = cg
# ➕₉ = min(max(2.220446049250313e-16, 1 - calfa), 1.0e12)
# solution_error += abs(➕₉ - (1 - calfa))
# epinfma = 0
# spinf = 1
# ➕₁₈ = min(1.0e12, max(eps(), Pratio))
# ➕₁₉ = min(1.0e12, max(eps(), rk))
# ➕₂₀ = min(1.0e12, max(eps(), calfa))
# ➕₂₁ = min(1.0e12, max(eps(), dp))
# params_and_solved_vars = [ctou, clandaw, curvw, csigma, chabb, cprobw, csigl, cfc, constebeta, ctrend, calfa, curvp, cprobp, Pratio, afunc, dp, gy, rk, zcap, ➕₃, ➕₈, ➕₉, ➕₁₈, ➕₁₉, ➕₂₀, ➕₂₁]
# lbs = [-1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16]

# lbs[1:17] .=  .2
# lbs[3:5] .=  4
# lbs[6:8] .=  6
# lbs[10:11] .=  7

# ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]

# ubs[1:17] .=  25
# inits = [max.(lbs[1:length(closest_solution[5])], min.(ubs[1:length(closest_solution[5])], closest_solution[5].^0 .- 1 .+ .7688)), closest_solution[6]]
# block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[3]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())
# solution = block_solver_AD(params_and_solved_vars, 3, 𝓂.ss_solve_blocks[3], inits, lbs, ubs, solver_parameters, cold_start, verbose)
# iters += (solution[2])[2]
# solution_error += (solution[2])[1]
# solution_error += abs(➕₁₈ - Pratio) + abs(➕₁₉ - rk) + abs(➕₂₀ - calfa) + abs(➕₂₁ - dp)
# sol = solution[1]
# ➕₅ = lab
# sol_found = [c, dw, gam1, gam2, gam3, gamw1, gamw2, gamw3, inve, k, kp, lab, mc, w, wdotl, wnew, y, ➕₅]

# 𝓂.ss_solve_blocks[3](params_and_solved_vars, sol_found)



# k - (kp * zcap) / (ctrend / 100 + 1)
# 1 - wnew / dw
# ((-cprobp * gam3 * (ctrend / 100 + 1)) / (➕₈ ^ csigma * (constebeta / 100 + 1)) + gam3) - y
# (-clandaw * gamw2 + (gamw1 * wnew * (curvw * (1 - clandaw) + 1)) / (1 + (curvw * (1 - clandaw)) / clandaw)) - (curvw * gamw3 * wnew ^ ((clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1) + 1) * (1 - clandaw) * (clandaw - 1)) / (clandaw * (1 + (curvw * (1 - clandaw)) / clandaw))
# -lab + ➕₅
# (-dw * (1 + (curvw * wdotl * (1 - clandaw)) / clandaw)) / (1 + (curvw * (1 - clandaw)) / clandaw) + w
# ((-cprobw * gamw1 * (ctrend / 100 + 1)) / (➕₈ ^ csigma * (constebeta / 100 + 1)) - dw ^ ((clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1)) * lab) + gamw1
# -(k ^ calfa) * lab ^ (1 - calfa) + y * (cfc - 1) + y
# ((Pratio * gam1 * (curvp * (1 - cfc) + 1)) / (1 + (curvp * (1 - cfc)) / cfc) - (➕₁₈ ^ ((cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1) + 1) * curvp * gam3 * (1 - cfc) * (cfc - 1)) / (cfc * (1 + (curvp * (1 - cfc)) / cfc))) - cfc * gam2


# ((-cprobp * gam2 * (ctrend / 100 + 1)) / (➕₈ ^ csigma * (constebeta / 100 + 1)) - ➕₁₉ ^ ((cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1)) * mc * y) + gam2


# gam2 - (y * mc * spinf * dp ^ (cfc * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)) + gam2 * xi / xi * (1 + ctrend / 100) * cprobp * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma) * (pinf ^ (1 - cindp) * pinf ^ cindp / pinf) ^ (( - cfc) * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)))

# gam2 - (y * mc * spinf * dp ^ (cfc * (1 + curvp * (1 - cfc) / cfc) / (cfc - 1)) + gam2 * xi / xi * (1 + ctrend / 100) * cprobp * 1 / (1 + constebeta / 100) * (1 + ctrend / 100) ^ (-csigma))



# (-inve - (kp * (1 - ctou)) / (ctrend / 100 + 1)) + kp
# ((-cprobp * gam1 * (ctrend / 100 + 1)) / (➕₈ ^ csigma * (constebeta / 100 + 1)) - ➕₁₉ ^ ((cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1)) * y) + gam1
# (-calfa * lab * w) / (rk * (1 - calfa)) + k
# mc - (➕₂₀ ^ calfa * w ^ (1 - calfa) * ➕₉ ^ (calfa - 1)) / ➕₂₁ ^ calfa
# ((-cprobw * gamw2 * (ctrend / 100 + 1)) / (➕₈ ^ csigma * (constebeta / 100 + 1)) - dw ^ ((clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1)) * lab * ➕₅ ^ csigl * ((-c * chabb) / (ctrend / 100 + 1) + c)) + gamw2
# ((((-afunc * kp) / (ctrend / 100 + 1) - c) - gy * y) - inve) + y
# ((-cprobw * gamw3 * (ctrend / 100 + 1)) / (➕₈ ^ csigma * (constebeta / 100 + 1)) + gamw3) - lab
# (-cprobw * wdotl + wdotl) - (wnew * (1 - cprobw)) / dw



# c = sol[1]
# dw = sol[2]
# gam1 = sol[3]
# gam2 = sol[4]
# gam3 = sol[5]
# gamw1 = sol[6]
# gamw2 = sol[7]
# gamw3 = sol[8]
# inve = sol[9]
# k = sol[10]
# kp = sol[11]
# lab = sol[12]
# mc = sol[13]
# w = sol[14]
# wdotl = sol[15]
# wnew = sol[16]
# y = sol[17]
# ➕₅ = sol[18]
# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
# 		sol
# 	else
# 		ℱ.value.(sol)
# 	end]
# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
# 		params_and_solved_vars
# 	else
# 		ℱ.value.(params_and_solved_vars)
# 	end]
# Sfuncflex = 0
# SfuncDflex = 0
# pkflex = 1
# params_and_solved_vars = [ctou, csigma, czcap, constebeta, ➕₈]
# lbs = [-1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16]
# ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 700.0, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
# inits = [max.(lbs[1:length(closest_solution[7])], min.(ubs[1:length(closest_solution[7])], closest_solution[7])), closest_solution[8]]
# block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[4]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())
# solution = block_solver_AD(params_and_solved_vars, 4, 𝓂.ss_solve_blocks[4], inits, lbs, ubs, solver_parameters, cold_start, verbose)
# iters += (solution[2])[2]
# solution_error += (solution[2])[1]
# sol = solution[1]
# afuncDflex = sol[1]
# afuncflex = sol[2]
# rkflex = sol[3]
# zcapflex = sol[4]
# ➕₁₇ = sol[5]
# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
# 		sol
# 	else
# 		ℱ.value.(sol)
# 	end]
# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
# 		params_and_solved_vars
# 	else
# 		ℱ.value.(params_and_solved_vars)
# 	end]
# mcflex = mc
# ➕₂₂ = min(1.0e12, max(eps(), rkflex))
# ➕₂₃ = min(1.0e12, max(eps(), calfa))
# ➕₂₄ = min(1.0e12, max(eps(), ➕₂₃ ^ calfa * mcflex * ➕₉ ^ (1 - calfa)))
# solution_error += abs(➕₂₂ - rkflex) + abs(➕₂₃ - calfa) + abs(➕₂₄ - ➕₂₃ ^ calfa * mcflex * ➕₉ ^ (1 - calfa))
# wflex = ➕₂₂ ^ (calfa / (calfa - 1)) / ➕₂₄ ^ (1 / (calfa - 1))
# params_and_solved_vars = [ctou, clandaw, curvw, chabb, csigl, cfc, ctrend, calfa, afuncflex, gy, rkflex, wflex, zcapflex]
# lbs = [-1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12]
# ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
# inits = [max.(lbs[1:length(closest_solution[9])], min.(ubs[1:length(closest_solution[9])], closest_solution[9])), closest_solution[10]]
# block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[5]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())
# solution = block_solver_AD(params_and_solved_vars, 5, 𝓂.ss_solve_blocks[5], inits, lbs, ubs, solver_parameters, cold_start, verbose)
# iters += (solution[2])[2]
# solution_error += (solution[2])[1]
# sol = solution[1]
# cflex = sol[1]
# inveflex = sol[2]
# kflex = sol[3]
# kpflex = sol[4]
# labflex = sol[5]
# yflex = sol[6]
# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
# 		sol
# 	else
# 		ℱ.value.(sol)
# 	end]
# NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
# 		params_and_solved_vars
# 	else
# 		ℱ.value.(params_and_solved_vars)
# 	end]
# ➕₁₃ = min(max(2.220446049250313e-16, y / yflex), 1.0e12)
# solution_error += abs(➕₁₃ - y / yflex)
# ➕₁₆ = min(max(2.220446049250313e-16, (cflex * (-100chabb + ctrend + 100)) / (ctrend + 100)), 1.0e12)
# solution_error += abs(➕₁₆ - (cflex * (-100chabb + ctrend + 100)) / (ctrend + 100))
# ➕₆ = min(max(-1.0e12, (➕₅ ^ (csigl + 1) * (csigma - 1)) / (csigl + 1)), 700.0)
# solution_error += abs(➕₆ - (➕₅ ^ (csigl + 1) * (csigma - 1)) / (csigl + 1))
# ms = 1
# ➕₁₂ = min(max(2.220446049250313e-16, ➕₁₃ ^ (-cry / crpi)), 1.0e12)
# solution_error += abs(➕₁₂ - ➕₁₃ ^ (-cry / crpi))
# pinf = (constepinf * ➕₁₂) / 100
# b = 1
# r = (pinf * ➕₈ ^ csigma * (constebeta + 100)) / 100
# ➕₇ = min(max(2.220446049250313e-16, (c * (-100chabb + ctrend + 100)) / (ctrend + 100)), 1.0e12)
# solution_error += abs(➕₇ - (c * (-100chabb + ctrend + 100)) / (ctrend + 100))
# xi = exp(➕₆) / ➕₇ ^ csigma
# ➕₁₅ = min(max(-1.0e12, (labflex ^ (csigl + 1) * (csigma - 1)) / (csigl + 1)), 700.0)
# solution_error += abs(➕₁₅ - (labflex ^ (csigl + 1) * (csigma - 1)) / (csigl + 1))
# rrflex = (➕₈ ^ csigma * (constebeta + 100)) / 100
# xiflex = exp(➕₁₅) / ➕₁₆ ^ csigma




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %Declare steady state for all variables here. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# gy_ss    = log(cg);
# spinf_ss = log(1);%-log(clandap);
# a        = 1;
# b        = 1;
# ms       = 1;
# sw       = 1;
# epinfma  = 0;
# ewma     = 0;
# qs       = 1;

# % steady state params
# cgamma=1+ctrend/100 ;
# cbeta=1/(1+constebeta/100);
# clandap=cfc;

# %Steady state for all endogenous variables in logs

# rk_ss    = log(cbeta^(-1)*cgamma^csigma - (1-ctou));
# mc_ss    = - (log(clandap) + spinf_ss);
# w_ss     = 1/(1-calfa)*(mc_ss + calfa*log(calfa)+(1-calfa)*log(1-calfa) - calfa*rk_ss);
# ikp_ss   = log(1 - (1-ctou)/cgamma);
# ik_ss    = log(cgamma) + ikp_ss;
# klab_ss  = log(calfa/(1-calfa)) + w_ss - rk_ss;
# ky_ss    = (1-calfa)*klab_ss + log(cfc);
# cy_ss    = log(1 - exp(ik_ss)*exp(ky_ss) - exp(gy_ss));
# lab_ss   = 1/(1+csigl)*(w_ss - log(clandaw) + ky_ss - klab_ss - cy_ss - log(1-chabb/cgamma));
# k_ss     = klab_ss + lab_ss;
# y_ss     = k_ss - ky_ss;
# c_ss     = cy_ss +y_ss;
# inve_ss  = ik_ss + k_ss;
# kp_ss    = inve_ss - ikp_ss;
# r_ss     = log((1+constepinf/100)/(cbeta*cgamma^(-csigma)));
# zcap_ss  = 0;
# pk_ss    = 0;
# xi_ss    = -csigma*(c_ss + log(1-chabb/cgamma))+(csigma-1)/(1+csigl)*exp(lab_ss)^(1+csigl);
# gamw1_ss = lab_ss + (clandaw/(clandaw-1)-curvw) * w_ss - log(1-cbeta*cgamma^(1 - csigma)*cprobw);
# gamw2_ss = gamw1_ss + log(1-chabb/cgamma) + c_ss + csigl*lab_ss;
# gamw3_ss = lab_ss - log(1-cbeta*cgamma^(1 - csigma)*cprobw);
# gam1_ss  = y_ss - log(1-cbeta*cgamma^(1 - csigma)*cprobp);
# gam2_ss  = gam1_ss + mc_ss;
# gam3_ss  = gam1_ss;
# pinf_ss  = log(1+constepinf/100);

# qsaux    = qs;
# afunc    = 0;
# afuncD   = exp(rk_ss);
# Sfunc    = 0;
# SfuncD   = 0;




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %converting quantity variables from logs to levels.
# %converting interest rates and inflation rates from net to gross rates.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# c      = exp(c_ss);
# y      = exp(y_ss);
# lab    = exp(lab_ss);
# inve   = exp(inve_ss);
# kp     = exp(kp_ss);
# k      = kp/cgamma; %capital services
# zcap   = exp(zcap_ss);
# Pratio = 1;
# wnew   = exp(w_ss);
# pinf = exp(pinf_ss);
# pdot   = 1;
# wdot   = 1;
# dp     = 1;
# dw     = wnew;
# pdotl  = 1;
# wdotl  = 1;
# rk     = exp(rk_ss);
# w      = exp(w_ss);
# mc     = exp(mc_ss);
# xi     = exp(xi_ss);
# r    = exp(r_ss);
# pk = exp(pk_ss);
# gam1 = exp(gam1_ss);
# gam2 = exp(gam2_ss);
# gam3 = exp(gam3_ss);
# gamw1 = exp(gamw1_ss);
# gamw2 = exp(gamw2_ss);
# gamw3 = exp(gamw3_ss);

# gy = exp(gy_ss);
# spinf = exp(spinf_ss);


# %Flex price steady variables same as sticky variables with full indexation

# cflex      = c;
# yflex      = y;
# labflex    = lab;
# inveflex   = inve;
# kpflex     = kp;
# kflex      = k;
# zcapflex   = zcap;
# rkflex     = rk;
# wflex      = w;
# mcflex     = mc;
# xiflex     = xi;
# rrflex     = r/pinf;
# pkflex     = pk;
# afuncflex  = afunc;
# afuncDflex = afuncD; 
# Sfuncflex  = Sfunc;
# SfuncDflex = SfuncD;

# % Observation equations
# dy = ctrend;
# % pinfobs = 100*log(pinf);
# pinfobs = constepinf;
# dc = ctrend;
# dinve = ctrend;
# labobs = 0;
# dwobs = ctrend;
# robs = 100*(r-1);
# % robs = r-1;

# ygap = 0;
# rnot = r;
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# % Doing some steady state checks
# test1 = y - (c+inve+gy*y+afunc*kp/cgamma);
# test2 = y - (a*((k)^calfa)*(lab^(1-calfa)))/cfc;
# test3 = y - k*((a*( ( (calfa/(1-calfa))*w/rk )^(calfa-1)) )/cfc);