using MacroModelling

@model JQ_2012 begin
	w[0] / c[0] ^ siggma - alppha / (1 - n[0]) = 0

	c[0] ^ (-siggma) = betta * (R[0] - tau) / (1 - tau) * c[1] ^ (-siggma)

	w[0] * n[0] + b[-1] - b[0] / R[0] + d[0] - c[0] = 0

	(1 - theta) * z[0] * k[-1] ^ theta * n[0] ^ (-theta) = w[0] * 1 / (1 - mu[0] * (1 + 2 * kappa * (d[0] - d[ss])))

	betta * (c[0] / c[1]) ^ siggma * (1 + 2 * kappa * (d[0] - d[ss])) / (1 + 2 * kappa * (d[1] - d[ss])) * (1 - delta + theta * (1 - (1 + 2 * kappa * (d[1] - d[ss])) * mu[1]) * z[1] * k[0] ^ (theta - 1) * n[1] ^ (1 - theta)) + (1 + 2 * kappa * (d[0] - d[ss])) * mu[0] * xi[0] = 1

	(1 + 2 * kappa * (d[0] - d[ss])) / (1 + 2 * kappa * (d[1] - d[ss])) * (c[0] / c[1]) ^ siggma * betta * R[0] + (1 + 2 * kappa * (d[0] - d[ss])) * mu[0] * xi[0] * R[0] * (1 - tau) / (R[0] - tau) = 1

	b[0] / R[0] + k[-1] * (1 - delta) + z[0] * k[-1] ^ theta * n[0] ^ (1 - theta) - w[0] * n[0] - b[-1] - k[0] - (d[0] + kappa * (d[0] - d[ss]) ^ 2) = 0

	xi[0] * (k[0] - b[0] * (1 - tau) / (R[0] - tau)) = z[0] * k[-1] ^ theta * n[0] ^ (1 - theta)

	log(z[0] / z[ss]) = A11 * log(z[-1] / z[ss]) + A12 * log(xi[-1] / xi_bar) + eps_z[x]

	log(xi[0] / xi_bar) = log(z[-1] / z[ss]) * A21 + log(xi[-1] / xi_bar) * A22 + eps_xi[x]

	y[0] = z[0] * k[-1] ^ theta * n[0] ^ (1 - theta)

	invest[0] = k[0] - k[-1] * (1 - delta)

	v[0] = d[0] + c[0] * betta / c[1] * v[1]

	r[0] = (R[0] - tau) / (1 - tau) - 1

	# yhat[0] = log(y[0]) - log(y[ss])

	# chat[0] = log(c[0]) - log(c[ss])

	# ihat[0] = log(invest[0]) - log(invest[ss])

	# nhat[0] = log(n[0]) - log(n[ss])

	# muhat[0] = log(mu[0]) - log(mu[ss])

	# byhat[0] = (b[-1] / (1 + r[-1]) - b[0] / (1 + r[0])) / y[0]

	# dyhat[0] = d[0] / y[0]

	# vyhat[0] = log(v[0] / (k[-1] - b[-1])) - log((v[0] / (k[-1] - b[-1])))

end


@parameters JQ_2012 begin

	n[ss] = .3 | alppha

	# BY_ratio = 3.36

	b[ss] / (1 + r[ss]) / y[ss] = 3.36 | xi_bar

	# xi_bar = .163

	z[ss] = 1 | betta

	# betta = 0.9825

	siggma = 1

	theta = 0.36

	delta = 0.025

	tau = 0.35

	kappa = 0.146

	kappa_store = kappa

	sigma_xi = 0.0098

	sigma_z = 0.0045

	covariance_z_xi =0

	A11 = 0.9457

	A12 = -0.0091

	A21 = 0.0321

	A22 = 0.9703

	b > .1

	-100 < xi < 100

end

plot(JQ_2012)
get_solution(JQ_2012)


JQ_2012.SS_solve_func


using Optimization, OptimizationNLopt, NLboxsolve
# get_SS(m,verbose = true)
𝓂 = JQ_2012
params = 𝓂.parameter_values
fail_fast_solvers_only = false
verbose = true



siggma = params[1]
theta = params[2]
delta = params[3]
tau = params[4]
kappa = params[5]
A21 = params[11]
A22 = params[12]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:738 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:739 =#
kappa_store = kappa
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:740 =#
NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:741 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:742 =#
nonnegativity_auxilliary₄ = max(eps(), 1)
nonnegativity_auxilliary₃ = max(eps(), 1)
nonnegativity_auxilliary₅ = max(eps(), 1)
nonnegativity_auxilliary₇ = max(eps(), 1)
lbs = [1.1920928955078125e-7, 1.1920928955078125e-7, -100.0, -1.0e12]
ubs = [1.0e12, 1.0e12, 100.0, 1.0e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))
block_solver_RD = block_solver_AD([A21, A22], 1, 𝓂.ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
solution = block_solver_RD([A21, A22])
solution_error += sum(abs2, (𝓂.ss_solve_blocks[1])([A21, A22], solution, 0))
sol = solution
nonnegativity_auxilliary₆ = sol[1]
nonnegativity_auxilliary₈ = sol[2]
xi = sol[3]
xi_bar = sol[4]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
			sol
		else
			ℱ.value.(sol)
		end]
z = 1
nonnegativity_auxilliary₂ = max(eps(), 1)
n = 0.3
nonnegativity_auxilliary₁ = max(eps(), 1)
lbs = [-1.0e12, 0.10000000000000023, -1.0e12, 1.1920928955078125e-7, -1.0e12, -1.0e12, -1.0e12]
ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))
block_solver_RD = block_solver_AD([siggma, theta, delta, tau, kappa, xi], 2, 𝓂.ss_solve_blocks[2], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
solution = block_solver_RD([siggma, theta, delta, tau, kappa, xi])
solution_error += sum(abs2, (𝓂.ss_solve_blocks[2])([siggma, theta, delta, tau, kappa, xi], solution, 0))
sol = solution
R = sol[1]
b = sol[2]
betta = sol[3]
k = sol[4]
mu = sol[5]
r = sol[6]
y = sol[7]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
			sol
		else
			ℱ.value.(sol)
		end]
invest = delta * k
w = (3.33333333333333k) ^ theta * (((((2.0 * kappa * mu * theta - 2.0 * kappa * mu) + mu * theta) - mu) - theta) + 1.0)
d = ((-b - delta * k) - 0.3w) + b / R + (0.3 * (10.0k) ^ theta) / 3.0 ^ theta
c = (b + d + 0.3w) - b / R
alppha = (0.7w) / c ^ siggma
v = -d / (betta - 1)




var_past = @ignore_derivatives setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
var_present = @ignore_derivatives setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
var_future = @ignore_derivatives setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)

SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
# par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))
par = vcat(parameters,calibrated_parameters)

past_idx = @ignore_derivatives [indexin(sort([var_past; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_past,𝓂.exo_past))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_past =       length(past_idx) > 0 ? SS[past_idx] : zeros(0) #; zeros(length(𝓂.exo_past))...]

present_idx = @ignore_derivatives [indexin(sort([var_present; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_present,𝓂.exo_present))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_present =    length(present_idx) > 0 ? SS[present_idx] : zeros(0)#; zeros(length(𝓂.exo_present))...]

future_idx = @ignore_derivatives [indexin(sort([var_future; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_future,𝓂.exo_future))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_future =     length(future_idx) > 0 ? SS[future_idx] : zeros(0)#; zeros(length(𝓂.exo_future))...]

shocks_ss = zeros(length(𝓂.exo))

# return ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), [SS_future; SS_present; SS_past; shocks_ss])#, SS_and_pars
return Matrix(𝓂.model_jacobian([SS_future; SS_present; SS_past; shocks_ss], par, SS))




using MacroTools: postwalk

# create data containers
exo = Set()
aux = Set()
var = Set()
par = Set()
ss = Set()

dyn_ss_past = Set()
dyn_ss_future = Set()
dyn_ss_present = Set()

var_future = Set()
var_present = Set()
var_past = Set()

aux_future = Set()
aux_present = Set()
aux_past = Set()

parameters = []
parameter_values = Vector{Float64}(undef,0)

exo_list = []
ss_list = []

ss_calib_list = []
par_calib_list = []
var_list = []
dynamic_variables_list = []
dynamic_variables_future_list = []
# var_redundant_list = nothing
# var_redundant_calib_list = nothing
# var_solved_list = nothing
# var_solved_calib_list = nothing
# var_remaining_list = []
par_list = []
var_future_list = []
var_present_list = []
var_past_list = []

dyn_exo_future_list = []
dyn_exo_present_list = []
dyn_exo_past_list = []

# dyn_aux_future_list = []
# dyn_aux_present_list = []
# dyn_aux_past_list = []

exo_future = Set()
exo_present = Set()
exo_past = Set()

dyn_shift_var_future_list = []
dyn_shift_var_present_list = []
dyn_shift_var_past_list = []

dyn_shift2_var_past_list = []

dyn_var_future_list = []
dyn_var_present_list = []
dyn_var_past_list = []
dyn_exo_list = []
dyn_ss_list = []

solved_vars = [] 
solved_vals = []

non_linear_solved_vars = []
non_linear_solved_vals = []

# solved_sub_vals = []
# solved_sub_values = []
ss_solve_blocks = []
# ss_solve_blocks_no_transform = []
ss_solve_blocks_optim = []
# SS_init_guess = Vector{Float64}(undef,0)
# NSSS_solver_cache = CircularBuffer{Vector{Vector{Float64}}}(500)
SS_solve_func = nothing
nonlinear_solution_helper = nothing
SS_dependencies = nothing

ss_equations = []
equations = []
calibration_equations = []
calibration_equations_parameters = []

bounds⁺ = Set()

bounded_vars = []
lower_bounds = []
upper_bounds = []

t_future_equations = []
t_past_equations = []
t_present_equations = []
dyn_equations = []
dyn_equations_future = []


ex = :(betta * (c[0] / c[1]) ^ siggma * (1 + 2 * kappa * (d[0] - d[ss])) / (1 + 2 * kappa * (d[1] - d[ss])) * (1 - delta + theta * (1 - (1 + 2 * kappa * (d[1] - d[ss])) * mu[1]) * z[1] * k[0] ^ (theta - 1) * n[1] ^ (1 - theta)) + (1 + 2 * kappa * (d[0] - d[ss])) * mu[0] * xi[0] = 1)


# label all variables parameters and exogenous variables and timings across all equations
postwalk(x -> 
		x isa Expr ? 
			x.head == :ref ? 
				x.args[2] isa Int ? 
					x.args[2] == 0 ? 
						push!(var_present,x.args[1]) : 
					x.args[2] > 1 ? 
						begin
							time_idx = x.args[2]

							while time_idx > 1
								push!(aux_future,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(time_idx - 1)) * "⁾"))
								push!(aux_present,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(time_idx - 1)) * "⁾"))
								
								time_idx -= 1
							end

							push!(var_future,x.args[1])
						end : 
					1 >= x.args[2] > 0 ? 
						push!(var_future,x.args[1]) : 
					-1 <= x.args[2] < 0 ? 
						push!(var_past,x.args[1]) : 
					x.args[2] < -1 ? 
						begin
							time_idx = x.args[2]

							while time_idx < -1
								push!(aux_past,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(time_idx + 1)) * "⁾"))
								push!(aux_present,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(time_idx + 1)) * "⁾"))

								time_idx += 1
							end

							push!(var_past,x.args[1])
						end : 
					x :
				# issubset([x.args[2]],[:x :ex :exo :exogenous]) ?
				occursin(r"^(x|ex|exo|exogenous){1}$"i,string(x.args[2])) ?
					push!(exo,x.args[1]) :
				# issubset([x.args[2]],[:ss :SS :ℳ :StSt :steady :steadystate :steady_state :Steady_State]) ?
				occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
					push!(ss,x.args[1]) :
				x : 
			x.head == :call ? 
				for i in 2:length(x.args)
					x.args[i] isa Symbol ? 
						occursin(r"^(ss|stst|steady|steadystate|steady_state|x|ex|exo|exogenous){1}$"i,string(x.args[i])) ? 
							x :
						push!(par,x.args[i]) : 
					x
				end : 
			x :
		x,
ex)


