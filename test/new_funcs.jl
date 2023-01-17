using MacroModelling
import ForwardDiff as ℱ
using StatsFuns, SpecialFunctions

include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")

get_SS(m)

include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")

get_solution(m)

parameters = m.parameter_values

m.SS_solve_func(parameters, m, false, false)[1]

using Zygote
Zygote.jacobian(x -> m.SS_solve_func(x, m, false, true)[1],Float64[parameters...])[1]

using ForwardDiff
ForwardDiff.jacobian(x -> m.SS_solve_func(x, m, false, false)[1],Float64[parameters...])


using BenchmarkTools
@benchmark Zygote.jacobian(x -> m.SS_solve_func(x, m, false, false)[1],Float64[parameters...])[1]

@benchmark get_SS(m)

m.SS_solve_func






@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end


@parameters RBC_CME begin
    # alpha | k[ss] / (4 * y[ss]) = cap_share
    # cap_share = 1.66
    alpha = .157

    # beta | R[ss] = R_ss
    # R_ss = 1.0035
    beta = .999

    # delta | c[ss]/y[ss] = 1 - I_K_ratio
    # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
    # I_K_ratio = .15
    delta = .0226

    # Pibar | Pi[ss] = Pi_ss
    # Pi_ss = 1.0025
    Pibar = 1.0008

    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

solve!(RBC_CME, dynamics = true)



data = get_irf(RBC_CME, levels = true)[:,:,1]
data = simulate(RBC_CME, levels = true)[:,:,1]
observables = [:c,:k]

calculate_kalman_filter_loglikelihood(RBC_CME,data(observables),observables)

parameters = RBC_CME.parameter_values

using Zygote
Zygote.gradient(x->calculate_kalman_filter_loglikelihood(RBC_CME,data(observables),observables,parameters = x),parameters)

@test isapprox(425.7688745392835,calculate_kalman_filter_loglikelihood(RBC_CME,data(observables),observables),rtol = 1e-5)


using Optimization, Krylov

block_solver_AD(parameters_and_solved_vars::Vector{Float64}, 
    n_block::Int, 
    ss_solve_blocks::Function, 
    f::OptimizationFunction, 
    guess::Vector{Float64}, 
    lbs::Vector{Float64}, 
    ubs::Vector{Float64};
    tol = eps(Float64),
    maxtime = 120,
    starting_points = [.9, 1, 1.1, .75, 1.5, -.5, 2, .25],
    fail_fast_solvers_only = true,
    verbose = false) = ImplicitFunction(parameters_and_solved_vars -> block_solver(parameters_and_solved_vars,
                                                            n_block, 
                                                            ss_solve_blocks,
                                                            f,
                                                            guess,
                                                            lbs,
                                                            ubs;
                                                            tol = tol,
                                                            maxtime = maxtime,
                                                            starting_points = starting_points,
                                                            fail_fast_solvers_only = fail_fast_solvers_only,
                                                            verbose = verbose)[1],  
                                        ss_solve_blocks)#, Krylov.bicgstab)

params = m.parameter_values
𝓂 = m

cap_share = params[1]
R_ss = params[2]
I_K_ratio = params[3]
phi_pi = params[4]
Pi_real = params[7]
rhoz = params[8]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:707 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:708 =#
Pi_ss = R_ss - Pi_real
rho_z_delta = rhoz
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:709 =#
# NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:710 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:711 =#
nonnegativity_auxilliary₂ = max(eps(), 1)
log_ZZ_avg = 0
A = 1
ZZ_avg = 1
R = R_ss
Pi = Pi_ss
beta = Pi / R
nonnegativity_auxilliary₁ = max(eps(), (R * beta) ^ (1 / phi_pi))
z_delta = 1
lbs = [2.220446049250313e-16, -1.0e12, -1.0e12, 1.1920928955078125e-7, -1.0e12]
ubs = [0.9999999999999998, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
f = OptimizationFunction(𝓂.ss_solve_blocks_optim[1], Optimization.AutoForwardDiff())
inits = max.(lbs, min.(ubs, fill(.5,length(ubs))))


function undo_transformer(x)
    # return sinh.(sinh.(sinh.(x)))
    return sinh(sinh(x))
    # return sinh(x)
    # return x
end


function ss_solve_blocks(parameters_and_solved_vars,guess)
    guess = undo_transformer.(guess) 
    alpha = (guess[1])
    c = (guess[2])
    delta = (guess[3])
    k = (guess[4])
    y = (guess[5])
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:511 =#
    cap_share = parameters_and_solved_vars[1]
    I_K_ratio = parameters_and_solved_vars[2]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:512 =#
    beta = parameters_and_solved_vars[3]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:515 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:516 =#
    return [-beta * ((alpha * k ^ (alpha - 1) - delta) + 1) + 1, -(k ^ alpha) + y, ((-c + k * (1 - delta)) - k) + k ^ alpha, (I_K_ratio + c / y) - 1, -cap_share + k / (4y)]
end

using ImplicitDifferentiation
bl = block_solver_AD([cap_share, I_K_ratio, beta], 1, ss_solve_blocks, f, inits, lbs, ubs, fail_fast_solvers_only = true, verbose = true);

block_solver([cap_share, I_K_ratio, beta], 1, ss_solve_blocks, f, inits, lbs, ubs, fail_fast_solvers_only = true, verbose = true)

bl([cap_share, I_K_ratio, beta])

using Zygote, ForwardDiff, FiniteDifferences

Zygote.jacobian(bl, [cap_share, I_K_ratio, beta])[1]

ForwardDiff.jacobian(bl, [cap_share, I_K_ratio, beta])
ForwardDiff.jacobian(x->block_solver(x, 1, ss_solve_blocks, f, inits, lbs, ubs, fail_fast_solvers_only = true, verbose = true)[1], [cap_share, I_K_ratio, beta])

FiniteDifferences.jacobian(central_fdm(3,1),bl, [cap_share, I_K_ratio, beta])[1]


function SS_solve_func(parameters, 𝓂, fail_fast_solvers_only, verbose)
    params_flt = typeof(parameters) == Vector{Float64} ? parameters : ℱ.value.(parameters)
    closest_solution_init = 𝓂.NSSS_solver_cache[findmin([sum(abs2,pars[end] - params_flt) for pars in 𝓂.NSSS_solver_cache])[2]]
    solved_scale = 0
    range_length = fail_fast_solvers_only ? [1] : [2^i for i in 0:5]
    for r in range_length
        for scale in range(0,1,r+1)[2:end]
            if scale <= solved_scale continue end
            closest_solution = 𝓂.NSSS_solver_cache[findmin([sum(abs2,pars[end] - params_flt) for pars in 𝓂.NSSS_solver_cache])[2]]
            params = all(isfinite.(closest_solution_init[end])) && parameters != closest_solution_init[end] ? scale * parameters + (1 - scale) * closest_solution_init[end] : parameters
            params_scaled_flt = typeof(params) == Vector{Float64} ? params : ℱ.value.(params)

            alpha = params[1]
            R_ss = params[2]
            delta = params[3]
            phi_pi = params[4]
            Pi_real = params[7]
            rhoz = params[8]
            
            Pi_ss = R_ss - Pi_real
            rho_z_delta = rhoz

            NSSS_solver_cache_tmp = []
            solution_error = 0.0
            
            Pi = Pi_ss
            z_delta = 1
            R = R_ss
            beta = Pi / R
            nonnegativity_auxilliary₁ = max(eps(), (R * beta) ^ (1 / phi_pi))
            A = 1
            ZZ_avg = 1
            ZZ_avg_fut = 1
            k = ((beta * (delta - 1) + 1) / (alpha * beta)) ^ (1 / (alpha - 1))
            c = -delta * k + k ^ alpha
            c_normlogpdf = -0.5 * c ^ 2 - 0.918938533204673
            nonnegativity_auxilliary₃ = max(eps(), c - 1)
            nonnegativity_auxilliary₂ = max(eps(), 1)
            log_ZZ_avg = 0
            Pibar = Pi / nonnegativity_auxilliary₁
            c_norminvcdf = -1.4142135623731 * erfcinv(2.0nonnegativity_auxilliary₃)
            y = k ^ alpha
            eps_z = 0
            eps_zᴸ⁽⁻¹⁾ = 0
            eps_zᴸ⁽¹⁾ = 0

            if scale == 1
                # return (KeyedArray([A, Pi, R, ZZ_avg, ZZ_avg_fut, c, c_norminvcdf, c_normlogpdf, eps_z, eps_zᴸ⁽¹⁾, eps_zᴸ⁽⁻¹⁾, k, log_ZZ_avg, y, z_delta, beta, Pibar], Variables_and_parameters = [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]), solution_error)
                return ([A, Pi, R, ZZ_avg, ZZ_avg_fut, c, c_norminvcdf, c_normlogpdf, eps_z, eps_zᴸ⁽¹⁾, eps_zᴸ⁽⁻¹⁾, k, log_ZZ_avg, y, z_delta, beta, Pibar], solution_error)
            end
        end
    end
end

m.SS_solve_func(parameters, m, false, false)[1]

using Zygote

Zygote.jacobian(x -> m.SS_solve_func(x, m, false, false)[1],Float64[parameters...])[1]



m.SS_solve_func(parameters, m, false, false)

using Zygote

Zygote.jacobian(x -> m.SS_solve_func(x, m, false, false),Float64[parameters...])



include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")

get_SS(m)

function SS_solve_func(params, 𝓂, fail_fast_solvers_only, verbose)
    cap_share = params[1]
    R_ss = params[2]
    I_K_ratio = params[3]
    phi_pi = params[4]
    Pi_real = params[7]
    rhoz = params[8]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:679 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:680 =#
    Pi_ss = R_ss - Pi_real
    rho_z_delta = rhoz

    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:682 =#
    solution_error = 0.0
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:683 =#
    A = 1
    nonnegativity_auxilliary₂ = max(eps(), 1)
    R = R_ss
    Pi = Pi_ss
    beta = Pi / R
    nonnegativity_auxilliary₁ = max(eps(), (R * beta) ^ (1 / phi_pi))
    Pibar = Pi / nonnegativity_auxilliary₁
    z_delta = 1
    lbs = [2.220446049250313e-16, -1.0e12, -1.0e12, 1.1920928955078125e-7, -1.0e12]
    ubs = [0.9999999999999998, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
    f = OptimizationFunction(𝓂.ss_solve_blocks_optim[1], Optimization.AutoForwardDiff())
    inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

    block_solver_AD = ImplicitFunction(x->block_solver(x, 1, 𝓂.ss_solve_blocks[1], f, inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose), 𝓂.ss_solve_blocks[1])

    solution = block_solver_AD([cap_share, I_K_ratio, beta])
    solution_error += solution[2]
    sol = solution[1]
    alpha = sol[1]
    c = sol[2]
    delta = sol[3]
    k = sol[4]
    y = sol[5]
    
    ZZ_avg = 1
    ZZ_avg_fut = 1
    c_normlogpdf = -0.5 * c ^ 2 - 0.918938533204673
    nonnegativity_auxilliary₃ = max(eps(), c - 1)
    log_ZZ_avg = 0
    c_norminvcdf = -1.4142135623731 * erfcinv(2.0nonnegativity_auxilliary₃)
    eps_z = 0
    eps_zᴸ⁽⁻¹⁾ = 0
    eps_zᴸ⁽¹⁾ = 0

    return ([A, Pi, R, ZZ_avg, ZZ_avg_fut, c, c_norminvcdf, c_normlogpdf, eps_z, eps_zᴸ⁽¹⁾, eps_zᴸ⁽⁻¹⁾, k, log_ZZ_avg, y, z_delta, alpha, beta, delta, Pibar], solution_error)
end


using Optimization, OptimizationNLopt, ImplicitDifferentiation, StatsFuns, SpecialFunctions, AxisKeys, ComponentArrays

SS_and_pars = SS_solve_func(Float64[parameters...], m, false, false)[1]


using Zygote

Zygote.jacobian(x -> SS_solve_func(x, m, false, false)[1], Float64[parameters...])[1]

m.ss_solve_blocks[1]([parameters[[1,3]]...,SS_and_pars[:beta]...],[SS_and_pars[:alpha],SS_and_pars[:c],SS_and_pars[:delta],SS_and_pars[:k],SS_and_pars[:y]])



Zygote.jacobian(x -> m.ss_solve_blocks[1](x,[SS_and_pars[:alpha],SS_and_pars[:c],SS_and_pars[:delta],SS_and_pars[:k],SS_and_pars[:y]]), [parameters[[1,3]]...,SS_and_pars[:beta]...])[1]

Zygote.jacobian(x -> m.ss_solve_blocks[1]([parameters[[1,3]]...,SS_and_pars[:beta]...],x), [SS_and_pars[:alpha],SS_and_pars[:c],SS_and_pars[:delta],SS_and_pars[:k],SS_and_pars[:y]])[1]



m.parameters
SS_and_pars[:beta]

using ComponentArrays

ComponentArray(X = true, y = false, z = randn(100))

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end




get_solution(RBC)

using ImplicitDifferentiation

block_solver_AD(parameters_and_solved_vars::Vector{Float64}, 
n_block::Int, 
ss_solve_blocks::Function, 
# SS_optimizer, 
f::OptimizationFunction, 
guess::Vector{Float64}, 
lbs::Vector{Float64}, 
ubs::Vector{Float64};
tol = eps(Float64),
maxtime = 120,
starting_points = [.9, 1, 1.1, .75, 1.5, -.5, 2, .25],
fail_fast_solvers_only = true,
verbose = false) = ImplicitFunction(block_solver, ss_solve_blocks)

RBC.model_jacobian

symbolics = create_symbols_eqs!(RBC)


using ComponentArrays, SparseArrays
import ForwardDiff as ℱ


include("models/SW07.jl")
get_SS(m)

include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
get_solution(m)

m.dyn_exo_future_list
m.dyn_exo_present_list
m.dyn_exo_past_list
m.dyn_equations
m.var_future_list_aux_SS
m.var_past_list_aux_SS
m.dyn_equations

m.dyn_equations[1]|>dump
using ComponentArrays
𝓂= m 
parameters = m.parameter_values
SS_and_pars, _ = m.SS_solve_func(parameters, m, false, true)

var_past = setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
    var_present = setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
    var_future = setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)

    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
    calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
    par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))

    past_idx = [indexin(sort([var_past; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_past,𝓂.exo_past))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_past =       length(past_idx) > 0 ? SS[past_idx] : zeros(0) #; zeros(length(𝓂.exo_past))...]
    
    present_idx = [indexin(sort([var_present; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_present,𝓂.exo_present))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_present =    length(present_idx) > 0 ? SS[present_idx] : zeros(0)#; zeros(length(𝓂.exo_present))...]
    
    future_idx = [indexin(sort([var_future; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_future,𝓂.exo_future))]), sort(union(𝓂.var,𝓂.exo_present)))...]
    SS_future =     length(future_idx) > 0 ? SS[future_idx] : zeros(0)#; zeros(length(𝓂.exo_future))...]

    shocks_ss = zeros(length(𝓂.exo))


    collect(𝓂.model_jacobian([SS_future; SS_present; SS_past; shocks_ss], par, SS))

function get_symbols(ex)
    par = Set()
    postwalk(x ->   
    x isa Expr ? 
        x.head == :(=) ?
            for i in x.args
                i isa Symbol ? 
                    push!(par,i) :
                x
            end :
        x.head == :call ? 
            for i in 2:length(x.args)
                x.args[i] isa Symbol ? 
                    push!(par,x.args[i]) : 
                x
            end : 
        x : 
    x, ex)
    return par
end

using MacroTools: postwalk
all_symbols = get_symbols.(m.dyn_equations)
contains.(all_symbols[1].|>string,"₍₀₎")
setdiff.(all_symbols,m.dyn_exo_future_list)


function match_pattern(strings::Union{Set,Vector}, pattern::Regex)
    return filter(r -> match(pattern, string(r)) != nothing, strings)
end


all_symbols = get_symbols.(m.dyn_equations)

future_symbols = match_pattern.(all_symbols,r"₍₁₎")
present_symbols = match_pattern.(all_symbols,r"₍₀₎")
past_symbols = match_pattern.(all_symbols,r"₍₋₁₎")

intersect(future_symbols,m.var


get_solution(m)
get_moments(m)[2]


include("models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl")
get_solution(m)
get_moments(m)[2]
create_symbols_eqs!(m)
get_irf(m, algorithm = :second_order)
get_irf(m)

m.model_jacobian
m.model_hessian

𝓂 = m

symbolics = create_symbols_eqs!(𝓂)
parameters = 𝓂.parameter_values
SS_and_pars, _ = 𝓂.SS_solve_func(parameters, 𝓂, false, false)

var_past = setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
var_present = setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
var_future = setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)

SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))

past_idx = [indexin(sort([var_past; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_past,𝓂.exo_past))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_past =       length(past_idx) > 0 ? SS[past_idx] : zeros(0) #; zeros(length(𝓂.exo_past))...]

present_idx = [indexin(sort([var_present; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_present,𝓂.exo_present))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_present =    length(present_idx) > 0 ? SS[present_idx] : zeros(0)#; zeros(length(𝓂.exo_present))...]

future_idx = [indexin(sort([var_future; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux_future,𝓂.exo_future))]), sort(union(𝓂.var,𝓂.exo_present)))...]
SS_future =     length(future_idx) > 0 ? SS[future_idx] : zeros(0)#; zeros(length(𝓂.exo_future))...]

shocks_ss = zeros(length(𝓂.exo))

jacFD = ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), [SS_future; SS_present; SS_past; shocks_ss])



nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)

hessian = sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> (𝓂.model_function(x, par, SS)), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^2))



third_order_derivatives = sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), x), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^3))



jacRD = 𝓂.model_jacobian([SS_future; SS_present; SS_past; shocks_ss], par, SS) |>collect

hesRD = 𝓂.model_hessian([SS_future; SS_present; SS_past; shocks_ss], par, SS)

thirdRD = 𝓂.model_third_order_derivatives([SS_future; SS_present; SS_past; shocks_ss], par, SS)



jacRD - jacFD
findnz(jacRD)[3] .- findnz(jacFD)[3]
findnz(hesRD)[3] .- findnz(hessian)[3]


findnz(thirdRD)[3] .- findnz(third_order_derivatives)[3]


hesRD ≈ hessian
thirdRD ≈ third_order_derivatives

using BenchmarkTools
@benchmark ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), [SS_future; SS_present; SS_past; shocks_ss])

@benchmark sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> (𝓂.model_function(x, par, SS)), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^2))

@benchmark sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), x), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^3))

@benchmark 𝓂.model_jacobian([SS_future; SS_present; SS_past; shocks_ss], par, SS)
@benchmark 𝓂.model_hessian([SS_future; SS_present; SS_past; shocks_ss], par, SS)
@benchmark 𝓂.model_third_order_derivatives([SS_future; SS_present; SS_past; shocks_ss], par, SS)

@profview 𝓂.model_hessian([SS_future; SS_present; SS_past; shocks_ss], par, SS)

using BenchmarkTools
@benchmark jacFD = ℱ.jacobian(x -> 𝓂.model_function(x, par, SS), [SS_future; SS_present; SS_past; shocks_ss])#, SS_and_pars

@benchmark jacRD = 𝓂.model_jacobian([SS_future; SS_present; SS_past; shocks_ss], par, SS)

l



using SymPy

dyn_var_future_list = collect(reduce(union,symbolics.dyn_var_future_list))
dyn_var_present_list = collect(reduce(union,symbolics.dyn_var_present_list))
dyn_var_past_list = collect(reduce(union,symbolics.dyn_var_past_list))
dyn_exo_list = collect(reduce(union,symbolics.dyn_exo_list))

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_var_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_var_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_var_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))

vars = [dyn_var_future_list[indexin(sort(future),future)]...,
        dyn_var_present_list[indexin(sort(present),present)]...,
        dyn_var_past_list[indexin(sort(past),past)]...,
        dyn_exo_list[indexin(sort(exo),exo)]...]

eqs = symbolics.dyn_equations

first_order = []
second_order = []
third_order = []
row1 = Int[]
row2 = Int[]
row3 = Int[]
column1 = Int[]
column2 = Int[]
column3 = Int[]
i1 = 1
i2 = 1
i3 = 1

for (c1,var1) in enumerate(vars)
    for (r,eq) in enumerate(eqs)
        if var1 ∈ free_symbols(eq)
            deriv_first = diff(eq,var1)
            if deriv_first != 0 
                push!(first_order, :(out[$i1] = $deriv_first))
                push!(row1,r)
                push!(column1,c1)
                i1 += 1
                for (c2,var2) in enumerate(vars)
                    if var2 ∈ free_symbols(deriv_first)
                        deriv_second = diff(deriv_first,var2)
                        if deriv_second != 0 
                            push!(second_order, :(out[$i2] = $deriv_second))
                            push!(row2,r)
                            push!(column2,(c1 - 1) * length(vars) + c2)
                            i2 += 1
                            for (c3,var3) in enumerate(vars)
                                if var3 ∈ free_symbols(deriv_second)
                                    deriv_third = diff(deriv_second,var3)
                                    if deriv_third != 0 
                                        push!(third_order, :(out[$i3] = $deriv_third))
                                        push!(row3,r)
                                        push!(column3,(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3)
                                        i3 += 1
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


first_order
first_order[29].subs(PI,N(PI))
RBC.model_jacobian
RBC.model_hessian

nk = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + length(𝓂.exo)
        
hessian = sparse(reshape(ℱ.jacobian(x -> ℱ.jacobian(x -> (𝓂.model_function(x, par, SS)), x), [SS_future; SS_present; SS_past; shocks_ss] ), 𝓂.timings.nVars, nk^2))#, SS_and_pars

findnz(hessian)


mod_func3 = :(function model_jacobian(X::Vector{Real}, params::Vector{Real}, X̄::Vector{Real})
    $(alll...)
    $(paras...)
    $(𝓂.calibration_equations_no_var...)
    $(steady_state...)
    out = zeros($(length(deriv)))
    $(deriv...)
    sparse([$(row...)], [$(column...)], out, $(length(eqs)), $(length(vars)))
end)


# RBC.dyn_equations

# using Symbolics
# 𝓂 = RBC

# symbols_in_equation = union(𝓂.var,𝓂.par,𝓂.parameters,𝓂.parameters_as_function_of_parameters,𝓂.exo,𝓂.dynamic_variables,𝓂.nonnegativity_auxilliary_vars)#,𝓂.

# varexp = :(@variables ($(symbols_in_equation...)))

# eval(varexp)

# vars = [z₍₋₁₎,k₍₋₁₎,c₍₀₎,z₍₀₎,k₍₀₎,q₍₀₎,c₍₁₎,z₍₁₎,eps_z₍ₓ₎]

# eqs = eval.(RBC.dyn_equations)



# var_past = setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
# var_present = setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
# var_future = setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)
# 𝓂.exo

# vars = [[eval(Symbol(string(v)*"₍₋₁₎")) for v in var_past]...,
# [eval(Symbol(string(v)*"₍₀₎")) for v in var_present]...,
# [eval(Symbol(string(v)*"₍₁₎")) for v in var_future]...,
# [eval(Symbol(string(v)*"₍ₓ₎")) for v in 𝓂.exo]...]
# # @time Symbolics.sparsejacobian(eqs,vars)

# # @time Symbolics.sparsehessian(eqs,vars)

# # @time Symbolics.hessian(eqs,vars)

# # eqs

# using SparseArrays
# # jac = Symbolics.sparsejacobian(eqs,vars)
# # hess = Symbolics.sparsejacobian(findnz(jac)[3],vars)
# jacobian = Symbolics.jacobian(eqs,vars)
# jacobian[2,1] |> Expr
# expp = Symbolics._build_function(Symbolics.JuliaTarget(),Symbolics.jacobian(eqs,vars), force_SA = true)

# eval(expp)
# @time third_order = Symbolics.sparsejacobian(findnz(Symbolics.sparsejacobian(findnz(Symbolics.sparsejacobian(eqs,vars))[3],vars))[3],vars);

# [Symbolics.sparsejacobian(eqs,vars) for v1 in vars, v2 in vars];


using SymPy, SparseArrays

𝓂 = RBC


symbolics = create_symbols_eqs!(𝓂)


vars = [reduce(union,symbolics.dyn_var_past_list)...,
        reduce(union,symbolics.dyn_var_present_list)...,
        reduce(union,symbolics.dyn_var_future_list)...,
        reduce(union,symbolics.dyn_exo_list)...]

eqs = symbolics.dyn_equations




deriv = []
row = Int[]
column = Int[] 
i = 1
for (r,eq) in enumerate(eqs)
    for (c,var) in enumerate(vars)
        deriv_res = diff(eq,var)
        if deriv_res != 0 
            push!(deriv, :(out[$i] = $deriv_res))
            push!(row,r)
            push!(column,c)
            i += 1
        end
    end
end

mod_func2 = :(function model_function_uni_redux(X::Vector{Real}, params::Vector{Real}, X̄::Vector{Real})
out = zeros($(length(deriv)))
$(deriv...)
sparse([$(row...)], [$(column...)], out, $(length(eqs)), $(length(vars)))
end)



symbols_in_equation = union(𝓂.var,𝓂.par,𝓂.parameters,𝓂.parameters_as_function_of_parameters,𝓂.exo,𝓂.dynamic_variables,𝓂.nonnegativity_auxilliary_vars)#,𝓂.dynamic_variables_future)
l_bnds = Dict(𝓂.bounded_vars .=> 𝓂.lower_bounds)
u_bnds = Dict(𝓂.bounded_vars .=> 𝓂.upper_bounds)

symbols_pos = []
symbols_neg = []
symbols_none = []

for symb in symbols_in_equation
    if symb in 𝓂.bounded_vars
        if l_bnds[symb] >= 0
            push!(symbols_pos, symb)
        elseif u_bnds[symb] <= 0
            push!(symbols_neg, symb)
        end
    else
        push!(symbols_none, symb)
    end
end

expr =  quote
            @vars $(symbols_pos...)  real = true finite = true positive = true
            @vars $(symbols_neg...)  real = true finite = true negative = true 
            @vars $(symbols_none...) real = true finite = true 
        end

eval(expr)

var_past = setdiff(𝓂.var_past,𝓂.nonnegativity_auxilliary_vars)
var_present = setdiff(𝓂.var_present,𝓂.nonnegativity_auxilliary_vars)
var_future = setdiff(𝓂.var_future,𝓂.nonnegativity_auxilliary_vars)
𝓂.exo

vars = [[eval(Symbol(string(v)*"₍₋₁₎")) for v in var_past]...,
        [eval(Symbol(string(v)*"₍₀₎")) for v in var_present]...,
        [eval(Symbol(string(v)*"₍₁₎")) for v in var_future]...,
        [eval(Symbol(string(v)*"₍ₓ₎")) for v in 𝓂.exo]...]

eqs = eval.(RBC.dyn_equations)

# [diff(eq,var)  for (r,eq) in eqs, (c,var) in vars]

deriv = []
row = Int[]
column = Int[] 
i = 1
for (r,eq) in enumerate(eqs)
    for (c,var) in enumerate(vars)
        deriv_res = diff(eq,var)
        if deriv_res != 0 
            push!(deriv, :(out[$i] = $deriv_res))
            push!(row,r)
            push!(column,c)
            i += 1
        end
    end
end

mod_func2 = :(function model_function_uni_redux(X::Vector{Real}, params::Vector{Real}, X̄::Vector{Real})
out = zeros($(length(deriv)))
$(deriv...)
sparse([$(row...)], [$(column...)], out, $(length(eqs)), $(length(vars)))
end)




sparse(row,column,deriv.|>string.|>Meta.parse,length(eqs),length(vars))

sparse(row,column,,length(eqs),length(vars))

jac = eqs.jacobian(vars)
sparse(jac)
jac[3,1]|> string |> Meta.parse
[length(intersect(free_symbols(j), vars)) > 0 for j in jac]
length(free_symbols(x)) > 0
Float64(findnz(sparse(jac))[3][4])
jac = vec(eqs.jacobian(vars)).jacobian(vars)

jacobian = jacobian.jacobian(vars)
# vars = [z₍₋₁₎,k₍₋₁₎,c₍₀₎,z₍₀₎,k₍₀₎,q₍₀₎,c₍₁₎,z₍₁₎,eps_z₍ₓ₎]

@time hessian = [diff.(eqs,v1, v2) for v1 in vars, v2 in vars];

@time [diff.(eqs,v1, v2, v3) for v1 in vars, v2 in vars, v3 in vars];



hessian(eqs, vars)

using Symbolics




SymPy.diff.(eqs,k₍₋₁₎, k₍₋₁₎)

SymPy.diff.(eqs,[k₍₋₁₎,c₍₀₎])

sympify()

shock_series = zeros(RBC.timings.nExo,4)
shock_series[1,2] = 1
shock_series[1,4] = -1
plot(RBC, shocks = shock_series, save_plots = true, save_plots_format = :png)

# get_irf(RBC, shocks = shock_series)







using MacroModelling
@model SW03 begin
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


@parameters SW03 begin  
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


using AxisKeys
shock_series = KeyedArray(zeros(2,12), Shocks = [:eta_b, :eta_w], Periods = 1:12)
shock_series[1,2] = 1
shock_series[2,12] = -1
plot(SW03, shocks = shock_series, variables = [:W,:r_k,:w_star,:R])
plot(SW03, shocks = shock_series, variables = [:W,:r_k,:w_star,:R], save_plots = true, save_plots_format = :png)



@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end



𝓂 = RBC_CME
m = RBC_CME




old_par_vals = copy(m.parameter_values)

lvl_irfs  = get_irf(m, old_par_vals, verbose = true, levels = true)

new_sub_lvl_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = :none, initial_state = collect(lvl_irfs[:,5,1]), levels = true)

isapprox(collect(new_sub_lvl_irfs[:,1,:]), collect(lvl_irfs[:,6,1]),rtol = eps(Float32))


using AxisKeys

get_irf(𝓂,𝓂.parameter_values)
get_irf(𝓂)

get_irf(𝓂,𝓂.parameter_values, shocks = randn(𝓂.timings.nExo, 13))

get_irf(𝓂, shocks = randn(𝓂.timings.nExo, 13))

shock_hist = zeros(𝓂.timings.nExo, 13)
shock_hist[2,1] = 1
get_irf(𝓂,𝓂.parameter_values, shocks = shock_hist)

shock_hist[1,5] = 1
get_irf(𝓂, shocks = shock_hist, algorithm = :second_order)



shocks = zeros(𝓂.timings.nExo, 10)
shocks = KeyedArray(reshape(shocks[1,:],1,10), Shocks = [𝓂.timings.exo[1]], Periods = 1:size(shocks)[2])
shocks[1,4] = 1
get_irf(𝓂, shocks = shocks, algorithm = :second_order)


if shocks isa Matrix{Float64}
    @assert size(shocks)[1] == 𝓂.timings.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."
    shock_history = shocks
    shock_idx = 1:𝓂.timings.nExo
elseif shocks isa KeyedArray{Float64}
    shock_input = axiskeys(shocks)[1]
    @assert length(setdiff(shock_input, 𝓂.timings.exo)) == 0 "Provided shocks which are not part of the model."
    shock_history = zeros(𝓂.timings.nExo, size(shocks)[2])
    shock_idx = indexin(shock_input,𝓂.timings.exo)
    shock_history[shock_idx,:] = shocks
    sort!(shock_idx)
else
    shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)
end




include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
lvl_irfs  = get_irf(m, verbose = true, levels = true)

lvlv_init_irfs  = get_irf(m, verbose = true, levels = true, initial_state = collect(lvl_irfs(:,5,m.exo[1])))



𝓂 = m
state = :k



using Plots, AxisKeys
using MacroModelling: Symbol_input,ℳ, parse_variables_input_to_index

function plot_solution(𝓂::ℳ,
    state::Symbol;
    variables::Symbol_input = :all,
    algorithm::Union{Symbol,Vector{Symbol}} = :first_order,
    σ::Float64 = 2.0,
    parameters = nothing,
    show_plots::Bool = true,
    save_plots::Bool = false,
    save_plots_format::Symbol = :pdf,
    save_plots_path::String = ".",
    plots_per_page::Int = 4,
    verbose = false)

    @assert state ∈ 𝓂.timings.past_not_future_and_mixed "Invalid state. Choose one from:"*repr(𝓂.timings.past_not_future_and_mixed)

    @assert length(setdiff(algorithm isa Symbol ? [algorithm] : algorithm, [:third_order, :second_order, :first_order])) == 0 "Invalid algorithm. Choose any combination of: :third_order, :second_order, :first_order"

    if algorithm isa Symbol
        max_algorithm = algorithm
        algorithm = [algorithm]
    else
        if :third_order ∈ algorithm 
            max_algorithm = :third_order 
        elseif :second_order ∈ algorithm 
            max_algorithm = :second_order 
        else 
            max_algorithm = :first_order 
        end
    end

    solve!(𝓂, verbose = verbose, algorithm = max_algorithm, dynamics = true)

    SS_and_std = get_moments(𝓂, 
                            derivatives = false,
                            parameters = parameters,
                            verbose = verbose)


    full_NSSS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
    full_NSSS[indexin(𝓂.aux,full_NSSS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    full_SS = [SS_and_std[1](s) for s in full_NSSS]

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    default(size=(700,500),
            plot_titlefont = (10), 
            titlefont = (10), 
            guidefont = (8), 
            legendfontsize = 8, 
            tickfontsize = 8,
            framestyle = :box)

    vars_to_plot = intersect(axiskeys(SS_and_std[1])[1],𝓂.timings.var[var_idx])

    state_range = collect(range(-SS_and_std[2](state), SS_and_std[2](state), 100)) * σ

    state_selector = state .== 𝓂.timings.var

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1

    for k in vars_to_plot

        kk = Symbol(replace(string(k), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

        if !(plot_count % plots_per_page == 0)
            plot_count += 1
            
            if :first_order ∈ algorithm
                variable_first = [𝓂.solution.perturbation.first_order.state_update(state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_first = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_first]
            end

            if :second_order ∈ algorithm
                SSS = 𝓂.solution.perturbation.second_order.stochastic_steady_state

                variable_second = [𝓂.solution.perturbation.second_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_second = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_second]
            end

            if :third_order ∈ algorithm
                SSS = 𝓂.solution.perturbation.third_order.stochastic_steady_state

                variable_third = [𝓂.solution.perturbation.third_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_third = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_third]
            end

            push!(pp,begin 
                        if :third_order ∈ algorithm 
                            Pl = Plots.plot(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                variable_third, 
                                ylabel = string(k)*"₍₀₎", 
                                xlabel = string(state)*"₍₋₁₎",
                                label = "3rd order perturbation")
                        end
                        if :second_order ∈ algorithm
                            if :second_order == max_algorithm 
                                Pl = Plots.plot(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                    variable_second, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "2nd order perturbation")
                            else
                                Plots.plot!(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                    variable_second, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "2nd order perturbation")
                            end
                        end
                        if :first_order ∈ algorithm
                            if :first_order  == max_algorithm 
                                Pl = Plots.plot(state_range .+ SS_and_std[1](state), 
                                    variable_first, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "1st order perturbation")

                                Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], label = "Non Stochastic Steady State")
                            else
                                Plots.plot!(state_range .+ SS_and_std[1](state), 
                                    variable_first, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "1st order perturbation")

                                Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], label = "Non Stochastic Steady State")
                            end
                        end

                        if :second_order ∈ algorithm || :third_order ∈ algorithm
                            Plots.scatter!([SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], [SSS[indexin([k],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], label = "Stochastic Steady State")
                        end

                        Pl
                    end)
        else
            plot_count = 1

            if :first_order ∈ algorithm
                variable_first = [𝓂.solution.perturbation.first_order.state_update(state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_first = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_first]
            end

            if :second_order ∈ algorithm
                SSS = 𝓂.solution.perturbation.second_order.stochastic_steady_state

                variable_second = [𝓂.solution.perturbation.second_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_second = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_second]
            end

            if :third_order ∈ algorithm
                SSS = 𝓂.solution.perturbation.third_order.stochastic_steady_state

                variable_third = [𝓂.solution.perturbation.third_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

                variable_third = [(abs(x) > eps() ? x : 0.0) + SS_and_std[1](kk) for x in variable_third]
            end

            push!(pp,begin 
                        if :third_order ∈ algorithm 
                            Pl = Plots.plot(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                variable_third, 
                                ylabel = string(k)*"₍₀₎", 
                                xlabel = string(state)*"₍₋₁₎",
                                label = "3rd order perturbation")
                        end
                        if :second_order ∈ algorithm
                            if :second_order == max_algorithm 
                                Pl = Plots.plot(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                    variable_second, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "2nd order perturbation")
                            else
                                Plots.plot!(state_range .+ SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1], 
                                    variable_second, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "2nd order perturbation")
                            end
                        end
                        if :first_order ∈ algorithm
                            if :first_order  == max_algorithm 
                                Pl = Plots.plot(state_range .+ SS_and_std[1](state), 
                                    variable_first, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "1st order perturbation")

                                Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], label = "Non Stochastic Steady State")
                            else
                                Plots.plot!(state_range .+ SS_and_std[1](state), 
                                    variable_first, 
                                    ylabel = string(k)*"₍₀₎", 
                                    xlabel = string(state)*"₍₋₁₎",
                                    label = "1st order perturbation")

                                Plots.scatter!([SS_and_std[1](state)], [SS_and_std[1](kk)], label = "Non Stochastic Steady State")
                            end
                        end

                        if :second_order ∈ algorithm || :third_order ∈ algorithm
                            Plots.scatter!([SSS[indexin([state],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], [SSS[indexin([k],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]], label = "Stochastic Steady State")
                        end

                        Pl
                    end)

            p = Plots.plot(pp..., plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            if show_plots
                display(p)
            end

            if save_plots
                savefig(p, save_plots_path * "/solution__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        p = Plots.plot(pp..., plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

        if show_plots
            display(p)
        end

        if save_plots
            savefig(p, save_plots_path * "/solution__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end
end


# plot_solution(m, :k, variables = :Pi, algorithm = :third_order)


include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")

std(m, derivatives = false)
plot_solution(m, :z_delta, variables = :k)

plot_solution(m, :k)

get_stochastic_steady_state(m)

include("models/FS2000.jl")
plot_solution(m, :k, algorithm = [:first_order], plots_per_page = 6)

algorithm = [:third_order, :frst_order]
length(setdiff(algorithm isa Symbol ? [algorithm] : algorithm, [:third_order, :second_order, :first_order]))


setdiff([:first_order,:second_order],[:first_order,:second_order,:third_order])|>length
algorithm = [:second_order,:third_order]


if algorithm isa Symbol
    max_algorithm = algorithm
else
    if :third_order ∈ algorithm 
        max_algorithm = :third_order 
    elseif :second_order ∈ algorithm 
        max_algorithm = :second_order 
    else 
        max_algorithm = :first_order 
    end
end

st = :k

𝓂 = m
solve!(𝓂, dynamics = true, algorithm = :second_order)

SS_and_std = get_moments(𝓂, derivatives = false)

SSS = 𝓂.solution.perturbation.second_order.stochastic_steady_state

state_selector = st .== 𝓂.timings.var

delta = 1
state = 0 * SSS

shock = zeros(𝓂.timings.nExo)
while delta > eps(Float64)
    state_tmp = 𝓂.solution.perturbation.second_order.state_update(state,shock)
    delta = sum(abs,state_tmp - state)
    state = state_tmp
end

complete_state = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
complete_state[indexin(𝓂.aux,complete_state)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

full_SS = [𝓂.solution.non_stochastic_steady_state[s] for s in complete_state]

full_SS + state
SSS




state_range = collect(range(-SS_and_std[2](st), SS_and_std[2](st), 100)) * 2

[SSS - full_SS .+ state_selector * x for x in state_range]
variable_second = [𝓂.solution.perturbation.second_order.state_update(SSS - full_SS .+ state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]

variable_second .+ full_SS[state_selector]


SSS  - full_SS
m.solution.perturbation.second_order.state_update(full_SSS - full_SS, zeros(𝓂.timings.nExo)) + (full_SSS - full_SS)



SState = get_SS(m,derivatives = false)





[SSS[indexin([:y],sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))][1]]

shock = zeros(m.timings.nExo)
delta = 1
SS = 𝓂.solution.non_stochastic_steady_state
complete_state = sort(union(m.var,m.aux,m.exo_present))
complete_state[indexin(m.aux,complete_state)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  m.aux)

full_SS = [SS[s] for s in complete_state]

state = full_SS * 0
# state_tmp = nothing
delta = 1

while delta > eps(Float64)
    state_tmp = m.solution.perturbation.second_order.state_update(state,shock)
    delta = sum(abs,state_tmp - state)
    state = state_tmp
end

full_SS + state
SSs = m.solution.perturbation.second_order.stochastic_steady_state

full_SS = [SState(s) for s in full_NSSS]
m.solution.perturbation.second_order.state_update(full_SSS - full_SS,shock)

SS_and = get_moments(m, 
derivatives = false)
SS_and[1]
SSSS = get_stochastic_steady_state(m)

full_SSS = [SSSS(s) for s in full_NSSS]



𝓂 = RBC_CME


state = :k
variable = :A

get_solution(𝓂)

get_SS(𝓂,derivatives = false)([variable,state])
get_std(𝓂, derivatives = false)([variable,state])

out = get_moments(𝓂, derivatives = false)

state_range = collect(range(-out[2](state),out[2](state),100)) * 2

state_selector = state .== 𝓂.timings.var
variable_idx = indexin([variable],𝓂.timings.var)

variable_output = [𝓂.solution.perturbation.first_order.state_update(state_selector * x,zeros(𝓂.timings.nExo))[variable_idx][1] for x in state_range]
[x -> 𝓂.solution.perturbation.first_order.state_update(state_selector * x, zeros(𝓂.timings.nExo))[indexin([k],𝓂.timings.var)][1] for x in state_range]
# 𝓂.solution.perturbation.first_order.state_update(state_selector * state_range[1],zeros(𝓂.timings.nExo))[variable_idx]

Plots.plot(state_range .+ out[1](state), variable_output .* 0 .+ out[1](variable), ylabel = string(variable), xlabel = string(state),label = "1st order perturbation")


Plots.plot(state_range .+ out[1](state), variable_output .+ out[1](variable), ylabel = string(variable), xlabel = string(state),label = "1st order perturbation")



Plots.plot(state_range .+ out[1](state), x -> 𝓂.solution.perturbation.first_order.state_update(state_selector * (x .- out[1](state)),zeros(𝓂.timings.nExo))[variable_idx][1]  .+ out[1](variable), ylabel = string(variable), xlabel = string(state),label = "1st order perturbation")

Plots.scatter!([out[1](state)], [out[1](variable)], label = "Non Stochastic Steady State")



SSS = get_stochastic_steady_state(𝓂)


Plots.plot!(state_range .+ out[1](state), x -> 𝓂.solution.perturbation.second_order.state_update(state_selector * (x .- out[1](state)),zeros(𝓂.timings.nExo))[variable_idx][1] .+ out[1](variable), label = "2nd order perturbation")

Plots.scatter!([SSS(state)], [SSS(variable)], label = "Stochastic Steady State")









Plots.plot(state_range, variable_output)

range(-.5*(1+1/3),(1+1/3)*.5,100)

m.solution.perturbation.first_order.solution_matrix

pol = [[i,m.solution.perturbation.first_order.state_update([0,0.0,i],[0.0])[1]] for i in range(-.05*(1+1/3),(1+1/3)*.05,100)]
solve!(m,algorithm = :second_order, dynamics= true)
using Plots

pol2 = [[i,m.solution.perturbation.second_order.state_update([0,0.0,i],[0.0])[1]] for i in range(-.05*(1+1/3),(1+1/3)*.05,100)]

Plots.plot(reduce(hcat,pol)[1,:],reduce(hcat,pol)[2,:])
Plots.plot!(reduce(hcat,pol2)[1,:],reduce(hcat,pol2)[2,:])




ff = fevd(RBC_CME)

default(size=(700,500),
# leg = false,
# plot_titlefont = (10, fontt), 
# titlefont = (10, fontt), 
# guidefont = (8, fontt), 
plot_titlefont = (10), 
titlefont = (8), 
guidefont = (8), 
legendfontsize = 8, 
# tickfont = (8, fontt),
# tickfontfamily = fontt,
tickfontsize = 8,
# tickfontrotation = 9,
# rotation = 90,
# tickfontvalign = :center,
# topmargin = 10mm,
# rightmargin = 17mm, 
framestyle = :box)


include("models/SW03.jl")

ff = fevd(m)

using StatsPlots, AxisKeys

vars_to_plot = axiskeys(ff)[1][1:9]
shocks_to_plot = axiskeys(ff)[2]
pp = []
for k in vars_to_plot
    push!(pp,groupedbar(ff(k,:,:)', title = string(k), titlefont = (10), bar_position = :stack, legend = :none))
end
ppp = Plots.plot(pp...)

Plots.plot(ppp,Plots.bar(fill(0,1,length(axiskeys(ff)[2])), label = reshape(string.(axiskeys(ff)[2]),1,length(axiskeys(ff)[2])),  linewidth = 0 , framestyle = :none, legend = :inside, legend_columns=-1), layout = grid(2, 1, heights=[0.99, 0.01]))

Plots.plot(ppp,Plots.bar(fill(0,1,length(shocks_to_plot)), label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot)),  linewidth = 0 , framestyle = :none, legend = :left), layout = grid(1, 2, widths=[0.8, 0.2]))





using MacroModelling: ℳ, Symbol_input
using Plots, AxisKeys, StatsPlots


function plot_fevd(𝓂::ℳ;
    periods::Union{Vector{Int},Vector{Float64},UnitRange{Int64}} = 1:40,
    variables::Symbol_input = :all,
    parameters = nothing,
    show_plots::Bool = true,
    save_plots::Bool = false,
    save_plots_format::Symbol = :pdf,
    save_plots_path::String = ".",
    plots_per_page::Int = 9, 
    verbose = false)

    fevds = get_conditional_variance_decomposition(𝓂,
                                                    periods = periods,
                                                    parameters = parameters,
                                                    verbose = verbose)

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    default(size=(700,500),
            plot_titlefont = (10), 
            titlefont = (10), 
            guidefont = (8), 
            legendfontsize = 8, 
            tickfontsize = 8,
            framestyle = :box)

    vars_to_plot = axiskeys(fevds)[1][var_idx]
    
    shocks_to_plot = axiskeys(fevds)[2]

    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1

    for k in vars_to_plot
        if !(plot_count % plots_per_page == 0)
            plot_count += 1
            push!(pp,groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, legend = :none))
        else
            plot_count = 1

            push!(pp,groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, legend = :none))
            
            ppp = Plots.plot(pp...)

            p = Plots.plot(ppp,Plots.bar(fill(0,1,length(shocks_to_plot)), 
                                        label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot)), 
                                        linewidth = 0 , 
                                        framestyle = :none, 
                                        legend = :inside, 
                                        legend_columns = -1), 
                                        layout = grid(2, 1, heights=[0.99, 0.01]),
                                        plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            if show_plots
                display(p)
            end

            if save_plots
                savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        ppp = Plots.plot(pp...)

        p = Plots.plot(ppp,Plots.bar(fill(0,1,length(shocks_to_plot)), 
                                    label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot)), 
                                    linewidth = 0 , 
                                    framestyle = :none, 
                                    legend = :inside, 
                                    legend_columns = -1), 
                                    layout = grid(2, 1, heights=[0.99, 0.01]),
                                    plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

        if show_plots
            display(p)
        end

        if save_plots
            savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
        end
    end
end

plot_fevd(RBC_CME)



include("models/SW07.jl")
plot_fevd(m)









periods::Union{Vector{Int},Vector{Float64},UnitRange{Int64}} = 1:40
variables = :all
parameters = nothing
show_plots::Bool = true
save_plots::Bool = false
save_plots_format::Symbol = :pdf
save_plots_path::String = "."
plots_per_page::Int = 9
verbose = false

fevds = get_conditional_variance_decomposition(𝓂,
                                                periods = periods,
                                                parameters = parameters,
                                                verbose = verbose)

var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

default(size=(700,500),
        plot_titlefont = (10), 
        titlefont = (10), 
        guidefont = (8), 
        legendfontsize = 8, 
        tickfontsize = 8,
        framestyle = :box)

vars_to_plot = axiskeys(fevds)[1][var_idx]

for k in vars_to_plot
    n_subplots = length(var_idx)
    pp = []
    pane = 1
    plot_count = 1

    for i in 1:length(var_idx)
        if !(plot_count % plots_per_page == 0)
            plot_count += 1
                push!(pp,push!(pp,groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, legend = :none)))
        else
            plot_count = 1

            push!(pp,push!(pp,groupedbar(fevds(k,:,:)', title = string(k), bar_position = :stack, legend = :none)))
            
            ppp = Plots.plot(pp...)

            p = Plots.plot(ppp,Plots.bar(fill(0,1,length(shocks_to_plot)), 
                                        label = reshape(string.(shocks_to_plot),1,length(shocks_to_plot)), 
                                        linewidth = 0 , 
                                        framestyle = :none, 
                                        legend = :inside, 
                                        legend_columns = -1), 
                                        layout = grid(2, 1, heights=[0.99, 0.01]),
                                        plot_title = "Model: "*𝓂.model_name*"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")

            if show_plots
                display(p)
            end

            if save_plots
                savefig(p, save_plots_path * "/fevd__" * 𝓂.model_name * "__" * string(pane) * "." * string(save_plots_format))
            end

            pane += 1
            pp = []
        end
    end

    if length(pp) > 0
        println("Hi")
    end
end





push!(pp,Plots.bar(fill(0,1,2), label = reshape(string.(axiskeys(ff)[2]),1,2),  linewidth = 0 , framestyle = :none, legend = :inside))
Plots.plot(pp..., layout = @layout([grid(3,3);h]))
Plots.plot(pp..., layout = @layout([° ° °; ° ° °; ° _ _; °]))


get_conditional_variance_decomposition(RBC_CME)


@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;


get_moments(RBC)

var(RBC)

autocorr = get_autocorrelation(RBC)

get_correlation(RBC)

get_variance_decomposition(RBC)


include("models/FS2000.jl")

get_autocorrelation(m, parameters = (:alp => 0.33,:bet => 0.99,:gam => 0.003,:mst => 1.011,:rho => 0.7,:psi => 0.787,:del => 0.02, :z_e_a => 0.014,
:z_e_m => 0.005))

get_correlation(m)

get_variance_decomposition(m)

import LinearAlgebra as ℒ
𝓂 = m
subset_indices = collect(1:𝓂.timings.nVars)
T = 𝓂.timings

verbose = false

SS_and_pars, solution_error = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose)
    
∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)

𝑺₁ = calculate_first_order_solution(∇₁; T = 𝓂.timings)

A = @views 𝑺₁[subset_indices,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(subset_indices)))[indexin(T.past_not_future_and_mixed_idx,subset_indices),:]
C = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+1:end]




variances_by_shock = [ℒ.diag(calculate_covariance_forward(sol[:,[1:𝓂.timings.nPast_not_future_and_mixed..., 𝓂.timings.nPast_not_future_and_mixed+i]], T = 𝓂.timings, subset_indices = collect(1:𝓂.timings.nVars))) for i in 1:𝓂.timings.nExo]

variances_by_shock = [ℒ.diag(calculate_covariance_forward(sol[:,[1:𝓂.timings.nPast_not_future_and_mixed..., 𝓂.timings.nPast_not_future_and_mixed+i]], T = 𝓂.timings, subset_indices = collect(1:𝓂.timings.nVars))) for i in 1:𝓂.timings.nExo]


using AxisKeys, LinearMaps
import IterativeSolvers as ℐ
using MacroModelling: ℳ

function get_conditional_variance_decomposition(𝓂::ℳ; 
    periods::Union{Vector{Int},Vector{Float64}} = [1:20...,Inf],
    parameters = nothing,  
    verbose = false)

    periods = sort(periods)
    maxperiods = Int(maximum(periods[isfinite.(periods)]))
    var_container = zeros(size(𝑺₁)[1],𝓂.timings.nExo,length(periods))

    for i in 1:𝓂.timings.nExo
        C = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+i]
        CC = C * C'
        var = zeros(size(C)[1],size(C)[1])
        for k in 1:maxperiods
            var = A * var * A' + CC
            if k ∈ periods
                var_container[:,i,indexin(k,periods)] = ℒ.diag(var)
            end
        end
        if Inf in periods
            lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

            var_container[:,i,indexin(Inf,periods)] = ℒ.diag(reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC)))
        end
    end

    cond_var_decomp = var_container ./ sum(var_container,dims=2)

    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    KeyedArray(cond_var_decomp[indexin(sort(var),sort([var; 𝓂.aux; 𝓂.exo_present])),:,:]; Variables = sort(var), Shocks = 𝓂.timings.exo, Periods = periods)
end

get_conditional_variance_decomposition(m, periods = [1:5...])

C1 = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+1]
CC1 = C1 * C1'

CC11 = A * CC1 * A' + CC1


C2 = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+2]
CC2 = C2 * C2'

CC22 = A * CC2 * A' + CC2

CC - CC1 - CC2 .|> abs |> maximum

ℒ.diag(CC11 ./ (CC11 + CC22))

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC))




get_variance(m)
get_covariance(m)
get_standard_deviation(m)



get_autocorrelation(m)

m.parameter_values


get_solution(m)

outt = get_moments(RBC, variance = true, standard_deviation = false, non_stochastic_steady_state = false)

reduce(hcat,autocorr)
outt[1]
get_solution(RBC)



include("models/FS2000.jl")

get_autocorrelation(m, parameters = (:alp => 0.33,:bet => 0.99,:gam => 0.003,:mst => 1.011,:rho => 0.7,:psi => 0.787,:del => 0.02, :z_e_a => 0.014,
:z_e_m => 0.005))


get_correlation(m)



using MacroModelling: timings
using LinearMaps, SparseArrays
import IterativeSolvers as ℐ
import ForwardDiff as ℱ
import LinearAlgebra as ℒ

function calculate_covariance_forward(𝑺₁::AbstractMatrix{<: Number}; T::timings)
    A = 𝑺₁[:,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]
    C = 𝑺₁[:,T.nPast_not_future_and_mixed+1:end]
    
    CC = C * C'

    lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))
    
    reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC))

    # covar_dcmp = sparse(ℒ.triu(reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC))))

    # droptol!(covar_dcmp,eps(Float64))

    # return covar_dcmp
end


function calculate_covariance_forward(𝑺₁::AbstractMatrix{ℱ.Dual{Z,S,N}}; T::timings = T) where {Z,S,N}
    # unpack: AoS -> SoA
    𝑺₁̂ = ℱ.value.(𝑺₁)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, 𝑺₁)'

    # get f(vs)
    val = calculate_covariance_forward(𝑺₁̂, T = T)

    # get J(f, vs) * ps (cheating). Write your custom rule here
    B = ℱ.jacobian(x -> calculate_covariance_conditions(x, val, T = T), 𝑺₁̂)
    A = ℱ.jacobian(x -> calculate_covariance_conditions(𝑺₁̂, x, T = T), val)

    jvp = (-A \ B) * ps

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val))
end


function calculate_covariance_conditions(𝑺₁::AbstractMatrix{<: Number}, covar::AbstractMatrix{<: Number}; T::timings)
    A = 𝑺₁[:,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]
    C = 𝑺₁[:,T.nPast_not_future_and_mixed+1:end]
    
    A * covar * A' + C * C' - covar
end




𝓂 = m
parameters = 𝓂.parameter_values
SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, false, true)
    
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)
for i in 1:𝓂.timings.nExo
size(sol)[2] - 𝓂.timings.nPast_not_future_and_mixed
first_shock = calculate_covariance_forward(sol[:,1:𝓂.timings.nPast_not_future_and_mixed+1],T = 𝓂.timings) |> ℒ.diag# .|> sqrt
second_shock = calculate_covariance_forward(sol[:,[1:𝓂.timings.nPast_not_future_and_mixed...,𝓂.timings.nPast_not_future_and_mixed+2]],T = 𝓂.timings) |> ℒ.diag# .|> sqrt
all_shocks = calculate_covariance_forward(sol,T = 𝓂.timings) |> ℒ.diag#.|> sqrt


isapprox(first_shock + second_shock, all_shocks,rtol = eps(Float32))


first_shock ./ (first_shock + second_shock)



ℱ.jacobian(parameters->begin
    SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, false, true)
        
    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    ℒ.diag(calculate_covariance_forward(sol, T = 𝓂.timings))
    end,
    Float64[𝓂.parameter_values...])

covar_dcmp = calculate_covariance_forward(sol, T = 𝓂.timings)


A = sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]
C = sol[:,𝓂.timings.nPast_not_future_and_mixed+1:end]


covar_dcmp = A * covar_dcmp * A' + C * C'

stds = sqrt.(ℒ.diag(covar_dcmp))
# Autocorr
# A ^ 20  * covar_dcmp ./ (stds * stds') |> ℒ.diag
A ^ 20  * covar_dcmp ./ ℒ.diag(covar_dcmp) |> ℒ.diag



include("/Users/thorekockerols/GitHub/MacroModelling.jl/test/models/SW07.jl")
outt = get_moments(m, variance = true, standard_deviation = false, non_stochastic_steady_state = false)
outt[1]
get_SS(m)
import LinearAlgebra as ℒ
𝓂 = m
parameters = m.parameter_values
SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, false, true)
    
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

A = sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]
C = sol[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

using BenchmarkTools
@benchmark covar_dcmp = ((reshape((ℒ.I - ℒ.kron(A, conj(A))) \ reshape(C * C', prod(size(A)), 1), size(A))))

using SparseArrays, LinearMaps
import IterativeSolvers as ℐ
sA = sparse(A)
CC = C * C'
lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))
# lm2 = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)),size(CC)[1],size(CC)[2])

# lm = LinearMap{Float64}(x -> sA * reshape(x,size(CC)) * sA' + CC, length(CC))


# lm(vec(covar_dcmp))
@benchmark sol = reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC))
@benchmark sol = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))
sol = reshape(ℐ.bicgstabl(lm, (-CC)), size(CC))
lm(vec(sol)) + CC

lm(vec(zero(sol)))

covar_dcmp = A * covar_dcmp * A' + C * C'

stds = sqrt.(ℒ.diag(sol))
# Autocorr
ℒ.diag(A ^ 1  * sol ./ (stds * stds'))


sqrt.(ℒ.diag(covar_dcmp)) * sqrt.(ℒ.diag(covar_dcmp))'
