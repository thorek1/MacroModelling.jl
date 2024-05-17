
# import Pkg; Pkg.add(["Turing","CSV","DataFrames","AxisKeys","FiniteDifferences","Zygote","DynamicPPL","StatsPlots","BenchmarkTools"])

# import Pkg; Pkg.add(["Turing","CSV","DataFrames","AxisKeys","FiniteDifferences","Zygote","StatsPlots","BenchmarkTools"])
using MacroModelling
import Turing: NUTS, HMC, PG, IS, sample, logpdf, Truncated#, Normal, Beta, Gamma, InverseGamma,
using CSV, DataFrames, AxisKeys
import Zygote
import ForwardDiff
import ChainRulesCore: @ignore_derivatives, ignore_derivatives, rrule, NoTangent, @thunk
using Random
import FiniteDifferences
# import FastDifferentiation
import BenchmarkTools: @benchmark
import LinearAlgebra as ℒ
Random.seed!(1)
import Symbolics
import SparseArrays: sparse!, SparseMatrixCSC
import MacroModelling: get_and_check_observables, create_third_order_auxilliary_matrices, create_second_order_auxilliary_matrices, get_symbols, solve!, create_symbols_eqs!, remove_redundant_SS_vars!, check_bounds, write_functions_mapping!, get_relevant_steady_state_and_state_update, calculate_loglikelihood, get_initial_covariance, riccati_forward, match_pattern



# m = SGU_2003_debt_premium

include("../models/Ghironi_Melitz_2005.jl")
m = Ghironi_Melitz_2005


autocorr_3rd = get_autocorrelation(m, algorithm = :pruned_third_order)


include("../models/JQ_2012_RBC.jl")
m = JQ_2012_RBC

SS(m, algorithm = :pruned_second_order)
SS(m, algorithm = :pruned_third_order)


include("../models/RBC_baseline.jl")
m = RBC_baseline

# @profview include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")

# todos: use Symbolics function writer 
# use analytical derivatives for SS derivs output and the non_stochastic_steady_state function for that purpose
SS(Ghironi_Melitz_2005, algorithm = :pruned_second_order)
SS(m, algorithm = :pruned_third_order)

get_irf(m)

Zygote.jacobian(x -> get_non_stochastic_steady_state(m, x)[1], m.parameter_values)[1]

ForwardDiff.jacobian(x -> get_non_stochastic_steady_state(m, x)[1], m.parameter_values)

SS(m)

observables = [:y]

Random.seed!(1)
simulated_data = simulate(m)

get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

Zygote.jacobian(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)[1]

SSS(m)

include("../models/NAWM_EAUS_2008.jl")


SS(NAWM_EAUS_2008, algorithm = :pruned_second_order)
SS(NAWM_EAUS_2008, algorithm = :pruned_third_order)


m = NAWM_EAUS_2008

model = NAWM_EAUS_2008

observables = [:EA_R, :US_R, :EA_PI, :US_PI, :EA_YGAP, :US_YGAP]

Random.seed!(1)
simulated_data = simulate(model)

get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values)


Zygote.jacobian(x -> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)[1]

using BenchmarkTools
@benchmark get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values)
@benchmark Zygote.jacobian(x -> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)[1]
@profview get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values)



# symbolics = create_symbols_eqs!(NAWM_EAUS_2008)
# @profview symbolics = create_symbols_eqs!(NAWM_EAUS_2008)

# @profview remove_redundant_SS_vars!(NAWM_EAUS_2008, symbolics) 


@profview write_functions_mapping!(NAWM_EAUS_2008, 1)
# Remove redundant variables in non stochastic steady state problem:	10.531 seconds
# Set up non stochastic steady state problem:				8.669 seconds
# Take symbolic derivatives up to first order:				2.566 seconds
# Find non stochastic steady state:					13.728 seconds

# Remove redundant variables in non stochastic steady state problem:	12.436 seconds
# Set up non stochastic steady state problem:				12.699 seconds
# Take symbolic derivatives up to first order:				82.684 seconds
# Find non stochastic steady state:					16.254 seconds


𝓂 = m
max_perturbation_order = 1


future_varss  = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₁₎$")))
present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎$")))
past_varss    = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₋₁₎$")))
shock_varss   = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₓ₎$")))
ss_varss      = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₛₛ₎$")))

sort!(future_varss  ,by = x->replace(string(x),r"₍₁₎$"=>"")) #sort by name without time index because otherwise eps_zᴸ⁽⁻¹⁾₍₋₁₎ comes before eps_z₍₋₁₎
sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))
sort!(past_varss    ,by = x->replace(string(x),r"₍₋₁₎$"=>""))
sort!(shock_varss   ,by = x->replace(string(x),r"₍ₓ₎$"=>""))
sort!(ss_varss      ,by = x->replace(string(x),r"₍ₛₛ₎$"=>""))

steady_state = []
steady_state_no_time = []
for (i, var) in enumerate(ss_varss)
    push!(steady_state,:($var = X̄[$i]))
    push!(steady_state_no_time,:($(Symbol(replace(string(var),r"₍ₛₛ₎$"=>""))) = X̄[$i]))
    # ii += 1
end

ii = 1

alll = []
alll_no_time = []
for var in future_varss
    push!(alll,:($var = X[$ii]))
    push!(alll_no_time,:($(Symbol(replace(string(var), r"₍₁₎$"=>""))) = X[$ii])) # , r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""
    ii += 1
end

for var in present_varss
    push!(alll,:($var = X[$ii]))
    push!(alll_no_time,:($(Symbol(replace(string(var), r"₍₀₎$"=>""))) = X[$ii])) # , r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""
    ii += 1
end

for var in past_varss
    push!(alll,:($var = X[$ii]))
    push!(alll_no_time,:($(Symbol(replace(string(var), r"₍₋₁₎$"=>""))) = X[$ii])) # , r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""
    ii += 1
end

for var in shock_varss
    push!(alll,:($var = X[$ii]))
    # push!(alll_no_time,:($(Symbol(replace(string(var),r"₍ₛₛ₎$"=>""))) = X[$ii]))
    ii += 1
end



# paras = []
# push!(paras,:((;$(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)...)) = params))

paras = []
for (i, parss) in enumerate(vcat(𝓂.parameters,𝓂.calibration_equations_parameters))
    push!(paras,:($parss = params[$i]))
end

# # watch out with naming of parameters in model and functions
# mod_func2 = :(function model_function_uni_redux(X::Vector, params::Vector{Number}, X̄::Vector)
#     $(alll...)
#     $(paras...)
# 	$(𝓂.calibration_equations_no_var...)
#     $(steady_state...)
#     [$(𝓂.dyn_equations...)]
# end)


# 𝓂.model_function = @RuntimeGeneratedFunction(mod_func2)
# 𝓂.model_function = eval(mod_func2)

dyn_future_list = collect(reduce(union, 𝓂.dyn_future_list))
dyn_present_list = collect(reduce(union, 𝓂.dyn_present_list))
dyn_past_list = collect(reduce(union, 𝓂.dyn_past_list))
dyn_exo_list = collect(reduce(union,𝓂.dyn_exo_list))
dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.dyn_ss_list))) .* "₍ₛₛ₎")

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))

vars_raw = [dyn_future_list[indexin(sort(future),future)]...,
            dyn_present_list[indexin(sort(present),present)]...,
            dyn_past_list[indexin(sort(past),past)]...,
            dyn_exo_list[indexin(sort(exo),exo)]...]

Symbolics.@syms norminvcdf(x) norminv(x) qnorm(x) normlogpdf(x) normpdf(x) normcdf(x) pnorm(x) dnorm(x)

# overwrite SymPyCall names
eval(:(Symbolics.@variables $(reduce(union,get_symbols.(vcat(𝓂.dyn_equations, 𝓂.calibration_equations_no_var)))...)))

vars = eval(:(Symbolics.@variables $(vars_raw...)))

eqs = Symbolics.parse_expr_to_symbolic.(𝓂.dyn_equations,(@__MODULE__,))

# future_no_lead_lag = Symbol.(replace.(string.(future), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
# present_no_lead_lag = Symbol.(replace.(string.(present), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
# past_no_lead_lag = Symbol.(replace.(string.(past), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

# eval(:(Symbolics.@variables $(Set(vcat(future_no_lead_lag, present_no_lead_lag, past_no_lead_lag))...)))
eval(:(Symbolics.@variables $(Set(vcat(future, present, past, 𝓂.parameters))...)))

SS_and_pars = Symbol.(vcat(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), 𝓂.calibration_equations_parameters))

# remove time indices
# vars_no_time_transform = union(Dict(eval.(dyn_future_list) .=> eval.(future_no_lead_lag)), 
#                                 Dict(eval.(dyn_present_list) .=> eval.(present_no_lead_lag)), 
#                                 Dict(eval.(dyn_past_list) .=> eval.(past_no_lead_lag)),
#                                 Dict(eval.(dyn_exo_list) .=> 0))
vars_no_time_transform = union(Dict(eval.(dyn_future_list) .=> eval.(future)), 
                                Dict(eval.(dyn_present_list) .=> eval.(present)), 
                                Dict(eval.(dyn_past_list) .=> eval.(past)),
                                Dict(eval.(dyn_ss_list) .=> eval.(stst)),
                                Dict(eval.(dyn_exo_list) .=> 0))

                                max_perturbation_order = 3
if max_perturbation_order >= 2 
    nk = length(vars_raw)
    second_order_idxs = [nk * (i-1) + k for i in 1:nk for k in 1:i]
    if max_perturbation_order == 3
        third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]
    end
end

max_exprs_per_func = 80

calib_eqs = Dict([(eval(calib_eq.args[1]) => eval(calib_eq.args[2])) for calib_eq in reverse(𝓂.calibration_equations_no_var)])

eqs_sub = Symbolics.Num[]
for subst in eqs
    for calib_eq in calib_eqs
        subst = Symbolics.substitute(subst, calib_eq)
    end
    push!(eqs_sub, subst)
end

∂SS_equations_∂vars = Symbolics.sparsejacobian(eqs_sub, vars, simplify = false)# |> findnz

input_args = vcat(future_varss,
                    present_varss,
                    past_varss,
                    shock_varss,
                    𝓂.parameters,
                    𝓂.calibration_equations_parameters,
                    ss_varss)

funcs = Symbolics.build_function(∂SS_equations_∂vars, eval.(input_args), expression = false, parallel = Symbolics.MultithreadedForm(50, Threads.nthreads()))

𝓂.model_jacobian = funcs[1]
# for i in zip(∂SS_equations_∂vars...)
#     exx = :(function(X::Vector, params::Vector{Real}, X̄::Vector)
#     $(alll...)
#     $(paras...)
#     $(𝓂.calibration_equations_no_var...)
#     $(steady_state...)
#     return $(Symbolics.toexpr(i[3])), $(i[1]), $(i[2])
#     end)
#     push!(𝓂.model_jacobian, @RuntimeGeneratedFunction(exx))
# end

# mod_func3 = :(function model_jacobian(X::Vector, params::Vector{Real}, X̄::Vector)
#     $(alll...)
#     $(paras...)
#     $(𝓂.calibration_equations_no_var...)
#     $(steady_state...)
#     sparse(Int[$(∂SS_equations_∂vars[1]...)], Int[$(∂SS_equations_∂vars[2]...)], [$(Symbolics.toexpr.(∂SS_equations_∂vars[3])...)], $(length(eqs)), $(length(vars)))
# end)

# 𝓂.model_jacobian = @RuntimeGeneratedFunction(mod_func3)

# calib_eqs = Dict([(eval(calib_eq.args[1]) => eval(calib_eq.args[2])) for calib_eq in reverse(𝓂.calibration_equations_no_var)])

eqs_static = map(x -> Symbolics.substitute(x, vars_no_time_transform), findnz(∂SS_equations_∂vars)[3])

∂SS_equations_∂SS_and_pars = Symbolics.sparsejacobian(eqs_static, eval.(vcat(𝓂.parameters, SS_and_pars)), simplify = false) # |> findnz

idx_conversion = (findnz(∂SS_equations_∂vars)[1] + length(eqs) * (findnz(∂SS_equations_∂vars)[2] .- 1))

rows, cols, vals = findnz(∂SS_equations_∂SS_and_pars)

∂SS_equations_∂SS_and_pars_ext = sparse(cols, idx_conversion[rows], vals, (length(SS_and_pars) + length(𝓂.parameters)), (length(eqs) * length(vars)))

input_args_no_time = vcat(Symbol.(replace.(string.(future_varss), r"₍₁₎$"=>"")),
                    Symbol.(replace.(string.(present_varss), r"₍₀₎$"=>"")),
                    Symbol.(replace.(string.(past_varss), r"₍₋₁₎$"=>"")),
                    𝓂.parameters,
                    𝓂.calibration_equations_parameters,
                    Symbol.(replace.(string.(ss_varss),r"₍ₛₛ₎$"=>"")))


funcs = Symbolics.build_function(∂SS_equations_∂SS_and_pars_ext, eval.(input_args_no_time), expression = false)

𝓂.model_jacobian_SS_and_pars_vars = funcs[1]


# max_perturbation_order = 2

# hessian_rows = Int[]
# hessian_cols = Int[]
# hessian_vals = []

# for (i,eq) in enumerate(eqs_sub)
#     hessian = Symbolics.sparsehessian(eq, vars, simplify = false, full = true) |> findnz
#     if length(hessian[1]) > 0
#         push!(hessian_rows, fill(i, length(hessian[3]))...)
#         # push!(hessian_cols, indexin((hessian[1] .- 1) .* length(vars) .+ hessian[2], second_order_idxs)...)
#         push!(hessian_cols, ((hessian[1] .- 1) .* length(vars) .+ hessian[2])...)
#         push!(hessian_vals, hessian[3]...)
#     end
# end



# if max_perturbation_order >= 2
    
    hessian_rows = Int[]
    hessian_cols = Int[]
    hessian_vals = []

    for (i,eq) in enumerate(eqs_sub)
        hessian = Symbolics.sparsehessian(eq, vars, simplify = false, full = true) |> findnz

        push!(hessian_rows, fill(i, length(hessian[3]))...)
        # push!(hessian_cols, indexin((hessian[1] .- 1) .* length(vars) .+ hessian[2], second_order_idxs)...)
        push!(hessian_cols, ((hessian[1] .- 1) .* length(vars) .+ hessian[2])...)
        push!(hessian_vals, hessian[3]...)
    end

    # if max_perturbation_order == 3 # && 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
        third_order_derivatives = Symbolics.sparsejacobian(hessian_vals, vars) |> findnz

        third_order_rows = hessian_rows[third_order_derivatives[1]]
        third_order_cols = (hessian_cols[third_order_derivatives[1]] .- 1) .* length(vars) .+ third_order_derivatives[2]
        third_order_vals = third_order_derivatives[3]

        reducer³ = [i ∈ third_order_idxs for i in third_order_cols]

        reduced_third_order_rows = third_order_rows[reducer³]
        reduced_third_order_cols = Int.(indexin(third_order_cols[reducer³], third_order_idxs))
        reduced_third_order_vals = third_order_vals[reducer³]

        ∂SS_equations_∂vars_∂vars_∂vars = sparse!(reduced_third_order_rows, reduced_third_order_cols, reduced_third_order_vals, length(eqs), length(third_order_idxs))

        min_n_funcs = length(∂SS_equations_∂vars_∂vars_∂vars.nzval) ÷ max_exprs_per_func
        
        if min_n_funcs == 0
            parallel = Symbolics.SerialForm()
        # elseif min_n_funcs == 1
        #     parallel = Symbolics.ShardedForm(max_exprs_per_func, 2)
        else
            # parallel = Symbolics.MultithreadedForm(max_exprs_per_func, min_n_funcs)
            parallel = Symbolics.ShardedForm(max_exprs_per_func, min_n_funcs + 1)
        end
        
        funcs = Symbolics.build_function(∂SS_equations_∂vars_∂vars_∂vars, eval.(input_args), expression = false, parallel = parallel)

        𝓂.model_third_order_derivatives = funcs[1]

        if 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            𝓂.solution.perturbation.third_order_auxilliary_matrices = create_third_order_auxilliary_matrices(𝓂.timings, Int.(indexin(intersect(third_order_idxs, unique(third_order_cols)), third_order_idxs)))
        end
    # end
    
    
        reducer² = [i ∈ second_order_idxs for i in hessian_cols]
        
        reduced_hessian_rows = hessian_rows[reducer²]
        reduced_hessian_cols = Int.(indexin(hessian_cols[reducer²], second_order_idxs))
        reduced_hessian_vals = hessian_vals[reducer²]

        ∂SS_equations_∂vars_∂vars = sparse(reduced_hessian_rows, reduced_hessian_cols, reduced_hessian_vals, length(eqs), length(second_order_idxs))


    # ∂SS_equations_∂vars_∂vars = sparse(hessian_rows[hessian_cols .∈ second_order_idxs], hessian_cols[hessian_cols .∈ second_order_idxs], hessian_vals[hessian_cols .∈ second_order_idxs], length(eqs), length(second_order_idxs))

    input_args = vcat(future_varss,
                        present_varss,
                        past_varss,
                        shock_varss,
                        𝓂.parameters,
                        𝓂.calibration_equations_parameters,
                        ss_varss)
    
    min_n_funcs = length(∂SS_equations_∂vars_∂vars.nzval) ÷ max_exprs_per_func

    if min_n_funcs == 0
        parallel = Symbolics.SerialForm()
    # elseif min_n_funcs == 1
    #     parallel = Symbolics.ShardedForm(max_exprs_per_func, 2)
    else
        # parallel = Symbolics.MultithreadedForm(max_exprs_per_func, min_n_funcs)
        parallel = Symbolics.ShardedForm(max_exprs_per_func, min_n_funcs + 1)
    end
    
    funcs = Symbolics.build_function(∂SS_equations_∂vars_∂vars, eval.(input_args), expression = false, parallel = parallel)

    𝓂.model_hessian = funcs[1];






parameters = 𝓂.parameter_values
SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, false, 𝓂.solver_parameters)

SSSS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
# par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))
par = vcat(parameters,calibrated_parameters)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

𝓂.model_hessian([SSSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SSSS[dyn_ss_idx]])

@benchmark 𝓂.model_hessian([SSSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SSSS[dyn_ss_idx]])

𝓂.model_third_order_derivatives([SSSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SSSS[dyn_ss_idx]])





# import SparseArrays
∂SS_equations_∂vars_∂vars = SparseArrays.sparse!(hessian_rows, hessian_cols, hessian_vals, length(eqs), length(vars)^2)

# ∂SS_equations_∂vars_∂vars * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂


input_args = vcat(future_varss,
                    present_varss,
                    past_varss,
                    shock_varss,
                    𝓂.parameters,
                    𝓂.calibration_equations_parameters,
                    ss_varss)

funcs = Symbolics.build_function(∂SS_equations_∂vars_∂vars, eval.(input_args), expression = false, parallel = Symbolics.MultithreadedForm(50, Threads.nthreads()));

funcs = Symbolics.build_function(∂SS_equations_∂vars_∂vars, eval.(input_args), expression = false, parallel = Symbolics.ShardedForm(50,4));

𝓂.model_hessian = funcs[1];



parameters = 𝓂.parameter_values
SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, false, 𝓂.solver_parameters)


SSSS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
# par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))
par = vcat(parameters,calibrated_parameters)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

𝓂.model_hessian([SSSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SSSS[dyn_ss_idx]])

@benchmark 𝓂.model_hessian([SSSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SSSS[dyn_ss_idx]])

𝓂.model_third_order_derivatives([SSSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SSSS[dyn_ss_idx]])



funcs = Symbolics.build_function(∂SS_equations_∂vars_∂vars, eval.(input_args), expression = false, parallel = Symbolics.MultithreadedForm(50, Threads.nthreads()));

max_exprs_per_func = 200
funcs = Symbolics.build_function(∂SS_equations_∂vars_∂vars, eval.(input_args), expression = false, parallel = Symbolics.MultithreadedForm(max_exprs_per_func, length(∂SS_equations_∂vars_∂vars.nzval) ÷ max_exprs_per_func));

# funcs = Symbolics.build_function(∂SS_equations_∂vars_∂vars, eval.(input_args), expression = false, parallel = Symbolics.ShardedForm(50,2*Threads.nthreads()));

𝓂.model_hessian = funcs[1];


@benchmark funcs[1]([SSSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SSSS[dyn_ss_idx]])

@profview funcs[1]([SSSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SSSS[dyn_ss_idx]])



# third order derivs
third_order_derivatives = Symbolics.sparsejacobian(∂SS_equations_∂vars_∂vars.nzval, vars) |> findnz

hessian = ∂SS_equations_∂vars_∂vars |> findnz
third_order_rows = hessian[1][third_order_derivatives[1]]
third_order_cols = (hessian[2][third_order_derivatives[1]] .- 1) .* length(vars) .+ third_order_derivatives[2]
third_order_vals = third_order_derivatives[3]


hessian_cols == hessian[1]

hessian_rows == hessian[2]

third_order_derivatives = Symbolics.sparsejacobian(hessian_vals, vars) |> findnz

third_order_rows = hessian_cols[third_order_derivatives[1]]
third_order_cols = (hessian_rows[third_order_derivatives[1]] .- 1) .* length(vars) .+ third_order_derivatives[2]
third_order_vals = third_order_derivatives[3]


∂SS_equations_∂vars_∂vars_∂vars = SparseArrays.sparse!(third_order_rows, third_order_cols, third_order_vals, length(eqs), length(vars)^3)


sort(findnz(∂SS_equations_∂vars_∂vars_∂vars)[1]) == sort(findnz(∇₃)[1])
sort(findnz(∂SS_equations_∂vars_∂vars_∂vars)[2]) == sort(findnz(∇₃)[2])
∇₃ = calculate_third_order_derivatives(𝓂.parameter_values, SS_and_pars, 𝓂)

unique(third_order_cols)
Int.(indexin(intersect(third_order_idxs,unique(third_order_cols)),third_order_idxs))

unique((indexin(third_order_cols, third_order_idxs)))
import MacroModelling: create_third_order_auxilliary_matrices
create_third_order_auxilliary_matrices(𝓂.timings, Int.(indexin(intersect(third_order_idxs,unique(third_order_cols)),third_order_idxs)))
    

third_order_rows = Int[]
third_order_cols = Int[]
third_order_vals = []

for (i,eq) in enumerate(eqs_sub)
    third_order = Symbolics.sparsehessian(eq, vars, simplify = false, full = true) |> findnz

    push!(third_order_rows, fill(i, length(hessian[3]))...)
    # push!(third_order_cols, indexin((hessian[1] .- 1) .* length(vars) .+ hessian[2], second_order_idxs)...)
    push!(third_order_cols, ((hessian[1] .- 1) .* length(vars) .+ hessian[2])...)
    push!(third_order_vals, hessian[3]...)
end

bb =sparse(row2, column2, second_order, length(eqs), length(second_order_idxs))



aa - bb
aa.nzval
bb.nzval
# if max_perturbation_order >= 2 
    first_order = []
    first_order_parameter = []
    first_order_SS_and_pars_var = []
    second_order = []
    third_order = []
    row1 = Int[]
    row1p = Int[]
    row1SSp = Int[]
    row2 = Int[]
    row3 = Int[]
    column1 = Int[]
    column1p = Int[]
    column1SSp = Int[]
    column2 = Int[]
    column3 = Int[]
    # column3ext = Int[]
    i1 = 1
    i1p = 1
    i1SSp = 1
    i2 = 1
    i3 = 1



    for (c1,var1) in enumerate(vars)
        for (r,eq) in enumerate(eqs)
            if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
                deriv_first = Symbolics.derivative(eq,var1)
    
                deriv_first_subst = copy(deriv_first)
    
                # substitute in calibration equations without targets
                for calib_eq in reverse(𝓂.calibration_equations_no_var)
                    deriv_first_subst = Symbolics.substitute(deriv_first_subst, Dict(eval(calib_eq.args[1]) => eval(calib_eq.args[2])))
                end
                    i1 += 1
                    if max_perturbation_order >= 2 
                        for (c2,var2) in enumerate(vars)
                            # if Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first))
                            if (((c1 - 1) * length(vars) + c2) ∈ second_order_idxs) && (Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first)))
                                deriv_second = Symbolics.derivative(deriv_first,var2)
                                # if deriv_second != 0 
                                #     deriv_expr = Meta.parse(string(deriv_second.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
                                #     push!(second_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                                    # push!(second_order,Symbolics.toexpr(deriv_second))
                                    push!(second_order,(deriv_second))
                                    push!(row2,r)
                                    # push!(column2,(c1 - 1) * length(vars) + c2)
                                    push!(column2, Int.(indexin([(c1 - 1) * length(vars) + c2], second_order_idxs))...)
                                    i2 += 1
                                    if max_perturbation_order == 3
                                        for (c3,var3) in enumerate(vars)
                                            # if Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second))
                                                # push!(column3ext,(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3)
                                                if (((c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3) ∈ third_order_idxs) && (Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second)))
                                                    deriv_third = Symbolics.derivative(deriv_second,var3)
                                                    # if deriv_third != 0 
                                                    #     deriv_expr = Meta.parse(string(deriv_third.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
                                                    #     push!(third_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                                                        push!(third_order,Symbolics.toexpr(deriv_third))
                                                        push!(row3,r)
                                                        # push!(column3,(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3)
                                                        push!(column3, Int.(indexin([(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3], third_order_idxs))...)
                                                        i3 += 1
                                                    # end
                                                end
                                            # end
                                        end
                                    end
                                # end
                            end
                        end
                    end
                # end
            end
        end
    end
# end



sparse(row2, column2, second_order, length(eqs), length(second_order_idxs))




input_vals = [SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss; par; SS[dyn_ss_idx]]
tmp = funcs[1](input_vals)
findnz(tmp)
funcs[2](tmp, input_vals)
findnz(tmp)

jacc ≈ tmp
parameters = 𝓂.parameter_values
SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, false, 𝓂.solver_parameters)



SSS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]
# par = ComponentVector(vcat(parameters,calibrated_parameters),Axis(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)))
par = vcat(parameters,calibrated_parameters)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

# return 𝒜.jacobian(𝒷(), x -> 𝓂.model_function(x, par, SS), [SS_future; SS_present; SS_past; shocks_ss])#, SS_and_pars
# return Matrix(𝓂.model_jacobian(([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx])))
jacc = 𝓂.model_jacobian([SSS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SSS[dyn_ss_idx])







# for i in zip(∂SS_equations_∂vars...)
#     exx = :(function(X::Vector, params::Vector{Real}, X̄::Vector)
#     $(alll...)
#     $(paras...)
#     $(𝓂.calibration_equations_no_var...)
#     $(steady_state...)
#     return $(Symbolics.toexpr(i[3])), $(i[1]), $(i[2])
#     end)
#     push!(𝓂.model_jacobian, @RuntimeGeneratedFunction(exx))
# end

mod_func3 = :(function model_jacobian(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll...)
    $(paras...)
    $(𝓂.calibration_equations_no_var...)
    $(steady_state...)
    sparse(Int[$(∂SS_equations_∂vars[1]...)], Int[$(∂SS_equations_∂vars[2]...)], [$(Symbolics.toexpr.(∂SS_equations_∂vars[3])...)], $(length(eqs)), $(length(vars)))
end)

# @benchmark begin
# eval(:(FastDifferentiation.@variables $(vars_raw...)))

# eval(:(FastDifferentiation.@variables $(reduce(union,get_symbols.(vcat(𝓂.dyn_equations, 𝓂.calibration_equations_no_var, 𝓂.calibration_equations)))...)))

# ∂SS_equations_∂vars = FastDifferentiation.sparse_jacobian(eval.(𝓂.dyn_equations), eval.(vars_raw)) |> findnz
# end

# FastDifferentiation.make_Expr(∂SS_equations_∂vars, eval.(vcat(vars_raw,reduce(union,get_symbols.(vcat(𝓂.dyn_equations, 𝓂.calibration_equations_no_var, 𝓂.calibration_equations)))...)),false,false)

# @benchmark begin
vars = eval(:(Symbolics.@variables $(vars_raw...)))

eval(:(Symbolics.@variables $(reduce(union,get_symbols.(vcat(𝓂.dyn_equations, 𝓂.calibration_equations_no_var, 𝓂.calibration_equations)))...)))

∂SS_equations_∂vars = Symbolics.sparsejacobian(eqs, vars, simplify = true)# |> findnz
# end

Symbolics.build_function((∂SS_equations_∂vars), vars)# ,parallel = Symbolics.MultithreadedForm())

model_jacobian = []
for i in zip(∂SS_equations_∂vars...)
    exx = :(function(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll...)
    $(paras...)
    $(𝓂.calibration_equations_no_var...)
    $(steady_state...)
    return $(Symbolics.toexpr(i[3])), $(i[2]), $(i[1])
    end)
    # push!(𝓂.model_jacobian, @RuntimeGeneratedFunction(exx))
    push!(model_jacobian, (exx))
end

 
calib_eqs = [(eval(calib_eq.args[1]) => eval(calib_eq.args[2])) for calib_eq in reverse(𝓂.calibration_equations_no_var)]

calib_eqs = Dict([(eval(calib_eq.args[1]) => eval(calib_eq.args[2])) for calib_eq in reverse(𝓂.calibration_equations_no_var)])

@benchmark  begin
eqs_static = Symbolics.Num[]
for sse in ∂SS_equations_∂vars[3]
    subst = sse
    # for calib_eq in calib_eqs
    #     subst = Symbolics.substitute(subst, Dict(eval(calib_eq.args[1]) => eval(calib_eq.args[2])))
    # end
    # subst = Symbolics.fast_substitute(subst, vars_no_time_transform_pair)
    subst = Symbolics.substitute(subst, vars_no_time_transform)
    # subst = Symbolics.simplify(subst)
    push!(eqs_static,subst)
end
end

∂SS_equations_∂pars = Symbolics.sparsejacobian(eqs_static, eval.(𝓂.parameters), simplify = true) |> findnz

∂SS_equations_∂SS_and_pars = Symbolics.sparsejacobian(eqs_static, eval.(SS_and_pars), simplify = true) |> findnz

idx_conversion = (∂SS_equations_∂vars[1] + length(eqs) * (∂SS_equations_∂vars[2] .- 1))

model_jacobian_parameters = []
for i in zip(∂SS_equations_∂pars...)
    exx = :(function(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll...)
    $(paras...)
    # $(𝓂.calibration_equations_no_var...)
    $(steady_state...)
    return $(Symbolics.toexpr(i[3])), $(i[2]), $(idx_conversion[i[1]])
    end)
    # push!(𝓂.model_jacobian, @RuntimeGeneratedFunction(exx))
    push!(model_jacobian_parameters, (exx))
end



model_jacobian_SS_and_pars_vars = []
for i in zip(∂SS_equations_∂SS_and_pars...)
    exx = :(function(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll...)
    $(paras...)
    # $(𝓂.calibration_equations_no_var...)
    $(steady_state...)
    return $(Symbolics.toexpr(i[3])), $(i[2]), $(idx_conversion[i[1]])
    end)
    # push!(𝓂.model_jacobian, @RuntimeGeneratedFunction(exx))
    push!(model_jacobian_SS_and_pars_vars, (exx))
end



mod_func3p = :(function model_jacobian_parameters(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll_no_time...)
    $(paras...)
    # $(𝓂.calibration_equations_no_var...)
    $(steady_state_no_time...)
    sparse(Int[$(column1p...)], Int[$(row1p...)], Float64[$(first_order_parameter...)], $(length(𝓂.parameters)), $(length(eqs) * length(vars)))
end)

𝓂.model_jacobian_parameters = @RuntimeGeneratedFunction(mod_func3p)


mod_func3SSp = :(function model_jacobian_SS_and_pars_vars(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll_no_time...)
    $(paras...)
    # $(𝓂.calibration_equations_no_var...)
    $(steady_state_no_time...)
    sparse(Int[$(column1SSp...)], Int[$(row1SSp...)], Float64[$(first_order_SS_and_pars_var...)], $(length(SS_and_pars)), $(length(eqs) * length(vars)))
end)

𝓂.model_jacobian_SS_and_pars_vars = @RuntimeGeneratedFunction(mod_func3SSp)


# ∂SS_equations_∂vars_∂vars = Symbolics.sparsehessian(eqs[1], vars, simplify = true, full = false) |> findnz


if max_perturbation_order >= 1
    first_order = []
    first_order_parameter = []
    first_order_SS_and_pars_var = []
    second_order = []
    third_order = []
    row1 = Int[]
    row1p = Int[]
    row1SSp = Int[]
    row2 = Int[]
    row3 = Int[]
    column1 = Int[]
    column1p = Int[]
    column1SSp = Int[]
    column2 = Int[]
    column3 = Int[]
    # column3ext = Int[]
    i1 = 1
    i1p = 1
    i1SSp = 1
    i2 = 1
    i3 = 1



    for (c1,var1) in enumerate(vars)
        for (r,eq) in enumerate(eqs)
            if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
                deriv_first = Symbolics.derivative(eq,var1)

                deriv_first_subst = copy(deriv_first)

                # substitute in calibration equations without targets
                for calib_eq in reverse(𝓂.calibration_equations_no_var)
                    deriv_first_subst = Symbolics.substitute(deriv_first_subst, Dict(eval(calib_eq.args[1]) => eval(calib_eq.args[2])))
                end

                for (p1,p) in enumerate(𝓂.parameters)
                    if Symbol(p) ∈ Symbol.(Symbolics.get_variables(deriv_first_subst))
                        deriv_first_no_time = Symbolics.substitute(deriv_first_subst, vars_no_time_transform)

                        deriv_first_parameters = Symbolics.derivative(deriv_first_no_time, eval(p))

                        deriv_first_parameters_expr = Symbolics.toexpr(deriv_first_parameters)

                        push!(first_order_parameter, deriv_first_parameters_expr)
                        push!(row1p, r + length(eqs) * (c1 - 1))
                        push!(column1p, p1)

                        i1p += 1
                    end
                end

                for (SSp1,SSp) in enumerate(SS_and_pars)
                    deriv_first_no_time = Symbolics.substitute(deriv_first_subst, vars_no_time_transform)
                    
                    if Symbol(SSp) ∈ Symbol.(Symbolics.get_variables(deriv_first_no_time))
                        deriv_first_SS_and_pars_var = Symbolics.derivative(deriv_first_no_time, eval(SSp))

                        deriv_first_SS_and_pars_var_expr = Symbolics.toexpr(deriv_first_SS_and_pars_var)

                        push!(first_order_SS_and_pars_var, deriv_first_SS_and_pars_var_expr)
                        push!(row1SSp, r + length(eqs) * (c1 - 1))
                        push!(column1SSp, SSp1)

                        i1SSp += 1
                    end
                end
                
                # if deriv_first != 0 
                #     deriv_expr = Meta.parse(string(deriv_first.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
                #     push!(first_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                    deriv_first_expr = Symbolics.toexpr(deriv_first)
                    # deriv_first_expr_safe = postwalk(x -> x isa Expr ? 
                    #                                     x.args[1] == :^ ? 
                    #                                         :(NaNMath.pow($(x.args[2:end]...))) : 
                    #                                     x : 
                    #                                 x, 
                    #                         deriv_first_expr)

                    push!(first_order, deriv_first_expr)
                    push!(row1,r)
                    push!(column1,c1)
                    i1 += 1
                    if max_perturbation_order >= 2 
                        for (c2,var2) in enumerate(vars)
                            # if Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first))
                            if (((c1 - 1) * length(vars) + c2) ∈ second_order_idxs) && (Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first)))
                                deriv_second = Symbolics.derivative(deriv_first,var2)
                                # if deriv_second != 0 
                                #     deriv_expr = Meta.parse(string(deriv_second.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
                                #     push!(second_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                                    push!(second_order,Symbolics.toexpr(deriv_second))
                                    push!(row2,r)
                                    # push!(column2,(c1 - 1) * length(vars) + c2)
                                    push!(column2, Int.(indexin([(c1 - 1) * length(vars) + c2], second_order_idxs))...)
                                    i2 += 1
                                    if max_perturbation_order == 3
                                        for (c3,var3) in enumerate(vars)
                                            # if Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second))
                                                # push!(column3ext,(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3)
                                                if (((c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3) ∈ third_order_idxs) && (Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second)))
                                                    deriv_third = Symbolics.derivative(deriv_second,var3)
                                                    # if deriv_third != 0 
                                                    #     deriv_expr = Meta.parse(string(deriv_third.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
                                                    #     push!(third_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                                                        push!(third_order,Symbolics.toexpr(deriv_third))
                                                        push!(row3,r)
                                                        # push!(column3,(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3)
                                                        push!(column3, Int.(indexin([(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3], third_order_idxs))...)
                                                        i3 += 1
                                                    # end
                                                end
                                            # end
                                        end
                                    end
                                # end
                            end
                        end
                    end
                # end
            end
        end
    end
end
    

mod_func3 = :(function model_jacobian(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll...)
    $(paras...)
    $(𝓂.calibration_equations_no_var...)
    $(steady_state...)
    sparse(Int[$(row1...)], Int[$(column1...)], [$(first_order...)], $(length(eqs)), $(length(vars)))
end)

𝓂.model_jacobian = @RuntimeGeneratedFunction(mod_func3)


# for (l,first) in enumerate(first_order)
#     exx = :(function(X::Vector, params::Vector{Real}, X̄::Vector)
#     $(alll...)
#     $(paras...)
#     $(𝓂.calibration_equations_no_var...)
#     $(steady_state...)
#     return $first, $(row1[l]), $(column1[l])
#     end)
#     push!(𝓂.model_jacobian,@RuntimeGeneratedFunction(exx))
# end

# 𝓂.model_jacobian = FWrap{Tuple{Vector{Float64}, Vector{Number}, Vector{Float64}}, SparseMatrixCSC{Float64}}(@RuntimeGeneratedFunction(mod_func3))

# 𝓂.model_jacobian = eval(mod_func3)

mod_func3p = :(function model_jacobian_parameters(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll_no_time...)
    $(paras...)
    # $(𝓂.calibration_equations_no_var...)
    $(steady_state_no_time...)
    sparse(Int[$(column1p...)], Int[$(row1p...)], Float64[$(first_order_parameter...)], $(length(𝓂.parameters)), $(length(eqs) * length(vars)))
end)

𝓂.model_jacobian_parameters = @RuntimeGeneratedFunction(mod_func3p)


mod_func3SSp = :(function model_jacobian_SS_and_pars_vars(X::Vector, params::Vector{Real}, X̄::Vector)
    $(alll_no_time...)
    $(paras...)
    # $(𝓂.calibration_equations_no_var...)
    $(steady_state_no_time...)
    sparse(Int[$(column1SSp...)], Int[$(row1SSp...)], Float64[$(first_order_SS_and_pars_var...)], $(length(SS_and_pars)), $(length(eqs) * length(vars)))
end)

𝓂.model_jacobian_SS_and_pars_vars = @RuntimeGeneratedFunction(mod_func3SSp)






SS_and_pars, _ = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, false, 𝓂.solver_parameters)

TT, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(:first_order), x, 𝓂, 1e-12)



SS_and_pars, (solution_error, iters) = get_non_stochastic_steady_state(𝓂, 𝓂.parameter_values)

state = zeros(𝓂.timings.nVars)

TT = 𝓂.timings

sp∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix

∇₁ = Matrix(sp∇₁)

𝐒₁, solved = calculate_first_order_solution(∇₁; T = TT)


SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)

SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_equations)),union(𝓂.parameters_in_equations))))

unknowns = union(vars_in_ss_equations, 𝓂.calibration_equations_parameters)

∂SS_equations_∂parameters = 𝓂.∂SS_equations_∂parameters(𝓂.parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
∂SS_equations_∂SS_and_pars = 𝓂.∂SS_equations_∂SS_and_pars(𝓂.parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix

JVP = -(∂SS_equations_∂SS_and_pars \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]
jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters), )

for (i,v) in enumerate(SS_and_pars_names)
    if v in unknowns
        jvp[i,:] = JVP[indexin([v], unknowns),:]
    end
end


parameter_values = 𝓂.parameter_values

parameter_values = parameters_combined
SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(parameter_values, 𝓂, false, false, 𝓂.solver_parameters)

SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
    
SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

# vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_equations)),union(𝓂.parameters_in_equations))))

unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)

∂SS_equations_∂parameters = 𝓂.∂SS_equations_∂parameters(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
∂SS_equations_∂SS_and_pars = 𝓂.∂SS_equations_∂SS_and_pars(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix

JVP = -(∂SS_equations_∂SS_and_pars \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]
jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))

for (i,v) in enumerate(SS_and_pars_names)
    if v in unknowns
        jvp[i,:] = JVP[indexin([v], unknowns),:]
    end
end

jvp' * ∂SS_and_pars


vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_equations)),union(𝓂.parameters_in_equations))))

unknowns = union(vars_in_ss_equations, 𝓂.calibration_equations_parameters)

ss_equations = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

Symbolics.@syms norminvcdf(x) norminv(x) qnorm(x) normlogpdf(x) normpdf(x) normcdf(x) pnorm(x) dnorm(x)

# overwrite SymPyCall names
eval(:(Symbolics.@variables $(setdiff(union(𝓂.parameters_in_equations, 𝓂.parameters_as_function_of_parameters), 𝓂.parameters)...)))

vars = eval(:(Symbolics.@variables $(unknowns...)))

pars = eval(:(Symbolics.@variables $(𝓂.parameters...)))

eqs = Symbolics.parse_expr_to_symbolic.(ss_equations,(@__MODULE__,))


eqs = Symbolics.Num[]
for sse in ss_equations
    subst = Symbolics.parse_expr_to_symbolic.([sse],(@__MODULE__,))[1]
    for calib_eq in reverse(𝓂.calibration_equations_no_var)
        subst = Symbolics.substitute(subst, Dict(eval(calib_eq.args[1]) => eval(calib_eq.args[2])))
    end
    push!(eqs,subst)
end

@benchmark begin
    ∂SS_equations_∂parameters = Symbolics.sparsejacobian(eqs,pars)

    ∂SS_equations_∂SS_and_pars = Symbolics.sparsejacobian(eqs,vars)
end

∂SS_equations_∂parameters = Symbolics.sparsejacobian(eqs,pars) |> findnz
∂SS_equations_∂parameters[3] .|> Symbolics.toexpr
first_order_SS_and_pars_var|>findnz

@benchmark begin
    
first_order_parameter = []
first_order_SS_and_pars_var = []
row1p = Int[]
row1SSp = Int[]
column1p = Int[]
column1SSp = Int[]
i1p = 1
i1SSp = 1

for (r,eq) in enumerate(eqs)
    for (c1,var1) in enumerate(vars)
        if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
            deriv_first = Symbolics.derivative(eq,var1)

            deriv_first_subst = copy(deriv_first)

            # substitute in calibration equations without targets
            for calib_eq in reverse(𝓂.calibration_equations_no_var)
                deriv_first_subst = Symbolics.substitute(deriv_first_subst, Dict(eval(calib_eq.args[1]) => eval(calib_eq.args[2])))
            end
            
            # if deriv_first != 0 
            #     deriv_expr = Meta.parse(string(deriv_first.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
            #     push!(first_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                deriv_first_expr = Symbolics.toexpr(deriv_first_subst)
                # deriv_first_expr_safe = postwalk(x -> x isa Expr ? 
                #                                     x.args[1] == :^ ? 
                #                                         :(NaNMath.pow($(x.args[2:end]...))) : 
                #                                     x : 
                #                                 x, 
                #                         deriv_first_expr)

                # push!(first_order_SS_and_pars_var, deriv_first_subst)
                push!(first_order_SS_and_pars_var, deriv_first_expr)
                push!(row1SSp,r)
                push!(column1SSp,c1)
                i1SSp += 1
            # end
        end
    end

    for (p1,par1) in enumerate(pars)
        if Symbol(par1) ∈ Symbol.(Symbolics.get_variables(eq))
            deriv_first = Symbolics.derivative(eq,par1)

            deriv_first_subst = copy(deriv_first)

            # substitute in calibration equations without targets
            for calib_eq in reverse(𝓂.calibration_equations_no_var)
                deriv_first_subst = Symbolics.substitute(deriv_first_subst, Dict(eval(calib_eq.args[1]) => eval(calib_eq.args[2])))
            end
            
            # if deriv_first != 0 
            #     deriv_expr = Meta.parse(string(deriv_first.subs(SPyPyC.PI,SPyPyC.N(SPyPyC.PI))))
            #     push!(first_order, :($(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, deriv_expr))))
                deriv_first_expr = Symbolics.toexpr(deriv_first_subst)
                # deriv_first_expr_safe = postwalk(x -> x isa Expr ? 
                #                                     x.args[1] == :^ ? 
                #                                         :(NaNMath.pow($(x.args[2:end]...))) : 
                #                                     x : 
                #                                 x, 
                #                         deriv_first_expr)

                # push!(first_order_parameter, deriv_first_subst)
                push!(first_order_parameter, deriv_first_expr)
                push!(row1p,r)
                push!(column1p,p1)
                i1p += 1
            # end
        end
    end
end
end

pars = []
for (i, p) in enumerate(𝓂.parameters)
    push!(pars, :($p = parameters[$i]))
end

unknwns = []
for (i, u) in enumerate(union(vars_in_ss_equations, 𝓂.calibration_equations_parameters))
    push!(unknwns, :($u = unknowns[$i]))
end

∂SS_equations_∂parameters_exp = :(function calculate_∂SS_equations_∂parameters(parameters::Vector{Float64}, unknowns::Vector{Float64})
    $(pars...)
    # $(𝓂.calibration_equations_no_var...)
    $(unknwns...)
    sparse(Int[$(column1p...)], Int[$(row1p...)], Float64[$(first_order_parameter...)], $(length(𝓂.parameters)), $(length(eqs)))
end)


𝓂.∂SS_equations_∂parameters = @RuntimeGeneratedFunction(∂SS_equations_∂parameters_exp)



∂SS_equations_∂SS_and_pars_exp = :(function calculate_∂SS_equations_∂SS_and_pars(parameters::Vector{Float64}, unknowns::Vector{Float64})
    $(pars...)
    # $(𝓂.calibration_equations_no_var...)
    $(unknwns...)
    sparse(Int[$(column1SSp...)], Int[$(row1SSp...)], Float64[$(first_order_SS_and_pars_var...)], $(length(vars)), $(length(eqs)))
end)


𝓂.∂SS_equations_∂SS_and_pars = @RuntimeGeneratedFunction(∂SS_equations_∂SS_and_pars_exp)
