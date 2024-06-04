
# TODO: speedtesting multithreaded execution and toexpr vs string performance, and simplification
using Revise

using MacroModelling
using Zygote, Random, ForwardDiff
import Accessors
import Polyester
import Symbolics
import SparseArrays: SparseMatrixCSC
import LinearAlgebra as ℒ
import RecursiveFactorization as RF
import MacroModelling: get_non_stochastic_steady_state, get_symbols, write_derivatives_function, match_pattern, create_second_order_auxilliary_matrices, create_third_order_auxilliary_matrices
# include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
# include("models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl")

# include("models/RBC_CME.jl")

# model = m

# get_moments(model, parameter_derivatives = :alpha)

# observables = [:R]

# Random.seed!(1)
# simulated_data = simulate(model)

# # get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values)

# back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)[1]
# forw_grad = ForwardDiff.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)



# include("models/Backus_Kehoe_Kydland_1992.jl")
# model = Backus_Kehoe_Kydland_1992
# get_parameters(model)
# get_moments(model, parameter_derivatives = :K_ss)



include("../models/Smets_Wouters_2007.jl")
model = Smets_Wouters_2007

# include("../models/QUEST3_2009.jl")
# model = QUEST3_2009

# # SS(model)
get_mean(model, algorithm = :pruned_second_order)
# # @time SSS(model, algorithm = :pruned_second_order, parameters = :STD_EPS_ETA => 0.145)

SSS(model, algorithm = :pruned_third_order)

get_irf(model, algorithm = :pruned_third_order)

get_std(model, algorithm = :pruned_third_order)
model.model_third_order_derivatives[2]
# model.model_hessian[2]
# model.model_hessian[1]|>length

# include("../models/NAWM_EAUS_2008.jl")
# model = NAWM_EAUS_2008

# get_parameters(model, values = true)
# SSS(model, algorithm = :pruned_second_order)
# @time SSS(model, algorithm = :pruned_second_order, parameters = :σ_EA_R => .98)
# SSS(model, algorithm = :pruned_third_order)
# include("../models/Ghironi_Melitz_2005.jl")
# model = Ghironi_Melitz_2005

𝓂 = model
max_perturbation_order = 3
max_exprs_per_func = 1




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

dyn_future_list = collect(reduce(union, 𝓂.dyn_future_list))
dyn_present_list = collect(reduce(union, 𝓂.dyn_present_list))
dyn_past_list = collect(reduce(union, 𝓂.dyn_past_list))
dyn_exo_list = collect(reduce(union,𝓂.dyn_exo_list))
# dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.dyn_ss_list))) .* "₍ₛₛ₎")

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
# stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))

vars_raw = [dyn_future_list[indexin(sort(future),future)]...,
            dyn_present_list[indexin(sort(present),present)]...,
            dyn_past_list[indexin(sort(past),past)]...,
            dyn_exo_list[indexin(sort(exo),exo)]...]

Symbolics.@syms norminvcdf(x) norminv(x) qnorm(x) normlogpdf(x) normpdf(x) normcdf(x) pnorm(x) dnorm(x) erfc(x) erfcinv(x)

# overwrite SymPyCall names
input_args = vcat(future_varss,
                    present_varss,
                    past_varss,
                    ss_varss,
                    𝓂.parameters,
                    𝓂.calibration_equations_parameters,
                    shock_varss)

eval(:(Symbolics.@variables $(input_args...)))

Symbolics.@variables 𝔛[1:length(input_args)]

SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)

calib_eq_no_vars = reduce(union, get_symbols.(𝓂.calibration_equations_no_var), init = []) |> collect

eval(:(Symbolics.@variables $((vcat(SS_and_pars_names_lead_lag, calib_eq_no_vars))...)))

vars = eval(:(Symbolics.@variables $(vars_raw...)))

eqs = Symbolics.parse_expr_to_symbolic.(𝓂.dyn_equations,(@__MODULE__,))

final_indices = vcat(𝓂.parameters, SS_and_pars_names_lead_lag)

input_X = Pair{Symbolics.Num, Symbolics.Num}[]
input_X_no_time = Pair{Symbolics.Num, Symbolics.Num}[]

for (v,input) in enumerate(input_args)
    push!(input_X, eval(input) => eval(𝔛[v]))

    if input ∈ shock_varss
        push!(input_X_no_time, eval(𝔛[v]) => 0)
    else
        input_no_time = Symbol(replace(string(input), r"₍₁₎$"=>"", r"₍₀₎$"=>"" , r"₍₋₁₎$"=>"", r"₍ₛₛ₎$"=>"", r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

        vv = indexin([input_no_time], final_indices)
        
        if vv[1] isa Int
            push!(input_X_no_time, eval(𝔛[v]) => eval(𝔛[vv[1]]))
        end
    end
end

vars_X = map(x -> Symbolics.substitute(x, input_X), vars)

calib_eqs = Dict([(eval(calib_eq.args[1]) => eval(calib_eq.args[2])) for calib_eq in reverse(𝓂.calibration_equations_no_var)])

eqs_sub = Symbolics.Num[]
for subst in eqs
    for _ in calib_eqs
        for calib_eq in calib_eqs
            subst = Symbolics.substitute(subst, calib_eq)
        end
    end
    # subst = Symbolics.fixpoint_sub(subst, calib_eqs)
    subst = Symbolics.substitute(subst, input_X)
    subst = Symbolics.simplify(subst)
    push!(eqs_sub, subst)
end

if max_perturbation_order >= 2 
    nk = length(vars_raw)
    second_order_idxs = [nk * (i-1) + k for i in 1:nk for k in 1:i]
    if max_perturbation_order == 3
        third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]
    end
end

first_order = Symbolics.Num[]
second_order = Symbolics.Num[]
third_order = Symbolics.Num[]
row1 = Int[]
row2 = Int[]
row3 = Int[]
column1 = Int[]
column2 = Int[]
column3 = Int[]

# Polyester.@batch for rc1 in 0:length(vars_X) * length(eqs_sub) - 1
# for rc1 in 0:length(vars_X) * length(eqs_sub) - 1
for (c1, var1) in enumerate(vars_X)
    for (r, eq) in enumerate(eqs_sub)
    # r, c1 = divrem(rc1, length(vars_X)) .+ 1
    # var1 = vars_X[c1]
    # eq = eqs_sub[r]
        if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
            deriv_first = Symbolics.derivative(eq, var1)
            
            push!(first_order, deriv_first)
            push!(row1, r)
            # push!(row1, r...)
            push!(column1, c1)
            if max_perturbation_order >= 2 
                for (c2, var2) in enumerate(vars_X)
                    if (((c1 - 1) * length(vars) + c2) ∈ second_order_idxs) && (Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first)))
                        deriv_second = Symbolics.derivative(deriv_first, var2)
                        
                        push!(second_order, deriv_second)
                        push!(row2, r)
                        push!(column2, Int.(indexin([(c1 - 1) * length(vars) + c2], second_order_idxs))...)
                        if max_perturbation_order == 3
                            for (c3, var3) in enumerate(vars_X)
                                if (((c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3) ∈ third_order_idxs) && (Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second)))
                                    deriv_third = Symbolics.derivative(deriv_second,var3)

                                    push!(third_order, deriv_third)
                                    push!(row3, r)
                                    push!(column3, Int.(indexin([(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3], third_order_idxs))...)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
    

if max_perturbation_order >= 1
    if 𝓂.model_jacobian[2] == Int[]
        write_auxilliary_indices!(𝓂)

        write_derivatives_of_ss_equations!(𝓂::ℳ, max_exprs_per_func = max_exprs_per_func)

        # derivative of jacobian wrt SS_and_pars and parameters
        eqs_static = map(x -> Symbolics.substitute(x, input_X_no_time), first_order)

        ∂jacobian_∂SS_and_pars = Symbolics.sparsejacobian(eqs_static, eval.(𝔛[1:(length(final_indices))]), simplify = false) # |> findnz

        idx_conversion = (row1 + length(eqs) * (column1 .- 1))

        cols, rows, vals = findnz(∂jacobian_∂SS_and_pars) #transposed

        converted_cols = idx_conversion[cols]

        perm_vals = sortperm(converted_cols) # sparse reorders the rows and cols and sorts by column. need to do that also for the values

        min_n_funcs = length(vals) ÷ max_exprs_per_func + 1

        funcs = Function[]

        if min_n_funcs == 1
            push!(funcs, write_derivatives_function(vals[perm_vals], 1:length(vals), Val(:string)))
        else
            Polyester.@batch for i in 1:min(min_n_funcs, length(vals))
                indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(vals) : i * max_exprs_per_func)

                push!(funcs, write_derivatives_function(vals[perm_vals][indices], indices, Val(:string)))
            end
        end

        𝓂.model_jacobian_SS_and_pars_vars = (funcs, sparse(rows, converted_cols, zero(cols), length(final_indices), length(eqs) * length(vars)))

        # first order
        min_n_funcs = length(first_order) ÷ max_exprs_per_func + 1

        funcs = Function[]

        if min_n_funcs == 1
            push!(funcs, write_derivatives_function(first_order, 1:length(first_order), Val(:string)))
        else
            for i in 1:min(min_n_funcs, length(first_order))
                indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(first_order) : i * max_exprs_per_func)

                push!(funcs, write_derivatives_function(first_order[indices], indices, Val(:string)))
            end
        end

        𝓂.model_jacobian = (funcs, row1 .+ (column1 .- 1) .* length(eqs_sub),  zeros(length(eqs_sub), length(vars)))
    end
end
    
if max_perturbation_order >= 2
# second order
    if 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝛔 == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
        𝓂.solution.perturbation.second_order_auxilliary_matrices = create_second_order_auxilliary_matrices(𝓂.timings)

        perm_vals = sortperm(column2) # sparse reorders the rows and cols and sorts by column. need to do that also for the values

        min_n_funcs = length(second_order) ÷ max_exprs_per_func + 1

        funcs = Function[]
    
        if min_n_funcs == 1
            push!(funcs, write_derivatives_function(second_order[perm_vals], 1:length(second_order), Val(:string)))
        else
            Polyester.@batch for i in 1:min(min_n_funcs, length(second_order))
                indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(second_order) : i * max_exprs_per_func)
        
                push!(funcs, write_derivatives_function(second_order[perm_vals][indices], indices, Val(:string)))
            end
        end

        𝓂.model_hessian = (funcs, sparse(row2, column2, zero(column2), length(eqs_sub), size(𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂,1)))
    end
end

if max_perturbation_order == 3
# third order
    if 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
        𝓂.solution.perturbation.third_order_auxilliary_matrices = create_third_order_auxilliary_matrices(𝓂.timings, unique(column3))
    
        perm_vals = sortperm(column3) # sparse reorders the rows and cols and sorts by column. need to do that also for the values

        min_n_funcs = length(third_order) ÷ max_exprs_per_func + 1

        funcs = Function[]
    
        if min_n_funcs == 1
            push!(funcs, write_derivatives_function(third_order[perm_vals], 1:length(third_order), Val(:string)))
        else
            # Polyester.@batch for i in 1:min(min_n_funcs, length(third_order))
            for i in 1:min(min_n_funcs, length(third_order))
                println(i)
                indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(third_order) : i * max_exprs_per_func)
        
                push!(funcs, write_derivatives_function(third_order[perm_vals][indices], indices, Val(:string)))
            end
        end

        𝓂.model_third_order_derivatives = (funcs, sparse(row3, column3, zero(column3), length(eqs_sub), size(𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃,1)))
    end
end



import MacroModelling: expand_steady_state, kron³
parameters = 𝓂.parameter_values
verbose = true
tol = 1e-12

SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
all_SS = expand_steady_state(SS_and_pars,𝓂)

∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix

𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)
sum(abs.(𝐒₂) .< eps())
∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

import MacroModelling: A_mult_kron_power_3_B, mat_mult_kron, kron³, solve_matrix_equation_forward
𝑺₁ = 𝐒₁
M₂ = 𝓂.solution.perturbation.second_order_auxilliary_matrices
M₃ = 𝓂.solution.perturbation.third_order_auxilliary_matrices
T = 𝓂.timings


    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]


    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    B = spinv * ∇₁₊
    droptol!(B,tol)

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    # 𝐗₃ = -∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux)
    𝐗₃ = -A_mult_kron_power_3_B(∇₃, aux)

    tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)
    out = - ∇₃ * tmpkron - ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron * M₃.𝐏₁ᵣ̃ - ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron * M₃.𝐏₂ᵣ̃
    𝐗₃ += out
    
    # tmp𝐗₃ = -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
    tmp𝐗₃ = -mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)

    tmpkron1 = -∇₂ *  ℒ.kron(𝐒₁₊╱𝟎,𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔,𝐒₁₋╱𝟏ₑ)
    out2 = tmpkron1 * tmpkron2 +  tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
    
    𝐗₃ += (tmp𝐗₃ + out2 + -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * M₂.𝛔)) * M₃.𝐏# |> findnz
    
    𝐗₃ += @views -∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]) * M₃.𝐏
    droptol!(𝐗₃,tol)
    
    X = spinv * 𝐗₃ * M₃.𝐂₃
    droptol!(X,tol)
    
    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
    
    C = M₃.𝐔₃ * tmpkron + M₃.𝐔₃ * M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐔₃ * M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃
    C *= M₃.𝐂₃
    # C += M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) # no speed up here from A_mult_kron_power_3_B; this is the bottleneck. ideally have this return reduced space directly. TODO: write compressed kron3
    C += kron³(𝐒₁₋╱𝟏ₑ, M₃)
    droptol!(C,tol)
    
    using BenchmarkTools
    @benchmark kron³(𝐒₁₋╱𝟏ₑ, M₃)
    @benchmark M₃.𝐔₃ * kron(𝐒₁₋╱𝟏ₑ,kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃
    
    r1,c1,v1 = findnz(B)
    r2,c2,v2 = findnz(C)
    r3,c3,v3 = findnz(X)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    push!(coordinates,(r2,c2))
    push!(coordinates,(r3,c3))
    
    values = vcat(v1, v2, v3)

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(B))
    push!(dimensions,size(C))
    push!(dimensions,size(X))

    𝐒₃, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres, sparse_output = true)

    if !solved
        return 𝐒₃, solved
    end

    𝐒₃ *= M₃.𝐔₃






    findnz(M₃.𝐔₃)
    findnz(M₃.𝐂₃)
rows, cols, vals = findnz(𝐒₁₋╱𝟏ₑ)

combination(rows,3)
rows = 1:6

nk = n₋ + nₑ + 1
third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]

third_order_idxs = [((i,k,l), length(unique((i,k,l))) == 1 ? 1 : length(unique((i,k,l))) == 2 ? 3 : 6) for i in 1:nk for k in 1:i for l in 1:k]


third_order_idxs = Dict{Tuple{Int,Int,Int},Tuple{Int,Int}}()
idx = 1
for i in 1:nk 
    for k in 1:i 
        for l in 1:k
            factor = length(unique((i,k,l))) == 1 ? 1 : length(unique((i,k,l))) == 2 ? 3 : 6
            third_order_idxs[(i,k,l)] = (factor,idx)
            idx += 1
        end
    end
end

valls = []
for (r1, c1) in zip(rows,cols)
    for (r2, c2) in zip(rows,cols)
        for (r3, c3) in zip(rows,cols)
            sorted_rows = Tuple(sort([r1,r2,r3],rev=true))
            sorted_cols = (c1,c2,c3) # Tuple(sort([c1,c2,c3],rev=true))
            if haskey(third_order_idxs, sorted_rows) && haskey(third_order_idxs, sorted_cols)
                push!(valls,third_order_idxs[sorted_rows][1])
            end
        end
    end
end

idxs = Set{Vector{Int}}()
for k in unique(rows)
    for j in unique(rows)
        for i in unique(rows)
            sorted_ids = sort([k,j,i])
            push!(idxs, sorted_ids)
        end
    end
end
idxs|>typeof


S = 𝐒₁₋╱𝟏ₑ

M₃.𝐔₃[5,:]


result = spzeros(size(M₃.𝐂₃, 2), size(M₃.𝐔₃, 1))

for (r1, c1, v1) in zip(rows,cols,vals)
    for (r2, c2 ,v2) in zip(rows,cols,vals)
        for (r3, c3, v3) in zip(rows,cols,vals)
# for i in 1:length(vals)
#     for k in 1:i
#         for l in 1:k
#             r1, c1, v1 = [rows[i],cols[i],vals[i]]
#             r2, c2, v2 = [rows[k],cols[k],vals[k]]
#             r3, c3, v3 = [rows[l],cols[l],vals[l]]
            # sorted_rows = (r1, r2, r3) # Tuple(sort([r1,r2,r3],rev=true))
            sorted_cols = (c1, c2, c3) # Tuple(sort([c1,c2,c3],rev=true))
            sorted_rows = Tuple(sort([r1,r2,r3],rev=true))
            # sorted_cols = Tuple(sort([c1,c2,c3],rev=true))
            if haskey(third_order_idxs, sorted_rows) && haskey(third_order_idxs, sorted_cols)# && result[third_order_idxs[sorted_rows][2], third_order_idxs[sorted_cols][2]] == 0
                result[third_order_idxs[sorted_rows][2], third_order_idxs[sorted_cols][2]] += v1*v2*v3# * min(third_order_idxs[sorted_rows][1], third_order_idxs[sorted_cols][1])
            end
        end
    end
end

vals[1] * vals[1] * vals[3]

third_order_idxs[(3,1,1)]
findnz(result)[1] == findnz(M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃)[1]
findnz(result)[2] == findnz(M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃)[2]
findnz(result)[3]  - findnz(M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃)[3]

isapprox(result, M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) * M₃.𝐂₃)
# Fill in the result matrix by iterating over the elements of S
for i in 1:size(S, 1)
    for j in 1:size(S, 2)
        if S[i, j] != 0
            for k in 1:size(S, 1)
                if S[k, j] != 0
                    for l in 1:size(S, 1)
                        if S[l, j] != 0
                            sorted_ids = sort([i, k, l])
                            row_index = (sorted_ids[3] - 1) * nₑ₋^2 + (sorted_ids[2] - 1) * nₑ₋ + sorted_ids[1]
                            col_index = nₑ₋^2 * (i - 1) + nₑ₋ * (k - 1) + l
                            result[col_index, row_index] += S[i, j] * S[k, j] * S[l, j]
                        end
                    end
                end
            end
        end
    end
end





    r1,c1,v1 = findnz(B)
    r2,c2,v2 = findnz(C)
    r3,c3,v3 = findnz(X)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    push!(coordinates,(r2,c2))
    push!(coordinates,(r3,c3))
    
    values = vcat(v1, v2, v3)

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(B))
    push!(dimensions,size(C))
    push!(dimensions,size(X))

    𝐒₃, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres, sparse_output = true)

    if !solved
        return 𝐒₃, solved
    end

    𝐒₃ *= M₃.𝐔₃




𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝓂.solution.perturbation.second_order_auxilliary_matrices, 𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]




𝓂.model_third_order_derivatives[1] |> length
𝓂.model_third_order_derivatives[2]
𝓂.model_hessian[1] |> length



SS_and_pars = get_non_stochastic_steady_state(𝓂, 𝓂.parameter_values)[1]

SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]

par = vcat(𝓂.parameter_values,calibrated_parameters)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

X = [SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; SS[dyn_ss_idx]; par; shocks_ss]


import Polyester
@time begin
    vals = zeros(length(𝓂.model_hessian[1]))

    Polyester.@batch per=thread for f in 𝓂.model_third_order_derivatives[1]
        out = f(X)
        vals[out[2]] .= out[1]
    end
end


import Polyester
@time begin
    vals = zeros(length(𝓂.model_third_order_derivatives[1]))

    Polyester.@batch per=thread for f in 𝓂.model_third_order_derivatives[1]
        out = f(X)
        vals[out[2]] .= out[1]
    end
end

using BenchmarkTools
@benchmark begin
    vals = zeros(length(𝓂.model_third_order_derivatives[1]))

    Polyester.@batch per=thread for f in 𝓂.model_third_order_derivatives[1]
        out = f(X)
        vals[out[2]] .= out[1]
    end
end


@time begin
    vals = []

    for f in 𝓂.model_third_order_derivatives[1]
        push!(vals, f(X)[1]...)
    end
end

@benchmark begin
    vals = []
    
    for f in 𝓂.model_third_order_derivatives[1]
        push!(vals, f(X)[1]...)
    end
end


vals = []

for f in 𝓂.model_third_order_derivatives[1]
    push!(vals, f(X)...)
end


vals = []
elapsed_time = 0.0
for f in 𝓂.model_third_order_derivatives[1]
    start_time = time()
    @time push!(vals, f(X)...)
    elapsed_time += time() - start_time
end


import Polyester
for f in zip(1:length(𝓂.model_third_order_derivatives[1]), 𝓂.model_third_order_derivatives[1])
    println(f[1])
end

vals = zeros(length(𝓂.model_third_order_derivatives[1]))

itter = zip(1:length(𝓂.model_third_order_derivatives[1]), 𝓂.model_third_order_derivatives[1]) |>collect
# Use Polyester.@batch to parallelize the loop
Polyester.@batch for f in itter
    vals[f[1]] .= f[2](X)
end

# Assuming 𝓂.model_third_order_derivatives[1] is a list of functions and X is your input

batch_size = ceil(Int, length(𝓂.model_third_order_derivatives[1]) / Threads.nthreads())

@time begin
    vals = Float64[]
    thread_id = Int[]
    # Use Polyester.@batch to parallelize the loop
    Polyester.@batch per=core for f in 𝓂.model_third_order_derivatives[1]
        push!(vals, f(X)...)
        push!(thread_id, Threads.threadid())
    end
end

valss = []

for f in 𝓂.model_third_order_derivatives[1]
    push!(valss, f(X)...)
end

vals == valss

# Assuming X and 𝓂.model_third_order_derivatives[1] are already defined
functions = 𝓂.model_third_order_derivatives[1]
n = length(functions)
results = zeros(n)  # Shared array to store results from each thread
counter = Threads.Atomic{Int}(1)  # Atomic counter to track positions

# Parallel loop with Polyester, preserving order
Polyester.@batch per=thread for f in functions
    i = counter[]  # Get the current position
    counter[] += 1  # Increment the counter
    println(Threads.threadid())
    results[i] = f(X)[1]
end



using BenchmarkTools

@benchmark begin
    vals = []
    thread_id = []
    # Use Polyester.@batch to parallelize the loop
    Polyester.@batch for f in 𝓂.model_third_order_derivatives[1]
        push!(vals, f(X)...)
        push!(thread_id, Threads.threadid())
    end
end


# Convert the atomic vector to a regular vector
result = vals[]

elapsed_time
# 50 -> 185
# 25 -> 118
# 10 -> 57
#  5 -> 38
#  3 -> 31
#  1 -> 24
#MT1 -> 16.7
Accessors.@reset 𝓂.model_third_order_derivatives[2].nzval = vals

return 𝓂.model_third_order_derivatives[2] * 𝓂.solution.perturbation.third_order_auxilliary_matrices.𝐔∇₃






𝓂 = m
max_exprs_per_func = 200
import Symbolics
# derivative of SS equations wrt parameters and SS_and_pars
# unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
SS_and_pars = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))

SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
        
SS_and_pars_names = Symbol.(replace.(string.(SS_and_pars_names_lead_lag), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

ss_equations = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

Symbolics.@syms norminvcdf(x) norminv(x) qnorm(x) normlogpdf(x) normpdf(x) normcdf(x) pnorm(x) dnorm(x) erfc(x) erfcinv(x)

# overwrite SymPyCall names
other_pars = setdiff(union(𝓂.parameters_in_equations, 𝓂.parameters_as_function_of_parameters), 𝓂.parameters)

if length(other_pars) > 0
    eval(:(Symbolics.@variables $(other_pars...)))
end

vars = eval(:(Symbolics.@variables $(SS_and_pars...)))

vars_no_lead_lag = eval(:(Symbolics.@variables $(SS_and_pars_names...)))

pars = eval(:(Symbolics.@variables $(𝓂.parameters...)))

input_args = vcat(𝓂.parameters, SS_and_pars)

Symbolics.@variables 𝔛[1:length(input_args)]

input_X_no_time = Pair{Symbolics.Num, Symbolics.Num}[]

for (v,input) in enumerate(input_args)
    push!(input_X_no_time, eval(input) => eval(𝔛[v]))
end

ss_eqs = Symbolics.parse_expr_to_symbolic.(ss_equations,(@__MODULE__,))

calib_eqs = Dict([(eval(calib_eq.args[1]) => eval(calib_eq.args[2])) for calib_eq in reverse(𝓂.calibration_equations_no_var)])

eqs = Symbolics.Num[]
for subst in ss_eqs
    subst = Symbolics.fixpoint_sub(subst, calib_eqs)
    subst = Symbolics.substitute(subst, input_X_no_time)
    push!(eqs, subst)
end

# ∂SS_equations_∂parameters = Symbolics.sparsejacobian(ss_eqs, eval.(𝓂.parameters)) |> findnz

∂SS_equations_∂parameters = Symbolics.sparsejacobian(eqs, eval.(𝔛[1:length(pars)])) |> findnz

min_n_funcs = length(∂SS_equations_∂parameters[3]) ÷ max_exprs_per_func

funcs = Function[]

if min_n_funcs == 0
    push!(funcs, write_derivatives_function(∂SS_equations_∂parameters[3], Val(:Symbolics)))
else
    for i in 1:min_n_funcs
        indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(∂SS_equations_∂parameters[3]) : i * max_exprs_per_func)
        push!(funcs, write_derivatives_function(∂SS_equations_∂parameters[3][indices], Val(:Symbolics)))
    end
end

𝓂.∂SS_equations_∂parameters = (funcs, sparse(∂SS_equations_∂parameters[1], ∂SS_equations_∂parameters[2], zeros(Float64,length(∂SS_equations_∂parameters[3])), length(eqs), length(pars)))

# 𝓂.∂SS_equations_∂parameters = write_sparse_derivatives_function(∂SS_equations_∂parameters[1], 
#                                                                     ∂SS_equations_∂parameters[2], 
#                                                                     ∂SS_equations_∂parameters[3],
#                                                                     length(eqs), 
#                                                                     length(pars),
#                                                                     Val(:string));

∂SS_equations_∂SS_and_pars = Symbolics.sparsejacobian(eqs, eval.(𝔛[length(pars)+1:end])) |> findnz

min_n_funcs = length(∂SS_equations_∂SS_and_pars[3]) ÷ max_exprs_per_func

funcs = Function[]

if min_n_funcs == 0
    push!(funcs, write_derivatives_function(∂SS_equations_∂SS_and_pars[3], Val(:Symbolics)))
else
    for i in 1:min_n_funcs
        indices = ((i - 1) * max_exprs_per_func + 1):(i == min_n_funcs ? length(∂SS_equations_∂SS_and_pars[3]) : i * max_exprs_per_func)

        push!(funcs, write_derivatives_function(∂SS_equations_∂SS_and_pars[3][indices], Val(:Symbolics)))
    end
end

𝓂.∂SS_equations_∂SS_and_pars = (funcs, ∂SS_equations_∂SS_and_pars[1] .+ (∂SS_equations_∂SS_and_pars[2] .- 1) .* length(eqs), zeros(length(eqs), length(vars)))

# 𝓂.∂SS_equations_∂SS_and_pars = write_sparse_derivatives_function(∂SS_equations_∂SS_and_pars[1], 
#                                                                     ∂SS_equations_∂SS_and_pars[2], 
#                                                                     ∂SS_equations_∂SS_and_pars[3],
#                                                                     length(eqs), 
#                                                                     length(vars),
#                                                                     Val(:string));




SS_and_pars, (solution_error, iters)  = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, false, 𝓂.solver_parameters)

SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
    
SS_and_pars_names = vcat(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)

# unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)
unknowns = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))), 𝓂.calibration_equations_parameters))
# ∂SS_equations_∂parameters = try 𝓂.∂SS_equations_∂parameters(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
# catch
#     return (SS_and_pars, (10, iters)), x -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
# end

X = [𝓂.parameter_values; SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]]

vals = Float64[]

for f in 𝓂.∂SS_equations_∂parameters[1]
    push!(vals, f(X)...)
end

Accessors.@reset 𝓂.∂SS_equations_∂parameters[2].nzval = vals

∂SS_equations_∂parameters = 𝓂.∂SS_equations_∂parameters[2]


vals = Float64[]

for f in 𝓂.∂SS_equations_∂SS_and_pars[1]
    push!(vals, f(X)...)
end

𝓂.∂SS_equations_∂SS_and_pars[3] .*= 0
𝓂.∂SS_equations_∂SS_and_pars[3][𝓂.∂SS_equations_∂SS_and_pars[2]] .+= vals

∂SS_equations_∂SS_and_pars = 𝓂.∂SS_equations_∂SS_and_pars[3]

# ∂SS_equations_∂parameters = 𝓂.∂SS_equations_∂parameters(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix
# ∂SS_equations_∂SS_and_pars = 𝓂.∂SS_equations_∂SS_and_pars(parameter_values, SS_and_pars[indexin(unknowns, SS_and_pars_names_lead_lag)]) |> Matrix

∂SS_equations_∂SS_and_pars_lu = RF.lu!(∂SS_equations_∂SS_and_pars, check = false)

JVP = -(∂SS_equations_∂SS_and_pars_lu \ ∂SS_equations_∂parameters)#[indexin(SS_and_pars_names, unknowns),:]

jvp = zeros(length(SS_and_pars_names_lead_lag), length(𝓂.parameters))

for (i,v) in enumerate(SS_and_pars_names)
    if v in unknowns
        jvp[i,:] = JVP[indexin([v], unknowns),:]
    end
end





# get_non_stochastic_steady_state(m,m.parameter_values)[1]

# Zygote.jacobian(x->get_non_stochastic_steady_state(m,x)[1], m.parameter_values)[1]






