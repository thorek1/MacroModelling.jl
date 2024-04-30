
using MacroModelling
import Turing: NUTS, HMC, PG, IS, sample, logpdf, Truncated#, Normal, Beta, Gamma, InverseGamma,
using CSV, DataFrames, AxisKeys
import Zygote
import ForwardDiff
import ChainRulesCore: @ignore_derivatives, ignore_derivatives, rrule, NoTangent, @thunk
using Random
import FiniteDifferences
import BenchmarkTools: @benchmark
import LinearAlgebra as ℒ
Random.seed!(1)
import MacroModelling: get_and_check_observables, solve!, check_bounds, get_relevant_steady_state_and_state_update, calculate_loglikelihood, get_initial_covariance, riccati_forward

# include("../models/RBC_baseline.jl")
include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")
# include("../models/Smets_Wouters_2007_linear.jl")
# 𝓂 = Smets_Wouters_2007_linear
# 𝓂 = RBC_baseline
𝓂 = m
# parameter_values = parameters_combined
algorithm = :first_order
filter = :kalman
warmup_iterations = 0
presample_periods = 0
initial_covariance = :diagonal
tol = 1e-12
verbose = false
T = 𝓂.timings

𝓂.model_jacobian

𝓂.model_jacobian_parameters

𝓂.model_jacobian_SS_and_pars_vars


# derivatives wrt solution
SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix
# calculate_hessian(𝓂.parameter_values, SS_and_pars, 𝓂)
# get_solution(RBC_baseline, algorithm = :second_order)

fini_grad_SS_and_pars = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5,1), SS_and_pars->calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂), SS_and_pars)[1] |> sparse

fini_grad_parameters = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5,1), p->calculate_jacobian(p, SS_and_pars, 𝓂), 𝓂.parameter_values)[1] |> sparse

droptol!(fini_grad_parameters, 1e-12)
droptol!(fini_grad_SS_and_pars, 1e-12)


𝓂.model_jacobian_SS_and_pars_vars
𝓂.parameters
get_equations(𝓂)

parameters = 𝓂.parameter_values

SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]
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
analytical_jac_parameters = 𝓂.model_jacobian_parameters([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx])
analytical_jac_SS_and_pars_vars = 𝓂.model_jacobian_SS_and_pars_vars([SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; shocks_ss], par, SS[dyn_ss_idx])




fini_grad_parameters ≈ analytical_jac_parameters
fini_grad_SS_and_pars ≈ analytical_jac_SS_and_pars_vars


findnz(analytical_jac_parameters)[1]
findnz(fini_grad_parameters)[1]

findnz(analytical_jac_SS_and_pars_vars)[1]
findnz(fini_grad_SS_and_pars)[1]

findnz(analytical_jac_SS_and_pars_vars)[2]
findnz(fini_grad_SS_and_pars)[2]

findnz(analytical_jac_SS_and_pars_vars)[3]
findnz(fini_grad_SS_and_pars)[3]

analytical_jac' * vec(∇₁)

analytical_jac'
# derivatives wrt jacobian calc
# using SymbolicUtils, Symbolics
# @variables t x y z(t) a c, i_y , k_y, β, δ , α
# ex = x + y + sin(z)
# substitute(ex, Dict([x => z, sin(z) => z^2]))

# expr = [:(δ = i_y / k_y), :(β = 1 / (α / k_y + (1 - δ)))]


# expr |> dump
# expr.args[1]
# expr.args[2]
# substitute(ex, Dict([eval(expr[2].args[1]) => eval(expr[2].args[2]), eval(expr[1].args[1]) => eval(expr[1].args[2]), sin(z) => z^2]))

# # c[0] ^ (-σ) = β * c[1] ^ (-σ) * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)
# # β * c₁ ^ (-σ) * (α * z₁ * (k₀ / l₁) ^ (α - 1) + 1 - δ)
# (*)((*)((*)((*)(-1, (^)(c₍₁₎, (+)(-1, (*)(-1, σ)))), (+)((+)(-1, δ), (*)((*)((*)(-1, (^)((/)(k₍₀₎, l₍₁₎), (+)(-1, α))), z₍₁₎), α))), β), σ)
# (-1 * (c₁ ^ (-1 + (-1 * σ))) + (-1 + δ + (-1 * (k₀ / l₁) ^ (-1 + α) * z₁) * α)) * β * σ


using SymbolicUtils, Symbolics
import MacroModelling: match_pattern, get_symbols
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
    push!(alll_no_time,:($(Symbol(replace(string(var), r"₍₁₎$"=>"", r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))) = X[$ii]))
    ii += 1
end

for var in present_varss
    push!(alll,:($var = X[$ii]))
    push!(alll_no_time,:($(Symbol(replace(string(var), r"₍₀₎$"=>"", r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))) = X[$ii]))
    ii += 1
end

for var in past_varss
    push!(alll,:($var = X[$ii]))
    push!(alll_no_time,:($(Symbol(replace(string(var), r"₍₋₁₎$"=>"", r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))) = X[$ii]))
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

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))

vars_raw = [dyn_future_list[indexin(sort(future),future)]...,
        dyn_present_list[indexin(sort(present),present)]...,
        dyn_past_list[indexin(sort(past),past)]...,
        dyn_exo_list[indexin(sort(exo),exo)]...]

# overwrite SymPyCall names
eval(:(Symbolics.@variables $(reduce(union,get_symbols.(vcat(𝓂.dyn_equations, 𝓂.calibration_equations_no_var)))...)))

vars = eval(:(Symbolics.@variables $(vars_raw...)))

Symbolics.@syms normlogpdf(x) norminvcdf(x)

eqs = Symbolics.parse_expr_to_symbolic.(𝓂.dyn_equations,(@__MODULE__,))

future_no_lead_lag = Symbol.(replace.(string.(future), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
present_no_lead_lag = Symbol.(replace.(string.(present), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
past_no_lead_lag = Symbol.(replace.(string.(past), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

eval(:(Symbolics.@variables $(Set(vcat(future_no_lead_lag, present_no_lead_lag, past_no_lead_lag))...)))

SS_and_pars = Symbol.(vcat(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""), 𝓂.calibration_equations_parameters))

# remove time indices
vars_no_time_transform = union(Dict(eval.(dyn_future_list) .=> eval.(future_no_lead_lag)), 
                                Dict(eval.(dyn_present_list) .=> eval.(present_no_lead_lag)), 
                                Dict(eval.(dyn_past_list) .=> eval.(past_no_lead_lag)),
                                Dict(eval.(dyn_exo_list) .=> 0))


if max_perturbation_order >= 2 
    nk = length(vars_raw)
    second_order_idxs = [nk * (i-1) + k for i in 1:nk for k in 1:i]
    if max_perturbation_order == 3
        third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]
    end
end

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











σ = 1

α = 1/3

i_y = 0.25

k_y = 10.4

δ = i_y / k_y

β = 1 / (α / k_y + (1 - δ))

𝓂.SS_check_func





A, solved = riccati_forward(∇₁; T = 𝓂.timings)


fini_grad = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), ∇₁ -> riccati_forward(∇₁; T = 𝓂.timings)[1], ∇₁)[1]


expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

A =  ∇₁[:,1:T.nFuture_not_past_and_mixed] * sparse(expand[1]) |> Matrix
B =  ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] |> Matrix
C =  ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2] |> Matrix


A
B

ab = A * B

ab * B'
A' * ab

zyg_gradA = Zygote.jacobian(A -> A * B, A)[1]
zyg_gradB = Zygote.jacobian(B -> A * B, B)[1]

C
# ∇̂₀ = ℒ.lu(B, check = false)
# B = (∇̂₀ \ A)
# A = (∇̂₀ \ C)

function calc_quad_sol(∇₊,∇₀,∇₋; tol::Float64 = 1e-12)
    ∇̂₀ = ℒ.lu(∇₀, check = false)
    A = (∇̂₀ \ ∇₋)
    B = (∇̂₀ \ ∇₊)

    # A = ∇₀ \ ∇₋
    # B = ∇₀ \ ∇₊

    C = copy(A)
    C̄ = similar(A)
    C² = similar(A)

    maxiter = 10000  # Maximum number of iterations

    # error = one(tol) + tol
    # iter = 0

    # tol_not_reached = true
    
    for iter in 1:maxiter
        copy!(C̄,C)  # Store the current C̄ before updating it

        ℒ.mul!(C²,C,C)
        ℒ.mul!(C,B,C²)
        ℒ.axpy!(1,A,C)
        # Update C̄ based on the given formula
        # C = A + B * C^2
        
        # Check for convergence
        if iter % 200 == 0
            ℒ.axpy!(-1,C,C̄)

            tol_reached = true

            for element in C̄
                if abs(element) > tol
                    tol_reached = false
                    break
                end
            end

            if tol_reached
                return C, iter
            end

            # error = maximum(abs, C̄)
            # error = maximum(abs, C - C̄)
        end

    end

    return C, iter
end

using SpeedMapping


expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

A =  ∇₁[:,1:T.nFuture_not_past_and_mixed] * sparse(expand[1]) |> Matrix
B =  ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] |> Matrix
C =  ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2] |> Matrix

∇̂₀ = ℒ.lu(B, check = false)
B = (∇̂₀ \ A)
A = (∇̂₀ \ C)

D = similar(C)

sol = speedmapping(zero(A); 
                    m! = (C̄, C) -> begin 
                                    ℒ.mul!(D,C,C)
                                    ℒ.mul!(C̄,B,D)
                                    ℒ.axpy!(1,A,C̄)
                                end, 
                    tol = 1e-12, maps_limit = 10000)

X = -sol.minimizer

expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

A =  ∇₁[:,1:T.nFuture_not_past_and_mixed] * sparse(expand[1]) |> Matrix
B =  ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] |> Matrix
C =  ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2] |> Matrix


A * X * X + B * X + C



# X̄ + 2 * A * X * S + B' * S
inv(X' * A + A' * X + B')
A * X * X
S = -inv(A' * X' + B')

S = -inv(2 * A * X + B')

C̄ = S
B̄ = X * C̄
Ā = X * B̄

Ā = sparse(Ā)
droptol!(Ā, 1e-12)

fini_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), C -> calc_quad_sol(A,B,C)[1][3,4],C)[1]


fini_grad


@benchmark sol = speedmapping(zero(A); 
m! = (C̄, C) -> begin 
                ℒ.mul!(D,C,C)
                ℒ.mul!(C̄,B,D)
                ℒ.axpy!(1,A,C̄)
            end, 
tol = 1e-12, maps_limit = 10000)



@profview for i in 1:1000 sol = speedmapping(zero(A); 
m! = (C̄, C) -> begin 
                ℒ.mul!(D,C,C)
                ℒ.mul!(C̄,B,D)
                ℒ.axpy!(1,A,C̄)
            end, 
tol = tol, maps_limit = 10000) end
sol.minimizer

calc_quad_sol(A,B,C)[1]


@benchmark calc_quad_sol(A,B,C)

@profview for o in 1:10000 calc_quad_sol(A,B,C) end



@benchmark S₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)


