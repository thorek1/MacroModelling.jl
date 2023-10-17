using MacroModelling

@model testmax begin 
    1  /  c[0] = (β  /  c[1]) * (r[1] + (1 - δ))

    r̂[0] = α * exp(z[0]) * k[-1]^(α - 1)

    r[0] = max(r̄,r̂[0]) | ϵᶻ > 0

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
end

@parameters testmax begin
    r̄ = 0
    σᶻ= 1#0.01
    ρᶻ= 0.8#2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_irf(testmax)(:r,:,:ϵᶻᵒᵇᶜ⁽⁻¹⁾)
get_solution(testmax)

# using Optimization, Ipopt, OptimizationMOI, OptimizationOptimJL, LineSearches, OptimizationNLopt, KNITRO
import MacroTools: postwalk, unblock
import MacroModelling: parse_for_loops, convert_to_ss_equation, simplify,create_symbols_eqs!,remove_redundant_SS_vars!,get_symbols, parse_occasionally_binding_constraints, parse_algorithm_to_state_update
import DataStructures: CircularBuffer
import Subscripts: super, sub
import LinearAlgebra as ℒ
using BenchmarkTools, JuMP, StatusSwitchingQP

𝓂 = testmax
algorithm = :first_order
state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂)
T = 𝓂.timings
periods = 40

initial_state = zeros(T.nVars)
# Y = zeros(Real,T.nVars,periods,1)
# T.exo
obc_shocks = [i[1] for i in 𝓂.obc_shock_bounds]

shock_history = zeros(T.nExo,periods)
shock_history[1,1] = -4


reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())
# shock_history[16,1]

obc_shock_idx = contains.(string.(T.exo),"ᵒᵇᶜ")

function calc_state(x::Vector{S}) where S
    Y = zeros(AffExpr,T.nVars,periods,1)
    # println(collect(x))
    shock_hst = AffExpr.(copy(shock_history[:,1]))
    shock_hst[obc_shock_idx] .= x

    Y[:,1,1] = state_update(initial_state, shock_hst)

    for t in 1:periods-1
        Y[:,t+1,1] = state_update(Y[:,t,1],shock_history[:,t+1])
    end

    Y .+= reference_steady_state[1:T.nVars]

    return Y[4,:,:]
end

# get bound on shocks
# get minmax condition

# fill(a[3],sum(obc_shock_idx)÷length(𝓂.obc_shock_bounds))

model = Model(StatusSwitchingQP.Optimizer)
set_silent(model)

periods_per_shock = sum(obc_shock_idx)÷length(𝓂.obc_shock_bounds)
num_shocks = length(𝓂.obc_shock_bounds)

# Create the variables over the full set of indices first.
@variable(model, x[1:num_shocks*periods_per_shock])

# Now loop through obc_shock_bounds to set the bounds on these variables.
for (idx, v) in enumerate(𝓂.obc_shock_bounds)
    is_upper_bound = v[2]
    bound = v[3]
    idxs = (idx - 1) * periods_per_shock + 1:idx * periods_per_shock
    if is_upper_bound
        set_upper_bound.(x[idxs], bound)
    else
        set_lower_bound.(x[idxs], bound)
    end
end

@objective(model, Min, x' * ℒ.I * x)
@constraint(model, calc_state(x) .>= 0)
JuMP.optimize!(model)


@profview for i in 1:100 JuMP.optimize!(model) end

value.(x)

# ≤≥
:(a-b-c)|>dump
testmax.dyn_equations[3]|>dump

eq = testmax.dyn_equations[3]
[i for (i,c) in enumerate(condition_list) if c isa Expr]

import MacroModelling: get_symbols, match_pattern

#check if equation contains maxmin

expression_for_placeholder = []
ismax = []
eqs = postwalk(x -> 
            x isa Expr ?
                x.head == :call ?
                    x.args[1] ∈ [:max,:min] ?
                        begin 
                            if length(intersect(get_symbols(x.args[2]),testmax.var)) == 0
                                push!(expression_for_placeholder,x.args[2])
                            else
                                push!(expression_for_placeholder,x.args[3])
                            end
                            if x.args[1] == :max
                                push!(ismax,true)
                                :max_placeholder
                            else
                                push!(ismax,false)
                                :min_placeholder
                            end
                        end :
                    x :
                x :
            x,
        eq)

# solve for the placeholder
using SymPyPythonCall

@syms max_placeholder, min_placeholder, r₍₀₎

eq_to_solve = eval(eqs)

if ismax[1]
    sol = solve(eq_to_solve, max_placeholder)
else
    sol = solve(eq_to_solve, min_placeholder)
end


final_expr = Expr(:call, :-, ismax[1] ? :($(expression_for_placeholder[1])) : :(-$(expression_for_placeholder[1])), Meta.parse(string(sol[1])))






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
for (i, var) in enumerate(ss_varss)
    push!(steady_state,:($var = X̄[$i]))
    # ii += 1
end

ii = 1

alll = []
for var in future_varss
    push!(alll,:($var = X[$ii]))
    ii += 1
end

for var in present_varss
    push!(alll,:($var = X[$ii]))
    ii += 1
end

for var in past_varss
    push!(alll,:($var = X[$ii]))
    ii += 1
end

for var in shock_varss
    push!(alll,:($var = X[$ii]))
    ii += 1
end


# paras = []
# push!(paras,:((;$(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)...)) = params))

paras = []
for (i, parss) in enumerate(vcat(𝓂.parameters,𝓂.calibration_equations_parameters))
    push!(paras,:($parss = params[$i]))
end





mod_func3 = :(function model_jacobian(X::Vector, params::Vector{Real}, X̄::Vector)
$(alll...)
$(paras...)
$(𝓂.calibration_equations_no_var...)
$(steady_state...)
$final_expr
end)



intersect(get_symbols(:r̄),testmax.var)
# replace maxmin with the side containing parameters only


check_minmax()
max(r̄,r̂[0]) = r[0]

r[0] = max(r̄,r̂[0]) 
r[0] - r̄ > 0
:(begin
r[0] = max(r̄,r̂[0]) + v[0]
r[0] - r̄ - v[0] > 0
end)

using MacroTools
MacroTools.postwalk(:(r[0] = max(r̄,r̂[0]) + v[0]))

r[0] = max(r̄,r̂[0]) + v[0]
r[0] - r̄ - v[0] > 0

:(ϵᶻ > 0) |> dump


using MacroTools

function parse_obc_shock_bounds(expr::Expr)
    # Determine the order of the shock and bound in the expression
    shock_first = isa(expr.args[2], Symbol)
    
    # Extract the shock and bound from the expression
    shock = shock_first ? expr.args[2] : expr.args[3]
    bound_expr = shock_first ? expr.args[3] : expr.args[2]
    
    # Evaluate the bound expression to get a numerical value
    bound = eval(bound_expr) |> Float64
    
    # Determine whether the bound is a lower or upper bound based on the comparison operator and order
    is_upper_bound = (expr.args[1] in (:<, :≤) && shock_first) || (expr.args[1] in (:>, :≥) && !shock_first)
    
    return shock, is_upper_bound, bound
end

# Example usage:
expressions = [:(ϵᶻ > 0), :(1/2 < ϵᶻ), :(ϵᶻ > 1/3+1), :(ϵᶻ ≤ -1), :(1 ≥ ϵᶻ)]
parsed_expressions = [parse_expression(expr) for expr in expressions]



sol_mat = 𝓂.solution.perturbation.first_order.solution_matrix

function bound_violations(x,p)
    # shock_history[2:end,1] = x

    # Y[:,1,1] = sol_mat * [initial_state[𝓂.timings.past_not_future_and_mixed_idx]; shock_history[:,1]] #state_update(initial_state,shock_history[:,1])

    # for t in 1:periods-1
    #     Y[:,t+1,1] = sol_mat * [Y[𝓂.timings.past_not_future_and_mixed_idx,t,1]; shock_history[:,t+1]] #state_update(Y[:,t,1],shock_history[:,t+1])
    # end

    # Y .+= reference_steady_state[1:T.nVars]

    # target = sum(abs,Y[4,:,:] .- max.(0,Y[5,:,:])) + sum(abs2,x)
    return sum(x.^2)#target
end

# bound_violation(4*zeros(21),[])

# bound_violations([zeros(14)...,2,zeros(6)...],[])

function shock_constraints(res, x, p)
    # shock_history[2:end,1] = x

    Y = zeros(Real,T.nVars,periods,1)
    Y[:,1,1] = sol_mat * [initial_state[𝓂.timings.past_not_future_and_mixed_idx]; vcat(shock_history[1,1],x)] #state_update(initial_state,shock_history[:,1])

    for t in 1:periods-1
        Y[:,t+1,1] = sol_mat * [Y[𝓂.timings.past_not_future_and_mixed_idx,t,1]; shock_history[:,t+1]] #state_update(Y[:,t,1],shock_history[:,t+1])
    end

    Y .+= reference_steady_state[1:T.nVars]

    res .= vcat(x, Y[4,:,:])
end

# T.exo

x0 = zeros(T.nExo-1)
# x0[indexin([:ϵᶻᵒᵇᶜ⁽⁻⁰⁾], T.exo)[1] - 1] = 2

# bound_violations(x0*.2,())
# shock_constraints(zeros(T.nExo-1 + periods), x0, [])

# optprob = OptimizationFunction(bound_violations, Optimization.AutoForwardDiff(), cons = shock_constraints)
# prob = OptimizationProblem(optprob, x0, (), lcons = zeros(T.nExo-1 + periods), ucons = fill(Inf, T.nExo-1 + periods))

# # vcat(zeros(14),3,zeros(6 + periods))[16]
# # T.exo[16]
# # shock_constraints(zeros(21 + periods),vcat(zeros(14),2,zeros(6 + periods)),[])

# sol = solve(prob, Ipopt.Optimizer())

# sol = solve(prob, MadNLP.Optimizer())

# sol = solve(prob, AmplNLWriter.Optimizer("couenne"))




function calc_state(x::Vector{S}) where S
    Y = zeros(AffExpr,T.nVars,periods,1)
    
    Y[:,1,1] = state_update(initial_state, vcat(shock_history[1,1],x))

    for t in 1:periods-1
        Y[:,t+1,1] = state_update(Y[:,t,1],shock_history[:,t+1])
    end

    Y .+= reference_steady_state[1:T.nVars]
    return Y[4,:,:]
end


using BenchmarkTools, JuMP, StatusSwitchingQP
# using BenchmarkTools, JuMP, Ipopt, COSMO, Clarabel, DAQP, HiGHS, MadNLP, OSQP, SCS, StatusSwitchingQP, Hypatia
# import MultiObjectiveAlgorithms as MOA
# model = Model(Ipopt.Optimizer)

import LinearAlgebra as ℒ



model = Model(StatusSwitchingQP.Optimizer)
set_silent(model)
@variable(model, x[i = 1:T.nExo - 1] >= 0)
@objective(model, Min, x' * ℒ.I * x)
@constraint(model, calc_state(x) .>= 0)
JuMP.optimize!(model)

value.(x)

@benchmark begin
    model = Model(COSMO.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end


@benchmark begin
    model = Model(MadNLP.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end

# favorite
@benchmark begin
    model = Model(StatusSwitchingQP.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end

value.(x)

@benchmark begin
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end


@benchmark begin
    model = Model(Hypatia.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end





@benchmark begin
    model = Model(SCS.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end


@benchmark begin
    model = Model(OSQP.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end


@benchmark begin
    model = Model(DAQP.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end


@benchmark begin
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end



@benchmark begin
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end


@benchmark begin
    model = Model(ProxSDP.Optimizer)
    set_silent(model)
    @variable(model, x[1:11] >= 0)
    @objective(model, Min, x'*Q*x)
    @constraint(model, calc_state(x) .>= 0)
    JuMP.optimize!(model)
end




solution_summary(model)

value.(x)



# sol = solve(pro, SCIP.Optimizer())
# sol = solve(prob, Pavito.Optimizer())
# sol = solve(prob, Juniper.Optimizer())
# sol = solve(prob, COSMO.Optimizer())
# sol = solve(prob, BARON.Optimizer())
# sol = solve(prob, knitro.Optimizer())
# sol = solve(prob, highs.Optimizer())
# sol = solve(prob, EAGO.Optimizer())

bound_violations(sol*.9, ())
shock_constraints(zeros(T.nExo-1 + periods), sol, [])

sol = solve(prob, NelderMead())

sol = solve(prob, LBFGS(linesearch = LineSearches.BackTracking(order=3)))


sol = solve(prob, IPNewton())




T.var

Y .+ reference_steady_state[1:𝓂.timings.nVars]


using StatsPlots
plot_irf(testmax, variables = :r, shocks = :ϵᶻ, parameters = :σᶻ=> 1, negative_shock = true)


:(r[0] = max(r̄,r̂[0])+1 | ϵᶻ > 0) |> dump

eq = :(z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x])
eq|>dump

obc_shock = :ϵᶻ

max_obc_shift = 40
obc_shifts = [Expr(:ref,Meta.parse(string(obc_shock) * "ᵒᵇᶜ⁽⁻"*super(string(i))*"⁾"),:(x-$i)) for i in 1:max_obc_shift]

super("1")
eqs = postwalk(x -> 
                x isa Expr ?
                    x.head == :ref ?
                        x.args[1] == obc_shock ?
                            Expr(:call,:+, x, obc_shifts...) :
                        x :
                    x :
                x,
            eq)


:(1+2+3)|>dump
Expr(:+,1,1,2)
eq = :(r[0] = max(r̄,r̂[0]) | ϵᶻ > 0)
import MacroTools: postwalk
function parse_occasionally_binding_constraints(equations_block)
    eqs = []
    condition_list = []

    for arg in equations_block.args
        if isa(arg,Expr)
            condition = []
            eq = postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        x.args[1] ∈ [:>, :<, :≤, :≥] ?
                            x.args[2].args[1] == :| ?
                                begin
                                    condition = Expr(x.head, x.args[1], x.args[2].args[end], x.args[end])
                                    x.args[2].args[2]
                                end :
                            x :
                        x :
                    x :
                x,
            arg)
            push!(condition_list, condition)
            push!(eqs,eq)
        end
    end

    return Expr(:block,eqs...), condition_list
end

eq = :(begin
    1  /  c[0] = (β  /  c[1]) * (r[1] + (1 - δ))

    r̂[0] = α * exp(z[0]) * k[-1]^(α - 1)

    r[0] = max(r̄,r̂[0]) | ϵᶻ > 0

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
end)
model_ex, condition_list = parse_occasionally_binding_constraints(eq)


obc_shock_bounds = Tuple{Symbol, Bool, Float64}[]

for c in condition_list
    if c isa Expr
        push!(obc_shock_bounds, parse_obc_shock_bounds(c))
    end
end

obc_shocks = Symbol[]

for a in condition_list 
    if a isa Expr
        s = get_symbols(a)
        for ss in s
            push!(obc_shocks,ss)
        end
    end
end


max_obc_shift = 40

eqs_with_obc_shocks = []
for eq in eqs
    eqq = postwalk(x -> 
                    x isa Expr ?
                        x.head == :ref ?
                            x.args[1] ∈ obc_shocks ?
                                begin
                                    obc_shock = intersect(x.args[1], obc_shocks)
                                    obc_shifts = [Expr(:ref,Meta.parse(string(obc_shock) * "ᵒᵇᶜ⁽⁻"*super(string(i))*"⁾"),:(x-$i)) for i in 1:max_obc_shift]
                                    Expr(:call,:+, x, obc_shifts...) 
                                end :
                            x :
                        x :
                    x,
    eq)
    push!(eqs_with_obc_shocks, eqq)
end


function parse_occasionally_binding_constraints(equations_block; max_obc_shift::Int = 20)
    eqs = []
    condition_list = []

    for arg in equations_block.args
        if isa(arg,Expr)
            condition = []
            eq = postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        x.args[1] ∈ [:>, :<, :≤, :≥] ?
                            x.args[2].args[1] == :| ?
                                begin
                                    condition = Expr(x.head, x.args[1], x.args[2].args[end], x.args[end])
                                    x.args[2].args[2]
                                end :
                            x :
                        x :
                    x :
                x,
            arg)
            push!(condition_list, condition)
            push!(eqs,eq)
        end
    end

    obc_shocks = Symbol[]

    for a in condition_list 
        if a isa Expr
            s = get_symbols(a)
            for ss in s
                push!(obc_shocks,ss)
            end
        end
    end

    eqs_with_obc_shocks = []
    for eq in eqs
        eqq = postwalk(x -> 
                        x isa Expr ?
                            x.head == :ref ?
                                x.args[1] ∈ obc_shocks ?
                                    begin
                                        obc_shock = intersect([x.args[1]], obc_shocks)[1]
                                        obc_shifts = [Expr(:ref,Meta.parse(string(obc_shock) * "ᵒᵇᶜ⁽⁻"*super(string(i))*"⁾"),:(x-$i)) for i in 1:max_obc_shift]
                                        Expr(:call,:+, x, obc_shifts...) 
                                    end :
                                x :
                            x :
                        x,
        eq)
        push!(eqs_with_obc_shocks, eqq)
    end

    return Expr(:block,eqs_with_obc_shocks...), condition_list
end


parse_occasionally_binding_constraints(eq)

separate_shock_related_to_constraint_from_equation(eq)

Expr(:call,:>,0,1)


SS(testmax)
SS(testmax, parameters = :r̄ => 0.00)

get_irf(testmax)
get_solution(testmax)
testmax.SS_solve_func(testmax.parameter_values, testmax, true)


symbolics = create_symbols_eqs!(testmax)
remove_redundant_SS_vars!(testmax, symbolics) 

solve_for = :r̂
eq = symbolics.ss_equations[3]|>string|>Meta.parse

eq |> get_symbols

eqs = postwalk(x -> 
    x isa Expr ?
        x.head == :call ? 
            x.args[1] ∈ [:Max,:Min] ?
                solve_for ∈ get_symbols(x.args[2]) ?
                    x.args[2] :
                solve_for ∈ get_symbols(x.args[3]) ?
                    x.args[3] :
                x :
            x :
        x :
    x,
eq)


r̄ = 0
σᶻ= 0.01
ρᶻ= 0.2
δ = 0.02
α = 0.5
β = 0.95

r = (δ - 1) + 1 / β
z = 0
r = max(r̂, r̄)
k = (r̂ / α) ^ (1 / (α - 1))
q = k ^ α
c = -k * δ + q


using SymPyPythonCall

@syms a,b,c
Symbol(a)
a = max(b,c)

solve(a - max(b,c),b)

get_SS(testmax, parameters = :r̄ => 0)

testmax.SS_solve_func
max(0,1)

import MacroTools: postwalk, unblock
import MacroModelling: parse_for_loops, convert_to_ss_equation, simplify,create_symbols_eqs!,remove_redundant_SS_vars!,get_symbols, parse_occasionally_binding_constraints, parse_algorithm_to_state_update
import DataStructures: CircularBuffer
import Subscripts: super, sub


precompile_model = false

ex = :(begin 
    1  /  c[0] = (β  /  c[1]) * (r[1] + (1 - δ))

    r̂[0] = α * exp(z[0]) * k[-1]^(α - 1)

    r[0] = max(r̄,r̂[0])

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
end)




# create data containers
parameters = []
parameter_values = Vector{Float64}(undef,0)

ss_calib_list = []
par_calib_list = []

solved_vars = [] 
solved_vals = []

ss_solve_blocks = []

NSSS_solver_cache = CircularBuffer{Vector{Vector{Float64}}}(500)
SS_solve_func = x->x
SS_dependencies = nothing

original_equations = []
calibration_equations = []
calibration_equations_parameters = []

bounded_vars = []
lower_bounds = []
upper_bounds = []

dyn_equations = []

➕_vars = []
ss_and_aux_equations = []
aux_vars_created = Set()

unique_➕_vars = []

ss_eq_aux_ind = Int[]
dyn_eq_aux_ind = Int[]

model_ex = parse_for_loops(ex)


model_ex.args[1]

# write down dynamic equations and add auxilliary variables for leads and lags > 1
for (i,arg) in enumerate(model_ex.args)
    if isa(arg,Expr)
        # write down dynamic equations
        t_ex = postwalk(x -> 
            x isa Expr ? 
                x.head == :(=) ? 
                    Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                    x.head == :ref ?
                        occursin(r"^(x|ex|exo|exogenous){1}$"i,string(x.args[2])) ?
                            begin
                                Symbol(string(x.args[1]) * "₍ₓ₎") 
                            end :
                        occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}(\-|\+){1}\s{1}\d+$))"i,string(x.args[2])) ?
                            x.args[2].args[1] == :(+) ?
                                begin
                                    k = x.args[2].args[3]
            
                                    while k > 2 # create auxilliary dynamic equation for exogenous variables with lead > 1
                                        if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎") ∈ aux_vars_created
                                            break
                                        else
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))
                
                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 2)) * "⁾₍₁₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                            
                                            k -= 1
                                        end
                                    end

                                    if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎") ∉ aux_vars_created && k > 1
                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))
                
                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₁₎")))
                                        push!(dyn_eq_aux_ind,length(dyn_equations))
                                    end

                                    if Symbol(string(x.args[1]) * "₍₀₎") ∉ aux_vars_created
                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "₍₀₎"))
                                        
                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "₍₀₎"),Symbol(string(x.args[1]) * "₍ₓ₎")))
                                        push!(dyn_eq_aux_ind,length(dyn_equations))
                                    end

                                    if x.args[2].args[3] > 1
                                        Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(x.args[2].args[3] - 1)) * "⁾₍₁₎")
                                    else
                                        Symbol(string(x.args[1]) * "₍₁₎")
                                    end
                                end :
                            x.args[2].args[1] == :(-) ?
                                begin
                                    k = - x.args[2].args[3]
                
                                    while k < -2 # create auxilliary dynamic equations for exogenous variables with lag < -1
                                        if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") ∈ aux_vars_created
                                            break
                                        else
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                
                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 2)) * "⁾₍₋₁₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                            
                                            k += 1
                                        end
                                    end
                
                                    if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") ∉ aux_vars_created && k < -1
                                    
                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))
                
                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₋₁₎")))
                                        push!(dyn_eq_aux_ind,length(dyn_equations))
                                    end
                                    
                                    if Symbol(string(x.args[1]) * "₍₀₎") ∉ aux_vars_created
                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "₍₀₎"))
                                        
                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "₍₀₎"),Symbol(string(x.args[1]) * "₍ₓ₎")))
                                        push!(dyn_eq_aux_ind,length(dyn_equations))
                                    end

                                    if  - x.args[2].args[3] < -1
                                        Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(x.args[2].args[3] - 1)) * "⁾₍₋₁₎")
                                    else
                                        Symbol(string(x.args[1]) * "₍₋₁₎")
                                    end
                                end :
                            x.args[1] : 
                        occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                            begin
                                Symbol(string(x.args[1]) * "₍ₛₛ₎") 
                            end :
                        x.args[2] isa Int ? 
                            x.args[2] > 1 ? 
                                begin
                                    k = x.args[2]

                                    while k > 2 # create auxilliary dynamic equations for endogenous variables with lead > 1
                                        if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎") ∈ aux_vars_created
                                            break
                                        else
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))

                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 2)) * "⁾₍₁₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                            
                                            k -= 1
                                        end
                                    end

                                    if Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎") ∉ aux_vars_created
                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"))

                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(k - 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₁₎")))
                                        push!(dyn_eq_aux_ind,length(dyn_equations))
                                    end
                                    Symbol(string(x.args[1]) * "ᴸ⁽" * super(string(x.args[2] - 1)) * "⁾₍₁₎")
                                end :
                            1 >= x.args[2] >= 0 ? 
                                begin
                                    Symbol(string(x.args[1]) * "₍" * sub(string(x.args[2])) * "₎")
                                end :  
                            -1 <= x.args[2] < 0 ? 
                                begin
                                    Symbol(string(x.args[1]) * "₍₋" * sub(string(x.args[2])) * "₎")
                                end :
                            x.args[2] < -1 ?  # create auxilliary dynamic equations for endogenous variables with lag < -1
                                begin
                                    k = x.args[2]

                                    while k < -2
                                        if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") ∈ aux_vars_created
                                            break
                                        else
                                            push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))

                                            push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 2)) * "⁾₍₋₁₎")))
                                            push!(dyn_eq_aux_ind,length(dyn_equations))
                                            
                                            k += 1
                                        end
                                    end

                                    if Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎") ∉ aux_vars_created
                                        push!(aux_vars_created,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"))

                                        push!(dyn_equations,Expr(:call,:-,Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(k + 1)) * "⁾₍₀₎"),Symbol(string(x.args[1]) * "₍₋₁₎")))
                                        push!(dyn_eq_aux_ind,length(dyn_equations))
                                    end

                                    Symbol(string(x.args[1]) * "ᴸ⁽⁻" * super(string(x.args[2] + 1)) * "⁾₍₋₁₎")
                                end :
                        x.args[1] :
                    x.args[1] : 
                unblock(x) : 
            x,
        model_ex.args[i])

        push!(dyn_equations,unblock(t_ex))
        
        # write down ss equations including nonnegativity auxilliary variables
        # find nonegative variables, parameters, or terms
        eqs = postwalk(x -> 
            x isa Expr ? 
                x.head == :(=) ? 
                    Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                        x.head == :ref ?
                            occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
                    x : 
                x.head == :call ?
                    # x.args[1] == :max ?
                    x.args[1] == :* ?
                        x.args[2] isa Int ?
                            x.args[3] isa Int ?
                                x :
                            Expr(:call, :*, x.args[3:end]..., x.args[2]) : # 2beta => beta * 2 
                        x :
                    x.args[1] ∈ [:^] ?
                        !(x.args[3] isa Int) ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                    begin
                                        push!(bounded_vars,x.args[2])
                                        push!(lower_bounds,eps(Float32))
                                        push!(upper_bounds,1e12+rand())
                                        x
                                    end :
                            x.args[2].head == :ref ?
                                x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                    begin
                                        push!(bounded_vars,x.args[2].args[1])
                                        push!(lower_bounds,eps(Float32))
                                        push!(upper_bounds,1e12+rand())
                                        x
                                    end :
                                x :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile_model
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if x.args[2] ∈ unique_➕_vars
                                            ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                            replacement = Expr(:ref,Symbol("➕" * sub(string(➕_vars_idx))),0)
                                        else
                                            push!(unique_➕_vars,x.args[2])
                                            push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                            push!(lower_bounds,eps(Float32))
                                            push!(upper_bounds,1e12+rand())
                                            push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the cond_var_decomp
                                            push!(ss_eq_aux_ind,length(ss_and_aux_equations))

                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)
                                        end
                                    end

                                    :($(replacement) ^ $(x.args[3]))
                                end :
                            x :
                        x :
                    x.args[2] isa Float64 ?
                        x :
                    x.args[1] ∈ [:log] ?
                        x.args[2] isa Symbol ? # nonnegative parameters 
                            begin
                                push!(bounded_vars,x.args[2])
                                push!(lower_bounds,eps(Float32))
                                push!(upper_bounds,1e12+rand())
                                x
                            end :
                        x.args[2].head == :ref ?
                            x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                begin
                                    push!(bounded_vars,x.args[2].args[1])
                                    push!(lower_bounds,eps(Float32))
                                    push!(upper_bounds,1e12+rand())
                                    x
                                end :
                            x :
                        x.args[2].head == :call ? # nonnegative expressions
                            begin
                                if precompile_model
                                    replacement = x.args[2]
                                else
                                    replacement = simplify(x.args[2])
                                end

                                if !(replacement isa Int) # check if the nonnegative term is just a constant
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Expr(:ref,Symbol("➕" * sub(string(➕_vars_idx))),0)
                                    else
                                        push!(unique_➕_vars,x.args[2])
                                        push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                        push!(lower_bounds,eps(Float32))
                                        push!(upper_bounds,1e12+rand())
                                        push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the code
                                        push!(ss_eq_aux_ind,length(ss_and_aux_equations))
                                        
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)
                                    end
                                end
                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x :
                    x.args[1] ∈ [:norminvcdf, :norminv, :qnorm] ?
                        x.args[2] isa Symbol ? # nonnegative parameters 
                            begin
                                push!(bounded_vars,x.args[2])
                                push!(lower_bounds,eps())
                                push!(upper_bounds,1-eps())
                                x
                            end :
                        x.args[2].head == :ref ?
                            x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                begin
                                    push!(bounded_vars,x.args[2].args[1])
                                    push!(lower_bounds,eps())
                                    push!(upper_bounds,1-eps())
                                    x
                                end :
                            x :
                        x.args[2].head == :call ? # nonnegative expressions
                            begin
                                if precompile_model
                                    replacement = x.args[2]
                                else
                                    replacement = simplify(x.args[2])
                                end

                                if !(replacement isa Int) # check if the nonnegative term is just a constant
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Expr(:ref,Symbol("➕" * sub(string(➕_vars_idx))),0)
                                    else
                                        push!(unique_➕_vars,x.args[2])
                                        push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                        push!(lower_bounds,eps())
                                        push!(upper_bounds,1-eps())

                                        push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the code
                                        push!(ss_eq_aux_ind,length(ss_and_aux_equations))
                                        
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)
                                    end
                                end
                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x :
                    x.args[1] ∈ [:exp] ?
                        x.args[2] isa Symbol ? # nonnegative parameters 
                            begin
                                push!(bounded_vars,x.args[2])
                                push!(lower_bounds,-1e12+rand())
                                push!(upper_bounds,700)
                                x
                            end :
                        x.args[2].head == :ref ?
                            x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                begin
                                    push!(bounded_vars,x.args[2].args[1])
                                    push!(lower_bounds,-1e12+rand())
                                    push!(upper_bounds,700)
                                    x
                                end :
                            x :
                        x.args[2].head == :call ? # nonnegative expressions
                            begin
                                if precompile_model
                                    replacement = x.args[2]
                                else
                                    replacement = simplify(x.args[2])
                                end

                                # println(replacement)
                                if !(replacement isa Int) # check if the nonnegative term is just a constant
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Expr(:ref,Symbol("➕" * sub(string(➕_vars_idx))),0)
                                    else
                                        push!(unique_➕_vars,x.args[2])
                                        push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                        push!(lower_bounds,-1e12+rand())
                                        push!(upper_bounds,700)

                                        push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the code
                                        push!(ss_eq_aux_ind,length(ss_and_aux_equations))
                                        
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)
                                    end
                                end
                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x :
                    x.args[1] ∈ [:erfcinv] ?
                        x.args[2] isa Symbol ? # nonnegative parameters 
                            begin
                                push!(bounded_vars,x.args[2])
                                push!(lower_bounds,eps())
                                push!(upper_bounds,2-eps())
                                x
                            end :
                        x.args[2].head == :ref ?
                            x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                begin
                                    push!(bounded_vars,x.args[2].args[1])
                                    push!(lower_bounds,eps())
                                    push!(upper_bounds,2-eps())
                                    x
                                end :
                            x :
                        x.args[2].head == :call ? # nonnegative expressions
                            begin
                                if precompile_model
                                    replacement = x.args[2]
                                else
                                    replacement = simplify(x.args[2])
                                end

                                # println(replacement)
                                if !(replacement isa Int) # check if the nonnegative term is just a constant
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Expr(:ref,Symbol("➕" * sub(string(➕_vars_idx))),0)
                                    else
                                        push!(unique_➕_vars,x.args[2])
                                        push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                        push!(lower_bounds,eps())
                                        push!(upper_bounds,2-eps())
                                        push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)+1))),0))), x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the code
                                        push!(ss_eq_aux_ind,length(ss_and_aux_equations))
                                        
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Expr(:ref,Symbol("➕" * sub(string(length(➕_vars)))),0)
                                    end
                                end
                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x :
                    x :
                x :
            x,
        model_ex.args[i])
        push!(ss_and_aux_equations,unblock(eqs))
    end
end


# go through changed SS equations including nonnegative auxilliary variables
ss_aux_equations = []

# tag vars and pars in changed SS equations
var_list_aux_SS = []
ss_list_aux_SS = []
par_list_aux_SS = []

var_future_list_aux_SS = []
var_present_list_aux_SS = []
var_past_list_aux_SS = []

# # label all variables parameters and exogenous variables and timings for changed SS equations including nonnegativity auxilliary variables
for (idx,eq) in enumerate(ss_and_aux_equations)
    var_tmp = Set()
    ss_tmp = Set()
    par_tmp = Set()
    var_future_tmp = Set()
    var_present_tmp = Set()
    var_past_tmp = Set()

    # remove terms multiplied with 0
    eq = postwalk(x -> 
        x isa Expr ? 
            x.head == :call ? 
                x.args[1] == :* ?
                    any(x.args[2:end] .== 0) ? 
                        0 :
                    x :
                x :
            x :
        x,
    eq)

    # label all variables parameters and exogenous variables and timings for individual equations
    postwalk(x -> 
        x isa Expr ? 
            x.head == :call ? 
                for i in 2:length(x.args)
                    x.args[i] isa Symbol ? 
                        occursin(r"^(ss|stst|steady|steadystate|steady_state|x|ex|exo|exogenous){1}$"i,string(x.args[i])) ? 
                            x :
                        push!(par_tmp,x.args[i]) : 
                    x
                end :
            x.head == :ref ? 
                x.args[2] isa Int ? 
                    x.args[2] == 0 ? 
                        push!(var_present_tmp,x.args[1]) : 
                    x.args[2] > 0 ? 
                        push!(var_future_tmp,x.args[1]) : 
                    x.args[2] < 0 ? 
                        push!(var_past_tmp,x.args[1]) : 
                    x :
                occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\-{1}\s{1}\d+$))"i,string(x.args[2])) ?
                    push!(var_past_tmp,x.args[1]) : 
                occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\+{1}\s{1}\d+$))"i,string(x.args[2])) ?
                    push!(var_future_tmp,x.args[1]) : 
                occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    push!(ss_tmp,x.args[1]) :
                x : 
            x :
        x,
    eq)

    var_tmp = union(var_future_tmp,var_present_tmp,var_past_tmp)
    
    push!(var_list_aux_SS,var_tmp)
    push!(ss_list_aux_SS,ss_tmp)
    push!(par_list_aux_SS,par_tmp)
    push!(var_future_list_aux_SS,var_future_tmp)
    push!(var_present_list_aux_SS,var_present_tmp)
    push!(var_past_list_aux_SS,var_past_tmp)


    # write down SS equations including nonnegativity auxilliary variables
    prs_ex = convert_to_ss_equation(eq)
    
    if idx ∈ ss_eq_aux_ind
        if precompile_model
            ss_aux_equation = Expr(:call,:-,unblock(prs_ex).args[2],unblock(prs_ex).args[3]) 
        else
            ss_aux_equation = Expr(:call,:-,unblock(prs_ex).args[2],simplify(unblock(prs_ex).args[3])) # simplify RHS if nonnegative auxilliary variable
        end
    else
        if precompile_model
            ss_aux_equation = unblock(prs_ex)
        else
            ss_aux_equation = simplify(unblock(prs_ex))
        end
    end
    ss_aux_equation_expr = if ss_aux_equation isa Symbol Expr(:call,:-,ss_aux_equation,0) else ss_aux_equation end

    push!(ss_aux_equations,ss_aux_equation_expr)
end