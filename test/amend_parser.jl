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

using Optimization, Ipopt, OptimizationMOI, OptimizationOptimJL
import MacroTools: postwalk, unblock
import MacroModelling: parse_for_loops, convert_to_ss_equation, simplify,create_symbols_eqs!,remove_redundant_SS_vars!,get_symbols, parse_occasionally_binding_constraints, parse_algorithm_to_state_update
import DataStructures: CircularBuffer
import Subscripts: super, sub

𝓂 = testmax
algorithm = :first_order
state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂)
T = 𝓂.timings
periods = 40

initial_state = zeros(Real, T.nVars)
Y = zeros(Real,T.nVars,periods,1)
T.exo
shock_history = zeros(Real,T.nExo,periods)
shock_history[1,1] = -3


reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())
shock_history[16,1]

sol_mat = 𝓂.solution.perturbation.first_order.solution_matrix
function bound_violations(x,p)
    shock_history[2:end,1] = x

    Y[:,1,1] = sol_mat * [initial_state[𝓂.timings.past_not_future_and_mixed_idx]; shock_history[:,1]] #state_update(initial_state,shock_history[:,1])

    for t in 1:periods-1
        Y[:,t+1,1] = sol_mat * [Y[𝓂.timings.past_not_future_and_mixed_idx,t,1]; shock_history[:,t+1]] #state_update(Y[:,t,1],shock_history[:,t+1])
    end

    Y .+= reference_steady_state[1:T.nVars]

    target = sum(abs,Y[4,:,:] .- max.(0,Y[5,:,:])) + sum(abs2,x)
    return target
end

bound_violation(4*zeros(21),[])

bound_violation([zeros(14)...,2,zeros(6)...],[])

T.exo[16]
function shock_constraints(res, x, p)
    res .= x
end

x0 = zeros(21) .+ .1

optprob = OptimizationFunction(bound_violations, Optimization.AutoForwardDiff(), cons = shock_constraints)
prob = OptimizationProblem(optprob, x0, (), lcons = zeros(21), ucons = fill(Inf,21))

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