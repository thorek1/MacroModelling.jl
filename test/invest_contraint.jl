import Subscripts: super

obc_shocks = []
eqs = []

max_obc_shift = 3

push!(obc_shocks, Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0))


obc_name = "ϵᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1))

obc_var = Expr(:ref, Meta.parse(obc_name * "ꜝᴸ⁽⁻⁰⁾"), 0)
obc_shock = Expr(:ref, Meta.parse(obc_name * "ꜝ⁽⁻⁰⁾"), :x)


for obc in obc_shocks
    push!(eqs, :($(obc) = $(Expr(:ref, obc.args[1], -1)) * 0.9 + $obc_var + $obc_shock))

    for i in 1:max_obc_shift
        push!(eqs, :($(Expr(:ref, Meta.parse(obc_name * "ꜝᴸ⁽⁻" * super(string(i)) * "⁾"), 0)) = $(Expr(:ref, Meta.parse(obc_name * "ꜝᴸ⁽⁻" * super(string(i-1)) * "⁾"), -1)) + $(Expr(:ref, Meta.parse(obc_name * "ꜝ⁽⁻" * super(string(i)) * "⁾"), :x))))
    end
end


# i = 1
# :($(Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝᴸ⁽⁻" * super(string(i)) * "⁾"), 0)) = $(Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝᴸ⁽⁻" * super(string(i-1)) * "⁾"), -i)) + $(Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ⁽⁻" * super(string(i)) * "⁾"), :x)))





# ϵll⁻¹[0] = ϵll⁻²[-1] + ϵll⁻¹[x]



# :(min(b₍₀₎ - M * y₍₀₎, -lb₍₀₎)) |> dump

# :( isapprox(-lb₍₀₎, 0, atol = 1e-14) ? b₍₀₎ - M * y₍₀₎ : -lb₍₀₎) |> dump


# import MacroTools: postwalk

# exp = :(α * exp(z[0]) * k[-1]^(α - 1)) 
# exp |> dump




# function check_for_dynamic_variables(ex::Expr)
#     dynamic_indicator = Bool[]

#     postwalk(x -> 
#         x isa Expr ?
#             x.head == :ref ? 
#                 occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
#                     x :
#                 begin
#                     push!(dynamic_indicator,true)
#                     x
#                 end :
#             x :
#         x,
#     ex)

#     any(dynamic_indicator)
# end



using MacroModelling

import_model("test/GI2017.mod")



using MacroModelling

@model testmax begin
    1  /  c[0] = (β  /  c[1]) * (r[1] + (1 - δ))

    r̂[0] = α * exp(z[0]) * k[-1]^(α - 1)

    # r̂[0] = max(r̄, r[0])
    # 0 = max(r̄ - r[0], r̂[0] - r[0])
    r[0] = max(r̄, r̂[0])

    # r̂[0] = r[0] + ϵll[x-3]

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

    # ϵll⁻¹[0] = ϵll⁻²[-1] + ϵll⁻¹[x]

    # ϵll⁻²[0] = ϵll⁻³[-1] + ϵll⁻²[x]

    # ϵll⁻³[0] = ϵll⁻³[x]

end

@parameters testmax begin
    r̄ = 0
    σᶻ= 1#0.01
    ρᶻ= 0.8#2
    δ = 0.02
    α = 0.5
    β = 0.95
end

SS(testmax)



@model testmax begin
    1  /  c[0] = (β  /  c[1]) * (r[1] + (1 - δ))

    r̂[0] = α * exp(z[0]) * k[-1]^(α - 1)

    # r̂[0] = max(r̄, r[0])
    # 0 = max(r̄ - r[0], r̂[0] - r[0])
    # r[0] = max(r̄, r̂[0])

    r̂[0] = r[0]# + ϵll[x-3]

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

    # ϵll⁻¹[0] = ϵll⁻²[-1] + ϵll⁻¹[x]

    # ϵll⁻²[0] = ϵll⁻³[-1] + ϵll⁻²[x]

    # ϵll⁻³[0] = ϵll⁻³[x]

end

@parameters testmax begin
    r̄ = 0
    σᶻ= 1#0.01
    ρᶻ= 0.8#2
    δ = 0.02
    α = 0.5
    β = 0.95
end



SSS(testmax, parameters = :σᶻ => .1, algorithm = :second_order)

get_solution(testmax, algorithm = :second_order)
get_solution(testmax)
testmax.ss_aux_equations
testmax.dyn_equations
testmax.parameters
testmax.exo
:(ϵllᴸ⁽⁻²⁾₍₀₎ - ϵllᴸ⁽⁻¹⁾₍₋₁₎)
:(ϵllᴸ⁽⁻¹⁾₍₀₎ - ϵll₍₋₁₎)
:(ϵll₍₀₎ - ϵll₍ₓ₎)

𝓂 = testmax

obc_shock_idx = contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")

periods_per_shock = 𝓂.max_obc_shift + 1

num_shocks = sum(obc_shock_idx)÷periods_per_shock

relevent_part = match.(r"ᵒᵇᶜ.*(?=⁽)", string.(𝓂.timings.exo))
isnothing(relevent_part[end])

for i in relevent_part
    println(!isnothing(i))
    # if !isnothing(relevent_part)
    #     # println(i)
    #     i.match
    # end
end
testmax.dyn_equations

testmax.ss_aux_equations

:(ϵᵒᵇᶜ⁺ꜝ¹ꜝ₍₀₎ - (ϵᵒᵇᶜ⁺ꜝ¹ꜝ₍₋₁₎ * 0.9 + ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁰⁾₍₀₎ + ϵᵒᵇᶜ⁺ꜝ²ꜝ⁽⁻⁰⁾₍ₓ₎))
:(ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻¹⁾₍₀₎ - (ϵᵒᵇᶜ⁺ꜝ²ꜝᴸ⁽⁻⁰⁾₍₋₁₎ + ϵᵒᵇᶜ⁺ꜝ²ꜝ⁽⁻¹⁾₍ₓ₎))
import StatsPlots

plot_irf(testmax, ignore_obc = true)

plot_irf(testmax, ignore_obc = true, variables = :all)
plot_irf(testmax, variables = :all)

plot_irf(testmax, ignore_obc = false)

plot_irf(testmax, negative_shock = false, parameters = (:σᶻ => 8.0, :r̄ => .0))

plot_irf(testmax, negative_shock = false, parameters = :σᶻ => 8, variables = :all)
plot_irf(testmax, ignore_obc = true, negative_shock = true, parameters = :σᶻ => .1, variables = :all, algorithm = :second_order)

plot_irf(testmax, negative_shock = false, parameters = (:r̄ => .05,:σᶻ => 8.0), variables = :all)

plot_irf(testmax, negative_shock = false, parameters = :σᶻ => -1.1, ignore_obc = true)
plot_irf(testmax, negative_shock = false, ignore_obc = true, variables = :all)

plot_irf(testmax, negative_shock = false, ignore_obc = true, variables = :all, shocks = :ϵᵒᵇᶜ⁺ꜝ¹ꜝ⁽⁰⁾)

plot_irf(testmax, negative_shock = false, ignore_obc = true, shocks = :ϵᵒᵇᶜ⁺ꜝ¹ꜝˡ⁽⁰⁾, variables = :all)
plot_irf(testmax, negative_shock = false, ignore_obc = true, shocks = :ϵᵒᵇᶜ⁺ꜝ¹ꜝʳ⁽⁰⁾, variables = :all)

testmax.obc_violation_equations
testmax.obc_violation_function


testmax.exo

using MacroModelling

@model RBC begin
    c[0]^(-γ) - λ[0] = β * (c[1]^(-γ) * ((1 - δ) + α * exp(z[1]) * k[0]^(α - 1)) - (1 - δ) * λ[1])
    # c[0]^(-γ) = β * (c[1]^(-γ) * ((1 - δ) + α * exp(z[1]) * k[0]^(α - 1)))
    c[0] + i[0] = exp(z[0]) * k[-1]^α
    k[0] = (1 - δ) * k[-1] + i[0]
    # λ[0] * (i[0] - ϕ * i[ss]) = λ1[0]
    # λ[0] = λ[-1] * .9 + .1 * λ[ss] - .0001*eps_λ[x-1]
    z[0] = ρ * z[-1] + std_z * eps_z[x]

    # i[0] ≥ ϕ * i[ss] | λ[0], eps_zⁱ > 0
    0 = min(i[0] - ϕ * i[ss], λ[0])# + λ1[0] #| eps_zⁱ > 0

    # î[0] = min(i[0], ϕ * i[ss]) | eps_zⁱ > 0
    # bind[0] =  i[0] - ϕ * i[ss]

    # λ1[0] = λ1[-1] * .9  - eps_λ[x]
    # i[0] - ϕ * i[ss] + λ1[0]
end

@parameters RBC begin
    std_z = 0.15
    std_zⁱ= 0.15
    ρ = 0.2
    ρⁱ= 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
    ϕ = .9
    γ = 1
    # i > 0
    # ϕ < 1
end

SS(RBC)
SSS(RBC, algorithm = :pruned_second_order)


RBC.obc_violation_function
get_solution(RBC)
get_solution(RBC)(:eps_z₍ₓ₎, :χᵒᵇᶜ⁻ꜝ¹ꜝˡ)
get_solution(RBC)(:eps_z₍ₓ₎, :i)
get_solution(RBC)(:eps_z₍ₓ₎, :χᵒᵇᶜ⁻ꜝ¹ꜝʳ)

# contains.(string.(RBC.timings.exo), "ᵒᵇᶜ")
using StatsPlots
plot_irf(RBC)

plot_irf(RBC, negative_shock = true)

plot_irf(RBC, variables = :all)

plot_irf(RBC, variables = :all, negative_shock = true)

plot_irf(RBC, ignore_obc = true, variables = :all)
plot_irf(RBC, ignore_obc = true, variables = :all, negative_shock = true)

plot_irf(RBC, ignore_obc = true, variables = :all, shocks = :all)

plot_irf(RBC, negative_shock = true, parameters = :std_z => .15)

plot_irf(RBC, negative_shock = true, ignore_obc = true)

plot_irf(RBC, ignore_obc = true, variables = :all, negative_shock = true)

plot_irf(RBC, ignore_obc = false, variables = :all, negative_shock = true)


plot_irf(RBC, ignore_obc = false, variables = :all, negative_shock = true)

plot_irf(RBC, ignore_obc = true, variables = :all, negative_shock = false)

plot_irf(RBC, ignore_obc = true, variables = :all, shocks = :all, negative_shock = false)

plot_irf(RBC, ignore_obc = false, variables = :all, negative_shock = true, parameters = :ϕ => .98)
plot_irf(RBC, ignore_obc = true, variables = :all,  shocks = :all)
plot_irf(RBC, ignore_obc = true, shocks = :eps_λ, variables = :all, parameters = :ϕ => .99)

get_solution(RBC, algorithm = :quadratic_iteration)

RBC.obc_violation_function

SS_and_pars, solution_error = RBC.SS_solve_func(RBC.parameter_values, RBC, true)

RBC.SS_solve_func

RBC.ss_solve_blocks[1]

RBC.dyn_equations#_aux_equations


using MacroModelling

@model borrcon begin
    c[0] = y[0] + b[0] - R * b[-1]

    # b[0] = M * y[0] + λ[0]
    # λ[0] = λ[-1] * .9 - .01 * eps_λ[x] #  + .1 * λ[ss]
    # 0 = -max(b[0] - M * y[0], -lb[0])
    0 = max(bind[0], -lb[0])

    bind[0] = b[0] - M * y[0]

    lb[0] = 1/c[0]^GAMMAC - BETA * R / c[1]^GAMMAC

    log(y[0]) = RHO * log(y[-1]) + SIGMA * u[x]
end

@parameters borrcon begin
    R = 1.05
    BETA = 0.945
    RHO   = 0.9
    SIGMA = 0.05
    M = 1
    GAMMAC = 1
    # bind[ss] = 0.01 | BETA
    # lb[ss] = 0 | BETA
    # 1.0001 > b > .99999
    # c < 0.950001
    # lb < 1e-6
end

SS(borrcon)#, parameters = :M => -.1)
# SS(borrcon, parameters = :M => 1)

get_solution(borrcon, algorithm = :pruned_second_order)
SS_and_pars, solution_error = borrcon.SS_solve_func(borrcon.parameter_values, borrcon, true)

get_irf(borrcon)

using StatsPlots

plot_irf(borrcon)
plot_irf(borrcon, variables = :all)
plot_irf(borrcon, ignore_obc = true)
plot_irf(borrcon, ignore_obc = true, variables = :all)
plot_irf(borrcon, negative_shock = true, variables = :all)
plot_irf(borrcon, negative_shock = true)
plot_irf(borrcon, negative_shock = false, variables = :all, shocks = :all)
plot_irf(borrcon, negative_shock = false, ignore_obc = true, variables = :all, shocks = :all)
plot_irf(borrcon, parameters = :M => .99)


get_solution(borrcon)

borrcon.ss_aux_equations
borrcon.dyn_equations

borrcon.ss_solve_blocks
borrcon.SS_solve_func
borrcon.obc_violation_function
# assume that lagrange multiplier is always < 0


T  =borrcon.timings
reference_steady_state = borrcon.solution.non_stochastic_steady_state
Dict(borrcon.var .=> reference_steady_state[1:T.nVars])

borrcon.dyn_equations[4] |> dump
eq = borrcon.dyn_equations[4]
replace(string(:Χᵒᵇᶜꜝ¹ꜝ₍₀₎), "₍₀₎" => "")
import MacroTools: postwalk

postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        length(x.args) == 3 ?
                        x.args[3] isa Expr ?
                            x.args[3].args[1] ∈ [:Min, :min, :Max, :max] ?
                                    begin
                                        plchldr = replace(string(x.args[2]), "₍₀₎" => "")

                                        ineq_plchldr_1 = replace(string(x.args[3].args[2]), "₍₀₎" => "")

                                        :($plchldr ≈ $ineq_plchldr_1 ? $(x.args[3].args[2]) : $(x.args[3].args[3]))
                                    end :
                                x :
                            x :
                        x :
                    x :
                x,
            eq)

            







# indexin(contains.(string.(borrcon.var), "Χᵒᵇᶜ")

borrcon.var[contains.(string.(borrcon.var), "Χᵒᵇᶜ") .|| contains.(string.(borrcon.var), "χᵒᵇᶜ")]

obc_idxs = Set()
push!(obc_idxs, findall(x->contains(string(x), "Χᵒᵇᶜ") , borrcon.var)...)
push!(obc_idxs, findall(x->contains(string(x), "χᵒᵇᶜ") , borrcon.var)...)


steady_state_obc = []
for i in obc_idxs
    push!(steady_state_obc,:($(𝓂.var[i]) = reference_steady_state[$i]))
end



:(0 = max(b[0] - M * y[0], -lb[0]) + λ[0]) |> dump

arg = :(0 = max(b[0] - M * y[0], -lb[0]) + λ[0])
obc_shocks = []
eqs = []
postwalk(x -> 
                    x isa Expr ?
                        x.head == :call ? 
                            x.args[1] ∈ [:max, :min] ?
                                begin
                                    obc_shock = Expr(:ref, Meta.parse("ϵᵒᵇᶜꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0)

                                    obc_vars_left = Expr(:ref, Meta.parse("χᵒᵇᶜꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝˡ" ), 0)
                                    obc_vars_right = Expr(:ref, Meta.parse("χᵒᵇᶜꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝʳ" ), 0)

                                    push!(eqs, :($obc_vars_left = $(x.args[2])))
                                    push!(eqs, :($obc_vars_right = $(x.args[3])))

                                    obc_inequality = Expr(:ref, Meta.parse("Χᵒᵇᶜꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ" ), 0)

                                    push!(eqs, :($obc_inequality = $(Expr(x.head, x.args[1], obc_vars_left, obc_vars_right)) + $obc_shock))

                                    push!(obc_shocks, obc_shock)

                                    obc_inequality
                                end :
                            x :
                        x :
                    x,
            arg)




import MacroTools: postwalk
import Subscripts: super

arg = :(0 = max(b[0] - M * y[0], -lb[0]))

obc_shocks = Expr[]

eq = postwalk(x -> 
        x isa Expr ?
            x.head == :call ? 
                x.args[1] ∈ [:max, :min] ?
                    begin
                        obc_shock = Expr(:ref,Meta.parse("ϵᵒᵇᶜꜝ"*super(string(length(obc_shocks) + 1))),0)

                        push!(obc_shocks, obc_shock)

                        Expr(x.head, x.args[1], Expr(:call, :(+), x.args[2], obc_shock), Expr(:call, :(+), x.args[3], obc_shock))
                    end :
                x :
            x :
        x,
arg)

eqs = []
max_obc_shift = 10

for obc in obc_shocks
    obc_shifts = [Expr(:ref,Meta.parse(string(obc.args[1]) * "ꜝ⁽⁻"*super(string(i))*"⁾"),i > 0 ? :(x - $i) : :x) for i in 0:max_obc_shift]
    push!(eq, :($(obc) = $(Expr(:call,:+, obc_shifts...))))
end

import MacroTools: postwalk
import Subscripts: super
import MacroModelling: get_symbols

function parse_occasionally_binding_constraints(equations_block; max_obc_shift::Int = 10)
    eqs = []
    obc_shocks = Expr[]

    for arg in equations_block.args
        if isa(arg,Expr)

            eq = postwalk(x -> 
                    x isa Expr ?
                        x.head == :call ? 
                            x.args[1] ∈ [:max, :min] ?
                                begin
                                    obc_shock = Expr(:ref,Meta.parse("ϵᵒᵇᶜꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0)
            
                                    push!(obc_shocks, obc_shock)
            
                                    Expr(x.head, x.args[1], Expr(:call, :(+), x.args[2], obc_shock), Expr(:call, :(+), x.args[3], obc_shock))
                                end :
                            x :
                        x :
                    x,
            arg)
            
            push!(eqs, eq)
        end
    end

    for obc in obc_shocks
        obc_shifts = [Expr(:ref, Meta.parse(string(obc.args[1]) * "⁽⁻" * super(string(i)) * "⁾"), i > 0 ? :(x - $i) : :x) for i in 0:max_obc_shift]
        push!(eqs, :($(obc) = $(Expr(:call, :+, obc_shifts...))))
    end

    return Expr(:block, eqs...)
end


eqs = parse_occasionally_binding_constraints(:(begin 
c[0]^(-γ) + λ[0] = β * (c[1]^(-γ) * ((1 - δ) + α * exp(z[1]) * k[0]^(α - 1)) - (1 - δ) * λ[0])
c[0] + i[0] = exp(z[0]) * k[-1]^α
k[0] = (1 - δ) * k[-1] + exp(zⁱ[0]) * i[0]
λ[0] * (i[0] - ϕ * i[ss]) = 0
z[0] = ρ * z[-1] + std_z * eps_z[x]
zⁱ[0] = ρⁱ * zⁱ[-1] + std_zⁱ * eps_zⁱ[x]

0 = max(i[0] - ϕ * i[ss], -λ[0])
end))

eqs_with_obc_shocks = []
inequalities[1][1].args[2]

inequalities[1][1]|>dump
for (i,ineq) in enumerate(inequalities)
    if ineq[1].args[1] ∈ [:≤,:<]
        push!(eqs_with_obc_shocks, Expr(:(=), Expr(:ref,Meta.parse("λᵒᵇᶜꜝ"*super(string(i))),0), Expr(:call, :(-), ineq[1].args[3], ineq[1].args[2])))
    elseif ineq[1].args[1] ∈ [:≥,:>]
        push!(eqs_with_obc_shocks, Expr(:(=), Expr(:ref,Meta.parse("λᵒᵇᶜꜝ"*super(string(i))),0), Expr(:call, :(-), ineq[1].args[2], ineq[1].args[3])))
    end
end
:(λᵒᵇᶜ[0] =  i[0] - ϕ * i[ss])|>dump
