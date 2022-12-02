using MacroModelling;

@model m begin
    K[0] = (1 - δ) * K[-1] + I[0]
    Z[0] = (1 - ρ) * μ + ρ * Z[-1] + eps_z[x]
    I[1]  = ((ρ + δ - Z[0])/(1 - δ))  + ((1 + ρ)/(1 - δ)) * I[0]
end

@parameters m begin
    ρ = 0.05
    δ = 0.10
    μ = .17
    σ = .2
end

get_solution(m)

range(-.5*(1+1/3),(1+1/3)*.5,100)
m.solution.perturbation.first_order.solution_matrix
pol = [[i,m.solution.perturbation.first_order.state_update([0,0.0,i],[0.0])[1]] for i in range(-.05*(1+1/3),(1+1/3)*.05,100)]
solve!(m,algorithm = :second_order, dynamics= true)
using Plots

pol2 = [[i,m.solution.perturbation.second_order.state_update([0,0.0,i],[0.0])[1]] for i in range(-.05*(1+1/3),(1+1/3)*.05,100)]

Plots.plot(reduce(hcat,pol)[1,:],reduce(hcat,pol)[2,:])
Plots.plot!(reduce(hcat,pol2)[1,:],reduce(hcat,pol2)[2,:])
