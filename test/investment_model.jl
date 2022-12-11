using MacroModelling, BlockTriangularForm, SparseArrays


@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    c_normcdf[0]= normcdf(c[0])
    c_normpdf[0]= normpdf(c[0])
    c_norminvcdf[0]= norminvcdf(c[0]-1)
    c_norminv[0]= norminv(c[0]-1)
    c_qnorm[0]= qnorm(c[0]-1)
    c_dnorm[0]= dnorm(c[0])
    c_pnorm[0]= pnorm(c[0])
    c_normlogpdf[0]= normlogpdf(c[0])
    # c_norm[0]= cdf(Normal(),c[0])
    c_inv[0] = erfcinv(c[0])
    # c_binomlogpdf[0]= binomlogpdf(c[0])
    # A[0]=exp(z[0])
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
    # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
    # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
    # y_avg[0] = log(y[0] / y[-4])
    # y_growth[0] = log(y[1] / y[2])
    # y_growthl[0] = log(y[0] / y[1])
    # y_growthl1[0] = log(y[-1] / y[0])
    # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
end


@parameters RBC_CME begin
    alpha | k[ss] / (4 * y[ss]) = cap_share
    cap_share = 1.66
    # alpha = .157

    beta | R[ss] = R_ss
    R_ss = 1.0035
    # beta = .999

    # delta | c[ss]/y[ss] = 1 - I_K_ratio
    # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
    # I_K_ratio = .15
    delta = .0226

    Pibar | Pi[ss] = Pi_ss
    Pi_ss = R_ss - Pi_real
    Pi_real = .001
    # Pibar = 1.0008

    phi_pi = 1.5
    rhoz = rho_z_delta
    std_eps = .0068
    rho_z_delta = 9 / 10
    std_z_delta = .005
end

get_moments(RBC_CME)





𝓂 = RBC_CME
parameters = 𝓂.parameter_values

cap_share = parameters[1]
R_ss = parameters[2]
delta = parameters[3]
phi_pi = parameters[5]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:516 =#
R = R_ss
Pi = Pi_ss
beta = Pi / R
A = 1
lbs = [-1.0e12, 2.220446049250313e-16, -1.0e12]
ubs = [1.0e12, 1.0e12, 1.0e12]
𝓂.SS_init_guess = initial_guess
f = OptimizationFunction(𝓂.ss_solve_blocks_optim[1], Optimization.AutoForwardDiff())
inits = max.(lbs, min.(ubs, 𝓂.SS_init_guess[[17, 14, 15]]))
sol = block_solver([cap_share, delta, beta], 1, 𝓂.ss_solve_blocks[1], 𝓂.SS_optimizer, f, inits, lbs, ubs)

RBC_CME.calibration_equations
RBC_CME.calibration_equations_no_var
RBC_CME.parameters
RBC_CME.parameter_values
RBC_CME.calibration_equations_no_var

RBC_CME.ss_no_var_calib_list
RBC_CME.par_no_var_calib_list


import MacroTools: postwalk, unblock


ex = :(begin 
    alpha | k[ss] / (4 * y[ss]) = cap_share
    omega = 1 / 6
    Pibar = 1.01
    nu = .36
    kappa = (Pi_ss / R_ss) ^ 2
    R_ss = omega / 2 * Pibar
    Pi_ss = R_ss ^ nu
    alpha = 1 / 3
    beta = .999
end)

dump(ex)



calib_equations = []
calib_equations_no_var = []
calib_values_no_var = []

calib_parameters_no_var = []


calib_eq_parameters = []
calib_equations_list = []

ss_calib_list = []
par_calib_list = []


calib_equations_no_var_list = []

ss_no_var_calib_list = []
par_no_var_calib_list = []

calib_parameters = []
calib_values = []


bounds = []

# label all variables parameters and exogenous vairables and timings across all equations
postwalk(x -> 
    x isa Expr ?
        x.head == :(=) ? 
            x.args[1] isa Symbol ?
                typeof(x.args[2]) ∈ [Int, Float64] ?
                    begin # this is normal calibration by setting values of parameters
                        push!(calib_values,x.args[2])
                        push!(calib_parameters,x.args[1])
                    end :
                begin # this is normal calibration by setting values of parameters
                    # push!(calib_equations_no_var,Expr(:(=),x.args[1], unblock(x.args[2])))
                    push!(calib_values_no_var,unblock(x.args[2]))
                    push!(calib_parameters_no_var,x.args[1])
                end :
            begin # this is calibration by targeting SS values
                push!(calib_eq_parameters,x.args[1].args[2])
                push!(calib_equations,Expr(:(=),x.args[1].args[3], unblock(x.args[2])))
            end :
        x.head == :comparison ? 
            push!(bounds,x) :
        x.head == :call ?
            issubset([x.args[1]], [:(<) :(>) :(<=) :(>=)]) ?
                push!(bounds,x) :
            x :
        x :
    x,
ex)


for (i, v) in enumerate(calib_values_no_var)
    out = try eval(v) catch e end
    if out isa Float64
        push!(calib_parameters, calib_parameters_no_var[i])
        push!(calib_values, out)
    else
        push!(calib_equations_no_var, Expr(:(=),calib_parameters_no_var[i], calib_values_no_var[i]))
    end
end

calib_parameters_no_var = setdiff(calib_parameters_no_var,calib_parameters)

for (i, cal_eq) in enumerate(calib_equations)
    ss_tmp = Set()
    par_tmp = Set()

    # parse SS variables
    postwalk(x -> 
        x isa Expr ? 
            x.head == :ref ?
                occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    push!(ss_tmp,x.args[1]) :
                x : 
            x :
        x,
        cal_eq)

    # get SS variables per non_linear_solved_vals
    postwalk(x -> 
    x isa Symbol ? 
        occursin(r"^(\+|\-|\*|\/|\^|ss|stst|steady|steadystate|steady_state){1}$"i,string(x)) ?
            x :
            begin
                diffed = setdiff([x],ss_tmp)
                if !isempty(diffed)
                    push!(par_tmp,diffed[1])
                end
            end :
    x,
    cal_eq)

    push!(ss_calib_list,ss_tmp)
    push!(par_calib_list,par_tmp)
    
    # write down calibration equations
    prs_ex = postwalk(x -> 
        x isa Expr ? 
            x.head == :(=) ? 
                Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                    x.head == :ref ?
                        occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    x.args[1] : 
                x : 
            x.head == :call ?
                x.args[1] == :* ?
                    x.args[2] isa Int ?
                        x.args[3] isa Int ?
                            x :
                        :($(x.args[3]) * $(x.args[2])) :
                    x :
                x :
            unblock(x) : 
        x,
        cal_eq)
    push!(calib_equations_list,unblock(prs_ex))
end


for (i, cal_eq) in enumerate(calib_equations_no_var)
    ss_tmp = Set()
    par_tmp = Set()

    # parse SS variables
    postwalk(x -> 
        x isa Expr ? 
            x.head == :ref ?
                occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    push!(ss_tmp,x.args[1]) :
                x : 
            x :
        x,
        cal_eq)

    # get SS variables per non_linear_solved_vals
    postwalk(x -> 
    x isa Symbol ? 
        occursin(r"^(\+|\-|\*|\/|\^|ss|stst|steady|steadystate|steady_state){1}$"i,string(x)) ?
            x :
            begin
                diffed = setdiff([x],ss_tmp)
                if !isempty(diffed)
                    push!(par_tmp,diffed[1])
                end
            end :
    x,
    cal_eq)

    push!(ss_no_var_calib_list,ss_tmp)
    push!(par_no_var_calib_list,setdiff(par_tmp,calib_parameters))
    
    # write down calibration equations
    prs_ex = postwalk(x -> 
        x isa Expr ? 
            # x.head == :(=) ? 
                # Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                    x.head == :ref ?
                        occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    x.args[1] : 
                x : 
                # x :
            x.head == :call ?
                x.args[1] == :* ?
                    x.args[2] isa Int ?
                        x.args[3] isa Int ?
                            x :
                        :($(x.args[3]) * $(x.args[2])) :
                    x :
                x :
            unblock(x) : 
        x,
        cal_eq)
    push!(calib_equations_no_var_list,unblock(prs_ex))
end




incidence_matrix = fill(0,length(calib_parameters_no_var),length(calib_parameters_no_var))

for i in 1:length(calib_parameters_no_var)
    for k in 1:length(calib_parameters_no_var)
        incidence_matrix[i,k] = collect(calib_parameters_no_var)[i] ∈ collect(par_no_var_calib_list)[k]
    end
end

Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence_matrix))

@assert length(Q) == n_blocks "Check the parameter definitions. They are either incomplete or have more than only the defined parameter on the LHS."

calib_equations_no_var_list = calib_equations_no_var_list[Q]




calib_values_no_var[2] |> eval





nnaux = [:(nonnegativity_auxilliary₃₁ = max(eps(), -((C * h - C)))),
:(nonnegativity_auxilliary₂₄ = max(eps(), -((C * h - C)))),
:(nonnegativity_auxilliary₃ = max(eps(), -(-K * z))),
:(nonnegativity_auxilliary₂₀ = max(eps(), -(-K * z))),
:(nonnegativity_auxilliary₁ = max(eps(), -((C * h - C)))),
:(nonnegativity_auxilliary₂₅ = max(eps(), w_star / W)),
:(nonnegativity_auxilliary₅₂ = max(eps(), -((C * h - C)))),
:(nonnegativity_auxilliary₃₉ = max(eps(), -((C * h - C)))),
:(nonnegativity_auxilliary₉ = max(eps(), -(-K * z))),
:(nonnegativity_auxilliary₂₉ = max(eps(), -(-L / nonnegativity_auxilliary₂₈ ^ ((lambda_w + 1) / lambda_w)))),
:(nonnegativity_auxilliary₂₈ = max(eps(), w_star / W)),
:(nonnegativity_auxilliary₃₃ = max(eps(), -((C * h - C))))]


import MacroTools: postwalk
using BlockTriangularForm, SparseArrays

function get_symbols(ex)
    list = Set()
    postwalk(x -> x isa Symbol ? push!(list, x) : x, ex)
    return list
end

nn_symbols = map(x->intersect(SW03.nonnegativity_auxilliary_vars,x),get_ex_symbols.(nnaux))

all_symbols = reduce(vcat,nn_symbols) |> Set


incidence_matrix = fill(0,length(all_symbols),length(all_symbols))


for i in 1:length(all_symbols)
    for k in 1:length(all_symbols)
        incidence_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
    end
end

Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence_matrix))

nnaux = nnaux[Q]


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

@testset "Model without shocks" begin
    @model m begin
        K[0] = (1 - δ) * K[-1] + I[0]
        Z[0] = (1 - ρ) * μ + ρ * Z[-1] 
        I[1]  = ((ρ + δ - Z[0])/(1 - δ))  + ((1 + ρ)/(1 - δ)) * I[0]
    end

    @parameters m begin
        ρ = 0.05
        δ = 0.10
        μ = .17
        σ = .2
    end

    m_ss = get_steady_state(m)
    @test isapprox(m_ss(:,:Steady_state),[1/7.5,1/.75,.17],rtol = eps(Float32))

    m_sol = get_solution(m) 
    @test isapprox(m_sol(:,:K),[1/.75,.9,.04975124378109454],rtol = eps(Float32))
end

get_irf(m, initial_state = init)

plot_irf(m, initial_state = init, shocks = :none, save_plots = true, save_plots_path = "~/Downloads", save_plots_format = :png)

plot(m, initial_state = init)
m.timings.nExo




using MacroModelling;

@model m begin
    Z[0] = (1 - ρ) * μ + ρ * Z[-1]
    I[1]  = (ρ + δ - Z[0]) / (1 - δ)  + (1 + ρ) / (1 - δ) * I[0]
end
# Model: m
# Variables: 2
# Shocks: 0
# Parameters: 3
# Auxiliary variables: 0

@parameters m begin
    ρ = 0.05
    δ = 0.10
    μ = .17
    σ = .2
end
m_ss = get_steady_state(m)
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Variables_and_calibrated_parameters ∈ 2-element Vector{Symbol}
# →   Steady_state_and_∂steady_state∂parameter ∈ 4-element Vector{Symbol}
# And data, 2×4 Matrix{Float64}:
#         (:Steady_state)  (:ρ)      (:δ)      (:μ)
#   (:I)   0.133333        -7.55556  -7.55556   6.66667
#   (:Z)   0.17             0.0       0.0       1.0

m.SS_solve_func
# RuntimeGeneratedFunction(#=in MacroModelling=#, #=using MacroModelling=#, :((parameters, initial_guess, 𝓂)->begin
# 
# 
#           ρ = parameters[1]
#           δ = parameters[2]
#           μ = parameters[3]
# 
#           Z = μ
#           I = ((Z - δ) - ρ) / (δ + ρ)
#           SS_init_guess = [I, Z]
#           𝓂.SS_init_guess = if typeof(SS_init_guess) == Vector{Float64}
#                   SS_init_guess
#               else
#                   ℱ.value.(SS_init_guess)
#               end
#           return ComponentVector([I, Z], Axis([sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]))
#       end))

m_sol = get_solution(m) 
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Steady_state__States__Shocks ∈ 2-element Vector{Symbol}
# →   Variable ∈ 2-element Vector{Symbol}
# And data, 2×2 adjoint(::Matrix{Float64}) with eltype Float64:
#                    (:I)        (:Z)
#   (:Steady_state)   0.133333    0.17
#   (:Z₍₋₁₎)          0.0497512   0.05

init = m_ss(:,:Steady_state) |> collect
init[2] *= 1.5
get_irf(m, initial_state = init, shocks = :none)

plot_irf(m, initial_state = init, shocks = :none)

# , save_plots = true, save_plots_path = "~/Downloads", save_plots_format = :png)