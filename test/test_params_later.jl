using Revise
# using Test
using MacroModelling

include("models/Backus_Kehoe_Kydland_1992.jl")

@model Backus_Kehoe_Kydland_1992_incomplete begin
    for co in [H, F]
        Y{co}[0] = ((LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co} * Z{co}[-1]^(-nu{co}))^(-1/nu{co})

        K{co}[0] = (1-delta{co})*K{co}[-1] + S{co}[0]

        X{co}[0] = for lag in (-4+1):0 phi{co} * S{co}[lag] end

        A{co}[0] = (1-eta{co}) * A{co}[-1] + N{co}[0]

        L{co}[0] = 1 - alpha{co} * N{co}[0] - (1-alpha{co})*eta{co} * A{co}[-1]

        U{co}[0] = (C{co}[0]^mu{co}*L{co}[0]^(1-mu{co}))^gamma{co}

        psi{co} * mu{co} / C{co}[0]*U{co}[0] = LGM[0]

        psi{co} * (1-mu{co}) / L{co}[0] * U{co}[0] * (-alpha{co}) = - LGM[0] * (1-theta{co}) / N{co}[0] * (LAMBDA{co}[0] * K{co}[-4]^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co})*Y{co}[0]^(1+nu{co})

        for lag in 0:(4-1)  
            beta{co}^lag * LGM[lag]*phi{co}
        end +
        for lag in 1:4
            -beta{co}^lag * LGM[lag] * phi{co} * (1-delta{co})
        end = beta{co}^4 * LGM[+4] * theta{co} / K{co}[0] * (LAMBDA{co}[+4] * K{co}[0]^theta{co} * N{co}[+4]^(1-theta{co})) ^ (-nu{co})* Y{co}[+4]^(1+nu{co})

        LGM[0] = beta{co} * LGM[+1] * (1+sigma{co} * Z{co}[0]^(-nu{co}-1)*Y{co}[+1]^(1+nu{co}))

        NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1]))/Y{co}[0]
    end

    (LAMBDA{H}[0]-1) = rho{H}{H}*(LAMBDA{H}[-1]-1) + rho{H}{F}*(LAMBDA{F}[-1]-1) + Z_E{H} * E{H}[x]

    (LAMBDA{F}[0]-1) = rho{F}{F}*(LAMBDA{F}[-1]-1) + rho{F}{H}*(LAMBDA{H}[-1]-1) + Z_E{F} * E{F}[x]

    for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end

    dLGM[0] = LGM[1] / LGM[0]

    dLGM_ann[0] = for operator = :*, lag in -3:0 dLGM[lag] end
end

@parameters Backus_Kehoe_Kydland_1992_incomplete begin
    K_ss = 11.0148
    # K[ss] = K_ss | beta
    # K[ss] = 10 | beta
    # F_H_ratio = 1
    K{F}[ss] / K{H}[ss] = F_H_ratio | beta{F}
    K{H}[ss] = K_ss | beta{H}

    # beta    =    0.99
    # mu      =    0.34
    # gamma   =    -1.0
    # alpha   =    1
    # eta     =    0.5
    # theta   =    0.36
    # nu      =    3
    # sigma   =    0.01
    # delta   =    0.025
    # phi     =    1/4
    # psi     =    0.5

    # Z_E = 0.00852
    
    # rho{H}{H} = 0.906
    rho{F}{F} = rho{H}{H}
    # rho{H}{F} = 0.088
    rho{F}{H} = rho{H}{F}
end

pars = [ 
        "kk" => 0.0,
        "F_H_ratio" => 1.0,
        "K_ss"      => 11.0148,
        "Z_E{F}"    => 0.00852,
        "Z_E{H}"    => 0.00852,
        "alpha{F}"  => 1.0,
        "alpha{H}"  => 1.0,
        "delta{F}"  => 0.025,
        "delta{H}"  => 0.025,
        "eta{F}"    => 0.5,
        "eta{H}"    => 0.5,
        "gamma{F}"  => -1.0,
        "gamma{H}"  => -1.0,
        "mu{F}"     => 0.34,
        "mu{H}"     => 0.34,
        "nu{F}"     => 3.0,
        "nu{H}"     => 3.0,
        "phi{F}"    => 0.25,
        "phi{H}"    => 0.25,
        "psi{F}"    => 0.5,
        "psi{H}"    => 0.5,
        "rho{H}{F}" => 0.088,
        "rho{H}{H}" => 0.906,
        "sigma{F}"  => 0.01,
        "sigma{H}"  => 0.01,
        "theta{F}"  => 0.36,
        "theta{H}"  => 0.36
        ]

# Backus_Kehoe_Kydland_1992_incomplete.parameters

# params_full = get_parameters(Backus_Kehoe_Kydland_1992, values = true)

cov1 = get_cov(Backus_Kehoe_Kydland_1992_incomplete, parameters = pars, verbose = true)


Backus_Kehoe_Kydland_1992_incomplete.jacobian_SS_and_pars[1]

Backus_Kehoe_Kydland_1992.jacobian_SS_and_pars[1]

Backus_Kehoe_Kydland_1992.jacobian[2]

import Symbolics
import MacroModelling: get_symbols, match_pattern, replace_symbols


𝓂 = Backus_Kehoe_Kydland_1992_incomplete


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
    dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.dyn_ss_list))) .* "₍ₛₛ₎")

    future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
    present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
    past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
    exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
    stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))

    vars_raw = vcat(dyn_future_list[indexin(sort(future),future)],
                    dyn_present_list[indexin(sort(present),present)],
                    dyn_past_list[indexin(sort(past),past)],
                    dyn_exo_list[indexin(sort(exo),exo)])

    dyn_var_future_idx = 𝓂.solution.perturbation.auxiliary_indices.dyn_var_future_idx
    dyn_var_present_idx = 𝓂.solution.perturbation.auxiliary_indices.dyn_var_present_idx
    dyn_var_past_idx = 𝓂.solution.perturbation.auxiliary_indices.dyn_var_past_idx
    dyn_ss_idx = 𝓂.solution.perturbation.auxiliary_indices.dyn_ss_idx

    dyn_var_idxs = vcat(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx)

    pars_ext = vcat(𝓂.parameters, 𝓂.calibration_equations_parameters)
    parameters_and_SS = vcat(pars_ext, dyn_ss_list[indexin(sort(stst),stst)])

    np = length(parameters_and_SS)
    nv = length(vars_raw)
    nc = length(𝓂.calibration_equations)
    nps = length(𝓂.parameters)
    nxs = maximum(dyn_var_idxs) + nc

    Symbolics.@variables 𝔓[1:np] 𝔙[1:nv]

    parameter_dict = Dict{Symbol, Symbol}()
    back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
    calib_vars = Symbol[]
    calib_expr = []
    SS_mapping = Dict{Symbolics.Num, Symbolics.Num}()


    for (i,v) in enumerate(parameters_and_SS)
        push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
        if i > nps
            if i > length(pars_ext)
                push!(SS_mapping, 𝔓[i] => 𝔙[dyn_ss_idx[i-length(pars_ext)]])
            else
                push!(SS_mapping, 𝔓[i] => 𝔙[nxs + i - nps - nc])
            end
        end
    end

    for (i,v) in enumerate(vars_raw)
        push!(parameter_dict, v => :($(Symbol("𝔙_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔙_$i"))), @__MODULE__) => 𝔙[i])
        if i <= length(dyn_var_idxs)
            push!(SS_mapping, 𝔙[i] => 𝔙[dyn_var_idxs[i]])
        else
            push!(SS_mapping, 𝔙[i] => 0)
        end
    end


    for v in 𝓂.calibration_equations_no_var
        push!(calib_vars, v.args[1])
        push!(calib_expr, v.args[2])
    end


    calib_replacements = Dict{Symbol,Any}()
    for (i,x) in enumerate(calib_vars)
        replacement = Dict(x => calib_expr[i])
        for ii in i+1:length(calib_vars)
            calib_expr[ii] = replace_symbols(calib_expr[ii], replacement)
        end
        push!(calib_replacements, x => calib_expr[i])
    end


    dyn_equations = 𝓂.dyn_equations |> 
        x -> replace_symbols.(x, Ref(calib_replacements)) |> 
        x -> replace_symbols.(x, Ref(parameter_dict)) |> 
        x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
        x -> Symbolics.substitute.(x, Ref(back_to_array_dict))



𝓂 = Backus_Kehoe_Kydland_1992


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
    dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.dyn_ss_list))) .* "₍ₛₛ₎")

    future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
    present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
    past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
    exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
    stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))

    vars_raw = vcat(dyn_future_list[indexin(sort(future),future)],
                    dyn_present_list[indexin(sort(present),present)],
                    dyn_past_list[indexin(sort(past),past)],
                    dyn_exo_list[indexin(sort(exo),exo)])

    dyn_var_future_idx = 𝓂.solution.perturbation.auxiliary_indices.dyn_var_future_idx
    dyn_var_present_idx = 𝓂.solution.perturbation.auxiliary_indices.dyn_var_present_idx
    dyn_var_past_idx = 𝓂.solution.perturbation.auxiliary_indices.dyn_var_past_idx
    dyn_ss_idx = 𝓂.solution.perturbation.auxiliary_indices.dyn_ss_idx

    dyn_var_idxs = vcat(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx)

    pars_ext = vcat(𝓂.parameters, 𝓂.calibration_equations_parameters)
    parameters_and_SS = vcat(pars_ext, dyn_ss_list[indexin(sort(stst),stst)])

    np = length(parameters_and_SS)
    nv = length(vars_raw)
    nc = length(𝓂.calibration_equations)
    nps = length(𝓂.parameters)
    nxs = maximum(dyn_var_idxs) + nc


    Symbolics.@variables 𝔓[1:np] 𝔙[1:nv]

    parameter_dict = Dict{Symbol, Symbol}()
    back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
    calib_vars = Symbol[]
    calib_expr = []
    SS_mapping = Dict{Symbolics.Num, Symbolics.Num}()


    for (i,v) in enumerate(parameters_and_SS)
        push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
        if i > nps
            if i > length(pars_ext)
                push!(SS_mapping, 𝔓[i] => 𝔙[dyn_ss_idx[i-length(pars_ext)]])
            else
                push!(SS_mapping, 𝔓[i] => 𝔙[nxs + i - nps - nc])
            end
        end
    end

    for (i,v) in enumerate(vars_raw)
        push!(parameter_dict, v => :($(Symbol("𝔙_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔙_$i"))), @__MODULE__) => 𝔙[i])
        if i <= length(dyn_var_idxs)
            push!(SS_mapping, 𝔙[i] => 𝔙[dyn_var_idxs[i]])
        else
            push!(SS_mapping, 𝔙[i] => 0)
        end
    end


    for v in 𝓂.calibration_equations_no_var
        push!(calib_vars, v.args[1])
        push!(calib_expr, v.args[2])
    end


    calib_replacements = Dict{Symbol,Any}()
    for (i,x) in enumerate(calib_vars)
        replacement = Dict(x => calib_expr[i])
        for ii in i+1:length(calib_vars)
            calib_expr[ii] = replace_symbols(calib_expr[ii], replacement)
        end
        push!(calib_replacements, x => calib_expr[i])
    end


    dyn_equations = 𝓂.dyn_equations |> 
        x -> replace_symbols.(x, Ref(calib_replacements)) |> 
        x -> replace_symbols.(x, Ref(parameter_dict)) |> 
        x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
        x -> Symbolics.substitute.(x, Ref(back_to_array_dict))


# Backus_Kehoe_Kydland_1992_incomplete.NSSS_solver_cache

Backus_Kehoe_Kydland_1992_incomplete.SS_solve_func

Backus_Kehoe_Kydland_1992_incomplete.missing_parameters

Backus_Kehoe_Kydland_1992_incomplete.ss_solve_blocks_in_place[11]

1
# Backus_Kehoe_Kydland_1992_incomplete.bounds

# Backus_Kehoe_Kydland_1992.bounds

# Backus_Kehoe_Kydland_1992.SS_solve_func

# bcov2 = get_cov(Backus_Kehoe_Kydland_1992)

# @test cov1 ≈ cov2
# Backus_Kehoe_Kydland_1992.parameters
# Backus_Kehoe_Kydland_1992_incomplete.parameters
# setdiff(axiskeys(cov1,1), axiskeys(cov2,1))

# include("../models/Gali_2015_chapter_3_obc.jl")

# @model Gali_2015_chapter_3_obc_incomplete begin
#     W_real[0] = C[0] ^ σ * N[0] ^ φ

#     Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]

#     R[0] = 1 / Q[0]

#     Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

#     R[0] = Pi[1] * realinterest[0]

#     R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))

#     C[0] = Y[0]

#     log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]

#     log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]

#     nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]

#     MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])

#     1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)

#     S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]

#     Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)

#     x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]

#     x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]

#     log_y[0] = log(Y[0])

#     log_W_real[0] = log(W_real[0])

#     log_N[0] = log(N[0])

#     pi_ann[0] = 4 * log(Pi[0])

#     i_ann[0] = 4 * log(R[0])

#     r_real_ann[0] = 4 * log(realinterest[0])

#     M_real[0] = Y[0] / R[0] ^ η

# end

# @parameters Gali_2015_chapter_3_obc_incomplete begin
#     σ = 1

#     φ = 5

#     ϕᵖⁱ = 1.5
    
#     ϕʸ = 0.125

#     θ = 0.75

#     ρ_ν = 0.5

#     ρ_z = 0.5

#     ρ_a = 0.9

#     β = 0.99

#     η = 3.77

#     α = 0.25

#     ϵ = 9

#     τ = 0

#     std_a = .01

#     std_z = .05

#     std_nu = .0025

#     R > 1.0001
# end

# cov1 = get_cov(Gali_2015_chapter_3_obc_incomplete, parameters = :R̄ => 1.0)

# cov2 = get_cov(Gali_2015_chapter_3_obc)

# @test cov1 ≈ cov2