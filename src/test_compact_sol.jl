using MacroModelling
import Turing, Pigeons
import Turing: NUTS, sample, logpdf
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL
import Zygote

include("../models/Smets_Wouters_2007.jl")

# load data
dat = CSV.read("test/data/usmodel.csv", DataFrame)

# load data
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables as written in csv file
observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# Subsample
# subset observables in data
sample_idx = 47:230 # 1960Q1-2004Q4

data = data(observables_old, sample_idx)

# declare observables as written in model
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => observables)


# Handling distributions with varying parameters using arraydist
dists = [
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ea
InverseGamma(0.1, 2.0, 0.025,5.0, μσ = true),   # z_eb
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eg
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eqs
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_em
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_epinf
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ew
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoa
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhob
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhog
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoqs
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoms
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhopinf
Beta(0.5, 0.2, 0.001,0.9999, μσ = true),        # crhow
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # cmap
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # cmaw
Normal(4.0, 1.5,   2.0, 15.0),                  # csadjcost
Normal(1.50,0.375, 0.25, 3.0),                  # csigma
Beta(0.7, 0.1, 0.001, 0.99, μσ = true),         # chabb
Beta(0.5, 0.1, 0.3, 0.95, μσ = true),           # cprobw
Normal(2.0, 0.75, 0.25, 10.0),                  # csigl
Beta(0.5, 0.10, 0.5, 0.95, μσ = true),          # cprobp
Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # cindw
Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # cindp
Beta(0.5, 0.15, 0.01, 0.99999, μσ = true),      # czcap
Normal(1.25, 0.125, 1.0, 3.0),                  # cfc
Normal(1.5, 0.25, 1.0, 3.0),                    # crpi
Beta(0.75, 0.10, 0.5, 0.975, μσ = true),        # crr
Normal(0.125, 0.05, 0.001, 0.5),                # cry
Normal(0.125, 0.05, 0.001, 0.5),                # crdy
Gamma(0.625, 0.1, 0.1, 2.0, μσ = true),         # constepinf
Gamma(0.25, 0.1, 0.01, 2.0, μσ = true),         # constebeta
Normal(0.0, 2.0, -10.0, 10.0),                  # constelab
Normal(0.4, 0.10, 0.1, 0.8),                    # ctrend
Normal(0.5, 0.25, 0.01, 2.0),                   # cgy
Normal(0.3, 0.05, 0.01, 1.0),                   # calfa
]

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01,:cmap => 0.01,:cmaw => 0.01])

fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw],Smets_Wouters_2007.parameters)]

inits = [Dict(get_parameters(Smets_Wouters_2007, values = true))[string(i)] for i in [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]]


z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = inits

ctou, clandaw, cg, curvp, curvw = fixed_parameters

parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

get_loglikelihood(Smets_Wouters_2007, data, parameters_combined, verbose = false, presample_periods = 4, initial_covariance = :diagonal)


import ChainRulesCore: @ignore_derivatives
import MacroModelling: get_and_check_observables, riccati_compact_forward, riccati_forward, create_timings_for_estimation!
import LinearAlgebra as ℒ
import ThreadedSparseArrays
import RecursiveFactorization as RF

explosive = false
𝓂 = Smets_Wouters_2007
    # data::KeyedArray{Float64}, 
    parameter_values = parameters_combined
    algorithm = :first_order
    # filter = :kalman
    warmup_iterations = 0
    presample_periods = 4
    initial_covariance = :diagonal
    tol = 1e-12
    verbose = false
    
    # checks to avoid errors further down the line and inform the user
    # @assert filter ∈ [:kalman, :inversion] "Currently only the Kalman filter (:kalman) for linear models and the inversion filter (:inversion) for linear and nonlinear models are supported."

    # checks to avoid errors further down the line and inform the user
    @assert initial_covariance ∈ [:theoretical, :diagonal] "Invalid method to initialise the Kalman filters covariance matrix. Supported methods are: the theoretical long run values (option `:theoretical`) or large values (10.0) along the diagonal (option `:diagonal`)."

    # if algorithm ∈ [:second_order,:pruned_second_order,:third_order,:pruned_third_order]
        # filter = :inversion
    # end

    observables = @ignore_derivatives get_and_check_observables(𝓂, data)

    @ignore_derivatives solve!(𝓂, verbose = verbose, algorithm = algorithm)

    # keep the parameters within bounds
    if length(𝓂.bounds) > 0 
        for (k,v) in 𝓂.bounds
            if k ∈ 𝓂.parameters
                if @ignore_derivatives min(max(parameter_values[indexin([k], 𝓂.parameters)][1], v[1]), v[2]) != parameter_values[indexin([k], 𝓂.parameters)][1]
                    return -Inf
                end
            end
        end
    end

    NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    obs_indices = @ignore_derivatives convert(Vector{Int},indexin(observables,NSSS_labels))
    
        SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)

        if solution_error > tol || isnan(solution_error)
            return -Inf
        end

        state = zeros(𝓂.timings.nVars)

        observables = @ignore_derivatives get_and_check_observables(𝓂, data)

        observables = union(observables, 𝓂.timings.present_only)[[1:9...,11:end...]]

        if !haskey(𝓂.estimation_helper, observables) create_timings_for_estimation!(𝓂, observables) end

        T, op_mat, present_idx = 𝓂.estimation_helper[observables]

        ∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂) |> Matrix

        𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)


        vars_to_exclude = setdiff(𝓂.timings.present_only, observables)

        # Mapping variables to their equation index
        variable_to_equation = Dict{Symbol, Vector{Int}}()
        for var in vars_to_exclude
            for (eq_idx, vars_set) in enumerate(𝓂.dyn_var_present_list)
            # for var in vars_set
                if var in vars_set
                    if haskey(variable_to_equation, var)
                        push!(variable_to_equation[var],eq_idx)
                    else
                        variable_to_equation[var] = [eq_idx]
                    end
                end
            end
        end
    
        ∇̂₁ = deepcopy(∇₁)
    
        rows_to_exclude = Int[]
        cant_exclude = Symbol[]

        for (ks, vidx) in variable_to_equation
            if all(.!(∇₁[vidx, 𝓂.timings.nFuture_not_past_and_mixed + indexin([ks] ,𝓂.timings.var)[1]] .== 0))
                for v in vidx
                    if v ∉ rows_to_exclude
                        push!(rows_to_exclude, v)
                        ∇₁[vidx,:] .-= ∇₁[v,:]' .* ∇₁[vidx, 𝓂.timings.nFuture_not_past_and_mixed + indexin([ks] ,𝓂.timings.var)[1]] ./ ∇₁[v, 𝓂.timings.nFuture_not_past_and_mixed + indexin([ks] ,𝓂.timings.var)[1]]
                        break
                    end
                end
            else
                push!(cant_exclude, ks)
            end
        end

        rows_to_include = setdiff(1:𝓂.timings.nVars, rows_to_exclude)
    
        cols_to_exclude = indexin(setdiff(𝓂.timings.present_only, union(observables, cant_exclude)), 𝓂.timings.var)

        present_idx = 𝓂.timings.nFuture_not_past_and_mixed .+ (setdiff(range(1, 𝓂.timings.nVars), cols_to_exclude))

        ∇̄₁ = ∇₁[rows_to_include, vcat(1:𝓂.timings.nFuture_not_past_and_mixed, present_idx , 𝓂.timings.nFuture_not_past_and_mixed + 𝓂.timings.nVars + 1 : size(∇₁,2))]
    

        # ∇₁ = Matrix(op_mat * ∇₁[:, vcat(1:𝓂.timings.nFuture_not_past_and_mixed, present_idx, 𝓂.timings.nFuture_not_past_and_mixed + 𝓂.timings.nVars + 1 : size(∇₁,2))])

        # findnz(op_mat)
        
        # ([1, 2, 3, 4, 6, 10, 23, 5, 6, 7  …  56, 57, 58, 59, 60, 61, 62, 63, 64, 65], [1, 2, 3, 4, 5, 5, 5, 6, 7, 8  …  57, 58, 59, 60, 61, 62, 63, 64, 65, 66], [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # 𝓂.dyn_equations[variable_to_equation[:qsaux]]

        # ∇₁[variable_to_equation[:qsaux],𝓂.timings.nFuture_not_past_and_mixed .+ range(1,𝓂.timings.nVars)]

        # ∇₁[variable_to_equation[:qsaux],𝓂.timings.nFuture_not_past_and_mixed .+ indexin([:qsaux],𝓂.timings.var)]
        # ∇₁[variable_to_equation[:qsaux],findall(vec(sum(abs, ∇₁[variable_to_equation[:qsaux],:], dims = 1) .> 0))]



        if !haskey(𝓂.estimation_helper, union(observables, cant_exclude)) create_timings_for_estimation!(𝓂, union(observables, cant_exclude)) end

        T, op_mat, present_idx = 𝓂.estimation_helper[union(observables, cant_exclude)]


    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
    ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

∇₊ = @views ∇̄₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
∇₀ = @views ∇̄₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇₋ = @views ∇̄₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
∇ₑ = @views ∇̄₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

∇̂₀ =  RF.lu(∇₀)

A = ∇₀ \ ∇₋
B = ∇₀ \ ∇₊

C = similar(A)
C̄ = similar(A)

sol = speedmapping(zero(A); m! = (C̄, C) -> C̄ .=  A + B * C^2, tol = tol, maps_limit = 10000)


C = -sol.minimizer

D = -(∇₊ * C + ∇₀) \ ∇ₑ

@views hcat(C[:,T.past_not_future_and_mixed_idx],D), sol.converged






    ∇₊ = @view ∇̄₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = @view ∇̄₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view ∇̄₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    Q    = ℒ.qr(collect(∇₀[:,T.present_only_idx]))
    Qinv = Q.Q'

    A₊ = Qinv * ∇₊
    A₀ = Qinv * ∇₀
    A₋ = Qinv * ∇₋

    dynIndex = T.nPresent_only+1:T.nVars

    Ã₊  = @view A₊[dynIndex,:]
    Ã₋  = @view A₋[dynIndex,:]
    Ã₀₊ = @view A₀[dynIndex, T.future_not_past_and_mixed_idx]
    Ã₀₋ = @views A₀[dynIndex, T.past_not_future_idx] * ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.not_mixed_in_past_idx,:]
    
    Z₊ = zeros(T.nMixed,T.nFuture_not_past_and_mixed)
    I₊ = @view ℒ.diagm(ones(T.nFuture_not_past_and_mixed))[T.mixed_in_future_idx,:]

    Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
    I₋ = @view ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

    D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
    E = vcat(hcat(-Ã₋,-Ã₀₊), hcat(Z₋, I₊))
    # this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
    schdcmp = try
        ℒ.schur(D, E)
    catch
        return zeros(T.nVars,T.nPast_not_future_and_mixed), false
    end

    # if explosive # returns false for NaN gen. eigenvalue which is correct here bc they are > 1
    #     eigenselect = abs.(schdcmp.β ./ schdcmp.α) .>= 1

    #     ℒ.ordschur!(schdcmp, eigenselect)

    #     Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
    #     Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    #     S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
    #     T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    #     Ẑ₁₁ = RF.lu(Z₁₁, check = false)

    #     if !ℒ.issuccess(Ẑ₁₁)
    #         Ẑ₁₁ = ℒ.svd(Z₁₁, check = false)
    #     end

    #     if !ℒ.issuccess(Ẑ₁₁)
    #         return zeros(T.nVars,T.nPast_not_future_and_mixed), false
    #     end
    # else
        eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

        try
            ℒ.ordschur!(schdcmp, eigenselect)
        catch
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end

        Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
        Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

        S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
        T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]


        Ẑ₁₁ = RF.lu(Z₁₁, check = false)

        if !ℒ.issuccess(Ẑ₁₁)
            Ẑ₁₁ = ℒ.svd(Z₁₁)
    
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    # end
    
    Ŝ₁₁ = RF.lu(S₁₁, check = false)

    if !ℒ.issuccess(Ŝ₁₁)
        return zeros(T.nVars,T.nPast_not_future_and_mixed), false
    end
    
    D      = Z₂₁ / Ẑ₁₁
    L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁

    sol = @views vcat(L[T.not_mixed_in_past_idx,:], D)

    Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = @view A₋[1:T.nPresent_only,:]

    Ā̂₀ᵤ = RF.lu(Ā₀ᵤ, check = false)

    if !ℒ.issuccess(Ā̂₀ᵤ)
        Ā̂₀ᵤ = ℒ.svd(collect(Ā₀ᵤ))
    end

    A    = @views vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    
    A[T.reorder,:]


    
    return A[T.reorder,:], true



import MacroModelling: get_symbols, match_pattern, timings, get_and_check_observables








        cols_to_exclude = indexin(T.var, setdiff(𝓂.timings.present_only, observables))
        remaining_vars = setdiff(T.var, setdiff(𝓂.timings.present_only, observables))

        present_idx = T.nFuture_not_past_and_mixed .+ (setdiff(range(1, T.nVars), cols_to_exclude))

        ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
        ∇₀ = @view ∇₁[:,present_idx]
        ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + length(present_idx) .+ range(1, T.nPast_not_future_and_mixed)]
    
        Q    = ℒ.qr(collect(∇₀[:, indexin(observables,remaining_vars)]))
        Qinv = Q.Q'
    
        A₊ = Qinv * ∇₊
        A₀ = Qinv * ∇₀
        A₋ = Qinv * ∇₋
    
        dynIndex = length(observables)+1:length(remaining_vars)
    
        Ã₊  = @view A₊[dynIndex,:]
        Ã₋  = @view A₋[dynIndex,:]
        Ã₀₊ = @view A₀[dynIndex, T.future_not_past_and_mixed_idx]
        Ã₀₋ = @views A₀[dynIndex, T.past_not_future_idx] * ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.not_mixed_in_past_idx,:]
    
        Z₊ = zeros(T.nMixed,T.nFuture_not_past_and_mixed)
        I₊ = @view ℒ.diagm(ones(T.nFuture_not_past_and_mixed))[T.mixed_in_future_idx,:]
    
        Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
        I₋ = @view ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]
    
        D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
        E = vcat(hcat(-Ã₋,-Ã₀₊), hcat(Z₋, I₊))
        # this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
        schdcmp = try
            ℒ.schur(D, E)
        catch
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    
        eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1
    
        try
            ℒ.ordschur!(schdcmp, eigenselect)
        catch
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    
        Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
        Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
    
        S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
        T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
    
        Ẑ₁₁ = RF.lu(Z₁₁, check = false)
    
        if !ℒ.issuccess(Ẑ₁₁)
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    
        Ŝ₁₁ = RF.lu(S₁₁, check = false)
    
        if !ℒ.issuccess(Ŝ₁₁)
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    
        D      = Z₂₁ / Ẑ₁₁
        L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
    
        sol = @views vcat(L[T.not_mixed_in_past_idx,:], D)
    
        Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
        A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
        Ã₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_but_not_only_idx]
        A₋ᵤ  = @view A₋[1:T.nPresent_only,:]
    
        Ā̂₀ᵤ = RF.lu(Ā₀ᵤ, check = false)
    
        if !ℒ.issuccess(Ā̂₀ᵤ)
            return zeros(T.nVars,T.nPast_not_future_and_mixed), false
        end
    
        A    = @views vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    
        return A[T.reorder,:][observables_and_states,:], true







        observables_and_states_indices = @ignore_derivatives sort(union(𝓂.timings.past_not_future_and_mixed_idx,𝓂.timings.future_not_past_and_mixed_idx,obs_indices))

        T = 𝓂.timings

        A, solved = riccati_compact_forward(∇₁, observables_and_states_indices; T = T)

    if !solved
        return hcat(A, zeros(size(A,1),T.nExo)), solved
    end

    Jm̂ = @view(ℒ.diagm(ones(T.nVars))[:,observables_and_states_indices])

    Jm = @view(ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:])

    ∇₊ = @views (∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:])[observables_and_states_indices,observables_and_states_indices]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)][observables_and_states_indices,observables_and_states_indices]
    ∇ₑ = @view ∇₁[observables_and_states_indices,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    inv(∇₊ * Jm̂ * A * Jm + ∇₀)[observables_and_states_indices,observables_and_states_indices]
    B = -((∇₊ * Jm̂ * A * Jm + ∇₀) \ ∇ₑ)#[observables_and_states_indices]



    A, solved = riccati_forward(∇₁; T = T)

    Jm = @view(ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:])
    
    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇ₑ = @view ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

    (∇₊ * A * Jm + ∇₀)

    inv(∇₊ * A * Jm + ∇₀)[observables_and_states_indices, observables_and_states_indices]
    B = -((∇₊ * A * Jm + ∇₀) \ ∇ₑ)

    return hcat(A, B), solved



    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    ∇₀
    𝓂.dyn_equations[5]
    𝓂.timings.present_only

    𝓂.timings.present_only_idx

    ∇₀[:,𝓂.timings.present_only_idx[1]]
    𝓂.dyn_var_present_list
    𝓂.dyn_equations[7]

    # iterate through 𝓂.timings.present_only, and then iterate through 𝓂.dyn_var_present_list if you find the element of the first loop in the element of the second loop

    ∇̂₀ = deepcopy(∇₀)
    ∇̂₁ = deepcopy(∇₁)
    # ∇₀ is a matrix containing the first order derivatives of the model equations with respect to the variables in the model. The rows of ∇₀ correspond to the equations in the model, while the columns correspond to the variables in the model. I want to eliminate equations by substituting them out (substract the row with a given variable from all rows where this variable occurs). the equations i want are those that contain the variables in 𝓂.timings.present_only. you can check which equation (column) contains the variables using the 𝓂.dyn_var_present_list (Vector{Set{Symbol}}). in the end i want to have the reduced square system. write some code to do so 
    # Mapping variables to their equation index
    variable_to_equation = Dict{Symbol, Vector{Int}}()
    for var in setdiff(𝓂.timings.present_only, observables)
        for (eq_idx, vars_set) in enumerate(𝓂.dyn_var_present_list)
        # for var in vars_set
            if var in vars_set
                if haskey(variable_to_equation, var)
                    push!(variable_to_equation[var],eq_idx)
                else
                    variable_to_equation[var] = [eq_idx]
                end
            end
        end
    end

    op_mat = spdiagm(ones(T.nVars))

    rows_to_exclude = Int[]

    for vidx in values(variable_to_equation)
        for v in vidx
            if v ∉ rows_to_exclude
                push!(rows_to_exclude, v)
                op_mat[vidx,:] .-= op_mat[v,:]'
                break
            end
        end
    end

    # Step 1: Creating the row selection matrix
    rows_to_include = setdiff(1:T.nVars, rows_to_exclude)
    
    op_mat[rows_to_include, :] * ∇̂₁[:, vcat(future_idx, present_idx, past_idx, exo_idx)]


    rows_to_exclude = [v[1] for v in values(variable_to_equation)]
    cols_to_exclude = indexin(T.var, setdiff(𝓂.timings.present_only, observables))
    future_idx = 1:T.nFuture_not_past_and_mixed
    present_idx = T.nFuture_not_past_and_mixed .+ (setdiff(range(1, T.nVars), cols_to_exclude))
    past_idx = T.nFuture_not_past_and_mixed + T.nVars .+ (1:T.nPast_not_future_and_mixed)
    exo_idx = T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1 : size(∇̂₁,2)
    
    ∇̂₁ = ∇̂₁[setdiff(1:size(∇̂₁,1),rows_to_exclude), vcat(future_idx, present_idx, past_idx, exo_idx)]


    

    ∇̂₀[variable_to_equation[:Pratio],:] .- ∇̂₀[variable_to_equation[:Pratio][1],:]'

    Jm = @view(ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:])
    Jm2 = @view(ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:])
    
    𝓂.var[vec(sum(abs.((abs.(∇₊ * Jm2) .> 0) + (abs.(∇₀) .> 0) + (abs.(∇₋ * Jm) .> 0)) .> 0, dims = 1)' .== 1)]
    sum(abs.((abs.(∇₊ * Jm2) .> 0) + (abs.(∇₀) .> 0) + (abs.(∇₋ * Jm) .> 0)) .> 0, dims = 2)
    sum(abs.(∇₊ * Jm2 + ∇₀ + ∇₋ * Jm) .> 0, dims = 2)'


    return hcat(A, B), solved


    𝐒₁ = hcat(A, B)

    𝐒₁, solved = MacroModelling.riccati_compact_forward(∇₁, observables_and_states_index; T = T)

    sp𝐒₁ = sparse(𝐒₁) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
    sp∇₁ = sparse(∇₁) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

    droptol!(sp𝐒₁, 10*eps())
    droptol!(sp∇₁, 10*eps())

    # expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 
    expand = [
        spdiagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC, 
        spdiagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
    ] 

    A = sp∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    B = sp∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

    sol_buf = sp𝐒₁ * expand[2]
    sol_buf2 = sol_buf * sol_buf

    spd𝐒₁a = (ℒ.kron(expand[2] * sp𝐒₁, A') + 
            ℒ.kron(expand[2] * expand[2]', sol_buf' * A' + B'))
            
    droptol!(spd𝐒₁a, 10*eps())

    d𝐒₁a = spd𝐒₁a' |> collect # bottleneck, reduce size, avoid conversion, subselect necessary part of matrix already here (as is done in the estimation part later)

    # Initialize empty spd∇₁a
    spd∇₁a = spzeros(length(sp𝐒₁), length(∇₁))

    # Directly allocate dA, dB, dC into spd∇₁a
    # Note: You need to calculate the column indices where each matrix starts and ends
    # This is conceptual; actual implementation would depend on how you can obtain or compute these indices
    dA_cols = 1:(T.nFuture_not_past_and_mixed * size(𝐒₁,1))
    dB_cols = dA_cols[end] .+ (1 : size(𝐒₁, 1)^2)
    dC_cols = dB_cols[end] .+ (1 : length(sp𝐒₁))

    spd∇₁a[:,dA_cols] = ℒ.kron(expand[1] * sol_buf2 * expand[2]' , ℒ.I(size(𝐒₁, 1)))'
    spd∇₁a[:,dB_cols] = ℒ.kron(sp𝐒₁, ℒ.I(size(𝐒₁, 1)))' 
    spd∇₁a[:,dC_cols] = ℒ.I(length(𝐒₁))

    d𝐒₁â = ℒ.lu(d𝐒₁a, check = false)
    
    if !ℒ.issuccess(d𝐒₁â)
        tmp = spd∇₁a'
        solved = false
    else
        tmp = -(d𝐒₁â \ spd∇₁a)' # bottleneck, reduce size, avoid conversion
    end

    return 𝐒₁, solved, tmp