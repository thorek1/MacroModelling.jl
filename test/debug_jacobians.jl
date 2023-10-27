using MacroModelling
import MacroModelling: parse_variables_input_to_index, String_input, multiplicate, calculate_covariance, product_moments, generateSumVectors, covariance_parameter_derivatives, jacobian_wrt_A, solve_sylvester_equation_forward, determine_efficient_order, jacobian_wrt_values
import ForwardDiff as ℱ
import LinearAlgebra as ℒ
using Krylov, LinearOperators, SpeedMapping, ThreadedSparseArrays

include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
SS(m)
# get_irf(m,algorithm = :pruned_third_order)
σdiff = get_std(m)

σdiff = get_std(m, algorithm = :pruned_second_order)


σdiff = get_std(m, algorithm = :pruned_third_order)
@profview for i in 1:10 σdiff = get_std(m, algorithm = :pruned_third_order) end
using BenchmarkTools
@benchmark σdiff = get_std(m, algorithm = :pruned_third_order)
# Time  (mean ± σ):   98.900 ms
# Memory estimate: 226.51 MiB, allocs estimate: 208768.

# ThreadedSparseArrays 
# Memory estimate: 213.68 MiB, allocs estimate: 208612.

# no sparsity but linearoperator - definite nope

# back to ImplicitDifferentiation
# Memory estimate: 86.72 MiB, allocs estimate: 104092.
# Time  (mean ± σ):   71.954 ms

𝓂 = m
algorithm = :first_order
verbose = true
silent = false
parameters = m.parameter_values
variables = :all_including_auxilliary
parameter_derivatives = :all
variance = false
standard_deviation = true
non_stochastic_steady_state = false
mean = false
derivatives = true
tol = eps()


MacroModelling.solve!(𝓂, parameters = parameters, algorithm = algorithm, verbose = verbose, silent = silent)

# write_parameters_input!(𝓂,parameters, verbose = verbose)

var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

parameter_derivatives = parameter_derivatives isa String_input ? parameter_derivatives .|> Meta.parse .|> replace_indices : parameter_derivatives

if parameter_derivatives == :all
    length_par = length(𝓂.parameters)
    param_idx = 1:length_par
elseif isa(parameter_derivatives,Symbol)
    @assert parameter_derivatives ∈ 𝓂.parameters string(parameter_derivatives) * " is not part of the free model parameters."

    param_idx = indexin([parameter_derivatives], 𝓂.parameters)
    length_par = 1
elseif length(parameter_derivatives) > 1
    for p in vec(collect(parameter_derivatives))
        @assert p ∈ 𝓂.parameters string(p) * " is not part of the free model parameters."
    end
    param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.parameters) |> sort
    length_par = length(parameter_derivatives)
end

NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

if length_par * length(NSSS) > 200 || (!variance && !standard_deviation && !non_stochastic_steady_state && !mean)
    derivatives = false
end

if parameter_derivatives != :all && (variance || standard_deviation || non_stochastic_steady_state || mean)
    derivatives = true
end


axis1 = 𝓂.var

if any(x -> contains(string(x), "◖"), axis1)
    axis1_decomposed = decompose_name.(axis1)
    axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
end

axis2 = 𝓂.timings.exo

if any(x -> contains(string(x), "◖"), axis2)
    axis2_decomposed = decompose_name.(axis2)
    axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
end



axis2 = vcat(:Standard_deviation, 𝓂.parameters[param_idx])
    
if any(x -> contains(string(x), "◖"), axis2)
    axis2_decomposed = decompose_name.(axis2)
    axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
end

covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)

dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])


dst_dev = ℱ.jacobian(x -> sqrt.(ℒ.diag(calculate_covariance(x, 𝓂, verbose = verbose)[1])), 𝓂.parameter_values[param_idx])






function solve_sylvester_equation_forw(ABC::Vector{Float64};
    coords::Vector{Tuple{Vector{Int}, Vector{Int}}},
    dims::Vector{Tuple{Int,Int}},
    sparse_output::Bool = false,
    solver::Symbol = :doubling)

    if length(coords) == 1
        lengthA = length(coords[1][1])
        vA = ABC[1:lengthA]
        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        C = reshape(ABC[lengthA+1:end],dims[2]...)
        if solver != :doubling
            B = A'
        end
    elseif length(coords) == 3
        lengthA = length(coords[1][1])
        lengthB = length(coords[2][1])

        vA = ABC[1:lengthA]
        vB = ABC[lengthA .+ (1:lengthB)]
        vC = ABC[lengthA + lengthB + 1:end]

        A = sparse(coords[1]...,vA,dims[1]...)# |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        B = sparse(coords[2]...,vB,dims[2]...)# |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        C = sparse(coords[3]...,vC,dims[3]...)# |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
    else
        lengthA = dims[1][1] * dims[1][2]
        A = reshape(ABC[1:lengthA],dims[1]...)
        C = reshape(ABC[lengthA+1:end],dims[2]...)
        if solver != :doubling
            B = A'
        end
    end
    

    if solver ∈ [:gmres, :bicgstab]
        function sylvester!(sol,𝐱)
            𝐗 = reshape(𝐱, size(C))
            sol .= vec(A * 𝐗 * B - 𝐗)
            return sol
        end
        
        sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

        if solver == :gmres
            𝐂, info = Krylov.gmres(sylvester, [vec(C);])
        elseif solver == :bicgstab
            𝐂, info = Krylov.bicgstab(sylvester, [vec(C);])
        end
        solved = info.solved
    elseif solver == :iterative
        iter = 1
        change = 1
        𝐂  = copy(C)
        𝐂¹ = copy(C)
        while change > eps(Float32) && iter < 10000
            𝐂¹ = A * 𝐂 * B - C
            if !(A isa DenseMatrix)
                droptol!(𝐂¹, eps())
            end
            if iter > 500
                change = maximum(abs, 𝐂¹ - 𝐂)
            end
            𝐂 = 𝐂¹
            iter += 1
        end
        solved = change < eps(Float32)
    elseif solver == :doubling
        iter = 1
        change = 1
        𝐂  = -C
        𝐂¹ = -C
        while change > eps(Float32) && iter < 500
            𝐂¹ = A * 𝐂 * A' + 𝐂
            A *= A
            if !(A isa DenseMatrix)
                droptol!(A, eps())
            end
            if iter > 10
                change = maximum(abs, 𝐂¹ - 𝐂)
            end
            𝐂 = 𝐂¹
            iter += 1
        end
        solved = change < eps(Float32)
    elseif solver == :speedmapping
        soll = speedmapping(collect(-C); m! = (X, x) -> X .= A * x * B - C, stabilize = true)

        𝐂 = soll.minimizer

        solved = soll.converged
    end

    return sparse_output ? sparse(reshape(𝐂, size(C))) : reshape(𝐂, size(C)), solved # return info on convergence
end



function solve_sylvester_equation_conditions(ABC::Vector{<: Real},
    X::AbstractMatrix{<: Real}, 
    solved::Bool;
    coords::Vector{Tuple{Vector{Int}, Vector{Int}}},
    dims::Vector{Tuple{Int,Int}},
    sparse_output::Bool = false,
    solver::Symbol = :doubling)

    solver = :gmres # ensure the AXB works always

    if length(coords) == 1
        lengthA = length(coords[1][1])
        vA = ABC[1:lengthA]
        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        C = reshape(ABC[lengthA+1:end],dims[2]...)
        if solver != :doubling
            B = A'
        end
    elseif length(coords) == 3
        lengthA = length(coords[1][1])
        lengthB = length(coords[2][1])
        
        vA = ABC[1:lengthA]
        vB = ABC[lengthA .+ (1:lengthB)]
        vC = ABC[lengthA + lengthB + 1:end]

        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        B = sparse(coords[2]...,vB,dims[2]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        C = sparse(coords[3]...,vC,dims[3]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
    else
        lengthA = dims[1][1] * dims[1][2]
        A = reshape(ABC[1:lengthA],dims[1]...)
        C = reshape(ABC[lengthA+1:end],dims[2]...)
        if solver != :doubling
            B = A'
        end
    end

    A * X * B - C - X
end



function solve_sylvester_equation_forw(abc::Vector{ℱ.Dual{Z,S,N}};
    coords::Vector{Tuple{Vector{Int}, Vector{Int}}},
    dims::Vector{Tuple{Int,Int}},
    sparse_output::Bool = false,
    solver::Symbol = :doubling) where {Z,S,N}

    # unpack: AoS -> SoA
    ABC = ℱ.value.(abc)

    # you can play with the dimension here, sometimes it makes sense to transpose
    partial_values = mapreduce(ℱ.partials, hcat, abc)'

    # get f(vs)
    val, solved = solve_sylvester_equation_forw(ABC, coords = coords, dims = dims, sparse_output = sparse_output, solver = solver)

    if length(coords) == 1
        lengthA = length(coords[1][1])

        vA = ABC[1:lengthA]
        A = sparse(coords[1]...,vA,dims[1]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        # C = reshape(ABC[lengthA+1:end],dims[2]...)
        droptol!(A,eps())

        B = sparse(A')
        # jacA = jac_wrt_A(A, -val)
        # println(typeof(jacA))
        # b = hcat(jacA, ℒ.I(length(val)))
        # droptol!(b,eps())

        # a = jacobian_wrt_values(A, B)
        # droptol!(a,eps())

        partials = zeros(dims[1][1] * dims[1][2] + dims[2][1] * dims[2][2], size(partial_values,2))
        partials[vcat(coords[1][1] + (coords[1][2] .- 1) * dims[1][1], dims[1][1] * dims[1][2] + 1:end),:] = partial_values


        reshape_matmul_b = LinearOperators.LinearOperator(Float64, length(val) * size(partials,2), 2*size(A,1)^2 * size(partials,2), false, false, 
        (sol,𝐱) -> begin 
            𝐗 = reshape(𝐱, (2* size(A,1)^2,size(partials,2))) |> sparse
            b = hcat(jac_wrt_A(A, -val), ℒ.I(length(val)))
            droptol!(b,eps())
            sol .= vec(b * 𝐗)
            return sol
        end)
    elseif length(coords) == 3
        lengthA = length(coords[1][1])
        lengthB = length(coords[2][1])

        vA = ABC[1:lengthA]
        vB = ABC[lengthA .+ (1:lengthB)]
        # vC = ABC[lengthA + lengthB + 1:end]

        A = sparse(coords[1]...,vA,dims[1]...)# |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        B = sparse(coords[2]...,vB,dims[2]...)# |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
        # C = sparse(coords[3]...,vC,dims[3]...) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

        # jacobian_A = ℒ.kron(-val * B, ℒ.I(size(A,1)))
        # jacobian_B = ℒ.kron(ℒ.I(size(B,1)), -A * val)

        # b = hcat(jacobian_A', jacobian_B, ℒ.I(length(val)))
        # droptol!(b,eps())

        # a = jacobian_wrt_values(A, B)
        # droptol!(a,eps())

        partials = spzeros(dims[1][1] * dims[1][2] + dims[2][1] * dims[2][2] + dims[3][1] * dims[3][2], size(partial_values,2))
        partials[vcat(
            coords[1][1] + (coords[1][2] .- 1) * dims[1][1], 
            coords[2][1] + (coords[2][2] .- 1) * dims[2][1] .+ dims[1][1] * dims[1][2], 
            coords[3][1] + (coords[3][2] .- 1) * dims[3][1] .+ dims[1][1] * dims[1][2] .+ dims[2][1] * dims[2][2]),:] = partial_values
        
        
        reshape_matmul_b = LinearOperators.LinearOperator(Float64, length(val) * size(partials,2), 2*size(A,1)^2 * size(partials,2), false, false, 
            (sol,𝐱) -> begin 
                𝐗 = reshape(𝐱, (2* size(A,1)^2,size(partials,2))) |> sparse

                jacobian_A = ℒ.kron(-val * B, ℒ.I(size(A,1)))
                jacobian_B = ℒ.kron(ℒ.I(size(B,1)), -A * val)

                b = hcat(jacobian_A', jacobian_B, ℒ.I(length(val)))
                droptol!(b,eps())

                sol .= vec(b * 𝐗)
                return sol
        end)
    else
        lengthA = dims[1][1] * dims[1][2]
        A = reshape(ABC[1:lengthA],dims[1]...) |> sparse
        droptol!(A,eps())
        # C = reshape(ABC[lengthA+1:end],dims[2]...)
        B = sparse(A')
        # jacobian_A = reshape(permutedims(reshape(ℒ.kron(ℒ.I(size(A,1)), -A * val) ,size(A,1), size(A,1), size(A,1), size(A,1)), [1, 2, 4, 3]), size(A,1) * size(A,1), size(A,1) * size(A,1))

        # spA = sparse(A)
        # droptol!(spA, eps())

        # b = hcat(jac_wrt_A(spA, -val), ℒ.I(length(val)))

        # a = reshape(permutedims(reshape(ℒ.I - ℒ.kron(A, B) ,size(B,1), size(A,1), size(A,1), size(B,1)), [2, 3, 4, 1]), size(A,1) * size(B,1), size(A,1) * size(B,1))

        partials = partial_values

        reshape_matmul_b = LinearOperators.LinearOperator(Float64, length(val) * size(partials,2), 2*size(A,1)^2 * size(partials,2), false, false, 
        (sol,𝐱) -> begin 
            𝐗 = reshape(𝐱, (2* size(A,1)^2,size(partials,2))) |> sparse
            b = hcat(jac_wrt_A(sparse(A), -val), ℒ.I(length(val)))
            droptol!(b,eps())
            sol .= vec(b * 𝐗)
            return sol
        end)
    end
    

    # get J(f, vs) * ps (cheating). Write your custom rule here. This used to be the conditions but here they are analytically derived.
    reshape_matmul_a = LinearOperators.LinearOperator(Float64, length(A) * size(partials,2), length(A) * size(partials,2), false, false, 
        (sol,𝐱) -> begin 
        𝐗 = reshape(𝐱, (length(A),size(partials,2))) |> sparse
        a = jacobian_wrt_values(A, B)
        droptol!(a,eps())
        sol .= vec(a * 𝐗)
        return sol
    end)


    # println(b)
    # println(partials)
    X, info = Krylov.gmres(reshape_matmul_a, -vec(reshape_matmul_b * vec(partials)))#, atol = tol)

    jvp = reshape(X, (size(b,1),size(partials,2)))
    # println(jvp)
    out = reshape(map(val, eachrow(jvp)) do v, p
            ℱ.Dual{Z}(v, p...) # Z is the tag
        end,size(val))

    # pack: SoA -> AoS
    return sparse_output ? sparse(out) : out, solved
end






function calc_cov(parameters)
    SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
        
    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) 

    sol, solved = calculate_first_order_solution(Matrix(∇₁); T = 𝓂.timings)

    A = @views sol[:, 1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    C = @views sol[:, 𝓂.timings.nPast_not_future_and_mixed+1:end]

    CC = C * C'

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(A))
    push!(dimensions,size(CC))

    values = vcat(vec(A), vec(collect(-CC)))

    covar_raw, _ = solve_sylvester_equation_forw(values, coords = coordinates, dims = dimensions, solver = :doubling)
    
    return sqrt.(ℒ.diag(covar_raw))
end

function calc_cov_2nd(parameters)

    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose)

    nᵉ = 𝓂.timings.nExo

    nˢ = 𝓂.timings.nPast_not_future_and_mixed

    iˢ = 𝓂.timings.past_not_future_and_mixed_idx

    Σᶻ₁ = Σʸ₁[iˢ, iˢ]

    # precalc second order
    ## mean
    I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

    ## covariance
    E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

    quadrup = multiplicate(nᵉ, 4)

    comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

    comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

    for j = 1:size(comb⁴,1)
        E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
    end

    e⁴ = quadrup * E_e⁴

    # second order
    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
    v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
    kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
    kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

    # first order
    s_to_y₁ = 𝐒₁[:, 1:nˢ]
    e_to_y₁ = 𝐒₁[:, (nˢ + 1):end]
    
    s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
    e_to_s₁ = 𝐒₁[iˢ, (nˢ + 1):end]


    # second order
    s_s_to_y₂ = 𝐒₂[:, kron_s_s]
    e_e_to_y₂ = 𝐒₂[:, kron_e_e]
    v_v_to_y₂ = 𝐒₂[:, kron_v_v]
    s_e_to_y₂ = 𝐒₂[:, kron_s_e]

    s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
    e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
    v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
    s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

    s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
    e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
    s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

    # # Set up in pruned state transition matrices
    ŝ_to_ŝ₂ = [ s_to_s₁             zeros(nˢ, nˢ + nˢ^2)
                zeros(nˢ, nˢ)       s_to_s₁             s_s_to_s₂ / 2
                zeros(nˢ^2, 2*nˢ)   s_to_s₁_by_s_to_s₁                  ]

    ê_to_ŝ₂ = [ e_to_s₁         zeros(nˢ, nᵉ^2 + nᵉ * nˢ)
                zeros(nˢ,nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
                zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁]

    ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2]

    ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

    ŝv₂ = [ zeros(nˢ) 
            vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec(ℒ.I(nᵉ))
            e_to_s₁_by_e_to_s₁ * vec(ℒ.I(nᵉ))]

    yv₂ = (vec(v_v_to_y₂) + e_e_to_y₂ * vec(ℒ.I(nᵉ))) / 2

    ## Mean
    μˢ⁺₂ = (ℒ.I - ŝ_to_ŝ₂) \ ŝv₂
    Δμˢ₂ = vec((ℒ.I - s_to_s₁) \ (s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂ + e_e_to_s₂ * vec(ℒ.I(nᵉ))) / 2))
    μʸ₂  = SS_and_pars[1:𝓂.timings.nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

    # if !covariance
    #     return μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂
    # end

    # Covariance
    Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
            zeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     zeros(nᵉ^2, nᵉ * nˢ)
            zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

    C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    
    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(ŝ_to_ŝ₂))
    push!(dimensions,size(C))
    
    values = vcat(vec(ŝ_to_ŝ₂), vec(collect(-C)))

    Σᶻ₂, info = solve_sylvester_equation_forward(values, coords = coordinates, dims = dimensions, solver = :doubling)
    # Σᶻ₂, info = solve_sylvester_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)
    # Σᶻ₂, info = solve_sylvester_equation_AD([vec(ŝ_to_ŝ₂); vec(-C)], dims = [size(ŝ_to_ŝ₂) ;size(C)])#, solver = :doubling)
    # Σᶻ₂, info = solve_sylvester_equation_forward([vec(ŝ_to_ŝ₂); vec(-C)], dims = [size(ŝ_to_ŝ₂) ;size(C)])
    
    Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'

    return sqrt.(ℒ.diag(Σʸ₂))
end




function calc_cov_3rd(parameters::Vector{T}) where T

    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose)

    nᵉ = 𝓂.timings.nExo

    nˢ = 𝓂.timings.nPast_not_future_and_mixed

    iˢ = 𝓂.timings.past_not_future_and_mixed_idx

    Σᶻ₁ = Σʸ₁[iˢ, iˢ]

    # precalc second order
    ## mean
    I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

    ## covariance
    E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

    quadrup = multiplicate(nᵉ, 4)

    comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

    comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

    for j = 1:size(comb⁴,1)
        E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
    end

    e⁴ = quadrup * E_e⁴

    # second order
    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
    v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
    kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
    kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

    # first order
    s_to_y₁ = 𝐒₁[:, 1:nˢ]
    e_to_y₁ = 𝐒₁[:, (nˢ + 1):end]
    
    s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
    e_to_s₁ = 𝐒₁[iˢ, (nˢ + 1):end]


    # second order
    s_s_to_y₂ = 𝐒₂[:, kron_s_s]
    e_e_to_y₂ = 𝐒₂[:, kron_e_e]
    v_v_to_y₂ = 𝐒₂[:, kron_v_v]
    s_e_to_y₂ = 𝐒₂[:, kron_s_e]

    s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
    e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
    v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
    s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

    s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
    e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
    s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

    # # Set up in pruned state transition matrices
    ŝ_to_ŝ₂ = [ s_to_s₁             zeros(nˢ, nˢ + nˢ^2)
                zeros(nˢ, nˢ)       s_to_s₁             s_s_to_s₂ / 2
                zeros(nˢ^2, 2*nˢ)   s_to_s₁_by_s_to_s₁                  ]
    # println(ŝ_to_ŝ₂)
    ê_to_ŝ₂ = [ e_to_s₁         zeros(nˢ, nᵉ^2 + nᵉ * nˢ)
                zeros(nˢ,nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
                zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁]

    ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2]

    ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

    ŝv₂ = [ zeros(nˢ) 
            vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec(ℒ.I(nᵉ))
            e_to_s₁_by_e_to_s₁ * vec(ℒ.I(nᵉ))]

    yv₂ = (vec(v_v_to_y₂) + e_e_to_y₂ * vec(ℒ.I(nᵉ))) / 2

    ## Mean
    μˢ⁺₂ = (ℒ.I - ŝ_to_ŝ₂) \ ŝv₂
    Δμˢ₂ = vec((ℒ.I - s_to_s₁) \ (s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂ + e_e_to_s₂ * vec(ℒ.I(nᵉ))) / 2))
    μʸ₂  = SS_and_pars[1:𝓂.timings.nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

    # if !covariance
    #     return μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂
    # end

    # Covariance
    Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
            zeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     zeros(nᵉ^2, nᵉ * nˢ)
            zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

    C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

    r1,c1,v1 = findnz(sparse(ŝ_to_ŝ₂))

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(ŝ_to_ŝ₂))
    push!(dimensions,size(C))
    
    values = vcat(v1, vec(collect(-C)))

    Σᶻ₂, info = solve_sylvester_equation_forw(values, coords = coordinates, dims = dimensions, solver = :doubling)
    # Σᶻ₂, info = solve_sylvester_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)
    # Σᶻ₂, info = solve_sylvester_equation_AD([vec(ŝ_to_ŝ₂); vec(-C)], dims = [size(ŝ_to_ŝ₂) ;size(C)])#, solver = :doubling)
    # Σᶻ₂, info = solve_sylvester_equation_forward([vec(ŝ_to_ŝ₂); vec(-C)], dims = [size(ŝ_to_ŝ₂) ;size(C)])
    
    Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'


    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
                                                𝓂.solution.perturbation.second_order_auxilliary_matrices, 
                                                𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    orders = determine_efficient_order(𝐒₁, 𝓂.timings, observables, tol = dependencies_tol)

    nᵉ = 𝓂.timings.nExo

    # precalc second order
    ## covariance
    E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

    quadrup = multiplicate(nᵉ, 4)

    comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

    comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

    for j = 1:size(comb⁴,1)
        E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
    end

    e⁴ = quadrup * E_e⁴

    # precalc third order
    sextup = multiplicate(nᵉ, 6)
    E_e⁶ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4 * (nᵉ + 4)÷5 * (nᵉ + 5)÷6)

    comb⁶   = reduce(vcat, generateSumVectors(nᵉ, 6))

    comb⁶ = comb⁶ isa Int64 ? reshape([comb⁶],1,1) : comb⁶

    for j = 1:size(comb⁶,1)
        E_e⁶[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁶[j,:])
    end

    e⁶ = sextup * E_e⁶

    Σʸ₃ = zeros(T, size(Σʸ₂))

    # if autocorrelation
    #     autocorr = zeros(T, size(Σʸ₂,1), length(autocorrelation_periods))
    # end

    # Threads.@threads for ords in orders 
    # for ords in orders 
        ords = orders[1]
        variance_observable, dependencies_all_vars = ords

        sort!(variance_observable)

        sort!(dependencies_all_vars)

        dependencies = intersect(𝓂.timings.past_not_future_and_mixed, dependencies_all_vars)

        obs_in_y = indexin(variance_observable, 𝓂.timings.var)

        dependencies_in_states_idx = indexin(dependencies, 𝓂.timings.past_not_future_and_mixed)

        dependencies_in_var_idx = Int.(indexin(dependencies, 𝓂.timings.var))

        nˢ = length(dependencies)

        iˢ = dependencies_in_var_idx

        Σ̂ᶻ₁ = Σʸ₁[iˢ, iˢ]

        dependencies_extended_idx = vcat(dependencies_in_states_idx, 
                dependencies_in_states_idx .+ 𝓂.timings.nPast_not_future_and_mixed, 
                findall(ℒ.kron(𝓂.timings.past_not_future_and_mixed .∈ (intersect(𝓂.timings.past_not_future_and_mixed,dependencies),), 𝓂.timings.past_not_future_and_mixed .∈ (intersect(𝓂.timings.past_not_future_and_mixed,dependencies),))) .+ 2*𝓂.timings.nPast_not_future_and_mixed)
        
        Σ̂ᶻ₂ = Σᶻ₂[dependencies_extended_idx, dependencies_extended_idx]
        
        Δ̂μˢ₂ = Δμˢ₂[dependencies_in_states_idx]

        s_in_s⁺ = BitVector(vcat(𝓂.timings.past_not_future_and_mixed .∈ (dependencies,), zeros(Bool, nᵉ + 1)))
        e_in_s⁺ = BitVector(vcat(zeros(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), ones(Bool, nᵉ)))
        v_in_s⁺ = BitVector(vcat(zeros(Bool, 𝓂.timings.nPast_not_future_and_mixed), 1, zeros(Bool, nᵉ)))

        # precalc second order
        ## mean
        I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

        e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ*nˢ)), nˢ*nᵉ^2, nˢ*nᵉ^2))
        e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ*nˢ^2, nᵉ*nˢ^2))
        ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
        s_s  = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))

        # first order
        s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
        e_to_y₁ = 𝐒₁[obs_in_y,:][:, (𝓂.timings.nPast_not_future_and_mixed + 1):end]
        
        s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
        e_to_s₁ = 𝐒₁[iˢ, (𝓂.timings.nPast_not_future_and_mixed + 1):end]

        # second order
        kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
        kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
        kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

        s_s_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_s]
        e_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_e_e]
        s_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_e]

        s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
        e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
        v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
        s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

        s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
        e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
        s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

        # third order
        kron_s_v = ℒ.kron(s_in_s⁺, v_in_s⁺)
        kron_e_v = ℒ.kron(e_in_s⁺, v_in_s⁺)

        s_s_s_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, s_in_s⁺)]
        s_s_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, e_in_s⁺)]
        s_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_e, e_in_s⁺)]
        e_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_e, e_in_s⁺)]
        s_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_v, v_in_s⁺)]
        e_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_v, v_in_s⁺)]

        s_s_s_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, s_in_s⁺)]
        s_s_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, e_in_s⁺)]
        s_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_e, e_in_s⁺)]
        e_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_e, e_in_s⁺)]
        s_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_v, v_in_s⁺)]
        e_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_v, v_in_s⁺)]

        # Set up pruned state transition matrices
        ŝ_to_ŝ₃ = [  s_to_s₁                spzeros(nˢ, 2*nˢ + 2*nˢ^2 + nˢ^3)
                                            spzeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2   spzeros(nˢ, nˢ + nˢ^2 + nˢ^3)
                                            spzeros(nˢ^2, 2 * nˢ)               s_to_s₁_by_s_to_s₁  spzeros(nˢ^2, nˢ + nˢ^2 + nˢ^3)
                                            s_v_v_to_s₃ / 2    spzeros(nˢ, nˢ + nˢ^2)      s_to_s₁       s_s_to_s₂    s_s_s_to_s₃ / 6
                                            ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    spzeros(nˢ^2, 2*nˢ + nˢ^2)     s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2)    
                                            spzeros(nˢ^3, 3*nˢ + 2*nˢ^2)   ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁)]

        ê_to_ŝ₃ = [ e_to_s₁   spzeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        spzeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   spzeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        spzeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁  spzeros(nˢ^2, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        e_v_v_to_s₃ / 2    spzeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                        ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    spzeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                        spzeros(nˢ^3, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss   ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es  ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6]

        ê_to_y₃ = [e_to_y₁ + e_v_v_to_y₃ / 2  e_e_to_y₂ / 2  s_e_to_y₂   s_e_to_y₂     s_s_e_to_y₃ / 2    s_e_e_to_y₃ / 2    e_e_e_to_y₃ / 6]

        μˢ₃δμˢ₁ = reshape((ℒ.I - s_to_s₁_by_s_to_s₁) \ vec( 
                                    (s_s_to_s₂  * reshape(ss_s * vec(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂'),nˢ^2, nˢ) +
                                    s_s_s_to_s₃ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ^3, nˢ) / 6 +
                                    s_e_e_to_s₃ * ℒ.kron(Σ̂ᶻ₁, vec(ℒ.I(nᵉ))) / 2 +
                                    s_v_v_to_s₃ * Σ̂ᶻ₁ / 2) * s_to_s₁' +
                                    (s_e_to_s₂  * ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ)) +
                                    e_e_e_to_s₃ * reshape(e⁴, nᵉ^3, nᵉ) / 6 +
                                    s_s_e_to_s₃ * ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ)) / 2 +
                                    e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                    ), nˢ, nˢ)

        Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σ̂ᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    reshape(e⁴, nᵉ, nᵉ^3)
                spzeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σ̂ᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δ̂μˢ₂ * Δ̂μˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δ̂μˢ₂, reshape(e⁴, nᵉ, nᵉ^3))
                ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σ̂ᶻ₁), reshape(e⁴, nᵉ, nᵉ^3))
                spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σ̂ᶻ₁, reshape(e⁴, nᵉ^2, nᵉ^2))    spzeros(nˢ*nᵉ^2,nᵉ^3)
                reshape(e⁴, nᵉ^3, nᵉ)  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', reshape(e⁴, nᵉ^3, nᵉ))     ℒ.kron(vec(Σ̂ᶻ₁)', reshape(e⁴, nᵉ^3, nᵉ))  spzeros(nᵉ^3, nˢ*nᵉ^2)     reshape(e⁶, nᵉ^3, nᵉ^3)]


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
                ℒ.kron(Σ̂ᶻ₁,vec(ℒ.I(nᵉ)))   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec(ℒ.I(nᵉ)))    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec(ℒ.I(nᵉ)))  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec(ℒ.I(nᵉ)))
                spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]
        
        droptol!(ŝ_to_ŝ₃, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())
        
        A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
        droptol!(A, eps())

        C = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A + A'
        droptol!(C, eps())

        r1,c1,v1 = findnz(ŝ_to_ŝ₃)

        coordinates = Tuple{Vector{Int}, Vector{Int}}[]
        push!(coordinates,(r1,c1))
        
        dimensions = Tuple{Int, Int}[]
        push!(dimensions,size(ŝ_to_ŝ₃))
        push!(dimensions,size(C))
        
        values = vcat(v1, vec(collect(-C)))

        Σᶻ₃, info = solve_sylvester_equation_forw(values, coords = coordinates, dims = dimensions, solver = :doubling)
        # Σᶻ₃, info = solve_sylvester_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃' + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

        for obs in variance_observable
            Σʸ₃[indexin([obs], 𝓂.timings.var), indexin(variance_observable, 𝓂.timings.var)] = Σʸ₃tmp[indexin([obs], variance_observable), :]
        end

    # end



    return sqrt.(ℒ.diag(Σʸ₃))
end


using SparseArrays

function jac_wrt_A(A::AbstractSparseArray{T}, X::Matrix{T}) where T
    # does this without creating dense arrays: reshape(permutedims(reshape(ℒ.I - ℒ.kron(A, B) ,size(B,1), size(A,1), size(A,1), size(B,1)), [2, 3, 4, 1]), size(A,1) * size(B,1), size(A,1) * size(B,1))

    # Compute the Kronecker product and subtract from identity
    C = ℒ.kron(ℒ.I(size(A,1)), A * X)|>sparse

    # Extract the row, column, and value indices from C
    rows, cols, vals = findnz(C)

    # Lists to store the 2D indices after the operations
    final_rows = zeros(Int,length(rows))
    final_cols = zeros(Int,length(rows))

    Threads.@threads for i = 1:length(rows)
        # Convert the 1D row index to its 2D components
        i1, i2 = divrem(rows[i]-1, size(A,1)) .+ 1

        # Convert the 1D column index to its 2D components
        j1, j2 = divrem(cols[i]-1, size(A,1)) .+ 1

        # Convert the 4D index (i1, j2, j1, i2) to a 2D index in the final matrix
        final_col, final_row = divrem(Base._sub2ind((size(A,1), size(A,1), size(A,1), size(A,1)), i2, i1, j1, j2) - 1, size(A,1) * size(A,1)) .+ 1

        # Store the 2D indices
        final_rows[i] = final_row
        final_cols[i] = final_col
    end

    r,c,_ = findnz(A) 
    
    non_zeros_only = spzeros(Int,size(A,1)^2,size(A,1)^2)
    
    non_zeros_only[CartesianIndex.(r .+ (c.-1) * size(A,1), r .+ (c.-1) * size(A,1))] .= 1

    println(typeof(A))
    intmdt = sparse(X * A')
    println(nnz(intmdt)/length(intmdt))

    println(typeof(intmdt))
    out_b = ℒ.kron(intmdt, ℒ.I(size(A,1)))' * non_zeros_only
    println(typeof(out_b))
    return sparse(final_rows, final_cols, vals, size(A,1) * size(A,1), size(A,1) * size(A,1)) + out_b
end


observables = :full_covar
dependencies_tol = eps()
T = m.timings

dst_dev = ℱ.jacobian(x -> calc_cov(x), 𝓂.parameter_values[param_idx])

calc_cov_2nd(𝓂.parameter_values)

dst_dev2 = ℱ.jacobian(x -> calc_cov_2nd(x), 𝓂.parameter_values[param_idx])

calc_cov_3rd(𝓂.parameter_values)

dst_dev3 = ℱ.jacobian(x -> calc_cov_3rd(x), 𝓂.parameter_values[param_idx])

using FiniteDifferences
verbose = false
fd_diff = FiniteDifferences.jacobian(central_fdm(4,1),x -> calc_cov(x), 𝓂.parameter_values[param_idx])[1]

fd_diff2 = FiniteDifferences.jacobian(central_fdm(4,1),x -> calc_cov_2nd(x), 𝓂.parameter_values[param_idx])[1]

fd_diff3 = FiniteDifferences.jacobian(central_fdm(4,1),x -> calc_cov_3rd(x), 𝓂.parameter_values[param_idx])[1]

isapprox(dst_dev,fd_diff,rtol = 1e-7)
isapprox(dst_dev2,fd_diff2,rtol = 1e-7)
isapprox(dst_dev3,fd_diff3,rtol = 1e-7)




b = sparse([1, 2, 3, 4, 5, 6, 7, 8, 15, 22, 29, 36, 43, 2, 8, 9, 10, 11, 12, 13, 14, 16, 23, 30, 37, 44, 3, 10, 15, 16, 17, 18, 19, 20, 21, 24, 31, 38, 45, 4, 11, 18, 22, 23, 24, 25, 26, 27, 28, 32, 39, 46, 5, 12, 19, 26, 29, 30, 31, 32, 33, 34, 35, 40, 47, 6, 13, 20, 27, 34, 36, 37, 38, 39, 40, 41, 42, 48, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13, 14, 16, 23, 30, 37, 44, 3, 10, 15, 16, 17, 18, 19, 20, 21, 24, 31, 38, 45, 4, 11, 18, 22, 23, 24, 25, 26, 27, 28, 32, 39, 46, 5, 12, 19, 26, 29, 30, 31, 32, 33, 34, 35, 40, 47, 6, 13, 20, 27, 34, 36, 37, 38, 39, 40, 41, 42, 48, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13, 14, 16, 23, 30, 37, 44, 3, 10, 15, 16, 17, 18, 19, 20, 21, 24, 31, 38, 45, 4, 11, 18, 22, 23, 24, 25, 26, 27, 28, 32, 39, 46, 5, 12, 19, 26, 29, 30, 31, 32, 33, 34, 35, 40, 47, 36, 37, 38, 39, 40, 41, 42, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98], [-0.00043806315789473666, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, -0.00021903157894736833, 2.767348015026762e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, -0.00021903157894736833, 1.383674007513381e-6, 4.155162692886903e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.000325307660022015, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.004030563150275439, -0.0003553645312401283, 5.622600081916384e-20, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0007107290624802567, 5.622600081916384e-20, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, -0.0016782420570060751, 0.00021333060279185635, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, -0.0016782420570060751, 0.00010666530139592818, 0.00032031510209720173, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.005113839279308015, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.08076253753165824, -0.00334109083024237, 0.0001153588170270328, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00668218166048474, 0.0001153588170270328, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, 5.62260008191638e-20, -1.2576572484385765e-6, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, 5.62260008191638e-20, -6.288286242192882e-7, -1.8883676540769768e-6, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 3.007982917839462e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00027705240434624753, 3.0246448933223264e-6, -0.00011842105263157913, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00023684210526315826, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 49, 98)


partials = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0; 0.005264651767825495 0.02127810051143547 1.0279089128508414 -0.02624238439776989 0.0 0.0 1.0099938087710185 0.11411531661345238; 0.00790485493617888 0.03191561156492047 1.5434013875997297 -0.01701796753705211 0.0 0.0 1.5499955036025694 0.17134372102982465; -0.02127645823933927 -0.007933294307888318 4.4564051312049635 0.0 0.0 0.0 7.961060837967423 2.0395768091488695; 0.4859348084363515 -0.014567600452916567 18.157134734000476 0.0 0.0 0.0 14.618587054504541 -0.6180377473702475; 0.46465835019701224 -0.022500894760804885 22.61353986520544 0.0 0.0 0.0 22.579647892471964 1.421539061778622; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0037373987139470207 -0.0035971993817412395 -0.05486934668004898 0.006671388065009553 0.0 0.0 -0.0508355892595231 4.718898484400319e-17; 0.005611690188697011 -0.005395717830148978 -0.08238611879316084 0.006352787559030381 0.0 0.0 -0.08181215534243785 7.085408422587112e-17; -0.019679817324683605 -0.001314383708171508 0.6400400943878555 0.0 0.0 0.0 1.3189840511503563 1.5966050825266647e-15; 0.01967981732468364 0.00031936502047696583 -0.6400400943878534 0.0 0.0 0.0 -0.32048279804869456 -1.5966050825266647e-15; 3.605063934384798e-17 -0.000995018687694542 1.000000000000002 0.0 0.0 0.0 0.9985012531016617 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; -1.0838562210045907e-33 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 8.118009813235019e-5 0.0011815189682540118 0.08847966878758272 -0.0034240316003972446 0.0 0.0 0.027477751836416108 -0.0007933466801269788; 0.00012189161367952133 0.0017722356776159781 0.13285189171322584 -0.0039268284966493365 0.0 0.0 0.04307471238567444 -0.0011912070725796859; -0.0455890905797699 0.002317701010745205 -4.666252896001847 0.0 0.0 0.0 -2.3258129642832523 -0.5059622303496306; -0.13979581815050332 0.001058873260640222 -7.222342783519393 0.0 0.0 0.0 -1.0625793170526643 0.2926403925828835; -9.872157831778827e-18 4.780552944776056e-19 -4.80448558795675e-16 0.0 0.0 0.0 -4.797284880083678e-16 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0; 5.898059818321144e-21 -0.0 1.887379141862766e-19 -0.0 -0.0136 -0.0 1.887379141862766e-19 -0.0; -2.704861086047285e-7 -1.0932215196097583e-6 -5.2811675700247666e-5 1.3482753939476523e-6 -0.00033792324482128267 -0.0 -5.189123746396887e-5 -4.586392453126312e-6; -4.061338802765761e-7 -1.639753198624369e-6 -7.929653351401279e-5 8.743453543481e-7 -0.000507390488047118 -0.0 -7.963532454064774e-5 -6.886451112287665e-6; 1.0931371433189531e-6 4.075950319963956e-7 -0.00022896019251879735 -0.0 -0.004344702107992636 -0.0 -0.0004090216146084594 -8.837560498652089e-5; -2.49662506023299e-5 7.484509388254025e-7 -0.0009328732334446472 -0.0 -0.014988229132196624 -0.0 -0.0007510705171114339 8.837560498652089e-5; -2.3873113459010948e-5 1.1560459708217983e-6 -0.0011618334259634443 -0.0 -0.01933293124018926 -0.0 -0.0011600921317198934 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; -2.704861086047285e-7 -1.0932215196097583e-6 -5.2811675700247666e-5 1.3482753939476523e-6 -0.00033792324482128267 -0.0 -5.189123746396887e-5 -4.586392453126312e-6; -1.3447776720127575e-8 -5.4415678158723504e-8 -2.631080598928588e-6 6.725840697428902e-8 -8.396479366951806e-6 -1.8169004172003877e-8 -2.5807716838532483e-6 -2.2775856623902527e-7; -2.0191786441867364e-8 -8.166226067624208e-8 -3.9505576773438975e-6 7.236627197678724e-8 -1.2607282361201206e-5 -2.7280691800407822e-8 -3.917844926561284e-6 -3.419786352431849e-7; -5.737686198781513e-8 -3.368007602716632e-7 -2.2205798670817705e-5 4.2376180483627956e-7 -0.00010795410544961313 9.867694361734233e-7 -2.6597472005638933e-5 -3.649230345887875e-6; -9.128897699897738e-7 -1.1816569252331998e-6 -8.07670493025312e-5 1.4725748694294862e-6 -0.0003724169869468207 1.8886478468873777e-6 -7.57034115304596e-5 -2.8831940094361038e-6; -9.776889193483231e-7 -1.5253324752417646e-6 -0.00010394228060554828 1.9166261385315476e-6 -0.00048037109239643376 -0.0 -0.00010259057524276972 -6.519736024765731e-6; -2.2550027258982654e-9 -3.281997134038848e-8 -2.457768577432816e-6 9.511198889992253e-8 -0.0 -1.3479244849769545e-5 -7.632708843448823e-7 5.947975458621918e-8; -4.061338802765761e-7 -1.639753198624369e-6 -7.929653351401279e-5 8.743453543481e-7 -0.000507390488047118 -0.0 -7.963532454064774e-5 -6.886451112287665e-6; -2.0191786441867364e-8 -8.166226067624208e-8 -3.9505576773438975e-6 7.236627197678724e-8 -1.2607282361201206e-5 -2.7280691800407822e-8 -3.917844926561284e-6 -3.419786352431849e-7; -3.031789181209063e-8 -1.2255150028667416e-7 -5.931747574884692e-6 6.568189902176796e-8 -1.8929787305933285e-5 -4.096185669083678e-8 -5.946932426122967e-6 -5.134796415958145e-7; -8.615114364803864e-8 -5.051609402503826e-7 -3.3341923640147985e-5 2.713360620142395e-7 -0.00016209268551424456 1.4816306172569033e-6 -4.0482050664033214e-5 -5.4793057138630665e-6; -1.3707005748400073e-6 -1.7723706347299228e-6 -0.00012127142240661855 9.483111718012533e-7 -0.0005591827128196528 2.8357976773338906e-6 -0.0001155577974420128 -4.329105020153314e-6; -1.467996255211036e-6 -2.287843316951849e-6 -0.0001560689455177073 1.2429160746904948e-6 -0.0007212753983338973 -0.0 -0.0001564855857740792 -9.789359253145755e-6; -3.3858781577639546e-9 -4.922876882266497e-8 -3.690330325367328e-6 1.0907856935137045e-7 -0.0 -2.0239035720813576e-5 -1.196519788490941e-6 8.930862901836054e-8; 1.0931371433189531e-6 4.075950319963956e-7 -0.00022896019251879735 -0.0 -0.004344702107992636 -0.0 -0.0004090216146084594 -8.837560498652089e-5; -5.737686198781513e-8 -3.368007602716632e-7 -2.2205798670817705e-5 4.2376180483627956e-7 -0.00010795410544961313 9.867694361734233e-7 -2.6597472005638933e-5 -3.649230345887875e-6; -8.615114364803864e-8 -5.051609402503826e-7 -3.3341923640147985e-5 2.713360620142395e-7 -0.00016209268551424456 1.4816306172569033e-6 -4.0482050664033214e-5 -5.4793057138630665e-6; 5.130222916423243e-7 2.6984954645751176e-7 -0.00016526659995335605 -0.0 -0.0013879732652349746 -5.359203569705735e-5 -0.00027079401987016366 -5.8225571980051944e-5; -7.232797127451814e-6 6.994771808216444e-7 -0.00058319853283498 -0.0 -0.004788190493068479 -0.00010257358924903931 -0.000701925350954652 -6.996814234783108e-5; -6.072648613369857e-6 9.48726624543744e-7 -0.0006966391024725066 -0.0 -0.006176163758303453 -0.0 -0.0009520471677298258 -0.000125629374596657; 1.2663636272158303e-6 -6.438058363181124e-8 0.00012961813600005125 -0.0 -0.0 0.0007320658146441299 6.460591567453478e-5 1.2020990246811601e-5; -2.49662506023299e-5 7.484509388254025e-7 -0.0009328732334446472 -0.0 -0.014988229132196624 -0.0 -0.0007510705171114339 8.837560498652089e-5; -9.128897699897738e-7 -1.1816569252331998e-6 -8.07670493025312e-5 1.4725748694294862e-6 -0.0003724169869468207 1.8886478468873777e-6 -7.57034115304596e-5 -2.8831940094361038e-6; -1.3707005748400073e-6 -1.7723706347299228e-6 -0.00012127142240661855 9.483111718012533e-7 -0.0005591827128196528 2.8357976773338906e-6 -0.0001155577974420128 -4.329105020153314e-6; -7.232797127451814e-6 6.994771808216444e-7 -0.00058319853283498 -0.0 -0.004788190493068479 -0.00010257358924903931 -0.000701925350954652 -6.996814234783108e-5; -5.6117590732395114e-5 1.657941602976253e-6 -0.0021124138131076563 -0.0 -0.01651816268523732 -0.00019632285048668796 -0.0016637443985869859 0.00019816185667571407; -6.180047800314419e-5 2.3380023849262065e-6 -0.002606544094101728 -0.0 -0.021306353178305797 -0.0 -0.0023461853932738924 0.000125629374596657; 3.883217170847313e-6 -2.9413146128895035e-8 0.0002006206328755387 -0.0 -0.0 0.0014011525630233416 2.951609214035176e-5 -1.2020990246811601e-5; -2.3873113459010948e-5 1.1560459708217983e-6 -0.0011618334259634443 -0.0 -0.01933293124018926 -0.0 -0.0011600921317198934 -0.0; -9.776889193483231e-7 -1.5253324752417646e-6 -0.00010394228060554828 1.9166261385315476e-6 -0.00048037109239643376 -0.0 -0.00010259057524276972 -6.519736024765731e-6; -1.467996255211036e-6 -2.287843316951849e-6 -0.0001560689455177073 1.2429160746904948e-6 -0.0007212753983338973 -0.0 -0.0001564855857740792 -9.789359253145755e-6; -6.072648613369857e-6 9.48726624543744e-7 -0.0006966391024725066 -0.0 -0.006176163758303453 -0.0 -0.0009520471677298258 -0.000125629374596657; -6.180047800314419e-5 2.3380023849262065e-6 -0.002606544094101728 -0.0 -0.021306353178305797 -0.0 -0.0023461853932738924 0.000125629374596657; -6.787312661651405e-5 3.286729009469951e-6 -0.0033031831965742342 -0.0 -0.02748251693660925 -0.0 -0.0032982325610037186 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; -2.2550027258982654e-9 -3.281997134038848e-8 -2.457768577432816e-6 9.511198889992253e-8 -0.0 -1.3479244849769545e-5 -7.632708843448823e-7 5.947975458621918e-8; -3.3858781577639546e-9 -4.922876882266497e-8 -3.690330325367328e-6 1.0907856935137045e-7 -0.0 -2.0239035720813576e-5 -1.196519788490941e-6 8.930862901836054e-8; 1.2663636272158303e-6 -6.438058363181124e-8 0.00012961813600005125 -0.0 -0.0 0.0007320658146441299 6.460591567453478e-5 1.2020990246811601e-5; 3.883217170847313e-6 -2.9413146128895035e-8 0.0002006206328755387 -0.0 -0.0 0.0014011525630233416 2.951609214035176e-5 -1.2020990246811601e-5; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.01 -0.0 -0.0]


b * partials

A = [0.8999999999999994 0.0 0.0 0.0 8.326672684688674e-17 0.0 -1.9121451373538808e-17; 0.0223625676719964 0.0 0.0 0.0 -0.003660625168836744 0.0 0.0012131320364793152; 0.03357731170900023 0.0 0.0 0.0 -0.00549641499789681 0.0 0.0018215132148732716; 0.28751705126421856 0.0 0.0 0.0 0.0497026832029341 0.0 -0.0658859233179717; 0.9918681043365404 0.0 0.0 0.0 0.9512948230314797 0.0 -0.1261037306721008; 1.279385155600759 0.0 0.0 0.0 0.02359750623441393 0.0 -2.7108884718944e-17; 0.0 0.0 0.0 0.0 0.0 0.0 0.9000000000000001]

val = [0.00024336842105263166 -9.636757436968988e-8 -1.4469555243891755e-7 0.0001611604341770818 0.0018647133966734181 0.000385560044332759 -6.247333424351537e-20; -9.636757436969007e-8 3.8887548781005304e-7 5.838950903003358e-7 -5.3762354741557e-6 -0.00010166037262935646 -2.654025385941416e-6 5.996438739217837e-7; -1.4469555243891797e-7 5.838950903003358e-7 8.767162939397377e-7 -8.072397453838355e-6 -0.00015264266922727208 -3.9850091892153684e-6 9.003630336366737e-7; 0.0001611604341770818 -5.3762354741557e-6 -8.072397453838357e-6 0.00017969552425705414 0.002612148433134713 0.0002894327795335548 -1.5366087666887907e-5; 0.0018647133966734181 -0.00010166037262935647 -0.00015264266922727208 0.0026121484331347125 0.04048750667115813 0.003603660173848351 -0.00012817646336336976; 0.000385560044332759 -2.6540253859414153e-6 -3.985009189215368e-6 0.0002894327795335548 0.003603660173848351 0.0006269300753765012 -2.7221804039900978e-6; -6.247333424351537e-20 5.996438739217836e-7 9.003630336366737e-7 -1.5366087666887907e-5 -0.00012817646336336976 -2.722180403990098e-6 0.00013157894736842124]


A
partials
b = hcat(jac_wrt_A(sparse(A), -val), ℒ.I(length(val)))


B = sparse(A')

reshape_matmul_a = LinearOperators.LinearOperator(Float64, length(A) * size(partials,2), length(A) * size(partials,2), false, false, 
(sol,𝐱) -> begin 
𝐗 = reshape(𝐱, (length(A),size(partials,2))) |> sparse
a = jacobian_wrt_values(A, B)
droptol!(a,eps())
sol .= vec(a * 𝐗)
return sol
end)

a*jvp|>vec
reshape_matmul_aa*vec(jvp)

reshape_matmul_b = LinearOperators.LinearOperator(Float64, length(val) * size(partials,2), 2*size(A,1)^2 * size(partials,2), false, false, 
    (sol,𝐱) -> begin 
    𝐗 = reshape(𝐱, (2* size(A,1)^2,size(partials,2))) |> sparse
    b = hcat(jac_wrt_A(sparse(A), -val), ℒ.I(length(val)))
    droptol!(b,eps())
    sol .= vec(b * 𝐗)
    return sol
end)

reshape_matmul_b * vec(partials)
b*partials


b = sparse([1, 2, 3, 4, 5, 6, 7, 8, 15, 22, 29, 36, 43, 2, 8, 9, 10, 11, 12, 13, 14, 16, 23, 30, 37, 44, 3, 10, 15, 16, 17, 18, 19, 20, 21, 24, 31, 38, 45, 4, 11, 18, 22, 23, 24, 25, 26, 27, 28, 32, 39, 46, 5, 12, 19, 26, 29, 30, 31, 32, 33, 34, 35, 40, 47, 6, 13, 20, 27, 34, 36, 37, 38, 39, 40, 41, 42, 48, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13, 14, 16, 23, 30, 37, 44, 3, 10, 15, 16, 17, 18, 19, 20, 21, 24, 31, 38, 45, 4, 11, 18, 22, 23, 24, 25, 26, 27, 28, 32, 39, 46, 5, 12, 19, 26, 29, 30, 31, 32, 33, 34, 35, 40, 47, 6, 13, 20, 27, 34, 36, 37, 38, 39, 40, 41, 42, 48, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13, 14, 16, 23, 30, 37, 44, 3, 10, 15, 16, 17, 18, 19, 20, 21, 24, 31, 38, 45, 4, 11, 18, 22, 23, 24, 25, 26, 27, 28, 32, 39, 46, 5, 12, 19, 26, 29, 30, 31, 32, 33, 34, 35, 40, 47, 36, 37, 38, 39, 40, 41, 42, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98], [-0.00043806315789473666, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, -0.00021903157894736833, 2.767348015026762e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, -0.00021903157894736833, 1.383674007513381e-6, 4.155162692886903e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.000325307660022015, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.004030563150275439, -0.0003553645312401283, 5.622600081916384e-20, -0.0003553645312401283, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0007107290624802567, 5.622600081916384e-20, 5.622600081916384e-20, -0.00021903157894736833, 1.383674007513381e-6, 2.0775813464434515e-6, -0.0001626538300110075, -0.0020152815751377194, -0.0003553645312401283, 5.622600081916384e-20, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 8.673081693272083e-8, -3.707129395110491e-7, -5.566240919690883e-7, 5.120008706204698e-6, 9.688018744264373e-5, 2.522222521043916e-6, -5.396794865296052e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, 1.302259971950257e-7, -5.566240919690882e-7, -8.357689919565852e-7, 7.687673920214168e-6, 0.00014546523905041483, 3.7871076805999382e-6, -8.103267302730065e-7, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.00014504439075937352, 5.976776276961131e-6, 8.974107222838732e-6, -0.00017717956773990884, -0.0026467108967527485, -0.0002678264560924426, 1.3829478900199118e-5, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, -0.0016782420570060751, 0.00021333060279185635, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, -0.0016782420570060751, 0.00010666530139592818, 0.00032031510209720173, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.005113839279308015, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.08076253753165824, -0.00334109083024237, 0.0001153588170270328, -0.00334109083024237, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00668218166048474, 0.0001153588170270328, 0.0001153588170270328, -0.0016782420570060751, 0.00010666530139592818, 0.00016015755104860087, -0.0025569196396540077, -0.04038126876582912, -0.00334109083024237, 0.0001153588170270328, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, -0.00034700403989948285, 4.5728389135717715e-6, 6.866100523345643e-6, -0.0002901460203930262, -0.00381091125473141, -0.000578317190731198, 2.4499623635910886e-6, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, 5.62260008191638e-20, -1.2576572484385765e-6, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, 5.62260008191638e-20, -6.288286242192882e-7, -1.8883676540769768e-6, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 3.007982917839462e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00027705240434624753, 3.0246448933223264e-6, -0.00011842105263157913, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00011842105263157913, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, 5.62260008191638e-20, -6.288286242192882e-7, -9.441838270384884e-7, 1.503991458919731e-5, 0.00013852620217312376, 3.0246448933223264e-6, -0.00023684210526315826, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 49, 98)

partials = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0; 0.005264651767825495 0.02127810051143547 1.0279089128508414 -0.02624238439776989 0.0 0.0 1.0099938087710185 0.11411531661345238; 0.00790485493617888 0.03191561156492047 1.5434013875997297 -0.01701796753705211 0.0 0.0 1.5499955036025694 0.17134372102982465; -0.02127645823933927 -0.007933294307888318 4.4564051312049635 0.0 0.0 0.0 7.961060837967423 2.0395768091488695; 0.4859348084363515 -0.014567600452916567 18.157134734000476 0.0 0.0 0.0 14.618587054504541 -0.6180377473702475; 0.46465835019701224 -0.022500894760804885 22.61353986520544 0.0 0.0 0.0 22.579647892471964 1.421539061778622; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0037373987139470207 -0.0035971993817412395 -0.05486934668004898 0.006671388065009553 0.0 0.0 -0.0508355892595231 4.718898484400319e-17; 0.005611690188697011 -0.005395717830148978 -0.08238611879316084 0.006352787559030381 0.0 0.0 -0.08181215534243785 7.085408422587112e-17; -0.019679817324683605 -0.001314383708171508 0.6400400943878555 0.0 0.0 0.0 1.3189840511503563 1.5966050825266647e-15; 0.01967981732468364 0.00031936502047696583 -0.6400400943878534 0.0 0.0 0.0 -0.32048279804869456 -1.5966050825266647e-15; 3.605063934384798e-17 -0.000995018687694542 1.000000000000002 0.0 0.0 0.0 0.9985012531016617 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; -1.0838562210045907e-33 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 8.118009813235019e-5 0.0011815189682540118 0.08847966878758272 -0.0034240316003972446 0.0 0.0 0.027477751836416108 -0.0007933466801269788; 0.00012189161367952133 0.0017722356776159781 0.13285189171322584 -0.0039268284966493365 0.0 0.0 0.04307471238567444 -0.0011912070725796859; -0.0455890905797699 0.002317701010745205 -4.666252896001847 0.0 0.0 0.0 -2.3258129642832523 -0.5059622303496306; -0.13979581815050332 0.001058873260640222 -7.222342783519393 0.0 0.0 0.0 -1.0625793170526643 0.2926403925828835; -9.872157831778827e-18 4.780552944776056e-19 -4.80448558795675e-16 0.0 0.0 0.0 -4.797284880083678e-16 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0; 5.898059818321144e-21 -0.0 1.887379141862766e-19 -0.0 -0.0136 -0.0 1.887379141862766e-19 -0.0; -2.704861086047285e-7 -1.0932215196097583e-6 -5.2811675700247666e-5 1.3482753939476523e-6 -0.00033792324482128267 -0.0 -5.189123746396887e-5 -4.586392453126312e-6; -4.061338802765761e-7 -1.639753198624369e-6 -7.929653351401279e-5 8.743453543481e-7 -0.000507390488047118 -0.0 -7.963532454064774e-5 -6.886451112287665e-6; 1.0931371433189531e-6 4.075950319963956e-7 -0.00022896019251879735 -0.0 -0.004344702107992636 -0.0 -0.0004090216146084594 -8.837560498652089e-5; -2.49662506023299e-5 7.484509388254025e-7 -0.0009328732334446472 -0.0 -0.014988229132196624 -0.0 -0.0007510705171114339 8.837560498652089e-5; -2.3873113459010948e-5 1.1560459708217983e-6 -0.0011618334259634443 -0.0 -0.01933293124018926 -0.0 -0.0011600921317198934 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; -2.704861086047285e-7 -1.0932215196097583e-6 -5.2811675700247666e-5 1.3482753939476523e-6 -0.00033792324482128267 -0.0 -5.189123746396887e-5 -4.586392453126312e-6; -1.3447776720127575e-8 -5.4415678158723504e-8 -2.631080598928588e-6 6.725840697428902e-8 -8.396479366951806e-6 -1.8169004172003877e-8 -2.5807716838532483e-6 -2.2775856623902527e-7; -2.0191786441867364e-8 -8.166226067624208e-8 -3.9505576773438975e-6 7.236627197678724e-8 -1.2607282361201206e-5 -2.7280691800407822e-8 -3.917844926561284e-6 -3.419786352431849e-7; -5.737686198781513e-8 -3.368007602716632e-7 -2.2205798670817705e-5 4.2376180483627956e-7 -0.00010795410544961313 9.867694361734233e-7 -2.6597472005638933e-5 -3.649230345887875e-6; -9.128897699897738e-7 -1.1816569252331998e-6 -8.07670493025312e-5 1.4725748694294862e-6 -0.0003724169869468207 1.8886478468873777e-6 -7.57034115304596e-5 -2.8831940094361038e-6; -9.776889193483231e-7 -1.5253324752417646e-6 -0.00010394228060554828 1.9166261385315476e-6 -0.00048037109239643376 -0.0 -0.00010259057524276972 -6.519736024765731e-6; -2.2550027258982654e-9 -3.281997134038848e-8 -2.457768577432816e-6 9.511198889992253e-8 -0.0 -1.3479244849769545e-5 -7.632708843448823e-7 5.947975458621918e-8; -4.061338802765761e-7 -1.639753198624369e-6 -7.929653351401279e-5 8.743453543481e-7 -0.000507390488047118 -0.0 -7.963532454064774e-5 -6.886451112287665e-6; -2.0191786441867364e-8 -8.166226067624208e-8 -3.9505576773438975e-6 7.236627197678724e-8 -1.2607282361201206e-5 -2.7280691800407822e-8 -3.917844926561284e-6 -3.419786352431849e-7; -3.031789181209063e-8 -1.2255150028667416e-7 -5.931747574884692e-6 6.568189902176796e-8 -1.8929787305933285e-5 -4.096185669083678e-8 -5.946932426122967e-6 -5.134796415958145e-7; -8.615114364803864e-8 -5.051609402503826e-7 -3.3341923640147985e-5 2.713360620142395e-7 -0.00016209268551424456 1.4816306172569033e-6 -4.0482050664033214e-5 -5.4793057138630665e-6; -1.3707005748400073e-6 -1.7723706347299228e-6 -0.00012127142240661855 9.483111718012533e-7 -0.0005591827128196528 2.8357976773338906e-6 -0.0001155577974420128 -4.329105020153314e-6; -1.467996255211036e-6 -2.287843316951849e-6 -0.0001560689455177073 1.2429160746904948e-6 -0.0007212753983338973 -0.0 -0.0001564855857740792 -9.789359253145755e-6; -3.3858781577639546e-9 -4.922876882266497e-8 -3.690330325367328e-6 1.0907856935137045e-7 -0.0 -2.0239035720813576e-5 -1.196519788490941e-6 8.930862901836054e-8; 1.0931371433189531e-6 4.075950319963956e-7 -0.00022896019251879735 -0.0 -0.004344702107992636 -0.0 -0.0004090216146084594 -8.837560498652089e-5; -5.737686198781513e-8 -3.368007602716632e-7 -2.2205798670817705e-5 4.2376180483627956e-7 -0.00010795410544961313 9.867694361734233e-7 -2.6597472005638933e-5 -3.649230345887875e-6; -8.615114364803864e-8 -5.051609402503826e-7 -3.3341923640147985e-5 2.713360620142395e-7 -0.00016209268551424456 1.4816306172569033e-6 -4.0482050664033214e-5 -5.4793057138630665e-6; 5.130222916423243e-7 2.6984954645751176e-7 -0.00016526659995335605 -0.0 -0.0013879732652349746 -5.359203569705735e-5 -0.00027079401987016366 -5.8225571980051944e-5; -7.232797127451814e-6 6.994771808216444e-7 -0.00058319853283498 -0.0 -0.004788190493068479 -0.00010257358924903931 -0.000701925350954652 -6.996814234783108e-5; -6.072648613369857e-6 9.48726624543744e-7 -0.0006966391024725066 -0.0 -0.006176163758303453 -0.0 -0.0009520471677298258 -0.000125629374596657; 1.2663636272158303e-6 -6.438058363181124e-8 0.00012961813600005125 -0.0 -0.0 0.0007320658146441299 6.460591567453478e-5 1.2020990246811601e-5; -2.49662506023299e-5 7.484509388254025e-7 -0.0009328732334446472 -0.0 -0.014988229132196624 -0.0 -0.0007510705171114339 8.837560498652089e-5; -9.128897699897738e-7 -1.1816569252331998e-6 -8.07670493025312e-5 1.4725748694294862e-6 -0.0003724169869468207 1.8886478468873777e-6 -7.57034115304596e-5 -2.8831940094361038e-6; -1.3707005748400073e-6 -1.7723706347299228e-6 -0.00012127142240661855 9.483111718012533e-7 -0.0005591827128196528 2.8357976773338906e-6 -0.0001155577974420128 -4.329105020153314e-6; -7.232797127451814e-6 6.994771808216444e-7 -0.00058319853283498 -0.0 -0.004788190493068479 -0.00010257358924903931 -0.000701925350954652 -6.996814234783108e-5; -5.6117590732395114e-5 1.657941602976253e-6 -0.0021124138131076563 -0.0 -0.01651816268523732 -0.00019632285048668796 -0.0016637443985869859 0.00019816185667571407; -6.180047800314419e-5 2.3380023849262065e-6 -0.002606544094101728 -0.0 -0.021306353178305797 -0.0 -0.0023461853932738924 0.000125629374596657; 3.883217170847313e-6 -2.9413146128895035e-8 0.0002006206328755387 -0.0 -0.0 0.0014011525630233416 2.951609214035176e-5 -1.2020990246811601e-5; -2.3873113459010948e-5 1.1560459708217983e-6 -0.0011618334259634443 -0.0 -0.01933293124018926 -0.0 -0.0011600921317198934 -0.0; -9.776889193483231e-7 -1.5253324752417646e-6 -0.00010394228060554828 1.9166261385315476e-6 -0.00048037109239643376 -0.0 -0.00010259057524276972 -6.519736024765731e-6; -1.467996255211036e-6 -2.287843316951849e-6 -0.0001560689455177073 1.2429160746904948e-6 -0.0007212753983338973 -0.0 -0.0001564855857740792 -9.789359253145755e-6; -6.072648613369857e-6 9.48726624543744e-7 -0.0006966391024725066 -0.0 -0.006176163758303453 -0.0 -0.0009520471677298258 -0.000125629374596657; -6.180047800314419e-5 2.3380023849262065e-6 -0.002606544094101728 -0.0 -0.021306353178305797 -0.0 -0.0023461853932738924 0.000125629374596657; -6.787312661651405e-5 3.286729009469951e-6 -0.0033031831965742342 -0.0 -0.02748251693660925 -0.0 -0.0032982325610037186 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; -2.2550027258982654e-9 -3.281997134038848e-8 -2.457768577432816e-6 9.511198889992253e-8 -0.0 -1.3479244849769545e-5 -7.632708843448823e-7 5.947975458621918e-8; -3.3858781577639546e-9 -4.922876882266497e-8 -3.690330325367328e-6 1.0907856935137045e-7 -0.0 -2.0239035720813576e-5 -1.196519788490941e-6 8.930862901836054e-8; 1.2663636272158303e-6 -6.438058363181124e-8 0.00012961813600005125 -0.0 -0.0 0.0007320658146441299 6.460591567453478e-5 1.2020990246811601e-5; 3.883217170847313e-6 -2.9413146128895035e-8 0.0002006206328755387 -0.0 -0.0 0.0014011525630233416 2.951609214035176e-5 -1.2020990246811601e-5; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.01 -0.0 -0.0]




jvp = [-3.1042420096426314e-20 0.0 -9.93357443085642e-19 0.0 0.0715789473684193 0.0 -9.93357443085642e-19 0.0023055955678669762; 3.929594726017388e-6 -2.0522181759352703e-7 9.80142524365059e-5 4.1000177453724345e-6 -2.83434042263703e-5 0.0 0.00010957251958524722 -1.3577425202159505e-5; 5.900271781882774e-6 -3.0799596014794427e-7 0.00014716803339655466 6.059697606326073e-6 -4.2557515423203425e-5 0.0 0.00016437839356933296 -2.0386453152670662e-5; 1.2356323648947992e-5 -5.4094959224882364e-6 0.0034720994336611658 0.0 0.047400127699140475 0.0 0.0054284291582180535 0.0024916211524111877; 0.0011431793762415965 -2.3660803841257108e-5 0.02666754270663117 0.0 0.5484451166686384 0.0 0.023743616654706038 0.026764746694082377; 0.0001499265297858648 -8.25683705674506e-6 0.008359513579983307 0.0 0.11340001303904376 0.0 0.008285735986445398 0.0038899198956684716; -6.755336557507677e-37 0.0 0.0 0.0 0.0 0.0 0.0 -5.9185264020171145e-19; 3.929594726017388e-6 -2.0522181759352703e-7 9.80142524365059e-5 4.1000177453724345e-6 -2.83434042263703e-5 0.0 0.00010957251958524722 -1.3577425202159505e-5; -1.6093324636648574e-7 7.570363463544308e-7 1.823466230430442e-5 -1.4221540921295618e-6 0.00011258756870630947 2.4311016834365775e-6 1.806500205343049e-5 3.806660503942313e-6; -2.416406674246274e-7 1.1361068346781844e-6 2.7379277240453945e-5 -1.7460956560205876e-6 0.00016904981326201844 3.6502900837834645e-6 2.7706972001224428e-5 5.7156865072909825e-6; 1.4120355018155537e-6 -5.113648376511058e-6 -0.0001785827368545907 1.2417514174242359e-5 -0.0015472913097022147 -4.6178008467217146e-5 -0.0002446893283271249 -6.161949963642431e-5; -3.442716048269863e-5 -9.886836244507225e-5 -0.002419822556309242 0.0002155150639540833 -0.02947263587559763 -0.0005813642609288624 -0.0024459709157329387 -0.0010927527998560295; 4.630926676653453e-6 -2.627251247198328e-6 -3.3841118544897934e-5 1.1057831200018292e-5 -0.0007706049157986987 -1.3587468890323159e-5 -1.757875937820664e-5 -4.602289388578257e-5; 1.0086314914774811e-7 5.850033561424393e-7 4.1759661685108064e-5 -1.2701928503705246e-6 0.0 0.00023985754956871208 1.2593006032865276e-5 7.180730732738605e-6; 5.900271781882774e-6 -3.0799596014794427e-7 0.00014716803339655466 6.059697606326073e-6 -4.2557515423203425e-5 0.0 0.00016437839356933296 -2.0386453152670662e-5; -2.416406674246274e-7 1.1361068346781844e-6 2.7379277240453996e-5 -1.7460956560205876e-6 0.00016904981326201838 3.6502900837834645e-6 2.770697200122448e-5 5.7156865072909825e-6; -3.628225582452904e-7 1.7049886827032877e-6 4.110988236029141e-5 -2.0372785666918742e-6 0.0002538276622570134 5.4808969063492096e-6 4.247644478681419e-5 8.582081910324e-6; 2.1201660240474856e-6 -7.670099726821208e-6 -0.00026814131137194494 1.3263252780511744e-5 -0.002323252113645178 -6.933610697783427e-5 -0.00037545237797330876 -9.252144820721237e-5; -5.1692252684867394e-5 -0.00014829874542365037 -0.003633354516593031 0.00022183328287613322 -0.044253052520442614 -0.0008729162631054107 -0.0038248781946476837 -0.0016407642413798715; 6.953319082326883e-6 -3.940846712760269e-6 -5.081231290744177e-5 1.3946619390558877e-5 -0.0011570603985097383 -2.0401533712876845e-5 -3.0369512960859583e-5 -6.910320301403659e-5; 1.514456411517558e-7 8.774853656652412e-7 6.270197581197928e-5 -1.3069477910620604e-6 0.0 0.00036014521345466695 1.9806469191628244e-5 1.0781840334618274e-5; 1.2356323648947992e-5 -5.4094959224882364e-6 0.0034720994336611658 0.0 0.047400127699140475 0.0 0.0054284291582180535 0.0024916211524111877; 1.4120355018155537e-6 -5.113648376511058e-6 -0.0001785827368545907 1.2417514174242359e-5 -0.001547291309702213 -4.617800846721714e-5 -0.0002446893283271248 -6.161949963642438e-5; 2.1201660240473987e-6 -7.670099726821208e-6 -0.00026814131137194494 1.3263252780511744e-5 -0.0023232521136451723 -6.933610697783427e-5 -0.0003754523779733092 -9.252144820721237e-5; 7.875397975343758e-5 -1.2395720095147393e-5 0.007851550399526397 0.0 0.05214246418644447 0.0009644584092554605 0.012439105115482848 0.0030189796964015217; 0.0025917950730236115 -0.00011901009205605153 0.0851515836030339 0.0 0.7604050038520475 0.010708568015075075 0.11942662737826837 0.03864693446872946; 0.0001612177589404694 -1.7020673375029998e-5 0.013512504955084033 0.0 0.0849476775553559 0.0002442703381350991 0.01708024573184638 0.004439497074500157; -1.1457063508408089e-5 5.271577395400857e-7 -0.0010614557048351607 0.0 0.0 -0.006146435066755075 -0.0005290027916285741 -0.00023345736752155617; 0.0011431793762415965 -2.3660803841257108e-5 0.02666754270663117 0.0 0.5484451166686384 0.0 0.023743616654706038 0.026764746694082377; -3.442716048269863e-5 -9.886836244507225e-5 -0.0024198225563092416 0.0002155150639540833 -0.029472635875597602 -0.0005813642609288624 -0.0024459709157329374 -0.00109275279985603; -5.169225268486657e-5 -0.00014829874542365005 -0.003633354516593028 0.00022183328287613322 -0.04425305252044261 -0.0008729162631054107 -0.0038248781946476802 -0.001640764241379871; 0.0025917950730236145 -0.00011901009205605153 0.0851515836030354 0.0 0.7604050038520476 0.010708568015075075 0.11942662737826837 0.03864693446872949; 0.061428211073523385 -0.0008388389325530337 0.8161829967516894 0.0 11.804924947636358 0.1403047396774141 0.8417748688171299 0.5304735062273263; 0.004034587422104815 -0.00014016030298122853 0.14431892494206355 0.0 1.0574719964059378 0.0033021544272307003 0.14065086404168853 0.05055849265866983; -0.00015787728542336767 8.201393314562576e-7 -0.006827727090260287 0.0 0.0 -0.05127058534534762 -0.0008230098191165109 -0.0016221691228076446; 0.0001499265297858648 -8.25683705674506e-6 0.008359513579983307 0.0 0.11340001303904376 0.0 0.008285735986445398 0.0038899198956684716; 4.6309266766541345e-6 -2.627251247198419e-6 -3.384111854489278e-5 1.1057831200018292e-5 -0.0007706049157986987 -1.3587468890323159e-5 -1.7578759378206262e-5 -4.6022893885781435e-5; 6.953319082327476e-6 -3.940846712760269e-6 -5.081231290744177e-5 1.3946619390558877e-5 -0.0011570603985097182 -2.0401533712876845e-5 -3.0369512960859085e-5 -6.910320301403926e-5; 0.0001612177589404693 -1.702067337503036e-5 0.013512504955084081 0.0 0.08494767755535615 0.00024427033813509907 0.01708024573184598 0.004439497074500173; 0.004034587422104814 -0.00014016030298122874 0.14431892494206355 0.0 1.0574719964059354 0.0033021544272307003 0.14065086404168858 0.05055849265866983; 0.0005013510013772744 -2.782341896934774e-5 0.02812215224805394 0.0 0.18433375186513518 7.812761401066824e-5 0.027920800935745713 0.006695650271577301; -3.3529592043452707e-6 1.3220209742064955e-7 -0.00026036441634839984 0.0 0.0 -0.001088872161596008 -0.00013266480476164426 -3.747587628317761e-5; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -5.9185264020171145e-19; 1.0086314914774811e-7 5.850033561424393e-7 4.1759661685108064e-5 -1.2701928503705246e-6 0.0 0.00023985754956871208 1.2593006032865276e-5 7.180730732738605e-6; 1.514456411517558e-7 8.774853656652412e-7 6.270197581197928e-5 -1.3069477910620604e-6 0.0 0.00036014521345466695 1.9806469191628244e-5 1.0781840334618274e-5; -1.1457063508408089e-5 5.271577395400857e-7 -0.0010614557048351607 0.0 0.0 -0.006146435066755075 -0.0005290027916285741 -0.00023345736752155617; -0.00015787728542336767 8.201393314562576e-7 -0.006827727090260287 0.0 0.0 -0.05127058534534762 -0.0008230098191165109 -0.0016221691228076446; -3.3529592043452643e-6 1.322020974206488e-7 -0.00026036441634839957 0.0 0.0 -0.001088872161596008 -0.00013266480476164472 -3.747587628317761e-5; 0.0 0.0 0.0 0.0 0.0 0.05263157894736713 0.0 0.0012465373961218587]


dst_dev2-fd_diff2
sqrt(eps())
if algorithm == :pruned_second_order
    covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)

    dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_second_order(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])

    if mean
        var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
    end
elseif algorithm == :pruned_third_order
    covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, verbose = verbose)

    dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_third_order(x, variables, param_idx, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)), 𝓂.parameter_values[param_idx])

    if mean
        var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
    end
else
    covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
    
    dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
end

standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))

st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)






if derivatives
    if non_stochastic_steady_state
        axis1 = [𝓂.var[var_idx]...,𝓂.calibration_equations_parameters...]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        axis2 = vcat(:Steady_state, 𝓂.parameters[param_idx])
    
        if any(x -> contains(string(x), "◖"), axis2)
            axis2_decomposed = decompose_name.(axis2)
            axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        end

        dNSSS = ℱ.jacobian(x -> collect(SS_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)[1]), 𝓂.parameter_values[param_idx])
        
        if length(𝓂.calibration_equations_parameters) > 0
            var_idx_ext = vcat(var_idx, 𝓂.timings.nVars .+ (1:length(𝓂.calibration_equations_parameters)))
        else
            var_idx_ext = var_idx
        end

        # dNSSS = ℱ.jacobian(x->𝓂.SS_solve_func(x, 𝓂),𝓂.parameter_values)
        SS =  KeyedArray(hcat(collect(NSSS[var_idx_ext]),dNSSS[var_idx_ext,:]);  Variables = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
    end
    
    axis1 = 𝓂.var[var_idx]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    if variance
        axis2 = vcat(:Variance, 𝓂.parameters[param_idx])
    
        if any(x -> contains(string(x), "◖"), axis2)
            axis2_decomposed = decompose_name.(axis2)
            axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        end

        if algorithm == :pruned_second_order
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)

            dvariance = ℱ.jacobian(x -> covariance_parameter_derivatives_second_order(x, param_idx, 𝓂, verbose = verbose), 𝓂.parameter_values[param_idx])

            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        elseif algorithm == :pruned_third_order
            covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, verbose = verbose)

            dvariance = ℱ.jacobian(x -> covariance_parameter_derivatives_third_order(x, variables, param_idx, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose), 𝓂.parameter_values[param_idx])

            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        else
            covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)

            dvariance = ℱ.jacobian(x -> covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose), 𝓂.parameter_values[param_idx])
        end

        vari = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))

        # dvariance = ℱ.jacobian(x-> convert(Vector{Number},max.(ℒ.diag(calculate_covariance(x, 𝓂)),eps(Float64))), Float64.(𝓂.parameter_values))
        
        
        varrs =  KeyedArray(hcat(vari[var_idx],dvariance[var_idx,:]);  Variables = axis1, Variance_and_∂variance∂parameter = axis2)

        if standard_deviation
            axis2 = vcat(:Standard_deviation, 𝓂.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))

            if algorithm == :pruned_second_order
                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_second_order(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
            elseif algorithm == :pruned_third_order
                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_third_order(x, variables, param_idx, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)), 𝓂.parameter_values[param_idx])
            else
                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
            end

            st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)
        end
    end

    if standard_deviation
        axis2 = vcat(:Standard_deviation, 𝓂.parameters[param_idx])
    
        if any(x -> contains(string(x), "◖"), axis2)
            axis2_decomposed = decompose_name.(axis2)
            axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        end

        if algorithm == :pruned_second_order
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)

            dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_second_order(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])

            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        elseif algorithm == :pruned_third_order
            covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, verbose = verbose)

            dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_third_order(x, variables, param_idx, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)), 𝓂.parameter_values[param_idx])

            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        else
            covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
            
            dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
        end

        standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))

        st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)
    end


    if mean && !(variance || standard_deviation || covariance)
        axis2 = vcat(:Mean, 𝓂.parameters[param_idx])
    
        if any(x -> contains(string(x), "◖"), axis2)
            axis2_decomposed = decompose_name.(axis2)
            axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        end

        state_μ, ___ = calculate_mean(𝓂.parameter_values, 𝓂, algorithm = algorithm, verbose = verbose)

        state_μ_dev = ℱ.jacobian(x -> mean_parameter_derivatives(x, param_idx, 𝓂, algorithm = algorithm, verbose = verbose), 𝓂.parameter_values[param_idx])
        
        var_means =  KeyedArray(hcat(state_μ[var_idx], state_μ_dev[var_idx, :]);  Variables = axis1, Mean_and_∂mean∂parameter = axis2)
    end


else
    if non_stochastic_steady_state
        axis1 = [𝓂.var[var_idx]...,𝓂.calibration_equations_parameters...]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        if length(𝓂.calibration_equations_parameters) > 0
            var_idx_ext = vcat(var_idx, 𝓂.timings.nVars .+ (1:length(𝓂.calibration_equations_parameters)))
        else
            var_idx_ext = var_idx
        end

        SS =  KeyedArray(collect(NSSS)[var_idx_ext];  Variables = axis1)
    end

    axis1 = 𝓂.var[var_idx]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    if mean && !(variance || standard_deviation || covariance)
        state_μ, ___ = calculate_mean(𝓂.parameter_values, 𝓂, algorithm = algorithm, verbose = verbose)
        var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
    end

    if variance
        if algorithm == :pruned_second_order
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        elseif algorithm == :pruned_third_order
            covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)
            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        else
            covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
        end

        varr = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))

        varrs = KeyedArray(varr[var_idx];  Variables = axis1)

        if standard_deviation
            st_dev = KeyedArray(sqrt.(varr)[var_idx];  Variables = axis1)
        end
    end

    if standard_deviation
        if algorithm == :pruned_second_order
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        elseif algorithm == :pruned_third_order
            covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)
            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        else
            covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
        end
        st_dev = KeyedArray(sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))[var_idx];  Variables = axis1)
    end

    if covariance
        if algorithm == :pruned_second_order
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        elseif algorithm == :pruned_third_order
            covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, :full_covar, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)
            if mean
                var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
            end
        else
            covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
        end
    end
end


ret = []
if non_stochastic_steady_state
    push!(ret,SS)
end
if mean
    push!(ret,var_means)
end
if standard_deviation
    push!(ret,st_dev)
end
if variance
    push!(ret,varrs)
end
if covariance
    axis1 = 𝓂.var[var_idx]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    push!(ret,KeyedArray(covar_dcmp[var_idx, var_idx]; Variables = axis1, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1))
end




σdiff2 = get_std(m, algorithm = :pruned_second_order)

