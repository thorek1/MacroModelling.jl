using Test
using MacroModelling
import LinearAlgebra as ℒ
import SparseArrays: sparse, sparsevec, findnz
using Random

const MMschur = MacroModelling

# Dense reference for the compressed second-order equation X = A X B + C and
# its adjoint, with the structure of the perturbation problem:
# A has nonzero columns only at i₊, B = 𝐔₂ (s ⊗ s + 𝛔) 𝐂₂, s = [hx 0 hu; 0 1 0; 0 0 0].
function schur_reference_problem(n, i₊, n₋, nₑ)
    nₑ₋ = n₋ + 1 + nₑ
    s = zeros(nₑ₋, nₑ₋)
    hx = randn(n₋, n₋)
    hx .*= 0.8 / maximum(abs.(ℒ.eigvals(hx)))
    s[1:n₋, 1:n₋] = hx
    s[1:n₋, n₋+2:end] = randn(n₋, nₑ)
    s[n₋+1, n₋+1] = 1
    Ā = randn(n, length(i₊))
    Ā .*= 0.7 / maximum(abs.(ℒ.eigvals(Ā[i₊, :])))
    A = zeros(n, n)
    A[:, i₊] = Ā
    colls2 = [nₑ₋ * (i-1) + k for i in 1:nₑ₋ for k in 1:i]
    𝐂₂ = sparse(colls2, 1:length(colls2), 1.0)
    𝐔₂ = 𝐂₂' * sparse([i <= k ? (k - 1) * nₑ₋ + i : (i - 1) * nₑ₋ + k for k in 1:nₑ₋ for i in 1:nₑ₋], 1:nₑ₋^2, 1.0)
    redu = sparsevec(nₑ₋ - nₑ + 1:nₑ₋, 1)
    redu_idxs = findnz(ℒ.kron(redu, redu))[1]
    𝛔 = sparse(redu_idxs[Int.(range(1, nₑ^2, nₑ))], fill(n₋ * (nₑ₋ + 1) + 1, nₑ), 1.0, nₑ₋^2, nₑ₋^2)
    B = Matrix(𝐔₂ * (ℒ.kron(s, s) + 𝛔) * 𝐂₂)
    return Ā, A, s, B
end

@testset "State-block Schur second-order solver" begin
    Random.seed!(7)
    for (n, i₊, n₋, nₑ) in [(8, [2, 4, 5, 7], 3, 2), (12, [1, 3, 6, 9, 10], 5, 3), (6, [2, 3], 1, 1)]
        Ā, A, s, B = schur_reference_problem(n, i₊, n₋, nₑ)
        nc = size(B, 1)
        C = randn(n, nc)
        Xref = reshape((ℒ.I - ℒ.kron(B', A)) \ vec(C), n, nc)
        X, ok, Z = MMschur.second_order_schur_solve(Ā, s, C, i₊, n₋, nₑ)
        @test ok
        @test isapprox(X, Xref, rtol = 1e-10)
        @test isapprox(Z, X[i₊, :] * B, rtol = 1e-10)

        G = randn(n, nc)
        Zref = reshape((ℒ.I - ℒ.kron(B, A')) \ vec(G), n, nc)
        ∂C, ok2 = MMschur.second_order_schur_adjoint(Ā, s, G, i₊, n₋, nₑ)
        @test ok2
        @test isapprox(∂C, Zref, rtol = 1e-10)
    end
end

@testset "Dense per-equation Hessian forcing" begin
    Random.seed!(8)
    N, q, n = 7, 5, 4
    np = N * (N + 1) ÷ 2
    ∇₂ = sparse([1, 1, 2, 3, 4, 4], [1, 5, 9, 12, 20, 28], randn(6), n, np)
    M = randn(N, q)
    @test isapprox(MMschur.second_order_hessian_forcing(∇₂, M), Matrix(∇₂ * MMschur.compressed_kron²(M)), rtol = 1e-12)
end

@testset "Schur and doubling give the same second-order solution" begin
    include("../models/RBC_baseline.jl")
    MMschur.clear_solution_caches!(RBC_baseline, :second_order)
    a = get_solution(RBC_baseline, algorithm = :second_order)
    MMschur.clear_solution_caches!(RBC_baseline, :second_order)
    b = get_solution(RBC_baseline, algorithm = :second_order, sylvester_algorithm = :doubling)
    @test isapprox(collect(a), collect(b), rtol = 1e-10)
end

@testset "State-block Schur third-order solver" begin
    Random.seed!(9)
    for (n, i₊, n₋, nₑ) in [(7, [2, 4, 5], 2, 2), (9, [1, 3, 6, 8], 3, 2), (6, [2, 3], 1, 1)]
        nₑ₋ = n₋ + 1 + nₑ
        Ā, A, s, _ = schur_reference_problem(n, i₊, n₋, nₑ)
        redu = sparsevec(nₑ₋ - nₑ + 1:nₑ₋, 1)
        redu_idxs = findnz(ℒ.kron(redu, redu))[1]
        𝛔 = sparse(redu_idxs[Int.(range(1, nₑ^2, nₑ))], fill(n₋ * (nₑ₋ + 1) + 1, nₑ), 1.0, nₑ₋^2, nₑ₋^2)
        B = Matrix(MMschur.compressed_permuted_mixed_kron(sparse(s), 𝛔) + MMschur.compressed_kron³(sparse(s)))
        nc = size(B, 1)
        C = randn(n, nc)
        Xref = reshape((ℒ.I - ℒ.kron(B', A)) \ vec(C), n, nc)
        X, ok, Z = MMschur.third_order_schur_solve(Ā, s, C, i₊, n₋, nₑ)
        @test ok
        @test isapprox(X, Xref, rtol = 1e-10)
        @test isapprox(Z, X[i₊, :] * B, rtol = 1e-10)

        G = randn(n, nc)
        Zref = reshape((ℒ.I - ℒ.kron(B, A')) \ vec(G), n, nc)
        ∂C, ok2 = MMschur.third_order_schur_adjoint(Ā, s, G, i₊, n₋, nₑ)
        @test ok2
        @test isapprox(∂C, Zref, rtol = 1e-10)
    end
end

@testset "Schur and doubling give the same third-order solution" begin
    include("../models/FS2000.jl")
    get_solution(FS2000)
    MMschur.clear_solution_caches!(FS2000, :third_order)
    a = get_solution(FS2000, algorithm = :third_order)
    MMschur.clear_solution_caches!(FS2000, :third_order)
    b = get_solution(FS2000, algorithm = :third_order, sylvester_algorithm = [:doubling, :doubling])
    @test isapprox(collect(a), collect(b), rtol = 1e-10)
end

@testset "Cold steady state and second order after a cache reset (GNSS_2010)" begin
    include("../models/GNSS_2010.jl")
    ss0 = collect(get_SS(GNSS_2010, derivatives = false))
    MMschur.clear_solution_caches!(GNSS_2010, :second_order)
    s2 = get_solution(GNSS_2010, algorithm = :second_order)
    @test size(s2, 2) == GNSS_2010.constants.post_model_macro.nVars
    @test isapprox(collect(get_SS(GNSS_2010, derivatives = false)), ss0, atol = 1e-10)
end

@testset "Standard Schur QME path matches the generalized Schur path" begin
    include("../models/Smets_Wouters_2007.jl")
    for m in (FS2000, Smets_Wouters_2007)
        SS, _ = MMschur.get_NSSS_and_parameters(m, m.parameter_values)
        ∇₁ = MMschur.calculate_jacobian(m.parameter_values, SS, m.caches, m.functions.jacobian, m.workspaces)
        sols = map((false, true)) do flag
            MMschur.QME_STANDARD_SCHUR[] = flag
            r = MMschur.calculate_first_order_solution(∇₁, m.constants, m.workspaces, m.caches;
                                                        opts = MMschur.merge_calculation_options(),
                                                        initial_guess = zeros(0, 0))
            @test r[3]
            copy(r[1])
        end
        MMschur.QME_STANDARD_SCHUR[] = true
        @test isapprox(sols[1], sols[2], rtol = 1e-10)
    end
end

@testset "Third-order moment dependencies keep states that enter with shocks" begin
    # z loads on σ₋ only through σ₋ ⊗ ε (second order) and σ₋ ⊗ σ₋ ⊗ ε (third order).
    @model sv_dependency_test begin
        z[0] = λ * z[-1] + σ[0] * ϵᶻ[x]
        log(σ[0]) = (1 - ρ) * log(σ̄) + ρ * log(σ[-1]) + η * ω[x]
    end
    @parameters sv_dependency_test silent = true begin
        λ = 0.95; ρ = 0.9; σ̄ = 0.1; η = 0.3
    end
    m = sv_dependency_test
    sd = get_std(m, algorithm = :pruned_third_order, derivatives = false)
    MMschur.invalidate_cache_validity!(m)
    Σ_full, = MMschur.calculate_third_order_moments(m.parameter_values, :full_covar, m)
    iz = findfirst(==(:z), m.constants.post_model_macro.var)
    @test isapprox(sd(:z), sqrt(Σ_full[iz, iz]), rtol = 1e-10)
    # a 2-million-period pruned simulation gives Var(z) ≈ 0.2161
    @test isapprox(sd(:z)^2, 0.2161, rtol = 0.02)
end

@testset "Structured pruned third-order covariance matches the generic Lyapunov solve" begin
    # Gaussian blocks in closed form, Lyapunov only on the other states
    include("../models/Backus_Kehoe_Kydland_1992.jl")
    for m in (FS2000, sv_dependency_test, Backus_Kehoe_Kydland_1992)
        get_std(m, algorithm = :pruned_third_order, derivatives = false)
        Σs = map((false, true)) do flag
            MMschur.PRUNED_THIRD_ORDER_STRUCTURED_COVARIANCE[] = flag
            MMschur.invalidate_cache_validity!(m)
            r = MMschur.calculate_third_order_moments(copy(m.parameter_values), Symbol.(get_variables(m)), m)
            @test r[4]
            copy(r[1])
        end
        MMschur.PRUNED_THIRD_ORDER_STRUCTURED_COVARIANCE[] = true
        MMschur.invalidate_cache_validity!(m)
        @test isapprox(Σs[1], Σs[2], rtol = 1e-9)
    end
end
