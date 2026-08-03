using MacroModelling
using Test
using Random
using ForwardDiff
using FiniteDifferences
using Zygote
import LinearAlgebra as ℒ
import AxisKeys: KeyedArray

# The cubic Kalman filter rests on one property: the pruned third-order recursion
# is exactly affine in the augmented state z = [x₁; x₂; x₃; a⊗a; a⊗b; a⊗a⊗a].
# These tests check that property directly, then the moments built on top of it,
# then the likelihood against a converged particle filter.

@testset "Cubic Kalman filter" begin
    opts = MacroModelling.merge_calculation_options()

    @model RBC_ckf begin
        1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α * exp(g[0])
        z[0] = ρz * z[-1] + std_z * eps_z[x]
        g[0] = ρg * g[-1] + std_g * eps_g[x]
    end
    @parameters RBC_ckf begin
        std_z = 0.02
        std_g = 0.02
        ρz = 0.4
        ρg = 0.6
        δ = 0.02
        α = 0.5
        β = 0.95
    end

    MacroModelling.solve!(RBC_ckf, algorithm = :pruned_third_order, dynamics = true, opts = opts)
    pars = RBC_ckf.parameter_values
    _, _, 𝐒, _, _ = MacroModelling.get_relevant_steady_state_and_state_update(Val(:pruned_third_order), pars, RBC_ckf, opts = opts)
    ssn = RBC_ckf.constants.post_complete_parameters.SS_and_pars_names
    obs = [:c, :q]
    obs_idx = convert(Vector{Int}, indexin(obs, ssn))

    sys = MacroModelling.build_cubic_kalman_system_from_constants(RBC_ckf.constants, 𝐒[1], 𝐒[2], 𝐒[3], obs_idx)
    nP = sys.nPast
    # q₁₁ and q₁₁₁ are carried compressed (symmetric); q₁₂ = a⊗b is not.
    @test sys.nz == 3sys.nr + nP * (nP + 1) ÷ 2 + nP^2 + nP * (nP + 1) * (nP + 2) ÷ 6

    # the compression maps must round-trip a genuine symmetric Kronecker product
    let a = randn(nP)
        q11 = ℒ.kron(a, a)
        q111 = ℒ.kron(ℒ.kron(a, a), a)
        @test q11[sys.can2][sys.exp2] ≈ q11
        @test q111[sys.can3][sys.exp3] ≈ q111
        @test length(sys.can2) == nP * (nP + 1) ÷ 2
        @test length(sys.can3) == nP * (nP + 1) * (nP + 2) ÷ 6
    end

    Random.seed!(3)
    ε = randn(sys.nExo)

    # 1. the step is affine in z — the property the whole filter depends on
    z1 = randn(sys.nz)
    z2 = randn(sys.nz)
    λ = 0.41
    lhs = MacroModelling.cubic_kalman_step(sys, λ .* z1 .+ (1 - λ) .* z2, ε)
    rhs = λ .* MacroModelling.cubic_kalman_step(sys, z1, ε) .+ (1 - λ) .* MacroModelling.cubic_kalman_step(sys, z2, ε)
    @test maximum(abs, lhs - rhs) < 1e-12

    # 2. on a consistent state it reproduces the pruned third-order recursion,
    #    including the Kronecker blocks, with every product recomputed directly
    # The reference forms every Kronecker product directly and only then
    # compresses, so it exercises the compressed algebra rather than assuming it.
    consistent(x1, x2, x3) = (a = sys.Pm * x1; b = sys.Pm * x2;
                              vcat(x1, x2, x3, ℒ.kron(a, a)[sys.can2], ℒ.kron(a, b),
                                   ℒ.kron(ℒ.kron(a, a), a)[sys.can3]))
    x1 = 0.01 .* randn(sys.nr)
    x2 = 0.005 .* randn(sys.nr)
    x3 = 0.002 .* randn(sys.nr)
    a = sys.Pm * x1; b = sys.Pm * x2; p = sys.Pm * x3
    aug1 = vcat(a, 1.0, ε); aug1h = vcat(a, 0.0, ε)
    aug2 = vcat(b, 0.0, zeros(sys.nExo)); aug3 = vcat(p, 0.0, zeros(sys.nExo))
    reference = consistent(sys.S1 * aug1,
                           sys.S1 * aug2 + sys.S2 * MacroModelling.compressed_kron²_power(aug1) / 2,
                           sys.S1 * aug3 + sys.S2 * MacroModelling.compressed_kron²(aug1h, aug2) +
                           sys.S3 * MacroModelling.compressed_kron³_power(aug1) / 6)
    stepped = MacroModelling.cubic_kalman_step(sys, consistent(x1, x2, x3), ε)
    @test maximum(abs, stepped - reference) < 1e-12
    @test maximum(abs, stepped[sys.i11] - reference[sys.i11]) < 1e-12
    @test maximum(abs, stepped[sys.i12] - reference[sys.i12]) < 1e-12
    @test maximum(abs, stepped[sys.i111] - reference[sys.i111]) < 1e-12

    # 3. the recovered transition reproduces the conditional mean exactly, and
    #    both quadrature moments agree with Monte Carlo
    nodes, wts = MacroModelling.gauss_hermite_tensor(sys.nExo, 4)
    𝒜, c = MacroModelling.build_cubic_kalman_transition(sys, nodes, wts)
    zt = 0.01 .* randn(sys.nz)
    mq, Sq = MacroModelling.cubic_kalman_moments(sys, zt, nodes, wts)
    @test maximum(abs, mq - (𝒜 * zt + c)) < 1e-12

    # 3b. the analytic assembly must reproduce the quadrature exactly — it is a
    #     closed form for the same integrals, not an approximation of them.
    basis = MacroModelling.cubic_noise_basis(sys.nExo)
    @test basis.N == binomial(sys.nExo + 3, 3)
    # the monomial moment vector and covariance against tensor Gauss-Hermite
    @test all(abs(basis.m[i] - sum(w * prod(ε .^ basis.exps[i]) for (ε, w) in zip(nodes, wts))) < 1e-10
              for i in 1:basis.N)
    𝒜a, ca, c₀, Λ = MacroModelling.build_cubic_kalman_system(sys, basis)
    @test maximum(abs, 𝒜a - 𝒜) < 1e-9
    @test maximum(abs, ca - c) < 1e-9
    @testset "conditional innovation covariance" begin
        Λnoise = Matrix(Λ[:, sys.noise_state_indices])
        Rstate = randn(sys.nz, sys.nz); Pc = Rstate * Rstate'
        Ctest = randn(sys.nz, basis.N)
        Pnoise = zeros(length(sys.noise_state_indices), length(sys.noise_state_indices))
        mixvec = zeros(sys.nz * basis.N); mixΨ = zeros(sys.nz, basis.N)
        CΨ = zeros(sys.nz, basis.N); Q = zeros(sys.nz, sys.nz)
        MacroModelling.cubic_kalman_noise_covariance!(Q, Ctest, Λnoise, basis.Ψ, Pc,
                                                       sys.noise_state_indices, Pnoise,
                                                       mixvec, mixΨ, CΨ)
        expected = Ctest * basis.Ψ * Ctest'
        Pload = Pc[sys.noise_state_indices, sys.noise_state_indices]
        for i in eachindex(sys.noise_state_indices), j in eachindex(sys.noise_state_indices)
            Di = reshape(view(Λnoise, :, i), sys.nz, basis.N)
            Dj = reshape(view(Λnoise, :, j), sys.nz, basis.N)
            expected .+= Pload[i, j] .* (Di * basis.Ψ * Dj')
        end
        @test Q ≈ (expected + expected') / 2
        Pc_outside = copy(Pc)
        outside = setdiff(1:sys.nz, sys.noise_state_indices)
        Pc_outside[outside, outside] .+= 10
        Qoutside = similar(Q)
        MacroModelling.cubic_kalman_noise_covariance!(Qoutside, Ctest, Λnoise, basis.Ψ,
                                                       Pc_outside, sys.noise_state_indices,
                                                       Pnoise, mixvec, mixΨ, CΨ)
        @test Qoutside ≈ Q
    end
    # Q(z) = C(z) Ψ C(z)' against the quadrature variance, at a non-trivial z
    Ca = reshape(c₀ + Λ * zt, sys.nz, basis.N)
    Qa = Ca * basis.Ψ * Ca'
    @test maximum(abs, Qa - Sq) / max(1e-12, maximum(abs, Sq)) < 1e-8

    Random.seed!(5)
    N = 200_000
    mm = zeros(sys.nz); SS = zeros(sys.nz, sys.nz)
    for _ in 1:N
        fz = MacroModelling.cubic_kalman_step(sys, zt, randn(sys.nExo))
        mm .+= fz
        SS .+= fz * fz'
    end
    mm ./= N; SS ./= N; SS .-= mm * mm'
    @test maximum(abs, mq - mm) / max(1e-12, maximum(abs, mq)) < 0.02
    @test maximum(abs, Sq - SS) / max(1e-12, maximum(abs, Sq)) < 0.05

    # 4. the likelihood matches a converged bootstrap particle filter
    Random.seed!(101)
    T = 60
    sim = get_irf(RBC_ckf, algorithm = :pruned_third_order, periods = T, shocks = :simulate, levels = false)
    Y = Matrix(sim(obs, :, :simulate))
    sd_obs = [sqrt(sum(abs2, Y[i, :] .- sum(Y[i, :]) / T) / (T - 1)) for i in eachindex(obs)]
    mev = (0.2 .* sd_obs) .^ 2
    NSSS = get_steady_state(RBC_ckf, derivatives = false)
    data = KeyedArray(Y .+ [NSSS(v) for v in obs]; Variable = obs, Time = 1:T)

    ll_ckf = get_loglikelihood(RBC_ckf, data, pars; algorithm = :pruned_third_order,
                               filter = :cubic_kalman, measurement_error = mev)
    @test isfinite(ll_ckf)

    ll_pf = [get_loglikelihood(RBC_ckf, data, pars; algorithm = :pruned_third_order,
                               filter = :bootstrap_particle, measurement_error = mev,
                               n_particles = 80_000, particle_rng = Random.Xoshiro(s)) for s in 1:4]
    m = sum(ll_pf) / length(ll_pf)
    # The particle filter's log-likelihood is downward-biased by about Var/2.
    @test abs(ll_ckf - (m + (sum(x -> (x - m)^2, ll_pf) / (length(ll_pf) - 1)) / 2)) < 0.05 * T

    # 4b. gradients. Both modes are checked against central differences, and
    #     reverse mode is checked on every measurement-error shape — a wrong `H`
    #     reaching the adjoint but not the primal gives a finite, plausible, wrong
    #     gradient rather than an error.
    f = p -> get_loglikelihood(RBC_ckf, data, p; algorithm = :pruned_third_order,
                               filter = :cubic_kalman, measurement_error = mev)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), f, pars)[1]
    @test !all(iszero, g_fd)
    @test maximum(abs.(ForwardDiff.gradient(f, pars) .- g_fd) ./ max.(1.0, abs.(g_fd))) < 1e-7
    @test maximum(abs.(Zygote.gradient(f, pars)[1] .- g_fd) ./ max.(1.0, abs.(g_fd))) < 1e-7

    f_nome = p -> get_loglikelihood(RBC_ckf, data, p; algorithm = :pruned_third_order,
                                    filter = :cubic_kalman)
    g_fd_nome = FiniteDifferences.grad(central_fdm(5, 1), f_nome, pars)[1]
    @test maximum(abs.(Zygote.gradient(f_nome, pars)[1] .- g_fd_nome) ./ max.(1.0, abs.(g_fd_nome))) < 1e-6

    mev_mat = [3e-5 1e-5; 1e-5 4e-5]     # non-diagonal covariance
    f_mat = p -> get_loglikelihood(RBC_ckf, data, p; algorithm = :pruned_third_order,
                                   filter = :cubic_kalman, measurement_error = mev_mat)
    g_fd_mat = FiniteDifferences.grad(central_fdm(5, 1), f_mat, pars)[1]
    @test maximum(abs.(Zygote.gradient(f_mat, pars)[1] .- g_fd_mat) ./ max.(1.0, abs.(g_fd_mat))) < 1e-6

    # 4c. the two hand-written adjoints the chain rests on, checked in isolation
    #     against ForwardDiff so a regression localises instead of just moving the
    #     end-to-end number.
    let ws = MacroModelling.cubic_kalman_workspace(sys), Pm = sys.Pm,
        nP = sys.nPast, nE = sys.nExo, na = sys.na,
        n1 = length(sys.S1), n2 = length(sys.S2), n3 = length(sys.S3)
        function rebuild(θ)
            S1 = reshape(θ[1:n1], size(sys.S1))
            S2 = reshape(θ[n1+1:n1+n2], size(sys.S2))
            S3 = reshape(θ[n1+n2+1:end], size(sys.S3))
            M, mc, V, B2, Wq, Wl_t, Bc, M2, M3 = MacroModelling.cubic_derived_matrices(S1, S2, Pm, nP, nE, na)
            # the live-column slices are views of S2/S3 and must follow them
            merge(sys, (; S1, S2, S3, M, mc, V, B2, Wq, Wl_t, Bc, M2, M3,
                        S2k2 = S2[:, sys.k2cols], S2k12 = S2[:, sys.k12cols],
                        S3k3 = S3[:, sys.k3cols]))
        end
        θ0 = vcat(vec(sys.S1), vec(sys.S2), vec(sys.S3))
        function fold(∂)
            MacroModelling.cubic_derived_pullback!(∂, sys)
            return vcat(vec(∂.S1), vec(∂.S2), vec(∂.S3))
        end

        # the step's adjoint
        zr = 0.05 .* randn(sys.nz); εr = randn(nE); ∂out = randn(sys.nz)
        function step_scalar(θ)
            s2 = rebuild(θ)
            w2 = MacroModelling.cubic_kalman_workspace(s2, eltype(θ))
            out = MacroModelling.cubic_kalman_step!(Vector{eltype(θ)}(undef, sys.nz), s2, zr, εr, w2)
            return ℒ.dot(∂out, out)
        end
        ∂s = MacroModelling.cubic_kalman_cotangents(sys)
        MacroModelling.cubic_kalman_step_pullback!(∂s, sys, zr, εr, ∂out, ws)
        gs = ForwardDiff.gradient(step_scalar, θ0)
        @test maximum(abs, fold(∂s) - gs) / max(1e-12, maximum(abs, gs)) < 1e-10

        # The build's adjoint. Unlike the step's, it folds the derived-block
        # cotangents onto S1/S2 itself, so no `fold` here.
        w𝒜 = randn(sys.nz, sys.nz); wc = randn(sys.nz)
        wc0 = randn(sys.nz * basis.N); wΛ = randn(sys.nz * basis.N, sys.nz)
        function build_scalar(θ)
            s2 = rebuild(θ)
            w2 = MacroModelling.cubic_kalman_workspace(s2, eltype(θ))
            A, cc, c0, L = MacroModelling.build_cubic_kalman_system(s2, basis; ws = w2)
            return ℒ.dot(w𝒜, A) + ℒ.dot(wc, cc) + ℒ.dot(wc0, c0) + ℒ.dot(wΛ, L)
        end
        ∂b = MacroModelling.cubic_kalman_cotangents(sys)
        MacroModelling.build_cubic_kalman_system_pullback!(∂b, sys, basis, w𝒜, wc, wc0, wΛ; ws = ws)
        gb = ForwardDiff.gradient(build_scalar, θ0)
        gb_mine = vcat(vec(∂b.S1), vec(∂b.S2), vec(∂b.S3))
        @test maximum(abs, gb_mine - gb) / max(1e-12, maximum(abs, gb)) < 1e-10
    end

    # 5. gating: the filter is only defined on the pruned third-order solution.
    #    At any other order it falls back to the inversion filter, which admits
    #    no measurement error — so none is passed here.
    ll_wrong = get_loglikelihood(RBC_ckf, data, pars; algorithm = :pruned_second_order,
                                 filter = :cubic_kalman)
    @test isfinite(ll_wrong)
end
