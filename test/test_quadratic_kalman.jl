using MacroModelling
using Test
import Random
import Statistics
import LinearAlgebra as ℒ
import ForwardDiff

# -----------------------------------------------------------------------------
# Quadratic Kalman filter (Monfort, Renne & Roussellet, 2015) on the pruned
# second-order solution.
#
# Three checks, in increasing strength:
#
#  1. On a *linear* model the second-order terms vanish, the augmented blocks go
#     inert, and the filter must reproduce the Kalman likelihood exactly. This
#     validates the plumbing — but note it exercises none of the quadratic
#     machinery, which is why it is the weakest of the three.
#  2. The augmented transition must reproduce the exact conditional mean of the
#     package's own pruned recursion, checked by Monte Carlo. This is what
#     validates the Kronecker algebra.
#  3. On a genuinely nonlinear model the particle filter is a near-exact
#     reference at the same measurement error, and the quadratic Kalman filter
#     must agree with it up to Monte-Carlo error.
# -----------------------------------------------------------------------------

@testset "Quadratic Kalman filter" begin

    @model RBC_qkf begin
        1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α * exp(g[0])
        z[0] = ρz * z[-1] + std_z * eps_z[x]
        g[0] = ρg * g[-1] + std_g * eps_g[x]
    end

    @parameters RBC_qkf begin
        std_z = 0.02
        std_g = 0.02
        ρz = 0.4
        ρg = 0.6
        δ = 0.02
        α = 0.5
        β = 0.95
    end

    obs = [:c, :q]
    Random.seed!(12345)
    data = simulate(RBC_qkf, periods = 60, algorithm = :pruned_second_order)(obs, :, :simulate)
    p = RBC_qkf.parameter_values

    opts = MacroModelling.merge_calculation_options()
    MacroModelling.solve!(RBC_qkf, algorithm = :pruned_second_order, dynamics = true, opts = opts)
    _, _, 𝐒, _, _ = MacroModelling.get_relevant_steady_state_and_state_update(
                        Val(:pruned_second_order), p, RBC_qkf, opts = opts)

    ssn = RBC_qkf.constants.post_complete_parameters.SS_and_pars_names
    obs_idx = convert(Vector{Int}, indexin(obs, ssn))
    NSSS = get_steady_state(RBC_qkf, derivatives = false)
    Y = collect(data) .- [NSSS(v) for v in obs]

    sys = MacroModelling.build_quadratic_kalman_system(RBC_qkf, 𝐒[1], 𝐒[2], obs_idx)
    # the Kronecker block is carried compressed (vech, not vec)
    @test sys.nq == sys.nPast * (sys.nPast + 1) ÷ 2
    @test sys.nz == 2 * sys.nr + sys.nq
    @test maximum(abs, sys.S2) > 1e-3    # the model really is nonlinear

    @testset "augmented transition reproduces the pruned conditional mean" begin
        Random.seed!(3)
        x1 = randn(sys.nr) * 0.02
        x2 = randn(sys.nr) * 0.002
        z  = vcat(x1, x2, sys.Lp * ℒ.kron(sys.P * x1, sys.P * x1))

        nmc = 200_000
        a1 = zeros(sys.nr); a2 = zeros(sys.nr); aq = zeros(sys.nPast^2)
        for _ in 1:nmc
            ε = randn(sys.nExo)
            # the retained-row system: aug₁ built from the same past states
            ā = sys.Ea * vcat(sys.P * x1, 1.0)
            aug1 = ā + sys.S * ε
            n1 = sys.S1 * aug1
            n2 = sys.S1 * (sys.Ea * vcat(sys.P * x2, 0.0)) + sys.S2 * ℒ.kron(aug1, aug1) / 2
            a1 .+= n1; a2 .+= n2
            aq .+= ℒ.kron(sys.P * n1, sys.P * n1)
        end
        a1 ./= nmc; a2 ./= nmc; aq ./= nmc

        pred = sys.𝒜 * z + sys.c
        rel(a, b) = maximum(abs, a - b) / max(1e-12, maximum(abs, b))
        tol = 20 / sqrt(nmc)          # generous multiple of the Monte-Carlo error
        @test rel(pred[1:sys.nr], a1) < tol
        @test rel(pred[sys.nr+1:2sys.nr], a2) < tol
        @test rel(pred[2sys.nr+1:end], sys.Lp * aq) < tol
    end

    @testset "matches the particle filter on a nonlinear model" begin
        # the particle filter is reliable here: two observables, so the weights do
        # not degenerate, and 60,000 particles put its Monte-Carlo error well below
        # the tolerance used
        mev = 1e-4
        qk = MacroModelling.run_quadratic_kalman(sys, Y; measurement_error = fill(mev, length(obs)))
        pf = [get_loglikelihood(RBC_qkf, data, p; algorithm = :pruned_second_order,
                                filter = :bootstrap_particle, measurement_error = mev,
                                n_particles = 60_000, particle_rng = Random.Xoshiro(50 + s))
              for s in 1:4]
        @test isfinite(qk)
        @test all(isfinite, pf)
        @test abs(qk - Statistics.mean(pf)) < 2.0

        # as the measurement error shrinks both approach the inversion filter's
        # zero-measurement-error limit from below
        inv_ll = get_loglikelihood(RBC_qkf, data, p; algorithm = :pruned_second_order,
                                   filter = :inversion)
        qk_tight = MacroModelling.run_quadratic_kalman(sys, Y;
                                                       measurement_error = fill(1e-5, length(obs)))
        @test isfinite(inv_ll)
        @test qk_tight > qk                    # less measurement error ⇒ higher density
        @test abs(qk_tight - inv_ll) < abs(qk - inv_ll)
    end

    @testset "public API, gating and derivatives" begin
        mev = 1e-4
        qk_api = get_loglikelihood(RBC_qkf, data, p; algorithm = :pruned_second_order,
                                   filter = :quadratic_kalman, measurement_error = mev)
        qk_int = MacroModelling.run_quadratic_kalman(sys, Y; measurement_error = fill(mev, length(obs)))
        @test isapprox(qk_api, qk_int, rtol = 1e-10)

        # the filter is defined only on the pruned second-order solution; asking for
        # it elsewhere falls back to the inversion filter rather than erroring
        @test get_loglikelihood(RBC_qkf, data, p; algorithm = :first_order,
                                filter = :quadratic_kalman) ==
              get_loglikelihood(RBC_qkf, data, p; algorithm = :first_order, filter = :inversion)

        # The implementation is type generic, so forward-mode AD flows through the
        # closed-form moment algebra — there is no finite differencing inside the
        # filter. Checked against central differences on the likelihood itself.
        f(x) = get_loglikelihood(RBC_qkf, data, x; algorithm = :pruned_second_order,
                                 filter = :quadratic_kalman, measurement_error = mev)
        g = ForwardDiff.gradient(f, p)
        @test all(isfinite, g)
        h = 1e-6
        fd = [(f(p + h * (1:length(p) .== i)) - f(p - h * (1:length(p) .== i))) / (2h)
              for i in eachindex(p)]
        @test maximum(abs.(g .- fd) ./ max.(abs.(fd), 1.0)) < 1e-5
    end

    @testset "hand-written reverse mode for the recursion" begin
        # Every cotangent of the taped recursion, against ForwardDiff. Random but
        # well-conditioned inputs; the point is the adjoint algebra, not a model.
        Random.seed!(7)
        nz, nE, nobs, nT, nPast_ = 9, 2, 2, 12, 3
        Pz = zeros(nPast_, nz); for i in 1:nPast_; Pz[i, i] = 1.0; end
        A0 = 0.3 * randn(nz, nz); A0 ./= (1.6 * maximum(abs, ℒ.eigvals(A0)))
        c0 = 0.01 * randn(nz); g0 = 0.05 * randn(nz * nE); L0 = 0.02 * randn(nz * nE, nPast_)
        Ch = zeros(nobs, nz); Ch[1,1] = 1.0; Ch[2,2] = 1.0; Ch[1,4] = 1.0; Ch[2,5] = 1.0
        QH0 = (M = 0.05 * randn(nz, nz); M * M')
        Hm0 = Matrix(0.01 * ℒ.I(nobs))
        Y0 = 0.05 * randn(nobs, nT); z00 = 0.01 * randn(nz)
        S00 = (M = 0.1 * randn(nz, nz); M * M')
        ps = 2
        f(A, c, QH, g, L, Hm, Y, z0, S0) =
            MacroModelling.quadratic_kalman_recursion(A, c, QH, g, L, Hm, Y, Ch, Pz, z0, S0,
                                                      nz, nE, ps, -Inf)
        ll, pb = MacroModelling.rrule(MacroModelling.quadratic_kalman_recursion,
                                      A0, c0, QH0, g0, L0, Hm0, Y0, Ch, Pz, z00, S00,
                                      nz, nE, ps, -Inf)
        ct = pb(1.0)
        @test isfinite(ll)
        rel(a, b) = maximum(abs, a .- b) / max(1e-10, maximum(abs, b))
        # ct = (NoTangent, 𝒜̄, c̄, Q̄H, ḡ0, Λ̄, H̄m, Ȳ, NoTangent, NoTangent, z̄0, Σ̄0, …)
        @test rel(ct[2], ForwardDiff.gradient(x -> f(x, c0, QH0, g0, L0, Hm0, Y0, z00, S00), A0)) < 1e-10
        @test rel(ct[3], ForwardDiff.gradient(x -> f(A0, x, QH0, g0, L0, Hm0, Y0, z00, S00), c0)) < 1e-10
        @test rel(ct[4], ForwardDiff.gradient(x -> f(A0, c0, x, g0, L0, Hm0, Y0, z00, S00), QH0)) < 1e-10
        @test rel(ct[5], ForwardDiff.gradient(x -> f(A0, c0, QH0, x, L0, Hm0, Y0, z00, S00), g0)) < 1e-10
        @test rel(ct[6], ForwardDiff.gradient(x -> f(A0, c0, QH0, g0, x, Hm0, Y0, z00, S00), L0)) < 1e-10
        @test rel(ct[7], ForwardDiff.gradient(x -> f(A0, c0, QH0, g0, L0, x, Y0, z00, S00), Hm0)) < 1e-10
        @test rel(ct[8], ForwardDiff.gradient(x -> f(A0, c0, QH0, g0, L0, Hm0, x, z00, S00), Y0)) < 1e-10
        @test rel(ct[11], ForwardDiff.gradient(x -> f(A0, c0, QH0, g0, L0, Hm0, Y0, x, S00), z00)) < 1e-10
        @test rel(ct[12], ForwardDiff.gradient(x -> f(A0, c0, QH0, g0, L0, Hm0, Y0, z00, x), S00)) < 1e-10
    end

    @testset "reduces to the Kalman filter on a linear model" begin
        # With 𝐒₂ = 0 the x₂ and Kronecker blocks are inert and the quadratic
        # Kalman filter is the Kalman filter. Exact agreement, not approximate.
        @model LIN_qkf begin
            zs[0] = rho_l * zs[-1] + sig_l * e1[x]
            ys[0] = zs[0] + 0 * ys[1]
        end
        @parameters LIN_qkf begin
            rho_l = 0.5
            sig_l = 0.01
        end

        Random.seed!(4242)
        dlin = simulate(LIN_qkf, periods = 80)([:ys], :, :simulate)
        plin = LIN_qkf.parameter_values
        optsl = MacroModelling.merge_calculation_options()
        MacroModelling.solve!(LIN_qkf, algorithm = :pruned_second_order, dynamics = true, opts = optsl)
        _, _, 𝐒l, _, _ = MacroModelling.get_relevant_steady_state_and_state_update(
                            Val(:pruned_second_order), plin, LIN_qkf, opts = optsl)
        @test maximum(abs, Matrix(𝐒l[2])) == 0.0     # premise: the model is linear

        ssnl = LIN_qkf.constants.post_complete_parameters.SS_and_pars_names
        oil  = convert(Vector{Int}, indexin([:ys], ssnl))
        NSSSl = get_steady_state(LIN_qkf, derivatives = false)
        Yl = collect(dlin) .- [NSSSl(:ys)]

        sysl = MacroModelling.build_quadratic_kalman_system(LIN_qkf, 𝐒l[1], 𝐒l[2], oil)
        mev = 1e-4
        qkl = MacroModelling.run_quadratic_kalman(sysl, Yl; measurement_error = [mev])
        kal = get_loglikelihood(LIN_qkf, dlin, plin; filter = :kalman, measurement_error = mev)
        @test isapprox(qkl, kal, rtol = 1e-9)
    end
end
