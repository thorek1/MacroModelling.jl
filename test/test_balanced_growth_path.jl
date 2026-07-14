using Test
using MacroModelling
using AxisKeys: axiskeys

# Balanced growth path (BGP) support: models written in levels with non-stationary
# I(1) variables on a common growth path. Steady state uses the IRIS-style
# two-time-point augmentation (level + growth unknowns), auto-detected so that
# fully stationary models are unaffected.

@testset verbose = true "Balanced growth path" begin

    @testset "random walk with drift (single trend)" begin
        @model RWdrift begin
            x[0]  = x[-1] + g + e[x]
            dx[0] = x[0] - x[-1]
        end
        @parameters RWdrift begin
            g = 0.02
        end

        # growth was auto-detected (growth symbols added as SS unknowns)
        @test any(s -> endswith(string(s), "ᴳ"), RWdrift.constants.post_model_macro.vars_in_ss_equations)

        ss = get_SS(RWdrift, derivatives = false)
        # dx (the stationary first difference) pins the drift; x is a free unit root → 0
        @test isapprox(ss(:dx, :Steady_state), 0.02; atol = 1e-10)
        @test isapprox(ss(:x, :Steady_state), 0.0; atol = 1e-10)
        @test collect(axiskeys(ss, 2)) == [:Steady_state, :Growth_rate]
        @test isapprox(ss(:x, :Growth_rate), 0.02; atol = 1e-10)
        @test isapprox(ss(:dx, :Growth_rate), 0.0; atol = 1e-10)

        # first-order solution and IRFs exist (one unit root) and are finite
        sol = get_solution(RWdrift)
        @test all(isfinite, collect(sol))
        irf = get_irf(RWdrift)
        @test all(isfinite, collect(irf))

        # ∂SS/∂param must propagate through the growth unknowns (dx = g ⇒ ∂dx/∂g = 1;
        # x is a free unit root ⇒ ∂x/∂g = 0). The augmented SS Jacobian is singular,
        # so this exercises the minimum-norm derivative path.
        ssd  = get_SS(RWdrift)   # derivatives = true
        M    = collect(ssd)
        rows = axiskeys(ssd, 1)
        @test isapprox(M[findfirst(==(:dx), rows), 3], 1.0; atol = 1e-6)
        @test isapprox(M[findfirst(==(:x),  rows), 3], 0.0; atol = 1e-6)

        # levels output of a trending variable carries the deterministic BGP drift
        # x_t = anchor + xᴳ·t (here xᴳ = g); the stationary dx stays at g; and
        # deviations (levels = false) carry no drift.
        lir = get_irf(RWdrift, shocks = :none, periods = 4, levels = true)
        @test isapprox(collect(lir(:x,  :, :))[:], [0.02, 0.04, 0.06, 0.08]; atol = 1e-8)
        @test isapprox(collect(lir(:dx, :, :))[:], fill(0.02, 4); atol = 1e-8)
        dir = get_irf(RWdrift, shocks = :none, periods = 4, levels = false)
        @test all(iszero, collect(dir(:x, :, :)))

        moments = get_moments(RWdrift, mean = false, variance = true,
                              standard_deviation = true, covariance = true,
                              correlation = true, derivatives = false)
        expected_moment_names = [:dx, :Delta_x]
        for moment_name in (:variance, :standard_deviation, :covariance, :correlation)
            result = moments[moment_name]
            @test collect(axiskeys(result, 1)) == expected_moment_names
            @test all(isfinite, collect(result))
        end

        for algorithm in (:pruned_second_order, :pruned_third_order)
            higher_order_moments = get_moments(RWdrift, mean = false,
                                                variance = true, covariance = true,
                                                algorithm = algorithm, derivatives = false)
            @test collect(axiskeys(higher_order_moments[:covariance], 1)) == expected_moment_names
            @test all(isfinite, collect(higher_order_moments[:covariance]))
        end

        statistics = get_statistics(RWdrift, RWdrift.parameter_values,
                                    variance = [:dx, :x], standard_deviation = [:dx, :x],
                                    covariance = [:dx, :x], correlation = [:dx, :x])
        for statistic_name in (:variance, :standard_deviation, :covariance, :correlation)
            @test all(isfinite, collect(statistics[statistic_name]))
        end
    end

    @testset "cointegration (two trends, shared growth path)" begin
        # a is an I(1) trend; c is stationary; b = a + c is cointegrated with a,
        # so b inherits a's growth. db = b - b[-1] must equal that growth in SS.
        @model BGPcoint begin
            a[0]  = a[-1] + ga + ea[x]
            c[0]  = 0.5 * c[-1] + ec[x]
            b[0]  = a[0] + c[0]
            db[0] = b[0] - b[-1]
        end
        @parameters BGPcoint begin
            ga = 0.03
        end

        ss = get_SS(BGPcoint, derivatives = false)
        @test isapprox(ss(:c, :Steady_state), 0.0; atol = 1e-10)            # stationary level
        @test isapprox(ss(:db, :Steady_state), 0.03; atol = 1e-10)          # bᴳ = aᴳ = ga (cointegration growth identity)
        # cointegration level identity holds at the (anchored) particular solution
        @test isapprox(ss(:b, :Steady_state), ss(:a, :Steady_state) + ss(:c, :Steady_state); atol = 1e-10)
    end

    @testset "steady-state level anchor (x[ss] = value)" begin
        # `x[ss] = xbar` pins the trend's level to the parameter; the growth is still
        # pinned by the dynamic law, and the dynamics are invariant to the anchor.
        @model RWanchor begin
            x[0]  = x[-1] + g + e[x]
            dx[0] = x[0] - x[-1]
            x[ss] = xbar
        end
        @parameters RWanchor begin
            g    = 0.02
            xbar = 5.0
        end

        @test RWanchor.equations.ss_anchors == Dict(:x => :xbar)

        ss = get_SS(RWanchor, derivatives = false)
        @test isapprox(ss(:x, :Steady_state), 5.0; atol = 1e-8)    # level pinned to the anchor
        @test isapprox(ss(:dx, :Steady_state), 0.02; atol = 1e-8)  # growth still pinned by the law

        # levels drift from the anchored level
        lir = get_irf(RWanchor, shocks = :none, periods = 3, levels = true)
        @test isapprox(collect(lir(:x, :, :))[:], [5.02, 5.04, 5.06]; atol = 1e-8)
    end

    @testset "stationary model is unaffected" begin
        @model RBCbgp begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + σ * e[x]
        end
        @parameters RBCbgp begin
            σ = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95
        end

        # no growth symbols introduced for a stationary model
        @test !any(s -> endswith(string(s), "ᴳ"), RBCbgp.constants.post_model_macro.vars_in_ss_equations)

        ss = get_SS(RBCbgp, derivatives = false)
        @test all(isfinite, collect(ss))
        sol = get_solution(RBCbgp)
        @test all(isfinite, collect(sol))
    end

    @testset "expectations model solves at higher orders" begin
        @model RBCexpect begin
            c[0] ^ (-σ) = β * c[1] ^ (-σ) * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)
            ψ * c[0] ^ σ / (1 - l[0]) = w[0]
            k[0] = (1 - δ) * k[-1] + i[0]
            y[0] = c[0] + i[0] + g[0]
            y[0] = z[0] * k[-1] ^ α * l[0] ^ (1 - α)
            w[0] = y[0] * (1 - α) / l[0]
            r[0] = y[0] * α * 4 / k[-1]
            z[0] = (1 - ρᶻ) + ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
            g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]
        end

        @parameters RBCexpect begin
            σᶻ = 0.066
            σᵍ = 0.104
            σ = 1
            α = 1 / 3
            i_y = 0.25
            k_y = 10.4
            ρᶻ = 0.97
            ρᵍ = 1.01
            g_y = 0.2038
            ḡ | ḡ = g_y * y[ss]
            δ = i_y / k_y
            β = 1 / (α / k_y + (1 - δ))
            ψ | l[ss] = 1 / 3
        end

        ss = get_SS(RBCexpect)
        @test collect(axiskeys(ss, 2)) == [:Steady_state, :Growth_rate]
        @test all(isfinite, collect(ss))
        @test isapprox(ss(:g, :Growth_rate), 0.0; atol = 1e-10)

        # With levels disabled the IRF is measured against the BGP path rather
        # than the fixed steady-state level.
        irf = get_irf(RBCexpect, shocks = :none, periods = 3)
        level_irf = get_irf(RBCexpect, shocks = :none, periods = 3, levels = true)
        bgp = ss(:g, :Steady_state) .+ ss(:g, :Growth_rate) .* (1:3)
        @test isapprox(collect(irf(:g, :, :)),
                       collect(level_irf(:g, :, :)) .- bgp;
                       atol = 1e-10)

        for algorithm in (:first_order, :pruned_second_order, :pruned_third_order)
            sol = get_solution(RBCexpect, algorithm = algorithm)
            @test all(isfinite, collect(sol))
        end
    end

end
