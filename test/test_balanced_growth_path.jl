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
        @test isapprox(ss(:dx), 0.02; atol = 1e-10)
        @test isapprox(ss(:x), 0.0; atol = 1e-10)

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
        @test isapprox(M[findfirst(==(:dx), rows), 2], 1.0; atol = 1e-6)
        @test isapprox(M[findfirst(==(:x),  rows), 2], 0.0; atol = 1e-6)

        # levels output of a trending variable carries the deterministic BGP drift
        # x_t = anchor + xᴳ·t (here xᴳ = g); the stationary dx stays at g; and
        # deviations (levels = false) carry no drift.
        lir = get_irf(RWdrift, shocks = :none, periods = 4, levels = true)
        @test isapprox(collect(lir(:x,  :, :))[:], [0.02, 0.04, 0.06, 0.08]; atol = 1e-8)
        @test isapprox(collect(lir(:dx, :, :))[:], fill(0.02, 4); atol = 1e-8)
        dir = get_irf(RWdrift, shocks = :none, periods = 4, levels = false)
        @test all(iszero, collect(dir(:x, :, :)))
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
        @test isapprox(ss(:c), 0.0; atol = 1e-10)            # stationary level
        @test isapprox(ss(:db), 0.03; atol = 1e-10)          # bᴳ = aᴳ = ga (cointegration growth identity)
        # cointegration level identity holds at the (anchored) particular solution
        @test isapprox(ss(:b), ss(:a) + ss(:c); atol = 1e-10)
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

end
