using Test
using MacroModelling
using AxisKeys: axiskeys

@testset verbose = true "Symbolic balanced growth path stationarization" begin
    @testset "pure additive trends are rejected" begin
        @test_throws ArgumentError begin
            @model AdditiveBGPRejected begin
                x[0] = x[-1] + g + e[x]
                dx[0] = x[0] - x[-1]
            end
            @parameters AdditiveBGPRejected begin
                g = 0.02
            end
        end
    end

    @testset "multiplicative trend public APIs" begin
        @model MultiplicativeBGP begin
            x[0] = x[-1] * g[0]
            g[0] = 1.02 + σg * eg[x]
            z[0] = 0.5 * z[-1] + σz * ez[x]
            y[0] = x[0] * (1 + z[0])
        end
        @parameters MultiplicativeBGP begin
            σg = 0.01
            σz = 0.01
        end

        model_variables = MultiplicativeBGP.constants.post_model_macro.var
        @test any(==(Symbol("xᴳ")), model_variables)
        @test all(name -> !endswith(string(name), "ᴳ"),
                  axiskeys(get_SS(MultiplicativeBGP, derivatives = false), 1))

        ss = get_SS(MultiplicativeBGP, derivatives = false)
        @test collect(axiskeys(ss, 2)) == [:Steady_state, :Growth_rate]
        @test isapprox(ss(:g, :Steady_state), 1.02; atol = 1e-10)
        @test isapprox(ss(:x, :Growth_rate), log(1.02); atol = 1e-10)
        @test isapprox(ss(:y, :Growth_rate), log(1.02); atol = 1e-10)
        @test isapprox(ss(:z, :Growth_rate), 0.0; atol = 1e-10)
        ss_with_derivatives = get_SS(MultiplicativeBGP)
        @test all(isfinite, collect(ss_with_derivatives))

        solution = get_solution(MultiplicativeBGP)
        @test all(isfinite, collect(solution))

        stationary_irf = get_irf(MultiplicativeBGP, shocks = :none,
                                 periods = 3, levels = false)
        level_irf = get_irf(MultiplicativeBGP, shocks = :none,
                            periods = 3, levels = true)
        @test all(iszero, collect(stationary_irf(:x, :, :)))
        @test isapprox(collect(level_irf(:x, :, :))[:],
                       [1.02, 1.0404, 1.061208]; atol = 1e-8)
        @test isapprox(collect(level_irf(:y, :, :))[:],
                       [1.02, 1.0404, 1.061208]; atol = 1e-8)
        stochastic_level_irf = get_irf(MultiplicativeBGP, periods = 3,
                                       levels = true)
        @test all(isfinite, collect(stochastic_level_irf))

        moments = get_moments(MultiplicativeBGP, mean = false,
                              variance = true, covariance = true,
                              derivatives = false)
        covariance = moments[:covariance]
        @test all(name -> !endswith(string(name), "ᴳ"),
                  axiskeys(covariance, 1))
        @test all(isfinite, collect(covariance))

        derivative_moments = get_moments(
            MultiplicativeBGP, mean = false, variance = true,
            covariance = true, derivatives = true,
            parameter_derivatives = :all)
        @test all(isfinite, collect(derivative_moments[:covariance]))

        for algorithm in (:pruned_second_order, :pruned_third_order)
            higher_order_moments = get_moments(
                MultiplicativeBGP, mean = true, variance = true,
                covariance = true, derivatives = false,
                algorithm = algorithm)
            @test all(isfinite, collect(higher_order_moments[:mean]))
            @test all(isfinite, collect(higher_order_moments[:covariance]))
            @test all(name -> !endswith(string(name), "ᴳ"),
                      axiskeys(higher_order_moments[:covariance], 1))
        end
    end

    @testset "forward expectations receive lead growth factors" begin
        @model MultiplicativeExpectations begin
            a[0] = a[-1] * g[0]
            g[0] = 1.02 + σg * eg[x]
            x[0] = 0.5 * x[-1] + β * x[1] + (0.5 - β) * a[0]
        end
        @parameters MultiplicativeExpectations begin
            σg = 0.01
            β = 0.2
        end

        metadata = MultiplicativeExpectations.equations.stationarization
        @test metadata !== nothing
        stationary_equation_strings = string.(metadata.stationary_equations)
        @test any(
            equation -> any(
                growth_variable -> occursin(string(growth_variable), equation) &&
                                   occursin("[1]", equation),
                metadata.growth_variables),
            stationary_equation_strings)
        @test all(isfinite,
                  collect(get_SS(MultiplicativeExpectations, derivatives = false)))
        @test all(isfinite, collect(get_solution(MultiplicativeExpectations)))
    end

    @testset "equation updates rebuild stationarization" begin
        @model RebuildBGP begin
            x[0] = x[-1] * g[0]
            g[0] = 1.02 + σg * eg[x]
            z[0] = 0.5 * z[-1] + σz * ez[x]
            y[0] = x[0] * (1 + z[0])
        end
        @parameters RebuildBGP begin
            σg = 0.01
            σz = 0.01
        end

        update_equations!(RebuildBGP, 2, :(g[0] = 1.03 + σg * eg[x]))
        ss = get_SS(RebuildBGP, derivatives = false)
        @test isapprox(ss(:g, :Steady_state), 1.03; atol = 1e-10)
        @test isapprox(ss(:x, :Growth_rate), log(1.03); atol = 1e-10)
    end

    @testset "stationary models remain unchanged" begin
        @model StationaryRegression begin
            1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + σ * e[x]
        end
        @parameters StationaryRegression begin
            σ = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        @test StationaryRegression.equations.stationarization === nothing
        @test all(isfinite,
                  collect(get_SS(StationaryRegression, derivatives = false)))
        @test all(isfinite, collect(get_solution(StationaryRegression)))
    end
end
