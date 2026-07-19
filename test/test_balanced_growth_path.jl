using Test
using MacroModelling
using AxisKeys: axiskeys

@testset "steady-state expression parser edge cases" begin
    @test MacroModelling.additive_terms(:(1 - x[ss])) ==
          Tuple{Float64, Any}[(1.0, 1), (-1.0, :(x[ss]))]
    @test MacroModelling.simplify(:(1 / 3)) isa Real
end

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

        @testset "mixed additive and multiplicative trends are rejected" begin
            @test_throws ArgumentError begin
                @model MixedAdditiveBGPRejected begin
                    x[0] = x[-1] + g + ex[x]
                    y[0] = y[-1] * μ
                end
                @parameters MixedAdditiveBGPRejected begin
                    g = 0.02
                    μ = 1.02
                end
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
        solution = get_solution(MultiplicativeExpectations)
        @test all(isfinite, collect(solution))
        for algorithm in (:pruned_second_order, :pruned_third_order)
            higher_order_solution = get_solution(
                MultiplicativeExpectations;
                algorithm = algorithm,
                silent = true,
            )
            @test all(isfinite, collect(higher_order_solution))
        end

        expectation_parameters = copy(MultiplicativeExpectations.parameter_values)
        expectation_SS_and_pars = MacroModelling.get_NSSS_and_parameters(
            MultiplicativeExpectations,
            expectation_parameters,
        )[1]
        expectation_internal_SS_and_pars = MacroModelling.internal_steady_state_and_parameters(
            expectation_SS_and_pars,
            MultiplicativeExpectations,
        )
        expectation_stationary_hessian = MacroModelling.calculate_hessian(
            expectation_parameters,
            expectation_internal_SS_and_pars,
            MultiplicativeExpectations.caches,
            MultiplicativeExpectations.functions.hessian,
            MultiplicativeExpectations.workspaces;
            caching = false,
        )
        expectation_direct_hessian = MacroModelling.calculate_bgp_hessian(
            MultiplicativeExpectations,
            expectation_parameters,
            expectation_SS_and_pars;
            caching = false,
        )
        @test isapprox(Matrix(expectation_direct_hessian),
                       Matrix(expectation_stationary_hessian); atol = 1e-10)

        expectation_stationary_third = MacroModelling.calculate_third_order_derivatives(
            expectation_parameters,
            expectation_internal_SS_and_pars,
            MultiplicativeExpectations.caches,
            MultiplicativeExpectations.functions.third_order_derivatives,
            MultiplicativeExpectations.workspaces;
            caching = false,
        )
        expectation_direct_third = MacroModelling.calculate_bgp_third_order_derivatives(
            MultiplicativeExpectations,
            expectation_parameters,
            expectation_SS_and_pars;
            caching = false,
        )
        @test isapprox(Matrix(expectation_direct_third),
                       Matrix(expectation_stationary_third); atol = 1e-10)

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

    @testset "multiple independent trend drivers" begin
        @model MultipleTrendDriversRegression begin
            a[0] = a[-1] * ga[0]
            n[0] = n[-1] * gn[0]
            ga[0] = 1.02 + σa * ea[x]
            gn[0] = 1.01 + σn * en[x]
            z[0] = 0.8 * z[-1] + σz * ez[x]
            y[0] = a[0] * n[0]^α * (1 + z[0])
            c[0] = y[0] * (1 - 0.2 * z[0])
        end
        @parameters MultipleTrendDriversRegression begin
            α = 0.6
            σa = 0.005
            σn = 0.005
            σz = 0.01
        end

        metadata = MultipleTrendDriversRegression.equations.stationarization
        metadata_id = objectid(metadata)
        profile = MultipleTrendDriversRegression.equations.bgp_detection
        @test metadata !== nothing
        @test profile.candidate_drivers == [:a, :n]
        @test metadata.trend_drivers == [:a, :n]
        @test Set(metadata.growth_variables) == Set([Symbol("aᴳ"), Symbol("nᴳ")])

        ss = get_SS(MultipleTrendDriversRegression, derivatives = false)
        @test isapprox(ss(:a, :Growth_rate), log(1.02); atol = 1e-10)
        @test isapprox(ss(:n, :Growth_rate), log(1.01); atol = 1e-10)
        @test isapprox(ss(:y, :Growth_rate),
                       log(1.02) + 0.6 * log(1.01); atol = 1e-10)
        @test isapprox(ss(:c, :Growth_rate), ss(:y, :Growth_rate); atol = 1e-10)
        solution = get_solution(MultipleTrendDriversRegression)
        @test all(isfinite, collect(solution))

        for algorithm in (:pruned_second_order, :pruned_third_order)
            higher_order_solution = get_solution(
                MultipleTrendDriversRegression;
                algorithm = algorithm,
                silent = true,
            )
            @test all(isfinite, collect(higher_order_solution))
        end

        parameters = copy(MultipleTrendDriversRegression.parameter_values)
        SS_and_pars, (solution_error, _) = MacroModelling.get_NSSS_and_parameters(
            MultipleTrendDriversRegression,
            parameters,
            caching = false,
        )
        @test solution_error < 1e-8
        @test any(endswith(string(name), "ᴳ")
                  for name in MultipleTrendDriversRegression.constants.post_complete_parameters.nsss_sol_names)
        internal_SS_and_pars = MacroModelling.internal_steady_state_and_parameters(
            SS_and_pars,
            MultipleTrendDriversRegression,
        )
        stationary_hessian = MacroModelling.calculate_hessian(
            parameters,
            internal_SS_and_pars,
            MultipleTrendDriversRegression.caches,
            MultipleTrendDriversRegression.functions.hessian,
            MultipleTrendDriversRegression.workspaces;
            caching = false,
        )
        direct_hessian = MacroModelling.calculate_bgp_hessian(
            MultipleTrendDriversRegression,
            parameters,
            SS_and_pars;
            caching = false,
        )
        @test isapprox(Matrix(direct_hessian), Matrix(stationary_hessian); atol = 1e-10)

        stationary_third_order = MacroModelling.calculate_third_order_derivatives(
            parameters,
            internal_SS_and_pars,
            MultipleTrendDriversRegression.caches,
            MultipleTrendDriversRegression.functions.third_order_derivatives,
            MultipleTrendDriversRegression.workspaces;
            caching = false,
        )
        direct_third_order = MacroModelling.calculate_bgp_third_order_derivatives(
            MultipleTrendDriversRegression,
            parameters,
            SS_and_pars;
            caching = false,
        )
        @test isapprox(Matrix(direct_third_order), Matrix(stationary_third_order); atol = 1e-10)

        solve!(MultipleTrendDriversRegression;
               parameters = Dict(:α => 0.7, :σa => 0.005,
                                 :σn => 0.005, :σz => 0.01),
               silent = true)
        updated_ss = get_SS(MultipleTrendDriversRegression, derivatives = false)
        @test objectid(MultipleTrendDriversRegression.equations.stationarization) == metadata_id
        @test isapprox(updated_ss(:y, :Growth_rate),
                       log(1.02) + 0.7 * log(1.01); atol = 1e-10)
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
        @test all(isfinite,         collect(get_solution(StationaryRegression)))
    end

    @testset "parameter-dependent growth is updated without rebuilding" begin
        @model ParameterDependentGrowth begin
            x[0] = x[-1] * g[0]
            y[0] = x[0]^α
            g[0] = μ + σg * eg[x]
        end
        @parameters ParameterDependentGrowth begin
            α = 0.5
            μ = 1.02
            σg = 0.01
        end

        metadata = ParameterDependentGrowth.equations.stationarization
        metadata_id = objectid(metadata)
        initial_ss = get_SS(ParameterDependentGrowth, derivatives = false)

        solve!(ParameterDependentGrowth;
               parameters = Dict(:α => 0.7, :μ => 1.02, :σg => 0.01),
               silent = true)
        updated_ss = get_SS(ParameterDependentGrowth, derivatives = false)

        @test objectid(ParameterDependentGrowth.equations.stationarization) == metadata_id
        @test isapprox(updated_ss(:y, :Growth_rate), 0.7 * log(1.02); atol = 1e-10)
        @test initial_ss(:y, :Growth_rate) != updated_ss(:y, :Growth_rate)
    end

    @testset "stationary and BGP representations switch lazily" begin
        @model LazyBGPMode begin
            x[0] = ρ * x[-1]
            y[0] = x[0]
        end
        @parameters LazyBGPMode begin
            ρ = 0.8
        end

        @test LazyBGPMode.equations.stationarization === nothing
        solve!(LazyBGPMode; parameters = [1.1], silent = true)
        @test LazyBGPMode.equations.stationarization !== nothing
        @test Symbol("xᴳ") ∈ LazyBGPMode.constants.post_model_macro.var

        solve!(LazyBGPMode; parameters = [0.9], silent = true)
        @test LazyBGPMode.equations.stationarization === nothing
        @test Symbol("xᴳ") ∉ LazyBGPMode.constants.post_model_macro.var
    end

    @testset "missing parameters initialize BGP dispatch" begin
        @model MissingParameterBGP begin
            x[0] = x[-1] * g[0]
            g[0] = (1 - ρ) * μ + ρ * g[-1] + σg * eg[x]
        end
        @parameters MissingParameterBGP begin
            ρ = 0.2
            σg = 0.01
        end

        @test MissingParameterBGP.equations.bgp_detection === nothing
        solve!(MissingParameterBGP;
               parameters = Dict(:μ => 1.02),
               silent = true)
        @test MissingParameterBGP.equations.stationarization !== nothing
    end
end
