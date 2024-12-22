# using Revise
using Test
using MacroModelling
using Random
using AxisKeys, SparseArrays
import Zygote, FiniteDifferences
import StatsPlots, Turing # has to come before Aqua, otherwise exports are not recognised
using Aqua
if VERSION < v"1.12"
    using JET
end
import LinearAlgebra as ℒ

test_set = ENV["TEST_SET"]

println("Running test set: $test_set")
println("Threads used: ", Threads.nthreads())

include("functionality_tests.jl")

# @testset verbose = true "Code formatting (JuliaFormatter.jl)" begin
#     @test format(MacroModelling; verbose=true, overwrite=true)
# end

# if test_set == "solver0"
#     transform = 0
#     include("optim_solver_params.jl")
# elseif test_set == "solver1"
#     transform = 1
#     include("optim_solver_params.jl")
# elseif test_set == "solver2"
#     transform = 2
#     include("optim_solver_params.jl")
# elseif test_set == "solver3"
#     transform = 3
#     include("optim_solver_params.jl")
# end

if test_set == "estimate_sw07"
    include("test_sw07_estimation.jl")
end

if test_set == "estimation"
    include("test_estimation.jl")
end

if test_set == "1st_order_inversion_estimation"
    include("test_1st_order_inversion_filter_estimation.jl")
end

if test_set == "2nd_order_estimation"
    include("test_2nd_order_estimation.jl")
end

if test_set == "3rd_order_estimation"
    include("test_3rd_order_estimation.jl")
end


if test_set == "higher_order"
    plots = true
    test_higher_order = true

    @testset verbose = true "Test various models: NSSS and 1st order solution" begin
        include("test_models.jl")
    end
    GC.gc()

    @testset verbose = true "FS2000 second order" begin
        include("../models/FS2000.jl")
        functionality_test(FS2000, algorithm = :second_order, plots = plots)
    end
    # FS2000 = nothing
    GC.gc()

    @testset verbose = true "FS2000 pruned second order" begin
        # include("../models/FS2000.jl")
        functionality_test(FS2000, algorithm = :pruned_second_order, plots = plots)
    end
    # FS2000 = nothing
    GC.gc()

    @testset verbose = true "FS2000 third order" begin
        # include("../models/FS2000.jl")
        functionality_test(FS2000, algorithm = :third_order, plots = plots)
    end
    # FS2000 = nothing
    GC.gc()

    @testset verbose = true "FS2000 pruned third order" begin
        # include("../models/FS2000.jl")
        functionality_test(FS2000, algorithm = :pruned_third_order, plots = plots)
    end
    FS2000 = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations and parameter definitions second order" begin
        include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
        functionality_test(m, algorithm = :second_order, plots = plots)
    end
    # m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations and parameter definitions third order" begin
        # include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
        functionality_test(m, algorithm = :third_order, plots = plots)
    end
    m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations, parameter definitions, special functions, variables in steady state, and leads/lag > 1 on endogenous and exogenous variables pruned second order" begin
        include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
        functionality_test(m, algorithm = :pruned_second_order, plots = plots)
    end
    # m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations, parameter definitions, special functions, variables in steady state, and leads/lag > 1 on endogenous and exogenous variables pruned third order" begin
        # include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
        functionality_test(m, algorithm = :pruned_third_order, plots = plots)
    end
    m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations second order" begin
        include("models/RBC_CME_calibration_equations.jl")
        functionality_test(m, algorithm = :second_order, plots = plots)
    end
    # m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations third order" begin
        # include("models/RBC_CME_calibration_equations.jl")
        functionality_test(m, algorithm = :third_order, plots = plots)
    end
    m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME second order" begin
        include("models/RBC_CME.jl")
        functionality_test(m, algorithm = :second_order, plots = plots)
    end
    # m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME third order" begin
        # include("models/RBC_CME.jl")
        functionality_test(m, algorithm = :third_order, plots = plots)
    end
    m = nothing
    GC.gc()
end





if test_set == "plots"
    plots = true
	Random.seed!(1)

    @testset verbose = true "Backus_Kehoe_Kydland_1992" begin
        include("../models/Backus_Kehoe_Kydland_1992.jl")
        functionality_test(Backus_Kehoe_Kydland_1992, plots = plots)
    end
    Backus_Kehoe_Kydland_1992 = nothing
    GC.gc()

    @testset verbose = true "Gali 2015 ELB" begin
        include("../models/Gali_2015_chapter_3_obc.jl")
        functionality_test(Gali_2015_chapter_3_obc, plots = plots)
    end
    Gali_2015_chapter_3_obc = nothing
    GC.gc()

    @testset verbose = true "FS2000" begin
        include("../models/FS2000.jl")
        functionality_test(FS2000, plots = plots)
    end
    FS2000 = nothing
    GC.gc()

    @testset verbose = true "Smets and Wouters (2007) linear" begin
        include("../models/Smets_Wouters_2007_linear.jl")
        functionality_test(Smets_Wouters_2007_linear, plots = plots)
    end
    Smets_Wouters_2007_linear = nothing
    GC.gc()

    @testset verbose = true "Smets and Wouters (2007) nonlinear" begin
        include("../models/Smets_Wouters_2007.jl")
        functionality_test(Smets_Wouters_2007, plots = plots)
    end
    Smets_Wouters_2007 = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations, parameter definitions, special functions, variables in steady state, and leads/lag > 1 on endogenous and exogenous variables" begin
        include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
        functionality_test(m, plots = plots)
        
        observables = [:R, :k]

        Random.seed!(1)
        simulated_data = simulate(m)

        get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

        back_grad = Zygote.gradient(x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        for i in 1:100        
            local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
            if isfinite(ℒ.norm(fin_grad))
                println("Finite differences worked after $i iterations")
                @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
                break
            end
        end

        # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
    end
    m = nothing
    GC.gc()



    @testset verbose = true "RBC_CME with calibration equations, parameter definitions, special functions, variables in steady state, and leads/lag > 1 on endogenous and exogenous variables numerical SS" begin
        include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")
        functionality_test(m, plots = plots)
        
        observables = [:R, :k]

        Random.seed!(1)
        simulated_data = simulate(m)

        get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

        back_grad = Zygote.gradient(x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        for i in 1:100        
            local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
            if isfinite(ℒ.norm(fin_grad))
                println("Finite differences worked after $i iterations")
                @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
                break
            end
        end

        # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
    end
    m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations, parameter definitions, and special functions" begin
        include("models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl")
        functionality_test(m, plots = plots)

        observables = [:R, :k]

        Random.seed!(1)
        simulated_data = simulate(m)

        get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

        back_grad = Zygote.gradient(x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
    
        for i in 1:100        
            local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x, verbose = true), m.parameter_values)
            if isfinite(ℒ.norm(fin_grad))
                println("Finite differences worked after $i iterations")
                @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
                break
            end
        end

        # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
    end
    m = nothing
    GC.gc()

    @testset verbose = true "RBC_CME with calibration equations and parameter definitions" begin
        include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
        functionality_test(m, plots = plots)

        observables = [:R, :k]

        Random.seed!(1)
        simulated_data = simulate(m)

        get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

        back_grad = Zygote.gradient(x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        for i in 1:100        
            local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
            if isfinite(ℒ.norm(fin_grad))
                println("Finite differences worked after $i iterations")
                @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
                break
            end
        end

        # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
    end
    m = nothing
    GC.gc()


    @testset verbose = true "RBC_CME with calibration equations" begin
        include("models/RBC_CME_calibration_equations.jl")
        functionality_test(m, plots = plots)
        
        observables = [:R, :k]

        Random.seed!(1)
        simulated_data = simulate(m)

        get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

        back_grad = Zygote.gradient(x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        for i in 1:100        
            local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
            if isfinite(ℒ.norm(fin_grad))
                println("Finite differences worked after $i iterations")
                @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
                break
            end
        end

        # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
    end
    m = nothing
    GC.gc()



    @testset verbose = true "RBC_CME" begin
        include("models/RBC_CME.jl")
        functionality_test(m, plots = plots)

        observables = [:R, :k]

        Random.seed!(1)
        simulated_data = simulate(m)

        get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

        back_grad = Zygote.gradient(x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

        for i in 1:100        
            local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
            if isfinite(ℒ.norm(fin_grad))
                println("Finite differences worked after $i iterations")
                @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
                break
            end
        end

        # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
    end
    m = nothing
    GC.gc()



    @testset verbose = true "Smets_Wouters_2003 with calibration equations" begin
        include("../models/Smets_Wouters_2003.jl")
        functionality_test(Smets_Wouters_2003, plots = plots)
    end
    Smets_Wouters_2003 = nothing
    GC.gc()
end



if test_set == "basic"
    plots = false
    test_higher_order = false

    @testset verbose = true "Code quality (Aqua.jl)" begin
        # Aqua.test_all(MacroModelling)
        @testset "Compare Project.toml and test/Project.toml" Aqua.test_project_extras(MacroModelling)
        @testset "Stale dependencies" Aqua.test_stale_deps(MacroModelling)#; ignore = [:Aqua, :JET])
        @testset "Unbound type parameters" Aqua.test_unbound_args(MacroModelling)
        @testset "Undefined exports" Aqua.test_undefined_exports(MacroModelling)
        @testset "Piracy" Aqua.test_piracies(MacroModelling)
        @testset "Method ambiguity" Aqua.test_ambiguities(MacroModelling, recursive = false)
        @testset "Compat" Aqua.test_deps_compat(MacroModelling)#; ignore = [:Aqua, :JET])
        # @testset "Persistent tasks" Aqua.test_persistent_tasks(MacroModelling)
    end
    GC.gc()
    
    @testset verbose = true "Static checking (JET.jl)" begin
        if VERSION < v"1.12"
            JET.test_package(MacroModelling; target_defined_modules = true, toplevel_logger = nothing)
        end
    end

    @testset verbose = true "Test various models: NSSS and 1st order solution" begin
        include("test_models.jl")
    end
    GC.gc()

    @testset verbose = true "Standalone functions" begin
        include("test_standalone_function.jl")
    end
    GC.gc()


    @testset verbose = true "Model without shocks" begin
        @model m begin
            K[0] = (1 - δ) * K[-1] + I[0]
            Z[0] = (1 - ρ) * μ + ρ * Z[-1] 
            I[1]  = ((ρ + δ - Z[0])/(1 - δ))  + ((1 + ρ)/(1 - δ)) * I[0]
        end

        @parameters m verbose = true begin
            ρ = 0.05
            δ = 0.10
            μ = .17
            σ = .2
        end

        m_ss = get_steady_state(m)
        @test isapprox(m_ss(:,:Steady_state),[1/7.5,1/.75,.17],rtol = eps(Float32))

        m_sol = get_solution(m) 
        @test isapprox(m_sol(:,:K),[1/.75,.9,.04975124378109454],rtol = eps(Float32))

        init = m_ss(:,:Steady_state) |> collect
        init[2] *= 1.5
        get_irf(m, initial_state = init, shocks = :none)

        plot_irf(m, initial_state = init, shocks = :none)
        @test true
    end
    m = nothing



    @testset verbose = true "Non-stochastic steady state guess" begin
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME verbose = true guess = Dict(:alpha => .2, :beta => .99) begin
            alpha | k[ss] / (4 * y[ss]) = cap_share
            cap_share = 1.66
            # alpha = .157

            beta | R[ss] = R_ss # beta needs to enter into function: block in order to solve
            R_ss = 1.0035
            # beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            delta | delta * k[ss] / y[ss] = I_K_ratio #check why this doesnt solve for y; because delta is not recognised as a free parameter here.
            I_K_ratio = .15
            # delta = .0226

            Pibar | Pi[ss] = Pi_ss
            Pi_ss = 1.0025
            # Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005

            
            # cap_share > 0
            # R_ss > 0
            # Pi_ss > 0
            # I_K_ratio > 0 

            # 0 < alpha < 1 
            # 0 < beta < 1
            # 0 < delta < 1
            # 0 < Pibar
            # 0 <= rhoz < 1
            # phi_pi > 0

            # 0 < A < 1
            # 0 < k < 50
            # 0 < y < 10
            # 0 < c < 10
        end
        
        @test RBC_CME.guess == Dict(:alpha => .2, :beta => .99)

        @test get_steady_state(RBC_CME, verbose = true)(RBC_CME.var,:Steady_state) ≈ [1.0, 1.0025, 1.0035, 1.2081023824176236, 9.437411552284384, 1.4212969205027686, 1.0]

        RBC_CME = nothing
    end

    @testset verbose = true "Distribution functions, general and SS" begin
        
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            c_normcdf[0]= normcdf(c[0])
            c_normpdf[0]= normpdf(c[0])
            c_norminvcdf[0]= norminvcdf(c[0]-1)
            c_norminv[0]= norminv(c[0]-1)
            c_qnorm[0]= qnorm(c[0]-1)
            c_dnorm[0]= dnorm(c[0])
            c_pnorm[0]= pnorm(c[0])
            c_normlogpdf[0]= normlogpdf(c[0])
            # c_norm[0]= cdf(Normal(),c[0])
            c_inv[0] = erfcinv(c[0])
            # c_binomlogpdf[0]= binomlogpdf(c[0])
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end

        get_solution(RBC_CME)


        @model finacc begin
            R[0] * beta = C[1] / C[0]
            C[0] = w[0] * L[0] - B[0] + R[-1] * B[-1] + (1-v) * (Rk[-1] * Q[-1] * K[-1] - (R[-1] + mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1])) * (Q[-1] * K[-1] - N[-1])) - We 
            w[0] = C[0] / (1-L[0])
            K[0] = (1-delta) * K[-1] + I[0] 
            Q[0] = 1 + chi * (I[0] / K[-1] - delta)
            Y[0] = A[0] * K[-1]^alpha * L[0]^(1-alpha)
            Rk[-1] = (alpha * Y[0] / K[-1] + Q[0] * (1-delta))/Q[-1]
            w[0] = (1-alpha) * Y[0] / L[0]
            N[0] = v * (Rk[-1] * Q[-1] * K[-1] - (R[-1] + mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1])) * (Q[-1] * K[-1] - N[-1])) + We 
            0 = (omegabar[0] * (1 - F[0]) + (1 - mu) * G[0]) * Rk[0] / R[0] * Q[0] * K[0] / N[0] - (Q[0] * K[0] / N[0] - 1)
            0 = (1 - (omegabar[0] * (1 - F[0]) + G[0])) * Rk[0] / R[0] + (1 - F[0]) / (1 - F[0] - omegabar[0] * mu * (normpdf((log(omegabar[0]) + sigma^2/2)  / sigma)/ omegabar[0]  / sigma)) * ((omegabar[0] * (1 - F[0]) + (1 - mu) * G[0]) * Rk[0] / R[0] - 1) 
            G[0] = normcdf(((log(omegabar[0])+sigma^2/2)/sigma) - sigma)
            F[0] = normcdf((log(omegabar[0])+sigma^2/2)/sigma)
            EFP[0] = (mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1]))
            Y[0] + walras[0] = C[0] + I[0] + EFP[0] * (Q[-1] * K[-1] - N[-1])
            B[0] = Q[0] * K[0] - N[0]  
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
        end


        @parameters finacc verbose = true begin
            beta = 0.99 
            delta = 0.02 
            We = 1e-12 
            alpha = 0.36 
            chi = 0 
            v = 0.978
            mu = 0.94
            sigma = 0.2449489742783178
            rhoz = .9
            std_eps = .0068

            .5 > omegabar > .44
            K > 15
            0 < L < .45
        end

        # solve!(finacc)
        @test isapprox(get_steady_state(finacc,derivatives = false)[:,1],[1.0, 7.004987166460695, 1.2762549358842095, 0.0008293608419033882, 0.0009318065746306208, 0.0003952537570055814, 0.30743973601435376, 15.371986800781423, 0.4435430773517457, 8.366999635233856, 1.0000000000593001, 1.0101010101010102, 1.0172249577970442, 1.5895043340984303, 0.4529051354389826, 2.2935377097663356, -1.4597012487627126e-10], rtol = 1e-6)
    end
    RBC_CME = nothing
    finacc = nothing

    @testset verbose = true "Lead and lag > 1" begin
        
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end

        sol = get_solution(RBC_CME)

        # sol1 = get_solution(RBC_CME, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol, sol1, atol = 1e-4)

        # sol2 = get_solution(RBC_CME, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol, sol2, atol = 1e-4)

        sol3 = get_solution(RBC_CME, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol, sol3, atol = 1e-4)


        # exo multi lead/lag >> 1
        @model RBC_CME_exo_mult begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * (eps_z[x-8] + eps_z[x-4] + eps_z[x+4] + eps_z_s[x])
            # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_exo_mult verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_exo_mult
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_exo_mult = nothing


        # endo/exo multi lead/lag >> 1
        @model RBC_CME_all_mult begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * (eps_z[x-8] + eps_z[x-4] + eps_z[x+4] + eps_z_s[x])
            ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            ZZ_avg_fut[0] = (A[0] + A[1] + A[2] + A[3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_all_mult verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_all_mult
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_all_mult = nothing


        # exo lead >> 1
        @model RBC_CME_exo_lead1 begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x+8]
            # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_exo_lead1 verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_exo_lead1
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_exo_lead1 = nothing



        # exo multi lag >> 1
        @model RBC_CME_exo_lag_mult begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * (eps_z[x-8] + eps_z[x-4] + eps_z_s[x])
            # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_exo_lag_mult verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_exo_lag_mult
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_exo_lag_mult = nothing


        # exo lag >> 1
        @model RBC_CME_exo_lag1 begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x-8]
            # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_exo_lag1 verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_exo_lag1
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_exo_lag1 = nothing


        # exo lead > 1
        @model RBC_CME_exo_lead begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x+1]
            # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_exo_lead verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_exo_lead
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_exo_lead = nothing


        # exo lag > 1
        @model RBC_CME_exo_lag begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x-1]
            # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_exo_lag verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_exo_lag

        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_exo_lag = nothing

        # Lags > 1
        @model RBC_CME_lag begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            ZZ_dev[0] = log(c[0]/c[ss])
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_lag verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end

        m = RBC_CME_lag
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_lag = nothing

        # Leads > 1
        @model RBC_CME_lead begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            ZZ_avg[0] = (A[0] + A[1] + A[2] + A[3]) / 4
            ZZ_dev[0] = log(c[0]/c[ss])
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end
        
        @parameters RBC_CME_lead verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_lead
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_lead = nothing


        # Leads and lags > 1
        @model RBC_CME_lead_lag begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            ZZ_avg_f[0] = (A[0] + A[1] + A[2] + A[3]) / 4
            ZZ_avg_b[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_lead_lag verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_lead_lag
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_lead_lag = nothing


        # Leads and lags > 10
        @model RBC_CME_lead_lag10 begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            ZZ_avg_f[0] = (A[0] + A[1] + A[2] + A[3]) / 4
            ZZ_avg_b[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            ZZ_avg_f10[0] = (A[0] + A[10]) / 2
            # ZZ_avg_b10[0] = (A[0] + A[-10]) / 2
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_lead_lag10 verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_lead_lag10
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_lead_lag10 = nothing


        # Leads and lags > 10
        @model RBC_CME_lead_lag20 begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            ZZ_avg_f[0] = (A[0] + A[1] + A[2] + A[3]) / 4
            ZZ_avg_b[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            ZZ_avg_ff[0] = (A[0] + A[10]) / 2
            ZZ_avg_bb[0] = (A[0] + A[-10]) / 2
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME_lead_lag20 verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        
        m = RBC_CME_lead_lag20
        
        sol_exo = get_solution(m)

        # sol_exo1 = get_solution(m, algorithm = :linear_time_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        # @test isapprox(sol_exo, sol_exo1, atol = 1e-4)


        # sol_exo2 = get_solution(m, algorithm = :quadratic_iteration, verbose = true)

        # @test isapprox(sol(axiskeys(sol, 1)[1:end-1],:), sol_exo2(axiskeys(sol, 1)[1:end-1], axiskeys(sol, 2)), atol = 1e-4)

        sol_exo3 = get_solution(m, quadratic_matrix_equation_algorithm = :doubling, verbose = true)

        @test isapprox(sol_exo, sol_exo3, atol = 1e-4)

        m = nothing
        RBC_CME_lead_lag20 = nothing
    end


    @testset verbose = true "Steady state RBC CME model" begin
        # Basic test
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            # A_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
            # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
            # y_avg[0] = log(y[0] / y[-4])
            # y_growth[0] = log(y[1] / y[2])
            # y_growthl[0] = log(y[0] / y[1])
            # y_growthl1[0] = log(y[-1] / y[0])
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end
        # get_steady_state(RBC_CME)[1]
        # using NLopt
        # RBC_CME.SS_optimizer = NLopt.LD_LBFGS
        # solve!(RBC_CME)
        @test get_steady_state(RBC_CME)(RBC_CME.var,:Steady_state) ≈ [1.0, 1.0024019205374952, 1.003405325870413, 1.2092444352939415, 9.467573947982233, 1.42321160651834, 1.0]
        # get_moments(RBC_CME)[1]
        # irf(RBC_CME)
        
        RBC_CME = nothing

        
        # Symbolic test
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME symbolic = true verbose = true begin
            # alpha | k[ss] / (4 * y[ss]) = cap_share
            # cap_share = 1.66
            alpha = .157

            # beta | R[ss] = R_ss
            # R_ss = 1.0035
            beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
            # I_K_ratio = .15
            delta = .0226

            # Pibar | Pi[ss] = Pi_ss
            # Pi_ss = 1.0025
            Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005

            # cap_share > 0
            # R_ss > 0
            # Pi_ss > 0
            # I_K_ratio > 0 

            # 0 < alpha < 1 
            0 < beta < 1
            # 0 < delta < 1
            0 < Pibar
            # 0 <= rhoz < 1
            phi_pi > 0

            # 0 < A < 1
            # 0 < k < 50
            0 < Pi
            0 < R
        end
        # get_steady_state(RBC_CME)[1]
        # using NLopt
        # RBC_CME.SS_optimizer = NLopt.LD_LBFGS
        # solve!(RBC_CME,symbolic_SS = true)
        @test get_steady_state(RBC_CME)(RBC_CME.var,:Steady_state) ≈ [1.0, 1.0024019205374952, 1.003405325870413, 1.2092444352939415, 9.467573947982233, 1.42321160651834, 1.0]
        # get_moments(RBC_CME)[1]

        RBC_CME = nothing




        # Numerical test with calibration targets
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME verbose = true begin
            alpha | k[ss] / (4 * y[ss]) = cap_share
            cap_share = 1.66
            # alpha = .157

            beta | R[ss] = R_ss # beta needs to enter into function: block in order to solve
            R_ss = 1.0035
            # beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            delta | delta * k[ss] / y[ss] = I_K_ratio #check why this doesnt solve for y; because delta is not recognised as a free parameter here.
            I_K_ratio = .15
            # delta = .0226

            Pibar | Pi[ss] = Pi_ss
            Pi_ss = 1.0025
            # Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005

            
            # cap_share > 0
            # R_ss > 0
            # Pi_ss > 0
            # I_K_ratio > 0 

            # 0 < alpha < 1 
            # 0 < beta < 1
            # 0 < delta < 1
            # 0 < Pibar
            # 0 <= rhoz < 1
            # phi_pi > 0

            # 0 < A < 1
            # 0 < k < 50
            # 0 < y < 10
            # 0 < c < 10
        end
        # get_steady_state(RBC_CME)[1]
        # using NLopt
        # RBC_CME.SS_optimizer = NLopt.LD_LBFGS
        # solve!(RBC_CME, verbose = true)
        # RBC_CME.SS_init_guess[1:7] = [1.0, 1.0025, 1.0035, 1.2081023828249515, 9.437411555244328, 1.4212969209705313, 1.0]
        # get_steady_state(RBC_CME)
        @test get_steady_state(RBC_CME, verbose = true)(RBC_CME.var,:Steady_state) ≈ [1.0, 1.0025, 1.0035, 1.2081023824176236, 9.437411552284384, 1.4212969205027686, 1.0]
        # get_moments(RBC_CME)[1]

        # RBC_CME.ss_solve_blocks[1]([0.15662344139650963, 1.2081023828249515, 0.02259036144578319, 9.437411555244328, 1.4212969209705313],RBC_CME)

        RBC_CME = nothing


        # Symbolic test with calibration targets
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME symbolic = true verbose = true begin
            alpha | k[ss] / (4 * y[ss]) = cap_share
            cap_share = 1.66
            # alpha = .157

            beta | R[ss] = R_ss
            R_ss = 1.0035
            # beta = .999

            delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss] / y[ss] = I_K_ratio # this doesnt solve symbolically
            I_K_ratio = .15
            # delta = .0226

            Pibar | Pi[ss] = Pi_ss
            Pi_ss = 1.0025
            # Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
            
            # cap_share > 0
            # R_ss > 0
            # Pi_ss > 0
            # I_K_ratio > 0 

            # 0 < alpha < .5
            # 0 < beta < 1
            # 0 < delta < 1
            # 0 < Pibar
            # 0 <= rhoz < 1
            # phi_pi > 0

            # 0 < A < 1
            # 0 < k < 50
            # 0 < y < 10
            # 0 < c < 10
        end
        # # get_steady_state(RBC_CME)[1]
        # # using NLopt
        # # RBC_CME.SS_optimizer = NLopt.LD_LBFGS
        # # get_steady_state(RBC_CME)
        @test isapprox(get_steady_state(RBC_CME, verbose = true)(RBC_CME.var,:Steady_state), [1.0, 1.0025, 1.0035, 1.2081023828249515, 9.437411555244328, 1.4212969209705313, 1.0],rtol = eps(Float32))
        # get_moments(RBC_CME)[1]
        
        RBC_CME = nothing
    end




    # using MacroModelling: @model, @parameters, get_steady_state, solve!

    @testset verbose = true "Steady state Smets_Wouters_2003 model" begin 
        
        @model Smets_Wouters_2003 begin
            -q[0] + beta * ((1 - tau) * q[1] + epsilon_b[1] * (r_k[1] * z[1] - psi^-1 * r_k[ss] * (-1 + exp(psi * (-1 + z[1])))) * (C[1] - h * C[0])^(-sigma_c)) = 0
            -q_f[0] + beta * ((1 - tau) * q_f[1] + epsilon_b[1] * (r_k_f[1] * z_f[1] - psi^-1 * r_k_f[ss] * (-1 + exp(psi * (-1 + z_f[1])))) * (C_f[1] - h * C_f[0])^(-sigma_c)) = 0
            -r_k[0] + alpha * epsilon_a[0] * mc[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^(-1 + alpha) = 0
            -r_k_f[0] + alpha * epsilon_a[0] * mc_f[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^(-1 + alpha) = 0
            -G[0] + T[0] = 0
            -G[0] + G_bar * epsilon_G[0] = 0
            -G_f[0] + T_f[0] = 0
            -G_f[0] + G_bar * epsilon_G[0] = 0
            -L[0] + nu_w[0]^-1 * L_s[0] = 0
            -L_s_f[0] + L_f[0] * (W_i_f[0] * W_f[0]^-1)^(lambda_w^-1 * (-1 - lambda_w)) = 0
            L_s_f[0] - L_f[0] = 0
            L_s_f[0] + lambda_w^-1 * L_f[0] * W_f[0]^-1 * (-1 - lambda_w) * (-W_disutil_f[0] + W_i_f[0]) * (W_i_f[0] * W_f[0]^-1)^(-1 + lambda_w^-1 * (-1 - lambda_w)) = 0
            Pi_ws_f[0] - L_s_f[0] * (-W_disutil_f[0] + W_i_f[0]) = 0
            Pi_ps_f[0] - Y_f[0] * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) = 0
            -Q[0] + epsilon_b[0]^-1 * q[0] * (C[0] - h * C[-1])^(sigma_c) = 0
            -Q_f[0] + epsilon_b[0]^-1 * q_f[0] * (C_f[0] - h * C_f[-1])^(sigma_c) = 0
            -W[0] + epsilon_a[0] * mc[0] * (1 - alpha) * L[0]^(-alpha) * (K[-1] * z[0])^alpha = 0
            -W_f[0] + epsilon_a[0] * mc_f[0] * (1 - alpha) * L_f[0]^(-alpha) * (K_f[-1] * z_f[0])^alpha = 0
            -Y_f[0] + Y_s_f[0] = 0
            Y_s[0] - nu_p[0] * Y[0] = 0
            -Y_s_f[0] + Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) = 0
            beta * epsilon_b[1] * (C_f[1] - h * C_f[0])^(-sigma_c) - epsilon_b[0] * R_f[0]^-1 * (C_f[0] - h * C_f[-1])^(-sigma_c) = 0
            beta * epsilon_b[1] * pi[1]^-1 * (C[1] - h * C[0])^(-sigma_c) - epsilon_b[0] * R[0]^-1 * (C[0] - h * C[-1])^(-sigma_c) = 0
            Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) - lambda_p^-1 * Y_f[0] * (1 + lambda_p) * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-1 - lambda_p^-1 * (1 + lambda_p)) = 0
            epsilon_b[0] * W_disutil_f[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) - omega * epsilon_b[0] * epsilon_L[0] * L_s_f[0]^sigma_l = 0
            -1 + xi_p * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1) + (1 - xi_p) * pi_star[0]^(-lambda_p^-1) = 0
            -1 + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1) + xi_w * (W[-1] * W[0]^-1)^(-lambda_w^-1) * (pi[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1) = 0
            -Phi - Y_s[0] + epsilon_a[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^alpha = 0
            -Phi - Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_a[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^alpha = 0
            eta_b[exo] - log(epsilon_b[0]) + rho_b * log(epsilon_b[-1]) = 0
            -eta_L[exo] - log(epsilon_L[0]) + rho_L * log(epsilon_L[-1]) = 0
            eta_I[exo] - log(epsilon_I[0]) + rho_I * log(epsilon_I[-1]) = 0
            eta_w[exo] - f_1[0] + f_2[0] = 0
            eta_a[exo] - log(epsilon_a[0]) + rho_a * log(epsilon_a[-1]) = 0
            eta_p[exo] - g_1[0] + g_2[0] * (1 + lambda_p) = 0
            eta_G[exo] - log(epsilon_G[0]) + rho_G * log(epsilon_G[-1]) = 0
            -f_1[0] + beta * xi_w * f_1[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1) + epsilon_b[0] * w_star[0] * L[0] * (1 + lambda_w)^-1 * (C[0] - h * C[-1])^(-sigma_c) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)) = 0
            -f_2[0] + beta * xi_w * f_2[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) + omega * epsilon_b[0] * epsilon_L[0] * (L[0] * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)))^(1 + sigma_l) = 0
            -g_1[0] + beta * xi_p * pi_star[0] * g_1[1] * pi_star[1]^-1 * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1) + epsilon_b[0] * pi_star[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c) = 0
            -g_2[0] + beta * xi_p * g_2[1] * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_b[0] * mc[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c) = 0
            -nu_w[0] + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)) + xi_w * nu_w[-1] * (W[-1] * pi[0]^-1 * W[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w)) = 0
            -nu_p[0] + (1 - xi_p) * pi_star[0]^(-lambda_p^-1 * (1 + lambda_p)) + xi_p * nu_p[-1] * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p)) = 0
            -K[0] + K[-1] * (1 - tau) + I[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2) = 0
            -K_f[0] + K_f[-1] * (1 - tau) + I_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2) = 0
            U[0] - beta * U[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C[0] - h * C[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s[0]^(1 + sigma_l)) = 0
            U_f[0] - beta * U_f[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C_f[0] - h * C_f[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s_f[0]^(1 + sigma_l)) = 0
            -epsilon_b[0] * (C[0] - h * C[-1])^(-sigma_c) + q[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2 - varphi * I[-1]^-1 * epsilon_I[0] * I[0] * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])) + beta * varphi * I[0]^-2 * epsilon_I[1] * q[1] * I[1]^2 * (-1 + I[0]^-1 * epsilon_I[1] * I[1]) = 0
            -epsilon_b[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) + q_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2 - varphi * I_f[-1]^-1 * epsilon_I[0] * I_f[0] * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])) + beta * varphi * I_f[0]^-2 * epsilon_I[1] * q_f[1] * I_f[1]^2 * (-1 + I_f[0]^-1 * epsilon_I[1] * I_f[1]) = 0
            eta_pi[exo] - log(pi_obj[0]) + rho_pi_bar * log(pi_obj[-1]) + log(calibr_pi_obj) * (1 - rho_pi_bar) = 0
            -C[0] - I[0] - T[0] + Y[0] - psi^-1 * r_k[ss] * K[-1] * (-1 + exp(psi * (-1 + z[0]))) = 0
            -calibr_pi + eta_R[exo] - log(R[ss]^-1 * R[0]) + r_Delta_pi * (-log(pi[ss]^-1 * pi[-1]) + log(pi[ss]^-1 * pi[0])) + r_Delta_y * (-log(Y[ss]^-1 * Y[-1]) + log(Y[ss]^-1 * Y[0]) + log(Y_f[ss]^-1 * Y_f[-1]) - log(Y_f[ss]^-1 * Y_f[0])) + rho * log(R[ss]^-1 * R[-1]) + (1 - rho) * (log(pi_obj[0]) + r_pi * (-log(pi_obj[0]) + log(pi[ss]^-1 * pi[-1])) + r_Y * (log(Y[ss]^-1 * Y[0]) - log(Y_f[ss]^-1 * Y_f[0]))) = 0
            -C_f[0] - I_f[0] + Pi_ws_f[0] - T_f[0] + Y_f[0] + L_s_f[0] * W_disutil_f[0] - L_f[0] * W_f[0] - psi^-1 * r_k_f[ss] * K_f[-1] * (-1 + exp(psi * (-1 + z_f[0]))) = 0
            epsilon_b[0] * (K[-1] * r_k[0] - r_k[ss] * K[-1] * exp(psi * (-1 + z[0]))) * (C[0] - h * C[-1])^(-sigma_c) = 0
            epsilon_b[0] * (K_f[-1] * r_k_f[0] - r_k_f[ss] * K_f[-1] * exp(psi * (-1 + z_f[0]))) * (C_f[0] - h * C_f[-1])^(-sigma_c) = 0
        end


        @parameters Smets_Wouters_2003 verbose = true begin  
            calibr_pi_obj | 1 = pi_obj[ss]
            calibr_pi | pi[ss] = pi_obj[ss]
            # Phi | Y_s[ss] * .408 = Phi
            # Phi = .408 * Y_j[ss]
            # (Y_j[ss] + Phi) / Y_j[ss] = 1.408 -> Phi; | this seems problematic because of the parameter
            # lambda_p | .6 = C_f[ss] / Y_f[ss]
            # lambda_w | L[ss] = .33
            # G_bar | .18 = G[ss] / Y[ss]
            # calibr_pi_obj = 0
            # calibr_pi = 1
            lambda_p = .368
            G_bar = .362
            lambda_w = 0.5
            Phi = .819

            alpha = 0.3
            beta = 0.99
            gamma_w = 0.763
            gamma_p = 0.469
            h = 0.573
            omega = 1
            psi = 0.169
            r_pi = 1.684
            r_Y = 0.099
            r_Delta_pi = 0.14
            r_Delta_y = 0.159
            rho = 0.961
            rho_b = 0.855
            rho_L = 0.889
            rho_I = 0.927
            rho_a = 0.823
            rho_G = 0.949
            rho_pi_bar = 0.924
            sigma_c = 1.353
            sigma_l = 2.4
            tau = 0.025
            varphi = 6.771
            xi_w = 0.737
            xi_p = 0.908

            # Putting non-negative constraint on first block is enough
            # 0 < K
            # 0 < I
            # 0 < Y_s
            # 0 < q
            # 0 < r_k
            # 5 < f_1
            # 0 < L
            # 0 < W
            # 30 < g_1
            # 0 < z
            # 0 < mc
            # 0 < w_star
            # 5 < f_2
            # 0 < Y
            # 0 < g_2
            # 0 < C
        end


        # solve!(Smets_Wouters_2003, verbose = true)


        @test isapprox(get_steady_state(Smets_Wouters_2003, verbose = true)(Smets_Wouters_2003.timings.var,[:Steady_state]),
                        [  1.2043777509278788
                        1.2043777484127967
                        0.362
                        0.362
                        0.44153840098985714
                        0.44153839784516097
                        17.66153603957938
                        17.66153591381742
                        1.2891159430437658
                        1.2891159432893282
                        1.289115942962812
                        1.289115943290125
                        0.9999999999999677
                        0.5401411855429173
                        0.482173806623137
                        0.999999999999352
                        1.0000000000002556
                        1.0101010101010102
                        1.0101010101010102
                        0.362
                        0.362
                        -427.9858908413812
                        -427.98589116567274
                        1.122103431093411
                        0.7480689524203904
                        1.1221034286309022
                        1.122103428630708
                        2.0079161519182205
                        2.0079161462568305
                        2.0079161519185624
                        2.007916146256947
                        1.0
                        1.0
                        1.0
                        1.0
                        1.0
                        8.770699454739315
                        8.770699454739393
                        48.84717584575426
                        35.70699988724729
                        0.7309941520471651
                        0.730994152046712
                        1.0
                        0.9999999999372022
                        1.0
                        1.0
                        1.0
                        2.4590033452182944
                        2.459003352168301
                        0.035101010082054955
                        0.03510101010809239
                        1.1221034311168996
                        0.9999999999999698
                        1.0000000000000089],
                        rtol = eps(Float32)
        )

        Smets_Wouters_2003 = nothing
    #     # x = 1


    # #     Smets_Wouters_2003 = nothing

    # #     @model Smets_Wouters_2003 begin
    # #         -q[0] + beta * ((1 - tau) * q[1] + epsilon_b[1] * (r_k[1] * z[1] - psi^-1 * r_k[ss] * (-1 + exp(psi * (-1 + z[1])))) * (C[1] - h * C[0])^(-sigma_c)) = 0
    # #         -q_f[0] + beta * ((1 - tau) * q_f[1] + epsilon_b[1] * (r_k_f[1] * z_f[1] - psi^-1 * r_k_f[ss] * (-1 + exp(psi * (-1 + z_f[1])))) * (C_f[1] - h * C_f[0])^(-sigma_c)) = 0
    # #         -r_k[0] + alpha * epsilon_a[0] * mc[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^(-1 + alpha) = 0
    # #         -r_k_f[0] + alpha * epsilon_a[0] * mc_f[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^(-1 + alpha) = 0
    # #         -G[0] + T[0] = 0
    # #         -G[0] + G_bar * epsilon_G[0] = 0
    # #         -G_f[0] + T_f[0] = 0
    # #         -G_f[0] + G_bar * epsilon_G[0] = 0
    # #         -L[0] + nu_w[0]^-1 * L_s[0] = 0
    # #         -L_s_f[0] + L_f[0] * (W_i_f[0] * W_f[0]^-1)^(lambda_w^-1 * (-1 - lambda_w)) = 0
    # #         L_s_f[0] - L_f[0] = 0
    # #         L_s_f[0] + lambda_w^-1 * L_f[0] * W_f[0]^-1 * (-1 - lambda_w) * (-W_disutil_f[0] + W_i_f[0]) * (W_i_f[0] * W_f[0]^-1)^(-1 + lambda_w^-1 * (-1 - lambda_w)) = 0
    # #         Pi_ws_f[0] - L_s_f[0] * (-W_disutil_f[0] + W_i_f[0]) = 0
    # #         Pi_ps_f[0] - Y_f[0] * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) = 0
    # #         -Q[0] + epsilon_b[0]^-1 * q[0] * (C[0] - h * C[-1])^(sigma_c) = 0
    # #         -Q_f[0] + epsilon_b[0]^-1 * q_f[0] * (C_f[0] - h * C_f[-1])^(sigma_c) = 0
    # #         -W[0] + epsilon_a[0] * mc[0] * (1 - alpha) * L[0]^(-alpha) * (K[-1] * z[0])^alpha = 0
    # #         -W_f[0] + epsilon_a[0] * mc_f[0] * (1 - alpha) * L_f[0]^(-alpha) * (K_f[-1] * z_f[0])^alpha = 0
    # #         -Y_f[0] + Y_s_f[0] = 0
    # #         Y_s[0] - nu_p[0] * Y[0] = 0
    # #         -Y_s_f[0] + Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) = 0
    # #         beta * epsilon_b[1] * (C_f[1] - h * C_f[0])^(-sigma_c) - epsilon_b[0] * R_f[0]^-1 * (C_f[0] - h * C_f[-1])^(-sigma_c) = 0
    # #         beta * epsilon_b[1] * pi[1]^-1 * (C[1] - h * C[0])^(-sigma_c) - epsilon_b[0] * R[0]^-1 * (C[0] - h * C[-1])^(-sigma_c) = 0
    # #         Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) - lambda_p^-1 * Y_f[0] * (1 + lambda_p) * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-1 - lambda_p^-1 * (1 + lambda_p)) = 0
    # #         epsilon_b[0] * W_disutil_f[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) - omega * epsilon_b[0] * epsilon_L[0] * L_s_f[0]^sigma_l = 0
    # #         -1 + xi_p * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1) + (1 - xi_p) * pi_star[0]^(-lambda_p^-1) = 0
    # #         -1 + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1) + xi_w * (W[-1] * W[0]^-1)^(-lambda_w^-1) * (pi[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1) = 0
    # #         -Phi - Y_s[0] + epsilon_a[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^alpha = 0
    # #         -Phi - Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_a[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^alpha = 0
    # #         eta_b[exo] - log(epsilon_b[0]) + rho_b * log(epsilon_b[-1]) = 0
    # #         -eta_L[exo] - log(epsilon_L[0]) + rho_L * log(epsilon_L[-1]) = 0
    # #         eta_I[exo] - log(epsilon_I[0]) + rho_I * log(epsilon_I[-1]) = 0
    # #         eta_w[exo] - f_1[0] + f_2[0] = 0
    # #         eta_a[exo] - log(epsilon_a[0]) + rho_a * log(epsilon_a[-1]) = 0
    # #         eta_p[exo] - g_1[0] + g_2[0] * (1 + lambda_p) = 0
    # #         eta_G[exo] - log(epsilon_G[0]) + rho_G * log(epsilon_G[-1]) = 0
    # #         -f_1[0] + beta * xi_w * f_1[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1) + epsilon_b[0] * w_star[0] * L[0] * (1 + lambda_w)^-1 * (C[0] - h * C[-1])^(-sigma_c) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)) = 0
    # #         -f_2[0] + beta * xi_w * f_2[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) + omega * epsilon_b[0] * epsilon_L[0] * (L[0] * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)))^(1 + sigma_l) = 0
    # #         -g_1[0] + beta * xi_p * pi_star[0] * g_1[1] * pi_star[1]^-1 * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1) + epsilon_b[0] * pi_star[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c) = 0
    # #         -g_2[0] + beta * xi_p * g_2[1] * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_b[0] * mc[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c) = 0
    # #         -nu_w[0] + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)) + xi_w * nu_w[-1] * (W[-1] * pi[0]^-1 * W[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w)) = 0
    # #         -nu_p[0] + (1 - xi_p) * pi_star[0]^(-lambda_p^-1 * (1 + lambda_p)) + xi_p * nu_p[-1] * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p)) = 0
    # #         -K[0] + K[-1] * (1 - tau) + I[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2) = 0
    # #         -K_f[0] + K_f[-1] * (1 - tau) + I_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2) = 0
    # #         U[0] - beta * U[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C[0] - h * C[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s[0]^(1 + sigma_l)) = 0
    # #         U_f[0] - beta * U_f[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C_f[0] - h * C_f[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s_f[0]^(1 + sigma_l)) = 0
    # #         -epsilon_b[0] * (C[0] - h * C[-1])^(-sigma_c) + q[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2 - varphi * I[-1]^-1 * epsilon_I[0] * I[0] * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])) + beta * varphi * I[0]^-2 * epsilon_I[1] * q[1] * I[1]^2 * (-1 + I[0]^-1 * epsilon_I[1] * I[1]) = 0
    # #         -epsilon_b[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) + q_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2 - varphi * I_f[-1]^-1 * epsilon_I[0] * I_f[0] * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])) + beta * varphi * I_f[0]^-2 * epsilon_I[1] * q_f[1] * I_f[1]^2 * (-1 + I_f[0]^-1 * epsilon_I[1] * I_f[1]) = 0
    # #         eta_pi[exo] - log(pi_obj[0]) + rho_pi_bar * log(pi_obj[-1]) + log(calibr_pi_obj) * (1 - rho_pi_bar) = 0
    # #         -C[0] - I[0] - T[0] + Y[0] - psi^-1 * r_k[ss] * K[-1] * (-1 + exp(psi * (-1 + z[0]))) = 0
    # #         -calibr_pi + eta_R[exo] - log(R[ss]^-1 * R[0]) + r_Delta_pi * (-log(pi[ss]^-1 * pi[-1]) + log(pi[ss]^-1 * pi[0])) + r_Delta_y * (-log(Y[ss]^-1 * Y[-1]) + log(Y[ss]^-1 * Y[0]) + log(Y_f[ss]^-1 * Y_f[-1]) - log(Y_f[ss]^-1 * Y_f[0])) + rho * log(R[ss]^-1 * R[-1]) + (1 - rho) * (log(pi_obj[0]) + r_pi * (-log(pi_obj[0]) + log(pi[ss]^-1 * pi[-1])) + r_Y * (log(Y[ss]^-1 * Y[0]) - log(Y_f[ss]^-1 * Y_f[0]))) = 0
    # #         -C_f[0] - I_f[0] + Pi_ws_f[0] - T_f[0] + Y_f[0] + L_s_f[0] * W_disutil_f[0] - L_f[0] * W_f[0] - psi^-1 * r_k_f[ss] * K_f[-1] * (-1 + exp(psi * (-1 + z_f[0]))) = 0
    # #         epsilon_b[0] * (K[-1] * r_k[0] - r_k[ss] * K[-1] * exp(psi * (-1 + z[0]))) * (C[0] - h * C[-1])^(-sigma_c) = 0
    # #         epsilon_b[0] * (K_f[-1] * r_k_f[0] - r_k_f[ss] * K_f[-1] * exp(psi * (-1 + z_f[0]))) * (C_f[0] - h * C_f[-1])^(-sigma_c) = 0
    # #     end


    # #     @parameters Smets_Wouters_2003 begin  
    # #         calibr_pi_obj | 1 = pi_obj[ss]
    # #         calibr_pi | pi[ss] = pi_obj[ss]
    # #         Phi | (Y_s[ss] + Phi) / Y_s[ss] = 1.408
    # #         # lambda_p | .6 = C_f[ss] / Y_f[ss]
    # #         # lambda_w | L[ss] = .33
    # #         G_bar | .18 = G[ss] / Y[ss]

    # #         lambda_p = .368
    # #         # G_bar = .362
    # #         lambda_w = 0.5
    # #         # Phi = .819

    # #         alpha = 0.3
    # #         beta = 0.99
    # #         gamma_w = 0.763
    # #         gamma_p = 0.469
    # #         h = 0.573
    # #         omega = 1
    # #         psi = 0.169
    # #         r_pi = 1.684
    # #         r_Y = 0.099
    # #         r_Delta_pi = 0.14
    # #         r_Delta_y = 0.159
    # #         rho = 0.961
    # #         rho_b = 0.855
    # #         rho_L = 0.889
    # #         rho_I = 0.927
    # #         rho_a = 0.823
    # #         rho_G = 0.949
    # #         rho_pi_bar = 0.924
    # #         sigma_c = 1.353
    # #         sigma_l = 2.4
    # #         tau = 0.025
    # #         varphi = 6.771
    # #         xi_w = 0.737
    # #         xi_p = 0.908

    # #     end


    # #     solve!(Smets_Wouters_2003, symbolic_SS = false)

    # #     # get_steady_state(Smets_Wouters_2003)

    # #     @test get_steady_state(Smets_Wouters_2003)[1] ≈ [     1.20465991441435
    # #     1.204659917151701
    # #     0.3613478048030788
    # #     0.3613478048030788
    # #     0.4414800855444218
    # #     0.4414800896382151
    # #    17.659203422264238
    # #    17.65920357698873
    # #     1.2889457095271066
    # #     1.2889457096070582
    # #     1.2889457095307755
    # #     1.2889457098239414
    # #     1.0000000000366498
    # #     0.5400259611608715
    # #     0.48211013259048446
    # #     1.00000000000172
    # #     1.000000000127065
    # #     1.0101010101010102
    # #     1.0101010101010102
    # #     0.3613478047907606
    # #     0.3613478048030788
    # #     -427.92495898028676
    # #     -427.9249587468684
    # #        1.1221034247496608
    # #        0.7480689524616317
    # #        1.122103428477167
    # #        1.1221034282377538
    # #        2.0074878047372287
    # #        2.00748781245403
    # #        2.007487804732286
    # #        2.0074878121606647
    # #        1.0
    # #        1.0
    # #        1.0
    # #        1.0
    # #        1.0
    # #        8.766762166589194
    # #        8.766762166588967
    # #       48.8212791635492
    # #       35.68806956399776
    # #       0.730994152045567
    # #       0.7309941520886629
    # #       1.0
    # #       1.0000000000028464
    # #       1.0
    # #       1.0
    # #       1.0
    # #       2.4582240979093846
    # #       2.4582240906598867
    # #       0.03510101014899653
    # #       0.035101010136073356
    # #       1.1221034247485961
    # #       1.0000000000000178
    # #       0.9999999999583465]


    end


    @testset verbose = true "First order perturbation" begin
        # Numerical test with calibration targets
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
        end


        @parameters RBC_CME verbose = true begin
            alpha | k[ss] / (4 * y[ss]) = cap_share
            cap_share = 1.66
            # alpha = .157

            beta | R[ss] = R_ss # beta needs to enter into function: block in order to solve
            R_ss = 1.0035
            # beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            delta | delta * k[ss] / y[ss] = I_K_ratio #check why this doesnt solve for y; because delta is not recognised as a free parameter here.
            I_K_ratio = .15
            # delta = .0226

            Pibar | Pi[ss] = Pi_ss
            Pi_ss = 1.0025
            # Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end

        get_solution(RBC_CME)
        @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]], [    0.0          0.0068
                                                                    6.73489e-6   0.000168887
                                                                    1.01124e-5   0.000253583
                                                                    -0.000365783  0.00217203
                                                                    -0.00070019   0.00749279
                                                                    0.0          0.00966482
                                                                    0.005        0.0], atol = 1e-6)

        get_solution(RBC_CME, parameters = :I_K_ratio => .1)
        @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]],[  0.0          0.0068
            3.42408e-6   0.000111417
            5.14124e-6   0.000167292
        -0.000196196  0.00190741
        -0.000430554  0.0066164
            0.0          0.00852381
            0.005        0.0], atol = 1e-6)

        get_solution(RBC_CME, parameters = :cap_share => 1.5)
        @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]],[ 0.0          0.0068
        4.00629e-6   0.000118171
        6.01543e-6   0.000177434
    -0.000207089  0.00201698
    -0.00041124   0.00639229
        0.0          0.00840927
        0.005        0.0], atol = 1e-6)

        RBC_CME = nothing
    end





    # @testset verbose = true "First order: linear time iteration" begin
    #     # Numerical test with calibration targets
    #     @model RBC_CME begin
    #         y[0]=A[0]*k[-1]^alpha
    #         1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    #         1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    #         R[0] * beta =(Pi[0]/Pibar)^phi_pi
    #         A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    #         z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    #         A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
    #     end


    #     @parameters RBC_CME verbose = true begin
    #         alpha | k[ss] / (4 * y[ss]) = cap_share
    #         cap_share = 1.66
    #         # alpha = .157

    #         beta | R[ss] = R_ss # beta needs to enter into function: block in order to solve
    #         R_ss = 1.0035
    #         # beta = .999

    #         # delta | c[ss]/y[ss] = 1 - I_K_ratio
    #         delta | delta * k[ss] / y[ss] = I_K_ratio #check why this doesnt solve for y; because delta is not recognised as a free parameter here.
    #         I_K_ratio = .15
    #         # delta = .0226

    #         Pibar | Pi[ss] = Pi_ss
    #         Pi_ss = 1.0025
    #         # Pibar = 1.0008

    #         phi_pi = 1.5
    #         rhoz = .9
    #         std_eps = .0068
    #         rho_z_delta = .9
    #         std_z_delta = .005
    #     end

    #     get_solution(RBC_CME, algorithm = :linear_time_iteration)

    #     @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]], [    0.0          0.0068
    #                                                                 6.73489e-6   0.000168887
    #                                                                 1.01124e-5   0.000253583
    #                                                                 -0.000365783  0.00217203
    #                                                                 -0.00070019   0.00749279
    #                                                                 0.0          0.00966482
    #                                                                 0.005        0.0], atol = 1e-6)


    #     get_solution(RBC_CME, algorithm = :linear_time_iteration, parameters = :I_K_ratio => .1)

    #     @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]],[  0.0          0.0068
    #         3.42408e-6   0.000111417
    #         5.14124e-6   0.000167292
    #     -0.000196196  0.00190741
    #     -0.000430554  0.0066164
    #         0.0          0.00852381
    #         0.005        0.0], atol = 1e-6)


    #     get_solution(RBC_CME, algorithm = :linear_time_iteration, parameters = :cap_share => 1.5)

    #     @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]],[ 0.0          0.0068
    #     4.00629e-6   0.000118171
    #     6.01543e-6   0.000177434
    # -0.000207089  0.00201698
    # -0.00041124   0.00639229
    #     0.0          0.00840927
    #     0.005        0.0], atol = 1e-6)

    #     RBC_CME = nothing
    # end





    @testset verbose = true "First order: doubling" begin
        # Numerical test with calibration targets
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
        end


        @parameters RBC_CME verbose = true begin
            alpha | k[ss] / (4 * y[ss]) = cap_share
            cap_share = 1.66
            # alpha = .157

            beta | R[ss] = R_ss # beta needs to enter into function: block in order to solve
            R_ss = 1.0035
            # beta = .999

            # delta | c[ss]/y[ss] = 1 - I_K_ratio
            delta | delta * k[ss] / y[ss] = I_K_ratio #check why this doesnt solve for y; because delta is not recognised as a free parameter here.
            I_K_ratio = .15
            # delta = .0226

            Pibar | Pi[ss] = Pi_ss
            Pi_ss = 1.0025
            # Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
        end

        get_solution(RBC_CME, quadratic_matrix_equation_algorithm = :doubling)

        @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]], [    0.0          0.0068
                                                                    6.73489e-6   0.000168887
                                                                    1.01124e-5   0.000253583
                                                                    -0.000365783  0.00217203
                                                                    -0.00070019   0.00749279
                                                                    0.0          0.00966482
                                                                    0.005        0.0], atol = 1e-6)


        get_solution(RBC_CME, quadratic_matrix_equation_algorithm = :doubling, parameters = :I_K_ratio => .1)

        @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]],[  0.0          0.0068
            3.42408e-6   0.000111417
            5.14124e-6   0.000167292
        -0.000196196  0.00190741
        -0.000430554  0.0066164
            0.0          0.00852381
            0.005        0.0], atol = 1e-6)


        get_solution(RBC_CME, quadratic_matrix_equation_algorithm = :doubling, parameters = :cap_share => 1.5)

        @test isapprox(RBC_CME.solution.perturbation.first_order.solution_matrix[:,[(end-RBC_CME.timings.nExo+1):end...]],[ 0.0          0.0068
        4.00629e-6   0.000118171
        6.01543e-6   0.000177434
    -0.000207089  0.00201698
    -0.00041124   0.00639229
        0.0          0.00840927
        0.005        0.0], atol = 1e-6)

        RBC_CME = nothing
    end





    @testset verbose = true "Plotting" begin
        # Symbolic test with calibration targets
        @model RBC_CME begin
            y[0]=A[0]*k[-1]^alpha
            1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
            1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
            R[0] * beta =(Pi[0]/Pibar)^phi_pi
            # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
            A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
            z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
            # z[0]=rhoz*z[-1]+std_eps*eps_z[x]
            # A[0]=exp(z[0])
            A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
            # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
        end


        @parameters RBC_CME verbose = true begin
            alpha | k[ss] / (4 * y[ss]) = cap_share
            cap_share = 1.66
            # alpha = .157

            beta | R[ss] = R_ss
            R_ss = 1.0035
            # beta = .999

            delta | c[ss]/y[ss] = 1 - I_K_ratio
            # delta | delta * k[ss] / y[ss] = I_K_ratio # this doesnt solve symbolically
            I_K_ratio = .15
            # delta = .0226

            Pibar | Pi[ss] = Pi_ss
            Pi_ss = 1.0025
            # Pibar = 1.0008

            phi_pi = 1.5
            rhoz = .9
            std_eps = .0068
            rho_z_delta = .9
            std_z_delta = .005
            
            # cap_share > 0
            # R_ss > 0
            # Pi_ss > 0
            # I_K_ratio > 0 

            # 0 < alpha < 1 
            # 0 < beta < 1
            # 0 < delta < 1
            # 0 < Pibar
            # 0 <= rhoz < 1
            # phi_pi > 0

            # 0 < A < 1
            # 0 < k < 50
            # 0 < y < 10
            # 0 < c < 10
        end
        plot_irf(RBC_CME)
        @test true

        RBC_CME = nothing
    end
    GC.gc()
end