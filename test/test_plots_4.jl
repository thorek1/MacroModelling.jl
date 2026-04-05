using Test
using MacroModelling
import MacroModelling: clear_solution_caches!
using Random
import SpecialFunctions: erfcinv
using AxisKeys, SparseArrays
import Mooncake, FiniteDifferences, Zygote
import DifferentiationInterface, ADTypes
import StatsPlots
import LinearAlgebra as ℒ

include("functionality_tests.jl")

plots = true
Random.seed!(1)

include("models/Caldara_et_al_2012_estim.jl")

@testset verbose = true "RBC_CME with calibration equations, parameter definitions, special functions, variables in steady state, and leads/lag > 1 on endogenous and exogenous variables" begin
    include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")
    functionality_test(m, Caldara_et_al_2012_estim, plots = plots)
    
    observables = [:R, :k]

    Random.seed!(1)
    simulated_data = simulate(m)

    get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), ADTypes.AutoMooncake(config = nothing), m.parameter_values)
    zygote_back_grad = Zygote.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)[1]

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
            @test isapprox(zygote_back_grad, fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
end
m = nothing
GC.gc()

@testset verbose = true "RBC_CME with calibration equations, parameter definitions, special functions, variables in steady state, and leads/lag > 1 on endogenous and exogenous variables numerical SS" begin
    include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")

    functionality_test(m, Caldara_et_al_2012_estim, plots = plots)
    
    observables = [:R, :k]

    Random.seed!(1)
    simulated_data = simulate(m)

    get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), ADTypes.AutoMooncake(config = nothing), m.parameter_values)
    zygote_back_grad = Zygote.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)[1]

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
            @test isapprox(zygote_back_grad, fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
end
m = nothing
GC.gc()

@testset verbose = true "RBC_CME with calibration equations, parameter definitions, and special functions" begin
    include("models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl")
    functionality_test(m, Caldara_et_al_2012_estim, plots = plots)

    observables = [:R, :k]

    Random.seed!(1)
    simulated_data = simulate(m)

    get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), ADTypes.AutoMooncake(config = nothing), m.parameter_values)
    zygote_back_grad = Zygote.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)[1]

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x, verbose = true), m.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
            @test isapprox(zygote_back_grad, fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
end
m = nothing
GC.gc()

@testset verbose = true "RBC_CME with calibration equations and parameter definitions" begin
    include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
    functionality_test(m, Caldara_et_al_2012_estim, plots = plots)

    observables = [:R, :k]

    Random.seed!(1)
    simulated_data = simulate(m)

    get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), ADTypes.AutoMooncake(config = nothing), m.parameter_values)
    zygote_back_grad = Zygote.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)[1]

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
            @test isapprox(zygote_back_grad, fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
end
m = nothing
GC.gc()

@testset verbose = true "RBC_CME with calibration equations" begin
    include("models/RBC_CME_calibration_equations.jl")
    functionality_test(m, Caldara_et_al_2012_estim, plots = plots)
    
    observables = [:R, :k]

    Random.seed!(1)
    simulated_data = simulate(m)

    get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), ADTypes.AutoMooncake(config = nothing), m.parameter_values)
    zygote_back_grad = Zygote.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)[1]

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
            @test isapprox(zygote_back_grad, fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
end
m = nothing
GC.gc()

@testset verbose = true "RBC_CME" begin
    include("models/RBC_CME.jl")
    functionality_test(m, Caldara_et_al_2012_estim, plots = plots)

    observables = [:R, :k]

    Random.seed!(1)
    simulated_data = simulate(m)

    get_loglikelihood(m, simulated_data(observables, :, :simulate), m.parameter_values)

    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), ADTypes.AutoMooncake(config = nothing), m.parameter_values)
    zygote_back_grad = Zygote.gradient(x -> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)[1]

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(m, simulated_data(observables, :, :simulate), x), m.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
            @test isapprox(zygote_back_grad, fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad, fin_grad[1], rtol = 1e-6)
end
m = nothing
GC.gc()
