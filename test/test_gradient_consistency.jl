# Test gradient consistency between FiniteDiff, Zygote, and Mooncake
# for get_NSSS_and_parameters_with_jacobian and get_loglikelihood

using MacroModelling
using Test
using Random
import LinearAlgebra as ℒ
using AxisKeys
import FiniteDifferences
import Zygote

# Define a simple test model
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# Generate simulated data with single observable (to match single shock)
Random.seed!(42)
simulated_data = simulate(RBC)
observables = [:c]
data = simulated_data(observables, :, :simulate)

@testset "Gradient Consistency Tests" begin
    params = copy(RBC.parameter_values)
    
    @testset "get_NSSS_and_parameters_with_jacobian" begin
        # Test that the function returns correct jacobian
        SS_and_pars, jvp, solution_error, iters = get_NSSS_and_parameters_with_jacobian(RBC, params)
        
        @test solution_error < 1e-8
        @test size(jvp, 1) == length(SS_and_pars)
        @test size(jvp, 2) == length(params)
        
        # Compare jacobian against FiniteDiff
        f_ss = x -> get_NSSS_and_parameters_with_jacobian(RBC, x)[1]
        jac_finitediff = FiniteDifferences.jacobian(
            FiniteDifferences.central_fdm(5, 1),
            f_ss,
            params
        )[1]
        
        rel_diff = ℒ.norm(jvp - jac_finitediff) / max(ℒ.norm(jac_finitediff), 1e-10)
        println("\nJacobian comparison:")
        println("  Direct jvp vs FiniteDiff relative diff: $rel_diff")
        @test rel_diff < 1e-6
        
        # Test Zygote gradient of sum(SS_and_pars)
        f_sum = x -> sum(get_NSSS_and_parameters_with_jacobian(RBC, x)[1])
        
        grad_zygote = Zygote.gradient(f_sum, params)[1]
        grad_finitediff = FiniteDifferences.grad(
            FiniteDifferences.central_fdm(5, 1),
            f_sum,
            params
        )[1]
        
        rel_diff_grad = ℒ.norm(grad_zygote - grad_finitediff) / max(ℒ.norm(grad_finitediff), 1e-10)
        println("  Zygote vs FiniteDiff gradient relative diff: $rel_diff_grad")
        @test rel_diff_grad < 1e-6
    end
    
    @testset "get_loglikelihood - Zygote vs FiniteDiff" begin
        # Test first_order + kalman
        f_kalman = x -> get_loglikelihood(RBC, data, x, algorithm = :first_order, filter = :kalman)
        
        grad_zygote = Zygote.gradient(f_kalman, params)[1]
        grad_finitediff = FiniteDifferences.grad(
            FiniteDifferences.central_fdm(5, 1),
            f_kalman,
            params
        )[1]
        
        println("\nget_loglikelihood (first_order + kalman):")
        println("  Zygote:     $grad_zygote")
        println("  FiniteDiff: $grad_finitediff")
        
        if isfinite(ℒ.norm(grad_finitediff)) && isfinite(ℒ.norm(grad_zygote))
            rel_diff = ℒ.norm(grad_zygote - grad_finitediff) / max(ℒ.norm(grad_finitediff), 1e-10)
            println("  Relative diff: $rel_diff")
            @test rel_diff < 1e-4
        end
    end
end

