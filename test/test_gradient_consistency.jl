# Test gradient consistency between FiniteDiff, Zygote, and Mooncake
# for get_loglikelihood and get_NSSS_and_parameters_with_jacobian

using MacroModelling
using Test
using Random
import LinearAlgebra as ℒ
using AxisKeys
using DifferentiationInterface
import FiniteDifferences
import Zygote
import Mooncake

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
observables = [:c]  # Must have same number of observables as shocks
data = simulated_data(observables, :, :simulate)

@testset "Gradient Consistency Tests" begin
    params = copy(RBC.parameter_values)
    
    @testset "get_NSSS_and_parameters_with_jacobian" begin
        # Test that the function returns the correct jacobian
        SS_and_pars, jvp, solution_error, iters = get_NSSS_and_parameters_with_jacobian(RBC, params)
        
        @test solution_error < 1e-8
        @test size(jvp, 1) == length(SS_and_pars)
        @test size(jvp, 2) == length(params)
        
        # Compare jacobian against FiniteDiff
        f_ss = x -> get_NSSS_and_parameters_with_jacobian(RBC, x)[1]  # Returns just SS_and_pars
        jac_finitediff = FiniteDifferences.jacobian(
            FiniteDifferences.central_fdm(5, 1),
            f_ss,
            params
        )[1]
        
        # Check that jacobian from function matches FiniteDiff
        rel_diff = ℒ.norm(jvp - jac_finitediff) / max(ℒ.norm(jac_finitediff), 1e-10)
        println("\nJacobian comparison:")
        println("  Direct jvp vs FiniteDiff relative diff: $rel_diff")
        @test rel_diff < 1e-6
        
        # Test gradient of sum(SS_and_pars) using Zygote
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
        
        # Test Mooncake gradient
        grad_mooncake = DifferentiationInterface.gradient(
            f_sum,
            DifferentiationInterface.AutoMooncake(; config = nothing),
            params
        )
        
        rel_diff_mooncake = ℒ.norm(grad_mooncake - grad_zygote) / max(ℒ.norm(grad_zygote), 1e-10)
        println("  Mooncake vs Zygote gradient relative diff: $rel_diff_mooncake")
        @test rel_diff_mooncake < 1e-10
    end
    
    # Test configurations for get_loglikelihood
    test_configs = [
        (algorithm = :first_order, filter = :kalman, name = "first_order + kalman"),
        (algorithm = :first_order, filter = :inversion, name = "first_order + inversion"),
    ]
    
    @testset "get_loglikelihood" begin
        for config in test_configs
            @testset "$(config.name)" begin
                # Create wrapper function for this config
                f_config = x -> get_loglikelihood(RBC, data, x, 
                                                   algorithm = config.algorithm, 
                                                   filter = config.filter)
                
                # Compute gradients
                grad_finitediff = nothing
                for i in 1:10  # Retry loop for numerical stability
                    grad_finitediff = FiniteDifferences.grad(
                        FiniteDifferences.central_fdm(5, 1), 
                        f_config, 
                        params
                    )[1]
                    if isfinite(ℒ.norm(grad_finitediff))
                        break
                    end
                end
                
                grad_zygote = Zygote.gradient(f_config, params)[1]
                
                # Mooncake gradient via DifferentiationInterface
                grad_mooncake = DifferentiationInterface.gradient(
                    f_config, 
                    DifferentiationInterface.AutoMooncake(; config = nothing), 
                    params
                )
                
                println("\n$(config.name):")
                println("  FiniteDiff: $grad_finitediff")
                println("  Zygote:     $grad_zygote")
                println("  Mooncake:   $grad_mooncake")
                
                # Test Zygote vs FiniteDiff
                if isfinite(ℒ.norm(grad_finitediff)) && isfinite(ℒ.norm(grad_zygote))
                    rel_diff_zygote = ℒ.norm(grad_zygote - grad_finitediff) / max(ℒ.norm(grad_finitediff), 1e-10)
                    println("  Zygote vs FiniteDiff relative diff: $rel_diff_zygote")
                    @test rel_diff_zygote < 1e-4  # Allow some tolerance for numerical differences
                end
                
                # Test Mooncake vs Zygote (should be identical since they use the same rrule)
                if isfinite(ℒ.norm(grad_zygote)) && isfinite(ℒ.norm(grad_mooncake))
                    rel_diff_mooncake = ℒ.norm(grad_mooncake - grad_zygote) / max(ℒ.norm(grad_zygote), 1e-10)
                    println("  Mooncake vs Zygote relative diff: $rel_diff_mooncake")
                    @test rel_diff_mooncake < 1e-10  # Should be nearly identical
                end
            end
        end
    end
end

