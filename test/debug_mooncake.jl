using DifferentiationInterface
import Mooncake
using MacroModelling
using Zygote
using FiniteDifferences
import LinearAlgebra as ℒ
using Random
using BenchmarkTools
# Note on Mooncake performance:
# Mooncake.jl has an inherent "time to first gradient" compilation cost.
# The prepare_gradient step can take 30-90+ seconds, but subsequent gradient
# evaluations are very fast (~0.01-0.1s). This is a fundamental design aspect
# of Mooncake and not a bug. For estimation workflows with many gradient 
# evaluations, the prep cost is quickly amortized.

# Define a simple RBC model for testing gradients
@model RBC_AD begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC_AD begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

params = copy(RBC_AD.parameter_values)

func_to_diff(x) = MacroModelling.get_NSSS_and_parameters(RBC_AD, x)[1]
mooncake_backend = DifferentiationInterface.AutoMooncake(; config=nothing)
mooncake_prep = DifferentiationInterface.prepare_jacobian(func_to_diff, mooncake_backend, params)
mooncake_jac = DifferentiationInterface.jacobian(func_to_diff, mooncake_prep, mooncake_backend, params)
# @profview mooncake_prep = DifferentiationInterface.prepare_jacobian(func_to_diff, mooncake_backend, params)
@profview for i in 1:100 mooncake_jac = DifferentiationInterface.jacobian(func_to_diff, mooncake_prep, mooncake_backend, params) end

zygote_backend = DifferentiationInterface.AutoZygote()
zygote_prep = DifferentiationInterface.prepare_jacobian(func_to_diff, zygote_backend, params)
zygote_jac = DifferentiationInterface.jacobian(func_to_diff, zygote_prep, zygote_backend, params)
@benchmark zygote_jac = DifferentiationInterface.jacobian(func_to_diff, zygote_prep, zygote_backend, params)

@profview for i in 1:100 zygote_jac = DifferentiationInterface.jacobian(func_to_diff, zygote_prep, zygote_backend, params) end

    # function nsss_sum(p)
    #     SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(RBC_AD, p)
    #     return sum(SS_and_pars)
    # end

    # params = copy(RBC_AD.parameter_values)
    # result = compare_gradients(nsss_sum, params)


# Helper function to compare gradients across backends
# The retry loop (max 100 iterations) for finite differences is needed because numerical
# differentiation can occasionally produce NaN/Inf values due to floating point issues
function compare_gradients(func, params; rtol=1e-5, test_fin_diff=true, max_fin_diff_retries=100, print_times=true)
    # Get Zygote gradient and measure time
    zygote_time = @elapsed zygote_grad = Zygote.gradient(func, params)[1]
    
    if print_times
        println("  Zygote time: $(round(zygote_time, digits=3))s")
        # println("  Mooncake prep time: $(round(mooncake_prep_time, digits=3))s (one-time cost), eval time: $(round(mooncake_time, digits=3))s")
    end
    # Get Mooncake gradient via DifferentiationInterface and measure time
    # Note: Mooncake prep time is expected to be slower than Zygote (30-90s vs 1-10s)
    # but subsequent gradient evaluations are very fast
    mooncake_backend = DifferentiationInterface.AutoMooncake(; config=nothing)
    mooncake_prep_time = @elapsed mooncake_prep = DifferentiationInterface.prepare_gradient(func, mooncake_backend, params)
    mooncake_time = @elapsed mooncake_grad = DifferentiationInterface.gradient(func, mooncake_prep, mooncake_backend, params)
    
    if print_times
        # println("  Zygote time: $(round(zygote_time, digits=3))s")
        println("  Mooncake prep time: $(round(mooncake_prep_time, digits=3))s (one-time cost), eval time: $(round(mooncake_time, digits=3))s")
    end

    # Get FiniteDifferences gradient (with retry for numerical stability)
    fin_grad = nothing
    if test_fin_diff
        for i in 1:max_fin_diff_retries
            local fin_grad_try = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), func, params)[1]
            if isfinite(ℒ.norm(fin_grad_try))
                fin_grad = fin_grad_try
                break
            end
        end
    end
    
    return (zygote_grad=zygote_grad, mooncake_grad=mooncake_grad, fin_grad=fin_grad,
            zygote_time=zygote_time, mooncake_prep_time=mooncake_prep_time, mooncake_time=mooncake_time,
            rtol=rtol)
end

# Test 1: NSSS gradient comparison
@testset "NSSS gradient comparison" begin
    function nsss_sum(p)
        SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(RBC_AD, p)
        return sum(SS_and_pars)
    end

    params = copy(RBC_AD.parameter_values)
    result = compare_gradients(nsss_sum, params)
    
    @test isapprox(result.zygote_grad, result.mooncake_grad, rtol=result.rtol)
    if result.fin_grad !== nothing
        @test isapprox(result.zygote_grad, result.fin_grad, rtol=result.rtol)
        @test isapprox(result.mooncake_grad, result.fin_grad, rtol=result.rtol)
    end
end

# Test 2: Jacobian gradient comparison
@testset "Jacobian gradient comparison" begin
    function jacobian_sum(p)
        SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(RBC_AD, p)
        jac = MacroModelling.calculate_jacobian(p, SS_and_pars, RBC_AD)
        return sum(jac)
    end

    params = copy(RBC_AD.parameter_values)
    result = compare_gradients(jacobian_sum, params)
    
    @test isapprox(result.zygote_grad, result.mooncake_grad, rtol=result.rtol)
    if result.fin_grad !== nothing
        @test isapprox(result.zygote_grad, result.fin_grad, rtol=result.rtol)
    end
end
