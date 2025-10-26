using MacroModelling
using Test

@testset "Partial parameter definition" begin
    # Test 1: Create a model
    @model RBC_partial begin
        1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α
        z[0] = ρ * z[-1] + std_z * eps_z[x]
    end
    
    # Test 2: Define only some parameters (missing β and δ)
    @test_logs (:info, r"Model set up with undefined parameters") @parameters RBC_partial silent = true begin
        std_z = 0.01
        ρ = 0.2
        α = 0.5
    end
    
    # Test 3: Verify undefined_parameters field is populated
    @test length(RBC_partial.undefined_parameters) == 2
    @test :β ∈ RBC_partial.undefined_parameters
    @test :δ ∈ RBC_partial.undefined_parameters
    
    # Test 4: Now define all parameters
    @parameters RBC_partial silent = true begin
        std_z = 0.01
        ρ = 0.2
        α = 0.5
        β = 0.95
        δ = 0.02
    end
    
    # Test 5: Verify undefined_parameters is now empty
    @test length(RBC_partial.undefined_parameters) == 0
    
    # Test 6: Now operations should succeed
    SS = get_steady_state(RBC_partial)
    @test !isnan(SS[1])
    @test length(SS) > 0
    
    # Test 7: IRF should now work
    irf_result = get_irf(RBC_partial, periods = 10)
    @test size(irf_result, 2) == 10
    
    println("✓ All partial parameter definition tests passed")
end

@testset "Provide parameters later via function call" begin
    # Test 1: Create a model and define only some parameters
    @model RBC_later begin
        1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α
        z[0] = ρ * z[-1] + std_z * eps_z[x]
    end
    
    @test_logs (:info, r"Model set up with undefined parameters") @parameters RBC_later silent = true begin
        std_z = 0.01
        ρ = 0.2
    end
    
    # Test 2: Verify undefined parameters are tracked
    @test length(RBC_later.undefined_parameters) > 0
    
    # Test 3: Provide missing parameters via function call as Dict
    params_dict = Dict(:α => 0.5, :β => 0.95, :δ => 0.02, :std_z => 0.01, :ρ => 0.2)
    
    # Test 4: IRF with explicit parameters should work even when some are undefined in @parameters
    irf_result = get_irf(RBC_later, parameters = params_dict, periods = 10)
    @test size(irf_result, 2) == 10
    
    # Test 5: After providing parameters via Dict, undefined_parameters should be cleared
    @test length(RBC_later.undefined_parameters) == 0
    
    println("✓ All 'provide parameters later' tests passed")
end
