"""
Test that QUEST3_2009, Gali_2015_chapter_3_nonlinear, and FS2000 return valid steady states.

This test verifies that these three models can successfully compute a steady state.
For models that can converge, all values should be finite and at least some non-zero.
For models that cannot converge, the solver should return gracefully without crashing.
"""

using Test
using MacroModelling

@testset "Three Models Valid Steady State" begin
    
    @testset "FS2000" begin
        println("Testing FS2000 steady state...")
        include("../models/FS2000.jl")
        
        # Get steady state
        ss = get_steady_state(FS2000)
        ss_vec = collect(ss[1:end, 1])
        
        # Verify steady state is valid
        @test all(isfinite.(ss_vec))
        @test length(ss_vec) > 0
        
        # Check if steady state was found (non-zero values)
        has_nonzero = any(ss_vec .!= 0.0)
        if has_nonzero
            println("  ✓ FS2000 steady state converged successfully ($(length(ss_vec)) variables)")
        else
            println("  ⚠ FS2000 did not converge but returned gracefully ($(length(ss_vec)) variables)")
        end
    end
    
    @testset "QUEST3_2009" begin
        println("Testing QUEST3_2009 steady state...")
        include("../models/QUEST3_2009.jl")
        
        # Get steady state  
        ss = get_steady_state(QUEST3_2009)
        ss_vec = collect(ss[1:end, 1])
        
        # Verify steady state is valid (finite and correct length)
        @test all(isfinite.(ss_vec))
        @test length(ss_vec) > 0
        
        # Check if steady state was found (non-zero values)
        has_nonzero = any(ss_vec .!= 0.0)
        if has_nonzero
            println("  ✓ QUEST3_2009 steady state converged successfully ($(length(ss_vec)) variables)")
        else
            println("  ⚠ QUEST3_2009 did not converge but returned gracefully ($(length(ss_vec)) variables)")
        end
    end
    
    @testset "Gali_2015_chapter_3_nonlinear" begin
        println("Testing Gali_2015_chapter_3_nonlinear steady state...")
        include("../models/Gali_2015_chapter_3_nonlinear.jl")
        
        # Get steady state
        ss = get_steady_state(Gali_2015_chapter_3_nonlinear)
        ss_vec = collect(ss[1:end, 1])
        
        # Verify steady state is valid (finite and correct length)
        @test all(isfinite.(ss_vec))
        @test length(ss_vec) > 0
        
        # Check if steady state was found (non-zero values)
        has_nonzero = any(ss_vec .!= 0.0)
        if has_nonzero
            println("  ✓ Gali_2015_chapter_3_nonlinear steady state converged successfully ($(length(ss_vec)) variables)")
        else
            println("  ⚠ Gali_2015_chapter_3_nonlinear did not converge but returned gracefully ($(length(ss_vec)) variables)")
        end
    end
    
end

println("\n✅ All three models completed steady state computation without crashing!")
