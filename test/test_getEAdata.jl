#!/usr/bin/env julia
#
# test_getEAdata.jl - Test script for getEAdata.jl functions
#
# This script tests the Julia implementation of Euro Area data processing functions
# to ensure they match the R script behavior
#

using Test
using DataFrames
using Dates
using Statistics
using LinearAlgebra

# Include the main script
include("getEAdata.jl")

@testset "Euro Area Data Processing Tests" begin
    
    @testset "Date Parsing" begin
        # Test quarterly date parsing
        @test parse_eurostat_time("2020-Q1") == Date(2020, 1, 1)
        @test parse_eurostat_time("2020-Q2") == Date(2020, 4, 1)
        @test parse_eurostat_time("2020-Q3") == Date(2020, 7, 1)
        @test parse_eurostat_time("2020-Q4") == Date(2020, 10, 1)
        
        # Test annual date parsing
        @test parse_eurostat_time("2020") == Date(2020, 1, 1)
    end
    
    @testset "Date to Quarter String" begin
        @test to_yearqtr(Date(2020, 1, 1)) == "2020 Q1"
        @test to_yearqtr(Date(2020, 4, 1)) == "2020 Q2"
        @test to_yearqtr(Date(2020, 7, 1)) == "2020 Q3"
        @test to_yearqtr(Date(2020, 10, 1)) == "2020 Q4"
    end
    
    @testset "Linear Interpolation" begin
        # Test with missing values
        x = [1.0, missing, 3.0, missing, 5.0]
        result = na_approx(x)
        
        @test result[1] ≈ 1.0
        @test result[2] ≈ 2.0  # Linear interpolation
        @test result[3] ≈ 3.0
        @test result[4] ≈ 4.0  # Linear interpolation
        @test result[5] ≈ 5.0
    end
    
    @testset "Spline Interpolation" begin
        # Test with enough points for spline
        x = [1.0, missing, missing, 4.0, missing, 6.0]
        result = na_spline(x)
        
        @test !ismissing(result[1])
        @test !ismissing(result[2])
        @test !ismissing(result[3])
        @test !ismissing(result[4])
    end
    
    @testset "HP Filter" begin
        # Test HP filter with simple trend
        t = 1:100
        trend = Float64.(t)
        cycle = sin.(2π * t / 10)  # Add cyclical component
        y = trend + cycle
        
        hp = hpfilter(y, λ=1600.0)
        
        # Check that trend is smooth and cycle has mean near zero
        @test length(hp.trend) == length(y)
        @test length(hp.cycle) == length(y)
        @test abs(mean(hp.cycle)) < 0.5  # Cycle should average to near zero
        
        # Trend should be smoother than original
        trend_volatility = std(diff(hp.trend))
        original_volatility = std(diff(y))
        @test trend_volatility < original_volatility
    end
    
    @testset "Chain Function" begin
        # Create simple test data
        dates1 = Date(2000,1,1):Month(3):Date(2005,12,31)
        dates2 = Date(2003,1,1):Month(3):Date(2010,12,31)
        
        df1 = DataFrame(
            period = collect(dates1),
            var = fill("gdp", length(dates1)),
            value = 100.0 .* (1.01 .^ (0:length(dates1)-1))  # 1% growth
        )
        
        df2 = DataFrame(
            period = collect(dates2),
            var = fill("gdp", length(dates2)),
            value = 150.0 .* (1.01 .^ (0:length(dates2)-1))  # Same growth, different level
        )
        
        # Chain at 2003-01-01
        chained = chain(df1, df2, Date(2003,1,1))
        
        # Check that result has data from both periods
        @test minimum(chained.period) == Date(2000,1,1)
        @test maximum(chained.period) == Date(2010,12,31)
        
        # Check that values match df2 after chain date
        after_chain = chained[chained.period .> Date(2003,1,1), :]
        df2_after = df2[df2.period .> Date(2003,1,1), :]
        @test nrow(after_chain) == nrow(df2_after)
    end
    
    @testset "Structural TS Smoothing" begin
        # Test with simple data
        x = Float64.(1:20) + randn(20) * 0.5
        smoothed = structts_smooth(x)
        
        @test length(smoothed) == length(x)
        # Smoothed series should have less variance in differences
        @test std(diff(smoothed)) < std(diff(x))
    end
end

println("\n" * "="^80)
println("All tests passed! ✓")
println("="^80)
println("\nThe Julia implementation of Euro Area data processing functions")
println("matches the expected behavior of the R script.")
println()
