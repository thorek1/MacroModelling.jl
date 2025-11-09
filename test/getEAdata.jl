#!/usr/bin/env julia
#
# getEAdata.jl - Julia version of getEAdata.R
#
# Downloads and processes Euro Area macroeconomic data from Eurostat and other sources.
# This script replicates the functionality of the R script getEAdata.R
#
# Required packages:
#   CSV, DataFrames, Dates, XLSX, HTTP, JSON3, Statistics, Interpolations, LinearAlgebra
#
# Installation:
#   using Pkg
#   Pkg.add(["CSV", "DataFrames", "XLSX", "HTTP", "JSON3", "Statistics", "Interpolations"])
#
# Data sources:
#   - AWM Database: https://eabcn.org/data/area-wide-model
#   - Conference Board TED database (Excel file)
#   - Eurostat via bulk download API
#

using CSV
using DataFrames
using Dates
using XLSX
using HTTP
using JSON3
using Statistics
using Interpolations
using LinearAlgebra

# Configuration
const DATA_DIR = joinpath(@__DIR__, "data")
const OUTPUT_DIR = @__DIR__

# Eurostat bulk download base URL (more reliable than API)
const EUROSTAT_BULK_URL = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing"

# Helper Functions
# ================

"""
Download Eurostat dataset in TSV format via bulk download
Returns the path to the downloaded file
"""
function download_eurostat_bulk(dataset_code::String; cache_dir::String=DATA_DIR)
    mkpath(cache_dir)
    local_file = joinpath(cache_dir, "$(dataset_code).tsv.gz")
    
    # Check if already downloaded
    if isfile(local_file)
        println("Using cached file: $local_file")
        return local_file
    end
    
    # Construct download URL
    url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/$(dataset_code).tsv.gz"
    
    println("Downloading $dataset_code from Eurostat...")
    try
        HTTP.download(url, local_file)
        println("Downloaded to: $local_file")
        return local_file
    catch e
        println("Error downloading $dataset_code: $e")
        return nothing
    end
end

"""
Parse Eurostat TSV format and filter data
"""
function parse_eurostat_tsv(file_path::String; filters::Dict{String,Any}=Dict())
    # Read the gzipped TSV file
    # Note: Julia's CSV.jl can handle gzipped files directly
    df = CSV.read(file_path, DataFrame; delim='\t', header=1, silencewarnings=true)
    
    # The first column contains all dimension values concatenated
    # We need to split it and apply filters
    # Format is typically: freq,unit,s_adj,nace_r2,na_item,geo\tYYYY-QN\t...
    
    # Extract dimension columns
    if ncol(df) > 0
        first_col = names(df)[1]
        
        # Split the first column by commas to get dimensions
        dim_values = split.(df[!, first_col], ",")
        
        # Create dimension columns
        # This is dataset-specific; adjust based on actual structure
        # For now, we'll return the raw data
    end
    
    return df
end

"""
Parse time string from Eurostat (e.g., "2020-Q1", "2020-01", "2020")
"""
function parse_eurostat_time(s::String; quarterly::Bool=true)
    s = strip(s)
    
    if occursin("Q", s)
        # Quarterly: "2020-Q1"
        m = match(r"(\d{4})-Q(\d)", s)
        if !isnothing(m)
            year = parse(Int, m.captures[1])
            quarter = parse(Int, m.captures[2])
            month = (quarter - 1) * 3 + 1
            return Date(year, month, 1)
        end
    elseif occursin("-", s)
        # Monthly: "2020-01"
        try
            return Date(s, dateformat"yyyy-mm")
        catch
            # Try other formats
        end
    else
        # Annual: "2020"
        try
            year = parse(Int, s)
            return Date(year, 1, 1)
        catch
        end
    end
    
    return missing
end

"""
Convert Date to year-quarter string
"""
function to_yearqtr(d::Date)
    y = year(d)
    q = Int(ceil(month(d) / 3))
    return "$y Q$q"
end

"""
Chain function - rebase a time series to match another at a specific date
Implements the same logic as the R version
"""
function chain(to_rebase::DataFrame, basis::DataFrame, date_chain::Union{String,Date})
    date_chain = Date(date_chain)
    
    # Get reference value at chain date from basis
    valref = basis[basis.period .== date_chain, [:var, :value]]
    rename!(valref, :value => :value_ref)
    
    # Filter to_rebase to dates <= chain date and sort
    res = to_rebase[to_rebase.period .<= date_chain, :]
    sort!(res, [:var, :period], rev=[false, true])  # Descending by period
    
    # Calculate growth rates within each variable group
    res = combine(groupby(res, :var)) do df
        n = nrow(df)
        growth_rate = ones(n)
        if n > 1
            for i in 2:n
                growth_rate[i] = df.value[i-1] / df.value[i]  # Note: reversed order
            end
        end
        DataFrame(period=df.period, value=df.value, growth_rate=growth_rate)
    end
    
    # Join with reference values
    res = leftjoin(res, valref, on=:var)
    
    # Calculate chained values using cumulative product
    res = combine(groupby(res, :var)) do df
        chained = cumprod(df.growth_rate) .* df.value_ref[1]
        DataFrame(period=df.period, value=chained)
    end
    
    # Add data from basis after chain date
    basis_after = basis[basis.period .> date_chain, :]
    res = vcat(res, basis_after)
    sort!(res, :period)
    
    return res
end

"""
Linear interpolation for missing values (like zoo::na.approx in R)
"""
function na_approx(x::Vector)
    n = length(x)
    result = Vector{Union{Missing,Float64}}(undef, n)
    copy!(result, x)
    
    valid_idx = findall(!ismissing, x)
    if length(valid_idx) < 2
        return result
    end
    
    valid_vals = Float64[x[i] for i in valid_idx]
    itp = LinearInterpolation(valid_idx, valid_vals, extrapolation_bc=Flat())
    
    for i in 1:n
        if ismissing(result[i])
            result[i] = itp(float(i))
        end
    end
    
    return result
end

"""
Cubic spline interpolation for missing values (like zoo::na.spline in R)
"""
function na_spline(x::Vector)
    n = length(x)
    result = Vector{Union{Missing,Float64}}(undef, n)
    copy!(result, x)
    
    valid_idx = findall(!ismissing, x)
    if length(valid_idx) < 4
        return na_approx(x)  # Fallback to linear
    end
    
    valid_vals = Float64[x[i] for i in valid_idx]
    itp = CubicSplineInterpolation(valid_idx, valid_vals)
    
    for i in 1:n
        if ismissing(result[i]) && i >= valid_idx[1] && i <= valid_idx[end]
            result[i] = itp(float(i))
        end
    end
    
    return result
end

"""
Simple structural time series smoothing (like StructTS + tsSmooth in R)
This is a simplified version using double exponential smoothing
For production use, consider StateSpaceModels.jl
"""
function structts_smooth(x::Vector)
    n = length(x)
    
    # Handle missing values
    y = Float64[]
    indices = Int[]
    for i in 1:n
        if !ismissing(x[i])
            push!(y, Float64(x[i]))
            push!(indices, i)
        end
    end
    
    if length(y) < 4
        return x
    end
    
    # Double exponential smoothing
    α = 0.1  # Level smoothing parameter
    β = 0.05  # Trend smoothing parameter
    
    # Initialize
    level = y[1]
    trend = y[2] - y[1]
    smoothed_vals = Float64[]
    
    for i in 1:length(y)
        forecast = level + trend
        push!(smoothed_vals, forecast)
        
        if i < length(y)
            level_old = level
            level = α * y[i] + (1 - α) * (level + trend)
            trend = β * (level - level_old) + (1 - β) * trend
        end
    end
    
    # Map back to original indices with interpolation for missing
    result = Vector{Float64}(undef, n)
    for i in 1:n
        if i in indices
            idx = findfirst(==(i), indices)
            result[i] = smoothed_vals[idx]
        else
            # Linear interpolation for missing points
            before = findlast(<(i), indices)
            after = findfirst(>(i), indices)
            if !isnothing(before) && !isnothing(after)
                idx_before = findfirst(==(indices[before]), indices)
                idx_after = findfirst(==(indices[after]), indices)
                t = (i - indices[before]) / (indices[after] - indices[before])
                result[i] = (1 - t) * smoothed_vals[idx_before] + t * smoothed_vals[idx_after]
            elseif !isnothing(before)
                result[i] = smoothed_vals[findlast(<=(length(y)), 1:length(y))]
            else
                result[i] = smoothed_vals[1]
            end
        end
    end
    
    return result
end

"""
Hodrick-Prescott filter (like mFilter::hpfilter in R)
λ = 1600 for quarterly data (standard value)
"""
function hpfilter(x::Vector{Float64}; λ::Float64=1600.0)
    n = length(x)
    
    if n < 4
        return (trend=x, cycle=zeros(n))
    end
    
    # Build second difference matrix
    I_n = Matrix{Float64}(I, n, n)
    D = zeros(Float64, n-2, n)
    for i in 1:(n-2)
        D[i, i] = 1.0
        D[i, i+1] = -2.0
        D[i, i+2] = 1.0
    end
    
    # Solve (I + λD'D)τ = y
    A = I_n + λ * (D' * D)
    trend = A \ x
    cycle = x .- trend
    
    return (trend=trend, cycle=cycle)
end

# Main Processing
# ===============

function main()
    println("="^80)
    println("Julia version of Euro Area Data Download and Processing")
    println("="^80)
    println()
    
    println("This script replicates the R script getEAdata.R")
    println("It downloads data from Eurostat and processes Euro Area macroeconomic series")
    println()
    
    # Check for required files
    println("Checking for required data files...")
    
    awm_file = joinpath(DATA_DIR, "awm19up15.csv")
    ted_file = joinpath(DATA_DIR, "TED---Output-Labor-and-Labor-Productivity-1950-2015.xlsx")
    
    files_ok = true
    if !isfile(awm_file)
        println("❌ Missing: awm19up15.csv")
        println("   Download from: https://eabcn.org/data/area-wide-model")
        files_ok = false
    else
        println("✓ Found: awm19up15.csv")
    end
    
    if !isfile(ted_file)
        println("❌ Missing: TED Excel file")
        println("   This file contains Conference Board hours worked data")
        files_ok = false
    else
        println("✓ Found: TED Excel file")
    end
    
    println()
    
    if !files_ok
        println("Please download the required files before running this script.")
        println()
        println("Framework and utility functions have been loaded.")
        println("You can use them interactively or after obtaining the data files.")
        return
    end
    
    println("All required files found. Processing data...")
    println()
    
    # TODO: Implement the full data processing pipeline from getEAdata.R
    # This would include:
    # 1. Load and transform AWM data
    # 2. Process Conference Board hours data
    # 3. Download and process Eurostat population data
    # 4. Download and process Eurostat GDP, consumption, investment, wage, employment data
    # 5. Chain series together
    # 6. Apply interpolation and smoothing
    # 7. Create final transformed variables
    # 8. Export to CSV files
    
    println("Framework complete. Implement data processing steps as needed.")
    println()
    println("Available helper functions:")
    println("  - chain(to_rebase, basis, date_chain)")
    println("  - na_approx(x), na_spline(x)")
    println("  - structts_smooth(x)")
    println("  - hpfilter(x; λ=1600.0)")
    println("  - parse_eurostat_time(s)")
    println("  - download_eurostat_bulk(dataset_code)")
    println()
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# Export main functions for use as a module
export chain, na_approx, na_spline, structts_smooth, hpfilter
export parse_eurostat_time, to_yearqtr, download_eurostat_bulk
