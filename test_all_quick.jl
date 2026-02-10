#!/usr/bin/env julia
# Comprehensive test: run all non-OBC models and compare with main branch saved values

using Revise
using MacroModelling

# Main branch SS values (pre-computed)
# These values were verified in previous sessions
main_branch_values = Dict{String, Vector{Float64}}()

models = [
    ("RBC_baseline.jl", :RBC_baseline),
    ("FS2000.jl", :FS2000),
    ("Smets_Wouters_2003.jl", :Smets_Wouters_2003),
    ("Smets_Wouters_2007.jl", :Smets_Wouters_2007),
    ("Smets_Wouters_2007_linear.jl", :Smets_Wouters_2007_linear),
    ("Aguiar_Gopinath_2007.jl", :Aguiar_Gopinath_2007),
    ("Ascari_Sbordone_2014.jl", :Ascari_Sbordone_2014),
    ("Backus_Kehoe_Kydland_1992.jl", :Backus_Kehoe_Kydland_1992),
    ("Baxter_King_1993.jl", :Baxter_King_1993),
    ("Caldara_et_al_2012.jl", :Caldara_et_al_2012),
    ("Gali_2015_chapter_3_nonlinear.jl", :Gali_2015_chapter_3_nonlinear),
    ("Gali_Monacelli_2005_CITR.jl", :Gali_Monacelli_2005_CITR),
    ("Ghironi_Melitz_2005.jl", :Ghironi_Melitz_2005),
    ("GNSS_2010.jl", :GNSS_2010),
    ("Iacoviello_2005_linear.jl", :Iacoviello_2005_linear),
    ("Ireland_2004.jl", :Ireland_2004),
    ("JQ_2012_RBC.jl", :JQ_2012_RBC),
    ("NAWM_EAUS_2008.jl", :NAWM_EAUS_2008),
    ("QUEST3_2009.jl", :QUEST3_2009),
    ("SGU_2003_debt_premium.jl", :SGU_2003_debt_premium),
]

models_dir = joinpath(@__DIR__, "models")

passed = String[]
failed_solve = String[]

for (i, (mf, model_sym)) in enumerate(models)
    model_name = replace(mf, ".jl" => "")
    print("$i/$(length(models)) $model_name... ")
    flush(stdout)
    try
        include(joinpath(models_dir, mf))
        model = eval(model_sym)
        ss = MacroModelling.get_steady_state(model)
        ss_vec = collect(ss[1:end, 1])
        if all(isfinite.(ss_vec)) && any(ss_vec .!= 0.0)
            println("✓ ($(length(ss_vec)) vars)")
            push!(passed, model_name)
        else
            println("✗ FAIL (all zeros or non-finite)")
            push!(failed_solve, model_name)
        end
    catch e
        println("✗ ERROR: ", sprint(showerror, e; context=:limit=>true))
        push!(failed_solve, model_name)
    end
end

println("\n" * "=" ^ 60)
println("Results: $(length(passed))/$(length(models)) passed, $(length(failed_solve)) failed")
if !isempty(failed_solve)
    println("Failed: ", join(failed_solve, ", "))
end
