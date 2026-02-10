#!/usr/bin/env julia
# Compare SS values between refactored branch and main branch for all non-OBC models

using Revise
using MacroModelling

# Map model files to their model variable names
models = [
    ("Aguiar_Gopinath_2007.jl", :Aguiar_Gopinath_2007),
    ("Ascari_Sbordone_2014.jl", :Ascari_Sbordone_2014),
    ("Backus_Kehoe_Kydland_1992.jl", :Backus_Kehoe_Kydland_1992),
    ("Baxter_King_1993.jl", :Baxter_King_1993),
    ("Caldara_et_al_2012.jl", :Caldara_et_al_2012),
    ("FS2000.jl", :FS2000),
    ("Gali_2015_chapter_3_nonlinear.jl", :Gali_2015_chapter_3_nonlinear),
    ("Gali_Monacelli_2005_CITR.jl", :Gali_Monacelli_2005_CITR),
    ("Ghironi_Melitz_2005.jl", :Ghironi_Melitz_2005),
    ("GNSS_2010.jl", :GNSS_2010),
    ("Guerrieri_Iacoviello_2017.jl", :Guerrieri_Iacoviello_2017),
    ("Iacoviello_2005_linear.jl", :Iacoviello_2005_linear),
    ("Ireland_2004.jl", :Ireland_2004),
    ("JQ_2012_RBC.jl", :JQ_2012_RBC),
    ("NAWM_EAUS_2008.jl", :NAWM_EAUS_2008),
    ("QUEST3_2009.jl", :QUEST3_2009),
    ("RBC_baseline.jl", :RBC_baseline),
    ("SGU_2003_debt_premium.jl", :SGU_2003_debt_premium),
    ("Smets_Wouters_2003.jl", :Smets_Wouters_2003),
    ("Smets_Wouters_2007.jl", :Smets_Wouters_2007),
    ("Smets_Wouters_2007_linear.jl", :Smets_Wouters_2007_linear),
]

models_dir = joinpath(@__DIR__, "models")

results = Dict{String, Any}()
passed = String[]
failed = String[]
errored = String[]

for (mf, model_sym) in models
    model_name = replace(mf, ".jl" => "")
    print("Testing $model_name... ")
    flush(stdout)
    try
        include(joinpath(models_dir, mf))
        model = eval(model_sym)
        
        ss = MacroModelling.get_steady_state(model, verbose=false)
        
        if all(isfinite, ss.data)
            results[model_name] = Dict(
                "names" => [string(n) for n in ss.axis1.val],
                "values" => Float64.(ss.data[:, 1])
            )
            println("OK ($(length(ss.data)) values)")
            push!(passed, model_name)
        else
            println("FAIL: Non-finite values in SS")
            push!(failed, model_name)
        end
    catch e
        println("ERROR: ", sprint(showerror, e))
        push!(errored, model_name)
    end
end

println("\n" * "="^60)
println("Results: $(length(passed)) passed, $(length(failed)) failed, $(length(errored)) errors")
println("="^60)

if !isempty(failed)
    println("\nFailed models:")
    for m in failed; println("  - $m"); end
end
if !isempty(errored)
    println("\nError models:")
    for m in errored; println("  - $m"); end
end

# Save results
using Serialization
serialize("refactored_ss_results.jls", results)
println("\nResults saved to refactored_ss_results.jls")
