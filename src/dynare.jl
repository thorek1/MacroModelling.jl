using DynarePreprocessor_jll
using JSON

function translate_mod_file(name::AbstractString)
    args = [basename(name), "language=julia", "json=compute"]
    
    directory = dirname(name)

    directory_2 = replace(basename(name),r"\.mod$"=>"")

    if length(directory) > 0
        current_directory = pwd()
        cd(directory)
    end
    
    dynare_preprocessor_path = dynare_preprocessor()

    mkpath(directory_2)

    run(pipeline(`$dynare_preprocessor_path $args`, stdout = directory_2 * "/log.txt"))

    son = JSON.parsefile(directory_2 * "/model/json/modfile.json");

    vars = [i["name"] for i in son["endogenous"]];
    shocks = [i["name"] for i in son["exogenous"]];
    eqs_orig = [i["lhs"] * " = " * i["rhs"] for i in son["model"]];
    
    eqs = []
    for eq in eqs_orig
        eq = replace(eq, r"(\w+)\((-?\d+)\)" => s"\1[\2]")
        for v in vars
            eq = replace(eq, Regex("\\b$(v)\\b") => v * "[0]")
        end
        for x in shocks
            eq = replace(eq, Regex("\\b$(x)\\b") => x * "[x]")
        end
        eq = replace(eq, r"\[0\]\[1\]" => "[1]", 
                            r"\[0\]\[-1\]" => "[-1]", 
                            r"\*" => " * ", 
                            r"\+" => " + ", 
                            r"(?<!\[|\^\()\-" => " - ", 
                            r"\/" => " / ", 
                            r"\^" => " ^ ")
        push!(eqs,eq)
    end
    
    pars = []
    for s in son["statements"]
        if s["statementName"] == "native" && contains(s["string"], "=")
            if contains(s["string"], "options_")
                break
            else
                push!(pars, replace(s["string"],";"=>""))
            end
        elseif s["statementName"] == "param_init"
            push!(pars, s["name"] * " = " * s["value"])
        else 
            break
        end
    end
    
    open(directory_2 * ".jl", "w") do io
        println(io,"using MacroModelling\n")
        println(io,"@model "*directory_2*" begin")
        [println(io,"\t"*eq*"\n") for eq in eqs]
        println(io,"end\n\n")
        println(io,"@parameters "*directory_2*" begin")
        [println(io,"\t"*par*"\n") for par in pars]
        println(io,"end\n")
    end

    rm(directory_2, recursive = true)

    if length(directory) > 0
        cd(current_directory)
    end

    @info "Created " * directory * "/" * directory_2 * ".jl"

    @warn "This is an experimental function. Parameters are not guaranteed to be translated correctly. Please check before running the model."
end

translate_dynare_file   =   translate_mod_file
import_model            =   translate_mod_file
import_dynare           =   translate_mod_file