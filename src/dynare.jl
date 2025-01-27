using DynarePreprocessor_jll
using JSON


"""
$(SIGNATURES)
Reads in a `dynare` .mod-file, adapts the syntax, tries to capture parameter definitions, and writes a julia file in the same folder containing the model equations and parameters in `MacroModelling.jl` syntax. This function is not guaranteed to produce working code. It's purpose is to make it easier to port a model from `dynare` to `MacroModelling.jl`. 

The recommended workflow is to use this function to translate a .mod-file, and then adapt the output so that it runs and corresponds to the input.

Note that this function copies the .mod-file to a temporary folder and executes it there. All references within that .mod-file are therefore not valid (because those filesare not copied) and must be made copied into the .mod-file.

# Arguments
- `path_to_mod_file` [Type: `AbstractString`]: path including filename of the .mod-file to be translated
"""
function translate_mod_file(path_to_mod_file::AbstractString)
    directory = dirname(path_to_mod_file)

    directory_2 = replace(basename(path_to_mod_file), r"\.mod$" => "")

    tmp = tempdir()

    mkpath(tmp * "/" * directory_2)

    cp(path_to_mod_file, tmp * "/" * basename(path_to_mod_file), force = true)

    args = [tmp * "/" * basename(path_to_mod_file), "language=julia", "json=compute"]

    if length(directory) > 0
        current_directory = pwd()
        cd(directory)
    end

    dynare_preprocessor_path = dynare_preprocessor()

    function parse_model()
        try
            run(pipeline(`$dynare_preprocessor_path $args`, stdout = "log.txt"))
        catch
            error("Failed to parse the model. Dynare preprocessor output:\n\n", read("log.txt", String))
        end
    end

    cd(parse_model, tmp)

    son = JSON.parsefile(tmp * "/" * directory_2 * "/model/json/modfile.json")

    @assert son isa Dict "Failed to parse the model."

    vars = [i["name"] for i in son["endogenous"]]
    shocks = [i["name"] for i in son["exogenous"]]
    eqs_orig = [i["lhs"] * " = " * i["rhs"] for i in son["model"]]

    eqs = []
    for eq in eqs_orig
        eq = replace(eq, r"(\w+)\((-?\d+)\)" => s"\1[\2]")
        for v in vars
            eq = replace(eq, Regex("(?<!\\b)\\($(v)\\)") => v * "[ss]")
            eq = replace(eq, Regex("\\b$(v)\\b(?!\\[)") => v * "[0]")
        end
        for x in shocks
            eq = replace(eq, Regex("\\b$(x)\\b") => x * "[x]")
        end
        eq = replace(
            eq,
            r"\[0\]\[1\]" => "[1]",
            r"\[0\]\[-1\]" => "[-1]",
            r"\*" => " * ",
            r"\+" => " + ",
            r"(?<!\[|\^\()\-" => " - ",
            r"\/" => " / ",
            r"\^" => " ^ ",
        )
        push!(eqs, eq)
    end

    pars = []
    for s in son["statements"]
        if s["statementName"] == "native"
            if contains(s["string"], "=")
                if contains(s["string"], "options_")
                    break
                else
                    push!(pars, replace(s["string"], ";" => ""))
                end
            elseif contains(s["string"], r"^\#")
                continue
            end
        elseif s["statementName"] == "param_init"
            push!(pars, s["name"] * " = " * s["value"])
        else
            break
        end
    end

    open(directory_2 * ".jl", "w") do io
        println(io, "using MacroModelling\n")
        println(io, "@model " * directory_2 * " begin")
        [println(io, "\t" * eq * "\n") for eq in eqs]
        println(io, "end\n\n")
        println(io, "@parameters " * directory_2 * " begin")
        [println(io, "\t" * par * "\n") for par in pars]
        println(io, "end\n")
    end

    # rm(directory_2, recursive = true)

    if length(directory) > 0
        cd(current_directory)
    end

    @info "Created " * directory * "/" * directory_2 * ".jl"

    @warn "This is an experimental function. Manual adjustments are most likely necessary. Please check before running the model."
end

"""
See [`translate_mod_file`](@ref)
"""
translate_dynare_file = translate_mod_file

"""
See [`translate_mod_file`](@ref)
"""
import_model = translate_mod_file

"""
See [`translate_mod_file`](@ref)
"""
import_dynare = translate_mod_file


"""
$(SIGNATURES)
Writes a `dynare` .mod-file in the current working directory. This function is not guaranteed to produce working code. It's purpose is to make it easier to port a model from `MacroModelling.jl` to `dynare`. 

The recommended workflow is to use this function to write a .mod-file, and then adapt the output so that it runs and corresponds to the input.

# Arguments
- $MODEL®
"""
function write_mod_file(m::ℳ)
    NSSS = get_SS(m, derivatives = false)

    index_in_name = NSSS.keys isa Base.RefValue{Vector{String}}

    open(m.model_name * ".mod", "w") do io
        println(io, "var ")
        [print(io, translate_symbol_to_ascii(v) * " ") for v in setdiff(m.vars_in_ss_equations, m.➕_vars)]

        println(io, ";\n\nvarexo ")
        [print(io, translate_symbol_to_ascii(e) * " ") for e in m.exo]

        println(io, ";\n\nparameters ")
        [print(io, translate_symbol_to_ascii(p) * " ") for p in m.parameters_in_equations]


        println(io, ";\n\n% Parameter definitions:")
        for (i, p) in enumerate(m.parameters)
            println(io, "\t" * translate_symbol_to_ascii(p) * "\t=\t" * string(m.parameter_values[i]) * ";")
        end

        for p in m.calibration_equations_parameters
            println(io, "\t" * translate_symbol_to_ascii(p) * "\t=\t" * string(NSSS(index_in_name ? replace(string(p), "◖" => "{", "◗" => "}") : p)) * ";") 
        end

        [
            println(io, "\t" * replace(
                    string(translate_expression_to_ascii(e)),
                    r"\b(\d+(\.\d+)?)([_\p{L}]\w*)\b" => s"\1*\3",
                    r"norminv(?=\()" => s"norminvcdf",
                    r"qnorm(?=\()" => s"norminvcdf",
                    r"pnorm(?=\()" => s"normcdf",
                    r"dnorm(?=\()" => s"normpdf",
                ) * ";") for 
                e in m.calibration_equations_no_var
        ]

        println(io, "\nmodel;")
        [
            println(
                io,
                "\t" *
                replace(
                    string(translate_expression_to_ascii(e)),
                    r"\[(-?\d+)\]" => s"(\1)",
                    r"(\w+)\[(ss|stst|steady|steadystate|steady_state){1}\]" =>
                        s"STEADY_STATE(\1)",
                    r"(\w+)\[(x|ex|exo|exogenous){1}\]" => s"\1",
                    r"(\w+)\[(x|ex|exo|exogenous){1}(\s*(\-|\+)\s*(\d{1}))\]" =>
                        s"\1(\4\5)",
                    r"norminv(?=\()" => s"norminvcdf",
                    r"qnorm(?=\()" => s"norminvcdf",
                    r"pnorm(?=\()" => s"normcdf",
                    r"dnorm(?=\()" => s"normpdf",
                ) *
                ";\n",
            ) for e in m.original_equations
        ]

        println(io, "end;\n\nshocks;")
        [println(io, "var\t" * translate_symbol_to_ascii(e) * "\t=\t1;") for e in m.exo]

        println(io, "end;\n\ninitval;")
        for v in setdiff(m.vars_in_ss_equations, m.➕_vars)
            print(io, "\t" * translate_symbol_to_ascii(v) * "\t=\t" * string(NSSS(index_in_name ? replace(string(v), "◖" => "{", "◗" => "}") : v)) * ";\n") 
        end

        println(io, "end;\n\nstoch_simul(order = 1, irf = 40);")
    end

    @info "Created " * m.model_name * ".mod"

    # @warn "This is an experimental function. Manual adjustments are most likely necessary. Please check before running the model."
end

"""
See [`write_mod_file`](@ref)
"""
export_dynare = write_mod_file

"""
See [`write_mod_file`](@ref)
"""
export_to_dynare = write_mod_file

"""
See [`write_mod_file`](@ref)
"""
export_mod_file = write_mod_file

"""
See [`write_mod_file`](@ref)
"""
write_dynare_file = write_mod_file

"""
See [`write_mod_file`](@ref)
"""
write_to_dynare_file = write_mod_file

"""
See [`write_mod_file`](@ref)
"""
write_to_dynare = write_mod_file

"""
See [`write_mod_file`](@ref)
"""
export_model = write_mod_file
