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

    current_directory = pwd()

    if length(directory) > 0
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

    @static if isdefined(JSON, :Object)
        @assert son isa Dict || son isa JSON.Object "Failed to parse the model."
    else
        @assert son isa Dict "Failed to parse the model."
    end

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
        # Dynare 7 emits steady-state references as STEADY_STATE(var), while
        # older releases emit the indexed form that MacroModelling accepts.
        eq = replace(eq, r"\bSTEADY_STATE\(([^()]*)\[0\]\)" => s"\1[ss]")
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
function write_mod_file(𝓂::ℳ; order::Int = 1, pruning::Bool = false, irf_periods::Int = 40)
    NSSS = get_SS(𝓂, derivatives = false)

    index_in_name = NSSS.keys isa Base.RefValue{Vector{String}}

    open(𝓂.model_name * ".mod", "w") do io
        println(io, "var ")
        [print(io, translate_symbol_to_ascii(v) * " ") for v in setdiff(𝓂.constants.post_model_macro.vars_in_ss_equations, 𝓂.constants.post_model_macro.➕_vars)]

        println(io, ";\n\nvarexo ")
        [print(io, translate_symbol_to_ascii(e) * " ") for e in 𝓂.constants.post_model_macro.exo]

        println(io, ";\n\nparameters ")
        [print(io, translate_symbol_to_ascii(p) * " ") for p in 𝓂.constants.post_model_macro.parameters_in_equations]


        println(io, ";\n\n% Parameter definitions:")
        for (i, p) in enumerate(𝓂.constants.post_complete_parameters.parameters)
            println(io, "\t" * translate_symbol_to_ascii(p) * "\t=\t" * string(𝓂.parameter_values[i]) * ";")
        end

        for p in 𝓂.equations.calibration_parameters
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
                e in 𝓂.equations.calibration_no_var
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
            ) for e in 𝓂.equations.original
        ]

        println(io, "end;\n\nshocks;")
        [println(io, "var\t" * translate_symbol_to_ascii(e) * "\t=\t1;") for e in 𝓂.constants.post_model_macro.exo]

        println(io, "end;\n\ninitval;")
        for v in setdiff(𝓂.constants.post_model_macro.vars_in_ss_equations, 𝓂.constants.post_model_macro.➕_vars)
            print(io, "\t" * translate_symbol_to_ascii(v) * "\t=\t" * string(NSSS(index_in_name ? replace(string(v), "◖" => "{", "◗" => "}") : v)) * ";\n") 
        end

        stoch_opts = "order = $order, irf = $irf_periods"
        if pruning
            stoch_opts *= ", pruning"
        end
        if order > 2
            stoch_opts *= ", k_order_solver"
        end
        println(io, "end;\n\nstoch_simul($stoch_opts);")
    end

    @info "Created " * 𝓂.model_name * ".mod"

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




function translate_symbol_to_ascii(x::Symbol)
    ss = Unicode.normalize(replace(string(x),  "◖" => "__", "◗" => "__"), :NFD)

    outstr = ""

    for i in ss
        out = REPL.symbol_latex(string(i))[2:end]
        if out == ""
            outstr *= string(i)
        else
            outstr *= replace(out,  
                        r"\!" => s"_",
                        r"\(" => s"_",
                        r"\)" => s"_",
                        r"\^" => s"_",
                        r"\_\^" => s"_",
                        r"\+" => s"plus",
                        r"\-" => s"minus",
                        r"\*" => s"times")
            if i != ss[end]
                outstr *= "_"
            end
        end
    end

    return outstr
end


function translate_expression_to_ascii(exp::Expr)
    postwalk(x -> 
                x isa Symbol ?
                    begin
                        x_tmp = translate_symbol_to_ascii(x)

                        if x_tmp == string(x)
                            x
                        else
                            Symbol(x_tmp)
                        end
                    end :
                x,
    exp)
end
