using DynarePreprocessor_jll
using JSON

Base.@kwdef struct CommandLineOptions
    compilemodule::Bool = true
end


@noinline function parseJSON(modfilename::String)
    modelstring::String =
        open(f -> read(f, String), modfilename * "/model/json/modfile.json")
    modeljson = JSON.parse(modelstring)
    return modeljson
end


function make_context(modeljson, modfilename, commandlineoptions)
    @debug "$(now()): get symbol_table"
    (symboltable, endo_nbr, exo_nbr, exo_det_nbr, param_nbr, orig_endo_nbr, aux_vars) =
        get_symbol_table(modeljson)
    @debug "$(now()): get Modelfile"
    modfileinfo = ModFileInfo(modfilename)
    check_function_files!(modfileinfo, modfilename)
    @debug "$(now()): get model"
    model = get_model(
        modfilename,
        modfileinfo,
        modeljson["model_info"],
        commandlineoptions,
        endo_nbr,
        exo_nbr,
        exo_det_nbr,
        param_nbr,
        orig_endo_nbr,
        aux_vars,
        Vector{Int64}(modeljson["dynamic_g1_sparse_rowval"]),
        Vector{Int64}(modeljson["dynamic_g1_sparse_colptr"]),
        Vector{Int64}(modeljson["static_g1_sparse_rowval"]),
        Vector{Int64}(modeljson["static_g1_sparse_colptr"]),
        Vector{Int64}(modeljson["dynamic_tmp_nbr"]),
        Vector{Int64}(modeljson["static_tmp_nbr"]),
    )
    varobs = get_varobs(modeljson)
    @debug "$(now()): make_container"
    global context = make_containers(
        modfileinfo,
        modfilename,
        endo_nbr,
        exo_nbr,
        exo_det_nbr,
        param_nbr,
        model,
        symboltable,
        varobs,
        commandlineoptions,
    )
    get_mcps!(context.models[1].mcps, modeljson["model"])
    return context
end

function parser(modfilename::String, commandlineoptions::CommandLineOptions)
    @debug "$(now()): Start $(nameof(var"#self#"))"

    modeljson = parseJSON(modfilename)
    context = make_context(modeljson, modfilename, commandlineoptions)
    context.work.analytical_steadystate_variables = DFunctions.load_model_functions(modfilename)
    if haskey(modeljson, "statements")
        parse_statements!(context, modeljson["statements"])
    end
    @info "$(now()): End $(nameof(var"#self#"))"
    return context
end



macro dynare(modfile_arg::String, args...)
    @info "Dynare version: $(module_version(Dynare))"
    modname = get_modname(modfile_arg)
    @info "$(now()): Starting @dynare $modfile_arg"
    arglist = []
    compilemodule = false
    preprocessing = true
    for (i, a) in enumerate(args)
        if a == "nocompile"
            compilemodule = false
        elseif a == "nopreprocessing"
            preprocessing = false
        else
            push!(arglist, a)
        end
    end
    if preprocessing
        modfilename = modname * ".mod"
        dynare_preprocess(modfilename, arglist)
    end
    @info "$(now()): End of preprocessing"
    options = CommandLineOptions(compilemodule)
    context = parser(modname, options)
    return context
end

function get_modname(modfilename::String)
    if occursin(r"\.mod$", modfilename)
        modname::String = modfilename[1:length(modfilename)-4]
    else
        modname = modfilename
    end
    return modname
end

function dynare_preprocess(modfilename::String, args::Vector{Any})
    dynare_args = [basename(modfilename), "language=julia", "json=compute"]
    offset = 0
    for a in args
        astring = string(a)
        if !occursin(r"^json=", astring)
            push!(dynare_args, astring)
        end
    end
    println(dynare_args)
    run_dynare(modfilename, dynare_args)
    println("")
end

function run_dynare(modfilename::String, dynare_args::Vector{String})
    @info "Dynare preprocessor version: $(module_version(DynarePreprocessor_jll))"
    directory = dirname(modfilename)
    if length(directory) > 0
        current_directory = pwd()
        cd(directory)
    end

    dynare_preprocessor_path = dynare_preprocessor()
    run(`$dynare_preprocessor_path $dynare_args`)

    if length(directory) > 0
        cd(current_directory)
    end
end