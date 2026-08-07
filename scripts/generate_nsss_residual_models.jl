using MacroModelling
using SparseArrays
using LinearAlgebra

import MacroModelling.BlockTriangularForm

const project_root = normpath(joinpath(@__DIR__, ".."))
const model_directory = joinpath(project_root, "models")
const output_directory = joinpath(project_root, "nsss_residuals")

const normcdf = MacroModelling.SymPyWorkspace.normcdf
const pnorm = MacroModelling.SymPyWorkspace.pnorm
const normpdf = MacroModelling.SymPyWorkspace.normpdf
const dnorm = MacroModelling.SymPyWorkspace.dnorm
const normlogpdf = MacroModelling.SymPyWorkspace.normlogpdf
const norminvcdf = MacroModelling.SymPyWorkspace.norminvcdf
const norminv = MacroModelling.SymPyWorkspace.norminv
const qnorm = MacroModelling.SymPyWorkspace.qnorm
const erfcinv = MacroModelling.SymPyWorkspace.erfcinv
const erfc = MacroModelling.SymPyWorkspace.erfc
const Max = max
const Min = min

function collect_expression_symbols!(symbols::Set{Symbol}, expression)
    if expression isa Symbol
        push!(symbols, expression)
    elseif expression isa Expr
        for argument in expression.args
            collect_expression_symbols!(symbols, argument)
        end
    end
    return symbols
end

function expression_symbols(expression)
    collect_expression_symbols!(Set{Symbol}(), expression)
end

function display_symbol(symbol::Symbol)
    replace(string(symbol), "◖" => "{", "◗" => "}")
end

function display_symbols(symbols)
    [display_symbol(Symbol(symbol)) for symbol in symbols]
end

function replace_expression(expression, replacements::Dict{Symbol, Any})
    if expression isa Symbol
        return get(replacements, expression, expression)
    elseif expression isa Expr
        return Expr(expression.head,
                    [replace_expression(argument, replacements) for argument in expression.args]...)
    end
    return expression
end

function expression_source(expression)
    string(expression)
end

function parameter_definition_order(model, complete_parameter_names)
    definitions = Dict{Symbol, Any}()
    for equation in model.equations.calibration_no_var
        definitions[equation.args[1]] = equation.args[2]
    end

    derived_names = Set(keys(definitions))
    ordered = Symbol[]
    remaining = Set(keys(definitions))
    while !isempty(remaining)
        progress = false
        for name in sort(collect(remaining), by = string)
            dependencies = intersect(expression_symbols(definitions[name]), derived_names)
            if isempty(setdiff(dependencies, Set(ordered)))
                push!(ordered, name)
                delete!(remaining, name)
                progress = true
            end
        end
        progress || error("Cyclic parameter definitions in $(model.model_name): $(collect(remaining))")
    end
    return [(name, definitions[name]) for name in ordered]
end

function complete_parameter_names(model)
    free_names = collect(model.constants.post_complete_parameters.parameters)
    calibration_names = Set(model.equations.calibration_parameters)
    equation_names = [name for name in model.constants.post_model_macro.parameters_in_equations
                      if !(name in calibration_names)]
    definition_names = [equation.args[1] for equation in model.equations.calibration_no_var]
    unique(vcat(free_names, equation_names, definition_names))
end

function make_blocks(model, auxiliary_equations, calibration_equations)
    post_model = model.constants.post_model_macro
    calibration_parameters = collect(model.equations.calibration_parameters)
    unknowns = collect(union(post_model.vars_in_ss_equations, calibration_parameters))
    equations = vcat(auxiliary_equations, calibration_equations)

    incidence = spzeros(Int, length(unknowns), length(equations))
    unknown_indices = Dict(name => index for (index, name) in enumerate(unknowns))
    for (equation_index, equation) in enumerate(equations)
        for symbol in intersect(expression_symbols(equation), Set(unknowns))
            incidence[unknown_indices[symbol], equation_index] = 1
        end
    end

    if isempty(unknowns)
        return [(index = 0, variables = Symbol[], equation_indices = collect(eachindex(equations)))], collect(eachindex(equations))
    end

    Q, P, R, _, block_count = BlockTriangularForm.order(incidence)
    block_labels = Int[]
    for block in 1:block_count
        for _ in R[block]:(R[block + 1] - 1)
            push!(block_labels, block_count - block + 1)
        end
    end
    push!(block_labels, 1)

    variable_matrix = hcat(P, block_labels)'
    equation_matrix = hcat(Q, block_labels)'

    unmatched = equation_matrix[1, :] .< 0
    if any(unmatched)
        keep = .!unmatched
        variable_matrix = variable_matrix[:, keep]
        equation_matrix = equation_matrix[:, keep]
        if !isempty(variable_matrix)
            old_blocks = sort(unique(variable_matrix[2, :]))
            block_remap = Dict(old => new for (new, old) in enumerate(old_blocks))
            for column in axes(variable_matrix, 2)
                variable_matrix[2, column] = block_remap[variable_matrix[2, column]]
                equation_matrix[2, column] = block_remap[equation_matrix[2, column]]
            end
            block_count = length(old_blocks)
        else
            block_count = 0
        end
    end

    blocks = NamedTuple[]
    used_equations = Int[]
    for block in 1:block_count
        equation_columns = findall(equation_matrix[2, :] .== block)
        equation_indices = Int.(equation_matrix[1, equation_columns])
        variable_columns = findall(variable_matrix[2, :] .== block)
        variables = unknowns[Int.(variable_matrix[1, variable_columns])]
        append!(used_equations, equation_indices)
        push!(blocks, (index = block,
                       variables = collect(variables),
                       equation_indices = equation_indices))
    end

    unused_equations = setdiff(collect(eachindex(equations)), used_equations)
    if !isempty(unused_equations)
        push!(blocks, (index = 0,
                       variables = Symbol[],
                       equation_indices = unused_equations))
    end

    equation_order = reduce(vcat, [block.equation_indices for block in blocks]; init = Int[])
    return blocks, equation_order
end

function parameter_values_for_model(model, complete_names)
    parameter_values = model.parameter_values
    values = Dict{Symbol, Float64}()
    free_names = collect(model.constants.post_complete_parameters.parameters)
    for (index, name) in enumerate(free_names)
        values[name] = Float64(parameter_values[index])
    end

    extended_names = model.constants.post_complete_parameters.nsss_param_names_ext
    extended_values = model.workspaces.nsss_solver.params_vec_buffer
    for (index, name) in enumerate(extended_names)
        if index <= length(extended_values)
            values[name] = Float64(extended_values[index])
        end
    end

    missing = setdiff(complete_names, collect(keys(values)))
    if !isempty(missing)
        definitions = parameter_definition_order(model, complete_names)
        for (name, expression) in definitions
            replacements = Dict{Symbol, Any}(key => value for (key, value) in values)
            values[name] = Float64(Core.eval(@__MODULE__, replace_expression(expression, replacements)))
        end
    end

    [get(values, name) do
        error("No value available for parameter $(name) in $(model.model_name)")
    end for name in complete_names]
end

function solution_values_for_model(model, original_names, auxiliary_names, complete_names, complete_values)
    nsss_values, (solution_error, _) = MacroModelling.get_NSSS_and_parameters(
        model,
        copy(model.parameter_values),
        opts = MacroModelling.merge_calculation_options(verbose = false),
        caching = false,
    )
    if !isfinite(solution_error) || solution_error > 1e-6
        error("NSSS solve failed for $(model.model_name) with residual $(solution_error)")
    end

    values = Dict{Symbol, Float64}()
    full_names = model.constants.post_complete_parameters.nsss_sol_names
    full_values = model.workspaces.nsss_solver.sol_vec_buffer
    for (index, name) in enumerate(full_names)
        if index <= length(full_values) && isfinite(full_values[index])
            values[name] = Float64(full_values[index])
        end
    end

    output_indices = model.constants.post_complete_parameters.nsss_output_indices
    for (index, output_index) in enumerate(output_indices)
        if index <= length(nsss_values)
            values[full_names[output_index]] = Float64(nsss_values[index])
        end
    end

    auxiliary_values_map = copy(values)
    all_auxiliary_names = collect(model.constants.post_model_macro.➕_vars)
    parameter_replacements = Dict{Symbol, Any}(
        name => complete_values[index] for (index, name) in enumerate(complete_names)
    )

    auxiliary_set = Set(auxiliary_names)
    auxiliary_equations = model.equations.steady_state_aux
    for _ in 1:max(length(auxiliary_equations), 1)
        changed = false
        for equation in auxiliary_equations
            if !(equation isa Expr && equation.head == :call && equation.args[1] == :- && length(equation.args) >= 3)
                continue
            end
            auxiliary_name = equation.args[2]
            if !(auxiliary_name in auxiliary_set)
                continue
            end
            dependencies = intersect(expression_symbols(equation.args[3]), auxiliary_set)
            if all(haskey(auxiliary_values_map, dependency) for dependency in dependencies)
                replacements = copy(parameter_replacements)
                for (name, value) in auxiliary_values_map
                    replacements[name] = value
                end
                evaluated = Core.eval(@__MODULE__, replace_expression(equation.args[3], replacements))
                new_value = Float64(evaluated)
                if !haskey(auxiliary_values_map, auxiliary_name) || auxiliary_values_map[auxiliary_name] != new_value
                    auxiliary_values_map[auxiliary_name] = new_value
                    changed = true
                end
            end
        end
        !changed && break
    end

    defaulted_names = Symbol[]
    for name in union(original_names, auxiliary_names, all_auxiliary_names)
        if !haskey(values, name)
            if startswith(string(name), "ϵᵒᵇᶜ")
                values[name] = 0.0
                auxiliary_values_map[name] = 0.0
                push!(defaulted_names, name)
            else
                error("No NSSS value available for $(name) in $(model.model_name)")
            end
        end
    end

    original_solution = [values[name] for name in original_names]
    residual_buffer = zeros(Float64, length(model.equations.steady_state) + length(model.equations.calibration))
    model.functions.NSSS_check(residual_buffer, model.parameter_values, original_solution)
    residual_norm = norm(residual_buffer)
    for _ in 1:8
        if !isfinite(residual_norm) || residual_norm < 1e-8
            break
        end
        jacobian = zeros(Float64, length(residual_buffer), length(original_solution))
        model.functions.NSSS_∂equations_∂SS_and_pars(jacobian, model.parameter_values, original_solution)
        update = jacobian \ residual_buffer
        candidate = original_solution - update
        model.functions.NSSS_check(residual_buffer, model.parameter_values, candidate)
        candidate_norm = norm(residual_buffer)
        if !isfinite(candidate_norm) || candidate_norm >= residual_norm
            break
        end
        original_solution = candidate
        residual_norm = candidate_norm
    end
    for (index, name) in enumerate(original_names)
        values[name] = original_solution[index]
    end

    original_values = [values[name] for name in original_names]
    auxiliary_values = [auxiliary_values_map[name] for name in auxiliary_names]
    all_auxiliary_values = [auxiliary_values_map[name] for name in all_auxiliary_names]
    return original_values, auxiliary_values, all_auxiliary_names, all_auxiliary_values,
           defaulted_names, solution_error, residual_norm
end

function bounds_for_names(model, names, fixed_zero_names = Symbol[])
    lower = fill(-Inf, length(names))
    upper = fill(Inf, length(names))
    indices = Dict(name => index for (index, name) in enumerate(names))

    for (name, bounds) in model.constants.post_parameters_macro.bounds
        if haskey(indices, name)
            index = indices[name]
            lower[index] = max(lower[index], Float64(bounds[1]))
            upper[index] = min(upper[index], Float64(bounds[2]))
        end
    end

    for name in fixed_zero_names
        if haskey(indices, name)
            index = indices[name]
            lower[index] = 0.0
            upper[index] = 0.0
        end
    end

    solver_constants = model.constants.nsss_solver
    full_names = model.constants.post_complete_parameters.nsss_sol_names
    for step in 1:solver_constants.n_steps
        write_range = solver_constants.write_ranges[step]
        write_indices = solver_constants.write_indices[write_range]
        bounds_range = solver_constants.bounds_ranges[step]
        for local_index in 1:min(length(write_indices), length(bounds_range))
            bound_index = bounds_range[local_index]
            if solver_constants.has_bounds[bound_index]
                name = full_names[write_indices[local_index]]
                if haskey(indices, name)
                    index = indices[name]
                    lower[index] = max(lower[index], solver_constants.lower_bounds[bound_index])
                    upper[index] = min(upper[index], solver_constants.upper_bounds[bound_index])
                end
            end
        end

        numerical_range = solver_constants.numerical_bounds_ranges[step]
        for local_index in 1:min(length(write_indices), length(numerical_range))
            bound_index = numerical_range[local_index]
            name = full_names[write_indices[local_index]]
            if haskey(indices, name)
                index = indices[name]
                lower[index] = max(lower[index], solver_constants.numerical_lbs[bound_index])
                upper[index] = min(upper[index], solver_constants.numerical_ubs[bound_index])
            end
        end
    end
    return lower, upper
end

function write_string_vector(io, constant_name, values)
    println(io, "const $(constant_name) = [")
    for value in values
        println(io, "    ", repr(display_symbol(Symbol(value))), ",")
    end
    println(io, "]")
end

function write_symbol_vector(io, constant_name, values)
    println(io, "const $(constant_name) = Symbol[")
    for value in values
        println(io, "    ", repr(Symbol(value)), ",")
    end
    println(io, "]")
end

function write_float_vector(io, constant_name, values)
    println(io, "const $(constant_name) = Float64[")
    for value in values
        println(io, "    ", repr(Float64(value)), ",")
    end
    println(io, "]")
end

function write_expression_vector(io, constant_name, equations)
    println(io, "const $(constant_name) = Expr[")
    for equation in equations
        println(io, "    ", repr(equation), ",")
    end
    println(io, "]")
end

function write_residual_function(io, function_name, solution_constant, solution, equations, parameter_map)
    println(io, "function $(function_name)(parameters::AbstractVector, solution::AbstractVector)")
    println(io, "    @assert length(parameters) == length(PARAMETER_NAMES)")
    println(io, "    @assert length(solution) == length($(solution_constant))")
    println(io, "    complete_parameters = complete_parameter_values(parameters)")
    println(io, "    return [")
    solution_map = Dict{Symbol, Any}(name => :(solution[$index]) for (index, name) in enumerate(solution))
    replacements = copy(parameter_map)
    merge!(replacements, solution_map)
    for equation in equations
        println(io, "        ", expression_source(replace_expression(equation, replacements)), ",")
    end
    println(io, "    ]")
    println(io, "end")
end

function write_model_file(model, output_path, source_file)
    model_name = Symbol(model.model_name)
    module_name = Symbol(string(model_name), "NsssResiduals")
    original_equations = collect(model.equations.steady_state)
    auxiliary_equations = collect(model.equations.steady_state_aux)
    calibration_equations = collect(model.equations.calibration)
    original_residual_equations = vcat(original_equations, calibration_equations)
    auxiliary_residual_equations = vcat(auxiliary_equations, calibration_equations)

    original_solution_names = collect(union(
        model.constants.post_model_macro.vars_in_ss_equations_no_aux,
        model.equations.calibration_parameters,
    ))
    auxiliary_solution_names = collect(union(
        model.constants.post_model_macro.vars_in_ss_equations,
        model.equations.calibration_parameters,
    ))
    free_parameter_names = collect(model.constants.post_complete_parameters.parameters)
    complete_names = complete_parameter_names(model)
    complete_values = parameter_values_for_model(model, complete_names)
    original_values, auxiliary_values, all_auxiliary_names, all_auxiliary_values,
    defaulted_names, solution_error, residual_norm = solution_values_for_model(
        model,
        original_solution_names,
        auxiliary_solution_names,
        complete_names,
        complete_values,
    )

    blocks, equation_order = make_blocks(model, auxiliary_equations, calibration_equations)
    original_lower, original_upper = bounds_for_names(model, original_solution_names, defaulted_names)
    auxiliary_lower, auxiliary_upper = bounds_for_names(model, auxiliary_solution_names, defaulted_names)
    all_auxiliary_lower, all_auxiliary_upper = bounds_for_names(model, all_auxiliary_names, defaulted_names)
    parameter_lower, parameter_upper = bounds_for_names(model, free_parameter_names)

    parameter_map = Dict{Symbol, Any}(
        name => :(complete_parameters[$index]) for (index, name) in enumerate(complete_names)
    )
    parameter_definition_list = parameter_definition_order(model, complete_names)
    free_indices = Dict(name => index for (index, name) in enumerate(free_parameter_names))
    definition_map = Dict(name => expression for (name, expression) in parameter_definition_list)

    open(output_path, "w") do io
        println(io, "module $(module_name)")
        println(io, "using MacroModelling")
        println(io, "")
        println(io, "const normcdf = MacroModelling.SymPyWorkspace.normcdf")
        println(io, "const pnorm = MacroModelling.SymPyWorkspace.pnorm")
        println(io, "const normpdf = MacroModelling.SymPyWorkspace.normpdf")
        println(io, "const dnorm = MacroModelling.SymPyWorkspace.dnorm")
        println(io, "const normlogpdf = MacroModelling.SymPyWorkspace.normlogpdf")
        println(io, "const norminvcdf = MacroModelling.SymPyWorkspace.norminvcdf")
        println(io, "const norminv = MacroModelling.SymPyWorkspace.norminv")
        println(io, "const qnorm = MacroModelling.SymPyWorkspace.qnorm")
        println(io, "const erfcinv = MacroModelling.SymPyWorkspace.erfcinv")
        println(io, "const erfc = MacroModelling.SymPyWorkspace.erfc")
        println(io, "const Max = max")
        println(io, "const Min = min")
        println(io, "")
        println(io, "const MODEL_NAME = ", repr(string(model_name)))
        println(io, "const SOURCE_MODEL_FILE = ", repr(source_file))
        println(io, "const NSSS_SOLUTION_ERROR = ", repr(Float64(solution_error)))
        println(io, "const NSSS_RESIDUAL_NORM = ", repr(Float64(residual_norm)))
        println(io, "")

        write_string_vector(io, "PARAMETER_NAMES", free_parameter_names)
        write_float_vector(io, "PARAMETER_VALUES", model.parameter_values)
        write_string_vector(io, "COMPLETE_PARAMETER_NAMES", complete_names)
        write_float_vector(io, "COMPLETE_PARAMETER_VALUES", complete_values)
        write_string_vector(io, "ORIGINAL_SOLUTION_NAMES", original_solution_names)
        write_float_vector(io, "ORIGINAL_SOLUTION_VALUES", original_values)
        write_string_vector(io, "AUXILIARY_SOLUTION_NAMES", auxiliary_solution_names)
        write_float_vector(io, "AUXILIARY_SOLUTION_VALUES", auxiliary_values)
        write_string_vector(io, "ALL_AUXILIARY_VARIABLE_NAMES", all_auxiliary_names)
        write_float_vector(io, "ALL_AUXILIARY_VARIABLE_VALUES", all_auxiliary_values)
        write_string_vector(io, "DEFAULTED_NSSS_SOLUTION_NAMES", defaulted_names)
        write_string_vector(io, "CALIBRATION_PARAMETER_NAMES", model.equations.calibration_parameters)
        println(io, "")

        write_expression_vector(io, "ORIGINAL_NSSS_EQUATIONS", original_equations)
        write_expression_vector(io, "CALIBRATION_EQUATIONS", calibration_equations)
        write_expression_vector(io, "AUXILIARY_NSSS_EQUATIONS", auxiliary_equations)
        write_expression_vector(io, "ORIGINAL_RESIDUAL_EQUATIONS", original_residual_equations)
        write_expression_vector(io, "AUXILIARY_RESIDUAL_EQUATIONS", auxiliary_residual_equations)
        println(io, "")

        write_string_vector(io, "PARAMETER_DEFINITION_NAMES", [name for (name, _) in parameter_definition_list])
        write_string_vector(io, "PARAMETER_DEFINITION_EXPRESSIONS", [string(expression) for (_, expression) in parameter_definition_list])

        write_string_vector(io, "PARAMETER_BOX_CONSTRAINT_NAMES", free_parameter_names)
        write_float_vector(io, "PARAMETER_BOX_LOWER_BOUNDS", parameter_lower)
        write_float_vector(io, "PARAMETER_BOX_UPPER_BOUNDS", parameter_upper)
        write_string_vector(io, "ORIGINAL_BOX_CONSTRAINT_NAMES", original_solution_names)
        write_float_vector(io, "ORIGINAL_BOX_LOWER_BOUNDS", original_lower)
        write_float_vector(io, "ORIGINAL_BOX_UPPER_BOUNDS", original_upper)
        write_string_vector(io, "AUXILIARY_BOX_CONSTRAINT_NAMES", auxiliary_solution_names)
        write_float_vector(io, "AUXILIARY_BOX_LOWER_BOUNDS", auxiliary_lower)
        write_float_vector(io, "AUXILIARY_BOX_UPPER_BOUNDS", auxiliary_upper)
        write_string_vector(io, "ALL_AUXILIARY_BOX_CONSTRAINT_NAMES", all_auxiliary_names)
        write_float_vector(io, "ALL_AUXILIARY_BOX_LOWER_BOUNDS", all_auxiliary_lower)
        write_float_vector(io, "ALL_AUXILIARY_BOX_UPPER_BOUNDS", all_auxiliary_upper)
        println(io, "")

        println(io, "const BLOCKS = [")
        for block in blocks
            println(io, "    (")
            println(io, "        index = ", block.index, ",")
            println(io, "        variables = ", repr(display_symbols(block.variables)), ",")
            println(io, "        equation_indices = ", repr(block.equation_indices), ",")
            println(io, "        equations = Expr[")
            for equation_index in block.equation_indices
                println(io, "            ", repr(auxiliary_residual_equations[equation_index]), ",")
            end
            println(io, "        ],")
            println(io, "    ),")
        end
        println(io, "]")
        println(io, "const BLOCK_EQUATION_ORDER = ", repr(equation_order))
        println(io, "")

        println(io, "function complete_parameter_values(parameters::AbstractVector)")
        println(io, "    @assert length(parameters) == length(PARAMETER_NAMES)")
        println(io, "    complete_parameters = Vector{eltype(parameters)}(undef, length(COMPLETE_PARAMETER_NAMES))")
        for (name, index) in free_indices
            complete_index = findfirst(==(name), complete_names)
            println(io, "    complete_parameters[", complete_index, "] = parameters[", index, "]")
        end
        complete_map = Dict{Symbol, Any}(
            name => :(complete_parameters[$index]) for (index, name) in enumerate(complete_names)
        )
        for (name, expression) in parameter_definition_list
            complete_index = findfirst(==(name), complete_names)
            println(io, "    complete_parameters[", complete_index, "] = ",
                    expression_source(replace_expression(expression, complete_map)))
        end
        println(io, "    return complete_parameters")
        println(io, "end")
        println(io, "")

        write_residual_function(io, "residuals_original", "ORIGINAL_SOLUTION_NAMES", original_solution_names,
                                original_residual_equations, parameter_map)
        println(io, "")
        write_residual_function(io, "residuals_auxiliary", "AUXILIARY_SOLUTION_NAMES", auxiliary_solution_names,
                                auxiliary_residual_equations, parameter_map)
        println(io, "")
        println(io, "function residuals_blocks(parameters::AbstractVector, solution::AbstractVector)")
        println(io, "    return residuals_auxiliary(parameters, solution)[BLOCK_EQUATION_ORDER]")
        println(io, "end")
        println(io, "")
        println(io, "export MODEL_NAME, SOURCE_MODEL_FILE, NSSS_SOLUTION_ERROR, NSSS_RESIDUAL_NORM")
        println(io, "export PARAMETER_NAMES, PARAMETER_VALUES, COMPLETE_PARAMETER_NAMES, COMPLETE_PARAMETER_VALUES")
        println(io, "export ORIGINAL_SOLUTION_NAMES, ORIGINAL_SOLUTION_VALUES")
        println(io, "export AUXILIARY_SOLUTION_NAMES, AUXILIARY_SOLUTION_VALUES")
        println(io, "export ALL_AUXILIARY_VARIABLE_NAMES, ALL_AUXILIARY_VARIABLE_VALUES")
        println(io, "export DEFAULTED_NSSS_SOLUTION_NAMES")
        println(io, "export ORIGINAL_NSSS_EQUATIONS, AUXILIARY_NSSS_EQUATIONS, CALIBRATION_EQUATIONS")
        println(io, "export BLOCKS, BLOCK_EQUATION_ORDER, residuals_original, residuals_auxiliary, residuals_blocks")
        println(io, "end")
    end
    return output_path
end

mkpath(output_directory)
model_files = sort(filter(file -> endswith(file, ".jl"), readdir(model_directory)))
requested_model = get(ENV, "NSSS_MODEL", "")
if !isempty(requested_model)
    model_files = filter(file -> first(splitext(file)) == requested_model, model_files)
end
for model_file in model_files
    model_path = joinpath(model_directory, model_file)
    model_name = Symbol(first(splitext(model_file)))
    println("Generating ", model_name, " from ", model_file)
    include(model_path)
    model = getfield(Main, model_name)
    output_path = joinpath(output_directory, string(model_name, ".jl"))
    write_model_file(model, output_path, joinpath("models", model_file))
    println("Wrote ", output_path)
end
