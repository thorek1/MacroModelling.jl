using Test

project_root = normpath(joinpath(@__DIR__, ".."))
source_files = sort(filter(file -> endswith(file, ".jl"), readdir(joinpath(project_root, "models"))))
output_directory = joinpath(project_root, "nsss_residuals")
output_files = sort(filter(file -> endswith(file, ".jl"), readdir(output_directory)))

@test length(output_files) == length(source_files)
@test Set(output_files) == Set(source_files)

for source_file in source_files
    model_name = first(splitext(source_file))
    module_name = Symbol(model_name, "NsssResiduals")
    include(joinpath(output_directory, source_file))
    residual_module = getfield(Main, module_name)

    parameters = residual_module.PARAMETER_VALUES
    original_solution = residual_module.ORIGINAL_SOLUTION_VALUES
    original_initial_solution = residual_module.ORIGINAL_INITIAL_SOLUTION_VALUES
    auxiliary_solution = residual_module.AUXILIARY_SOLUTION_VALUES
    auxiliary_initial_solution = residual_module.AUXILIARY_INITIAL_SOLUTION_VALUES
    all_auxiliary_initial_values = residual_module.ALL_AUXILIARY_VARIABLE_INITIAL_VALUES

    @test length(original_initial_solution) == length(residual_module.ORIGINAL_SOLUTION_NAMES)
    @test length(auxiliary_initial_solution) == length(residual_module.AUXILIARY_SOLUTION_NAMES)
    @test length(all_auxiliary_initial_values) == length(residual_module.ALL_AUXILIARY_VARIABLE_NAMES)
    @test all(isfinite, original_initial_solution)
    @test all(isfinite, auxiliary_initial_solution)
    @test all(isfinite, all_auxiliary_initial_values)

    original_residual = residual_module.residuals_original(parameters, original_solution)
    auxiliary_residual = residual_module.residuals_auxiliary(parameters, auxiliary_solution)
    previous_solutions = residual_module.BLOCK_PREVIOUS_SOLUTION_VALUES
    external_solutions = residual_module.BLOCK_EXTERNAL_SOLUTION_VALUES
    block_solutions = residual_module.BLOCK_SOLUTION_VALUES
    block_residual = residual_module.residuals_blocks(parameters, previous_solutions, external_solutions, block_solutions)

    @test all(isfinite, original_residual)
    @test all(isfinite, auxiliary_residual)
    @test all(isfinite, block_residual)
    @test maximum(abs, original_residual; init = 0.0) < 1e-7
    @test maximum(abs, auxiliary_residual; init = 0.0) < 1e-7
    @test length(residual_module.BLOCK_EQUATION_ORDER) == length(auxiliary_residual)
    @test length(residual_module.BLOCKS) == length(residual_module.BLOCK_SOLVE_ORDER)
    @test residual_module.BLOCK_SOLVE_ORDER == [
        block.index for block in sort(residual_module.BLOCKS, by = block -> block.solve_order)
    ]

    expected_block_residual = Float64[]
    for (block_index, block) in enumerate(residual_module.BLOCKS)
        @test block.index == block_index
        @test block.previous_solution_names == residual_module.BLOCK_PREVIOUS_SOLUTION_NAMES[block_index]
        @test block.external_solution_names == residual_module.BLOCK_EXTERNAL_SOLUTION_NAMES[block_index]
        @test block.solution_names == residual_module.BLOCK_SOLUTION_NAMES[block_index]
        @test block.previous_solution_values == residual_module.BLOCK_PREVIOUS_SOLUTION_VALUES[block_index]
        @test block.external_solution_values == residual_module.BLOCK_EXTERNAL_SOLUTION_VALUES[block_index]
        @test block.solution_values == residual_module.BLOCK_SOLUTION_VALUES[block_index]
        @test block.previous_solution_initial_values ==
              residual_module.BLOCK_PREVIOUS_SOLUTION_INITIAL_VALUES[block_index]
        @test block.external_solution_initial_values ==
              residual_module.BLOCK_EXTERNAL_SOLUTION_INITIAL_VALUES[block_index]
        @test block.solution_initial_values ==
              residual_module.BLOCK_SOLUTION_INITIAL_VALUES[block_index]
        @test all(isfinite, block.previous_solution_initial_values)
        @test all(isfinite, block.external_solution_initial_values)
        @test all(isfinite, block.solution_initial_values)
        @test isempty(intersect(block.previous_solution_names, block.solution_names))
        @test isempty(intersect(block.external_solution_names, block.solution_names))
        @test isempty(intersect(block.previous_solution_names, block.external_solution_names))
        @test all(name -> name in block.solution_names, block.domain_auxiliary_names)
        block_function = getfield(residual_module, Symbol("residuals_block_", block_index))
        block_residual_i = block_function(
            parameters,
            residual_module.BLOCK_PREVIOUS_SOLUTION_VALUES[block_index],
            residual_module.BLOCK_EXTERNAL_SOLUTION_VALUES[block_index],
            residual_module.BLOCK_SOLUTION_VALUES[block_index],
        )
        @test all(isfinite, block_residual_i)
        @test length(block_residual_i) == length(block.equations) + length(block.domain_auxiliary_equations)
        @test maximum(abs, block_residual_i; init = 0.0) < 1e-7
        append!(expected_block_residual, block_residual_i)
        @test all((block.box_lower_bounds .- 1e-12) .<= block.solution_values .<=
                  (block.box_upper_bounds .+ 1e-12))
    end
    @test block_residual == expected_block_residual

    original_lower = residual_module.ORIGINAL_BOX_LOWER_BOUNDS
    original_upper = residual_module.ORIGINAL_BOX_UPPER_BOUNDS
    auxiliary_lower = residual_module.AUXILIARY_BOX_LOWER_BOUNDS
    auxiliary_upper = residual_module.AUXILIARY_BOX_UPPER_BOUNDS
    all_auxiliary_lower = residual_module.ALL_AUXILIARY_BOX_LOWER_BOUNDS
    all_auxiliary_upper = residual_module.ALL_AUXILIARY_BOX_UPPER_BOUNDS
    @test all((original_lower .- 1e-12) .<= original_solution .<= (original_upper .+ 1e-12))
    @test all((auxiliary_lower .- 1e-12) .<= auxiliary_solution .<= (auxiliary_upper .+ 1e-12))
    @test all((all_auxiliary_lower .- 1e-12) .<= residual_module.ALL_AUXILIARY_VARIABLE_VALUES .<= (all_auxiliary_upper .+ 1e-12))
    @test all(iszero, auxiliary_solution[
        indexin(residual_module.DEFAULTED_NSSS_SOLUTION_NAMES,
                residual_module.AUXILIARY_SOLUTION_NAMES)
    ])
end

println("Verified ", length(source_files), " model residual exports")
