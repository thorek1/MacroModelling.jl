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
    auxiliary_solution = residual_module.AUXILIARY_SOLUTION_VALUES

    original_residual = residual_module.residuals_original(parameters, original_solution)
    auxiliary_residual = residual_module.residuals_auxiliary(parameters, auxiliary_solution)
    block_residual = residual_module.residuals_blocks(parameters, auxiliary_solution)

    @test all(isfinite, original_residual)
    @test all(isfinite, auxiliary_residual)
    @test all(isfinite, block_residual)
    @test maximum(abs, original_residual; init = 0.0) < 1e-7
    @test maximum(abs, auxiliary_residual; init = 0.0) < 1e-7
    @test block_residual == auxiliary_residual[residual_module.BLOCK_EQUATION_ORDER]
    @test length(residual_module.BLOCK_EQUATION_ORDER) == length(auxiliary_residual)

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
