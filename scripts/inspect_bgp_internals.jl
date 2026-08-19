#!/usr/bin/env julia

using MacroModelling

# This file is intentionally written as top-level "spaghetti" code.
# Run it with:
#
#     julia --project=. -i scripts/inspect_bgp_internals.jl
#
# In VS Code, execute one `# %%` block at a time. Every intermediate value is
# left in `Main`, so it can be inspected, modified, and rerun before continuing.

# %%
# The @model macro calls process_model_equations internally. After this block,
# inspect `InspectBGP.equations`, `InspectBGP.constants`, and `InspectBGP.workspaces`.
@model InspectBGP begin
    a[0] = a[-1] * g[0]
    g[0] = 1.02 + σg * eg[x]
    x[0] = 0.5 * x[-1] + β * x[1] + (0.5 - β) * a[0]
end

debug_model = InspectBGP
raw_model_equations = copy(debug_model.equations.original)
raw_dynamic_equations = copy(debug_model.equations.dynamic)
raw_steady_state_equations = copy(debug_model.equations.steady_state)
raw_model_variables = copy(debug_model.constants.post_model_macro.var)

@show raw_model_variables
display(raw_model_equations)

# %%
# This is the parameter parsing part of @parameters, written out so that the
# parsed block can be inspected before it is applied to the model.
parameter_block = quote
    σg = 0.01
    β = 0.2
end

parsed_parameters = MacroModelling.process_parameter_definitions(
    parameter_block,
    debug_model.constants.post_model_macro,
)

@show parsed_parameters.parameters
@show parsed_parameters.parameter_values
@show parsed_parameters.missing_parameters
@show parsed_parameters.calib_parameters_no_var
@show parsed_parameters.bounds

# %%
# Apply the parsed parameter information. These statements mirror the
# corresponding assignments in the @parameters macro.
parameter_guess = debug_model.constants.post_parameters_macro.guess
parameter_bounds = copy(debug_model.constants.post_parameters_macro.bounds)

for (name, bound) in parsed_parameters.bounds
    parameter_bounds[name] = haskey(parameter_bounds, name) ?
        (max(parameter_bounds[name][1], bound[1]),
         min(parameter_bounds[name][2], bound[2])) :
        bound
end

debug_model.constants.post_parameters_macro = MacroModelling.post_parameters_macro(
    parsed_parameters.calib_parameters_no_var,
    false,
    :single_equation,
    :ESCH,
    120.0,
    parameter_guess,
    parsed_parameters.ss_calib_list,
    parsed_parameters.par_calib_list,
    parameter_bounds,
)

debug_model.equations.calibration = parsed_parameters.equations.calibration
debug_model.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
debug_model.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters
debug_model.equations.calibration_original = parsed_parameters.equations.calibration_original

debug_model.constants.post_complete_parameters =
    MacroModelling.update_post_complete_parameters(
        debug_model.constants.post_complete_parameters;
        parameters = parsed_parameters.parameters,
        missing_parameters = parsed_parameters.missing_parameters,
    )
debug_model.parameter_values = parsed_parameters.parameter_values
debug_model.functions.functions_written = false

@show debug_model.parameter_values
@show debug_model.constants.post_complete_parameters.parameters

# %%
# Inspect the structural BGP detector before stationarization.
parameter_values_by_name = Dict(
    debug_model.constants.post_complete_parameters.parameters .=> debug_model.parameter_values,
)

bgp_profile_preview = MacroModelling.build_bgp_detection_metadata(
    raw_model_equations,
    debug_model.constants.post_complete_parameters.parameters,
    debug_model.parameter_values,
)

@show bgp_profile_preview.candidate_drivers
@show bgp_profile_preview.active_drivers
@show bgp_profile_preview.trigger_parameters
@show bgp_profile_preview.trigger_indices
@show bgp_profile_preview.mode

# %%
# Inspect the symbolic growth restriction and AST rewrite before applying it.
stationarization_preview, stationary_equations_preview =
    MacroModelling.build_stationarization_metadata(
        raw_model_equations,
        parameter_values_by_name;
        drivers_override = bgp_profile_preview.active_drivers,
    )

@show stationarization_preview.trend_drivers
@show stationarization_preview.trending_variables
@show stationarization_preview.growth_variables
@show stationarization_preview.growth_exponent_expressions
display(stationary_equations_preview)

# %%
# Parse the generated stationary equations separately. This is the same parser
# call made inside stationarize_model!, kept here as a visible checkpoint.
stationary_parse =
    MacroModelling.process_model_equations(
        Expr(:block, stationary_equations_preview...),
        debug_model.constants.post_model_macro.max_obc_horizon,
        debug_model.constants.post_parameters_macro.precompile;
        allow_single_variable_equations = true,
    )

stationary_parse_constants = stationary_parse[1]
stationary_parse_equations = stationary_parse[2]
stationary_parse_model_constants = stationary_parse[3]
stationary_parse_workspaces = stationary_parse[4]

@show stationary_parse_constants.var
@show stationary_parse_model_constants.post_model_macro.vars_in_ss_equations
display(stationary_parse_equations.original)
display(stationary_parse_equations.dynamic)
display(stationary_parse_equations.steady_state)

# %%
# Apply the actual representation switch. Inspect the model immediately after
# this line before proceeding to the NSSS setup.
MacroModelling.stationarize_model!(debug_model; verbose = true, silent = false)

active_bgp_profile = debug_model.equations.bgp_detection
active_stationarization = debug_model.equations.stationarization
active_model_variables = debug_model.constants.post_model_macro.var
active_dynamic_equations = debug_model.equations.dynamic
active_steady_state_equations = debug_model.equations.steady_state

@show active_bgp_profile.active_drivers
@show active_stationarization.growth_exponents
@show active_model_variables
display(active_dynamic_equations)
display(active_steady_state_equations)

# %%
# Build the steady-state solver functions and inspect the generated pipeline.
MacroModelling.set_up_steady_state_solver!(
    debug_model;
    verbose = true,
    silent = false,
    ss_symbolic_mode = :single_equation,
)

nsss_solver_constants = debug_model.constants.nsss_solver
nsss_solver_functions = debug_model.functions.nsss_solver
nsss_solver_workspace = debug_model.workspaces.nsss_solver

@show nsss_solver_constants.n_steps
@show nsss_solver_constants.descriptions
@show nsss_solver_constants.step_types
@show nsss_solver_constants.write_ranges

# %%
# Solve the NSSS through the same internal entry point used during setup.
steady_state_options = MacroModelling.merge_calculation_options(verbose = true)

steady_state_run = MacroModelling.solve_steady_state!(
    debug_model,
    steady_state_options,
    :ESCH,
    120.0;
    silent = false,
)

ss_and_pars = steady_state_run[1]
solution_error = steady_state_run[2]
found_solution = steady_state_run[3]
nsss_solution_names = copy(debug_model.constants.post_complete_parameters.nsss_sol_names)
nsss_full_solution = copy(debug_model.workspaces.nsss_solver.sol_vec_buffer)

@show solution_error
@show found_solution
@show nsss_solution_names
@show nsss_full_solution

# %%
# Evaluate the generated NSSS residual function by hand.
nsss_check_unknowns = union(
    setdiff(
        debug_model.constants.post_model_macro.vars_in_ss_equations,
        debug_model.constants.post_model_macro.➕_vars,
    ),
    debug_model.equations.calibration_parameters,
)
nsss_check_indices = Int.(indexin(nsss_check_unknowns, nsss_solution_names))
nsss_check_input = nsss_full_solution[nsss_check_indices]
nsss_residuals = zeros(Float64, length(debug_model.workspaces.nsss_solver.check_residual))

debug_model.functions.NSSS_check(
    nsss_residuals,
    debug_model.parameter_values,
    nsss_check_input,
)

@show nsss_check_unknowns
@show nsss_residuals

# %%
# Generate perturbation derivative functions, then evaluate the Jacobian.
MacroModelling.write_symbolic_derivatives!(
    debug_model;
    perturbation_order = 1,
    silent = false,
)

jacobian = MacroModelling.calculate_jacobian(
    debug_model.parameter_values,
    ss_and_pars,
    debug_model.caches,
    debug_model.functions.jacobian,
    debug_model.workspaces;
    caching = false,
)

@show size(jacobian)
display(jacobian)

# %%
# Solve the first-order perturbation system directly.
first_order_solution, qme_solution, first_order_solved =
    MacroModelling.calculate_first_order_solution(
        jacobian,
        debug_model.constants,
        debug_model.workspaces,
        debug_model.caches;
        opts = steady_state_options,
        initial_guess = debug_model.caches.qme_solution,
        parameter_values = debug_model.parameter_values,
    )

@show first_order_solved
@show size(first_order_solution)
display(first_order_solution)

# %%
# Finally compare the direct internal result with the public API.
public_steady_state = get_SS(debug_model, derivatives = false)
public_solution = get_solution(debug_model)
public_irf = get_irf(debug_model, shocks = :none, periods = 3)

display(public_steady_state)
display(public_solution)
display(public_irf)
