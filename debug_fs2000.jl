using Revise
using MacroModelling

include(joinpath(@__DIR__, "models/FS2000.jl"))

nsss = FS2000.NSSS
println("Number of solve steps: $(length(nsss.solve_steps))")
for (i, step) in enumerate(nsss.solve_steps)
    if step isa MacroModelling.AnalyticalNSSSStep
        println("  Step $i: Analytical - $(step.description), bounds=$(step.has_bounds), has_aux=$(step.aux_func! !== nothing), has_err=$(step.error_func! !== nothing)")
    elseif step isa MacroModelling.NumericalNSSSStep
        println("  Step $i: Numerical - $(step.description)")
    end
end

println("\n=== Manual step execution ===")
initial_params = FS2000.parameter_values
tol = MacroModelling.Tolerances()
params_vec = Vector{Float64}(undef, nsss.n_ext_params)
nsss.param_prep!(params_vec, initial_params)
sol_vec = zeros(Float64, nsss.n_sol)

println("tol.NSSS_acceptance_tol = $(tol.NSSS_acceptance_tol)")

cumulative_error = 0.0
for (i, step) in enumerate(nsss.solve_steps)
    if step isa MacroModelling.AnalyticalNSSSStep
        step_error, _, _ = MacroModelling.execute_step!(step, sol_vec, params_vec, nothing, FS2000, tol, false, false, MacroModelling.solver_parameters[], false)
        cumulative_error += step_error
        println("  Step $i ($(step.description)): step_error=$step_error, cumulative=$cumulative_error")
        if cumulative_error > tol.NSSS_acceptance_tol
            println("  *** BREAK: cumulative error $cumulative_error > tol $(tol.NSSS_acceptance_tol)")
            break
        end
    else
        println("  Step $i: Numerical block - skipping")
        break
    end
end
