#!/usr/bin/env julia
# Debug the step order and error_func for Gali_2015_chapter_3_nonlinear

using Revise
using MacroModelling

include("models/Gali_2015_chapter_3_nonlinear.jl")

# Print step details
nsss = Gali_2015_chapter_3_nonlinear.NSSS
println("Total steps: ", length(nsss.solve_steps))
for (i, step) in enumerate(nsss.solve_steps)
    if step isa MacroModelling.AnalyticalNSSSStep
        has_error = step.error_func! !== nothing
        has_aux = step.aux_func! !== nothing
        n_bounds = sum(step.has_bounds)
        n_clamp = sum(step.clamp_to_bounds)
        println("Step $i: Analytical - $(step.description)")
        println("         has_aux=$has_aux, has_error=$has_error, bounds=$n_bounds, clamp=$n_clamp")
        println("         write_indices=$(step.write_indices)")
    elseif step isa MacroModelling.NumericalNSSSStep
        has_aux_error = step.aux_error_func! !== nothing
        println("Step $i: Numerical - $(step.description)")
        println("         block_index=$(step.block_index), write_indices=$(step.write_indices)")
        println("         has_aux_error=$has_aux_error")
    end
end

# Now manually execute steps to see where Pi_star error comes from
params = Gali_2015_chapter_3_nonlinear.parameter_values
params_vec = Vector{Float64}(undef, nsss.n_ext_params)
nsss.param_prep!(params_vec, params)
sol_vec = zeros(Float64, nsss.n_sol)

println("\n--- Manual step execution ---")
for (i, step) in enumerate(nsss.solve_steps)
    if step isa MacroModelling.AnalyticalNSSSStep
        # Phase 1: aux
        if step.aux_func! !== nothing
            step.aux_func!(step.aux_buffer, sol_vec, params_vec)
            for (j, idx) in enumerate(step.aux_write_indices)
                sol_vec[idx] = step.aux_buffer[j]
            end
            println("Step $i aux: wrote $(step.aux_buffer) to indices $(step.aux_write_indices)")
        end
        
        # Error check
        if step.error_func! !== nothing
            step.error_func!(step.error_buffer, sol_vec, params_vec)
            err = sum(abs, step.error_buffer)
            println("Step $i error_func: buffer=$(step.error_buffer), total_err=$err")
            if err > 1e-12
                println("  → DOMAIN ERROR at step $i: $(step.description)")
            end
        end
        
        # Phase 2: main
        if !isempty(step.write_indices)
            step.eval_func!(step.buffer, sol_vec, params_vec)
            for (j, idx) in enumerate(step.write_indices)
                raw = step.buffer[j]
                if step.has_bounds[j]
                    clamped = clamp(raw, step.lower_bounds[j], step.upper_bounds[j])
                    bounds_err = abs(clamped - raw)
                    if step.clamp_to_bounds[j]
                        sol_vec[idx] = clamped
                    else
                        sol_vec[idx] = raw
                    end
                    if bounds_err > 1e-12
                        println("  → BOUNDS ERROR at step $i: raw=$raw → clamped=$clamped, err=$bounds_err")
                    end
                else
                    sol_vec[idx] = raw
                end
            end
            println("Step $i eval: $(step.description) → wrote to $(step.write_indices)")
        end
    else
        println("Step $i: NUMERICAL - $(step.description) [skipping]")
    end
end
