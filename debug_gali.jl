using Revise; using MacroModelling

include(joinpath(@__DIR__, "models/Gali_2015_chapter_3_nonlinear.jl"))

nsss = Gali_2015_chapter_3_nonlinear.NSSS

# List all steps
println("=== ALL STEPS ===")
for (i, step) in enumerate(nsss.solve_steps)
    println("Step $i: $(step.description) [$(typeof(step).name.name)]")
    if step isa MacroModelling.AnalyticalNSSSStep
        println("  write_indices: $(step.write_indices)")
        println("  has_bounds: $(step.has_bounds)")
        println("  lower_bounds: $(step.lower_bounds)")  
        println("  upper_bounds: $(step.upper_bounds)")
        println("  clamp_to_bounds: $(step.clamp_to_bounds)")
        println("  has aux_func: $(step.aux_func! !== nothing)")
        println("  has error_func: $(step.error_func! !== nothing)")
        println("  aux_write_indices: $(step.aux_write_indices)")
    elseif step isa MacroModelling.NumericalNSSSStep
        println("  block_index: $(step.block_index)")
        println("  write_indices: $(step.write_indices)")
        println("  param_gather_indices: $(step.param_gather_indices)")
        println("  var_gather_indices: $(step.var_gather_indices)")
        println("  lbs: $(step.lbs[1:min(5,length(step.lbs))])")
        println("  ubs: $(step.ubs[1:min(5,length(step.ubs))])")
    end
end

# Now trace execution step by step
println("\n=== STEP-BY-STEP EXECUTION ===")
params_vec = Vector{Float64}(undef, nsss.n_ext_params)
nsss.param_prep!(params_vec, Gali_2015_chapter_3_nonlinear.parameter_values)
println("params_vec: ", params_vec)

sol_vec = zeros(Float64, nsss.n_sol)
tol = MacroModelling.Tolerances()

for (i, step) in enumerate(nsss.solve_steps)
    # Save sol_vec before
    sol_before = copy(sol_vec)
    
    step_error, step_iters, step_cache = MacroModelling.execute_step!(
        step, sol_vec, params_vec, 
        Gali_2015_chapter_3_nonlinear.caches.solver_cache[end],
        Gali_2015_chapter_3_nonlinear, tol, false, false, 
        MacroModelling.solver_parameters[], false
    )
    
    # Print which indices changed
    changed = findall(sol_vec .!= sol_before)
    println("Step $i ($(step.description)): error=$step_error")
    for idx in changed
        println("  sol_vec[$idx]: $(sol_before[idx]) → $(sol_vec[idx])")
    end 
    
    # For the Pi_star step, show more detail
    if contains(step.description, "Pi_star") && step isa MacroModelling.AnalyticalNSSSStep
        println("  --- DETAILED Pi_star analysis ---")
        if step.aux_func! !== nothing
            step.aux_func!(step.aux_buffer, sol_vec, params_vec)
            println("  aux_buffer (raw): $(step.aux_buffer)")
            println("  aux_write_indices: $(step.aux_write_indices)")
            for (j, idx) in enumerate(step.aux_write_indices)
                println("  aux[$j] → sol_vec[$idx] = $(sol_vec[idx])")
            end
        end
        if step.error_func! !== nothing
            # Recompute error with current sol_vec
            step.error_func!(step.error_buffer, sol_vec, params_vec)
            println("  error_buffer: $(step.error_buffer)")
        end
        step.eval_func!(step.buffer, sol_vec, params_vec)
        println("  main eval buffer (raw): $(step.buffer)")
        println("  write_indices: $(step.write_indices)")
    end
    
    if step_error > tol.NSSS_acceptance_tol
        println("  *** WOULD BREAK HERE (error $step_error > $(tol.NSSS_acceptance_tol)) ***")
        # Continue anyway to see what would happen
    end
end

println("\n=== FINAL sol_vec ===")
println(sol_vec)

# Also show the output mapping
println("\n=== OUTPUT INDICES ===")
println("output_indices: ", nsss.output_indices)
println("n_sol: ", nsss.n_sol)
println("n_ext_params: ", nsss.n_ext_params)
