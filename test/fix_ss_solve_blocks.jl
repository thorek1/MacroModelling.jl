using MacroModelling

include("../test/models/RBC_CME.jl")

include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")
m.SS_solve_func
m.ss_solve_blocks
m.ss_solve_blocks_in_place[1].ss_problem.jac_buffer

m.ss_solve_blocks_in_place[1].extended_ss_problem.jac_buffer

m.NSSS_solver_cache[end][1]

m.SS_solve_func

guess = [ 0.15666405734651248
1.2082329932960632
0.022590361445783146
9.438431853512776
1.4214505803483093]


import MacroModelling: Tolerances, block_solver, levenberg_marquardt, solve_ss, newton
import LinearAlgebra as ℒ
import DataStructures: CircularBuffer

𝓂 = m
tol = Tolerances()
verbose = true
initial_parameters = 𝓂.parameter_values
cold_start = true
solver_parameters  = 𝓂.solver_parameters

# initial_parameters, 𝓂, tol, verbose, cold_start, solver_parameters)->begin
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3671 =#
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3678 =#
          initial_parameters = if typeof(initial_parameters) == Vector{Float64}
                  initial_parameters
              else
                  ℱ.value.(initial_parameters)
              end
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3680 =#
          initial_parameters_tmp = copy(initial_parameters)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3682 =#
          parameters = copy(initial_parameters)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3683 =#
          params_flt = copy(initial_parameters)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3685 =#
          current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - initial_parameters)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3686 =#
          closest_solution_init = 𝓂.NSSS_solver_cache[end]
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3688 =#
          for pars = 𝓂.NSSS_solver_cache
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3689 =#
              copy!(initial_parameters_tmp, pars[end])
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3691 =#
              ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3693 =#
              latest = sum(abs2, initial_parameters_tmp)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3694 =#
              if latest <= current_best
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3695 =#
                  current_best = latest
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3696 =#
                  closest_solution_init = pars
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3698 =#
          end
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3703 =#
          range_iters = 0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3704 =#
          solution_error = 1.0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
          solved_scale = 0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3707 =#
          scale = 1.0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3709 =#
          NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3710 =#
          push!(NSSS_solver_cache_scale, closest_solution_init)
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3712 =#
        #   while range_iters <= if cold_start
        #                   1
        #               else
        #                   500
        #               end && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3713 =#
              range_iters += 1
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3714 =#
              fail_fast_solvers_only = if range_iters > 1
                      true
                  else
                      false
                  end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
              if abs(solved_scale - scale) < 0.0001
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3718 =#
                #   break
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3731 =#
              current_best = sum(abs2, (NSSS_solver_cache_scale[end])[end] - initial_parameters)
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3732 =#
              closest_solution = NSSS_solver_cache_scale[end]
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3734 =#
              for pars = NSSS_solver_cache_scale
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3735 =#
                  copy!(initial_parameters_tmp, pars[end])
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3737 =#
                  ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3739 =#
                  latest = sum(abs2, initial_parameters_tmp)
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3741 =#
                  if latest <= current_best
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3742 =#
                      current_best = latest
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3743 =#
                      closest_solution = pars
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3745 =#
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3749 =#
              if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3750 =#
                  parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
              else
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3752 =#
                  parameters = copy(initial_parameters)
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3754 =#
              params_flt = parameters
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3758 =#
              cap_share = parameters[1]
              R_ss = parameters[2]
              I_K_ratio = parameters[3]
              phi_pi = parameters[4]
              Pi_real = parameters[7]
              rhoz = parameters[8]
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3759 =#
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3760 =#
              Pi_ss = R_ss - Pi_real
              rho_z_delta = rhoz
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3761 =#
              NSSS_solver_cache_tmp = []
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3762 =#
              solution_error = 0.0
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3763 =#
              iters = 0
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3764 =#
              Pi = Pi_ss
              ➕₃ = min(600, max(-1.0e12, R_ss - 1))
              solution_error += +(abs(➕₃ - (R_ss - 1)))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                      println("Failed for analytical variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                #   continue
              end
              R = exp(➕₃)
              beta = Pi / R
              solution_error += abs(min(max(1.1920928955078125e-7, beta), 0.9999998807907104) - beta)
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3527 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3527 =#
                      println("Failed for bounded variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3527 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3527 =#
                #   continue
              end
              ➕₁ = min(max(2.220446049250313e-16, (R * beta) ^ (1 / phi_pi)), 1.0e12)
              solution_error += abs(➕₁ - (R * beta) ^ (1 / phi_pi))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                  if verbose
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                #   continue
              end
              Pibar = Pi / ➕₁
              z_delta = 1
              A = 1
              params_and_solved_vars = [cap_share, I_K_ratio, beta]
              lbs = [1.1920928955078125e-7, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, 1.1920928955078125e-7]
              ubs = [0.9999998807907104, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 0.9999998807907104]
              inits = [max.(lbs[1:length(closest_solution[1])], min.(ubs[1:length(closest_solution[1])], closest_solution[1])), closest_solution[2]]


              solution = block_solver(params_and_solved_vars, 1, 𝓂.ss_solve_blocks_in_place[1], inits, lbs, ubs, solver_parameters, fail_fast_solvers_only, cold_start, verbose)


              SS_solve_block = 𝓂.ss_solve_blocks_in_place[1];

              parameters_and_solved_vars = params_and_solved_vars

              guess_and_pars_solved_vars = inits

              n_block = 1

    solved_yet = false

    guess = guess_and_pars_solved_vars[1]

    sol_values = guess

    closest_parameters_and_solved_vars = sum(abs, guess_and_pars_solved_vars[2]) == Inf ? parameters_and_solved_vars : guess_and_pars_solved_vars[2]

    # res = ss_solve_blocks(parameters_and_solved_vars, guess)

    SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.func_buffer, guess, parameters_and_solved_vars, 0) # TODO: make the block a struct
    # TODO: do the function creation with Symbolics as this will solve the compilation bottleneck for large functions

    res = SS_solve_block.ss_problem.func_buffer

    sol_minimum  = ℒ.norm(res)

    total_iters = [0,0]

    SS_optimizer = levenberg_marquardt

    SS_optimizer = newton

    # if cold_start
        guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only

        g = guesses[1]
        # g = rand(5) .+ 1
        p = solver_parameters[1]
        ext = false

        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
        g, 
        p,
        ext,
        false)

        sol_minimum < tol.NSSS_acceptance_tol
        SS_solve_block.ss_problem.jac_buffer

        SS_solve_block.ss_problem.func_buffer

        for g in guesses
            for p in solver_parameters
                for ext in [true, false] # try first the system where values and parameters can vary, next try the system where only values can vary
                    if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol# || rel_sol_minimum > rtol
                        if solved_yet continue end
                        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                            g, 
                                                            p,
                                                            ext,
                                                            false)
                        if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                            solved_yet = true
                        end
                    end
                end
            end
        end



              iters += (solution[2])[2]
              solution_error += (solution[2])[1]

              if solution_error > tol.NSSS_acceptance_tol
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2926 =#
                if verbose
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2926 =#
                    println("Failed after solving block with error $(solution_error)")
                end
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2926 =#
                scale = scale * 0.3 + solved_scale * 0.7
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2926 =#
                # continue
            end
            sol = solution[1]
            alpha = sol[1]
            c = sol[2]
            delta = sol[3]
            k = sol[4]
            y = sol[5]
            NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
                        sol
                    else
                        ℱ.value.(sol)
                    end]
            NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
                        params_and_solved_vars
                    else
                        ℱ.value.(params_and_solved_vars)
                    end]
            ➕₂ = min(max(2.220446049250313e-16, c - 1), 0.9999999999999998)
            solution_error += abs(➕₂ - (c - 1))
            if solution_error > tol.NSSS_acceptance_tol
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                if verbose
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                    println("Failed for analytical aux variables with error $(solution_error)")
                end
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                scale = scale * 0.3 + solved_scale * 0.7
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3508 =#
                # continue
            end
            ➕₄ = min(2 - eps(), max(eps(), 2.0➕₂))
            solution_error += +(abs(➕₄ - 2.0➕₂))
            if solution_error > tol.NSSS_acceptance_tol
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                if verbose
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                    println("Failed for analytical variables with error $(solution_error)")
                end
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                scale = scale * 0.3 + solved_scale * 0.7
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3519 =#
                continue
            end
            c_norminvcdf = -1.4142135623731 * erfcinv(➕₄)
            c_normlogpdf = -0.5 * c ^ 2 - 0.918938533204673
            log_ZZ_avg = 0
            ZZ_avg_fut = 1
            ZZ_avg = 1
            eps_z = 0
            eps_zᴸ⁽⁻¹⁾ = 0
            eps_zᴸ⁽¹⁾ = 0
            if length(NSSS_solver_cache_tmp) == 0
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3641 =#
                NSSS_solver_cache_tmp = [copy(params_flt)]
            else
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3641 =#
                NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)]
            end
            if current_best > 1.0e-8 && (solution_error < tol.NSSS_acceptance_tol && scale == 1)
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3651 =#
                reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3766 =#
            if solution_error < tol.NSSS_acceptance_tol
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3768 =#
                solved_scale = scale
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3769 =#
                if scale == 1
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3773 =#
                    return ([A, A, A, Pi, R, ZZ_avg, ZZ_avg_fut, c, c_norminvcdf, c_normlogpdf, eps_z, eps_z, eps_z, k, log_ZZ_avg, y, z_delta, beta, Pibar, alpha, delta], (solution_error, iters))
                else
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3775 =#
                    reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                end
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3778 =#
                if scale > 0.95
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3779 =#
                    scale = 1
                else
                    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3782 =#
                    scale = scale * 0.4 + 0.6
                end
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3792 =#
        end
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3793 =#
        return ([0.0], (1, 0))