using Revise
# ENV["JULIA_CONDAPKG_BACKEND"] = "MicroMamba"
using MacroModelling
using BenchmarkTools
import MacroModelling: clear_solution_caches!


@model FS2000 precompile = true begin
    dA[0] = exp(gam + z_e_a  *  e_a[x])
    log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  e_m[x]
    - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0
    W[0] = l[0] / n[0]
    - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0
    R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]
    1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0
    c[0] + k[0] = exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  *  e_a[x])) * k[-1]
    P[0] * c[0] = m[0]
    m[0] - 1 + d[0] = l[0]
    e[0] = exp(z_e_a  *  e_a[x])
    y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x]))
    gy_obs[0] = dA[0] * y[0] / y[-1]
    gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]
    log_gy_obs[0] = log(gy_obs[0])
    log_gp_obs[0] = log(gp_obs[0])
end

@parameters FS2000 silent = true precompile = true begin  
    alp     = 0.356
    bet     = 0.993
    gam     = 0.0085
    mst     = 1.0002
    rho     = 0.129
    psi     = 0.65
    del     = 0.01
    z_e_a   = 0.035449
    z_e_m   = 0.008862
end
m = FS2000
m = NAWM_EAUS_2008 
@benchmark begin
    # clear_solution_caches!(m, :first_order)
    get_SS(m, silent = true, derivatives = false)
end setup = clear_solution_caches!(m, :first_order)

clear_solution_caches!(m, :first_order)

@profview for i in 1:10
    clear_solution_caches!(m, :first_order)
    get_SS(m, silent = true, derivatives = false)
end

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

# [0.9999998807907104, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
# MacroModelling.solver_parameters(1.9479518608134938, 0.02343520604394183, 5.125002799990568, 0.02387522857907376, 0.2239226474715968, 4.889172213411495, 1.747880258818237, 2.8683242331457, 0.938229356687311, 1.4890887655876235, 1.6261504814901664, 11.26863249187599, 36.05486169712279, 6.091535897587629, 11.73936761697657, 3.189349432626493, 0.21045178305336348, 0.17122196312330415, 13.251662547139363, 5.282429995876679, 1, 0.0, 2)
# true
a ≈ b
a = [-12.433767642918585 0.0 0.0 -5.3762485340939765 5.3762502416961295 0.0 0.0 0.0; 12.433767642918585 -5.3762502416961295 -28.399665542074878 -23.0234170079809 0.0 0.0 0.0 0.0; -3.764232002406259 0.0 5.370859922641225 1.2120486067971766e-7 0.0 0.0 0.0 0.0; 0.0 1.0177608119544763 0.0 0.0 -1.0177608119544763 0.0 0.0 0.0; 0.0 0.0 0.0 0.2544402029886191 -0.2544402029886191 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 -1.937937047481161 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 -1.0111874208078342 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.4135047829953282]
b = [-12.433767642918587 0.0 0.0 -5.3762485340939765 5.3762502416961295 0.0 0.0 0.0; 12.433767642918587 -5.3762502416961295 -28.399665542074878 -23.023417007980903 0.0 0.0 0.0 0.0; 0.0 1.0177608119544763 0.0 0.0 -1.0177608119544763 0.0 0.0 0.0; -3.764232002406259 -0.0 5.370859922641225 1.2120486067971766e-7 -0.0 -0.0 -0.0 -0.0; 0.0 0.0 0.0 0.2544402029886191 -0.2544402029886191 0.0 0.0 0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -1.937937047481161 -0.0 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -1.0111874208078342 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -1.4135047829953282]
[0.8813735027258436, 2.366350881077193, 2.3663746058124353, 2.366374605151466, 2.366374573597027, 1.2803606388441917, 0.14944312018495765, 0.8806644521167746]

[0.8813735027258436, 2.366350881077193, 2.3663746058124353, 2.366374605151466, 2.366374573597027, 1.2803606388441917, 0.14944312018495765, 0.8806644521167746]

# [-12.433767626388102 0.0 0.0 -5.376248529783475 5.376250070701516 0.0 0.0 0.0; 12.433767626388102 -5.376124918137118 -28.399665514162745 -23.023416985040242 0.0 0.0 0.0 0.0; 0.0 1.0177371208803547 0.0 0.0 -1.0177362718097986 0.0 0.0 0.0; -3.7642320012329327 -0.0 5.370859921823067 1.2120486068320023e-7 -0.0 -0.0 -0.0 -0.0; 0.0 0.0 0.0 0.2544402111672444 -0.25444021144994905 0.0 0.0 0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -1.937937047481161 -0.0 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -1.0111874208078342 -0.0; -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -1.4135047829953282]
# [0.8414630548452718, 2.067702729890952, 1.5444697302150523, 2.217027361455363, 2.120867377098202, 1.2803606388441917, 0.14944312018495765, 0.8806644521167747]
# [0.8414630548452863, 2.067702729890989, 1.5444697302150652, 2.217027361455346, 2.120867377098219, 1.2803606388441917, 0.14944312018495765, 0.8806644521167747]

[0.15603018290473208, 1.0212331164230357, 0.022588440484669205, 2.940731629734951, 1.1503870139467807, 1.2803606388441915, 0.14944312018495765, 0.8806644521167747]
[0.15603018290473153, 1.0212331164230348, 0.02258844048466909, 2.9407316297349495, 1.1503870139467793, 1.2803606388441915, 0.14944312018495765, 0.8806644521167747]

import MacroModelling: Tolerances, block_solver, levenberg_marquardt, solve_ss, newton
import LinearAlgebra as ℒ
import DataStructures: CircularBuffer

𝓂 = m
tol = Tolerances()
verbose = true
initial_parameters = 𝓂.parameter_values
cold_start = true
solver_parameters  = 𝓂.solver_parameters[2:end]

          initial_parameters = if typeof(initial_parameters) == Vector{Float64}
                  initial_parameters
              else
                  ℱ.value.(initial_parameters)
              end
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3681 =#
          initial_parameters_tmp = copy(initial_parameters)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3683 =#
          parameters = copy(initial_parameters)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3684 =#
          params_flt = copy(initial_parameters)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3686 =#
          current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - initial_parameters)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3687 =#
          closest_solution_init = 𝓂.NSSS_solver_cache[end]
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3689 =#
          for pars = 𝓂.NSSS_solver_cache
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3690 =#
              copy!(initial_parameters_tmp, pars[end])
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3692 =#
              ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3694 =#
              latest = sum(abs2, initial_parameters_tmp)
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3695 =#
              if latest <= current_best
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3696 =#
                  current_best = latest
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3697 =#
                  closest_solution_init = pars
              end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3699 =#
          end
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3704 =#
          range_iters = 0
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3705 =#
          solution_error = 1.0
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3706 =#
          solved_scale = 0
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3708 =#
          scale = 1.0
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3710 =#
          NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3711 =#
          push!(NSSS_solver_cache_scale, closest_solution_init)
          #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3713 =#
        #   while range_iters <= if cold_start
        #                   1
        #               else
        #                   500
        #               end && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3714 =#
              range_iters += 1
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3715 =#
              fail_fast_solvers_only = if range_iters > 1
                      true
                  else
                      false
                  end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3717 =#
            #   if abs(solved_scale - scale) < 0.0001
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3719 =#
                #   break
            #   end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3732 =#
              current_best = sum(abs2, (NSSS_solver_cache_scale[end])[end] - initial_parameters)
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3733 =#
              closest_solution = NSSS_solver_cache_scale[end]
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3735 =#
              for pars = NSSS_solver_cache_scale
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3736 =#
                  copy!(initial_parameters_tmp, pars[end])
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3738 =#
                  ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3740 =#
                  latest = sum(abs2, initial_parameters_tmp)
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3742 =#
                  if latest <= current_best
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3743 =#
                      current_best = latest
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3744 =#
                      closest_solution = pars
                  end
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3746 =#
              end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3750 =#
              if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3751 =#
                  parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
              else
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3753 =#
                  parameters = copy(initial_parameters)
              end
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3755 =#
              params_flt = parameters
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3759 =#
              cap_share = parameters[1]
              R_ss = parameters[2]
              I_K_ratio = parameters[3]
              phi_pi = parameters[4]
              Pi_real = parameters[7]
              rhoz = parameters[8]
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3760 =#
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3761 =#
              Pi_ss = R_ss - Pi_real
              rho_z_delta = rhoz
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3762 =#
              NSSS_solver_cache_tmp = []
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3763 =#
              solution_error = 0.0
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3764 =#
              iters = 0
              #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3765 =#
              Pi = Pi_ss
              ➕₃ = min(600, max(-1.0e12, R_ss - 1))
              solution_error += +(abs(➕₃ - (R_ss - 1)))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3520 =#
                  if verbose
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3520 =#
                      println("Failed for analytical variables with error $(solution_error)")
                  end
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3520 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3520 =#
                #   continue
              end
              R = exp(➕₃)
              beta = Pi / R
              solution_error += abs(min(max(1.1920928955078125e-7, beta), 0.9999998807907104) - beta)
              if solution_error > tol.NSSS_acceptance_tol
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3528 =#
                  if verbose
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3528 =#
                      println("Failed for bounded variables with error $(solution_error)")
                  end
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3528 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3528 =#
                #   continue
              end
              ➕₁ = min(max(2.220446049250313e-16, (R * beta) ^ (1 / phi_pi)), 1.0e12)
              solution_error += abs(➕₁ - (R * beta) ^ (1 / phi_pi))
              if solution_error > tol.NSSS_acceptance_tol
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3509 =#
                  if verbose
                      #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3509 =#
                      println("Failed for analytical aux variables with error $(solution_error)")
                  end
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3509 =#
                  scale = scale * 0.3 + solved_scale * 0.7
                  #= /home/cdsw/MacroModelling.jl/src/MacroModelling.jl:3509 =#
                #   continue
              end
              Pibar = Pi / ➕₁
              A = 1
              z_delta = 1
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
          
                  rel_sol_minimum = 1.0
              
              if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                  solved_yet = true
          
                  if verbose
                      println("Block: $n_block, - Solved using previous solution; residual norm: $sol_minimum")
                  end
              end
          
              total_iters = [0,0]
          
              SS_optimizer = levenberg_marquardt
              SS_optimizer = newton

              
              guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only

              sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
              # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
              guesses[1], 
              solver_parameters[1],
                                                  false,
                                                  false)




              for g in guesses
                  for p in solver_parameters
                      for ext in [true, false] # try first the system where values and parameters can vary, next try the system where only values can vary
                          if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol# || rel_sol_minimum > rtol
                              if solved_yet continue end
      
                              sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                              # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
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