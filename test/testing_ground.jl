
using MacroModelling
import ForwardDiff as ℱ
using Optimization, OptimizationNLopt

context = dynare_preprocess("./test/models/example1", [])
context = @dynare "./test/models/example1";

@model finacc begin
    R[0] * beta = C[1] / C[0]
    C[0] = w[0] * L[0] - B[0] + R[-1] * B[-1] + (1-v) * (Rk[-1] * Q[-1] * K[-1] - (R[-1] + mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1])) * (Q[-1] * K[-1] - N[-1])) - We 
    w[0] = C[0] / (1-L[0])
    K[0] = (1-delta) * K[-1] + I[0] 
    Q[0] = 1 + chi * (I[0] / K[-1] - delta)
    Y[0] = A[0] * K[-1]^alpha * L[0]^(1-alpha)
    Rk[-1] = (alpha * Y[0] / K[-1] + Q[0] * (1-delta))/Q[-1]
    w[0] = (1-alpha) * Y[0] / L[0]
    N[0] = v * (Rk[-1] * Q[-1] * K[-1] - (R[-1] + mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1])) * (Q[-1] * K[-1] - N[-1])) + We 
    0 = (omegabar[0] * (1 - F[0]) + (1 - mu) * G[0]) * Rk[0] / R[0] * Q[0] * K[0] / N[0] - (Q[0] * K[0] / N[0] - 1)
    0 = (1 - (omegabar[0] * (1 - F[0]) + G[0])) * Rk[0] / R[0] + (1 - F[0]) / (1 - F[0] - omegabar[0] * mu * (normpdf((log(omegabar[0]) + sigma^2/2)  / sigma)/ omegabar[0]  / sigma)) * ((omegabar[0] * (1 - F[0]) + (1 - mu) * G[0]) * Rk[0] / R[0] - 1) 
    G[0] = normcdf(((log(omegabar[0])+sigma^2/2)/sigma) - sigma)
    F[0] = normcdf((log(omegabar[0])+sigma^2/2)/sigma)
    EFP[0] = (mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1]))
    Y[0] + walras[0] = C[0] + I[0] + EFP[0] * (Q[-1] * K[-1] - N[-1])
    B[0] = Q[0] * K[0] - N[0]  
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end


@macroexpand @parameters finacc begin
    beta = 0.99 
    delta = 0.02 
    We = 1e-12 
    alpha = 0.36 
    chi = 0 
    v = 0.978
    mu = 0.94
    sigma = 0.2449489742783178
    rhoz = .9
    std_eps = .0068

    .5 > omegabar > .44
    K > 15
    0 < L < .45
end


    solve!(finacc)
    𝓂 = finacc
    parameters = finacc.parameter_values

# transformation of NSSS problem
function transformer(x)
    # return asinh.(asinh.(asinh.(x)))
    return asinh.(asinh.(x))
    # return x
end

function undo_transformer(x)
    # return sinh.(sinh.(sinh.(x)))
    return sinh.(sinh.(x))
    # return x
end


function block_solver(parameters_and_solved_vars::Vector{Float64}, 
    n_block::Int, 
    ss_solve_blocks::Function, 
    # SS_optimizer, 
    f::OptimizationFunction, 
    guess::Vector{Float64}, 
    lbs::Vector{Float64}, 
    ubs::Vector{Float64};
    tol = eps(Float32),
    maxtime = 120,
    maxiters = 100000,
    multisolver = true)

# # try whatever solver you gave first
# # try catch in order to capture possible issues with bounds (ADAM doesnt work with bounds)
# sol = try 
#     solve(OptimizationProblem(f, guess, parameters_and_solved_vars, lb = lbs, ub = ubs), SS_optimizer(), maxtime = maxtime, maxiters = maxiters)
# catch e
#     solve(OptimizationProblem(f, guess, parameters_and_solved_vars), SS_optimizer(), maxtime = maxtime, maxiters = maxiters)
# end

# # if the provided guess does not work, try the standard starting point
# if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol, parameters_and_solved_vars)) > tol)
#     sol = try 
#         solve(OptimizationProblem(f, fill(.9,length(guess)), parameters_and_solved_vars, lb = lbs, ub = ubs), SS_optimizer(), maxtime = maxtime, maxiters = maxiters)
#     catch e
#         solve(OptimizationProblem(f, fill(.9,length(guess)), parameters_and_solved_vars), SS_optimizer(), maxtime = maxtime, maxiters = maxiters)
#     end
# end

# previous_SS_optimizer = SS_optimizer


# if the solver input was not L-BFGS and it didnt converge try that one next
# if ((sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)) && SS_optimizer != NLopt.LD_LBFGS
SS_optimizer = NLopt.LD_LBFGS

# println("Block: ",n_block," - Solution not found with ",string(previous_SS_optimizer),": maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying optimizer: ",string(SS_optimizer))

prob = OptimizationProblem(f, guess, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)

# if the previous non-converged best guess as a starting point does not work, try the standard starting point
if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol, parameters_and_solved_vars)) > tol)
standard_inits = max.(lbs,min.(ubs, fill(transformer(.9),length(guess))))
prob = OptimizationProblem(f, standard_inits, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)
end

# # if the the standard starting pointdoesnt work try the provided guess
# if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
#     prob = OptimizationProblem(f, guess, parameters_and_solved_vars, lb = lbs, ub = ubs)
#     sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)
# end

# if sol_new.minimum < sol.minimum
#     sol = sol_new
# end

previous_SS_optimizer = NLopt.LD_LBFGS
# end


# interpolate between closest converged solution and the new parametrisation
# if ((sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)) && isfinite(guess[2])
#     println("Block: ",n_block," - Solution not found with ",string(previous_SS_optimizer),": maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying interpolation.")
#     range_length = [2^i for i in 1:5]

#     for r in range_length
#         for scale in range(0,1,r+1)[2:end]
#             prob = OptimizationProblem(f, guess, scale * parameters_and_solved_vars + (1 - scale) * guess[2], lb = lbs, ub = ubs)
#             sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)

#             if sol_new.minimum < tol
#                 sol = sol_new
#                 break
#             end
#         end

#         if sol_new.minimum < tol
#             sol = sol_new
#             break
#         end
#     end

# end


# if SLSQP didnt converge try BOBYQA
if multisolver && ((sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol))
SS_optimizer = NLopt.LN_BOBYQA

println("Block: ",n_block," - Solution not found with ",string(previous_SS_optimizer),": maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying optimizer: ",string(SS_optimizer))

previous_sol_init = max.(lbs,min.(ubs, transformer(sol.u)))
prob = OptimizationProblem(f, previous_sol_init, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)

# if the previous non-converged best guess as a starting point does not work, try the standard starting point
if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
standard_inits = max.(lbs,min.(ubs, fill(transformer(.9),length(guess))))
prob = OptimizationProblem(f, standard_inits, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)
end

# if the the standard starting pointdoesnt work try the provided guess
if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
prob = OptimizationProblem(f, guess, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)
end

if sol_new.minimum < sol.minimum
sol = sol_new
end

previous_SS_optimizer = NLopt.LN_BOBYQA
end


# if L-BFGS didnt converge try SLSQP
if multisolver && ((sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol))
SS_optimizer = NLopt.LD_SLSQP

println("Block: ",n_block," - Solution not found with ",string(previous_SS_optimizer),": maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying optimizer: ",string(SS_optimizer))

previous_sol_init = max.(lbs,min.(ubs, transformer(sol.u)))
prob = OptimizationProblem(f, previous_sol_init, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)

# if the previous non-converged best guess as a starting point does not work, try the standard starting point
if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
standard_inits = max.(lbs,min.(ubs, fill(transformer(.9),length(guess))))
prob = OptimizationProblem(f, standard_inits, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)
end

# if the the standard starting pointdoesnt work try the provided guess
if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
prob = OptimizationProblem(f, guess, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)
end

if sol_new.minimum < sol.minimum
sol = sol_new
end

previous_SS_optimizer = NLopt.LD_SLSQP
end


# if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)
#     SS_optimizer = Optimisers.ADAM

#     println("Block: ",n_block," - Solution not found with ",string(previous_SS_optimizer),": maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying optimizer: ",string(SS_optimizer))

#     prob = OptimizationProblem(f, sol.u, parameters_and_solved_vars)#, lb = lbs, ub = ubs)
#     sol_new = solve(prob, SS_optimizer(), maxiters = maxiters)

#     if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
#         prob = OptimizationProblem(f, guess, parameters_and_solved_vars)#, lb = lbs, ub = ubs)
#         sol_new = solve(prob, SS_optimizer(), maxiters = maxiters)
#     end

#     if sol_new.minimum < sol.minimum
#         sol = sol_new
#     end

#     previous_SS_optimizer = Optimisers.ADAM
# end



# if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)
#     SS_optimizer = NLopt.LN_SBPLX

#     println("Block: ",n_block," - Solution not found with ",string(previous_SS_optimizer),": maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying optimizer: ",string(SS_optimizer))

#     prob = OptimizationProblem(f, sol.u, parameters_and_solved_vars, lb = lbs, ub = ubs)
#     sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)

#     if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
#         prob = OptimizationProblem(f, guess, parameters_and_solved_vars, lb = lbs, ub = ubs)
#         sol_new = solve(prob, SS_optimizer(), local_maxtime = maxtime, maxtime = maxtime)
#     end

#     if sol_new.minimum < sol.minimum
#         sol = sol_new
#     end

#     previous_SS_optimizer = NLopt.LN_SBPLX
# end



# if BOBYQA didnt converge try G_MLSL_LDS with BOBYQA
if multisolver && ((sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol))
println("Block: ",n_block," - Solution not found with ",string(previous_SS_optimizer),": maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying global solution.")

previous_sol_init = max.(lbs,min.(ubs, transformer(sol.u)))
prob = OptimizationProblem(f, previous_sol_init, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LN_BOBYQA(), population = length(ubs), local_maxtime = maxtime, maxtime = maxtime)

# if the previous non-converged best guess as a starting point does not work, try the standard starting point
if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
standard_inits = max.(lbs,min.(ubs, fill(transformer(.9),length(guess))))
prob = OptimizationProblem(f, standard_inits, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LN_BOBYQA(), population = length(ubs), local_maxtime = maxtime, maxtime = maxtime)
end

# if the the standard starting pointdoesnt work try the provided guess
if (sol_new.minimum > tol) | (maximum(abs,ss_solve_blocks(sol_new, parameters_and_solved_vars)) > tol)
prob = OptimizationProblem(f, guess, parameters_and_solved_vars, lb = lbs, ub = ubs)
sol_new = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LN_BOBYQA(), population = length(ubs), local_maxtime = maxtime, maxtime = maxtime)
end

if sol_new.minimum < sol.minimum
sol = sol_new
end
end

# if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)
#     println("Block: ",n_block," - Solution not found: maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying with positive domain.")

#     inits = max.(max.(lbs,eps()),min.(ubs,guess))
#     prob = OptimizationProblem(f, guess, inits, lb = max.(lbs,eps()), ub = ubs)
#     sol = solve(prob, SS_optimizer())
# end

# if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)
#     println("Block: ",n_block," - Solution not found: maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying optimizer: LN_BOBYQA.")
#     sol = solve(prob, NLopt.LN_BOBYQA(), local_maxtime = maxtime, maxtime = maxtime)
# end

# if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)
#     println("Block: ",n_block," - Solution not found: maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying optimizer: LN_SBPLX.")
#     sol = solve(prob, NLopt.LN_SBPLX(), local_maxtime = maxtime, maxtime = maxtime)
# end

# if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)
#     println("Block: ",n_block," - Local solution not found: maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Trying global solution.")
#     sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LN_BOBYQA(), population = length(ubs), local_maxtime = maxtime, maxtime = maxtime)
# end

# if none worked return an error
# if (sol.minimum > tol) | (maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)) > tol)
#     return error("Block: ",n_block," - Solution not found with G_MLSL_LDS and LN_BOBYQA: maximum residual = ",maximum(abs,ss_solve_blocks(sol,parameters_and_solved_vars)),". Consider changing bounds.")
# end

return sol.u, sol.minimum
end




        #   range_length = [Base.Generator for $(Expr(:opaque_closure, :(i->2 ^ i))), 0:5]
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:602 =#
          params_flt = if typeof(parameters) == Vector{Float64}
                  parameters
              else
                  ℱ.value.(parameters)
              end
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:603 =#
          closest_solution_init = 𝓂.NSSS_solver_cache[1]
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:606 =#
          solved_scale = 0
          #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:607 =#
        #   for r = range_length
        #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:608 =#
        #       for scale = (range(0, 1, r + 1))[2:end]
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:609 =#
                #   if scale <= solved_scale
                #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:609 =#
                #       continue
                #   end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:610 =#
                  closest_solution = 𝓂.NSSS_solver_cache[1]
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:611 =#
                  params = if all(isfinite.(closest_solution_init[end]))
                          scale * parameters + (1 - scale) * closest_solution_init[end]
                      else
                          parameters
                      end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:612 =#
                  params_scaled_flt = if typeof(params) == Vector{Float64}
                          params
                      else
                          ℱ.value.(params)
                      end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:614 =#
                  beta = params[1]
                  delta = params[2]
                  We = params[3]
                  alpha = params[4]
                  chi = params[5]
                  v = params[6]
                  mu = params[7]
                  sigma = params[8]
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:615 =#
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:616 =#
                  NSSS_solver_cache_tmp = []
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:617 =#
                  solution_error = 0.0
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:618 =#
                  R = 1 / beta
                  A = 1
                  lbs = [-1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 15.0, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 0.4400000000000002, -1.0e12]
                  ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 0.4499999999999998, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 0.4999999999999998, 1.0e12]
                  f = OptimizationFunction(𝓂.ss_solve_blocks_optim[1], Optimization.AutoForwardDiff())
                  inits = max.(lbs, min.(ubs, closest_solution[1]))
                  solution = block_solver([delta, We, alpha, chi, v, mu, sigma, R], 1, 𝓂.ss_solve_blocks[1], f, transformer(inits), transformer(lbs), transformer(ubs), multisolver = true)
                  solution_error = solution[2]
                  sol = undo_transformer(solution[1])
                  println(sol)
                  B = sol[1]
                  C = sol[2]
                  F = sol[3]
                  G = sol[4]
                  I = sol[5]
                  K = sol[6]
                  L = sol[7]
                  N = sol[8]
                  Q = sol[9]
                  Rk = sol[10]
                  Y = sol[11]
                  omegabar = sol[12]
                  w = sol[13]
                  omegabar = min(max(omegabar, 𝓂.lower_bounds[1]), 𝓂.upper_bounds[1])
                  K = min(max(K, 𝓂.lower_bounds[2]), 𝓂.upper_bounds[2])
                  L = min(max(L, 𝓂.lower_bounds[3]), 𝓂.upper_bounds[3])
                  push!(NSSS_solver_cache_tmp, if typeof(sol) == Vector{Float64}
                          sol
                      else
                          ℱ.value.(sol)
                      end)
                  EFP = (G * K * Q * Rk * mu) / (K * Q - N)
                  walras = (((C + EFP * K * Q) - EFP * N) + I) - Y
                  push!(NSSS_solver_cache_tmp, params_scaled_flt)
                  if minimum([Base.Generator for $(Expr(:opaque_closure, :(pars->sum(abs2, pars[end] - params_scaled_flt)))), 𝓂.NSSS_solver_cache]) > eps(Float64) && solution_error < 1.0e-6
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:589 =#
                      push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:590 =#
                      solved_scale = scale
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:621 =#
                  if solution_error < 1.0e-6 && scale == 1
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:622 =#
                      return ComponentVector([A, B, C, EFP, F, G, I, K, L, N, Q, R, Rk, Y, omegabar, w, walras], Axis([sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]))
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:624 =#
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:625 =#
          end