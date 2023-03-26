# using NLboxsolve



# overlap from NAWM and SW03:
# Row │ iter     tol          μ¹       μ²       p        λ¹       λ²       λᵖ       ρ        iter_2   tol_2        iter_1   tol_1       
#     │ Float64  Float64      Float64  Float64  Float64  Float64  Float64  Float64  Float64  Float64  Float64      Float64  Float64     
#─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#   1 │   107.0  2.18847e-12   1.0e-5   0.0001      2.0      0.4      0.6      0.9      0.1     32.0  8.28254e-9      26.0  6.23204e-10
#   2 │   104.0  7.38964e-13   1.0e-5   0.0001      2.0      0.6      0.6      0.9      0.1     30.0  4.54754e-9      26.0  6.15498e-10
#   3 │    25.0  4.43645e-13   0.0001   1.0e-5      1.4      0.5      0.4      0.9      0.1     17.0  3.04963e-11     23.0  8.73613e-15
#   4 │    83.0  6.77858e-11   0.0001   1.0e-5      1.4      0.7      0.7      0.7      0.1     22.0  9.96674e-9      29.0  5.48059e-9
#   5 │    62.0  2.27374e-13   0.0001   1.0e-5      1.5      0.7      0.4      0.5      0.1     19.0  3.95133e-11     15.0  3.31323e-9
#   6 │    58.0  1.21361e-11   0.0001   1.0e-5      1.6      0.4      0.7      0.7      0.1     29.0  2.78272e-9      29.0  5.74625e-9
#   7 │    47.0  7.61752e-10   0.0001   1.0e-5      1.6      0.5      0.7      0.5      0.1     29.0  4.80484e-9      29.0  6.17047e-9
#   8 │    63.0  3.20881e-11   0.0001   1.0e-5      1.7      0.5      0.7      0.5      0.1     29.0  5.50705e-9      27.0  6.92598e-9
#   9 │    86.0  1.64022e-10   0.0001   1.0e-5      1.9      0.5      0.7      0.9      0.1     57.0  2.39222e-9      31.0  6.84538e-9
#  10 │    46.0  2.38602e-9    0.0001   1.0e-5      1.9      0.6      0.7      0.9      0.1     29.0  3.04e-10        29.0  4.53669e-9 ****
#  11 │    27.0  4.26326e-13   0.0001   0.0001      1.8      0.5      0.4      0.9      0.1     25.0  1.07714e-11     22.0  2.85428e-9
#  12 │    62.0  8.20251e-11   0.0001   0.0001      1.8      0.5      0.5      0.9      0.1     30.0  2.18639e-10     24.0  4.18385e-9


import ForwardDiff as ℱ
import LinearAlgebra as ℒ
method = :lm_ar
verbose = true
transformer_option = 1



function minmax!(x::Vector{Float64},lb::Vector{Float64},ub::Vector{Float64})
    @inbounds for i in eachindex(x)
        x[i] = max(lb[i], min(x[i], ub[i]))
    end
end


transformer_option = 1

function levenberg_marquardt(f::Function, 
    initial_guess::Array{T,1}, 
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1}; 
    xtol::T = eps(), 
    ftol::T = 1e-8, 
    iterations::S = 1000, 
    r::T = .5, 
    ρ::T = .1, 
    p::T = 1.9,
    λ¹::T = .6, 
    λ²::T = .7,
    λᵖ::T = .9, 
    μ¹::T = .0001,
    μ²::T = .00001
    ) where {T <: AbstractFloat, S <: Integer}

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert lower_bounds < upper_bounds

    current_guess = copy(initial_guess)
    previous_guess = similar(current_guess)
    guess_update = similar(current_guess)
    ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))

    ∇̂ = similar(∇)

    largest_step = zero(T)
    largest_residual = zero(T)

	for iter in 1:iterations
        ∇ .= ℱ.jacobian(f,current_guess)

        if !all(isfinite,∇)
            return current_guess, (iter, Inf, Inf, fill(Inf,length(current_guess)))
        end

        previous_guess .= current_guess

        ḡ = sum(abs2,f(previous_guess))

        ∇̂ .= ∇' * ∇

        current_guess .+= -(∇̂ + μ¹ * sum(abs2, f(current_guess))^p * ℒ.I + μ² * ℒ.Diagonal(∇̂)) \ (∇' * f(current_guess))

        minmax!(current_guess, lower_bounds, upper_bounds)

        guess_update .= current_guess - previous_guess

        α = 1.0

        if sum(abs2,f(previous_guess + α * guess_update)) > ρ * ḡ 
            while sum(abs2,f(previous_guess + α * guess_update)) > ḡ  - 0.005 * α^2 * sum(abs2,guess_update)
                α *= r
            end
            μ¹ = μ¹ * λ¹ #max(μ¹ * λ¹, 1e-7)
            μ² = μ² * λ² #max(μ² * λ², 1e-7)
            p = λᵖ * p + (1 - λᵖ) * 1.1
        else
            μ¹ = min(μ¹ / λ¹, 1e-3)
            μ² = min(μ² / λ², 1e-3)
        end

        current_guess .= previous_guess + α * guess_update

        largest_step = maximum(abs, previous_guess - current_guess)
        largest_residual = maximum(abs, f(current_guess))
# println([sum(abs2, f(current_guess)),μ])
        if largest_step <= xtol || largest_residual <= ftol
            return current_guess, (iter, largest_step, largest_residual, f(current_guess))
        end
    end

    return current_guess, (iterations, largest_step, largest_residual, f(current_guess))
end







# transformation of NSSS problem
function transformer(x,lb,ub; option::Int = 2)
    # if option > 1
    #     for i in 1:option
    #         x .= asinh.(x)
    #     end
    #     return x
    if option == 6
        return  @. tanh((x * 2 - (ub + lb) / 2) / (ub - lb)) * (ub - lb) / 2 # project to unbounded
    elseif option == 5
        return asinh.(asinh.(asinh.(asinh.(asinh.(x ./ 100)))))
    elseif option == 4
        return asinh.(asinh.(asinh.(asinh.(x))))
    elseif option == 3
        return asinh.(asinh.(asinh.(x)))
    elseif option == 2
        return asinh.(asinh.(x))
    elseif option == 1
        return asinh.(x)
    elseif option == 0
        return x
    end
end

function undo_transformer(x,lb,ub; option::Int = 2)
    # if option > 1
    #     for i in 1:option
    #         x .= sinh.(x)
    #     end
    #     return x
    if option == 6
        return  @. atanh(x * 2 / (ub - lb)) * (ub - lb) / 2 + (ub + lb) / 2 # project to bounded
    elseif option == 5
        return sinh.(sinh.(sinh.(sinh.(sinh.(x))))) .* 100
    elseif option == 4
        return sinh.(sinh.(sinh.(sinh.(x))))
    elseif option == 3
        return sinh.(sinh.(sinh.(x)))
    elseif option == 2
        return sinh.(sinh.(x))
    elseif option == 1
        return sinh.(x)
    elseif option == 0
        return x
    end
end


ss_solve_blocks = []

push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
    guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:787 =#
    p_star = guess[1]
    pi = guess[2]
    ➕₁ = guess[3]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:788 =#
    alpha = parameters_and_solved_vars[1]
    theta = parameters_and_solved_vars[2]
    epsilon = parameters_and_solved_vars[3]
    var_rho = parameters_and_solved_vars[4]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:789 =#
    s = parameters_and_solved_vars[5]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:794 =#
    return [((-(pi ^ (epsilon / (1 - alpha))) * s * theta) / pi ^ ((epsilon * var_rho) / (1 - alpha)) + s) - (1 - theta) / p_star ^ (epsilon / (1 - alpha)), p_star - ➕₁ ^ (1 / (1 - epsilon)), ➕₁ - (-(pi ^ (var_rho * (1 - epsilon))) * pi ^ (epsilon - 1) * theta + 1) / (1 - theta)]
end)




params = [  0.99
0
0
0.75
10
1
0
0
0
1
2
0.125
0.8
0
0.01
0.01
0.01]



beta = params[1]
trend_inflation = params[2]
alpha = params[3]
theta = params[4]
epsilon = params[5]
sigma = params[6]
phi_par = params[10]
phi_pi = params[11]
phi_y = params[12]
rho_i = params[13]
var_rho = params[14]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:988 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:989 =#
Pi_bar = (1 + trend_inflation / 100) ^ (1 / 4)
i_bar = Pi_bar / beta - 1
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:990 =#
NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:991 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:992 =#
y = 3.0 ^ (alpha - 1)
v = 0
N = 0.333333333333333
A = 0
➕₃ = min(max(1.1920928955078125e-7, y), 1.0000000000002207e12)
s = 0.333333333333333 * ➕₃ ^ (1 / (alpha - 1))
lbs = [1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
ubs = [1.0000000000007645e12, 1.0000000000006613e12, 1.000000000000846e12]
# inits = max.(lbs, min.(ubs, closest_solution[1]))
# block_solver_RD = block_solver_AD([alpha, theta, epsilon, var_rho, s], 1, 𝓂.ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)


inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

# block_solver_RD = block_solver_AD([beta_p, piss, ind_d, kappa_d, phi_pie, rho_ib, phi_y, mk_d, ➕₃], 1, 𝓂.ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
# solution = block_solver_RD([beta_p, piss, ind_d, kappa_d, phi_pie, rho_ib, phi_y, mk_d, ➕₃])
# solution_error += sum(abs2, (𝓂.ss_solve_blocks[1])([beta_p, piss, ind_d, kappa_d, phi_pie, rho_ib, phi_y, mk_d, ➕₃], solution, 0, lbs, ubs))
transformer_option = 1

parameters_and_solved_vars = [alpha, theta, epsilon, var_rho, s]

previous_sol_init = inits

sol_new, info = levenberg_marquardt(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),
                                    transformer(previous_sol_init,lbs,ubs, option = transformer_option),
                                    transformer(lbs,lbs,ubs, option = transformer_option),
                                    transformer(ubs,lbs,ubs, option = transformer_option), 
                                    iterations = 100,
                                    μ¹ = .000001, 
                                    μ² = .0001, 
                                    # p = 1.9, 
                                    # λ¹ = .6, 
                                    # λ² = .7, 
                                    # λᵖ = .9, 
                                    # ρ = .1
                                    )#, λ² = .7, p = 1.4, λ¹ = .5,  μ = .001)#, λ = .5)#, p = 1.4)#, p =10.0)
solution = undo_transformer(sol_new,lbs,ubs, option = transformer_option)


info[[1,3]]




# go over parameters
parameter_find = []
itter = 1

for ρ in .05:.05:.15
    for λ¹ in .4:.1:.7
        for λ² in .4:.1:.7
            for λᵖ in .5:.2:.9
                for μ¹ in exp10.(-5:1:-3) 
                    for μ² in exp10.(-5:1:-3) 
                        for p in 1.4:.1:2.5
                            println(itter)
                            itter += 1
                            sol = levenberg_marquardt(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),
                            transformer(previous_sol_init,lbs,ubs, option = transformer_option),
                            transformer(lbs,lbs,ubs, option = transformer_option),
                            transformer(ubs,lbs,ubs, option = transformer_option),
                            iterations = 200, 
                            p = p, 
                            # σ¹ = σ¹, 
                            # σ² = σ², 
                            ρ = ρ,
                            # ϵ = .1,
                            # steps = 1,
                            λ¹ = λ¹,
                            λ² = λ²,
                            λᵖ = λᵖ,
                            μ¹ = μ¹,
                            μ² = μ²)
                            push!(parameter_find,[sol[2][[1,3]]..., μ¹, μ², p, λ¹, λ², λᵖ, ρ])
                        end
                    end
                end
            end
        end
    end
end
# end



using DataFrames, GLM, CSV

# simul = DataFrame(reduce(hcat,parameter_find)',[:iter,:tol,:ρ, :μ, :σ¹, :σ², :r, :ϵ, :steps, :γ])
# simul = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ, :σ², :r, :steps, :γ])
simulAS = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ¹, :μ², :p, :λ¹, :λ², :λᵖ, :ρ])
sort!(simulAS, [:μ¹, :μ², :p, :λ¹, :λ², :λᵖ, :ρ])
simulAS[simulAS.tol .> 1e-8,:iter] .= 200.0

simulASsub = simulAS[simulAS.tol .< 1e-8,:]


describe(simulASsub)