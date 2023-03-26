
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


function levenberg_marquardt(f::Function, 
    initial_guess::Array{T,1}, 
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1}; 
    xtol::T = eps(), 
    ftol::T = 1e-8, 
    iterations::S = 1000, 
    r::T = .5, 
    ρ::T = .1, 
    p::T = 1.6,
    λ¹::T = .4, 
    λ²::T = .7,
    λᵖ::T = .7, 
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
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:781 =#
    C = guess[1]
    I = guess[2]
    K = guess[3]
    L = guess[4]
    W = guess[5]
    Y = guess[6]
    Y_s = guess[7]
    f_1 = guess[8]
    f_2 = guess[9]
    g_1 = guess[10]
    g_2 = guess[11]
    mc = guess[12]
    q = guess[13]
    r_k = guess[14]
    w_star = guess[15]
    z = guess[16]
    ➕₁ = guess[17]
    ➕₁₄ = guess[18]
    ➕₁₇ = guess[19]
    ➕₂ = guess[20]
    ➕₅ = guess[21]
    ➕₈ = guess[22]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:782 =#
    lambda_p = parameters_and_solved_vars[1]
    lambda_w = parameters_and_solved_vars[2]
    Phi = parameters_and_solved_vars[3]
    alpha = parameters_and_solved_vars[4]
    beta = parameters_and_solved_vars[5]
    h = parameters_and_solved_vars[6]
    omega = parameters_and_solved_vars[7]
    psi = parameters_and_solved_vars[8]
    sigma_c = parameters_and_solved_vars[9]
    sigma_l = parameters_and_solved_vars[10]
    tau = parameters_and_solved_vars[11]
    varphi = parameters_and_solved_vars[12]
    xi_w = parameters_and_solved_vars[13]
    xi_p = parameters_and_solved_vars[14]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:783 =#
    T = parameters_and_solved_vars[15]
    ➕₁₁ = parameters_and_solved_vars[16]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:788 =#
    return [((L * w_star) / (➕₈ ^ sigma_c * (lambda_w + 1)) + beta * f_1 * xi_w) - f_1, -g_1 + g_2 * (lambda_p + 1), -W + (mc * ➕₅ ^ alpha * (1 - alpha)) / L ^ alpha, (-K * exp(➕₁₇) + K) / ➕₈ ^ sigma_c, (((-C - I) - (K * r_k * (exp(➕₁₇) - 1)) / psi) - T) + Y, (I + K * (1 - tau)) - K, ((Y * mc) / ➕₈ ^ sigma_c + beta * g_2 * xi_p) - g_2, q - 1 / ➕₈ ^ sigma_c, (L ^ (1 - alpha) * ➕₅ ^ alpha - Phi) - Y_s, -Y + Y_s, (beta * f_2 * xi_w - f_2) + omega * ➕₁₄ ^ (sigma_l + 1), L ^ (1 - alpha) * alpha * mc * ➕₅ ^ (alpha - 1) - r_k, beta * (q * (1 - tau) + (r_k * z - (r_k * (exp(➕₁) - 1)) / psi) / ➕₂ ^ sigma_c) - q, (Y / ➕₈ ^ sigma_c + beta * g_1 * xi_p) - g_1, -f_1 + f_2, ➕₂ - (-C * h + C), ➕₁₇ - psi * (z - 1), ➕₁ - psi * (z - 1), ➕₁₁ - w_star / W, ➕₁₄ - L / ➕₁₁ ^ ((lambda_w + 1) / lambda_w), ➕₈ - (-C * h + C), ➕₅ - K * z]
end)

push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
    guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:781 =#
    C_f = guess[1]
    I_f = guess[2]
    K_f = guess[3]
    L_f = guess[4]
    L_s_f = guess[5]
    P_j_f = guess[6]
    Pi_ws_f = guess[7]
    W_disutil_f = guess[8]
    W_f = guess[9]
    W_i_f = guess[10]
    Y_f = guess[11]
    Y_s_f = guess[12]
    mc_f = guess[13]
    q_f = guess[14]
    r_k_f = guess[15]
    z_f = guess[16]
    ➕₁₈ = guess[17]
    ➕₃ = guess[18]
    ➕₄ = guess[19]
    ➕₆ = guess[20]
    ➕₇ = guess[21]
    ➕₉ = guess[22]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:782 =#
    lambda_p = parameters_and_solved_vars[1]
    lambda_w = parameters_and_solved_vars[2]
    Phi = parameters_and_solved_vars[3]
    alpha = parameters_and_solved_vars[4]
    beta = parameters_and_solved_vars[5]
    h = parameters_and_solved_vars[6]
    omega = parameters_and_solved_vars[7]
    psi = parameters_and_solved_vars[8]
    sigma_c = parameters_and_solved_vars[9]
    sigma_l = parameters_and_solved_vars[10]
    tau = parameters_and_solved_vars[11]
    varphi = parameters_and_solved_vars[12]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:783 =#
    T_f = parameters_and_solved_vars[13]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:788 =#
    return [-Y_f + Y_s_f, (-K_f * exp(➕₁₈) + K_f) / ➕₉ ^ sigma_c, q_f - 1 / ➕₉ ^ sigma_c, (((((-C_f - I_f) - (K_f * r_k_f * (exp(➕₁₈) - 1)) / psi) - L_f * W_f) + L_s_f * W_disutil_f + Pi_ws_f) - T_f) + Y_f, (I_f + K_f * (1 - tau)) - K_f, (L_f ^ (1 - alpha) * ➕₆ ^ alpha - Phi) - Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p), -L_s_f * (-W_disutil_f + W_i_f) + Pi_ws_f, (L_f * ➕₇ ^ (-1 + (-lambda_w - 1) / lambda_w) * (-W_disutil_f + W_i_f) * (-lambda_w - 1)) / (W_f * lambda_w) + L_s_f, -(L_s_f ^ sigma_l) * omega + W_disutil_f / ➕₉ ^ sigma_c, L_f ^ (1 - alpha) * alpha * mc_f * ➕₆ ^ (alpha - 1) - r_k_f, -W_f + (mc_f * ➕₆ ^ alpha * (1 - alpha)) / L_f ^ alpha, (-(P_j_f ^ (-1 - (lambda_p + 1) / lambda_p)) * Y_f * (P_j_f - mc_f) * (lambda_p + 1)) / lambda_p + Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p), beta * (q_f * (1 - tau) + (r_k_f * z_f - (r_k_f * (exp(➕₃) - 1)) / psi) / ➕₄ ^ sigma_c) - q_f, -Y_s_f + Y_f / P_j_f ^ ((lambda_p + 1) / lambda_p), L_f * ➕₇ ^ ((-lambda_w - 1) / lambda_w) - L_s_f, -L_f + L_s_f, ➕₇ - W_i_f / W_f, ➕₄ - (-C_f * h + C_f), ➕₁₈ - psi * (z_f - 1), ➕₆ - K_f * z_f, ➕₃ - psi * (z_f - 1), ➕₉ - (-C_f * h + C_f)]
end)




params = [  0.368
0.362
0.5
0.819
0.3
0.99
0.763
0.469
0.573
1
0.169
1.684
0.099
0.14
0.159
0.961
0.855
0.889
0.927
0.823
0.949
0.924
1.353
2.4
0.025
6.771
0.737
0.908]


lambda_p = params[1]
G_bar = params[2]
lambda_w = params[3]
Phi = params[4]
alpha = params[5]
beta = params[6]
h = params[9]
omega = params[10]
psi = params[11]
sigma_c = params[23]
sigma_l = params[24]
tau = params[25]
varphi = params[26]
xi_w = params[27]
xi_p = params[28]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:982 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:983 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:984 =#
NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:985 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:986 =#
pi_obj = 1
pi = 1
➕₁₂ = max(eps(), 1)
➕₁₁ = max(eps(), 1)
➕₁₃ = max(eps(), 1)
epsilon_a = 1
➕₁₀ = max(eps(), 1)
pi_star = 1
nu_p = 1
epsilon_b = 1
epsilon_G = 1
G = G_bar
T = G
epsilon_I = 1
➕₁₅ = max(eps(), 1)
epsilon_L = 1
lbs = [-9.99999999999124e11, -9.999999999996576e11, -9.999999999998164e11, 1.1920928955078125e-7, -9.999999999994967e11, -9.999999999990083e11, -9.999999999999316e11, -9.999999999994413e11, -9.999999999999021e11, -9.999999999994248e11, -9.999999999992754e11, -9.999999999990753e11, -9.999999999999326e11, -9.999999999991036e11, -9.9999999999913e11, -9.999999999997437e11, -9.999999999993302e11, 1.1920928955078125e-7, -9.999999999994437e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
ubs = [1.0000000000007454e12, 1.0000000000009268e12, 1.0000000000006804e12, 1.0000000000000704e12, 1.0000000000000404e12, 1.0000000000001002e12, 1.0000000000006814e12, 1.0000000000008142e12, 1.0000000000003972e12, 1.0000000000000282e12, 1.0000000000006409e12, 1.000000000000585e12, 1.0000000000000121e12, 1.0000000000007094e12, 1.0000000000005623e12, 1.0000000000006578e12, 700.0, 1.0000000000003425e12, 700.0, 1.0000000000002819e12, 1.0000000000006672e12, 1.0000000000009396e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

parameters_and_solved_vars = [lambda_p, lambda_w, Phi, alpha, beta, h, omega, psi, sigma_c, sigma_l, tau, varphi, xi_w, xi_p, T, ➕₁₁]

previous_sol_init = inits

# using NLboxsolve
# sol_new = nlboxsolve(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option))


sol_new = levenberg_marquardt(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option))#, λ² = .9999, p = 1.0, λ¹ = .9999,  μ = 1e-4, iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)
sol_new[2][[1,3]]







# λ² = .9999, p = .25, λ¹ = .9,  μ = .0001
# go over parameters
parameter_find = []
itter = 1
# for transformer_option in 0:1

for ρ in .05:.05:.15
    for λ¹ in .4:.1:.7
        for λ² in .4:.1:.7
            for λᵖ in .5:.2:.9
                for μ¹ in exp10.(-5:1:-3) 
                    for μ² in exp10.(-5:1:-3) 
                        for p in 1.4:.1:2.0
                            # transformer_option = 1
                            println(itter)
                            itter += 1
                            sol = levenberg_marquardt(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option),
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


using DataFrames, GLM

# simul = DataFrame(reduce(hcat,parameter_find)',[:iter,:tol,:ρ, :μ, :σ¹, :σ², :r, :ϵ, :steps, :γ])
# simul = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ, :σ², :r, :steps, :γ])
simul = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ¹, :μ², :p, :λ¹, :λ², :λᵖ, :ρ])
sort!(simul, [:μ¹, :μ², :ρ, :p, :λ¹, :λ², :λᵖ])
simul[simul.tol .> 1e-8,:iter] .= 200.0


simulsub = simul[simul.tol .< 1e-8,:]



results = innerjoin(simulsub, simulNAWMsub, makeunique = true, on=["μ¹", "μ²", "p", "λ¹", "λ²", "λᵖ", "ρ"])
sort!(results, [:μ¹, :μ², :ρ, :p, :λ¹, :λ², :λᵖ])











#   block_solver_RD = block_solver_AD([lambda_p, lambda_w, Phi, alpha, beta, h, omega, psi, sigma_c, sigma_l, tau, varphi, xi_w, xi_p, T, ➕₁₁], 1, 𝓂.ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
#   solution = block_solver_RD([lambda_p, lambda_w, Phi, alpha, beta, h, omega, psi, sigma_c, sigma_l, tau, varphi, xi_w, xi_p, T, ➕₁₁])
solution_error += sum(abs2, (ss_solve_blocks[1])([lambda_p, lambda_w, Phi, alpha, beta, h, omega, psi, sigma_c, sigma_l, tau, varphi, xi_w, xi_p, T, ➕₁₁], solution, 0, lbs, ubs))
sol = solution
C = sol[1]
I = sol[2]
K = sol[3]
L = sol[4]
W = sol[5]
Y = sol[6]
Y_s = sol[7]
f_1 = sol[8]
f_2 = sol[9]
g_1 = sol[10]
g_2 = sol[11]
mc = sol[12]
q = sol[13]
r_k = sol[14]
w_star = sol[15]
z = sol[16]
➕₁ = sol[17]
➕₁₄ = sol[18]
➕₁₇ = sol[19]
➕₂ = sol[20]
➕₅ = sol[21]
➕₈ = sol[22]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
            sol
        else
            ℱ.value.(sol)
        end]
G_f = G_bar
T_f = G_f
lbs = [-9.999999999995978e11, -9.999999999998606e11, -9.999999999996747e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999998837e11, -9.999999999995592e11, -9.999999999996881e11, -9.999999999998801e11, -9.999999999996849e11, -9.999999999995737e11, -9.999999999995923e11, -9.999999999991996e11, -9.999999999991973e11, -9.999999999990516e11, -9.99999999999083e11, -9.999999999997275e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
ubs = [1.00000000000078e12, 1.0000000000002189e12, 1.0000000000000254e12, 1.0000000000007097e12, 1.0000000000009926e12, 1.0000000000003423e12, 1.0000000000009742e12, 1.0000000000007806e12, 1.00000000000085e12, 1.000000000000733e12, 1.0000000000008674e12, 1.0000000000006376e12, 1.000000000000421e12, 1.0000000000003641e12, 1.0000000000009923e12, 1.0000000000007113e12, 700.0, 700.0, 1.0000000000002742e12, 1.0000000000001934e12, 1.0000000000008767e12, 1.000000000000033e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

parameters_and_solved_vars = [lambda_p, lambda_w, Phi, alpha, beta, h, omega, psi, sigma_c, sigma_l, tau, varphi, T_f]

previous_sol_init = inits

sol_new = levenberg_marquardt(x->ss_solve_blocks[2](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)
sol_new[2][[1,3]]





# λ² = .9999, p = .25, λ¹ = .9,  μ = .0001
# go over parameters
parameter_find = []
itter = 1
# for transformer_option in 0:1


for ρ in .05:.05:.15
    for λ¹ in .4:.1:.7
        for λ² in .4:.1:.7
            for λᵖ in .5:.2:.9
                for μ¹ in exp10.(-5:1:-3) 
                    for μ² in exp10.(-5:1:-3) 
                        for p in 1.4:.1:2.0
                            # transformer_option = 1
                            println(itter)
                            itter += 1
                            sol = levenberg_marquardt(x->ss_solve_blocks[2](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option),
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


using DataFrames, GLM

# simul = DataFrame(reduce(hcat,parameter_find)',[:iter,:tol,:ρ, :μ, :σ¹, :σ², :r, :ϵ, :steps, :γ])
# simul = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ, :σ², :r, :steps, :γ])
simul2 = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ¹, :μ², :p, :λ¹, :λ², :λᵖ, :ρ])
sort!(simul2, [:μ¹, :μ², :ρ, :p, :λ¹, :λ², :λᵖ])
simul2[simul2.tol .> 1e-8,:iter] .= 200.0

simul2sub = simul2[simul2.tol .< 1e-8,:]

result = innerjoin(simulsub, simul2sub, makeunique = true, on=["μ¹", "μ²", "p", "λ¹", "λ²", "λᵖ", "ρ"])
sort!(result, [:μ¹, :μ², :ρ, :p, :λ¹, :λ², :λᵖ])

#   block_solver_RD = block_solver_AD([lambda_p, lambda_w, Phi, alpha, beta, h, omega, psi, sigma_c, sigma_l, tau, varphi, T_f], 2, 𝓂.ss_solve_blocks[2], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
# solution = block_solver_RD([lambda_p, lambda_w, Phi, alpha, beta, h, omega, psi, sigma_c, sigma_l, tau, varphi, T_f])
solution_error += sum(abs2, (ss_solve_blocks[2])([lambda_p, lambda_w, Phi, alpha, beta, h, omega, psi, sigma_c, sigma_l, tau, varphi, T_f], solution, 0, lbs, ubs))
sol = solution


