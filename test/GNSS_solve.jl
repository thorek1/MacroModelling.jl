using NLboxsolve

import ForwardDiff as ℱ
import LinearAlgebra as ℒ
method = :lm_ar
verbose = true
transformer_option = 4



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
    λ::T = .6,
    μ::T = .0005, # or 1e-7 for no transform (more iters anyway)
    p::T  = 1.1 # or 1.4
    ) where {T <: AbstractFloat, S <: Integer}

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert lower_bounds < upper_bounds

    current_guess = copy(initial_guess)
    previous_guess = similar(current_guess)
    guess_update = similar(current_guess)
    ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))

    largest_step = zero(T)
    largest_residual = zero(T)

	for iter in 1:iterations
        ∇ .= ℱ.jacobian(f,current_guess)

        if !all(isfinite,∇)
            return current_guess, (iter, Inf, Inf, fill(Inf,length(current_guess)))
        end

        previous_guess .= current_guess

        current_guess .+= -(∇' * ∇ + μ * sum(abs2, f(current_guess))^p * ℒ.I) \ (∇' * f(current_guess))

        μ *= λ

        minmax!(current_guess, lower_bounds, upper_bounds)

        α = 1.0

        ḡ = sum(abs2,f(previous_guess))

        guess_update .= current_guess - previous_guess

        if sum(abs2,f(previous_guess + α * guess_update)) > 0.8 * ḡ 
            while sum(abs2,f(previous_guess + α * guess_update)) > ḡ  - 0.005 * α^2 * sum(abs2,guess_update)
                α *= r
            end
        end

        current_guess .= previous_guess + α * guess_update

        largest_step = maximum(abs, previous_guess - current_guess)
        largest_residual = maximum(abs, f(current_guess))

        if largest_step <= xtol || largest_residual <= ftol
            return current_guess, (iter, largest_step, largest_residual, f(current_guess))
        end
    end

    return current_guess, (iterations, largest_step, largest_residual, f(current_guess))
end







function levenberg_marquardt_ar(f::Function, 
    initial_guess::Array{T,1}, 
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1}; 
    xtol::T = eps(), 
    ftol::T = 1e-8, 
    iterations::S = 1000, 
    r::T = .5, 
    μ::T = 1e-6, 
    ρ::T  = 0.8, 
    p::T  = 2.0, 
    σ¹::T = 0.005, 
    σ²::T = 0.005, 
    ϵ::T = .1,
    γ::T = eps(),
    steps::S = 1,
    trace::Bool = false) where {T <: AbstractFloat, S <: Integer}

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert lower_bounds < upper_bounds

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    current_guess = copy(initial_guess)
    ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))
    Â = similar(∇)
    previous_guess = similar(current_guess)
    intermediate_guess = similar(current_guess)
    # d̄ = similar(current_guess)

    largest_step = zero(T)
    largest_residual = zero(T)

    if trace 
        guess_history = []
        push!(guess_history,copy(current_guess)) 
    end

	for iter in 1:iterations
        ∇ .= ℱ.jacobian(f,current_guess)

        if !all(isfinite,∇)
            if trace 
                return current_guess, (iter, Inf, Inf, fill(Inf,length(current_guess)), guess_history)
            else
                return current_guess, (iter, Inf, Inf, fill(Inf,length(current_guess)))
            end
        end

        Â .= inv(-(∇' * ∇ + μ * sum(abs2, f(current_guess))^p * ℒ.I))

        previous_guess .= current_guess

        for step in 1:steps
            current_guess .+= Â * ∇' * f(current_guess)

            minmax!(current_guess, lower_bounds, upper_bounds)

            if step == 1# || sum(abs2,f(current_guess)) < N̄ 
                intermediate_guess .= current_guess 
            end

            # if step == steps println((current_guess - previous_guess)[1]) end

            if f(previous_guess)' * ∇ * (current_guess - previous_guess) > -γ
                # println((current_guess - previous_guess)[1])
                current_guess .= intermediate_guess
                break
            end
        end

        α = 1.0

        if sum(abs2,f(current_guess)) > ρ^2 * sum(abs2,f(previous_guess))
            while sum(abs2,f(previous_guess + α * (current_guess - previous_guess))) > (1 + ϵ - σ² * α^2) * sum(abs2,f(previous_guess)) - σ¹ * α^2 * sum(abs2,current_guess - previous_guess)
                α *= r
                ϵ *= r
            end
        end

        current_guess .= previous_guess + α * (current_guess - previous_guess)

        largest_step = maximum(abs, previous_guess - current_guess)
        largest_residual = maximum(abs, f(current_guess))

        if trace push!(guess_history,copy(current_guess)) end
        if largest_step <= xtol || largest_residual <= ftol
            if trace 
                return current_guess, (iter, largest_step, largest_residual, f(current_guess), guess_history)
            else
                return current_guess, (iter, largest_step, largest_residual, f(current_guess))
            end
        end

    end

    if trace 
        return current_guess, (iterations, largest_step, largest_residual, f(current_guess), guess_history)
    else
        return current_guess, (iterations, largest_step, largest_residual, f(current_guess))
    end
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
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:769 =#
    pie = guess[1]
    r_d = guess[2]
    r_ib = guess[3]
    ➕₄ = guess[4]
    ➕₅ = guess[5]
    ➕₆ = guess[6]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:770 =#
    beta_p = parameters_and_solved_vars[1]
    piss = parameters_and_solved_vars[2]
    ind_d = parameters_and_solved_vars[3]
    kappa_d = parameters_and_solved_vars[4]
    phi_pie = parameters_and_solved_vars[5]
    rho_ib = parameters_and_solved_vars[6]
    phi_y = parameters_and_solved_vars[7]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:771 =#
    mk_d = parameters_and_solved_vars[8]
    ➕₃ = parameters_and_solved_vars[9]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:776 =#
    return [(-1 + exp(mk_d) / (exp(mk_d) - 1)) - (r_ib * exp(mk_d)) / (r_d * (exp(mk_d) - 1)), (-beta_p * (r_d + 1)) / pie + 1, (r_ib - ➕₃ ^ (1 - rho_ib) * ➕₄ ^ rho_ib * ➕₆ ^ (1 - rho_ib)) + 1, ➕₅ - pie / piss, ➕₄ - (r_ib + 1), ➕₆ - ➕₅ ^ phi_pie]
end)

push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
    guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:769 =#
    B = guess[1]
    BE = guess[2]
    BH = guess[3]
    J_R = guess[4]
    K_b = guess[5]
    R_b = guess[6]
    Y = guess[7]
    b_e = guess[8]
    b_ee = guess[9]
    b_h = guess[10]
    b_i = guess[11]
    c_e = guess[12]
    c_i = guess[13]
    c_p = guess[14]
    d_b = guess[15]
    d_p = guess[16]
    h_i = guess[17]
    h_p = guess[18]
    j_B = guess[19]
    k_e = guess[20]
    l_i = guess[21]
    l_id = guess[22]
    l_p = guess[23]
    l_pd = guess[24]
    lam_e = guess[25]
    lam_i = guess[26]
    lam_p = guess[27]
    q_h = guess[28]
    r_be = guess[29]
    r_bh = guess[30]
    r_k = guess[31]
    s_e = guess[32]
    s_i = guess[33]
    u = guess[34]
    w_i = guess[35]
    w_p = guess[36]
    y_e = guess[37]
    ➕₁ = guess[38]
    ➕₂ = guess[39]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:770 =#
    beta_p = parameters_and_solved_vars[1]
    beta_i = parameters_and_solved_vars[2]
    j = parameters_and_solved_vars[3]
    phi = parameters_and_solved_vars[4]
    alpha = parameters_and_solved_vars[5]
    gamma_p = parameters_and_solved_vars[6]
    gamma_i = parameters_and_solved_vars[7]
    ni = parameters_and_solved_vars[8]
    gamma_b = parameters_and_solved_vars[9]
    gamma_e = parameters_and_solved_vars[10]
    deltak = parameters_and_solved_vars[11]
    piss = parameters_and_solved_vars[12]
    h = parameters_and_solved_vars[13]
    vi = parameters_and_solved_vars[14]
    ind_be = parameters_and_solved_vars[15]
    ind_bh = parameters_and_solved_vars[16]
    kappa_p = parameters_and_solved_vars[17]
    kappa_w = parameters_and_solved_vars[18]
    kappa_d = parameters_and_solved_vars[19]
    kappa_be = parameters_and_solved_vars[20]
    kappa_bh = parameters_and_solved_vars[21]
    kappa_kb = parameters_and_solved_vars[22]
    ind_p = parameters_and_solved_vars[23]
    ind_w = parameters_and_solved_vars[24]
    a_i = parameters_and_solved_vars[25]
    beta_e = parameters_and_solved_vars[26]
    eksi_1 = parameters_and_solved_vars[27]
    eksi_2 = parameters_and_solved_vars[28]
    delta_kb = parameters_and_solved_vars[29]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:771 =#
    eps_l = parameters_and_solved_vars[30]
    m_e = parameters_and_solved_vars[31]
    m_i = parameters_and_solved_vars[32]
    mk_be = parameters_and_solved_vars[33]
    mk_bh = parameters_and_solved_vars[34]
    pie = parameters_and_solved_vars[35]
    pie_wi = parameters_and_solved_vars[36]
    pie_wp = parameters_and_solved_vars[37]
    r_d = parameters_and_solved_vars[38]
    r_ib = parameters_and_solved_vars[39]
    x = parameters_and_solved_vars[40]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:776 =#
    return [w_p - (ni * y_e * (1 - alpha)) / (l_pd * x), (((-b_ee + (b_ee * (r_be + 1)) / pie + c_e) - k_e * (1 - deltak)) + k_e * (eksi_1 * (u - 1) + (eksi_2 * (u - 1) ^ 2) / 2) + k_e + l_id * w_i + l_pd * w_p) - y_e / x, Y - gamma_e * y_e, BH - b_h * gamma_b, (-K_b + b_e + b_h) - d_b, b_i * (r_bh + 1) - h_i * pie * q_h * exp(m_i), ((beta_p * kappa_w * pie_wp ^ 2 * (-(pie ^ ind_w) * piss ^ (1 - ind_w) + pie_wp)) / pie - kappa_w * pie_wp * (-(pie ^ ind_w) * piss ^ (1 - ind_w) + pie_wp)) + l_p * (1 - exp(eps_l)) + (l_p ^ (phi + 1) * exp(eps_l)) / (lam_p * w_p), -lam_p + (1 - a_i) / (-a_i * c_p + c_p), (beta_e * lam_e * (((-deltak - eksi_1 * (u - 1)) - (eksi_2 * (u - 1) ^ 2) / 2) + r_k * u + 1) - lam_e) + pie * s_e * (1 - deltak) * exp(m_e), b_h * gamma_b - b_i * gamma_i, gamma_e * l_pd - gamma_p * l_p, ((-beta_e * lam_e * (r_be + 1)) / pie + lam_e) - s_e * (r_be + 1), ((-beta_i * lam_i * (r_bh + 1)) / pie + lam_i) - s_i * (r_bh + 1), (-eksi_1 - eksi_2 * (u - 1)) + r_k, (-b_i + (b_i * (r_bh + 1)) / pie + c_i) - l_i * w_i, ((R_b * exp(mk_bh)) / (r_bh * (exp(mk_bh) - 1)) + 1) - exp(mk_bh) / (exp(mk_bh) - 1), (B - BE) - BH, J_R - Y * (((-kappa_p * (1 - piss ^ (1 - ind_p)) ^ 2) / 2 + 1) - 1 / x), (R_b - r_ib) + (K_b ^ 2 * kappa_kb * (-vi + K_b / B)) / B ^ 2, (beta_i * lam_i * q_h - lam_i * q_h) + pie * q_h * s_i * exp(m_i) + j / h_i, b_ee * (r_be + 1) - k_e * pie * (1 - deltak) * exp(m_e), (K_b * pie - K_b * (1 - delta_kb)) - j_B, ((R_b * exp(mk_be)) / (r_be * (exp(mk_be) - 1)) + 1) - exp(mk_be) / (exp(mk_be) - 1), w_i - (y_e * (1 - alpha) * (1 - ni)) / (l_id * x), (((K_b * kappa_kb * (-vi + K_b / B) ^ 2) / 2 - b_e * r_be) - b_h * r_bh) + d_b * r_d + j_B, (-alpha * k_e ^ (alpha - 1) * u ^ (alpha - 1) * ➕₂ ^ (1 - alpha)) / x + r_k, -lam_i + (1 - a_i) / (-a_i * c_i + c_i), (beta_p * lam_p * q_h - lam_p * q_h) + j / h_p, BE - b_e * gamma_b, y_e - ➕₁ ^ alpha * ➕₂ ^ (1 - alpha), b_e * gamma_b - b_ee * gamma_e, (-gamma_i * h_i - gamma_p * h_p) + h, -lam_e + (1 - a_i) / (-a_i * c_e + c_e), ((beta_i * kappa_w * pie_wi ^ 2 * (-(pie ^ ind_w) * piss ^ (1 - ind_w) + pie_wi)) / pie - kappa_w * pie_wi * (-(pie ^ ind_w) * piss ^ (1 - ind_w) + pie_wi)) + l_i * (1 - exp(eps_l)) + (l_i ^ (phi + 1) * exp(eps_l)) / (lam_i * w_i), ((-J_R / gamma_p + c_p + d_p) - (d_p * (r_d + 1)) / pie) - l_p * w_p, d_b * gamma_b - d_p * gamma_p, gamma_e * l_id - gamma_i * l_i, ➕₂ - l_id ^ (1 - ni) * l_pd ^ ni, ➕₁ - k_e * u]
end)




params = [   0.9943
0.975
0.2
1.0
0.7
0.35
0.25
-1.46025
2.932806
2.932806
0
6
5
1
1
0.8
1
1
0.025
1
1
0.09
0.0
0.0
0.0
0.3935275324257052
0.9390001567894549
0.9211790941478728
0.8938651443507468
0.9286486478061776
0.8380479641501677
0.8194621730335763
0.8342810056222126
0.5474914620444137
0.3047340963457367
0.639922254764848
0.8129795852441276
28.65019653869527
99.89828358530188
10.182155670839322
3.5029734165847466
9.36382331915174
10.086654447226444
11.068335540791962
1.9816026561910398
0.7685551455946952
0.3459149657035201
0.16051347848216171
0.27569624058316433
0.8559521971842566
0.0
0.0
0.0144
0.0062
0.0658
0.0034
0.0023
0.0488
0.0051
0.1454
0.0128
0.0018
1.0099
0.3721
0.05]

beta_p = params[1]
beta_i = params[2]
j = params[3]
phi = params[4]
m_i_ss = params[5]
m_e_ss = params[6]
alpha = params[7]
eps_d = params[8]
eps_bh = params[9]
eps_be = params[10]
eps_y_ss = params[12]
eps_l_ss = params[13]
gamma_p = params[14]
gamma_i = params[15]
ni = params[16]
gamma_b = params[17]
gamma_e = params[18]
deltak = params[19]
piss = params[20]
h = params[21]
vi = params[22]
ind_d = params[23]
ind_be = params[24]
ind_bh = params[25]
kappa_p = params[38]
kappa_w = params[39]
kappa_d = params[41]
kappa_be = params[42]
kappa_bh = params[43]
kappa_kb = params[44]
phi_pie = params[45]
rho_ib = params[46]
phi_y = params[47]
ind_p = params[48]
ind_w = params[49]
a_i = params[50]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:970 =#
piss = min(max(piss, 1.1920928955078125e-7), 1.0000000000001925e12)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:971 =#
beta_b = beta_p
beta_e = beta_i
mk_d_ss = eps_d / (eps_d - 1)
mk_bh_ss = eps_bh / (eps_bh - 1)
mk_be_ss = eps_be / (eps_be - 1)
r_ib_ss = ((eps_d - 1) * (piss / beta_p - 1)) / eps_d
r_be_ss = (eps_be * r_ib_ss) / (eps_be - 1)
r_bh_ss = (eps_bh * r_ib_ss) / (eps_bh - 1)
r_k_ss = (-((1 - deltak)) - ((piss * (1 - deltak) * m_e_ss) / beta_e) * (1 / (1 + r_be_ss) - beta_e / piss)) + 1 / beta_e
eksi_1 = r_k_ss
eksi_2 = r_k_ss * 0.1
eps_b = (eps_bh + eps_be) / 2
delta_kb = ((r_ib_ss / vi) * ((eps_d - eps_b) + eps_d * vi * (eps_b - 1))) / ((eps_d - 1) * (eps_b - 1))
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:972 =#
NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:973 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:974 =#
eps_y = log(eps_y_ss)
ee_z = 0
➕₃ = min(max(1.1920928955078125e-7, r_ib_ss + 1), 1.0000000000000715e12)
mk_d = log(mk_d_ss)
lbs = [1.1920928955078125e-7, -9.999999999999706e11, -9.999999999993268e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
ubs = [1.000000000000253e12, 1.0000000000007391e12, 1.0000000000007084e12, 1.0000000000009435e12, 1.0000000000009888e12, 1.0000000000009756e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

# block_solver_RD = block_solver_AD([beta_p, piss, ind_d, kappa_d, phi_pie, rho_ib, phi_y, mk_d, ➕₃], 1, 𝓂.ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
# solution = block_solver_RD([beta_p, piss, ind_d, kappa_d, phi_pie, rho_ib, phi_y, mk_d, ➕₃])
# solution_error += sum(abs2, (𝓂.ss_solve_blocks[1])([beta_p, piss, ind_d, kappa_d, phi_pie, rho_ib, phi_y, mk_d, ➕₃], solution, 0, lbs, ubs))
transformer_option = 1

parameters_and_solved_vars = [beta_p, piss, ind_d, kappa_d, phi_pie, rho_ib, phi_y, mk_d, ➕₃]

previous_sol_init = inits

sol_new, info = levenberg_marquardt(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)#, μ = .0005)#, λ = .5)#, p = 1.4)#, p =10.0)
solution = undo_transformer(sol_new,lbs,ubs, option = transformer_option)



# sol_new = nlboxsolve(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
# solution = undo_transformer(sol_new.zero,lbs,ubs, option = transformer_option)
                #   block_solver_RD = block_solver_AD([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR], 1, ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR])
solution_error += sum(abs2, (ss_solve_blocks[1])(parameters_and_solved_vars, solution, 0, lbs, ubs))


sol = solution
pie = sol[1]
r_d = sol[2]
r_ib = sol[3]
➕₄ = sol[4]
➕₅ = sol[5]
➕₆ = sol[6]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
            sol
        else
            ℱ.value.(sol)
        end]
x = -(exp(eps_y)) / (((((beta_p * kappa_p * pie ^ 2 - beta_p * kappa_p * pie ^ (ind_p + 1) * piss ^ (1 - ind_p)) - kappa_p * pie ^ 2) + kappa_p * pie ^ (ind_p + 1) * piss ^ (1 - ind_p)) - exp(eps_y)) + 1)
A_e = 0
m_e = log(m_e_ss)
mk_bh = log(mk_bh_ss)
m_i = log(m_i_ss)
ee_j = 0
pie_wi = pie
eps_l = log(eps_l_ss)
eps_K_b = 0
mk_be = log(mk_be_ss)
ee_qk = 0
q_k = 1
pie_wp = pie
lbs = [-9.999999999991332e11, -9.999999999990603e11, -9.999999999996316e11, -9.9999999999958e11, -9.999999999997006e11, -9.999999999994758e11, -9.999999999993079e11, -9.999999999991606e11, -9.99999999999722e11, -9.999999999990728e11, -9.999999999990931e11, -9.999999999990767e11, -9.999999999990826e11, -9.9999999999985e11, -9.999999999993181e11, -9.99999999999706e11, -9.999999999995851e11, -9.999999999999125e11, -9.999999999994717e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999991274e11, -9.999999999995262e11, -9.999999999993618e11, -9.999999999997921e11, -9.999999999996526e11, -9.999999999991731e11, -9.99999999999336e11, -9.999999999994158e11, -9.9999999999983e11, 1.1920928955078125e-7, -9.999999999999349e11, -9.999999999990642e11, -9.9999999999986e11, 1.1920928955078125e-7, 1.1920928955078125e-7]
ubs = [1.0000000000009529e12, 1.0000000000004498e12, 1.0000000000002745e12, 1.0000000000007039e12, 1.000000000000603e12, 1.0000000000004905e12, 1.000000000000751e12, 1.0000000000004178e12, 1.0000000000002572e12, 1.0000000000005719e12, 1.0000000000004489e12, 1.0000000000009875e12, 1.0000000000006892e12, 1.0000000000003691e12, 1.0000000000003909e12, 1.0000000000005836e12, 1.0000000000000676e12, 1.0000000000001016e12, 1.0000000000000944e12, 1.0000000000002822e12, 1.0000000000004904e12, 1.0000000000004064e12, 1.0000000000000146e12, 1.000000000000269e12, 1.00000000000006e12, 1.0000000000001852e12, 1.0000000000001589e12, 1.0000000000006929e12, 1.0000000000009417e12, 1.0000000000001357e12, 1.0000000000007079e12, 1.0000000000002041e12, 1.0000000000005435e12, 1.0000000000001443e12, 1.0000000000005546e12, 1.0000000000001842e12, 1.0000000000005153e12, 1.0000000000004794e12, 1.000000000000993e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))
# block_solver_RD = block_solver_AD([beta_p, beta_i, j, phi, alpha, gamma_p, gamma_i, ni, gamma_b, gamma_e, deltak, piss, h, vi, ind_be, ind_bh, kappa_p, kappa_w, kappa_d, kappa_be, kappa_bh, kappa_kb, ind_p, ind_w, a_i, beta_e, eksi_1, eksi_2, delta_kb, eps_l, m_e, m_i, mk_be, mk_bh, pie, pie_wi, pie_wp, r_d, r_ib, x], 2, 𝓂.ss_solve_blocks[2], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
# solution = block_solver_RD([beta_p, beta_i, j, phi, alpha, gamma_p, gamma_i, ni, gamma_b, gamma_e, deltak, piss, h, vi, ind_be, ind_bh, kappa_p, kappa_w, kappa_d, kappa_be, kappa_bh, kappa_kb, ind_p, ind_w, a_i, beta_e, eksi_1, eksi_2, delta_kb, eps_l, m_e, m_i, mk_be, mk_bh, pie, pie_wi, pie_wp, r_d, r_ib, x])
# solution_error += sum(abs2, (𝓂.ss_solve_blocks[2])([beta_p, beta_i, j, phi, alpha, gamma_p, gamma_i, ni, gamma_b, gamma_e, deltak, piss, h, vi, ind_be, ind_bh, kappa_p, kappa_w, kappa_d, kappa_be, kappa_bh, kappa_kb, ind_p, ind_w, a_i, beta_e, eksi_1, eksi_2, delta_kb, eps_l, m_e, m_i, mk_be, mk_bh, pie, pie_wi, pie_wp, r_d, r_ib, x], solution, 0, lbs, ubs))

previous_sol_init = inits

parameters_and_solved_vars = [beta_p, beta_i, j, phi, alpha, gamma_p, gamma_i, ni, gamma_b, gamma_e, deltak, piss, h, vi, ind_be, ind_bh, kappa_p, kappa_w, kappa_d, kappa_be, kappa_bh, kappa_kb, ind_p, ind_w, a_i, beta_e, eksi_1, eksi_2, delta_kb, eps_l, m_e, m_i, mk_be, mk_bh, pie, pie_wi, pie_wp, r_d, r_ib, x]


sol_new, info = levenberg_marquardt(x->ss_solve_blocks[2](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)#, μ = 3e-3)#,μ = 1e-5, p = 3.0)

info[[1,3]]
# sol_new = nlboxsolve(x->ss_solve_blocks[2](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
sol_new[2]
solution = undo_transformer(sol_new,lbs,ubs, option = transformer_option)
                #   block_solver_RD = block_solver_AD([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR], 1, ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR])
solution_error += sum(abs2, (ss_solve_blocks[2])(parameters_and_solved_vars, solution, 0, lbs, ubs))


# transformation of NSSS problem
function transformer(x,lb,ub; option::Int = 2)
        for i in 1:option
            x = asinh.(x ./ 1)
        end
        return x
end


function undo_transformer(x,lb,ub; option::Int = 2)
        for i in 1:option
            x = sinh.(x)
        end
        return x .* 1
end



function minmax!(x::Vector{Float64},lb::Vector{Float64},ub::Vector{Float64})
    for i in 1:length(x)
        if x[i] <= lb[i]
            x[i] = lb[i]
        elseif x[i] >= ub[i]
            x[i] = ub[i]
        end
    end
end

using Statistics
function levenberg_marquardt_ar(f::Function, x::Array{T,1}, lb::Array{T,1}, ub::Array{T,1}; xtol::T = eps(), ftol::T = 1e-8,iterations::S = 100000, r::T = .5, μ::T = 1e-4, ρ::T  = 0.8, σ1::T = 0.005, σ2::T = 0.005) where {T <: AbstractFloat, S <: Integer}

    @assert size(lb) == size(ub) == size(x)
    @assert lb < ub

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xk1 = copy(x)
    xk2 = copy(x)
    xk3 = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = Array{T,2}(undef,n,n)
    A = similar(jk)
    dk = similar(xk)
    d1k = similar(dk)
    d2k = similar(dk)
    d3k = similar(dk)
    d4k = similar(dk)

    lenx = zero(T)
    lenf = zero(T)
    xkhist = []
    αhist = []
    # Initialize solver-parameters
    γ  = eps()
    
    push!(xkhist,copy(xk))

	for iter in 1:iterations
        jk .= ℱ.jacobian(f,xk)

        xk_norm = ℒ.norm(f(xk))
        # xk_norm = median(f(xk).^2) * n
        Â = inv(-(jk'jk + μ * xk_norm * ℒ.I))
        # d1k .= -(jk'jk + μ * ℒ.norm(f(xk))^2 * ℒ.I) \ (jk'f(xk))
        d1k .= Â * (jk'f(xk))
        # d1k .= -(jk'jk + ℒ.diagm(μ * abs.(f(xk)))) \ (jk'f(xk))
        xk1 .= xk + d1k
        minmax!(xk1, lb, ub)
        # println(xk1[1])
        # println(d1k[1])

        # d2k .= -(jk'jk + μ * ℒ.norm(f(xk1))^2 * ℒ.I) \ (jk'f(xk1))
        d2k .= Â * (jk'f(xk1))
        xk2 .= xk1 + d2k
        minmax!(xk2, lb, ub)
        # println(xk2[1])
        # println(d2k[1])

        # d3k .= -(jk'jk + μ * ℒ.norm(f(xk2))^2 * ℒ.I) \ (jk'f(xk1))
        d3k .= Â * (jk'f(xk2))
        xk3 .= xk2 + d3k
        minmax!(xk3, lb, ub)
        # println(xk3[1])
        # println(d3k[1])

        # d4k .= -(jk'jk + μ * ℒ.norm(f(xk3))^2 * ℒ.I) \ (jk'f(xk3))
        d4k .= Â * (jk'f(xk3))
        z .= xk3 + d4k
        minmax!(z, lb, ub)
        # println(z[1])
        # println(d4k[1])

        dk .= d1k + d2k + d3k + d4k

        s .= z - xk
        println(s[1])

        if !all(isfinite,s)
            return xk, (iter, xkhist, αhist, Inf, Inf, fill(Inf,length(xk)))
        end

        # if ℒ.norm(f(z)) <= ρ * xk_norm
        # # # if maximum(abs2,f(z)) <= ρ * xk_norm
        #     α = 1.0
        # else
            if f(xk)'jk * dk > -γ
                s .= xk1 - xk
            end

        #     α = 1.0

        #     epsilon = 1/10

        #     while ℒ.norm(f(xk + α * s))^2 > (1 + epsilon) * xk_norm^2 - σ1 * α^2 * ℒ.norm(s)^2 - σ2 * α^2 * xk_norm^2
        #         α *= r

        #         epsilon *= r
        #     end
        # end

        xn .= xk + s
        # xn .= xk + α * s

        lenx = maximum(abs, xn - xk)
        lenf = maximum(abs, f(xn))

        xk .= xn

        # push!(αhist,copy(α))
        push!(xkhist,copy(xk))
        if lenx <= xtol || lenf <= ftol
            return xk, (iter, xkhist, αhist, lenx, lenf, f(xn))
        end

    end

    return xk, (iterations, xkhist, αhist, lenx, lenf, f(xk))
end





function levenberg_marquardt_ar(f::Function, x::Array{T,1}, lb::Array{T,1}, ub::Array{T,1}; xtol::T = eps(), ftol::T = 1e-8,iterations::S = 100000, r::T = .5, μ::T = 1e-4, ρ::T  = 0.8, σ1::T = 0.005, σ2::T = 0.005, steps::S = 4) where {T <: AbstractFloat, S <: Integer}

    @assert size(lb) == size(ub) == size(x)
    @assert lb < ub

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    # xk1 = copy(x)
    # xk2 = copy(x)
    # xk3 = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = Array{T,2}(undef,n,n)
    A = similar(jk)
    dk = similar(xk)
    d̄ = similar(xk)
    # d1k = similar(dk)
    # d2k = similar(dk)
    # d3k = similar(dk)
    # d4k = similar(dk)

    lenx = zero(T)
    lenf = zero(T)
    xkhist = []
    αhist = []
    # Initialize solver-parameters
    γ  = eps()
    
    push!(xkhist,copy(xk))

	for iter in 1:iterations
        jk .= ℱ.jacobian(f,xk)

        xk_norm = ℒ.norm(f(xk))

        xk̄ = copy(xk)

        for i in 1:steps
            d̄ .= -(jk'jk + μ * ℒ.norm(f(xk))^2 * ℒ.I) \ (jk'f(xk))
            xk .= xk + d̄
            minmax!(xk, lb, ub)
        end

        s .= xk̄ - xk

        if !all(isfinite,s)
            return xk, (iter, xkhist,  Inf, Inf, fill(Inf,length(xk)))
        end

        if ℒ.norm(f(xk)) <= ρ * xk_norm
            α = 1.0
        else
            if f(xk̄)'jk * s > -γ
                s /= 2
            end

            α = 1.0

            epsilon = 1/10

            while ℒ.norm(f(xk̄ + α * s))^2 > (1 + epsilon) * xk_norm^2 - σ1 * α^2 * ℒ.norm(s)^2 - σ2 * α^2 * xk_norm^2
                α *= r

                epsilon *= r
            end
        end

        xn .= xk + α * s

        lenx = maximum(abs, xn - xk)
        lenf = maximum(abs, f(xn))

        xk .= xn

        push!(αhist,copy(α))
        push!(xkhist,copy(xk))
        if lenx <= xtol || lenf <= ftol
            return xk, (iter, xkhist, αhist, lenx, lenf, f(xn))
        end

    end

    return xk, (iterations, xkhist, lenx, lenf, f(xk))
end




function levenberg_marquardt_ar(f::Function, 
    initial_guess::Array{T,1}, 
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1}; 
    xtol::T = eps(), 
    ftol::T = 1e-8, 
    iterations::S = 1000, 
    r::T = .5, 
    μ::T = 1e-6, 
    ρ::T  = 0.8, 
    p::T  = 1.0, 
    σ¹::T = 0.005, 
    σ²::T = 0.005, 
    ϵ::T = .1,
    γ::T = eps(),
    steps::S = 1,
    trace::Bool = false) where {T <: AbstractFloat, S <: Integer}

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert lower_bounds < upper_bounds

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    current_guess = copy(initial_guess)
    ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))
    Â = similar(∇)
    previous_guess = similar(current_guess)
    intermediate_guess = similar(current_guess)
    # d̄ = similar(current_guess)

    largest_step = zero(T)
    largest_residual = zero(T)

    if trace 
        guess_history = []
        push!(guess_history,copy(current_guess)) 
    end

	for iter in 1:iterations
        ∇ .= ℱ.jacobian(f,current_guess)

        if !all(isfinite,∇)
            if trace 
                return current_guess, (iter, Inf, Inf, fill(Inf,length(current_guess)), guess_history)
            else
                return current_guess, (iter, Inf, Inf, fill(Inf,length(current_guess)))
            end
        end

        Â .= inv(-(∇' * ∇ + μ * sum(abs2, f(current_guess))^p * ℒ.I))

        previous_guess .= current_guess

        for step in 1:steps
            current_guess .+= Â * ∇' * f(current_guess)

            minmax!(current_guess, lower_bounds, upper_bounds)

            if step == 1# || sum(abs2,f(current_guess)) < N̄ 
                intermediate_guess .= current_guess 
            end

            # if step == steps println((current_guess - previous_guess)[1]) end

            if f(previous_guess)' * ∇ * (current_guess - previous_guess) > -γ
                # println((current_guess - previous_guess)[1])
                current_guess .= intermediate_guess
                break
            end
        end

        α = 1.0

        if sum(abs2,f(current_guess)) > ρ^2 * sum(abs2,f(previous_guess))
            while sum(abs2,f(previous_guess + α * (current_guess - previous_guess))) > (1 + ϵ - σ² * α^2) * sum(abs2,f(previous_guess)) - σ¹ * α^2 * sum(abs2,current_guess - previous_guess)
                α *= r
                ϵ *= r
            end
        end

        current_guess .= previous_guess + α * (current_guess - previous_guess)

        largest_step = maximum(abs, previous_guess - current_guess)
        largest_residual = maximum(abs, f(current_guess))

        if trace push!(guess_history,copy(current_guess)) end
        if largest_step <= xtol || largest_residual <= ftol
            if trace 
                return current_guess, (iter, largest_step, largest_residual, f(current_guess), guess_history)
            else
                return current_guess, (iter, largest_step, largest_residual, f(current_guess))
            end
        end

    end

    if trace 
        return current_guess, (iterations, largest_step, largest_residual, f(current_guess), guess_history)
    else
        return current_guess, (iterations, largest_step, largest_residual, f(current_guess))
    end
end






# lbs = [-9.999999999992434e11, -9.999999999998973e11, -9.999999999992908e11, -9.99999999999766e11, -9.999999999997219e11, -9.999999999994574e11, -9.999999999993348e11, -9.999999999999745e11, -9.999999999999745e11, -9.999999999997137e11, -9.999999999998438e11, -9.99999999999494e11, -9.999999999992096e11, -9.9999999999944e11, -9.999999999992551e11, 20.0, -9.999999999990197e11, -9.999999999994269e11, -9.99999999999016e11, -9.999999999997462e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994875e11, -9.99999999999188e11, -9.999999999999135e11, -9.999999999994308e11, -9.999999999990367e11, 20.0, 20.0, 20.0, -9.999999999994858e11, -9.999999999993795e11, -9.999999999994908e11, -9.999999999997921e11, -9.999999999992015e11, -9.999999999999756e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999996508e11, -9.999999999993324e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999995457e11, 1.1920928955078125e-7, -9.999999999997122e11, -9.999999999990685e11, -9.99999999999497e11, -9.999999999999436e11, -9.999999999993546e11, -9.99999999999972e11, -9.999999999996284e11, -9.999999999994846e11, -9.99999999999161e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999996093e11, -9.99999999999547e11, -9.999999999992059e11, -9.99999999999126e11, -9.999999999994252e11, -9.999999999991068e11, -9.999999999995377e11, -9.999999999999745e11, 20.0, -9.999999999999745e11, 20.0, -9.999999999993511e11, -9.99999999999341e11, -9.999999999994292e11, -9.999999999996029e11, -9.999999999991389e11, 20.0, -9.99999999999839e11, -9.999999999997675e11, -9.999999999998387e11, -9.999999999994056e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994823e11, -9.999999999994779e11, -9.99999999999081e11, -9.999999999998727e11, -9.999999999995522e11, 20.0, 20.0, 20.0, -9.999999999998718e11, -9.999999999993461e11, -9.999999999992473e11, -9.999999999996089e11, -9.999999999993452e11, -9.999999999992495e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999992012e11, -9.999999999994204e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999995746e11, -9.999999999996866e11, -9.999999999995294e11, -9.999999999994865e11, -9.99999999999996e11, -9.999999999990101e11, -9.999999999994713e11, -9.999999999995946e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994125e11, -9.999999999992838e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
# ubs = [1.0000000000006466e12, 1.0000000000001799e12, 1.0000000000008628e12, 1.0000000000003022e12, 1.000000000000415e12, 1.0000000000006346e12, 1.0000000000002433e12, 1.0000000000004271e12, 1.0000000000003107e12, 1.0000000000006022e12, 1.0000000000008429e12, 1.000000000000343e12, 1.0000000000009706e12, 1.0000000000002526e12, 1.0000000000004196e12, 1.0000000000004094e12, 1.0000000000003304e12, 1.0000000000005702e12, 1.0000000000004358e12, 1.0000000000005243e12, 1.0000000000008948e12, 1.0000000000002933e12, 1.0000000000000917e12, 1.0000000000000221e12, 1.0000000000000438e12, 1.0000000000002003e12, 1.0000000000007571e12, 1.0000000000001656e12, 1.0000000000005831e12, 1.0000000000009275e12, 1.0000000000009824e12, 1.000000000000705e12, 1.0000000000003829e12, 1.0000000000001826e12, 1.0000000000002479e12, 1.0000000000007766e12, 1.0000000000003995e12, 1.0000000000009668e12, 1.0000000000002118e12, 1.0000000000001075e12, 1.0000000000007489e12, 1.0000000000006339e12, 1.0000000000005109e12, 1.0000000000004648e12, 1.0000000000004583e12, 1.0000000000009219e12, 1.0000000000000668e12, 1.000000000000175e12, 1.0000000000007239e12, 1.0000000000003716e12, 1.0000000000008591e12, 1.000000000000525e12, 1.0000000000000428e12, 1.0000000000004991e12, 1.0000000000000692e12, 1.0000000000006803e12, 1.0000000000000836e12, 1.000000000000833e12, 1.0000000000008341e12, 1.0000000000007124e12, 1.0000000000001855e12, 1.0000000000008895e12, 1.0000000000006974e12, 1.0000000000002766e12, 1.0000000000004583e12, 1.0000000000004031e12, 1.0000000000004636e12, 1.0000000000009358e12, 1.0000000000005801e12, 1.0000000000007516e12, 1.0000000000000852e12, 1.0000000000000955e12, 1.0000000000001885e12, 1.0000000000009397e12, 1.0000000000002449e12, 1.0000000000000017e12, 1.0000000000006707e12, 1.0000000000009272e12, 1.0000000000008794e12, 1.000000000000489e12, 1.0000000000006207e12, 1.0000000000005729e12, 1.0000000000000215e12, 1.0000000000001346e12, 1.0000000000001606e12, 1.0000000000006796e12, 1.0000000000000281e12, 1.0000000000009838e12, 1.000000000000054e12, 1.0000000000000393e12, 1.0000000000006516e12, 1.0000000000002982e12, 1.0000000000009418e12, 1.0000000000000588e12, 1.0000000000004891e12, 1.000000000000335e12, 1.0000000000000173e12, 1.0000000000001462e12, 1.0000000000002697e12, 1.0000000000000483e12, 1.0000000000003591e12, 1.0000000000009575e12, 1.0000000000005046e12, 1.0000000000000469e12, 1.0000000000009431e12, 1.0000000000007882e12, 1.000000000000962e12, 1.000000000000865e12, 1.000000000000155e12, 1.000000000000038e12, 1.0000000000003802e12, 1.0000000000004053e12, 1.0000000000004329e12, 1.0000000000006348e12, 1.0000000000009412e12, 1.0000000000003375e12, 1.0000000000007988e12, 1.000000000000482e12, 1.000000000000407e12, 1.0000000000001139e12, 1.0000000000006315e12, 1.0000000000000573e12, 1.0000000000002518e12, 1.000000000000906e12, 1.0000000000007684e12, 1.0000000000003804e12, 1.0000000000002927e12, 1.0000000000008252e12, 1.0000000000000266e12, 1.0000000000005348e12, 1.0000000000007694e12, 1.0000000000001473e12, 1.0000000000006338e12, 1.0000000000005836e12, 1.0000000000007792e12, 1.0000000000005366e12, 1.0000000000002231e12, 1.000000000000623e12, 1.0000000000003163e12, 1.0000000000008419e12, 1.0000000000006847e12, 1.0000000000005804e12, 1.000000000000068e12, 1.0000000000005236e12, 1.0000000000009244e12, 1.0000000000009698e12, 1.0000000000009288e12, 1.0000000000005693e12, 1.0000000000006023e12, 1.0000000000002853e12, 1.0000000000009161e12, 1.0000000000008955e12, 1.0000000000006903e12, 1.000000000000273e12, 1.0000000000002571e12, 1.0000000000000481e12, 1.000000000000343e12, 1.0000000000002885e12, 1.000000000000658e12]

lbs = [-9.999999999992169e11, -9.999999999996669e11, -9.999999999990638e11, -9.999999999995042e11, -9.999999999995912e11, -9.999999999992234e11, -9.999999999999098e11, -9.999999999991755e11, -9.999999999999569e11, -9.999999999990776e11, -9.999999999999224e11, -9.999999999996584e11, -9.999999999993726e11, -9.999999999999576e11, -9.999999999991099e11, -9.999999999990643e11, -9.999999999997766e11, -9.999999999995254e11, -9.999999999995583e11, -9.999999999992933e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999991045e11, -9.999999999996423e11, -9.999999999990361e11, -9.999999999997333e11, -9.999999999993528e11, -9.999999999993887e11, 1.1920928955078125e-7, -9.999999999990383e11, -9.999999999998053e11, -9.999999999993829e11, -9.999999999990548e11, -9.999999999990665e11, -9.999999999998585e11, -9.999999999995349e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999993796e11, -9.99999999999328e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999990261e11, 1.1920928955078125e-7, -9.999999999992701e11, -9.999999999990865e11, -9.999999999991748e11, -9.999999999996539e11, -9.999999999991104e11, -9.999999999999355e11, -9.999999999994198e11, -9.999999999997657e11, -9.999999999993402e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.99999999999841e11, -9.999999999993522e11, -9.999999999994008e11, -9.999999999994933e11, -9.999999999997007e11, -9.999999999996215e11, -9.999999999998083e11, -9.999999999995275e11, -9.999999999999718e11, -9.999999999990526e11, -9.999999999990874e11, -9.999999999994012e11, -9.999999999999148e11, -9.999999999993289e11, -9.999999999999839e11, -9.99999999999833e11, -9.999999999997343e11, -9.999999999996289e11, -9.999999999998129e11, -9.999999999997964e11, -9.999999999993507e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999998099e11, -9.999999999999852e11, -9.999999999991787e11, -9.999999999997802e11, -9.999999999998934e11, -9.99999999999491e11, 1.1920928955078125e-7, -9.999999999999861e11, -9.99999999999689e11, -9.999999999992256e11, -9.999999999996311e11, -9.999999999992483e11, -9.999999999999768e11, -9.999999999993004e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994946e11, -9.999999999998689e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999993158e11, -9.999999999995656e11, -9.99999999999144e11, -9.99999999999096e11, -9.999999999999077e11, -9.999999999991935e11, -9.999999999991974e11, -9.999999999992432e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999995476e11, -9.99999999999119e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
ubs = [1.0000000000007454e12, 1.0000000000006881e12, 1.0000000000002352e12, 1.0000000000000887e12, 1.000000000000267e12, 1.0000000000006276e12, 1.0000000000007416e12, 1.0000000000005256e12, 1.0000000000003162e12, 1.0000000000001301e12, 1.0000000000000929e12, 1.0000000000007092e12, 1.0000000000001482e12, 1.0000000000004445e12, 1.0000000000007737e12, 1.0000000000004109e12, 1.0000000000009922e12, 1.0000000000002102e12, 1.0000000000005298e12, 1.0000000000004242e12, 1.0000000000004824e12, 1.0000000000004956e12, 1.000000000000608e12, 1.0000000000003639e12, 1.0000000000002931e12, 1.0000000000005039e12, 1.0000000000004126e12, 1.0000000000008009e12, 1.0000000000003812e12, 1.0000000000004678e12, 1.000000000000962e12, 1.0000000000003231e12, 1.0000000000008568e12, 1.0000000000003435e12, 1.0000000000004156e12, 1.0000000000003132e12, 1.0000000000000248e12, 1.0000000000007047e12, 1.000000000000906e12, 1.0000000000009728e12, 1.0000000000009196e12, 1.0000000000004789e12, 1.0000000000002477e12, 1.0000000000007141e12, 1.0000000000008438e12, 1.000000000000949e12, 1.0000000000000721e12, 1.0000000000004729e12, 1.0000000000003215e12, 1.0000000000000095e12, 1.0000000000002747e12, 1.0000000000008464e12, 1.000000000000205e12, 1.0000000000000879e12, 1.0000000000002931e12, 1.0000000000007561e12, 1.0000000000007834e12, 1.0000000000008763e12, 1.0000000000002931e12, 1.0000000000001605e12, 1.0000000000001276e12, 1.0000000000001565e12, 1.0000000000005609e12, 1.000000000000137e12, 1.0000000000009791e12, 1.000000000000728e12, 1.0000000000004564e12, 1.0000000000005087e12, 1.0000000000002776e12, 1.0000000000005166e12, 1.000000000000132e12, 1.0000000000001534e12, 1.0000000000005536e12, 1.0000000000007208e12, 1.0000000000005076e12, 1.0000000000002491e12, 1.0000000000006022e12, 1.0000000000004696e12, 1.000000000000688e12, 1.0000000000003657e12, 1.0000000000001522e12, 1.0000000000009355e12, 1.0000000000009962e12, 1.0000000000005317e12, 1.0000000000006942e12, 1.0000000000009888e12, 1.0000000000001539e12, 1.0000000000006445e12, 1.0000000000002557e12, 1.0000000000009795e12, 1.0000000000008579e12, 1.0000000000006946e12, 1.0000000000003798e12, 1.0000000000008551e12, 1.0000000000004385e12, 1.000000000000184e12, 1.0000000000001947e12, 1.00000000000059e12, 1.0000000000001621e12, 1.0000000000006908e12, 1.0000000000004137e12, 1.0000000000001559e12, 1.0000000000007112e12, 1.0000000000000726e12, 1.0000000000008319e12, 1.0000000000009215e12, 1.0000000000006711e12, 1.000000000000423e12, 1.0000000000009786e12, 1.0000000000007246e12, 1.0000000000002511e12, 1.0000000000006433e12, 1.0000000000002512e12, 1.0000000000003032e12, 1.0000000000006791e12, 1.0000000000009635e12, 1.0000000000001199e12, 1.0000000000004841e12, 1.0000000000006301e12, 1.0000000000006458e12, 1.0000000000008599e12, 1.0000000000003329e12, 1.0000000000002465e12, 1.0000000000007041e12, 1.0000000000008188e12, 1.0000000000006385e12, 1.0000000000001459e12, 1.0000000000009069e12, 1.0000000000005725e12, 1.0000000000003309e12, 1.0000000000000153e12, 1.0000000000005326e12, 1.0000000000004305e12, 1.0000000000005038e12, 1.0000000000003544e12, 1.0000000000009727e12, 1.0000000000001838e12, 1.0000000000007097e12, 1.0000000000003715e12, 1.0000000000006802e12, 1.0000000000002219e12, 1.0000000000005836e12, 1.0000000000005159e12, 1.0000000000004913e12, 1.0000000000003772e12, 1.0000000000000349e12, 1.0000000000000896e12, 1.0000000000007861e12, 1.0000000000000171e12, 1.0000000000005046e12, 1.0000000000000026e12, 1.0000000000003008e12, 1.000000000000907e12, 1.0000000000001506e12, 1.000000000000707e12, 1.0000000000007765e12, 1.0000000000006825e12, 1.0000000000003625e12, 1.0000000000000052e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))





transformer_option = 1
using BenchmarkTools
@profview for i in 1:20 levenberg_marquardt(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option)) end#, steps = 4);

solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)
solls = reduce(hcat,[undo_transformer(i,lbs,ubs, option = transformer_option) for i in sol_new[2][2]])
solls_trans = reduce(hcat,sol_new[2][2])
sol_new[1]
sol_new[2][1]
sol_new[2][3]
sol_new[2][2]
sol_new[2][4]
sol_new[2][2][1]
# err = [sum(abs,ss_solve_blocks[7](parameters_and_solved_vars, i, transformer_option,lbs,ubs)) for i in sol_new[2][2]]
# normm = [ℒ.norm(ss_solve_blocks[7](parameters_and_solved_vars, i, transformer_option,lbs,ubs)) for i in sol_new[2][2]]

# sortperm(abs.(sol_new[1]))


# go over parameters
parameter_find = []
itter = 1
# for transformer_option in 0:1
# for σ¹ in exp10.(-4:1:-2) .* 5 
    # for σ² in exp10.(-4:1:-2) .* 5 
        # for γ in exp10.(-15:5:0)
            # for ϵ in .1:.2:.3
                for p in 1:.05:2
                    # for r in .4:.1:.7
                        for μ in exp10.(-8:.25:-6) 
                            # for steps in 1:1:5
                                # transformer_option = 1
                                println(itter)
                                itter += 1
                                sol = levenberg_marquardt(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),
                                transformer(previous_sol_init,lbs,ubs, option = transformer_option),
                                transformer(lbs,lbs,ubs, option = transformer_option),
                                transformer(ubs,lbs,ubs, option = transformer_option),
                                iterations = 1000, 
                                p = p, 
                                # σ¹ = σ¹, 
                                # σ² = σ², 
                                # r = r,
                                # ϵ = .1,
                                # steps = 1,
                                # γ = γ,
                                μ = μ)
                                push!(parameter_find,[sol[2][[1,3]]..., μ, p])
                            end
                        end
                    # end
                # end
            # end
#         end
#     end
# end


using DataFrames, GLM

# simul = DataFrame(reduce(hcat,parameter_find)',[:iter,:tol,:ρ, :μ, :σ¹, :σ², :r, :ϵ, :steps, :γ])
# simul = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ, :σ², :r, :steps, :γ])
simul = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ, :p])
sort!(simul, [:μ, :p])
simul[simul.tol .> 1e-8,:iter] .= 1000.0

simul[:,:iter] = simul.iter .< 1000

# ols = lm(@formula(iter ~ ρ+ μ+ σ¹+ σ²+ r+ ϵ+ steps+ γ), simul[simul.tol .< 1e-8 .&& simul.iter .< 50,:])
ols = glm(@formula(iter ~ μ * r + μ * p), simul,Binomial(), LogitLink())
ols = lm(@formula(iter ~ μ * p), simul[simul.tol .< 1e-8 .&& simul.iter .< 100, :])
# ols = lm(@formula(iter ~ μ + σ² + r + γ), simul[simul.tol .< 1e-8 .&& simul.iter .< 200 .&& simul.steps .== 1.0, :])
# ols = lm(@formula(iter ~ μ + σ² + r + steps + γ), simul[simul.tol .< 1e-8 .&& simul.iter .< 200 .&& simul.steps .== 1.0, :])
# ols = lm(@formula(iter ~ μ + σ² + r + steps + γ), simul)
simul[simul.tol .> 1e-8,:iter] .= 1000.0
simul = simul[simul.tol .< 1e-8 .&& simul.transformer_option .== 1.0, :]
sort!(simul, :iter)
simul[simul.steps .== 1.0,:]

parameter_find = [[levenberg_marquardt_ar(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),
                    transformer(previous_sol_init,lbs,ubs, option = transformer_option),
                    transformer(lbs,lbs,ubs, option = transformer_option),
                    transformer(ubs,lbs,ubs, option = transformer_option),
                    iterations = 1000, 
                    ρ = ρ, 
                    σ¹ = σ¹, 
                    σ² = σ², 
                    r = r,
                    ϵ = ϵ,
                    steps = steps,
                    γ = γ,
                    μ = μ)[2][[1,3]]..., ρ, μ, σ¹, σ², r, ϵ, steps, γ]
                    # for ρ in .5:.1:.9 
                        for μ in exp10.(-6:.5:-3) 
                            # for σ¹ in exp10.(-4:1:-2) .* 5 
                                for σ² in exp10.(-4:1:-2) .* 5 
                                    for r in .2:.2:.7 
                                        # for ϵ in .1:.1:.3
                                            for steps in 1:2:5
                                                for γ in exp10.(-15:6:-3)]

[[transformer_option, μ, r, p] 
for transformer_option in 0:1
    # for σ¹ in exp10.(-4:1:-2) .* 5 
        # for σ² in exp10.(-4:1:-2) .* 5 
            # for γ in exp10.(-15:5:0)
                # for ϵ in .1:.2:.3
                    for p in .5:.2:1.5
                        for r in .5:.07:.99
                            for μ in exp10.(-7:.5:-4)  ]

# r::T = .5, 
# μ::T = 1e-4, 
# ρ::T  = 0.8, 
# σ¹::T = 0.005, 
# σ²::T = 0.005, 
# ϵ::T = .1,
# γ::T = eps(),
# steps::S = 4,


log10(eps())


using Plots

Plots.plot(normm)
var1 = 22
var2 = 74 #74
var3 = 9 #9
# Create two vectors with random values
x = solls_trans[var1,:]
y = solls_trans[var2,:]
z = solls_trans[var3,:]
x = normm#solls[var1,:]
y = solls[var2,:]
z = solls[var3,:]


i = length(x)
Plots.plot(x[1:i], y[1:i], z[1:i], zcolor = z[1:i], m = (2, 0.8, :blues, Plots.stroke(0)), label = "Iteration $i")
Plots.plot!(x[1:i], y[1:i], fill(minimum(z),i), color = :black, line = :dash, linewidth = 1, alpha = .5, label = "")
xlims!(minimum(x), maximum(x))
ylims!(minimum(y), maximum(y))
zlims!(minimum(z), maximum(z))



# Initialize plot with first values of x and y
anim = @animate for i = 550:length(x)
    Plots.plot(x[1:i], y[1:i], z[1:i], zcolor = z[1:i], m = (2, 0.8, :blues, Plots.stroke(0)), label = "Iteration $i")
    Plots.plot!(x[1:i], y[1:i], fill(z[1],i), color = :black, line = :dash, linewidth = 1, alpha = .5)
    xlims!(minimum(x), maximum(x))
    ylims!(minimum(y), maximum(y))
    zlims!(minimum(z), maximum(z))
end

# Save animation as gif
gif(anim, "iteration.gif", fps = 5)

i = length(x)
Plots.plot(x[1:i], y[1:i], z[1:i], zcolor = z[1:i], m = (2, 0.8, :blues, Plots.stroke(0)), label = "Iteration $i")
Plots.plot!(x[1:i], y[1:i], fill(z[1],i), color = :black, line = :dash, linewidth = 1, alpha = .5, label = "")
xlims!(minimum(x), maximum(x))
ylims!(minimum(y), maximum(y))
zlims!(minimum(z), maximum(z))

p = plot([sin, cos], zeros(0), leg = false, xlims = (0, 2π), ylims = (-1, 1))
anim = Animation()
for x = range(0, stop = 2π, length = 20)
    push!(p, x, Float64[sin(x), cos(x)])
    frame(anim)
end



sol_new = nlboxsolve(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option),method = method, iterations = 1000)
solution = undo_transformer(sol_new.zero,lbs,ubs, option = transformer_option)



                #   block_solver_RD = block_solver_AD([EA_SIZE, EA_OMEGA, EA_BETA, EA_SIGMA, EA_KAPPA, EA_ZETA, EA_DELTA, EA_ETA, EA_ETAI, EA_ETAJ, EA_XII, EA_XIJ, EA_ALPHA, EA_THETA, EA_XIH, EA_XIX, EA_NUC, EA_MUC, EA_NUI, EA_MUI, EA_GAMMAU2, EA_BYTARGET, EA_PHITB, EA_TAUKBAR, EA_UPSILONT, EA_BFYTARGET, EA_PYBAR, EA_YBAR, EA_PIBAR, EA_PSIBAR, EA_QBAR, US_SIZE, US_OMEGA, US_BETA, US_SIGMA, US_KAPPA, US_ZETA, US_DELTA, US_ETA, US_ETAI, US_ETAJ, US_XII, US_XIJ, US_ALPHA, US_THETA, US_XIH, US_XIX, US_NUC, US_MUC, US_NUI, US_MUI, US_GAMMAU2, US_BYTARGET, US_PHITB, US_TAUKBAR, US_UPSILONT, US_PYBAR, US_YBAR, US_PIBAR, US_PSIBAR, US_QBAR, US_RER, EA_GAMMAIMC, EA_GAMMAIMCDAG, EA_GAMMAIMI, EA_GAMMAIMIDAG, EA_GAMMAVI, EA_GAMMAVIDER, EA_GAMMAVJ, EA_GAMMAVJDER, EA_GY, EA_PIC, EA_R, EA_TAUC, EA_TAUD, EA_TAUK, EA_TAUN, EA_TAUWF, EA_TAUWH, EA_TR, EA_TRJ, EA_VI, EA_VJ, EA_Z, US_GAMMAIMC, US_GAMMAIMCDAG, US_GAMMAIMI, US_GAMMAIMIDAG, US_GAMMAVI, US_GAMMAVIDER, US_GAMMAVJ, US_GAMMAVJDER, US_GY, US_PIC, US_R, US_TAUC, US_TAUD, US_TAUK, US_TAUN, US_TAUWF, US_TAUWH, US_TR, US_TRJ, US_VI, US_VJ, US_Z, ➕₃, ➕₄, ➕₆, ➕₇, ➕₈, ➕₁₂, ➕₁₃, ➕₁₅, ➕₁₆, ➕₁₈, ➕₁₉, ➕₂₂, ➕₂₉, ➕₃₁, ➕₃₃, ➕₃₆, ➕₃₇, ➕₃₉, ➕₄₀, ➕₄₁, ➕₄₅, ➕₄₆, ➕₄₈, ➕₄₉, ➕₅₁, ➕₅₂, ➕₅₅, ➕₆₂, ➕₆₄, ➕₆₆, ➕₆₇], 7, ss_solve_blocks[7], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([EA_SIZE, EA_OMEGA, EA_BETA, EA_SIGMA, EA_KAPPA, EA_ZETA, EA_DELTA, EA_ETA, EA_ETAI, EA_ETAJ, EA_XII, EA_XIJ, EA_ALPHA, EA_THETA, EA_XIH, EA_XIX, EA_NUC, EA_MUC, EA_NUI, EA_MUI, EA_GAMMAU2, EA_BYTARGET, EA_PHITB, EA_TAUKBAR, EA_UPSILONT, EA_BFYTARGET, EA_PYBAR, EA_YBAR, EA_PIBAR, EA_PSIBAR, EA_QBAR, US_SIZE, US_OMEGA, US_BETA, US_SIGMA, US_KAPPA, US_ZETA, US_DELTA, US_ETA, US_ETAI, US_ETAJ, US_XII, US_XIJ, US_ALPHA, US_THETA, US_XIH, US_XIX, US_NUC, US_MUC, US_NUI, US_MUI, US_GAMMAU2, US_BYTARGET, US_PHITB, US_TAUKBAR, US_UPSILONT, US_PYBAR, US_YBAR, US_PIBAR, US_PSIBAR, US_QBAR, US_RER, EA_GAMMAIMC, EA_GAMMAIMCDAG, EA_GAMMAIMI, EA_GAMMAIMIDAG, EA_GAMMAVI, EA_GAMMAVIDER, EA_GAMMAVJ, EA_GAMMAVJDER, EA_GY, EA_PIC, EA_R, EA_TAUC, EA_TAUD, EA_TAUK, EA_TAUN, EA_TAUWF, EA_TAUWH, EA_TR, EA_TRJ, EA_VI, EA_VJ, EA_Z, US_GAMMAIMC, US_GAMMAIMCDAG, US_GAMMAIMI, US_GAMMAIMIDAG, US_GAMMAVI, US_GAMMAVIDER, US_GAMMAVJ, US_GAMMAVJDER, US_GY, US_PIC, US_R, US_TAUC, US_TAUD, US_TAUK, US_TAUN, US_TAUWF, US_TAUWH, US_TR, US_TRJ, US_VI, US_VJ, US_Z, ➕₃, ➕₄, ➕₆, ➕₇, ➕₈, ➕₁₂, ➕₁₃, ➕₁₅, ➕₁₆, ➕₁₈, ➕₁₉, ➕₂₂, ➕₂₉, ➕₃₁, ➕₃₃, ➕₃₆, ➕₃₇, ➕₃₉, ➕₄₀, ➕₄₁, ➕₄₅, ➕₄₆, ➕₄₈, ➕₄₉, ➕₅₁, ➕₅₂, ➕₅₅, ➕₆₂, ➕₆₄, ➕₆₆, ➕₆₇])
solution_error += sum(abs2, (ss_solve_blocks[7])([EA_SIZE, EA_OMEGA, EA_BETA, EA_SIGMA, EA_KAPPA, EA_ZETA, EA_DELTA, EA_ETA, EA_ETAI, EA_ETAJ, EA_XII, EA_XIJ, EA_ALPHA, EA_THETA, EA_XIH, EA_XIX, EA_NUC, EA_MUC, EA_NUI, EA_MUI, EA_GAMMAU2, EA_BYTARGET, EA_PHITB, EA_TAUKBAR, EA_UPSILONT, EA_BFYTARGET, EA_PYBAR, EA_YBAR, EA_PIBAR, EA_PSIBAR, EA_QBAR, US_SIZE, US_OMEGA, US_BETA, US_SIGMA, US_KAPPA, US_ZETA, US_DELTA, US_ETA, US_ETAI, US_ETAJ, US_XII, US_XIJ, US_ALPHA, US_THETA, US_XIH, US_XIX, US_NUC, US_MUC, US_NUI, US_MUI, US_GAMMAU2, US_BYTARGET, US_PHITB, US_TAUKBAR, US_UPSILONT, US_PYBAR, US_YBAR, US_PIBAR, US_PSIBAR, US_QBAR, US_RER, EA_GAMMAIMC, EA_GAMMAIMCDAG, EA_GAMMAIMI, EA_GAMMAIMIDAG, EA_GAMMAVI, EA_GAMMAVIDER, EA_GAMMAVJ, EA_GAMMAVJDER, EA_GY, EA_PIC, EA_R, EA_TAUC, EA_TAUD, EA_TAUK, EA_TAUN, EA_TAUWF, EA_TAUWH, EA_TR, EA_TRJ, EA_VI, EA_VJ, EA_Z, US_GAMMAIMC, US_GAMMAIMCDAG, US_GAMMAIMI, US_GAMMAIMIDAG, US_GAMMAVI, US_GAMMAVIDER, US_GAMMAVJ, US_GAMMAVJDER, US_GY, US_PIC, US_R, US_TAUC, US_TAUD, US_TAUK, US_TAUN, US_TAUWF, US_TAUWH, US_TR, US_TRJ, US_VI, US_VJ, US_Z, ➕₃, ➕₄, ➕₆, ➕₇, ➕₈, ➕₁₂, ➕₁₃, ➕₁₅, ➕₁₆, ➕₁₈, ➕₁₉, ➕₂₂, ➕₂₉, ➕₃₁, ➕₃₃, ➕₃₆, ➕₃₇, ➕₃₉, ➕₄₀, ➕₄₁, ➕₄₅, ➕₄₆, ➕₄₈, ➕₄₉, ➕₅₁, ➕₅₂, ➕₅₅, ➕₆₂, ➕₆₄, ➕₆₆, ➕₆₇], solution, 0, lbs, ubs))
sol = solution
EAUS_RER = sol[1]
EA_B = sol[2]
EA_BF = sol[3]
EA_C = sol[4]
EA_CI = sol[5]
EA_CJ = sol[6]
EA_D = sol[7]
EA_FH = sol[8]
EA_FI = sol[9]
EA_FJ = sol[10]
EA_FX = sol[11]
EA_G = sol[12]
EA_GAMMAU = sol[13]
EA_GAMMAUDER = sol[14]
EA_GAMMAV = sol[15]
EA_GH = sol[16]
EA_GI = sol[17]
EA_GJ = sol[18]
EA_GX = sol[19]
EA_H = sol[20]
EA_HC = sol[21]
EA_HI = sol[22]
EA_I = sol[23]
EA_II = sol[24]
EA_IM = sol[25]
EA_IMC = sol[26]
EA_IMI = sol[27]
EA_K = sol[28]
EA_KD = sol[29]
EA_KI = sol[30]
EA_LAMBDAI = sol[31]
EA_LAMBDAJ = sol[32]
EA_M = sol[33]
EA_MC = sol[34]
EA_MI = sol[35]
EA_MJ = sol[36]
EA_ND = sol[37]
EA_NDI = sol[38]
EA_NDJ = sol[39]
EA_NJ = sol[40]
EA_PH = sol[41]
EA_PHTILDE = sol[42]
EA_PI = sol[43]
EA_PIM = sol[44]
EA_PIMTILDE = sol[45]
EA_PY = sol[46]
EA_Q = sol[47]
EA_QC = sol[48]
EA_QI = sol[49]
EA_RER = sol[50]
EA_RK = sol[51]
EA_SH = sol[52]
EA_SJ = sol[53]
EA_SX = sol[54]
EA_T = sol[55]
EA_TB = sol[56]
EA_TI = sol[57]
EA_TJ = sol[58]
EA_U = sol[59]
EA_W = sol[60]
EA_WI = sol[61]
EA_WITILDE = sol[62]
EA_WJ = sol[63]
EA_WJTILDE = sol[64]
EA_Y = sol[65]
EA_YS = sol[66]
USEA_RER = sol[67]
US_B = sol[68]
US_C = sol[69]
US_CI = sol[70]
US_CJ = sol[71]
US_D = sol[72]
US_FH = sol[73]
US_FI = sol[74]
US_FJ = sol[75]
US_FX = sol[76]
US_G = sol[77]
US_GAMMAU = sol[78]
US_GAMMAUDER = sol[79]
US_GAMMAV = sol[80]
US_GH = sol[81]
US_GI = sol[82]
US_GJ = sol[83]
US_GX = sol[84]
US_H = sol[85]
US_HC = sol[86]
US_HI = sol[87]
US_I = sol[88]
US_II = sol[89]
US_IM = sol[90]
US_IMC = sol[91]
US_IMI = sol[92]
US_K = sol[93]
US_KD = sol[94]
US_KI = sol[95]
US_LAMBDAI = sol[96]
US_LAMBDAJ = sol[97]
US_M = sol[98]
US_MC = sol[99]
US_MI = sol[100]
US_MJ = sol[101]
US_ND = sol[102]
US_NDI = sol[103]
US_NDJ = sol[104]
US_NJ = sol[105]
US_PH = sol[106]
US_PHTILDE = sol[107]
US_PI = sol[108]
US_PIM = sol[109]
US_PIMTILDE = sol[110]
US_PY = sol[111]
US_Q = sol[112]
US_QC = sol[113]
US_QI = sol[114]
US_RK = sol[115]
US_SH = sol[116]
US_SJ = sol[117]
US_SX = sol[118]
US_T = sol[119]
US_TI = sol[120]
US_TJ = sol[121]
US_U = sol[122]
US_W = sol[123]
US_WI = sol[124]
US_WITILDE = sol[125]
US_WJ = sol[126]
US_WJTILDE = sol[127]
US_Y = sol[128]
US_YS = sol[129]
➕₁ = sol[130]
➕₁₀ = sol[131]
➕₁₁ = sol[132]
➕₁₄ = sol[133]
➕₁₇ = sol[134]
➕₂₀ = sol[135]
➕₂₁ = sol[136]
➕₂₃ = sol[137]
➕₂₄ = sol[138]
➕₂₅ = sol[139]
➕₂₈ = sol[140]
➕₃₀ = sol[141]
➕₃₂ = sol[142]
➕₃₄ = sol[143]
➕₃₈ = sol[144]
➕₄₂ = sol[145]
➕₄₃ = sol[146]
➕₄₄ = sol[147]
➕₄₇ = sol[148]
➕₅ = sol[149]
➕₅₀ = sol[150]
➕₅₃ = sol[151]
➕₅₄ = sol[152]
➕₅₆ = sol[153]
➕₅₇ = sol[154]
➕₅₈ = sol[155]
➕₆₁ = sol[156]
➕₆₃ = sol[157]
➕₆₅ = sol[158]
➕₉ = sol[159]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
            sol
        else
            ℱ.value.(sol)
        end]
EA_YGAP = (EA_Y - EA_YBAR) / EA_YBAR
US_IY = (US_I * US_PI) / (US_PY * US_Y)
US_YGAP = (US_Y - US_YBAR) / US_YBAR
EA_UTILJ = ((EA_NJ ^ (EA_ZETA + 1) * EA_SIGMA - EA_NJ ^ (EA_ZETA + 1)) + EA_ZETA * ➕₅ ^ (1 - EA_SIGMA) + ➕₅ ^ (1 - EA_SIGMA)) / ((((((EA_BETA * EA_SIGMA * EA_ZETA + EA_BETA * EA_SIGMA) - EA_BETA * EA_ZETA) - EA_BETA) - EA_SIGMA * EA_ZETA) - EA_SIGMA) + EA_ZETA + 1)
EA_IMIY = (EA_IMI * EA_PIM) / (EA_PY * EA_Y)
➕₂₇ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (EA_CHII / 4 - 1 / 4) * EA_PIC ^ (1 - EA_CHII)), 1.0000000000007839e12)
➕₅₉ = min(max(1.1920928955078125e-7, US_WITILDE / US_WI), 1.0000000000005132e12)
➕₆₀ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (US_CHII / 4 - 1 / 4) * US_PIC ^ (1 - US_CHII)), 1.0000000000003038e12)
US_SI = (US_XII - 1) / (➕₅₉ ^ US_ETAI * (US_XII * ➕₆₀ ^ US_ETAI - 1))
US_NI = US_NDI * US_SI
US_UTILI = ((US_NI ^ (US_ZETA + 1) * US_SIGMA - US_NI ^ (US_ZETA + 1)) + US_ZETA * ➕₃₄ ^ (1 - US_SIGMA) + ➕₃₄ ^ (1 - US_SIGMA)) / ((((((US_BETA * US_SIGMA * US_ZETA + US_BETA * US_SIGMA) - US_BETA * US_ZETA) - US_BETA) - US_SIGMA * US_ZETA) - US_SIGMA) + US_ZETA + 1)
➕₂₆ = min(max(1.1920928955078125e-7, EA_WITILDE / EA_WI), 1.000000000000116e12)
EA_SI = (EA_XII - 1) / (➕₂₆ ^ EA_ETAI * (EA_XII * ➕₂₇ ^ EA_ETAI - 1))
EA_NI = EA_NDI * EA_SI
US_BY = US_B / (US_PYBAR * US_YBAR)
US_IMIY = (US_IMI * US_PIM) / (US_PY * US_Y)
US_BF = (-EA_BF * EA_SIZE) / US_SIZE
EA_RR = EA_R / EA_PIC
EA_TY = EA_T / (EA_PYBAR * EA_YBAR)
EA_IY = (EA_I * EA_PI) / (EA_PY * EA_Y)
US_RR = US_R / US_PIC
US_YGROWTH = 1
US_TY = US_T / (US_PYBAR * US_YBAR)
EA_TOT = EA_PIM / (EA_RER * US_PIM)
EA_BY = EA_B / (EA_PYBAR * EA_YBAR)
US_UTILJ = ((US_NJ ^ (US_ZETA + 1) * US_SIGMA - US_NJ ^ (US_ZETA + 1)) + US_ZETA * ➕₃₈ ^ (1 - US_SIGMA) + ➕₃₈ ^ (1 - US_SIGMA)) / ((((((US_BETA * US_SIGMA * US_ZETA + US_BETA * US_SIGMA) - US_BETA * US_ZETA) - US_BETA) - US_SIGMA * US_ZETA) - US_SIGMA) + US_ZETA + 1)
US_IMY = (US_IM * US_PIM) / (US_PY * US_Y)
EA_EPSILONM = -0.125 / (EA_R * ((EA_GAMMAV2 * EA_R + EA_R) - 1.0))
US_CY = US_C / (US_PY * US_Y)
EA_CY = EA_C / (EA_PY * EA_Y)
EA_IMY = (EA_IM * EA_PIM) / (EA_PY * EA_Y)
US_IMCY = (US_IMC * US_PIM) / (US_PY * US_Y)
US_EPSILONM = -0.125 / (US_R * ((US_GAMMAV2 * US_R + US_R) - 1.0))
EA_YGROWTH = 1
EA_UTILI = ((EA_NI ^ (EA_ZETA + 1) * EA_SIGMA - EA_NI ^ (EA_ZETA + 1)) + EA_ZETA * ➕₁ ^ (1 - EA_SIGMA) + ➕₁ ^ (1 - EA_SIGMA)) / ((((((EA_BETA * EA_SIGMA * EA_ZETA + EA_BETA * EA_SIGMA) - EA_BETA * EA_ZETA) - EA_BETA) - EA_SIGMA * EA_ZETA) - EA_SIGMA) + EA_ZETA + 1)
EA_YSHARE = (EA_PY * EA_SIZE * EA_Y * US_RER) / (EA_PY * EA_SIZE * EA_Y * US_RER + EA_RER * US_PY * US_SIZE * US_Y)
US_YSHARE = (EA_RER * US_PY * US_SIZE * US_Y) / (EA_PY * EA_SIZE * EA_Y * US_RER + EA_RER * US_PY * US_SIZE * US_Y)
EA_IMCY = (EA_IMC * EA_PIM) / (EA_PY * EA_Y)

                #   if length(NSSS_solver_cache_tmp) == 0
                #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:840 =#
                #       NSSS_solver_cache_tmp = [params_scaled_flt]
                #   else
                #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:840 =#
                #       NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., params_scaled_flt]
                #   end
                #   current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - params_flt)
                #   for pars = 𝓂.NSSS_solver_cache
                #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:845 =#
                #       latest = sum(abs2, pars[end] - params_flt)
                #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:846 =#
                #       if latest <= current_best
                #           #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:847 =#
                #           current_best = latest
                #       end
                #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:849 =#
                #   end
                #   if current_best > eps(Float32) && solution_error < eps(Float64)
                #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:852 =#
                #       reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
                #       #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:853 =#
                #       solved_scale = scale
                #   end
                #   #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:907 =#
                #   if scale == 1
                      #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:909 =#
return ([EAUS_RER, EA_B, EA_BF, EA_BY, EA_C, EA_CI, EA_CJ, EA_CY, EA_D, EA_EPSILONM, EA_FH, EA_FI, EA_FJ, EA_FX, EA_G, EA_GAMMAB, EA_GAMMAI, EA_GAMMAIDER, EA_GAMMAIMC, EA_GAMMAIMCDAG, EA_GAMMAIMI, EA_GAMMAIMIDAG, EA_GAMMAU, EA_GAMMAUDER, EA_GAMMAV, EA_GAMMAVI, EA_GAMMAVIDER, EA_GAMMAVJ, EA_GAMMAVJDER, EA_GH, EA_GI, EA_GJ, EA_GX, EA_GY, EA_H, EA_HC, EA_HI, EA_I, EA_II, EA_IM, EA_IMC, EA_IMCY, EA_IMI, EA_IMIY, EA_IMY, EA_IY, EA_K, EA_KD, EA_KI, EA_LAMBDAI, EA_LAMBDAJ, EA_M, EA_MC, EA_MI, EA_MJ, EA_M, EA_ND, EA_NDI, EA_NDJ, EA_NI, EA_NJ, EA_PH, EA_PHTILDE, EA_PI, EA_PIC, EA_PIC4, EA_PIC, EA_PIC, EA_PIH, EA_PIIM, EA_PIM, EA_PIMTILDE, EA_PY, EA_Q, EA_QC, EA_QI, EA_R, EA_RER, EA_RERDEP, EA_RK, EA_RP, EA_RR, EA_SH, EA_SI, EA_SJ, EA_SX, EA_T, EA_TAUC, EA_TAUD, EA_TAUK, EA_TAUN, EA_TAUWF, EA_TAUWH, EA_TB, EA_TI, EA_TJ, EA_TOT, EA_TR, EA_TRI, EA_TRJ, EA_TRY, EA_TY, EA_U, EA_UTILI, EA_UTILJ, EA_VI, EA_VJ, EA_W, EA_WI, EA_WITILDE, EA_WJ, EA_WJTILDE, EA_Y, EA_YGAP, EA_YGROWTH, EA_YS, EA_YSHARE, EA_Z, USEA_RER, US_B, US_BF, US_BY, US_C, US_CI, US_CJ, US_CY, US_D, US_EPSILONM, US_FH, US_FI, US_FJ, US_FX, US_G, US_GAMMAI, US_GAMMAIDER, US_GAMMAIMC, US_GAMMAIMCDAG, US_GAMMAIMI, US_GAMMAIMIDAG, US_GAMMAU, US_GAMMAUDER, US_GAMMAV, US_GAMMAVI, US_GAMMAVIDER, US_GAMMAVJ, US_GAMMAVJDER, US_GH, US_GI, US_GJ, US_GX, US_GY, US_H, US_HC, US_HI, US_I, US_II, US_IM, US_IMC, US_IMCY, US_IMI, US_IMIY, US_IMY, US_IY, US_K, US_KD, US_KI, US_LAMBDAI, US_LAMBDAJ, US_M, US_MC, US_MI, US_MJ, US_M, US_ND, US_NDI, US_NDJ, US_NI, US_NJ, US_PH, US_PHTILDE, US_PI, US_PIC, US_PIC4, US_PIC, US_PIC, US_PIH, US_PIIM, US_PIM, US_PIMTILDE, US_PY, US_Q, US_QC, US_QI, US_R, US_RK, US_RR, US_SH, US_SI, US_SJ, US_SX, US_T, US_TAUC, US_TAUD, US_TAUK, US_TAUN, US_TAUWF, US_TAUWH, US_TI, US_TJ, US_TR, US_TRI, US_TRJ, US_TRY, US_TY, US_U, US_UTILI, US_UTILJ, US_VI, US_VJ, US_W, US_WI, US_WITILDE, US_WJ, US_WJTILDE, US_Y, US_YGAP, US_YGROWTH, US_YS, US_YSHARE, US_Z], solution_error)
                  end
                  #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:911 =#
              end
              #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:912 =#
          end