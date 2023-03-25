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
    p::T = 1.8,
    λ¹::T = .5, 
    λ²::T = .5,
    λᵖ::T = .9, 
    μ¹::T = .0001,
    μ²::T = .0001
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

sol_new, info = levenberg_marquardt(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),
                                    transformer(previous_sol_init,lbs,ubs, option = transformer_option),
                                    transformer(lbs,lbs,ubs, option = transformer_option),
                                    transformer(ubs,lbs,ubs, option = transformer_option), 
                                    iterations = 100,
                                    # μ¹ = .0001, 
                                    # μ² = 1e-5, 
                                    # p = 1.9, 
                                    # λ¹ = .6, 
                                    # λ² = .7, 
                                    # λᵖ = .9, 
                                    # ρ = .1
                                    )#, λ² = .7, p = 1.4, λ¹ = .5,  μ = .001)#, λ = .5)#, p = 1.4)#, p =10.0)
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


sol_new, info = levenberg_marquardt(x->ss_solve_blocks[2](parameters_and_solved_vars, x, transformer_option,lbs,ubs),
                                    transformer(previous_sol_init,lbs,ubs, option = transformer_option),
                                    transformer(lbs,lbs,ubs, option = transformer_option),
                                    transformer(ubs,lbs,ubs, option = transformer_option),
                                    # μ¹ = .0001, 
                                    # μ² = 1e-5, 
                                    # p = 1.9, 
                                    # λ¹ = .6, 
                                    # λ² = .7, 
                                    # λᵖ = .9, 
                                    # ρ = .1
                                    )#, μ = 3e-3)#,μ = 1e-5, p = 3.0)

info[[1,3]]

# sol_new = nlboxsolve(x->ss_solve_blocks[2](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
sol_new[2]
solution = undo_transformer(sol_new,lbs,ubs, option = transformer_option)
                #   block_solver_RD = block_solver_AD([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR], 1, ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR])
solution_error += sum(abs2, (ss_solve_blocks[2])(parameters_and_solved_vars, solution, 0, lbs, ubs))
