# using NLboxsolve

method = :lm_ar
verbose = true
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
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:697 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:699 =#
    guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:701 =#
    US_PIC = guess[1]
    US_PIC4 = guess[2]
    US_R = guess[3]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:702 =#
    US_BETA = parameters_and_solved_vars[1]
    US_PI4TARGET = parameters_and_solved_vars[2]
    US_PHIRR = parameters_and_solved_vars[3]
    US_PHIRPI = parameters_and_solved_vars[4]
    US_PHIRGY = parameters_and_solved_vars[5]
    US_RRSTAR = parameters_and_solved_vars[6]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:703 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:708 =#
    return [-(US_PIC ^ 4) + US_PIC4, US_R - US_PIC / US_BETA, ((-US_PHIRR * (US_R ^ 4 - 1) + US_R ^ 4) - (1 - US_PHIRR) * ((US_PHIRPI * (-US_PI4TARGET + US_PIC4) + US_PI4TARGET * US_RRSTAR ^ 4) - 1)) - 1]
end)

push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:697 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:699 =#
guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:701 =#
EA_PIC = guess[1]
EA_PIC4 = guess[2]
EA_R = guess[3]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:702 =#
EA_BETA = parameters_and_solved_vars[1]
EA_PI4TARGET = parameters_and_solved_vars[2]
EA_PHIRR = parameters_and_solved_vars[3]
EA_PHIRPI = parameters_and_solved_vars[4]
EA_PHIRGY = parameters_and_solved_vars[5]
EA_RRSTAR = parameters_and_solved_vars[6]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:703 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:708 =#
return [-(EA_PIC ^ 4) + EA_PIC4, EA_R - EA_PIC / EA_BETA, ((-EA_PHIRR * (EA_R ^ 4 - 1) + EA_R ^ 4) - (1 - EA_PHIRR) * ((EA_PHIRPI * (-EA_PI4TARGET + EA_PIC4) + EA_PI4TARGET * EA_RRSTAR ^ 4) - 1)) - 1]
end)



push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:697 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:699 =#
guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:701 =#
US_GAMMAVJDER = guess[1]
US_VJ = guess[2]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:702 =#
US_BETA = parameters_and_solved_vars[1]
US_GAMMAV1 = parameters_and_solved_vars[2]
US_GAMMAV2 = parameters_and_solved_vars[3]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:703 =#
US_PIC = parameters_and_solved_vars[4]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:708 =#
return [-US_GAMMAV1 + US_GAMMAV2 / US_VJ ^ 2 + US_GAMMAVJDER, (US_BETA / US_PIC + US_GAMMAVJDER * US_VJ ^ 2) - 1]
end)




push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:697 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:699 =#
guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:701 =#
US_GAMMAVIDER = guess[1]
US_VI = guess[2]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:702 =#
US_BETA = parameters_and_solved_vars[1]
US_GAMMAV1 = parameters_and_solved_vars[2]
US_GAMMAV2 = parameters_and_solved_vars[3]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:703 =#
US_PIC = parameters_and_solved_vars[4]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:708 =#
return [(US_BETA / US_PIC + US_GAMMAVIDER * US_VI ^ 2) - 1, -US_GAMMAV1 + US_GAMMAV2 / US_VI ^ 2 + US_GAMMAVIDER]
end)



push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:697 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:699 =#
guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:701 =#
EA_GAMMAVJDER = guess[1]
EA_VJ = guess[2]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:702 =#
EA_BETA = parameters_and_solved_vars[1]
EA_GAMMAV1 = parameters_and_solved_vars[2]
EA_GAMMAV2 = parameters_and_solved_vars[3]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:703 =#
EA_PIC = parameters_and_solved_vars[4]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:708 =#
return [-EA_GAMMAV1 + EA_GAMMAV2 / EA_VJ ^ 2 + EA_GAMMAVJDER, (EA_BETA / EA_PIC + EA_GAMMAVJDER * EA_VJ ^ 2) - 1]
end)



push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:697 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:699 =#
guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:701 =#
EA_GAMMAVIDER = guess[1]
EA_VI = guess[2]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:702 =#
EA_BETA = parameters_and_solved_vars[1]
EA_GAMMAV1 = parameters_and_solved_vars[2]
EA_GAMMAV2 = parameters_and_solved_vars[3]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:703 =#
EA_PIC = parameters_and_solved_vars[4]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:708 =#
return [-EA_GAMMAV1 + EA_GAMMAV2 / EA_VI ^ 2 + EA_GAMMAVIDER, (EA_BETA / EA_PIC + EA_GAMMAVIDER * EA_VI ^ 2) - 1]
end)



push!(ss_solve_blocks,(parameters_and_solved_vars, guess, transformer_option, lbs, ubs) -> begin
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:697 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:699 =#
guess = undo_transformer(guess, lbs, ubs, option = transformer_option)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:701 =#
EAUS_RER = guess[1]
EA_B = guess[2]
EA_BF = guess[3]
EA_C = guess[4]
EA_CI = guess[5]
EA_CJ = guess[6]
EA_D = guess[7]
EA_FH = guess[8]
EA_FI = guess[9]
EA_FJ = guess[10]
EA_FX = guess[11]
EA_G = guess[12]
EA_GAMMAU = guess[13]
EA_GAMMAUDER = guess[14]
EA_GAMMAV = guess[15]
EA_GH = guess[16]
EA_GI = guess[17]
EA_GJ = guess[18]
EA_GX = guess[19]
EA_H = guess[20]
EA_HC = guess[21]
EA_HI = guess[22]
EA_I = guess[23]
EA_II = guess[24]
EA_IM = guess[25]
EA_IMC = guess[26]
EA_IMI = guess[27]
EA_K = guess[28]
EA_KD = guess[29]
EA_KI = guess[30]
EA_LAMBDAI = guess[31]
EA_LAMBDAJ = guess[32]
EA_M = guess[33]
EA_MC = guess[34]
EA_MI = guess[35]
EA_MJ = guess[36]
EA_ND = guess[37]
EA_NDI = guess[38]
EA_NDJ = guess[39]
EA_NJ = guess[40]
EA_PH = guess[41]
EA_PHTILDE = guess[42]
EA_PI = guess[43]
EA_PIM = guess[44]
EA_PIMTILDE = guess[45]
EA_PY = guess[46]
EA_Q = guess[47]
EA_QC = guess[48]
EA_QI = guess[49]
EA_RER = guess[50]
EA_RK = guess[51]
EA_SH = guess[52]
EA_SJ = guess[53]
EA_SX = guess[54]
EA_T = guess[55]
EA_TB = guess[56]
EA_TI = guess[57]
EA_TJ = guess[58]
EA_U = guess[59]
EA_W = guess[60]
EA_WI = guess[61]
EA_WITILDE = guess[62]
EA_WJ = guess[63]
EA_WJTILDE = guess[64]
EA_Y = guess[65]
EA_YS = guess[66]
USEA_RER = guess[67]
US_B = guess[68]
US_C = guess[69]
US_CI = guess[70]
US_CJ = guess[71]
US_D = guess[72]
US_FH = guess[73]
US_FI = guess[74]
US_FJ = guess[75]
US_FX = guess[76]
US_G = guess[77]
US_GAMMAU = guess[78]
US_GAMMAUDER = guess[79]
US_GAMMAV = guess[80]
US_GH = guess[81]
US_GI = guess[82]
US_GJ = guess[83]
US_GX = guess[84]
US_H = guess[85]
US_HC = guess[86]
US_HI = guess[87]
US_I = guess[88]
US_II = guess[89]
US_IM = guess[90]
US_IMC = guess[91]
US_IMI = guess[92]
US_K = guess[93]
US_KD = guess[94]
US_KI = guess[95]
US_LAMBDAI = guess[96]
US_LAMBDAJ = guess[97]
US_M = guess[98]
US_MC = guess[99]
US_MI = guess[100]
US_MJ = guess[101]
US_ND = guess[102]
US_NDI = guess[103]
US_NDJ = guess[104]
US_NJ = guess[105]
US_PH = guess[106]
US_PHTILDE = guess[107]
US_PI = guess[108]
US_PIM = guess[109]
US_PIMTILDE = guess[110]
US_PY = guess[111]
US_Q = guess[112]
US_QC = guess[113]
US_QI = guess[114]
US_RK = guess[115]
US_SH = guess[116]
US_SJ = guess[117]
US_SX = guess[118]
US_T = guess[119]
US_TI = guess[120]
US_TJ = guess[121]
US_U = guess[122]
US_W = guess[123]
US_WI = guess[124]
US_WITILDE = guess[125]
US_WJ = guess[126]
US_WJTILDE = guess[127]
US_Y = guess[128]
US_YS = guess[129]
➕₁ = guess[130]
➕₁₀ = guess[131]
➕₁₁ = guess[132]
➕₁₄ = guess[133]
➕₁₇ = guess[134]
➕₂₀ = guess[135]
➕₂₁ = guess[136]
➕₂₃ = guess[137]
➕₂₄ = guess[138]
➕₂₅ = guess[139]
➕₂₈ = guess[140]
➕₃₀ = guess[141]
➕₃₂ = guess[142]
➕₃₄ = guess[143]
➕₃₈ = guess[144]
➕₄₂ = guess[145]
➕₄₃ = guess[146]
➕₄₄ = guess[147]
➕₄₇ = guess[148]
➕₅ = guess[149]
➕₅₀ = guess[150]
➕₅₃ = guess[151]
➕₅₄ = guess[152]
➕₅₆ = guess[153]
➕₅₇ = guess[154]
➕₅₈ = guess[155]
➕₆₁ = guess[156]
➕₆₃ = guess[157]
➕₆₅ = guess[158]
➕₉ = guess[159]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:702 =#
EA_SIZE = parameters_and_solved_vars[1]
EA_OMEGA = parameters_and_solved_vars[2]
EA_BETA = parameters_and_solved_vars[3]
EA_SIGMA = parameters_and_solved_vars[4]
EA_KAPPA = parameters_and_solved_vars[5]
EA_ZETA = parameters_and_solved_vars[6]
EA_DELTA = parameters_and_solved_vars[7]
EA_ETA = parameters_and_solved_vars[8]
EA_ETAI = parameters_and_solved_vars[9]
EA_ETAJ = parameters_and_solved_vars[10]
EA_XII = parameters_and_solved_vars[11]
EA_XIJ = parameters_and_solved_vars[12]
EA_ALPHA = parameters_and_solved_vars[13]
EA_THETA = parameters_and_solved_vars[14]
EA_XIH = parameters_and_solved_vars[15]
EA_XIX = parameters_and_solved_vars[16]
EA_NUC = parameters_and_solved_vars[17]
EA_MUC = parameters_and_solved_vars[18]
EA_NUI = parameters_and_solved_vars[19]
EA_MUI = parameters_and_solved_vars[20]
EA_GAMMAU2 = parameters_and_solved_vars[21]
EA_BYTARGET = parameters_and_solved_vars[22]
EA_PHITB = parameters_and_solved_vars[23]
EA_TAUKBAR = parameters_and_solved_vars[24]
EA_UPSILONT = parameters_and_solved_vars[25]
EA_BFYTARGET = parameters_and_solved_vars[26]
EA_PYBAR = parameters_and_solved_vars[27]
EA_YBAR = parameters_and_solved_vars[28]
EA_PIBAR = parameters_and_solved_vars[29]
EA_PSIBAR = parameters_and_solved_vars[30]
EA_QBAR = parameters_and_solved_vars[31]
US_SIZE = parameters_and_solved_vars[32]
US_OMEGA = parameters_and_solved_vars[33]
US_BETA = parameters_and_solved_vars[34]
US_SIGMA = parameters_and_solved_vars[35]
US_KAPPA = parameters_and_solved_vars[36]
US_ZETA = parameters_and_solved_vars[37]
US_DELTA = parameters_and_solved_vars[38]
US_ETA = parameters_and_solved_vars[39]
US_ETAI = parameters_and_solved_vars[40]
US_ETAJ = parameters_and_solved_vars[41]
US_XII = parameters_and_solved_vars[42]
US_XIJ = parameters_and_solved_vars[43]
US_ALPHA = parameters_and_solved_vars[44]
US_THETA = parameters_and_solved_vars[45]
US_XIH = parameters_and_solved_vars[46]
US_XIX = parameters_and_solved_vars[47]
US_NUC = parameters_and_solved_vars[48]
US_MUC = parameters_and_solved_vars[49]
US_NUI = parameters_and_solved_vars[50]
US_MUI = parameters_and_solved_vars[51]
US_GAMMAU2 = parameters_and_solved_vars[52]
US_BYTARGET = parameters_and_solved_vars[53]
US_PHITB = parameters_and_solved_vars[54]
US_TAUKBAR = parameters_and_solved_vars[55]
US_UPSILONT = parameters_and_solved_vars[56]
US_PYBAR = parameters_and_solved_vars[57]
US_YBAR = parameters_and_solved_vars[58]
US_PIBAR = parameters_and_solved_vars[59]
US_PSIBAR = parameters_and_solved_vars[60]
US_QBAR = parameters_and_solved_vars[61]
US_RER = parameters_and_solved_vars[62]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:703 =#
EA_GAMMAIMC = parameters_and_solved_vars[63]
EA_GAMMAIMCDAG = parameters_and_solved_vars[64]
EA_GAMMAIMI = parameters_and_solved_vars[65]
EA_GAMMAIMIDAG = parameters_and_solved_vars[66]
EA_GAMMAVI = parameters_and_solved_vars[67]
EA_GAMMAVIDER = parameters_and_solved_vars[68]
EA_GAMMAVJ = parameters_and_solved_vars[69]
EA_GAMMAVJDER = parameters_and_solved_vars[70]
EA_GY = parameters_and_solved_vars[71]
EA_PIC = parameters_and_solved_vars[72]
EA_R = parameters_and_solved_vars[73]
EA_TAUC = parameters_and_solved_vars[74]
EA_TAUD = parameters_and_solved_vars[75]
EA_TAUK = parameters_and_solved_vars[76]
EA_TAUN = parameters_and_solved_vars[77]
EA_TAUWF = parameters_and_solved_vars[78]
EA_TAUWH = parameters_and_solved_vars[79]
EA_TR = parameters_and_solved_vars[80]
EA_TRJ = parameters_and_solved_vars[81]
EA_VI = parameters_and_solved_vars[82]
EA_VJ = parameters_and_solved_vars[83]
EA_Z = parameters_and_solved_vars[84]
US_GAMMAIMC = parameters_and_solved_vars[85]
US_GAMMAIMCDAG = parameters_and_solved_vars[86]
US_GAMMAIMI = parameters_and_solved_vars[87]
US_GAMMAIMIDAG = parameters_and_solved_vars[88]
US_GAMMAVI = parameters_and_solved_vars[89]
US_GAMMAVIDER = parameters_and_solved_vars[90]
US_GAMMAVJ = parameters_and_solved_vars[91]
US_GAMMAVJDER = parameters_and_solved_vars[92]
US_GY = parameters_and_solved_vars[93]
US_PIC = parameters_and_solved_vars[94]
US_R = parameters_and_solved_vars[95]
US_TAUC = parameters_and_solved_vars[96]
US_TAUD = parameters_and_solved_vars[97]
US_TAUK = parameters_and_solved_vars[98]
US_TAUN = parameters_and_solved_vars[99]
US_TAUWF = parameters_and_solved_vars[100]
US_TAUWH = parameters_and_solved_vars[101]
US_TR = parameters_and_solved_vars[102]
US_TRJ = parameters_and_solved_vars[103]
US_VI = parameters_and_solved_vars[104]
US_VJ = parameters_and_solved_vars[105]
US_Z = parameters_and_solved_vars[106]
➕₃ = parameters_and_solved_vars[107]
➕₄ = parameters_and_solved_vars[108]
➕₆ = parameters_and_solved_vars[109]
➕₇ = parameters_and_solved_vars[110]
➕₈ = parameters_and_solved_vars[111]
➕₁₂ = parameters_and_solved_vars[112]
➕₁₃ = parameters_and_solved_vars[113]
➕₁₅ = parameters_and_solved_vars[114]
➕₁₆ = parameters_and_solved_vars[115]
➕₁₈ = parameters_and_solved_vars[116]
➕₁₉ = parameters_and_solved_vars[117]
➕₂₂ = parameters_and_solved_vars[118]
➕₂₉ = parameters_and_solved_vars[119]
➕₃₁ = parameters_and_solved_vars[120]
➕₃₃ = parameters_and_solved_vars[121]
➕₃₆ = parameters_and_solved_vars[122]
➕₃₇ = parameters_and_solved_vars[123]
➕₃₉ = parameters_and_solved_vars[124]
➕₄₀ = parameters_and_solved_vars[125]
➕₄₁ = parameters_and_solved_vars[126]
➕₄₅ = parameters_and_solved_vars[127]
➕₄₆ = parameters_and_solved_vars[128]
➕₄₈ = parameters_and_solved_vars[129]
➕₄₉ = parameters_and_solved_vars[130]
➕₅₁ = parameters_and_solved_vars[131]
➕₅₂ = parameters_and_solved_vars[132]
➕₅₅ = parameters_and_solved_vars[133]
➕₆₂ = parameters_and_solved_vars[134]
➕₆₄ = parameters_and_solved_vars[135]
➕₆₆ = parameters_and_solved_vars[136]
➕₆₇ = parameters_and_solved_vars[137]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:708 =#
return [((((((((((-EA_B / EA_R + EA_B / EA_PIC) - EA_C * EA_TAUC) - EA_D * EA_TAUD) + EA_G * EA_PH) - EA_K * EA_TAUK * (-EA_PI * (EA_DELTA + EA_GAMMAU) + EA_RK * EA_U)) - EA_M) + EA_M / EA_PIC) - EA_ND * EA_TAUWF * EA_W) - EA_T) + EA_TR) - (EA_TAUN + EA_TAUWH) * (EA_NDI * EA_WI + EA_NDJ * EA_WJ), EA_PI - EA_Q, (-US_GAMMAU2 * (US_U - 1) + US_GAMMAUDER) - (-US_DELTA * US_PIBAR * US_TAUKBAR + US_QBAR * ((US_DELTA - 1) + 1 / US_BETA)) / (US_PIBAR * (1 - US_TAUKBAR)), EA_G * EA_PH - EA_GY * EA_PYBAR * EA_YBAR, (-EA_BETA * EA_FJ * EA_XIJ * ➕₆ ^ (EA_ETAJ * (EA_ZETA + 1)) + EA_FJ) - EA_NDJ ^ (EA_ZETA + 1) * EA_WJ ^ (EA_ETAJ * (EA_ZETA + 1)), -EA_BETA * (EA_DELTA * EA_PI * EA_TAUK + EA_Q * (1 - EA_DELTA) + (1 - EA_TAUK) * (-EA_GAMMAU * EA_PI + EA_RK * EA_U)) + EA_Q, (-EA_XIX * ➕₁₇ ^ (1 - EA_THETA) * ➕₁₈ ^ (1 - EA_THETA) + US_PIM ^ (1 - EA_THETA)) - US_PIMTILDE ^ (1 - EA_THETA) * (1 - EA_XIX), US_HI - (US_NUI * US_QI) / ➕₅₈ ^ US_MUI, ((((((((((-US_B / US_R + US_B / US_PIC) - US_C * US_TAUC) - US_D * US_TAUD) + US_G * US_PH) - US_K * US_TAUK * (-US_PI * (US_DELTA + US_GAMMAU) + US_RK * US_U)) - US_M) + US_M / US_PIC) - US_ND * US_TAUWF * US_W) - US_T) + US_TR) - (US_TAUN + US_TAUWH) * (US_NDI * US_WI + US_NDJ * US_WJ), (-(EA_HC ^ (1 - 1 / EA_MUC)) * EA_NUC ^ (1 / EA_MUC) + EA_QC ^ ((EA_MUC - 1) / EA_MUC)) - ➕₁₉ ^ (1 / EA_MUC) * ➕₂₀ ^ (1 - 1 / EA_MUC), (-US_CJ * (US_TAUC + 1)) / US_MJ + US_VJ, (-US_BETA * US_GI * US_XII * ➕₃₆ ^ (US_ETAI - 1) + US_GI) - US_LAMBDAI * US_NDI * US_WI ^ US_ETAI * ((-US_TAUN - US_TAUWH) + 1), ((-EA_IM * EA_SIZE * US_SX) / US_SIZE - US_H * US_SH) + US_YS, (US_PH ^ (1 - US_THETA) - US_PHTILDE ^ (1 - US_THETA) * (1 - US_XIH)) - US_XIH * ➕₄₇ ^ (1 - US_THETA) * ➕₄₈ ^ (1 - US_THETA), (-US_NUI * US_PH ^ (1 - US_MUI) + US_PI ^ (1 - US_MUI)) - ➕₅₇ ^ (1 - US_MUI) * (1 - US_NUI), (-(EA_WJ ^ (1 - EA_ETAJ)) * EA_XIJ * ➕₇ ^ (1 - EA_ETAJ) + EA_WJ ^ (1 - EA_ETAJ)) - EA_WJTILDE ^ (1 - EA_ETAJ) * (1 - EA_XIJ), (-EA_BETA * EA_FI * EA_XII * ➕₃ ^ (EA_ETAI * (EA_ZETA + 1)) + EA_FI) - EA_NDI ^ (EA_ZETA + 1) * EA_WI ^ (EA_ETAI * (EA_ZETA + 1)), US_LAMBDAJ * (US_GAMMAVJ + US_GAMMAVJDER * US_VJ + US_TAUC + 1) - 1 / ➕₃₈ ^ US_SIGMA, (-EA_CI * EA_GAMMAVI * (1 - EA_OMEGA) - EA_CJ * EA_GAMMAVJ * EA_OMEGA) + EA_GAMMAV, -(EA_KD ^ EA_ALPHA) * EA_ND ^ (1 - EA_ALPHA) * EA_Z + EA_PSIBAR + EA_YS, (US_M - US_MI * (1 - US_OMEGA)) - US_MJ * US_OMEGA, (-US_CI * (US_TAUC + 1)) / US_MI + US_VI, (-EA_CI * (EA_TAUC + 1)) / EA_MI + EA_VI, (-US_CI * US_GAMMAVI * (1 - US_OMEGA) - US_CJ * US_GAMMAVJ * US_OMEGA) + US_GAMMAV, (-EA_CJ * (EA_TAUC + 1)) / EA_MJ + EA_VJ, (US_GAMMAU - (US_GAMMAU2 * (US_U - 1) ^ 2) / 2) - ((US_U - 1) * (-US_DELTA * US_PIBAR * US_TAUKBAR + US_QBAR * ((US_DELTA - 1) + 1 / US_BETA))) / (US_PIBAR * (1 - US_TAUKBAR)), (-EA_SH * EA_XIH * ➕₃₁ ^ EA_THETA + EA_SH) - (1 - EA_XIH) / ➕₃₀ ^ EA_THETA, (-US_BETA * US_GJ * US_XIJ * ➕₃₉ ^ (US_ETAJ - 1) + US_GJ) - US_LAMBDAJ * US_NDJ * US_WJ ^ US_ETAJ * ((-US_TAUN - US_TAUWH) + 1), (-EA_BETA * EA_FH * EA_XIH * ➕₁₃ ^ EA_THETA + EA_FH) - EA_H * EA_MC, (-EA_ETAI * EA_FI) / (EA_GI * (EA_ETAI - 1)) + EA_WITILDE ^ (EA_ETAI * EA_ZETA + 1), (-EA_SJ * EA_XIJ * ➕₂₉ ^ EA_ETAJ + EA_SJ) - (1 - EA_XIJ) / ➕₂₈ ^ EA_ETAJ, (EA_C - EA_CI * (1 - EA_OMEGA)) - EA_CJ * EA_OMEGA, (EA_TJ - EA_T / EA_OMEGA) + (EA_TI * (1 - EA_OMEGA)) / EA_OMEGA, US_LAMBDAI * (US_GAMMAVI + US_GAMMAVIDER * US_VI + US_TAUC + 1) - 1 / ➕₃₄ ^ US_SIGMA, (-EA_ND * EA_OMEGA) / ➕₁₁ ^ EA_ETA + EA_NDJ, -EA_PHITB * (EA_B / (EA_PYBAR * EA_YBAR) - EA_BYTARGET) + EA_T / (EA_PYBAR * EA_YBAR), ((-EA_IM * EA_SIZE * US_MC) / US_SIZE - US_BETA * US_FX * US_XIX * ➕₄₉ ^ US_THETA) + US_FX, (-EA_SX * EA_XIX * ➕₃₃ ^ EA_THETA + EA_SX) - (1 - EA_XIX) / ➕₃₂ ^ EA_THETA, (-US_ND * (1 - US_OMEGA)) / ➕₄₃ ^ US_ETA + US_NDI, ((-EA_G + EA_H) - EA_HC) - EA_HI, (EA_D + EA_KD * EA_RK + EA_ND * EA_W * (EA_TAUWF + 1)) - EA_PY * EA_Y, (-US_SJ * US_XIJ * ➕₆₂ ^ US_ETAJ + US_SJ) - (1 - US_XIJ) / ➕₆₁ ^ US_ETAJ, -US_GAMMAUDER * US_PI + US_RK, (US_IM - US_IMC) - US_IMI, -US_PHITB * (US_B / (US_PYBAR * US_YBAR) - US_BYTARGET) + US_T / (US_PYBAR * US_YBAR), EAUS_RER - EA_RER / US_RER, (-US_FH * US_THETA) / (US_GH * (US_THETA - 1)) + US_PHTILDE / US_PH, -EA_NDJ * EA_SJ + EA_NJ, (-US_ETAJ * US_FJ) / (US_GJ * (US_ETAJ - 1)) + US_WJTILDE ^ (US_ETAJ * US_ZETA + 1), (-EA_FH * EA_THETA) / (EA_GH * (EA_THETA - 1)) + EA_PHTILDE / EA_PH, US_G * US_PH - US_GY * US_PYBAR * US_YBAR, ((((US_CJ * (US_GAMMAVJ + US_TAUC + 1) + US_MJ) - US_MJ / US_PIC) - US_NJ * US_WJ * ((-US_TAUN - US_TAUWH) + 1)) + US_TJ) - US_TRJ, US_I - US_II * (1 - US_OMEGA), (-EA_NUI * EA_PH ^ (1 - EA_MUI) + EA_PI ^ (1 - EA_MUI)) - ➕₂₄ ^ (1 - EA_MUI) * (1 - EA_NUI), USEA_RER - US_RER / EA_RER, (-EA_C - EA_GAMMAV) + EA_QC, (-EA_BETA * EA_GH * EA_XIH * ➕₁₃ ^ (EA_THETA - 1) + EA_GH) - EA_H * EA_PH, (-(US_HI ^ (1 - 1 / US_MUI)) * US_NUI ^ (1 / US_MUI) + US_QI ^ ((US_MUI - 1) / US_MUI)) - ➕₅₅ ^ (1 / US_MUI) * ➕₅₆ ^ (1 - 1 / US_MUI), (-US_ETAI * US_FI) / (US_GI * (US_ETAI - 1)) + US_WITILDE ^ (US_ETAI * US_ZETA + 1), (-EA_ALPHA * EA_MC * (EA_PSIBAR + EA_YS)) / EA_KD + EA_RK, (-US_SX * US_XIX * ➕₆₆ ^ US_THETA + US_SX) - (1 - US_XIX) / ➕₆₅ ^ US_THETA, (-EA_BETA * EA_GI * EA_XII * ➕₃ ^ (EA_ETAI - 1) + EA_GI) - EA_LAMBDAI * EA_NDI * EA_WI ^ EA_ETAI * ((-EA_TAUN - EA_TAUWH) + 1), ((-US_G + US_H) - US_HC) - US_HI, ((-EA_IM * EA_PIM * EA_SIZE * USEA_RER) / US_SIZE - US_BETA * US_GX * US_XIX * ➕₄₉ ^ (US_THETA - 1)) + US_GX, US_PI - US_Q, (-US_FX * US_THETA) / (US_GX * (US_THETA - 1)) + EA_PIMTILDE / EA_PIM, (-EA_ND * (1 - EA_OMEGA)) / ➕₁₀ ^ EA_ETA + EA_NDI, (-US_ND * US_OMEGA) / ➕₄₄ ^ US_ETA + US_NDJ, US_K * US_U - US_KD, (-(US_WJ ^ (1 - US_ETAJ)) * US_XIJ * ➕₄₀ ^ (1 - US_ETAJ) + US_WJ ^ (1 - US_ETAJ)) - US_WJTILDE ^ (1 - US_ETAJ) * (1 - US_XIJ), -US_BETA * (US_DELTA * US_PI * US_TAUK + US_Q * (1 - US_DELTA) + (1 - US_TAUK) * (-US_GAMMAU * US_PI + US_RK * US_U)) + US_Q, (-US_BETA * US_GH * US_XIH * ➕₄₆ ^ (US_THETA - 1) + US_GH) - US_H * US_PH, US_MC - (US_RK ^ US_ALPHA * ➕₄₁ ^ (US_ALPHA - 1) * ➕₄₂ ^ (1 - US_ALPHA)) / (US_ALPHA ^ US_ALPHA * US_Z), (-(EA_HI ^ (1 - 1 / EA_MUI)) * EA_NUI ^ (1 / EA_MUI) + EA_QI ^ ((EA_MUI - 1) / EA_MUI)) - ➕₂₂ ^ (1 / EA_MUI) * ➕₂₃ ^ (1 - 1 / EA_MUI), (EA_PIM ^ (1 - US_THETA) - EA_PIMTILDE ^ (1 - US_THETA) * (1 - US_XIX)) - US_XIX * ➕₅₀ ^ (1 - US_THETA) * ➕₅₁ ^ (1 - US_THETA), ((-EAUS_RER * US_IM * US_PIM * US_SIZE) / EA_SIZE - EA_BETA * EA_GX * EA_XIX * ➕₁₆ ^ (EA_THETA - 1)) + EA_GX, (EA_PH ^ (1 - EA_THETA) - EA_PHTILDE ^ (1 - EA_THETA) * (1 - EA_XIH)) - EA_XIH * ➕₁₄ ^ (1 - EA_THETA) * ➕₁₅ ^ (1 - EA_THETA), (-EA_NUC * EA_PH ^ (1 - EA_MUC) - ➕₂₁ ^ (1 - EA_MUC) * (1 - EA_NUC)) + 1, EA_I - EA_II * (1 - EA_OMEGA), (US_D + US_KD * US_RK + US_ND * US_W * (US_TAUWF + 1)) - US_PY * US_Y, (EA_IM - EA_IMC) - EA_IMI, (-US_GAMMAU * US_K - US_I) + US_QI, (-US_SH * US_XIH * ➕₆₄ ^ US_THETA + US_SH) - (1 - US_XIH) / ➕₆₃ ^ US_THETA, -(US_KD ^ US_ALPHA) * US_ND ^ (1 - US_ALPHA) * US_Z + US_PSIBAR + US_YS, EA_HC - (EA_NUC * EA_QC) / EA_PH ^ EA_MUC, EA_HI - (EA_NUI * EA_QI) / ➕₂₅ ^ EA_MUI, (-US_NUC * US_PH ^ (1 - US_MUC) - ➕₅₄ ^ (1 - US_MUC) * (1 - US_NUC)) + 1, ((((-EAUS_RER * US_IM * US_PIM * US_SIZE) / EA_SIZE - EA_G * EA_PH) - EA_PI * EA_QI) + EA_PIM * (EA_IMC + EA_IMI) + EA_PY * EA_Y) - EA_QC, EA_MC - (EA_RK ^ EA_ALPHA * ➕₈ ^ (EA_ALPHA - 1) * ➕₉ ^ (1 - EA_ALPHA)) / (EA_ALPHA ^ EA_ALPHA * EA_Z), ((((EA_CJ * (EA_GAMMAVJ + EA_TAUC + 1) + EA_MJ) - EA_MJ / EA_PIC) - EA_NJ * EA_WJ * ((-EA_TAUN - EA_TAUWH) + 1)) + EA_TJ) - EA_TRJ, (-(US_WI ^ (1 - US_ETAI)) * US_XII * ➕₃₇ ^ (1 - US_ETAI) + US_WI ^ (1 - US_ETAI)) - US_WITILDE ^ (1 - US_ETAI) * (1 - US_XII), ((((-EA_IM * EA_PIM * EA_SIZE * USEA_RER) / US_SIZE - US_G * US_PH) - US_PI * US_QI) + US_PIM * (US_IMC + US_IMI) + US_PY * US_Y) - US_QC, (-US_BETA * US_FI * US_XII * ➕₃₆ ^ (US_ETAI * (US_ZETA + 1)) + US_FI) - US_NDI ^ (US_ZETA + 1) * US_WI ^ (US_ETAI * (US_ZETA + 1)), (-US_ALPHA * US_MC * (US_PSIBAR + US_YS)) / US_KD + US_RK, (-(US_HC ^ (1 - 1 / US_MUC)) * US_NUC ^ (1 / US_MUC) + US_QC ^ ((US_MUC - 1) / US_MUC)) - ➕₅₂ ^ (1 / US_MUC) * ➕₅₃ ^ (1 - 1 / US_MUC), US_Y - US_YS, -US_NDJ * US_SJ + US_NJ, (-EA_II - EA_KI * (1 - EA_DELTA)) + EA_KI, -EA_T * EA_UPSILONT + EA_TI, (-EA_BF + EA_BF / US_R) - EA_TB / EA_RER, (-EA_BETA * EA_GJ * EA_XIJ * ➕₆ ^ (EA_ETAJ - 1) + EA_GJ) - EA_LAMBDAJ * EA_NDJ * EA_WJ ^ EA_ETAJ * ((-EA_TAUN - EA_TAUWH) + 1), EA_K - EA_KI * (1 - EA_OMEGA), (-EA_BETA * EA_FX * EA_XIX * ➕₁₆ ^ EA_THETA + EA_FX) - (EA_MC * US_IM * US_SIZE) / EA_SIZE, EA_Y - EA_YS, (-US_BETA * US_FJ * US_XIJ * ➕₃₉ ^ (US_ETAJ * (US_ZETA + 1)) + US_FJ) - US_NDJ ^ (US_ZETA + 1) * US_WJ ^ (US_ETAJ * (US_ZETA + 1)), (US_ND ^ (1 - 1 / US_ETA) - US_NDI ^ (1 - 1 / US_ETA) * ➕₄₅ ^ (1 / US_ETA)) - US_NDJ ^ (1 - 1 / US_ETA) * US_OMEGA ^ (1 / US_ETA), (-EA_ETAJ * EA_FJ) / (EA_GJ * (EA_ETAJ - 1)) + EA_WJTILDE ^ (EA_ETAJ * EA_ZETA + 1), (-(EA_WI ^ (1 - EA_ETAI)) * EA_XII * ➕₄ ^ (1 - EA_ETAI) + EA_WI ^ (1 - EA_ETAI)) - EA_WITILDE ^ (1 - EA_ETAI) * (1 - EA_XII), (EA_ND ^ (1 - 1 / EA_ETA) - EA_NDI ^ (1 - 1 / EA_ETA) * ➕₁₂ ^ (1 / EA_ETA)) - EA_NDJ ^ (1 - 1 / EA_ETA) * EA_OMEGA ^ (1 / EA_ETA), (-US_II - US_KI * (1 - US_DELTA)) + US_KI, (-EA_GAMMAU * EA_K - EA_I) + EA_QI, (EA_GAMMAU - (EA_GAMMAU2 * (EA_U - 1) ^ 2) / 2) - ((EA_U - 1) * (-EA_DELTA * EA_PIBAR * EA_TAUKBAR + EA_QBAR * ((EA_DELTA - 1) + 1 / EA_BETA))) / (EA_PIBAR * (1 - EA_TAUKBAR)), US_HC - (US_NUC * US_QC) / US_PH ^ US_MUC, (-EA_H * EA_SH + EA_YS) - (EA_SX * US_IM * US_SIZE) / EA_SIZE, US_K - US_KI * (1 - US_OMEGA), (US_TJ - US_T / US_OMEGA) + (US_TI * (1 - US_OMEGA)) / US_OMEGA, -EA_GAMMAUDER * EA_PI + EA_RK, (EA_IM * EA_PIM - (EA_RER * US_IM * US_PIM * US_SIZE) / EA_SIZE) + EA_TB, (-EA_FX * EA_THETA) / (EA_GX * (EA_THETA - 1)) + US_PIMTILDE / US_PIM, (-US_BETA * US_FH * US_XIH * ➕₄₆ ^ US_THETA + US_FH) - US_H * US_MC, (-US_C - US_GAMMAV) + US_QC, (EA_M - EA_MI * (1 - EA_OMEGA)) - EA_MJ * EA_OMEGA, (-EA_GAMMAU2 * (EA_U - 1) + EA_GAMMAUDER) - (-EA_DELTA * EA_PIBAR * EA_TAUKBAR + EA_QBAR * ((EA_DELTA - 1) + 1 / EA_BETA)) / (EA_PIBAR * (1 - EA_TAUKBAR)), EA_LAMBDAI * (EA_GAMMAVI + EA_GAMMAVIDER * EA_VI + EA_TAUC + 1) - 1 / ➕₁ ^ EA_SIGMA, EA_LAMBDAJ * (EA_GAMMAVJ + EA_GAMMAVJDER * EA_VJ + EA_TAUC + 1) - 1 / ➕₅ ^ EA_SIGMA, (US_C - US_CI * (1 - US_OMEGA)) - US_CJ * US_OMEGA, -US_T * US_UPSILONT + US_TI, EA_K * EA_U - EA_KD, ➕₂₈ - EA_WJTILDE / EA_WJ, ➕₅₀ - EA_PIM / EA_PIC, ➕₃₄ - (-US_CI * US_KAPPA + US_CI), ➕₂₅ - EA_PH / EA_PI, ➕₆₅ - EA_PIMTILDE / EA_PIM, ➕₁ - (-EA_CI * EA_KAPPA + EA_CI), ➕₄₃ - US_WI / US_W, ➕₂₄ - EA_PIM / EA_GAMMAIMIDAG, ➕₃₀ - EA_PHTILDE / EA_PH, ➕₁₄ - EA_PH / EA_PIC, ➕₄₇ - US_PH / US_PIC, ➕₁₀ - EA_WI / EA_W, ➕₅₆ - US_IMI * (1 - US_GAMMAIMI), ➕₅₃ - US_IMC * (1 - US_GAMMAIMC), ➕₁₁ - EA_WJ / EA_W, ➕₅₈ - US_PH / US_PI, ➕₃₂ - US_PIMTILDE / US_PIM, ➕₆₁ - US_WJTILDE / US_WJ, ➕₃₈ - (-US_CJ * US_KAPPA + US_CJ), ➕₅₄ - US_PIM / US_GAMMAIMCDAG, ➕₄₄ - US_WJ / US_W, ➕₁₇ - US_PIM / US_PIC, ➕₉ - EA_W * (EA_TAUWF + 1), ➕₂₁ - EA_PIM / EA_GAMMAIMCDAG, ➕₅ - (-EA_CJ * EA_KAPPA + EA_CJ), ➕₂₀ - EA_IMC * (1 - EA_GAMMAIMC), ➕₂₃ - EA_IMI * (1 - EA_GAMMAIMI), ➕₅₇ - US_PIM / US_GAMMAIMIDAG, ➕₄₂ - US_W * (US_TAUWF + 1), ➕₆₃ - US_PHTILDE / US_PH, ➕₆₇ - ((EA_BF * EA_RER) / (EA_PY * EA_Y * US_PIC) - EA_BFYTARGET)]
end)





params = [ 0.4194
0.25
0.992638
2.0
0.6
2.0
0.025
6.0
6.0
6.0
0.75
0.75
0.75
0.75
0.3
0.2
6.0
0.9
0.3
0.5
0.5
0.919622
1.5
0.418629
1.5
0.289073
0.150339
3.0
0.032765
0.007
2.5
0.0
0.01
2.4
0.1
0.18
0.195161
0.183
0.184123
0.122
0.118
0.219
1.2
0.6666666666666666
1.02
0.95
2.0
0.1
0.0
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
1.00645740523434
3.62698111871356
0.9
0.961117319822928
0.725396223742712
0.961117319822928
0
1
0.5806
0.25
0.992638
2.0
0.6
2.0
0.025
6.0
6.0
6.0
0.75
0.75
0.75
0.75
0.3
0.2
6.0
0.9
0.3
0.5
0.5
0.899734
1.5
0.673228
1.5
0.028706
0.150339
3.0
0.034697
0.007
2.5
0.0
0.01
2.4
0.1
0.16
0.079732
0.077
0.184123
0.154
0.071
0.071
1.2
0.6666666666666666
1.02
0.95
2.0
0.1
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.992282866960427
0
3.92445610588497
1.01776829477927
0.784891221176995
1.01776829477927
1
1]


EA_SIZE = params[1]
EA_OMEGA = params[2]
EA_BETA = params[3]
EA_SIGMA = params[4]
EA_KAPPA = params[5]
EA_ZETA = params[6]
EA_DELTA = params[7]
EA_ETA = params[8]
EA_ETAI = params[9]
EA_ETAJ = params[10]
EA_XII = params[11]
EA_XIJ = params[12]
EA_CHII = params[13]
EA_CHIJ = params[14]
EA_ALPHA = params[15]
EA_THETA = params[17]
EA_XIH = params[18]
EA_XIX = params[19]
EA_CHIH = params[20]
EA_CHIX = params[21]
EA_NUC = params[22]
EA_MUC = params[23]
EA_NUI = params[24]
EA_MUI = params[25]
EA_GAMMAV1 = params[26]
EA_GAMMAV2 = params[27]
EA_GAMMAU2 = params[30]
EA_GAMMAB1 = params[33]
EA_BYTARGET = params[34]
EA_PHITB = params[35]
EA_GYBAR = params[36]
EA_TRYBAR = params[37]
EA_TAUCBAR = params[38]
EA_TAUKBAR = params[39]
EA_TAUNBAR = params[40]
EA_TAUWHBAR = params[41]
EA_TAUWFBAR = params[42]
EA_UPSILONT = params[43]
EA_UPSILONTR = params[44]
EA_PI4TARGET = params[45]
EA_PHIRR = params[46]
EA_PHIRPI = params[47]
EA_PHIRGY = params[48]
EA_BFYTARGET = params[49]
EA_PYBAR = params[60]
EA_YBAR = params[61]
EA_PIBAR = params[63]
EA_PSIBAR = params[64]
EA_QBAR = params[65]
EA_TAUDBAR = params[66]
EA_ZBAR = params[67]
US_SIZE = params[68]
US_OMEGA = params[69]
US_BETA = params[70]
US_SIGMA = params[71]
US_KAPPA = params[72]
US_ZETA = params[73]
US_DELTA = params[74]
US_ETA = params[75]
US_ETAI = params[76]
US_ETAJ = params[77]
US_XII = params[78]
US_XIJ = params[79]
US_CHII = params[80]
US_CHIJ = params[81]
US_ALPHA = params[82]
US_THETA = params[84]
US_XIH = params[85]
US_XIX = params[86]
US_CHIH = params[87]
US_CHIX = params[88]
US_NUC = params[89]
US_MUC = params[90]
US_NUI = params[91]
US_MUI = params[92]
US_GAMMAV1 = params[93]
US_GAMMAV2 = params[94]
US_GAMMAU2 = params[97]
US_BYTARGET = params[101]
US_PHITB = params[102]
US_GYBAR = params[103]
US_TRYBAR = params[104]
US_TAUCBAR = params[105]
US_TAUKBAR = params[106]
US_TAUNBAR = params[107]
US_TAUWHBAR = params[108]
US_TAUWFBAR = params[109]
US_UPSILONT = params[110]
US_UPSILONTR = params[111]
US_PI4TARGET = params[112]
US_PHIRR = params[113]
US_PHIRPI = params[114]
US_PHIRGY = params[115]
US_PYBAR = params[126]
US_TAUDBAR = params[127]
US_YBAR = params[128]
US_PIBAR = params[129]
US_PSIBAR = params[130]
US_QBAR = params[131]
US_ZBAR = params[132]
US_RER = params[133]

#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:902 =#
EA_PI4TARGET = min(max(EA_PI4TARGET, 1.1920928955078125e-7), 1.000000000000723e12)
EA_ALPHA = min(max(EA_ALPHA, 1.1920928955078125e-7), 1.0000000000006951e12)
EA_OMEGA = min(max(EA_OMEGA, 1.1920928955078125e-7), 1.0000000000009675e12)
US_PI4TARGET = min(max(US_PI4TARGET, 1.1920928955078125e-7), 1.0000000000004036e12)
EA_NUC = min(max(EA_NUC, 1.1920928955078125e-7), 1.0000000000005804e12)
EA_NUI = min(max(EA_NUI, 1.1920928955078125e-7), 1.0000000000000125e12)
EA_ZBAR = min(max(EA_ZBAR, 1.1920928955078125e-7), 1.0000000000003737e12)
US_ALPHA = min(max(US_ALPHA, 1.1920928955078125e-7), 1.0000000000003065e12)
US_OMEGA = min(max(US_OMEGA, 1.1920928955078125e-7), 1.0000000000003402e12)
US_NUC = min(max(US_NUC, 1.1920928955078125e-7), 1.0000000000005594e12)
US_NUI = min(max(US_NUI, 1.1920928955078125e-7), 1.0000000000004137e12)
US_ZBAR = min(max(US_ZBAR, 1.1920928955078125e-7), 1.0000000000001163e12)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:903 =#
EA_RRSTAR = 1 / EA_BETA
US_RRSTAR = 1 / US_BETA
EA_interest_EXOG = EA_BETA ^ -1 * EA_PI4TARGET ^ (1 / 4)
US_interest_EXOG = US_BETA ^ -1 * US_PI4TARGET ^ (1 / 4)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:904 =#
NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:905 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:906 =#
US_TAUWH = US_TAUWHBAR
EA_GAMMAIMI = 0
EA_GAMMAIMIDAG = 1
EA_GAMMAIMC = 0
EA_GAMMAIMCDAG = 1
➕₂₂ = min(max(1.1920928955078125e-7, 1 - EA_NUI), 1.0000000000005713e12)
EA_TAUK = EA_TAUKBAR
EA_GAMMAIDER = 0
EA_GAMMAI = 0
lbs = [1.1920928955078125e-7, -9.999999999995553e11, -9.999999999999144e11]
ubs = [1.0000000000001346e12, 1.0000000000007758e12, 1.0000000000003999e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))
parameters_and_solved_vars = [US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR]

previous_sol_init = inits

sol_new = levenberg_marquardt(x->ss_solve_blocks[1](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)
                #   block_solver_RD = block_solver_AD([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR], 1, ss_solve_blocks[1], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR])
solution_error += sum(abs2, (ss_solve_blocks[1])([US_BETA, US_PI4TARGET, US_PHIRR, US_PHIRPI, US_PHIRGY, US_RRSTAR], solution, 0, lbs, ubs))
sol = solution
US_PIC = sol[1]
US_PIC4 = sol[2]
US_R = sol[3]

lbs = [1.1920928955078125e-7, -9.999999999996154e11, -9.999999999990419e11]
ubs = [1.0000000000006548e12, 1.0000000000003721e12, 1.0000000000004567e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

parameters_and_solved_vars = [EA_BETA, EA_PI4TARGET, EA_PHIRR, EA_PHIRPI, EA_PHIRGY, EA_RRSTAR]
previous_sol_init = inits

sol_new = levenberg_marquardt(x->ss_solve_blocks[2](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)

                #   block_solver_RD = block_solver_AD([EA_BETA, EA_PI4TARGET, EA_PHIRR, EA_PHIRPI, EA_PHIRGY, EA_RRSTAR], 2, ss_solve_blocks[2], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([EA_BETA, EA_PI4TARGET, EA_PHIRR, EA_PHIRPI, EA_PHIRGY, EA_RRSTAR])
solution_error += sum(abs2, (ss_solve_blocks[2])([EA_BETA, EA_PI4TARGET, EA_PHIRR, EA_PHIRPI, EA_PHIRGY, EA_RRSTAR], solution, 0, lbs, ubs))
sol = solution
EA_PIC = sol[1]
EA_PIC4 = sol[2]
EA_R = sol[3]

EA_PIIM = EA_PIC
➕₅₁ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (1 / 4 - US_CHIH / 4) * EA_PIIM ^ US_CHIX), 1.0000000000005076e12)
➕₆₆ = min(max(1.1920928955078125e-7, EA_PIIM ^ (1 - US_CHIX) * US_PI4TARGET ^ (US_CHIH / 4 - 1 / 4)), 1.0000000000000056e12)
US_GAMMAIMC = 0
US_GAMMAIMCDAG = 1
US_GAMMAIDER = 0
US_GAMMAI = 0
US_Z = US_ZBAR
➕₄₅ = min(max(1.1920928955078125e-7, 1 - US_OMEGA), 1.0000000000006392e12)
lbs = [-9.999999999996038e11, -9.99999999999996e11]
ubs = [1.000000000000112e12, 1.0000000000006305e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))
parameters_and_solved_vars = [US_BETA, US_GAMMAV1, US_GAMMAV2, US_PIC]

previous_sol_init = inits

sol_new = levenberg_marquardt(x->ss_solve_blocks[3](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)


                #   block_solver_RD = block_solver_AD([US_BETA, US_GAMMAV1, US_GAMMAV2, US_PIC], 3, ss_solve_blocks[3], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([US_BETA, US_GAMMAV1, US_GAMMAV2, US_PIC])
solution_error += sum(abs2, (ss_solve_blocks[3])([US_BETA, US_GAMMAV1, US_GAMMAV2, US_PIC], solution, 0, lbs, ubs))
sol = solution
US_GAMMAVJDER = sol[1]
US_VJ = sol[2]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
            sol
        else
            ℱ.value.(sol)
        end]
US_TAUC = US_TAUCBAR
➕₃₅ = min(max(1.1920928955078125e-7, US_GAMMAV1 * US_GAMMAV2), 1.0000000000002643e12)
US_GAMMAVJ = (US_GAMMAV1 * US_VJ + US_GAMMAV2 / US_VJ) - 2.0 * sqrt(➕₃₅)
➕₃₉ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (US_CHIJ / 4 - 1 / 4) * US_PIC ^ (1 - US_CHIJ)), 1.0000000000000668e12)
➕₄₀ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (1 / 4 - US_CHIJ / 4) * US_PIC ^ (US_CHIJ - 1)), 1.0000000000007162e12)
US_TAUN = US_TAUNBAR
US_TRY = US_TRYBAR
US_TR = US_PYBAR * US_TRY * US_YBAR
US_TRI = US_TR * US_UPSILONTR
US_TRJ = ((US_OMEGA * US_TRI + US_TR) - US_TRI) / US_OMEGA
➕₆₂ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (US_CHIJ / 4 - 1 / 4) * US_PIC ^ (1 - US_CHIJ)), 1.0000000000001808e12)
lbs = [-9.999999999994236e11, -9.999999999997894e11]
ubs = [1.0000000000008761e12, 1.0000000000006104e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))
parameters_and_solved_vars = [US_BETA, US_GAMMAV1, US_GAMMAV2, US_PIC]

previous_sol_init = inits

sol_new = levenberg_marquardt(x->ss_solve_blocks[4](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)

                #   block_solver_RD = block_solver_AD([US_BETA, US_GAMMAV1, US_GAMMAV2, US_PIC], 4, ss_solve_blocks[4], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([US_BETA, US_GAMMAV1, US_GAMMAV2, US_PIC])
solution_error += sum(abs2, (ss_solve_blocks[4])([US_BETA, US_GAMMAV1, US_GAMMAV2, US_PIC], solution, 0, lbs, ubs))
sol = solution
US_GAMMAVIDER = sol[1]
US_VI = sol[2]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
            sol
        else
            ℱ.value.(sol)
        end]
US_GAMMAVI = (US_GAMMAV1 * US_VI + US_GAMMAV2 / US_VI) - 2.0 * sqrt(➕₃₅)
➕₅₅ = min(max(1.1920928955078125e-7, 1 - US_NUI), 1.0000000000006082e12)
US_GAMMAIMI = 0
EA_PIH = EA_PIC
➕₃₁ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (EA_CHIH / 4 - 1 / 4) * EA_PIH ^ (1 - EA_CHIH)), 1.0000000000006049e12)
➕₁₅ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (1 / 4 - EA_CHIH / 4) * EA_PIH ^ EA_CHIH), 1.0000000000005957e12)
➕₁₃ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (EA_CHIH / 4 - 1 / 4) * EA_PIH ^ (1 - EA_CHIH)), 1.0000000000007091e12)
US_PIIM = US_PIC
➕₁₆ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (EA_CHIX / 4 - 1 / 4) * US_PIIM ^ (1 - EA_CHIX)), 1.0000000000009176e12)
➕₁₈ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (1 / 4 - EA_CHIH / 4) * US_PIIM ^ EA_CHIX), 1.0000000000000569e12)
➕₃₃ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (EA_CHIH / 4 - 1 / 4) * US_PIIM ^ (1 - EA_CHIX)), 1.0000000000009646e12)
➕₅₂ = min(max(1.1920928955078125e-7, 1 - US_NUC), 1.0000000000004541e12)
US_GY = US_GYBAR
US_TAUD = US_TAUDBAR
US_GAMMAIMIDAG = 1
US_TAUWF = US_TAUWFBAR
➕₃₇ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (1 / 4 - US_CHII / 4) * US_PIC ^ (US_CHII - 1)), 1.0000000000002687e12)
➕₃₆ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (US_CHII / 4 - 1 / 4) * US_PIC ^ (1 - US_CHII)), 1.0000000000005238e12)
US_TAUK = US_TAUKBAR
US_PIH = US_PIC
➕₄₈ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (1 / 4 - US_CHIH / 4) * US_PIH ^ US_CHIH), 1.0000000000008469e12)
➕₄₆ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (US_CHIH / 4 - 1 / 4) * US_PIH ^ (1 - US_CHIH)), 1.0000000000005497e12)
➕₄₁ = min(max(1.1920928955078125e-7, 1 - US_ALPHA), 1.0000000000004872e12)
➕₆₄ = min(max(1.1920928955078125e-7, US_PI4TARGET ^ (US_CHIH / 4 - 1 / 4) * US_PIH ^ (1 - US_CHIH)), 1.0000000000008112e12)
EA_RP = 0
EA_RERDEP = 1
EA_GAMMAB = 1 - US_PIC / (EA_BETA * US_R)
➕₆₇ = min(max(-9.9999999999974e11, log((EA_GAMMAB + EA_GAMMAB1) / EA_GAMMAB1)), 700.0)
➕₁₉ = min(max(1.1920928955078125e-7, 1 - EA_NUC), 1.0000000000005603e12)
➕₄₉ = min(max(1.1920928955078125e-7, EA_PIIM ^ (1 - US_CHIX) * US_PI4TARGET ^ (US_CHIX / 4 - 1 / 4)), 1.0000000000004805e12)
EA_GY = EA_GYBAR
EA_Z = EA_ZBAR
➕₈ = min(max(1.1920928955078125e-7, 1 - EA_ALPHA), 1.0000000000000879e12)
EA_TAUWF = EA_TAUWFBAR
➕₆ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (EA_CHIJ / 4 - 1 / 4) * EA_PIC ^ (1 - EA_CHIJ)), 1.0000000000005343e12)
EA_TAUWH = EA_TAUWHBAR
EA_TAUN = EA_TAUNBAR
➕₇ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (1 / 4 - EA_CHIJ / 4) * EA_PIC ^ (EA_CHIJ - 1)), 1.000000000000676e12)
lbs = [-9.999999999990764e11, -9.999999999992511e11]
ubs = [1.0000000000008254e12, 1.0000000000003884e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

parameters_and_solved_vars = [EA_BETA, EA_GAMMAV1, EA_GAMMAV2, EA_PIC]

previous_sol_init = inits

sol_new = levenberg_marquardt(x->ss_solve_blocks[5](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)


                #   block_solver_RD = block_solver_AD([EA_BETA, EA_GAMMAV1, EA_GAMMAV2, EA_PIC], 5, ss_solve_blocks[5], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
                #   solution = block_solver_RD([EA_BETA, EA_GAMMAV1, EA_GAMMAV2, EA_PIC])
solution_error += sum(abs2, (ss_solve_blocks[5])([EA_BETA, EA_GAMMAV1, EA_GAMMAV2, EA_PIC], solution, 0, lbs, ubs))
sol = solution
EA_GAMMAVJDER = sol[1]
EA_VJ = sol[2]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
            sol
        else
            ℱ.value.(sol)
        end]
➕₂ = min(max(1.1920928955078125e-7, EA_GAMMAV1 * EA_GAMMAV2), 1.0000000000005742e12)
EA_GAMMAVJ = (EA_GAMMAV1 * EA_VJ + EA_GAMMAV2 / EA_VJ) - 2.0 * sqrt(➕₂)
EA_TAUC = EA_TAUCBAR
lbs = [-9.999999999995089e11, -9.999999999995945e11]
ubs = [1.0000000000006375e12, 1.0000000000002753e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

parameters_and_solved_vars = [EA_BETA, EA_GAMMAV1, EA_GAMMAV2, EA_PIC]

previous_sol_init = inits

sol_new = levenberg_marquardt(x->ss_solve_blocks[6](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)


# block_solver_RD = block_solver_AD([EA_BETA, EA_GAMMAV1, EA_GAMMAV2, EA_PIC], 6, ss_solve_blocks[6], inits, lbs, ubs, fail_fast_solvers_only = fail_fast_solvers_only, verbose = verbose)
# solution = block_solver_RD([EA_BETA, EA_GAMMAV1, EA_GAMMAV2, EA_PIC])
solution_error += sum(abs2, (ss_solve_blocks[6])([EA_BETA, EA_GAMMAV1, EA_GAMMAV2, EA_PIC], solution, 0, lbs, ubs))
sol = solution
EA_GAMMAVIDER = sol[1]
EA_VI = sol[2]
NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
            sol
        else
            ℱ.value.(sol)
        end]
➕₁₂ = min(max(1.1920928955078125e-7, 1 - EA_OMEGA), 1.0000000000004594e12)
➕₃ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (EA_CHII / 4 - 1 / 4) * EA_PIC ^ (1 - EA_CHII)), 1.0000000000003591e12)
➕₄ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (1 / 4 - EA_CHII / 4) * EA_PIC ^ (EA_CHII - 1)), 1.0000000000002615e12)
EA_GAMMAVI = (EA_GAMMAV1 * EA_VI + EA_GAMMAV2 / EA_VI) - 2.0 * sqrt(➕₂)
➕₂₉ = min(max(1.1920928955078125e-7, EA_PI4TARGET ^ (EA_CHIJ / 4 - 1 / 4) * EA_PIC ^ (1 - EA_CHIJ)), 1.0000000000003284e12)
EA_TRY = EA_TRYBAR
EA_TR = EA_PYBAR * EA_TRY * EA_YBAR
EA_TRI = EA_TR * EA_UPSILONTR
EA_TRJ = ((EA_OMEGA * EA_TRI + EA_TR) - EA_TRI) / EA_OMEGA
EA_TAUD = EA_TAUDBAR

lbs = [-9.999999999992434e11, -9.999999999998973e11, -9.999999999992908e11, -9.99999999999766e11, -9.999999999997219e11, -9.999999999994574e11, -9.999999999993348e11, 20.0, 100.0, -9.999999999997137e11, -9.999999999998438e11, -9.99999999999494e11, -9.999999999992096e11, -9.9999999999944e11, -9.999999999992551e11, 20.0, -9.999999999990197e11, -9.999999999994269e11, -9.99999999999016e11, -9.999999999997462e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994875e11, -9.99999999999188e11, -9.999999999999135e11, -9.999999999994308e11, -9.999999999990367e11, 20.0, 20.0, 20.0, -9.999999999994858e11, -9.999999999993795e11, -9.999999999994908e11, -9.999999999997921e11, -9.999999999992015e11, -9.999999999999756e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999996508e11, -9.999999999993324e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999995457e11, 1.1920928955078125e-7, -9.999999999997122e11, -9.999999999990685e11, -9.99999999999497e11, -9.999999999999436e11, -9.999999999993546e11, -9.99999999999972e11, -9.999999999996284e11, -9.999999999994846e11, -9.99999999999161e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999996093e11, -9.99999999999547e11, -9.999999999992059e11, -9.99999999999126e11, -9.999999999994252e11, -9.999999999991068e11, -9.999999999995377e11, -9.999999999999745e11, 20.0, 100.0, 20.0, -9.999999999993511e11, -9.99999999999341e11, -9.999999999994292e11, -9.999999999996029e11, -9.999999999991389e11, 20.0, -9.99999999999839e11, -9.999999999997675e11, -9.999999999998387e11, -9.999999999994056e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994823e11, -9.999999999994779e11, -9.99999999999081e11, -9.999999999998727e11, -9.999999999995522e11, 20.0, 20.0, 20.0, -9.999999999998718e11, -9.999999999993461e11, -9.999999999992473e11, -9.999999999996089e11, -9.999999999993452e11, -9.999999999992495e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999992012e11, -9.999999999994204e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999995746e11, -9.999999999996866e11, -9.999999999995294e11, -9.999999999994865e11, -9.99999999999996e11, -9.999999999990101e11, -9.999999999994713e11, -9.999999999995946e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994125e11, -9.999999999992838e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
ubs = [1.0000000000006466e12, 1.0000000000001799e12, 1.0000000000008628e12, 1.0000000000003022e12, 1.000000000000415e12, 1.0000000000006346e12, 1.0000000000002433e12, 1.0000000000004271e12, 1.0000000000003107e12, 1.0000000000006022e12, 1.0000000000008429e12, 1.000000000000343e12, 1.0000000000009706e12, 1.0000000000002526e12, 1.0000000000004196e12, 1.0000000000004094e12, 1.0000000000003304e12, 1.0000000000005702e12, 1.0000000000004358e12, 1.0000000000005243e12, 1.0000000000008948e12, 1.0000000000002933e12, 1.0000000000000917e12, 1.0000000000000221e12, 1.0000000000000438e12, 1.0000000000002003e12, 1.0000000000007571e12, 1.0000000000001656e12, 1.0000000000005831e12, 1.0000000000009275e12, 1.0000000000009824e12, 1.000000000000705e12, 1.0000000000003829e12, 1.0000000000001826e12, 1.0000000000002479e12, 1.0000000000007766e12, 1.0000000000003995e12, 1.0000000000009668e12, 1.0000000000002118e12, 1.0000000000001075e12, 1.0000000000007489e12, 1.0000000000006339e12, 1.0000000000005109e12, 1.0000000000004648e12, 1.0000000000004583e12, 1.0000000000009219e12, 1.0000000000000668e12, 1.000000000000175e12, 1.0000000000007239e12, 1.0000000000003716e12, 1.0000000000008591e12, 1.000000000000525e12, 1.0000000000000428e12, 1.0000000000004991e12, 1.0000000000000692e12, 1.0000000000006803e12, 1.0000000000000836e12, 1.000000000000833e12, 1.0000000000008341e12, 1.0000000000007124e12, 1.0000000000001855e12, 1.0000000000008895e12, 1.0000000000006974e12, 1.0000000000002766e12, 1.0000000000004583e12, 1.0000000000004031e12, 1.0000000000004636e12, 1.0000000000009358e12, 1.0000000000005801e12, 1.0000000000007516e12, 1.0000000000000852e12, 1.0000000000000955e12, 1.0000000000001885e12, 1.0000000000009397e12, 1.0000000000002449e12, 1.0000000000000017e12, 1.0000000000006707e12, 1.0000000000009272e12, 1.0000000000008794e12, 1.000000000000489e12, 1.0000000000006207e12, 1.0000000000005729e12, 1.0000000000000215e12, 1.0000000000001346e12, 1.0000000000001606e12, 1.0000000000006796e12, 1.0000000000000281e12, 1.0000000000009838e12, 1.000000000000054e12, 1.0000000000000393e12, 1.0000000000006516e12, 1.0000000000002982e12, 1.0000000000009418e12, 1.0000000000000588e12, 1.0000000000004891e12, 1.000000000000335e12, 1.0000000000000173e12, 1.0000000000001462e12, 1.0000000000002697e12, 1.0000000000000483e12, 1.0000000000003591e12, 1.0000000000009575e12, 1.0000000000005046e12, 1.0000000000000469e12, 1.0000000000009431e12, 1.0000000000007882e12, 1.000000000000962e12, 1.000000000000865e12, 1.000000000000155e12, 1.000000000000038e12, 1.0000000000003802e12, 1.0000000000004053e12, 1.0000000000004329e12, 1.0000000000006348e12, 1.0000000000009412e12, 1.0000000000003375e12, 1.0000000000007988e12, 1.000000000000482e12, 1.000000000000407e12, 1.0000000000001139e12, 1.0000000000006315e12, 1.0000000000000573e12, 1.0000000000002518e12, 1.000000000000906e12, 1.0000000000007684e12, 1.0000000000003804e12, 1.0000000000002927e12, 1.0000000000008252e12, 1.0000000000000266e12, 1.0000000000005348e12, 1.0000000000007694e12, 1.0000000000001473e12, 1.0000000000006338e12, 1.0000000000005836e12, 1.0000000000007792e12, 1.0000000000005366e12, 1.0000000000002231e12, 1.000000000000623e12, 1.0000000000003163e12, 1.0000000000008419e12, 1.0000000000006847e12, 1.0000000000005804e12, 1.000000000000068e12, 1.0000000000005236e12, 1.0000000000009244e12, 1.0000000000009698e12, 1.0000000000009288e12, 1.0000000000005693e12, 1.0000000000006023e12, 1.0000000000002853e12, 1.0000000000009161e12, 1.0000000000008955e12, 1.0000000000006903e12, 1.000000000000273e12, 1.0000000000002571e12, 1.0000000000000481e12, 1.000000000000343e12, 1.0000000000002885e12, 1.000000000000658e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))

parameters_and_solved_vars = [EA_SIZE, EA_OMEGA, EA_BETA, EA_SIGMA, EA_KAPPA, EA_ZETA, EA_DELTA, EA_ETA, EA_ETAI, EA_ETAJ, EA_XII, EA_XIJ, EA_ALPHA, EA_THETA, EA_XIH, EA_XIX, EA_NUC, EA_MUC, EA_NUI, EA_MUI, EA_GAMMAU2, EA_BYTARGET, EA_PHITB, EA_TAUKBAR, EA_UPSILONT, EA_BFYTARGET, EA_PYBAR, EA_YBAR, EA_PIBAR, EA_PSIBAR, EA_QBAR, US_SIZE, US_OMEGA, US_BETA, US_SIGMA, US_KAPPA, US_ZETA, US_DELTA, US_ETA, US_ETAI, US_ETAJ, US_XII, US_XIJ, US_ALPHA, US_THETA, US_XIH, US_XIX, US_NUC, US_MUC, US_NUI, US_MUI, US_GAMMAU2, US_BYTARGET, US_PHITB, US_TAUKBAR, US_UPSILONT, US_PYBAR, US_YBAR, US_PIBAR, US_PSIBAR, US_QBAR, US_RER, EA_GAMMAIMC, EA_GAMMAIMCDAG, EA_GAMMAIMI, EA_GAMMAIMIDAG, EA_GAMMAVI, EA_GAMMAVIDER, EA_GAMMAVJ, EA_GAMMAVJDER, EA_GY, EA_PIC, EA_R, EA_TAUC, EA_TAUD, EA_TAUK, EA_TAUN, EA_TAUWF, EA_TAUWH, EA_TR, EA_TRJ, EA_VI, EA_VJ, EA_Z, US_GAMMAIMC, US_GAMMAIMCDAG, US_GAMMAIMI, US_GAMMAIMIDAG, US_GAMMAVI, US_GAMMAVIDER, US_GAMMAVJ, US_GAMMAVJDER, US_GY, US_PIC, US_R, US_TAUC, US_TAUD, US_TAUK, US_TAUN, US_TAUWF, US_TAUWH, US_TR, US_TRJ, US_VI, US_VJ, US_Z, ➕₃, ➕₄, ➕₆, ➕₇, ➕₈, ➕₁₂, ➕₁₃, ➕₁₅, ➕₁₆, ➕₁₈, ➕₁₉, ➕₂₂, ➕₂₉, ➕₃₁, ➕₃₃, ➕₃₆, ➕₃₇, ➕₃₉, ➕₄₀, ➕₄₁, ➕₄₅, ➕₄₆, ➕₄₈, ➕₄₉, ➕₅₁, ➕₅₂, ➕₅₅, ➕₆₂, ➕₆₄, ➕₆₆, ➕₆₇]

previous_sol_init = inits


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
    @inbounds for i in eachindex(x)
        x[i] = max(lb[i], min(x[i], ub[i]))
    end
end






# lbs = [-9.999999999992434e11, -9.999999999998973e11, -9.999999999992908e11, -9.99999999999766e11, -9.999999999997219e11, -9.999999999994574e11, -9.999999999993348e11, -9.999999999999745e11, -9.999999999999745e11, -9.999999999997137e11, -9.999999999998438e11, -9.99999999999494e11, -9.999999999992096e11, -9.9999999999944e11, -9.999999999992551e11, 20.0, -9.999999999990197e11, -9.999999999994269e11, -9.99999999999016e11, -9.999999999997462e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994875e11, -9.99999999999188e11, -9.999999999999135e11, -9.999999999994308e11, -9.999999999990367e11, 20.0, 20.0, 20.0, -9.999999999994858e11, -9.999999999993795e11, -9.999999999994908e11, -9.999999999997921e11, -9.999999999992015e11, -9.999999999999756e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999996508e11, -9.999999999993324e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999995457e11, 1.1920928955078125e-7, -9.999999999997122e11, -9.999999999990685e11, -9.99999999999497e11, -9.999999999999436e11, -9.999999999993546e11, -9.99999999999972e11, -9.999999999996284e11, -9.999999999994846e11, -9.99999999999161e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999996093e11, -9.99999999999547e11, -9.999999999992059e11, -9.99999999999126e11, -9.999999999994252e11, -9.999999999991068e11, -9.999999999995377e11, -9.999999999999745e11, 20.0, -9.999999999999745e11, 20.0, -9.999999999993511e11, -9.99999999999341e11, -9.999999999994292e11, -9.999999999996029e11, -9.999999999991389e11, 20.0, -9.99999999999839e11, -9.999999999997675e11, -9.999999999998387e11, -9.999999999994056e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994823e11, -9.999999999994779e11, -9.99999999999081e11, -9.999999999998727e11, -9.999999999995522e11, 20.0, 20.0, 20.0, -9.999999999998718e11, -9.999999999993461e11, -9.999999999992473e11, -9.999999999996089e11, -9.999999999993452e11, -9.999999999992495e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999992012e11, -9.999999999994204e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999995746e11, -9.999999999996866e11, -9.999999999995294e11, -9.999999999994865e11, -9.99999999999996e11, -9.999999999990101e11, -9.999999999994713e11, -9.999999999995946e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999994125e11, -9.999999999992838e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
# ubs = [1.0000000000006466e12, 1.0000000000001799e12, 1.0000000000008628e12, 1.0000000000003022e12, 1.000000000000415e12, 1.0000000000006346e12, 1.0000000000002433e12, 1.0000000000004271e12, 1.0000000000003107e12, 1.0000000000006022e12, 1.0000000000008429e12, 1.000000000000343e12, 1.0000000000009706e12, 1.0000000000002526e12, 1.0000000000004196e12, 1.0000000000004094e12, 1.0000000000003304e12, 1.0000000000005702e12, 1.0000000000004358e12, 1.0000000000005243e12, 1.0000000000008948e12, 1.0000000000002933e12, 1.0000000000000917e12, 1.0000000000000221e12, 1.0000000000000438e12, 1.0000000000002003e12, 1.0000000000007571e12, 1.0000000000001656e12, 1.0000000000005831e12, 1.0000000000009275e12, 1.0000000000009824e12, 1.000000000000705e12, 1.0000000000003829e12, 1.0000000000001826e12, 1.0000000000002479e12, 1.0000000000007766e12, 1.0000000000003995e12, 1.0000000000009668e12, 1.0000000000002118e12, 1.0000000000001075e12, 1.0000000000007489e12, 1.0000000000006339e12, 1.0000000000005109e12, 1.0000000000004648e12, 1.0000000000004583e12, 1.0000000000009219e12, 1.0000000000000668e12, 1.000000000000175e12, 1.0000000000007239e12, 1.0000000000003716e12, 1.0000000000008591e12, 1.000000000000525e12, 1.0000000000000428e12, 1.0000000000004991e12, 1.0000000000000692e12, 1.0000000000006803e12, 1.0000000000000836e12, 1.000000000000833e12, 1.0000000000008341e12, 1.0000000000007124e12, 1.0000000000001855e12, 1.0000000000008895e12, 1.0000000000006974e12, 1.0000000000002766e12, 1.0000000000004583e12, 1.0000000000004031e12, 1.0000000000004636e12, 1.0000000000009358e12, 1.0000000000005801e12, 1.0000000000007516e12, 1.0000000000000852e12, 1.0000000000000955e12, 1.0000000000001885e12, 1.0000000000009397e12, 1.0000000000002449e12, 1.0000000000000017e12, 1.0000000000006707e12, 1.0000000000009272e12, 1.0000000000008794e12, 1.000000000000489e12, 1.0000000000006207e12, 1.0000000000005729e12, 1.0000000000000215e12, 1.0000000000001346e12, 1.0000000000001606e12, 1.0000000000006796e12, 1.0000000000000281e12, 1.0000000000009838e12, 1.000000000000054e12, 1.0000000000000393e12, 1.0000000000006516e12, 1.0000000000002982e12, 1.0000000000009418e12, 1.0000000000000588e12, 1.0000000000004891e12, 1.000000000000335e12, 1.0000000000000173e12, 1.0000000000001462e12, 1.0000000000002697e12, 1.0000000000000483e12, 1.0000000000003591e12, 1.0000000000009575e12, 1.0000000000005046e12, 1.0000000000000469e12, 1.0000000000009431e12, 1.0000000000007882e12, 1.000000000000962e12, 1.000000000000865e12, 1.000000000000155e12, 1.000000000000038e12, 1.0000000000003802e12, 1.0000000000004053e12, 1.0000000000004329e12, 1.0000000000006348e12, 1.0000000000009412e12, 1.0000000000003375e12, 1.0000000000007988e12, 1.000000000000482e12, 1.000000000000407e12, 1.0000000000001139e12, 1.0000000000006315e12, 1.0000000000000573e12, 1.0000000000002518e12, 1.000000000000906e12, 1.0000000000007684e12, 1.0000000000003804e12, 1.0000000000002927e12, 1.0000000000008252e12, 1.0000000000000266e12, 1.0000000000005348e12, 1.0000000000007694e12, 1.0000000000001473e12, 1.0000000000006338e12, 1.0000000000005836e12, 1.0000000000007792e12, 1.0000000000005366e12, 1.0000000000002231e12, 1.000000000000623e12, 1.0000000000003163e12, 1.0000000000008419e12, 1.0000000000006847e12, 1.0000000000005804e12, 1.000000000000068e12, 1.0000000000005236e12, 1.0000000000009244e12, 1.0000000000009698e12, 1.0000000000009288e12, 1.0000000000005693e12, 1.0000000000006023e12, 1.0000000000002853e12, 1.0000000000009161e12, 1.0000000000008955e12, 1.0000000000006903e12, 1.000000000000273e12, 1.0000000000002571e12, 1.0000000000000481e12, 1.000000000000343e12, 1.0000000000002885e12, 1.000000000000658e12]
lbs = [-9.999999999991675e11, -9.999999999993151e11, -9.99999999999794e11, -9.999999999995188e11, -9.999999999996193e11, -9.99999999999576e11, -9.999999999992656e11, -9.999999999991495e11, -9.9999999999969e11, -9.999999999990642e11, -9.999999999996208e11, -9.999999999993491e11, -9.999999999995333e11, -9.999999999994442e11, -9.999999999994791e11, -9.999999999995477e11, -9.999999999990952e11, -9.999999999997767e11, -9.999999999996414e11, -9.999999999998275e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999990623e11, -9.999999999992137e11, -9.999999999996328e11, -9.999999999993987e11, -9.999999999999858e11, -9.999999999996173e11, 1.1920928955078125e-7, -9.999999999996832e11, -9.999999999990177e11, -9.999999999993583e11, -9.999999999998868e11, -9.999999999993961e11, -9.999999999999459e11, -9.999999999991407e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999999888e11, -9.999999999993003e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999990104e11, 1.1920928955078125e-7, -9.999999999994459e11, -9.99999999999065e11, -9.999999999991244e11, -9.999999999998722e11, -9.999999999997837e11, -9.999999999995964e11, -9.999999999991871e11, -9.999999999993824e11, -9.9999999999956e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999996936e11, -9.999999999994894e11, -9.999999999998044e11, -9.999999999995826e11, -9.99999999999162e11, -9.999999999991122e11, -9.999999999998221e11, -9.999999999999912e11, -9.999999999992771e11, -9.999999999996108e11, -9.99999999999764e11, -9.999999999997727e11, -9.999999999993313e11, -9.999999999990859e11, -9.999999999997048e11, -9.999999999994067e11, -9.999999999992821e11, -9.99999999999229e11, -9.999999999995188e11, -9.999999999991671e11, -9.99999999999604e11, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.99999999999238e11, -9.999999999994663e11, -9.999999999991552e11, -9.999999999991152e11, -9.999999999994614e11, -9.999999999995195e11, 1.1920928955078125e-7, -9.999999999992529e11, -9.999999999997014e11, -9.999999999990266e11, -9.999999999995443e11, -9.999999999999618e11, -9.999999999991464e11, -9.999999999992458e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999991044e11, -9.999999999997773e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999996566e11, -9.999999999999086e11, -9.999999999997627e11, -9.999999999999785e11, -9.999999999992761e11, -9.999999999997471e11, -9.999999999996912e11, -9.999999999990444e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, -9.999999999999197e11, -9.999999999994543e11, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7, 1.1920928955078125e-7]
ubs = [1.0000000000002256e12, 1.0000000000007655e12, 1.0000000000006888e12, 1.0000000000004225e12, 1.0000000000008085e12, 1.0000000000009457e12, 1.0000000000001338e12, 1.0000000000001289e12, 1.0000000000005166e12, 1.000000000000698e12, 1.0000000000009136e12, 1.0000000000009463e12, 1.000000000000823e12, 1.0000000000001427e12, 1.0000000000002933e12, 1.0000000000000548e12, 1.0000000000005068e12, 1.0000000000006989e12, 1.0000000000002219e12, 1.0000000000002935e12, 1.0000000000004312e12, 1.0000000000001277e12, 1.0000000000005071e12, 1.000000000000984e12, 1.0000000000009084e12, 1.0000000000000211e12, 1.0000000000005809e12, 1.0000000000008247e12, 1.0000000000004009e12, 1.0000000000008387e12, 1.0000000000001208e12, 1.000000000000695e12, 1.0000000000005365e12, 1.000000000000331e12, 1.000000000000933e12, 1.0000000000001266e12, 1.0000000000004985e12, 1.0000000000009509e12, 1.0000000000003026e12, 1.0000000000007415e12, 1.0000000000001667e12, 1.0000000000003767e12, 1.0000000000006222e12, 1.0000000000000004e12, 1.0000000000001616e12, 1.0000000000002041e12, 1.0000000000003461e12, 1.0000000000007223e12, 1.0000000000004011e12, 1.0000000000009613e12, 1.0000000000000627e12, 1.0000000000002721e12, 1.0000000000005239e12, 1.0000000000008849e12, 1.0000000000007006e12, 1.0000000000002106e12, 1.0000000000007965e12, 1.0000000000004995e12, 1.0000000000003071e12, 1.0000000000005492e12, 1.0000000000003132e12, 1.0000000000004603e12, 1.0000000000000956e12, 1.0000000000009186e12, 1.0000000000003708e12, 1.0000000000008861e12, 1.0000000000006646e12, 1.000000000000403e12, 1.0000000000003785e12, 1.0000000000005961e12, 1.0000000000001727e12, 1.0000000000009017e12, 1.0000000000006921e12, 1.0000000000004434e12, 1.0000000000004965e12, 1.0000000000009658e12, 1.0000000000007787e12, 1.0000000000009736e12, 1.0000000000005613e12, 1.0000000000003124e12, 1.0000000000003389e12, 1.0000000000004197e12, 1.0000000000003843e12, 1.0000000000002301e12, 1.0000000000000227e12, 1.0000000000007991e12, 1.0000000000003698e12, 1.0000000000007935e12, 1.0000000000008387e12, 1.0000000000003617e12, 1.0000000000007924e12, 1.0000000000007142e12, 1.0000000000008536e12, 1.0000000000005167e12, 1.0000000000003256e12, 1.0000000000002885e12, 1.0000000000006777e12, 1.0000000000007981e12, 1.0000000000009253e12, 1.0000000000007817e12, 1.0000000000004625e12, 1.000000000000134e12, 1.000000000000455e12, 1.0000000000008953e12, 1.0000000000008645e12, 1.0000000000009243e12, 1.0000000000008503e12, 1.0000000000004878e12, 1.0000000000008031e12, 1.0000000000000337e12, 1.0000000000005659e12, 1.0000000000007854e12, 1.000000000000109e12, 1.0000000000002245e12, 1.0000000000004321e12, 1.0000000000007397e12, 1.0000000000003533e12, 1.0000000000004644e12, 1.0000000000008707e12, 1.0000000000009493e12, 1.0000000000006072e12, 1.0000000000001941e12, 1.0000000000002389e12, 1.000000000000315e12, 1.0000000000006327e12, 1.0000000000004117e12, 1.0000000000001168e12, 1.0000000000001036e12, 1.0000000000003766e12, 1.000000000000633e12, 1.0000000000008148e12, 1.000000000000595e12, 1.0000000000003452e12, 1.0000000000003479e12, 1.0000000000009197e12, 1.0000000000006051e12, 1.0000000000003579e12, 1.0000000000000275e12, 1.0000000000000571e12, 1.0000000000009564e12, 1.0000000000009785e12, 1.0000000000003971e12, 1.0000000000000215e12, 1.0000000000003427e12, 1.0000000000005039e12, 1.0000000000006285e12, 1.0000000000008817e12, 1.0000000000001465e12, 1.0000000000009408e12, 1.0000000000002183e12, 1.0000000000004177e12, 1.0000000000007418e12, 1.0000000000009425e12, 1.0000000000005272e12, 1.0000000000009255e12, 1.0000000000001222e12, 1.0000000000003026e12, 1.0000000000009125e12, 1.0000000000007928e12]
inits = max.(lbs, min.(ubs, fill(.9,length(ubs))))


previous_sol_init = inits


# transformer_option = 1
# using BenchmarkTools
# @profview for i in 1:20 levenberg_marquardt(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option)) end#, steps = 4);

# sol_new = nlboxsolve(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option))
# sol_new


# λ²::T = .9, 
# p::T = 1.5, 
# λ¹::T = .9,  
# μ::T = .001
sol_new = levenberg_marquardt(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),
                                transformer(previous_sol_init,lbs,ubs, option = transformer_option),
                                transformer(lbs,lbs,ubs, option = transformer_option),
                                transformer(ubs,lbs,ubs, option = transformer_option),
                                # μ¹ = .0001, 
                                # μ² = 1e-5, 
                                # p = 1.5, 
                                # λ¹ = .7, 
                                # λ² = .4, 
                                # λᵖ = .5, 
                                # ρ = .1
                                )#, λ² = .9999, p = .25, λ¹ = .9,  μ = .0001, iterations = 100)#, p = 1.1)#, μ = 1e-4,λ = .6
# sum(abs2,sol_new[2][4])
sol_new[2][[1,3]]


# solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)
# solls = reduce(hcat,[undo_transformer(i,lbs,ubs, option = transformer_option) for i in sol_new[2][2]])
# solls_trans = reduce(hcat,sol_new[2][2])
# sol_new[1]
# sol_new[2][1]
# sol_new[2][3]
# sol_new[2][2]
# sol_new[2][4]
# sol_new[2][2][1]
# err = [sum(abs,ss_solve_blocks[7](parameters_and_solved_vars, i, transformer_option,lbs,ubs)) for i in sol_new[2][2]]
# normm = [ℒ.norm(ss_solve_blocks[7](parameters_and_solved_vars, i, transformer_option,lbs,ubs)) for i in sol_new[2][2]]

# sortperm(abs.(sol_new[1]))

# λ² = .9999, p = .25, λ¹ = .9,  μ = .0001
# go over parameters
parameter_find = []
itter = 1

for ρ in .1:.2:.5
    for λ¹ in .4:.1:.7
        for λ² in .4:.1:.7
            for λᵖ in .5:.2:.9
                for μ¹ in exp10.(-5:1:-4) 
                    for μ² in exp10.(-5:1:-4) 
                        for p in 1.3:.1:2.0
                            println(itter)
                            itter += 1
                            sol = levenberg_marquardt(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),
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



using DataFrames, GLM

# simul = DataFrame(reduce(hcat,parameter_find)',[:iter,:tol,:ρ, :μ, :σ¹, :σ², :r, :ϵ, :steps, :γ])
# simul = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ, :σ², :r, :steps, :γ])
simulNAWM = DataFrame(reduce(hcat, parameter_find)', [:iter, :tol, :μ¹, :μ², :p, :λ¹, :λ², :λᵖ, :ρ])
sort!(simulNAWM, [:μ¹, :μ², :p, :λ¹, :λ², :λᵖ, :ρ])
simulNAWM[simulNAWM.tol .> 1e-8,:iter] .= 200.0

simulNAWMsub = simulNAWM[simulNAWM.tol .< 1e-8,:]

results = innerjoin(simulNAWMsub, result, makeunique = true, on=["μ¹", "μ²", "p", "λ¹", "λ²", "λᵖ", "ρ"])
sort!(results, [:μ¹, :μ², :ρ, :p, :λ¹, :λ², :λᵖ])

# [p for ρ in .1:.2:.5
#     for λ¹ in .4:.1:.7
#         for λ² in .4:.1:.7
#             for λᵖ in .5:.2:.9
#                 for μ¹ in exp10.(-5:1:-4) 
#                     for μ² in exp10.(-5:1:-4) 
#                         for p in 1.3:.1:2.0]

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



sol_new = levenberg_marquardt(x->ss_solve_blocks[7](parameters_and_solved_vars, x, transformer_option,lbs,ubs),transformer(previous_sol_init,lbs,ubs, option = transformer_option),transformer(lbs,lbs,ubs, option = transformer_option),transformer(ubs,lbs,ubs, option = transformer_option), iterations = 1000)
solution = undo_transformer(sol_new[1],lbs,ubs, option = transformer_option)



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