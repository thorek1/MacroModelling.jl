using MacroModelling

include("models/SW03.jl")
include("models/GNSS_2010.jl")
include("models/Ghironi_Melitz_2005.jl")
include("models/SGU_2003_debt_premium.jl")
include("models/NAWM_EAUS_2008.jl") # stands out
include("models/JQ_2012_RBC.jl")
include("models/Ireland_2004.jl")
include("models/Caldara_et_al_2012.jl")
include("models/Gali_Monacelli_2005_CITR.jl")
include("models/Gali_2015_chapter_3_nonlinear.jl")
include("models/Aguiar_Gopinath_2007.jl")
include("models/Ascari_Sbordone_2014.jl")# stands out
include("models/FS2000.jl")


using Optimization, OptimizationNLopt
f = OptimizationFunction((x,u)-> begin 
    total_iters = 0
    par_inputs = Dict(  :r =>  x[1], 
                        :ρ =>  x[2], 
                        :p =>  x[3], 
                        :λ¹ => x[4], 
                        :λ² => x[5], 
                        :λᵖ => x[6], 
                        :μ¹ => x[7], 
                        :μ² => x[8])

    outGali_Monacelli_2005_CITR = Gali_Monacelli_2005_CITR.SS_solve_func(Gali_Monacelli_2005_CITR.parameter_values, Gali_Monacelli_2005_CITR, false, false, par_inputs, x[9])
    
    total_iters += outGali_Monacelli_2005_CITR[2] > 1e-8 ? 1000000 : outGali_Monacelli_2005_CITR[3]
    
    outGali_2015_chapter_3_nonlinear = Gali_2015_chapter_3_nonlinear.SS_solve_func(Gali_2015_chapter_3_nonlinear.parameter_values, Gali_2015_chapter_3_nonlinear, false, false, par_inputs, x[9])
    
    total_iters += outGali_2015_chapter_3_nonlinear[2] > 1e-8 ? 1000000 : outGali_2015_chapter_3_nonlinear[3]

    outAscari_Sbordone_2014 = Ascari_Sbordone_2014.SS_solve_func(Ascari_Sbordone_2014.parameter_values, Ascari_Sbordone_2014, false, false, par_inputs, x[9])
    
    total_iters += outAscari_Sbordone_2014[2] > 1e-8 ? 1000000 : outAscari_Sbordone_2014[3]

    outAguiar_Gopinath_2007 = Aguiar_Gopinath_2007.SS_solve_func(Aguiar_Gopinath_2007.parameter_values, Aguiar_Gopinath_2007, false, false, par_inputs, x[9])
    
    total_iters += outAguiar_Gopinath_2007[2] > 1e-8 ? 1000000 : outAguiar_Gopinath_2007[3]

    outSW03 = SW03.SS_solve_func(SW03.parameter_values, SW03, false, false, par_inputs, x[9])
    
    total_iters += outSW03[2] > 1e-8 ? 1000000 : outSW03[3]

    outCaldara_et_al_2012 = Caldara_et_al_2012.SS_solve_func(Caldara_et_al_2012.parameter_values, Caldara_et_al_2012, false, false, par_inputs, x[9])
    
    total_iters += outCaldara_et_al_2012[2] > 1e-8 ? 1000000 : outCaldara_et_al_2012[3]

    outGhironi_Melitz_2005 = Ghironi_Melitz_2005.SS_solve_func(Ghironi_Melitz_2005.parameter_values, Ghironi_Melitz_2005, false, false, par_inputs, x[9])
    
    total_iters += outGhironi_Melitz_2005[2] > 1e-8 ? 1000000 : outGhironi_Melitz_2005[3]

    outGNSS_2010 = GNSS_2010.SS_solve_func(GNSS_2010.parameter_values, GNSS_2010, false, false, par_inputs, x[9])

    total_iters += outGNSS_2010[2] > 1e-8 ? 1000000 : outGNSS_2010[3]

    outSGU_2003_debt_premium = SGU_2003_debt_premium.SS_solve_func(SGU_2003_debt_premium.parameter_values, SGU_2003_debt_premium, false, false, par_inputs, x[9])

    total_iters += outSGU_2003_debt_premium[2] > 1e-8 ? 1000000 : outSGU_2003_debt_premium[3]

    outNAWM_EAUS_2008 = NAWM_EAUS_2008.SS_solve_func(NAWM_EAUS_2008.parameter_values, NAWM_EAUS_2008, false, false, par_inputs, x[9])

    total_iters += outNAWM_EAUS_2008[2] > 1e-8 ? 1000000 : outNAWM_EAUS_2008[3]

    outJQ_2012_RBC = JQ_2012_RBC.SS_solve_func(JQ_2012_RBC.parameter_values, JQ_2012_RBC, false, false, par_inputs, x[9])

    total_iters += outJQ_2012_RBC[2] > 1e-8 ? 1000000 : outJQ_2012_RBC[3]

    outIreland_2004 = Ireland_2004.SS_solve_func(Ireland_2004.parameter_values, Ireland_2004, false, false, par_inputs, x[9])

    total_iters += outIreland_2004[2] > 1e-8 ? 1000000 : outIreland_2004[3]

    outFS2000 = m.SS_solve_func(m.parameter_values, m, false, false, par_inputs, x[9])

    total_iters += outFS2000[2] > 1e-8 ? 1000000 : outFS2000[3]

    total_iters >= 1000000 ? NaN : total_iters
    return Float64(total_iters)
end)

innit = [.90768,.026714,2.312,.022,.0085,.0345,0.004821,0.5106,.8128]
innit = [.9076,.026,2.31,.022,.0085,.0345,0.0048,0.51,.81]
innit = [.923466,.027651,2.22439,.02645,.008459,.041527,0.004649,0.5132,1.247]
lbs = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-10.0]
ubs = [1.0,1.0,10.0,1.0,1.0,1.0,1.0,1.0,10.0]
prob = OptimizationProblem(f, innit, [1], lb = lbs, ub = ubs)

f(innit,1)


maxt = 60

sol_BOBYQA  =   solve(prob, NLopt.LN_BOBYQA(),  maxtime = maxt); sol_BOBYQA.minimum # fast and solves
sol_COBYLA  =   solve(prob, NLopt.LN_COBYLA(),  maxtime = maxt); sol_COBYLA.minimum # slow
sol_NM      =   solve(prob, NLopt.LN_NELDERMEAD(),maxtime = maxt); sol_NM.minimum
sol_PRAXIS  =   solve(prob, NLopt.LN_PRAXIS(),  maxtime = maxt); sol_PRAXIS.minimum
sol_SBPLX   =   solve(prob, NLopt.LN_SBPLX(),   maxtime = maxt); sol_SBPLX.minimum

sol_AGS     =   solve(prob, NLopt.GN_AGS(),     maxtime = maxt); sol_AGS.minimum # slow and unreliable
sol_CRS2    =   solve(prob, NLopt.GN_CRS2_LM(), maxtime = maxt); sol_CRS2.minimum
sol_DIRECT  =   solve(prob, NLopt.GN_DIRECT_L_RAND(),  maxtime = maxt); sol_DIRECT.minimum
sol_ESCH    =   solve(prob, NLopt.GN_ESCH(),    maxtime = maxt); sol_ESCH.minimum
sol_ISRES   =   solve(prob, NLopt.GN_ISRES(),   maxtime = maxt); sol_ISRES.minimum



using OptimizationBBO

sol_BBO   =   solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(),   maxtime = maxt); sol_BBO.minimum

using OptimizationMetaheuristics
sol_Metaheuristics   =   solve(prob, WOA(),   maxtime = maxt); sol_Metaheuristics.minimum


using OptimizationMultistartOptimization
sol_Multi = solve(prob, MultistartOptimization.TikTak(20), NLopt.LN_SBPLX())

sol_Multi = solve(prob, NLopt.G_MLSL_LDS(), local_method =  NLopt.LN_SBPLX(), maxtime = maxt)

init = [0.9085377918862518
0.03861742976500958
2.3154091936573424
0.0218875125940833
0.008232256097897564
0.03415564785252073
0.004838443809642127
0.5128636449212913
0.8128464356325741]


innit = [ 0.9072851162787691
0.00769189510088665
2.312782406457346
0.02196011356205182
0.008505877227656958
0.034532455173425056
0.00483563092050365
0.5096198727629895
0.8114656217080681]

sqrt(170)
xx = sol_PRAXIS.u
xx = sol_BOBYQA.u
m.SS_solve_func(m.parameter_values, m, false, true, 
                    Dict(:r =>  xx[1], 
                        :ρ =>  xx[2], 
                        :p =>  xx[3], 
                        :λ¹ => xx[4], 
                        :λ² => xx[5], 
                        :λᵖ => xx[6], 
                        :μ¹ => xx[7], 
                        :μ² => xx[8]),xx[9])

GNSS_2010.SS_solve_func(GNSS_2010.parameter_values, GNSS_2010, false, true, 
                    Dict(:r =>  xx[1], 
                        :ρ =>  xx[2], 
                        :p =>  xx[3], 
                        :λ¹ => xx[4], 
                        :λ² => xx[5], 
                        :λᵖ => xx[6], 
                        :μ¹ => xx[7], 
                        :μ² => xx[8]),xx[9])


Ascari_Sbordone_2014.SS_solve_func(Ascari_Sbordone_2014.parameter_values, Ascari_Sbordone_2014, false, true, 
                        Dict(:r =>  xx[1], 
                            :ρ =>  xx[2], 
                            :p =>  xx[3], 
                            :λ¹ => xx[4], 
                            :λ² => xx[5], 
                            :λᵖ => xx[6], 
                            :μ¹ => xx[7], 
                            :μ² => xx[8]),xx[9])
    


NAWM_EAUS_2008.SS_solve_func(NAWM_EAUS_2008.parameter_values, NAWM_EAUS_2008, false, true, 
                        Dict(:r =>  xx[1], 
                            :ρ =>  xx[2], 
                            :p =>  xx[3], 
                            :λ¹ => xx[4], 
                            :λ² => xx[5], 
                            :λᵖ => xx[6], 
                            :μ¹ => xx[7], 
                            :μ² => xx[8]),xx[9])
    
prob = OptimizationProblem(f, 
transformer(previous_sol_init,lbs,ubs, option = transformer_option), 
(parameters_and_solved_vars,transformer_option, ss_solve_blocks, lbs, ubs), 
lb = transformer(lbs,lbs,ubs, option = transformer_option), 
ub = transformer(ubs,lbs,ubs, option = transformer_option))

    #             sol_new = solve(prob, SS_optimizer(), local_maxtime = timeout, maxtime = timeout)

    
m.SS_solve_func(m.parameter_values, m, false, false, Dict(), [.9])



using NLboxsolve
nlboxsolve



if norm(f(xn)) > γ*norm(f(xk))
    g .= jk'f(xk)
    if g'dk <= -ρ*norm(dk)^p
        α = 1.0
        while norm(f(xk+α*dk))^2 > norm(f(xk))^2 + 2*α*β*g'dk
            α = β*α
        end
        xn .= xk + α*dk
        box_projection!(xn,lb,ub)
    else
        α = 1.0
        while true
            xt .= xk-α*g
            box_projection!(xt,lb,ub)
            if norm(f(xt))^2 <= norm(f(xk))^2 + 2*σ*g'*(xt-xk)
                xn .=  xt
                break
            else
                α = β*α
            end
        end
    end
end



if sum(abs2,f(previous_guess + α * guess_update)) > ρ * ḡ 
    while sum(abs2,f(previous_guess + α * guess_update)) > ḡ  - 0.005 * α^2 * sum(abs2,guess_update)
        α *= r
    end
    μ¹ = μ¹ * λ¹ #max(μ¹ * λ¹, 1e-7)
    μ² = μ² * λ² #max(μ² * λ², 1e-7)
    p = λᵖ * p + (1 - λᵖ)
else
    μ¹ = min(μ¹ / λ¹, 1e-3)
    μ² = min(μ² / λ², 1e-3)
end




if norm(f(z)) <= ρ*norm(f(xk))
    α = 1.0
else
    if f(xk)'jk*dk > -γ
        dk = d1k
        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk
    end
    α = 1.0
    epsilon = 1/10
    while true
        if norm(f(xk+α*s))^2 > (1+epsilon)*norm(f(xk))^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(f(xk))^2
            α = r*α
            epsilon = r*epsilon
        else
            break
        end
    end
end
