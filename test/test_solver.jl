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
                        :ρ¹ => x[3], 
                        :p =>  x[4], 
                        :λ¹ => x[5], 
                        :λ² => x[6], 
                        :λᵖ => x[7], 
                        :μ¹ => x[8], 
                        :μ² => x[9])
# println(par_inputs)
    outAscari_Sbordone_2014 = try Ascari_Sbordone_2014.SS_solve_func(Ascari_Sbordone_2014.parameter_values, Ascari_Sbordone_2014, false, false, par_inputs, x[10]) catch end
    
    total_iters += outAscari_Sbordone_2014 isa Tuple{Vector{Float64}, Float64, Int64} ? outAscari_Sbordone_2014[2] > eps(Float64) ? 1000000 : outAscari_Sbordone_2014[3] : 1000000

    outSW03 = try SW03.SS_solve_func(SW03.parameter_values, SW03, false, false, par_inputs, x[10]) catch end
    
    total_iters += outSW03 isa Tuple{Vector{Float64}, Float64, Int64} ? outSW03[2] > eps(Float64) ? 1000000 : outSW03[3] : 1000000

    outNAWM_EAUS_2008 = try NAWM_EAUS_2008.SS_solve_func(NAWM_EAUS_2008.parameter_values, NAWM_EAUS_2008, false, false, par_inputs, x[10]) catch end

    total_iters += outNAWM_EAUS_2008 isa Tuple{Vector{Float64}, Float64, Int64} ? outNAWM_EAUS_2008[2] > eps(Float64) ? 1000000 : outNAWM_EAUS_2008[3] : 1000000

    outGali_Monacelli_2005_CITR = try Gali_Monacelli_2005_CITR.SS_solve_func(Gali_Monacelli_2005_CITR.parameter_values, Gali_Monacelli_2005_CITR, false, false, par_inputs, x[10]) catch end
    
    total_iters += outGali_Monacelli_2005_CITR isa Tuple{Vector{Float64}, Float64, Int64} ? outGali_Monacelli_2005_CITR[2] > eps(Float64) ? 1000000 : outGali_Monacelli_2005_CITR[3] : 1000000
    
    outGali_2015_chapter_3_nonlinear = try Gali_2015_chapter_3_nonlinear.SS_solve_func(Gali_2015_chapter_3_nonlinear.parameter_values, Gali_2015_chapter_3_nonlinear, false, false, par_inputs, x[10]) catch end
    
    total_iters += outGali_2015_chapter_3_nonlinear isa Tuple{Vector{Float64}, Float64, Int64} ? outGali_2015_chapter_3_nonlinear[2] > eps(Float64) ? 1000000 : outGali_2015_chapter_3_nonlinear[3] : 1000000

    outAguiar_Gopinath_2007 = try Aguiar_Gopinath_2007.SS_solve_func(Aguiar_Gopinath_2007.parameter_values, Aguiar_Gopinath_2007, false, false, par_inputs, x[10]) catch end
    
    total_iters += outAguiar_Gopinath_2007 isa Tuple{Vector{Float64}, Float64, Int64} ? outAguiar_Gopinath_2007[2] > eps(Float64) ? 1000000 : outAguiar_Gopinath_2007[3] : 1000000

    outCaldara_et_al_2012 = try Caldara_et_al_2012.SS_solve_func(Caldara_et_al_2012.parameter_values, Caldara_et_al_2012, false, false, par_inputs, x[10]) catch end
    
    total_iters += outCaldara_et_al_2012 isa Tuple{Vector{Float64}, Float64, Int64} ? outCaldara_et_al_2012[2] > eps(Float64) ? 1000000 : outCaldara_et_al_2012[3] : 1000000

    outGhironi_Melitz_2005 = try Ghironi_Melitz_2005.SS_solve_func(Ghironi_Melitz_2005.parameter_values, Ghironi_Melitz_2005, false, false, par_inputs, x[10]) catch end
    
    total_iters += outGhironi_Melitz_2005 isa Tuple{Vector{Float64}, Float64, Int64} ? outGhironi_Melitz_2005[2] > eps(Float64) ? 1000000 : outGhironi_Melitz_2005[3] : 1000000

    outGNSS_2010 = try GNSS_2010.SS_solve_func(GNSS_2010.parameter_values, GNSS_2010, false, false, par_inputs, x[10]) catch end

    total_iters += outGNSS_2010 isa Tuple{Vector{Float64}, Float64, Int64} ? outGNSS_2010[2] > eps(Float64) ? 1000000 : outGNSS_2010[3] : 1000000

    outSGU_2003_debt_premium = try SGU_2003_debt_premium.SS_solve_func(SGU_2003_debt_premium.parameter_values, SGU_2003_debt_premium, false, false, par_inputs, x[10]) catch end

    total_iters += outSGU_2003_debt_premium isa Tuple{Vector{Float64}, Float64, Int64} ? outSGU_2003_debt_premium[2] > eps(Float64) ? 1000000 : outSGU_2003_debt_premium[3] : 1000000

    outJQ_2012_RBC = try JQ_2012_RBC.SS_solve_func(JQ_2012_RBC.parameter_values, JQ_2012_RBC, false, false, par_inputs, x[10]) catch end

    total_iters += outJQ_2012_RBC isa Tuple{Vector{Float64}, Float64, Int64} ? outJQ_2012_RBC[2] > eps(Float64) ? 1000000 : outJQ_2012_RBC[3] : 1000000

    outIreland_2004 = try Ireland_2004.SS_solve_func(Ireland_2004.parameter_values, Ireland_2004, false, false, par_inputs, x[10]) catch end

    total_iters += outIreland_2004 isa Tuple{Vector{Float64}, Float64, Int64} ? outIreland_2004[2] > eps(Float64) ? 1000000 : outIreland_2004[3] : 1000000

    outFS2000 = try m.SS_solve_func(m.parameter_values, m, false, false, par_inputs, x[10]) catch end

    total_iters += outFS2000 isa Tuple{Vector{Float64}, Float64, Int64} ? outFS2000[2] > eps(Float64) ? 1000000 : outFS2000[3] : 1000000

    total_iters >= 1000000 ? NaN : total_iters
    println(total_iters)
    return Float64(total_iters)
end)

# innit = [.90768,.026714,2.312,.022,.0085,.0345,0.004821,0.5106,.8128]
# innit = [.9076,.026,.005,2.31,.022,.0085,.0345,0.0048,0.51,.81]

# innit = [.923466,.027651,0.005,2.22439,.02645,.008459,.041527,0.004649,0.5132,1.247]



# innit = [ 0.45567742151369633
# 0.08590458687304815
# 0.004361331164980951
# 1.9319469544673178
# 0.4764333429474666
# 0.008382020383011873
# 0.017055420068894822
# 0.7400358368511979
# 0.5742458069998582
# 1.783195678100702]


innit = [0.7601771571577032
0.0943561881214804
0.00023002168948737015
1.9319469544673178
0.01799771246781962
0.011659782477032993
0.5595714701091195
0.12338605153655616
0.5739263737494038
0.6764214269683286]



innit = [0.76015
0.09435
0.00023
1.932
0.018
0.0116
0.56
0.123
0.574
0.675]
lbs = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-30.0]
ubs = [1.0,1.0,1.0,20.0,1.0,1.0,1.0,1.0,1.0,30.0]
prob = OptimizationProblem(f, innit, [1], lb = lbs, ub = ubs)

f(innit,1)


maxt = 20 * 60


using OptimizationBBO
sol_BBO   =   solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(),   maxtime = maxt); sol_BBO.minimum #gets the far off parameters when only few models are involved
sol_BBO   =   solve(prob, BBO_dxnes(),   maxtime = maxt); sol_BBO.minimum
# sol_BBO   =   solve(prob, BBO_resampling_inheritance_memetic_search(),   maxtime = maxt); sol_BBO.minimum


sol_BOBYQA  =   solve(prob, NLopt.LN_BOBYQA()); sol_BOBYQA.minimum # fast and solves
sol_COBYLA  =   solve(prob, NLopt.LN_COBYLA()); sol_COBYLA.minimum # slow
sol_NM      =   solve(prob, NLopt.LN_NELDERMEAD()); sol_NM.minimum
sol_PRAXIS  =   solve(prob, NLopt.LN_PRAXIS()); sol_PRAXIS.minimum
sol_SBPLX   =   solve(prob, NLopt.LN_SBPLX()); sol_SBPLX.minimum


sol_AGS     =   solve(prob, NLopt.GN_AGS()); sol_AGS.minimum # slow and unreliable
sol_CRS2    =   solve(prob, NLopt.GN_CRS2_LM(), maxtime = maxt); sol_CRS2.minimum
sol_DIRECT  =   solve(prob, NLopt.GN_DIRECT_L_RAND(),  maxtime = maxt); sol_DIRECT.minimum
sol_ESCH    =   solve(prob, NLopt.GN_ESCH(),    maxtime = maxt); sol_ESCH.minimum
sol_ISRES   =   solve(prob, NLopt.GN_ISRES(),   maxtime = maxt); sol_ISRES.minimum
sol_Multi   =   solve(prob, NLopt.G_MLSL_LDS(), local_method =  NLopt.LN_BOBYQA(), maxtime = maxt); sol_Multi.minimum




using OptimizationMetaheuristics
sol_Metaheuristics   =   solve(prob, ECA(),   maxtime = maxt); sol_Metaheuristics.minimum


using OptimizationMultistartOptimization
sol_Multi = solve(prob, MultistartOptimization.TikTak(40), NLopt.LN_SBPLX()); sol_Multi.minimum

sol_Multi = solve(prob, NLopt.G_MLSL_LDS(), local_method =  NLopt.LN_SBPLX(), maxtime = maxt)





innit = [0.9085377918862518
0.03861742976500958
2.3154091936573424
0.0218875125940833
0.008232256097897564
0.03415564785252073
0.004838443809642127
0.5128636449212913
0.8128464356325741]


innit = [ 0.8393922298135471
0.4233763369516246
7.165513682560418
0.7314042269858947
0.014419253427264342
0.27053813459079223
0.3896905168010871
0.5201633597387922
0.7241371889045571]

innit = [ 0.9072851162787691
0.00769189510088665
2.312782406457346
0.02196011356205182
0.008505877227656958
0.034532455173425056
0.00483563092050365
0.5096198727629895
0.8114656217080681]

innit = xx
xx = innit
sqrt(170)
xx = sol_ESCH.u
xx = sol_PRAXIS.u

innit = [0.944993042979484
0.0032265667045325934
1.1413009165087014
0.34896994145261045
0.0957920419047839
0.15341302276628957
0.4443797645968635
0.7411904946987461
1.6576262439641378]


m.SS_solve_func(m.parameter_values, m, false, true, 
                    Dict(:r =>  xx[1], 
                    :ρ =>  xx[2], 
                    :ρ¹ => xx[3], 
                    :p =>  xx[4], 
                    :λ¹ => xx[5], 
                    :λ² => xx[6], 
                    :λᵖ => xx[7], 
                    :μ¹ => xx[8], 
                    :μ² => xx[9]),xx[10])

GNSS_2010.SS_solve_func(GNSS_2010.parameter_values, GNSS_2010, false, true, 
                    Dict(:r =>  xx[1], 
                    :ρ =>  xx[2], 
                    :ρ¹ => xx[3], 
                    :p =>  xx[4], 
                    :λ¹ => xx[5], 
                    :λ² => xx[6], 
                    :λᵖ => xx[7], 
                    :μ¹ => xx[8], 
                    :μ² => xx[9]),xx[10])


asdasd = Ascari_Sbordone_2014.SS_solve_func(Ascari_Sbordone_2014.parameter_values, Ascari_Sbordone_2014, false, true, 
                        Dict(:r =>  xx[1], 
                        :ρ =>  xx[2], 
                        :ρ¹ => xx[3], 
                        :p =>  xx[4], 
                        :λ¹ => xx[5], 
                        :λ² => xx[6], 
                        :λᵖ => xx[7], 
                        :μ¹ => xx[8], 
                        :μ² => xx[9]),xx[10])
   typeof(asdasd) 


NAWM_EAUS_2008.SS_solve_func(NAWM_EAUS_2008.parameter_values, NAWM_EAUS_2008, false, true, 
                        Dict(:r =>  xx[1], 
                        :ρ =>  xx[2], 
                        :ρ¹ => xx[3], 
                        :p =>  xx[4], 
                        :λ¹ => xx[5], 
                        :λ² => xx[6], 
                        :λᵖ => xx[7], 
                        :μ¹ => xx[8], 
                        :μ² => xx[9]),xx[10])
    
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
