using MacroModelling

include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")
include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
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
include("models/SW07.jl")


using Optimization, OptimizationNLopt
f = OptimizationFunction((x,verbose)-> begin 
    total_iters = 0

    x[1:2] = sort(x[1:2], rev = true)
    par_inputs = Dict(    
    :ϕ̄  =>  x[1],
    :ϕ̂  =>  x[2],
    :μ̄¹ =>  x[3],
    :μ̄² =>  x[4],
    :p̄¹ =>  x[5],
    :p̄² =>  x[6],
    :ρ  =>  x[7],
    :ρ¹ =>  x[8],
    :ρ² =>  x[9],
    :ρ³ =>  x[10],
    :ν  =>  x[11],
    :λ̅¹ =>  x[12],
    :λ̅² =>  x[13],
    :λ̂̅¹ =>  x[14],
    :λ̂̅² =>  x[15],
    :transformation_level   => Int(abs(round(x[16]))),
    :backtracking_order     => Int(abs(round(x[17])))
    )
# println(par_inputs)
    outSW07 = try SW07.SS_solve_func(SW07.parameter_values, SW07, false, verbose, par_inputs, x[end]) catch end
    
    total_iters += outSW07 isa Tuple{Vector{Float64}, Float64, Int64} ? (outSW07[2] > eps(Float64)) || !isfinite(outSW07[2]) ? 1000000 : outSW07[3] : 1000000

    outAscari_Sbordone_2014 = try Ascari_Sbordone_2014.SS_solve_func(Ascari_Sbordone_2014.parameter_values, Ascari_Sbordone_2014, false, verbose, par_inputs, x[end]) catch end
    # println(outAscari_Sbordone_2014[1][10])
    total_iters += outAscari_Sbordone_2014 isa Tuple{Vector{Float64}, Float64, Int64} ? (outAscari_Sbordone_2014[2] > eps(Float64)) || !isfinite(outAscari_Sbordone_2014[2]) ? 1000000 : outAscari_Sbordone_2014[3] : 1000000
    #  || !isapprox(outAscari_Sbordone_2014[1][10], 3.88351239274375, atol = 1e-6)

    outSW03 = try SW03.SS_solve_func(SW03.parameter_values, SW03, false, verbose, par_inputs, x[end]) catch end
    
    total_iters += outSW03 isa Tuple{Vector{Float64}, Float64, Int64} ? (outSW03[2] > eps(Float64)) || !isfinite(outSW03[2]) ? 1000000 : outSW03[3] : 1000000

    outNAWM_EAUS_2008 = try NAWM_EAUS_2008.SS_solve_func(NAWM_EAUS_2008.parameter_values, NAWM_EAUS_2008, false, verbose, par_inputs, x[end]) catch end

    total_iters += outNAWM_EAUS_2008 isa Tuple{Vector{Float64}, Float64, Int64} ? (outNAWM_EAUS_2008[2] > eps(Float64)) || !isfinite(outNAWM_EAUS_2008[2]) ? 1000000 : outNAWM_EAUS_2008[3] : 1000000

    # outGali_Monacelli_2005_CITR = try Gali_Monacelli_2005_CITR.SS_solve_func(Gali_Monacelli_2005_CITR.parameter_values, Gali_Monacelli_2005_CITR, false, verbose, par_inputs, x[end]) catch end
    
    # total_iters += outGali_Monacelli_2005_CITR isa Tuple{Vector{Float64}, Float64, Int64} ? (outGali_Monacelli_2005_CITR[2] > eps(Float64)) || !isfinite(outGali_Monacelli_2005_CITR[2]) ? 1000000 : outGali_Monacelli_2005_CITR[3] : 1000000
    
    # outGali_2015_chapter_3_nonlinear = try Gali_2015_chapter_3_nonlinear.SS_solve_func(Gali_2015_chapter_3_nonlinear.parameter_values, Gali_2015_chapter_3_nonlinear, false, verbose, par_inputs, x[end]) catch end
    
    # total_iters += outGali_2015_chapter_3_nonlinear isa Tuple{Vector{Float64}, Float64, Int64} ? (outGali_2015_chapter_3_nonlinear[2] > eps(Float64)) || !isfinite(outGali_2015_chapter_3_nonlinear[2]) ? 1000000 : outGali_2015_chapter_3_nonlinear[3] : 1000000

    # outAguiar_Gopinath_2007 = try Aguiar_Gopinath_2007.SS_solve_func(Aguiar_Gopinath_2007.parameter_values, Aguiar_Gopinath_2007, false, verbose, par_inputs, x[end]) catch end
    
    # total_iters += outAguiar_Gopinath_2007 isa Tuple{Vector{Float64}, Float64, Int64} ? (outAguiar_Gopinath_2007[2] > eps(Float64)) || !isfinite(outAguiar_Gopinath_2007[2]) ? 1000000 : outAguiar_Gopinath_2007[3] : 1000000

    # outCaldara_et_al_2012 = try Caldara_et_al_2012.SS_solve_func(Caldara_et_al_2012.parameter_values, Caldara_et_al_2012, false, verbose, par_inputs, x[end]) catch end
    
    # total_iters += outCaldara_et_al_2012 isa Tuple{Vector{Float64}, Float64, Int64} ? (outCaldara_et_al_2012[2] > eps(Float64)) || !isfinite(outCaldara_et_al_2012[2]) ? 1000000 : outCaldara_et_al_2012[3] : 1000000

    # outGhironi_Melitz_2005 = try Ghironi_Melitz_2005.SS_solve_func(Ghironi_Melitz_2005.parameter_values, Ghironi_Melitz_2005, false, verbose, par_inputs, x[end]) catch end
    
    # total_iters += outGhironi_Melitz_2005 isa Tuple{Vector{Float64}, Float64, Int64} ? (outGhironi_Melitz_2005[2] > eps(Float64)) || !isfinite(outGhironi_Melitz_2005[2]) ? 1000000 : outGhironi_Melitz_2005[3] : 1000000

    # outGNSS_2010 = try GNSS_2010.SS_solve_func(GNSS_2010.parameter_values, GNSS_2010, false, verbose, par_inputs, x[end]) catch end

    # total_iters += outGNSS_2010 isa Tuple{Vector{Float64}, Float64, Int64} ? (outGNSS_2010[2] > eps(Float64)) || !isfinite(outGNSS_2010[2]) ? 1000000 : outGNSS_2010[3] : 1000000

    # outSGU_2003_debt_premium = try SGU_2003_debt_premium.SS_solve_func(SGU_2003_debt_premium.parameter_values, SGU_2003_debt_premium, false, verbose, par_inputs, x[end]) catch end

    # total_iters += outSGU_2003_debt_premium isa Tuple{Vector{Float64}, Float64, Int64} ? (outSGU_2003_debt_premium[2] > eps(Float64)) || !isfinite(outSGU_2003_debt_premium[2]) ? 1000000 : outSGU_2003_debt_premium[3] : 1000000

    # outJQ_2012_RBC = try JQ_2012_RBC.SS_solve_func(JQ_2012_RBC.parameter_values, JQ_2012_RBC, false, verbose, par_inputs, x[end]) catch end

    # total_iters += outJQ_2012_RBC isa Tuple{Vector{Float64}, Float64, Int64} ? (outJQ_2012_RBC[2] > eps(Float64)) || !isfinite(outJQ_2012_RBC[2]) ? 1000000 : outJQ_2012_RBC[3] : 1000000

    # outIreland_2004 = try Ireland_2004.SS_solve_func(Ireland_2004.parameter_values, Ireland_2004, false, verbose, par_inputs, x[end]) catch end

    # total_iters += outIreland_2004 isa Tuple{Vector{Float64}, Float64, Int64} ? (outIreland_2004[2] > eps(Float64)) || !isfinite(outIreland_2004[2]) ? 1000000 : outIreland_2004[3] : 1000000

    # outFS2000 = try FS2000.SS_solve_func(FS2000.parameter_values, FS2000, false, verbose, par_inputs, x[end]) catch end

    # total_iters += outFS2000 isa Tuple{Vector{Float64}, Float64, Int64} ? (outFS2000[2] > eps(Float64)) || !isfinite(outFS2000[2]) ? 1000000 : outFS2000[3] : 1000000

    # outRBC_lead_lags = try RBC_lead_lags.SS_solve_func(RBC_lead_lags.parameter_values, RBC_lead_lags, false, verbose, par_inputs, x[end]) catch end

    # total_iters += outRBC_lead_lags isa Tuple{Vector{Float64}, Float64, Int64} ? (outRBC_lead_lags[2] > eps(Float64)) || !isfinite(outRBC_lead_lags[2]) ? 1000000 : outRBC_lead_lags[3] : 1000000

    # outRBC_param = try RBC_param.SS_solve_func(RBC_param.parameter_values, RBC_param, false, verbose, par_inputs, x[end]) catch end

    # total_iters += outRBC_param isa Tuple{Vector{Float64}, Float64, Int64} ? (outRBC_param[2] > eps(Float64)) || !isfinite(outRBC_param[2]) ? 1000000 : outRBC_param[3] : 1000000

    # total_iters >= 1000000 ? NaN : total_iters

    total_iters_pars = sum(abs2,vcat(x[[3,4]]...,[1 .- x[12:15]]...)) * 10

    # println(total_iters)
    return Float64(total_iters + total_iters_pars),total_iters
end)




innit = [ 
    .55
    .45

    0.001
    0.001
    1.0
    1.0

    0.005
    0.5
    0.5
    0.5

    0.01

    0.5
    0.5
    0.5
    0.5

    2.0
    2.5
    0.5
]



lbs = zero(innit)
# lbs .+= eps()*2
# lbs[5:10] .= -16
# lbs[12:15] .= 0
lbs[16] = -.5
lbs[17] = 2
lbs[end] = -1
ubs = zero(innit)
ubs .+= 1
ubs[1:6] .= 10
# ubs[5:10] .= 2
ubs[12:15] .= 10
ubs[16] = 4.5
ubs[17] = 3
ubs[end] = 2

prob = OptimizationProblem(f, innit, false, lb = lbs, ub = ubs)




innit = [ 
    5.0
    .875

    0.0027
    0.0
    8.04
    1.0

    0.076
    0.235
    0.51
    0.0

    0.62

    0.005
    0.005
    0.005
    0.005

    2.0
    2.5
    0.85
]




# ϕ̄::T    =       5.0,
# ϕ̂::T    =       0.8725,
# μ̄¹::T   =       0.0027,
# μ̄²::T   =       0.0,
# p̄¹::T   =       8.04,
# p̄²::T   =       0.0,
# ρ::T    =       0.076,
# ρ¹::T   =       0.235,
# ρ²::T   =       0.51,
# ρ³::T   =       0.0,
# ν::T    =       0.62,
# λ̅¹::T   =       0.422,
# λ̅²::T   =       .3,
# λ̂̅¹::T   =       0.5047,
# λ̂̅²::T   =       .3,

f(innit,true)

maxt = 10 * 60


sol_ESCH    =   solve(prob, NLopt.GN_ESCH(),    maxtime = maxt); sol_ESCH.minimum
innit = sol_ESCH.u

innit = [7.971710350206478
0.9041767277613695
0.02578560981066282
0.0
1.0
22.097277731204894
0.10321154244168337
0.16697538814219845
0.07074783314454779
0.010182004981312102
0.7942104062189025
0.8394091639953956
0.5276256406439965
0.2298040121252335
0.7636812309458207
0.012227999306191117
0.5112774682252668
0.9814967913661943
0.862118389647011
2.6999735986281466
2.050375187497662
0.8955836847010801]



innit = [7.971710350206478
0.9041767277613695
0.02578560981066282
0.0
1.0
0.0
0.0987991963495973
0.16697538814219845
0.07074783314454779
0.010182004981312102
0.7942104062189025
0.8394091639953956
1.0
0.49101785714380347
1.0
0.012802334805329335
1.0
0.9814967913661943
1.0
3
2
0.8955836847010801]


innit = [ 
8
0.904
0.026
0.0
1.0
0.0
0.1
0.17
0.07
0.01
0.8
0.84
1.0
0.5
1.0
0.0128
1.0
0.9815
1.0
3.0
2.0
0.897]

f(innit, false)

f(innit, true)

f(round.(innit, digits = 8),false)


sol_BOBYQA  =   solve(prob, NLopt.LN_BOBYQA()); sol_BOBYQA.minimum # fast and solves
sol_SBPLX   =   solve(prob, NLopt.LN_SBPLX()); sol_SBPLX.minimum



sol_COBYLA  =   solve(prob, NLopt.LN_COBYLA()); sol_COBYLA.minimum # slow
sol_NM      =   solve(prob, NLopt.LN_NELDERMEAD()); sol_NM.minimum
sol_PRAXIS  =   solve(prob, NLopt.LN_PRAXIS()); sol_PRAXIS.minimum

using OptimizationBBO
sol_BBO   =   solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(),   maxtime = maxt); sol_BBO.minimum #gets the far off parameters when only few models are involved
sol_BBO   =   solve(prob, BBO_dxnes(),   maxtime = maxt); sol_BBO.minimum
# sol_BBO   =   solve(prob, BBO_resampling_inheritance_memetic_search(),   maxtime = maxt); sol_BBO.minimum


sol_AGS     =   solve(prob, NLopt.GN_AGS()); sol_AGS.minimum # slow and unreliable
sol_CRS2    =   solve(prob, NLopt.GN_CRS2_LM(), maxtime = maxt); sol_CRS2.minimum
sol_DIRECT  =   solve(prob, NLopt.GN_DIRECT_L_RAND(),  maxtime = maxt); sol_DIRECT.minimum
sol_ISRES   =   solve(prob, NLopt.GN_ISRES(),   maxtime = maxt); sol_ISRES.minimum

sol_Multi   =   solve(prob, NLopt.G_MLSL_LDS(), local_method =  NLopt.LN_BOBYQA(), maxtime = maxt); sol_Multi.minimum


using OptimizationMetaheuristics
sol_Metaheuristics   =   solve(prob, ECA(),   maxtime = maxt); sol_Metaheuristics.minimum


using OptimizationMultistartOptimization
sol_Multi = solve(prob, MultistartOptimization.TikTak(40), NLopt.LN_SBPLX()); sol_Multi.minimum

sol_Multi = solve(prob, NLopt.G_MLSL_LDS(), local_method =  NLopt.LN_SBPLX(), maxtime = maxt)


innit = xx
xx = innit
sqrt(170)
xx = sol_SBPLX.u
xx = sol_ESCH.u


xx[20:21] .= Int.(round.(xx[20:21]))

SW07.SS_solve_func(SW07.parameter_values, SW07, false, true, 
                    Dict(
                        :ϕ̄  =>  maximum(xx[1:2]),
                        :ϕ̂  =>  minimum(xx[1:2]),
                        :μ̄¹ =>  xx[3],
                        :μ̄² =>  xx[4],
                        :p̄¹ =>  xx[5],
                        :p̄² =>  xx[6],
                        :ρ  =>  xx[7],
                        :ρ¹ =>  xx[8],
                        :ρ² =>  xx[9],
                        :ρ³ =>  xx[10],
                        :ν  =>  xx[11],
                        :λ¹ =>  xx[12],
                        :λ² =>  xx[13],
                        :λ̂¹ =>  xx[14],
                        :λ̂² =>  xx[15],
                        :λ̅¹ =>  xx[16],
                        :λ̅² =>  xx[17],
                        :λ̂̅¹ =>  xx[18],
                        :λ̂̅² =>  xx[19],
                        :transformation_level   => Int(abs(round(xx[20]))),
                        :backtracking_order     => Int(abs(round(xx[21])))
                    ),xx[end])


m.SS_solve_func(m.parameter_values, m, false, true, 
Dict(
    :ϕ̄  =>  maximum(xx[1:2]),
    :ϕ̂  =>  minimum(xx[1:2]),
    :μ̄¹ =>  xx[3],
    :μ̄² =>  xx[4],
    :p̄¹ =>  xx[5],
    :p̄² =>  xx[6],
    :ρ  =>  xx[7],
    :ρ¹ =>  xx[8],
    :ρ² =>  xx[9],
    :ρ³ =>  xx[10],
    :ν  =>  xx[11],
    :λ¹ =>  xx[12],
    :λ² =>  xx[13],
    :λ̂¹ =>  xx[14],
    :λ̂² =>  xx[15],
    :λ̅¹ =>  xx[16],
    :λ̅² =>  xx[17],
    :λ̂̅¹ =>  xx[18],
    :λ̂̅² =>  xx[19],
    :transformation_level   => Int(abs(round(xx[20]))),
    :backtracking_order     => Int(abs(round(xx[21])))
),xx[end])


GNSS_2010.SS_solve_func(GNSS_2010.parameter_values, GNSS_2010, false, true, 
Dict(
    :ϕ̄  =>  maximum(xx[1:2]),
    :ϕ̂  =>  minimum(xx[1:2]),
    :μ̄¹ =>  xx[3],
    :μ̄² =>  xx[4],
    :p̄¹ =>  xx[5],
    :p̄² =>  xx[6],
    :ρ  =>  xx[7],
    :ρ¹ =>  xx[8],
    :ρ² =>  xx[9],
    :ρ³ =>  xx[10],
    :ν  =>  xx[11],
    :λ¹ =>  xx[12],
    :λ² =>  xx[13],
    :λ̂¹ =>  xx[14],
    :λ̂² =>  xx[15],
    :λ̅¹ =>  xx[16],
    :λ̅² =>  xx[17],
    :λ̂̅¹ =>  xx[18],
    :λ̂̅² =>  xx[19],
    :transformation_level   => Int(abs(round(xx[20]))),
    :backtracking_order     => Int(abs(round(xx[21])))
),xx[end])


asdasd = Ascari_Sbordone_2014.SS_solve_func(Ascari_Sbordone_2014.parameter_values, Ascari_Sbordone_2014, false, true, 
Dict(
    :ϕ̄  =>  maximum(xx[1:2]),
    :ϕ̂  =>  minimum(xx[1:2]),
    :μ̄¹ =>  xx[3],
    :μ̄² =>  xx[4],
    :p̄¹ =>  xx[5],
    :p̄² =>  xx[6],
    :ρ  =>  xx[7],
    :ρ¹ =>  xx[8],
    :ρ² =>  xx[9],
    :ρ³ =>  xx[10],
    :ν  =>  xx[11],
    :λ¹ =>  xx[12],
    :λ² =>  xx[13],
    :λ̂¹ =>  xx[14],
    :λ̂² =>  xx[15],
    :λ̅¹ =>  xx[16],
    :λ̅² =>  xx[17],
    :λ̂̅¹ =>  xx[18],
    :λ̂̅² =>  xx[19],
    :transformation_level   => Int(abs(round(xx[20]))),
    :backtracking_order     => Int(abs(round(xx[21])))
),xx[end])

asdasd[1]
   typeof(asdasd) 


   SW03.SS_solve_func(SW03.parameter_values, SW03, false, true, 
   Dict(
       :ϕ̄  =>  maximum(xx[1:2]),
       :ϕ̂  =>  minimum(xx[1:2]),
       :μ̄¹ =>  xx[3],
       :μ̄² =>  xx[4],
       :p̄¹ =>  xx[5],
       :p̄² =>  xx[6],
       :ρ  =>  xx[7],
       :ρ¹ =>  xx[8],
       :ρ² =>  xx[9],
       :ρ³ =>  xx[10],
       :ν  =>  xx[11],
       :λ¹ =>  xx[12],
       :λ² =>  xx[13],
       :λ̂¹ =>  xx[14],
       :λ̂² =>  xx[15],
       :λ̅¹ =>  xx[16],
       :λ̅² =>  xx[17],
       :λ̂̅¹ =>  xx[18],
       :λ̂̅² =>  xx[19],
       :transformation_level   => Int(abs(round(xx[20]))),
       :backtracking_order     => Int(abs(round(xx[21])))
   ),xx[end])

    
   
NAWM_EAUS_2008.SS_solve_func(NAWM_EAUS_2008.parameter_values, NAWM_EAUS_2008, false, true, 
Dict(
    :ϕ̄  =>  maximum(xx[1:2]),
    :ϕ̂  =>  minimum(xx[1:2]),
    :μ̄¹ =>  xx[3],
    :μ̄² =>  xx[4],
    :p̄¹ =>  xx[5],
    :p̄² =>  xx[6],
    :ρ  =>  xx[7],
    :ρ¹ =>  xx[8],
    :ρ² =>  xx[9],
    :ρ³ =>  xx[10],
    :ν  =>  xx[11],
    :λ¹ =>  xx[12],
    :λ² =>  xx[13],
    :λ̂¹ =>  xx[14],
    :λ̂² =>  xx[15],
    :λ̅¹ =>  xx[16],
    :λ̅² =>  xx[17],
    :λ̂̅¹ =>  xx[18],
    :λ̂̅² =>  xx[19],
    :transformation_level   => Int(abs(round(xx[20]))),
    :backtracking_order     => Int(abs(round(xx[21])))
),xx[end])

    
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
