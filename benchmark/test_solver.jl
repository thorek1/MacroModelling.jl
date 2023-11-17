using MacroModelling

include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")
# include("../test/models/RBC_CME_calibration_equations_and_parameter_definitions.jl")
include("../models/Backus_Kehoe_Kydland_1992.jl")
include("../models/Baxter_King_1993.jl")
include("../models/SW03.jl")
include("../models/GNSS_2010.jl")
include("../models/Ghironi_Melitz_2005.jl")
include("../models/SGU_2003_debt_premium.jl")
include("../models/NAWM_EAUS_2008.jl") # stands out
include("../models/JQ_2012_RBC.jl")
include("../models/Ireland_2004.jl")
include("../models/Caldara_et_al_2012.jl")
include("../models/Gali_Monacelli_2005_CITR.jl")
include("../models/Gali_2015_chapter_3_nonlinear.jl")
include("../models/Aguiar_Gopinath_2007.jl")
include("../models/Ascari_Sbordone_2014.jl") # stands out
include("../models/FS2000.jl")
include("../models/SW07.jl")
include("../models/RBC_baseline.jl")
include("../models/Guerrieri_Iacoviello_2017.jl") # stands out


all_models = [
    m, 
    Backus_Kehoe_Kydland_1992, 
    Baxter_King_1993, 
    SW03, 
    GNSS_2010, 
    Ghironi_Melitz_2005, 
    SGU_2003_debt_premium, 
    NAWM_EAUS_2008, 
    JQ_2012_RBC, 
    Ireland_2004, 
    Caldara_et_al_2012, 
    Gali_Monacelli_2005_CITR, 
    Gali_2015_chapter_3_nonlinear, 
    Aguiar_Gopinath_2007, 
    Ascari_Sbordone_2014, 
    FS2000, 
    SW07, 
    Guerrieri_Iacoviello_2017
];



function calc_total_iters(model, par_inputs, starting_point)
    outmodel = try model.SS_solve_func(model.parameter_values, model, false, starting_point, par_inputs) catch end
    # outmodel = model.SS_solve_func(model.parameter_values, model, false, starting_point, par_inputs)
    
    outmodel isa Tuple{Vector{Float64}, Tuple{Float64, Int64}} ? 
        (outmodel[2][1] > eps(Float64)) || !isfinite(outmodel[2][1]) ? 
            1000000 : 
        outmodel[2][2] : 
    1000000
end

par_inputs = m.solver_parameters

calc_total_iters(m, par_inputs, 0.897)

using Optimization, OptimizationNLopt

f = OptimizationFunction((x,verbose)-> begin 
    total_iters = 0

    x[1:2] = sort(x[1:2], rev = true)

    par_inputs = MacroModelling.solver_parameters(   
        eps(),eps(),250,  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19],Int(abs(round(x[20]))),x[21],Int(abs(round(x[22])))
    )
    
    total_iters = 0
    for model in all_models
        total_iters += calc_total_iters(model, par_inputs, x[end])
    end

    total_iters_pars = sum(abs2,vcat(x[[3,4]]..., [1 .- x[12:19]]..., x[21] / 100)) * 10

    # println(total_iters)
    return Float64(total_iters + total_iters_pars), total_iters
end)

# innit = [ 
#     3.307670699324403
#     0.8887
#     0.023813243535282552
#     0.026833357448752496
#     2.442999997612768
#     1.49949375
#     0.0008014110419103691
#     0.041527311524283926
#     0.048143903530580914
#     0.0033122894690720854
#     0.5401102854078268
#     0.11292946464826276
#     0.00011413295389511516
#     0.6553524666109665
#     0.02388678809616366
#     0.99778
#     0.00011413295389511516
#     0.6553524666109665
#     0.02388678809616366
#     0.99778
#     1.0
#     3.0
# ]



# innit = [
#     2.1463884211036226
#     0.9929437566596109
#     0.6773423825308124
#     0.0
#     4.435883184171622
#     0.1545259444769216
#     0.16236330077946382
#     0.04628182054753979
#     0.5711216406099425
#     0.0
#     0.6804997522959056
#     0.8391805314686501
#     0.15117829923684067
#     0.08883112140047633
#     0.7130603312464207
#     0.935850006981596
# ]

# innit = [
#     5.0
#     0.8725
#     0.0027
#     0.0
#     8.04
#     0.0
#     0.076
#     0.235
#     0.51
#     0.0
#     0.62
#     0.422
#     1.0
#     0.5047
#     1.0
#     0.422
#     1.0
#     0.5047
#     1.0
#     1.0
#     0.0
#     2.0
#     0.7688
# ]


innit = [ 
    .55
    .45
    0.5
    0.5
    1
    1
    0.005
    0.5
    0.5
    0.5
    0.01
    1
    1
    1
    1
    1
    1
    1
    1
    2.5
    0.0
    2.55
    0.5
]



lbs = zero(innit)
# lbs .+= eps()*2
lbs[20] = -.5
lbs[21] = -100
lbs[22] = 2.45
lbs[end] = -100

ubs = zero(innit)
ubs .+= 1
ubs[1:2] .= 10
ubs[5:6] .= 100
ubs[20] = 4.5
ubs[21] = 100
ubs[22] = 2.55
ubs[end] = 100

prob = OptimizationProblem(f, innit, false, lb = lbs, ub = ubs)

f(innit,true)
# using BenchmarkTools

# max_minutes = 1 * 60
max_hours = 24 * 60 ^ 2
# Start 1200 (PT time)

sol_ESCH = solve(prob, NLopt.GN_ESCH(), maxtime = max_hours); sol_ESCH.minimum

innit2 = deepcopy(sol_ESCH.u)



@benchmark f(innit2,true)



x = innit2




x[1:2] = sort(x[1:2], rev = true)

par_inputs = MacroModelling.solver_parameters(   
    eps(),eps(),250,  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19],Int(abs(round(x[20]))),x[21],Int(abs(round(x[22])))
)

total_iters = 0
for model in all_models
    iter = calc_total_iters(model, par_inputs, x[end])
    println(iter, "\t ", model.model_name)
    total_iters += iter
end
total_iters


x[1:2] = sort(x[1:2], rev = true)

par_inputs = MacroModelling.solver_parameters(   
    eps(),eps(),250,  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19],Int(abs(round(x[20]))),x[21],Int(abs(round(x[22])))
)

@benchmark calc_total_iters(NAWM_EAUS_2008, par_inputs, x[23])

@benchmark calc_total_iters(Guerrieri_Iacoviello_2017, par_inputs, x[23])

calc_total_iters(Aguiar_Gopinath_2007, par_inputs, x[end])

calc_total_iters(Ascari_Sbordone_2014, par_inputs, x[23])

calc_total_iters(FS2000, par_inputs, x[23])

calc_total_iters(SW03, par_inputs, x[end])

calc_total_iters(SW07, par_inputs, x[end])


model = Backus_Kehoe_Kydland_1992
starting_point = .9#x[end]
outmodel = model.SS_solve_func(model.parameter_values, model, false, starting_point, par_inputs)
    
    outmodel isa Tuple{Vector{Float64}, Tuple{Float64, Int64}} ? 
        (outmodel[2][1] > eps(Float64)) || !isfinite(outmodel[2][1]) ? 
            1000000 : 
        outmodel[2][2] : 
    1000000



innit2 = [  2.9286687462521286
1.0934322881820666
8.040292279055312e-5
0.0010349673716525869
9.612040527215113
50.1236879266217
0.5669837295436179
0.5151760400524997
0.44424209515580787
0.0017803001645317015
0.20165426073261922
0.9893989333367081
0.9992664322446558
0.9947474775878444
0.9947619544912081
0.9758001846468876
0.992316474092166
0.9753918769395817
0.9907848466210624
0.5322997748902552
0.0007004948302871516
2.4590286886069483
25.260971258391464]

x = sol_ESCH.u


x[1:2] = sort(x[1:2], rev = true)

par_inputs = MacroModelling.solver_parameters(   
    eps(),eps(),250,  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19],Int(abs(round(x[20]))),x[21],Int(abs(round(x[22])))
)

calc_total_iters(NAWM_EAUS_2008, par_inputs, x[end])

calc_total_iters(Guerrieri_Iacoviello_2017,innit2[1:end-1],innit[end])
innit2
sort(innit2[1:2], rev = true)
x = innit2
par_inputs = MacroModelling.solver_parameters(   
        eps(),eps(),250,  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19],Int(abs(round(x[20]))),x[21],Int(abs(round(x[22])))
    )


calc_total_iters(Guerrieri_Iacoviello_2017,par_inputs,innit2[end])

Guerrieri_Iacoviello_2017.SS_solve_func(Guerrieri_Iacoviello_2017.parameter_values, Guerrieri_Iacoviello_2017, false, innit2[end], par_inputs)





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


prob = OptimizationProblem(f, innit2, false, lb = lbs, ub = ubs)

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












all_models = [
    # m, 
    # Backus_Kehoe_Kydland_1992, 
    # Baxter_King_1993, 
    # SW03, 
    # GNSS_2010, 
    # Ghironi_Melitz_2005, 
    # SGU_2003_debt_premium, 
    # NAWM_EAUS_2008, 
    # JQ_2012_RBC, 
    # Ireland_2004, 
    # Caldara_et_al_2012, 
    # Gali_Monacelli_2005_CITR, 
    # Gali_2015_chapter_3_nonlinear, 
    # Aguiar_Gopinath_2007, 
    # Ascari_Sbordone_2014, 
    # FS2000, 
    # SW07, 
    Guerrieri_Iacoviello_2017
];



function calc_total_iters(model, par_inputs, starting_point)
    outmodel = try model.SS_solve_func(model.parameter_values, model, false, starting_point, par_inputs) catch end
    # outmodel = model.SS_solve_func(model.parameter_values, model, false, starting_point, par_inputs)
    
    outmodel isa Tuple{Vector{Float64}, Tuple{Float64, Int64}} ? 
        (outmodel[2][1] > eps(Float64)) || !isfinite(outmodel[2][1]) ? 
            1000000 : 
        outmodel[2][2] : 
    1000000
end


f = OptimizationFunction((x,verbose)-> begin 
    total_iters = 0

    # x[1:2] = sort(x[1:2], rev = true)

    # par_inputs = MacroModelling.solver_parameters(   
    #     eps(),eps(),250,  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19],Int(abs(round(x[20]))),x[21],Int(abs(round(x[22])))
    # )
    
    total_iters = 0
    for model in all_models
        total_iters += calc_total_iters(model, model.solver_parameters, x[end])
    end

    # total_iters_pars = sum(abs2,vcat(x[[3,4]]..., [1 .- x[12:19]]..., x[21] / 100)) * 10

    return Float64(total_iters), total_iters
end)

innit = [ 
    0.5
]



lbs = zero(innit)
# lbs .+= eps()*2
# lbs[20] = -.5
# lbs[21] = -100
# lbs[22] = 2.45
lbs[end] = -100

ubs = zero(innit)
ubs .+= 1
# ubs[1:2] .= 10
# ubs[5:6] .= 100
# ubs[20] = 4.5
# ubs[21] = 100
# ubs[22] = 2.55
ubs[end] = 100

prob = OptimizationProblem(f, innit, false, lb = lbs, ub = ubs)

maxt = 10 * 60
using OptimizationBBO
sol_BBO   =   solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(),   maxtime = maxt); sol_BBO.minimum #gets the far off parameters when only few models are involved

sol_BBO.u
f(sol_BBO.u, false)


# 71.69060992965288 works
Guerrieri_Iacoviello_2017.SS_solve_func(Guerrieri_Iacoviello_2017.parameter_values, Guerrieri_Iacoviello_2017, false, sol_BBO.u[1], Guerrieri_Iacoviello_2017.solver_parameters)


sol_ESCH = solve(prob, NLopt.GN_ESCH(), maxtime = maxt); sol_ESCH.minimum

f(71.69,[])
Guerrieri_Iacoviello_2017.SS_solve_func(Guerrieri_Iacoviello_2017.parameter_values, Guerrieri_Iacoviello_2017, false, sol_ESCH.u[1], Guerrieri_Iacoviello_2017.solver_parameters)
Guerrieri_Iacoviello_2017.SS_solve_func(Guerrieri_Iacoviello_2017.parameter_values, Guerrieri_Iacoviello_2017, false, 0.0-10*eps(Float32), Guerrieri_Iacoviello_2017.solver_parameters)


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
