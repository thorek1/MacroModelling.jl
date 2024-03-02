using MacroModelling, Optimization, OptimizationNLopt, OptimizationMetaheuristics

# logic to implement

# when finding the NSSS the first time: use the system with aux variables as unknowns, parameters and solved vars as unknowns, separate the system into smaller blocks if possible
# do redundant var reduction only if the vars to reduce are not part of nonlinear expressions
# search for optim parameters that solve any of the models with a side condition of having the least necessary iterations
# for estimation use the small system (no aux, pars and solved vars) and again look for optim parameters that solve the system with the least necessary iterations during estimation


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
include("../models/SWnonlinear.jl")
# include("../models/RBC_baseline.jl") # no solver block / everything analytical
include("../models/Guerrieri_Iacoviello_2017.jl") # stands out


all_models = [
    SWnonlinear,
    Guerrieri_Iacoviello_2017,
    NAWM_EAUS_2008, 
    GNSS_2010, 
    Ascari_Sbordone_2014, 
    SW03, 
    Backus_Kehoe_Kydland_1992, 
    m, 
    Baxter_King_1993, 
    Ghironi_Melitz_2005, 
    SGU_2003_debt_premium, 
    JQ_2012_RBC, 
    Ireland_2004, 
    Caldara_et_al_2012, 
    Gali_Monacelli_2005_CITR, 
    Gali_2015_chapter_3_nonlinear, 
    Aguiar_Gopinath_2007, 
    FS2000, 
    SW07
];


function calc_total_iters(model, par_inputs, starting_point)
    outmodel = try model.SS_solve_func(model.parameter_values, model, false, starting_point, par_inputs) catch end

    outmodel isa Tuple{Vector{Float64}, Tuple{Float64, Int64}} ? 
        (outmodel[2][1] > 1e-12) || !isfinite(outmodel[2][1]) ? 
            10000 : 
        outmodel[2][2] : 
        10000
end

using Turing
# Function to calculate the posterior log likelihood
function evaluate_pars_loglikelihood(pars, models)
    log_lik = 0.0
    
    model_iters = zeros(Int, length(models))
    
    pars[1:2] = sort(pars[1:2], rev = true)

    # Apply prior distributions
    log_lik -= logpdf(filldist(MacroModelling.Gamma(1.0, 1.0, μσ = true), 19),pars[1:19])
    log_lik -= logpdf(MacroModelling.Normal(0.0, 2.0), pars[20])

    # Example solver parameters - this needs to be replaced with actual logic
    par_inputs = MacroModelling.solver_parameters(eps(), eps(), 500, pars[1:19]..., transform, 0.0, 2)

    # Iterate over all models and calculate the total iterations
    for (i,model) in enumerate(models)
        total_iters = calc_total_iters(model, par_inputs, pars[20])
        model_iters[i] = 1e1 * total_iters
    end
    
    return Float64(log_lik + sum(model_iters))
end

parameters = [2.9912988764832833, 0.8725, 0.0027, 0.028948770826150612, 8.04, 4.076413176215408, 0.06375413238034794, 0.24284340766769424, 0.5634017580097571, 0.009549630552246828, 0.6342888355132347, 0.5275522227754195, 1.0, 0.06178989216048817, 0.5234277812131813, 0.422, 0.011209254402846185, 0.5047, 0.6020757011698457, 0.7688]
# parameters = rand(20) .+ 1

evaluate_pars_loglikelihood(parameters, all_models)

lbs = fill(eps(),length(parameters))
lbs[20] = -20

ubs = fill(50.0,length(parameters))

prob = OptimizationProblem(evaluate_pars_loglikelihood, parameters, all_models, lb = lbs, ub = ubs)

max_minutes = 5 * 60^2 + 30 * 60

# max_minutes = 20

sol_GA = solve(prob, GA(), maxtime = max_minutes); sol_GA.minimum 
# sol_SA = solve(prob, SA(), maxtime = max_minutes); sol_SA.minimum 
# sol_ECA = solve(prob, ECA(), maxtime = max_minutes); sol_ECA.minimum 
# sol_PSO = solve(prob, PSO(), maxtime = max_minutes); sol_PSO.minimum 
# sol_WOA = solve(prob, WOA(), maxtime = max_minutes); sol_WOA.minimum 
# sol_ABC = solve(prob, ABC(), maxtime = max_minutes); sol_ABC.minimum 
# WOA terminates early
# sol_ESCH = solve(prob, NLopt.GN_ESCH(), maxtime = max_minutes); sol_ESCH.minimum
# sol_CRS = solve(prob, NLopt.GN_CRS2_LM(), maxtime = max_minutes); sol_CRS.minimum # gets nowhere
# sol_DIRECT = solve(prob, NLopt.GN_DIRECT(), maxtime = max_minutes); sol_DIRECT.minimum

pars = deepcopy(sol_GA.u)

# transform = 0
# pars = [0.0005091362638940568, 2.8685509141200138e-5, 5.853782207686227e-5, 0.00013196080350191824, 0.0002770476425730156, 0.0009519716244767909, 0.0001214119776051054, 0.00030908501576441285, 0.00015604633597157022, 0.00022641413680158165, 0.00019937397397335106, 0.00024166807372048577, 0.0006155904822177251, 6.834833988676723e-5, 0.0005660185689597568, 0.00023832991638765894, 1.3844380657673424e-5, 6.031788146343705e-5, 0.0002638201993322502, -0.0012275435299784476]

# transform = 1
# pars = [0.0009722629716867275, 0.0001226307621386892, 0.001394733545491217, 0.00020917332586393864, 3.443396112525556e-5, 0.000335726967547115, 0.0002127283513562217, 0.00022964278163688752, 0.0007721004173625069, 0.000674726682515611, 0.0007265065066785959, 0.0006368782202599225, 0.00033532317447594986, 0.00016604570894159111, 0.0007628267491510528, 1.4257916433958474e-5, 0.00017981947865976965, 0.0003429553550297806, 9.249519367471304e-6, -0.007678671054904385]

# transform = 2
# pars = [5.846510669966325e-5, 3.4129209353932815e-5, 0.0006505823521997761, 9.775916040749463e-5, 7.035264708621578e-5, 0.0007777756833533508, 0.002335262852991666, 0.00024831303889830245, 0.0005387508892999407, 0.00020225438940622587, 0.0001995261291163045, 0.000177631857961649, 3.403135735039661e-5, 0.0003017909959808597, 0.00022612713121778226, 0.0006430807474245832, 0.000496769666096488, 4.859031350196916e-5, 0.00029815039144675864, -0.014631202632603646]

# transform = 3
# pars = [0.001599643065980533, 0.00023357673646668051, 0.0008217384808418395, 0.0016514597308153382, 8.740547578718626e-6, 8.842504188831453e-5, 5.6630416478296754e-5, 0.0010582351341452108, 0.0002663753023310461, 0.0004692002779868142, 0.0005955395984410753, 0.00038998440665810537, 0.0007312132025819346, 0.0010101692249234668, 0.00026494260353431584, 0.0003154723515398832, 0.00018704238077452866, 0.00037738743688103533, 0.001210668814603666, -0.0008009605883678717]

println("Transform: $transform")
println("Parameters: $pars")

model_iters = zeros(Int, length(all_models))
    
pars[1:2] = sort(pars[1:2], rev = true)

# Example solver parameters - this needs to be replaced with actual logic
par_inputs = MacroModelling.solver_parameters(eps(), eps(), 250, pars[1:19]..., transform, 0.0, 2)

# Iterate over all models and calculate the total iterations
for (i,model) in enumerate(all_models)
    total_iters = calc_total_iters(model, par_inputs, pars[20])
    model_iters[i] = total_iters
end

println("Iterations per model: $model_iters")
println("Total iterations across model: $(sum(model_iters))")
