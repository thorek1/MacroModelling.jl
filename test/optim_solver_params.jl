using MacroModelling, Optimization, OptimizationNLopt

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
# include("../models/RBC_baseline.jl") # no solver block / everything analytical
include("../models/Guerrieri_Iacoviello_2017.jl") # stands out


all_models = [
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
function evaluate_pars_loglikelihood(pars, models, transform)
    log_lik = 0.0
    
    model_iters = zeros(Int, length(models))
    
    pars[1:2] = sort(pars[1:2], rev = true)

    # Apply prior distributions
    log_lik -= logpdf(filldist(MacroModelling.Gamma(1.0, 2.0, μσ = true), 19),pars[1:19])
    log_lik -= logpdf(MacroModelling.Normal(0.0, 10.0), pars[20])

    # Example solver parameters - this needs to be replaced with actual logic
    par_inputs = MacroModelling.solver_parameters(eps(), eps(), 500, pars[1:19]..., transform, 0.0, 2)

    # Iterate over all models and calculate the total iterations
    for (i,model) in enumerate(models)
        total_iters = calc_total_iters(model, par_inputs, pars[20])
        model_iters[i] = 1e2 * total_iters
    end
    
    return Float64(log_lik + sum(model_iters))
end

# parameters = repeat([2.9912988764832833, 0.8725, 0.0027, 0.028948770826150612, 8.04, 4.076413176215408, 0.06375413238034794, 0.24284340766769424, 0.5634017580097571, 0.009549630552246828, 0.6342888355132347, 0.5275522227754195, 1.0, 0.06178989216048817, 0.5234277812131813, 0.422, 0.011209254402846185, 0.5047, 0.6020757011698457, 0.7688],4)
parameters = rand(20) .+ 1

evaluate_pars_loglikelihood(parameters, all_models)

lbs = fill(eps(),length(parameters))
lbs[20] .= -20

ubs = fill(30.0,length(parameters))

prob = OptimizationProblem(evaluate_pars_loglikelihood, parameters, all_models, lb = lbs, ub = ubs)

max_minutes = 5 * 60^2

sol_ESCH = solve(prob, NLopt.GN_ESCH(), maxtime = max_minutes); sol_ESCH.minimum

pars = deepcopy(sol_ESCH.u)

println("Transform: $transform")
println("Parameters: $pars")

model_iters = zeros(Int, length(models))
    
pars[1:2] = sort(pars[1:2], rev = true)

# Example solver parameters - this needs to be replaced with actual logic
par_inputs = MacroModelling.solver_parameters(eps(), eps(), 250, pars[1:19]..., transform, 0.0, 2)

# Iterate over all models and calculate the total iterations
for (i,model) in enumerate(models)
    total_iters = calc_total_iters(model, par_inputs, pars[20])
    model_iters[i] = 1e2 * total_iters
end

println("Iterations per model: $model_iters")