using Test
using MacroModelling
using Random
import StatsPlots
using DelimitedFiles
using Dates
using AxisKeys, SparseArrays

include("test_helpers.jl")

Random.seed!(1)

@testset verbose = true "SW07 estim" begin
    include("../models/Smets_Wouters_2007.jl")

    # load data
    dat, header = readdlm("data/usmodel.csv", ',', header = true)
    dat = Float64.(dat)
    names = vec(Symbol.(strip.(header)))

    # load data
    data = KeyedArray(dat', Variable = names, Time = axes(dat, 1))

    # declare observables as written in csv file
    observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

    # Subsample
    # subset observables in data
    sample_idx = 47:230 # 1960Q1-2004Q4

    data = data(observables_old, sample_idx)

    # declare observables as written in model
    observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

    data = rekey(data, :Variable => observables)

    data_rekey = rekey(data, :Time => quarterly_dates(Date(1960, 1, 1), size(data,2)))


    plot_model_estimates(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24])

    plot_model_estimates!(Smets_Wouters_2007, data, parameters = [:csadjcost => 3, :calfa => 0.24])

    plot_model_estimates!(Smets_Wouters_2007, data, parameters = [:csadjcost => 3, :calfa => 0.28])


    plot_model_estimates(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24])

    plot_model_estimates!(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24], filter = :inversion)


    plot_model_estimates(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24])

    plot_model_estimates!(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24], filter = :inversion)

    plot_model_estimates!(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24], smooth = false)


    plot_model_estimates(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24], smooth = false)

    plot_model_estimates!(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24], smooth = false, presample_periods = 50)


    plot_model_estimates(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24])

    plot_model_estimates!(Smets_Wouters_2007, data[:,20:end], parameters = [:csadjcost => 6, :calfa => 0.24])


    plot_model_estimates(Smets_Wouters_2007, data_rekey, parameters = [:csadjcost => 6, :calfa => 0.24])

    plot_model_estimates!(Smets_Wouters_2007, data_rekey, parameters = [:csadjcost => 5, :calfa => 0.24])


    plot_model_estimates(Smets_Wouters_2007, data, parameters = [:csadjcost => 6, :calfa => 0.24])

    plot_model_estimates!(Smets_Wouters_2007, data_rekey, parameters = [:csadjcost => 5, :calfa => 0.24])

    # FS2000 model and data  
    include("../models/FS2000.jl")

    # load data
    dat, header = readdlm("data/FS2000_data.csv", ',', header = true)
    dat = Float64.(dat)
    names = vec(header)
    dataFS2000 = KeyedArray(dat', Variable = Symbol.("log_".*names), Time = axes(dat, 1))
    dataFS2000 = log.(dataFS2000)

    # declare observables
    observables = sort(Symbol.("log_".*names))

    # subset observables in data
    dataFS2000 = dataFS2000(observables,:)

    dataFS2000_rekey = rekey(dataFS2000, :Time => quarterly_dates(Date(1950, 1, 1), size(dataFS2000,2)))

    plot_model_estimates(FS2000, dataFS2000)

    plot_model_estimates(FS2000, dataFS2000_rekey[:,1:10])

    plot_shock_decomposition(FS2000, dataFS2000_rekey[:,1:10])

    plot_shock_decomposition(FS2000, dataFS2000_rekey)


    dataFS2000_rekey2 = rekey(dataFS2000, :Time => 1:1:size(dataFS2000,2))

    plot_shock_decomposition(FS2000, dataFS2000)

    plot_shock_decomposition(FS2000, dataFS2000_rekey2)


    plot_model_estimates(FS2000, dataFS2000_rekey, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))

    plot_model_estimates!(Smets_Wouters_2007, data_rekey)


    plot_model_estimates(FS2000, dataFS2000_rekey, parameters = :alp => 0.356, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))

    plot_model_estimates!(Smets_Wouters_2007, data_rekey)

    plot_model_estimates!(FS2000, dataFS2000_rekey, parameters = :alp => 0.3, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))


    plot_model_estimates!(Smets_Wouters_2007, data_rekey, parameters = :csigma => 0.3)

    plot_model_estimates(FS2000, dataFS2000_rekey, parameters = :alp => 0.356, shock_decomposition = true, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))


    estims = get_estimated_variables(Smets_Wouters_2007, data)

    plot_irf(Smets_Wouters_2007, shocks = :em, shock_size = 10)

    plot_irf!(Smets_Wouters_2007,initial_state = collect(estims[:,end]), shocks = :none, plot_type = :stack)

    plot_irf!(Smets_Wouters_2007, shocks = [:em, :ea], negative_shock = true, plot_type = :stack)
    
    shock_mat = randn(Smets_Wouters_2007.constants.post_model_macro.nExo,3)

    plot_irf!(Smets_Wouters_2007, shocks = shock_mat, plot_type = :stack)

    plot_irf!(Smets_Wouters_2007, shocks = shock_mat, plot_type = :stack)


    plot_irf(Smets_Wouters_2007, shocks = :em, periods = 5, variables = [:y, :k, :c])
    
    plot_irf!(FS2000, shocks = :e_m, periods = 5, plot_type = :stack, shock_size = 10, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w), variables = [:y, :k, :c])


    plot_irf(Smets_Wouters_2007, shocks = :em, periods = 5)
    
    plot_irf!(FS2000, shocks = :e_m, periods = 5, plot_type = :stack, shock_size = 10, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))

    plot_irf!(FS2000, shocks = [:e_m, :e_a], shock_size = 20, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))
    
    plot_irf!(Smets_Wouters_2007, shocks = [:em, :ea], shock_size = 0.5)
    
    

    cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,8), Variables = [:y], Periods = 1:8)
    cndtns_lvl[1,8] = 1.4

    plot_conditional_forecast(Smets_Wouters_2007, cndtns_lvl, initial_state = collect(estims[:,end]))


    cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,4), Variables = [:pinfobs], Periods = 1:4)
    cndtns_lvl[1,4] = 2

    plot_conditional_forecast!(Smets_Wouters_2007, cndtns_lvl, plot_type = :stack)
    

    
    cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,8), Variables = [:y], Periods = 1:8)
    cndtns_lvl[1,8] = 1.45

    plot_conditional_forecast!(FS2000, cndtns_lvl, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))


    cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,4), Variables = [:y], Periods = 1:4)
    cndtns_lvl[1,4] = 2.01

    plot_conditional_forecast!(FS2000, cndtns_lvl, plot_type = :stack, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))
    # conditons on #3 is nothing which makes sense since it is not showing

    shock_mat = sprandn(Smets_Wouters_2007.constants.post_model_macro.nExo, 10, .1)

    cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,4), Variables = [:pinfobs], Periods = 1:4)
    cndtns_lvl[1,4] = 2

    plot_conditional_forecast!(Smets_Wouters_2007, cndtns_lvl, shocks = shock_mat, plot_type = :stack)
    
    

    cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,8), Variables = [:y], Periods = 1:8)
    cndtns_lvl[1,8] = 1.4
    
    shock_mat = sprandn(Smets_Wouters_2007.constants.post_model_macro.nExo, 10, .1)

    plot_conditional_forecast(Smets_Wouters_2007, cndtns_lvl, shocks = shock_mat, label = "SW07 w shocks", variables = [:y, :k, :c])

    plot_conditional_forecast!(Smets_Wouters_2007, cndtns_lvl, variables = [:y,:w])

    plot_conditional_forecast!(FS2000, cndtns_lvl, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))
    
    shock_mat = sprandn(FS2000.constants.post_model_macro.nExo, 10, .1)

    plot_conditional_forecast!(FS2000, cndtns_lvl, shocks = shock_mat, label = :rand_shocks, rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))
    

    plot_solution(FS2000, :k)

    plot_solution!(FS2000, :k, algorithm = :second_order)


    plot_solution(Smets_Wouters_2007, :pinf)

    plot_solution!(Smets_Wouters_2007, :pinf, algorithm = :second_order)


    plot_solution(FS2000, :y)
    
    plot_solution!(Smets_Wouters_2007, :y, variables = [:y, :k, :c])

    plot_solution!(Smets_Wouters_2007, :y, algorithm = :second_order, variables = [:y, :k, :c])


    # tol-only and tol-varying tests (struct and NamedTuple formulations)
    plot_model_estimates(Smets_Wouters_2007, data,
        tol = Tolerances(nsss = NsssTolerances(acceptance_tol = 1e-10)))

    plot_model_estimates!(Smets_Wouters_2007, data,
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),)))

    plot_model_estimates!(Smets_Wouters_2007, data,
        tol = Tolerances(first_order = FirstOrderTolerances(qme = SolverTolerances(acceptance_tol = 1e-12))))


    plot_shock_decomposition(FS2000, dataFS2000_rekey,
        tol = Tolerances(first_order = FirstOrderTolerances(lyapunov = SolverTolerances(acceptance_tol = 1e-14))))

    plot_shock_decomposition(FS2000, dataFS2000_rekey,
        tol = Tolerances(nsss = (acceptance_tol = 1e-10,)))


    plot_irf(Smets_Wouters_2007, shocks = :em,
        tol = Tolerances(first_order = FirstOrderTolerances(qme = SolverTolerances(acceptance_tol = 1e-12))))

    plot_irf!(Smets_Wouters_2007, shocks = :em,
        tol = Tolerances(nsss = (xtol = 1e-14,)))

    plot_irf!(FS2000, shocks = :e_m,
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),)),
        rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))


    cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,8), Variables = [:y], Periods = 1:8)
    cndtns_lvl[1,8] = 1.4

    plot_conditional_forecast(Smets_Wouters_2007, cndtns_lvl,
        tol = Tolerances(nsss = NsssTolerances(ftol = 1e-16)))

    plot_conditional_forecast!(Smets_Wouters_2007, cndtns_lvl,
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),)))


    plot_solution(FS2000, :y,
        tol = Tolerances(nsss = NsssTolerances(acceptance_tol = 1e-10)))
    
    plot_solution!(FS2000, :y,
        tol = Tolerances(first_order = FirstOrderTolerances(lyapunov = SolverTolerances(acceptance_tol = 1e-14))))

    plot_solution!(FS2000, :y,
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),)))


    plot_solution(Smets_Wouters_2007, :y, algorithm = :second_order,
        tol = Tolerances(second_order = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-14))))

    plot_solution!(Smets_Wouters_2007, :y, algorithm = :second_order,
        tol = Tolerances(second_order = (sylvester = (acceptance_tol = 1e-14,), lyapunov = (acceptance_tol = 1e-14,))))


    # combined tol + other argument tests
    plot_model_estimates(Smets_Wouters_2007, data,
        parameters = [:csadjcost => 5, :calfa => 0.22],
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),)))

    plot_model_estimates!(Smets_Wouters_2007, data,
        parameters = [:csadjcost => 3, :calfa => 0.28], filter = :inversion,
        tol = Tolerances(nsss = NsssTolerances(acceptance_tol = 1e-10)))

    plot_shock_decomposition(FS2000, dataFS2000_rekey,
        tol = Tolerances(first_order = FirstOrderTolerances(qme = SolverTolerances(acceptance_tol = 1e-12))),
        rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))

    plot_irf(Smets_Wouters_2007, shocks = :em, periods = 10, variables = [:y, :k, :c],
        tol = Tolerances(first_order = FirstOrderTolerances(qme = SolverTolerances(acceptance_tol = 1e-12))))

    plot_irf!(FS2000, shocks = :e_m, shock_size = 10, periods = 5,
        tol = Tolerances(nsss = (xtol = 1e-14,)),
        rename_dictionary = Dict(:e_a => :ea, :e_m => :em, :R => :r, :W => :w))

    cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,8), Variables = [:y], Periods = 1:8)
    cndtns_lvl[1,8] = 1.35

    plot_conditional_forecast(Smets_Wouters_2007, cndtns_lvl,
        initial_state = collect(estims[:,end]),
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),)))

    plot_conditional_forecast!(Smets_Wouters_2007, cndtns_lvl, variables = [:y, :k],
        tol = Tolerances(nsss = NsssTolerances(ftol = 1e-16)))

    plot_solution(FS2000, :k, algorithm = :second_order,
        tol = Tolerances(second_order = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-14))))

    plot_solution!(Smets_Wouters_2007, :pinf, algorithm = :second_order, variables = [:pinf, :y],
        tol = Tolerances(second_order = (sylvester = (acceptance_tol = 1e-14,),)))

end

# multiple models
@testset verbose = true "Gali 2015 ELB plots" begin
    include("../models/Gali_2015_chapter_3_obc.jl")


    Random.seed!(14)
    plot_simulation(Gali_2015_chapter_3_obc, periods = 40, parameters = :R̄ => 1.0, ignore_obc = true)

    Random.seed!(14)
    plot_simulation!(Gali_2015_chapter_3_obc, periods = 40, parameters = :R̄ => 1.0)

    Random.seed!(14)
    plot_simulation!(Gali_2015_chapter_3_obc, periods = 40, parameters = :R̄ => 1.0025)


    Random.seed!(13)
    plot_simulation(Gali_2015_chapter_3_obc, algorithm = :pruned_second_order, 
    # periods = 40, 
    parameters = :R̄ => 1.0, ignore_obc = true)

    Random.seed!(13)
    plot_simulation!(Gali_2015_chapter_3_obc, algorithm = :pruned_second_order, 
    periods = 40, 
    parameters = :R̄ => 1.0)


    plot_irf(Gali_2015_chapter_3_obc, parameters = :R̄ => 1.0)

    plot_irf!(Gali_2015_chapter_3_obc, algorithm = :pruned_second_order, parameters = :R̄ => 1.0)


    plot_irf(Gali_2015_chapter_3_obc, parameters = :σ => 1.0)

    plot_irf!(Gali_2015_chapter_3_obc, parameters = :σ => 1.5)

    plot_irf!(Gali_2015_chapter_3_obc, parameters = :σ => 0.5)


    plot_irf(Gali_2015_chapter_3_obc, parameters = :σ => 1.0)

    plot_irf!(Gali_2015_chapter_3_obc, parameters = :σ => 1.0, generalised_irf = true)

    plot_irf!(Gali_2015_chapter_3_obc, parameters = :σ => 1.0, ignore_obc = true)


    plot_irf(Gali_2015_chapter_3_obc, parameters = :σ => 1.0, algorithm = :pruned_second_order)

    plot_irf!(Gali_2015_chapter_3_obc, parameters = :σ => 1.0, algorithm = :pruned_second_order, ignore_obc = true)

    plot_irf!(Gali_2015_chapter_3_obc, parameters = :σ => 1.0, algorithm = :pruned_second_order, ignore_obc = true, generalised_irf = true)


    # tol-only and tol-varying tests
    Random.seed!(14)
    plot_simulation(Gali_2015_chapter_3_obc, periods = 40, parameters = :R̄ => 1.0,
        tol = Tolerances(first_order = FirstOrderTolerances(lyapunov = SolverTolerances(acceptance_tol = 1e-14))))

    Random.seed!(14)
    plot_simulation!(Gali_2015_chapter_3_obc, periods = 40, parameters = :R̄ => 1.0,
        tol = Tolerances(first_order = (qme = (acceptance_tol = 1e-12,),)))


    plot_irf(Gali_2015_chapter_3_obc,
        tol = Tolerances(nsss = NsssTolerances(acceptance_tol = 1e-10)))

    plot_irf!(Gali_2015_chapter_3_obc,
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),)))


    # combined tol + other argument tests
    Random.seed!(14)
    plot_simulation(Gali_2015_chapter_3_obc, periods = 40, parameters = :R̄ => 1.0025, ignore_obc = true,
        tol = Tolerances(first_order = FirstOrderTolerances(lyapunov = SolverTolerances(acceptance_tol = 1e-14))))

    Random.seed!(13)
    plot_simulation!(Gali_2015_chapter_3_obc, algorithm = :pruned_second_order, periods = 40, parameters = :R̄ => 1.0,
        tol = Tolerances(first_order = (qme = (acceptance_tol = 1e-12,),)))

    plot_irf(Gali_2015_chapter_3_obc, parameters = :σ => 1.5,
        tol = Tolerances(nsss = NsssTolerances(acceptance_tol = 1e-10)))

    plot_irf!(Gali_2015_chapter_3_obc, parameters = :σ => 0.5, algorithm = :pruned_second_order,
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),),
                         second_order = (sylvester = (acceptance_tol = 1e-14,),)))
end

@testset verbose = true "Caldara et al 2012 plots" begin
    include("../models/Caldara_et_al_2012.jl")

    plot_irf(Caldara_et_al_2012, algorithm = :pruned_second_order)

    plot_irf!(Caldara_et_al_2012, algorithm = :second_order)


    plot_irf(Caldara_et_al_2012, algorithm = :pruned_second_order)

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_second_order, generalised_irf = true, generalised_irf_draws = 1000)


    plot_irf(Caldara_et_al_2012, algorithm = :pruned_second_order)

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order)


    plot_irf(Caldara_et_al_2012, algorithm = :second_order)

    plot_irf!(Caldara_et_al_2012, algorithm = :third_order)


    plot_irf(Caldara_et_al_2012, algorithm = :pruned_third_order)

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, generalised_irf = true)


    plot_irf(Caldara_et_al_2012, algorithm = :third_order)

    plot_irf!(Caldara_et_al_2012, algorithm = :third_order, generalised_irf = true)


    plot_irf(Caldara_et_al_2012, algorithm = :pruned_third_order)

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, shock_size = 2)

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, shock_size = 3)


    plot_irf(Caldara_et_al_2012, algorithm = :pruned_third_order, parameters = :ψ => 0.8)

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, parameters = :ψ => 1.5)

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, parameters = :ψ => 2.5)


    plot_irf(Caldara_et_al_2012, algorithm = :pruned_third_order, parameters = [:ψ => 0.5, :ζ => 0.3])

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, parameters = [:ψ => 0.5, :ζ => 0.25])

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, parameters = [:ψ => 0.5, :ζ => 0.35])


    # tol-only and tol-varying tests
    plot_irf(Caldara_et_al_2012, algorithm = :pruned_second_order,
        tol = Tolerances(second_order = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-14))))

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_second_order,
        tol = Tolerances(second_order = (sylvester = (acceptance_tol = 1e-14,), lyapunov = (acceptance_tol = 1e-14,))))

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order,
        tol = Tolerances(third_order = (sylvester = (acceptance_tol = 1e-14,),)))

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order,
        tol = Tolerances(nsss = (xtol = 1e-14,),
                         third_order = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-14),
                                                             lyapunov  = SolverTolerances(acceptance_tol = 1e-14))))


    # combined tol + other argument tests
    plot_irf(Caldara_et_al_2012, algorithm = :pruned_second_order, parameters = :ψ => 0.8,
        tol = Tolerances(second_order = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-14))))

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, shock_size = 2,
        tol = Tolerances(third_order = (sylvester = (acceptance_tol = 1e-14,), lyapunov = (acceptance_tol = 1e-14,))))

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, parameters = [:ψ => 0.5, :ζ => 0.3],
        tol = Tolerances(nsss = (xtol = 1e-14,),
                         third_order = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-14))))

    plot_irf!(Caldara_et_al_2012, algorithm = :pruned_third_order, generalised_irf = true,
        tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),),
                         third_order = (sylvester = (acceptance_tol = 1e-14,),)))
end
