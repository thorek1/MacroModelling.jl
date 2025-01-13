# if !test_higher_order
    include("../models/Guerrieri_Iacoviello_2017.jl")
    SSvals = get_SS(Guerrieri_Iacoviello_2017)

    @test isapprox(SSvals([:b,:c,:k,:r],[:Steady_state]),[3.7449, 1.01054, 16.5337, 1.01005],rtol = 1e-4)

    var_dec = get_var_decomp(Guerrieri_Iacoviello_2017)

    @test isapprox(var_dec([:dw,:k,:b],:eps_j) * 100, [0.38, 1.66, 83.52],rtol = 1e-3)
    @test isapprox(var_dec([:y,:r,:c],:eps_r) * 100, [7.35, 31.93, 0.59],rtol = 1e-3)

    write_to_dynare_file(Guerrieri_Iacoviello_2017)
    translate_dynare_file("Guerrieri_Iacoviello_2017.mod")
    include("Guerrieri_Iacoviello_2017.jl")
    # get_solution(Guerrieri_Iacoviello_2017)
    Guerrieri_Iacoviello_2017 = nothing


    include("models/SW07_nonlinear.jl")
    SSvals = get_SS(SW07_nonlinear)

    @test isapprox(SSvals([:robs, :y, :kflex, :ygap],[:Steady_state]),[1.62996, 1.36422, 7.55624, 0],rtol = 1e-4)

    var_dec = get_var_decomp(SW07_nonlinear)

    @test isapprox(var_dec([:y,:yflex,:labobs],:ea) * 100, [10.10, 89.47, 0.53],rtol = 1e-3)
    @test isapprox(var_dec([:y,:r,:c],:epinf) * 100, [36.42, 29.30, 22.62],rtol = 1e-3)

    model = SW07_nonlinear
    
    observables = [:k,:c,:y]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)
    
    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = false), model.parameter_values)
        
    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

    write_to_dynare_file(SW07_nonlinear)
    translate_dynare_file("SW07_nonlinear.mod")
    include("SW07_nonlinear.jl")
    get_solution(SW07_nonlinear)
    SW07_nonlinear = nothing


    include("models/Backus_Kehoe_Kydland_1992.jl")
    SSvals = get_SS(Backus_Kehoe_Kydland_1992)

    @test isapprox(SSvals(["A{F}","K{H}","L{F}","LGM"],["Steady_state"]),[0.606436, 11.0148, 0.696782, 0.278732],rtol = 1e-4)

    var_dec = get_var_decomp(Backus_Kehoe_Kydland_1992)

    @test isapprox(var_dec(["K{F}","Y{H}","Z{F}"],"E{F}") * 100, [51.34, 42.44, 52.59],rtol = 1e-3)
    @test isapprox(var_dec(["K{F}","Y{H}","Z{F}"],"E{H}") * 100, [48.66, 57.56, 47.41],rtol = 1e-3)

    model = Backus_Kehoe_Kydland_1992

    observables = ["C{F}","C{H}"]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
            break
        end
    end
    
    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

    write_to_dynare_file(Backus_Kehoe_Kydland_1992)
    translate_dynare_file("Backus_Kehoe_Kydland_1992.mod")
    include("Backus_Kehoe_Kydland_1992.jl")
    get_solution(Backus_Kehoe_Kydland_1992)
    Backus_Kehoe_Kydland_1992 = nothing


    include("../models/NAWM_EAUS_2008.jl")
    SSvals = get_SS(NAWM_EAUS_2008)

    @test isapprox(SSvals([:EAUS_RER,:EA_Y,:EA_K,:EA_C,:US_Y,:US_K],[:Steady_state]),[0.937577, 3.62701, 33.4238, 2.18955, 3.92449, 33.6712],rtol = 1e-5)

    var_dec = get_var_decomp(NAWM_EAUS_2008)

    @test isapprox(var_dec(:EAUS_RER,[:EA_EPSR,:EA_EPSRP,:US_EPSTAUN]) * 100, [22.15, 54.99, 0.03],rtol = 1e-3)
    @test isapprox(var_dec(:EA_Y,[:EA_EPSR,:EA_EPSZ,:US_EPSZ]) * 100, [50.02, 13.54, 4.35],rtol = 1e-3)
    @test isapprox(var_dec(:US_K,[:EA_EPSRP,:US_EPSR,:US_EPSZ]) * 100, [17.48, 26.83, 27.76],rtol = 1e-3)

    model = NAWM_EAUS_2008

    observables = [:EA_R, :US_R, :EA_PI, :US_PI, :EA_YGAP, :US_YGAP]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), vcat(x,model.parameter_values[11:end]), verbose = true), model.parameter_values[1:10])

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-3),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), vcat(x,model.parameter_values[11:end]), verbose = true), model.parameter_values[1:10])

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-3),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), vcat(x,model.parameter_values[11:end])), model.parameter_values[1:10])
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-2)
            break
        end
    end
    
    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-2)

    write_to_dynare_file(NAWM_EAUS_2008)
    translate_dynare_file("NAWM_EAUS_2008.mod")
    include("NAWM_EAUS_2008.jl")
    get_solution(NAWM_EAUS_2008)
    NAWM_EAUS_2008 = nothing




    include("../models/Baxter_King_1993.jl")
    moments = get_moments(Baxter_King_1993, derivatives = false)
    
    @test isapprox(moments[:non_stochastic_steady_state][end-1:end],[0.334599,5.29504],rtol = 1e-3)
    
    corrr = get_correlation(Baxter_King_1993)
    
    @test isapprox(corrr(:k,:l),0.8553,rtol = 1e-3)
    @test isapprox(corrr(:r,:w),-0.9898,rtol = 1e-3)
    
    model = Baxter_King_1993

    observables = [:k]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

    write_to_dynare_file(Baxter_King_1993)
    translate_dynare_file("Baxter_King_1993.mod")
    include("Baxter_King_1993.jl")
    get_solution(Baxter_King_1993)
    Baxter_King_1993 = nothing
    

    include("../models/Ireland_2004.jl")
    moments = get_moments(Ireland_2004, derivatives = false)

    @test isapprox(moments[:standard_deviation],[0.0709,0.0015,0.0077,0.0153,0.0075,0.0163,0.0062],atol = 1e-4)

    var_dec = get_variance_decomposition(Ireland_2004)

    @test isapprox(var_dec(:π̂,:) * 100, [4.51, 0.91, 87.44, 7.14],rtol = 1e-4)

    model = Ireland_2004

    observables = [:ŷ, :π̂]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

    write_to_dynare_file(Ireland_2004)
    translate_dynare_file("Ireland_2004.mod")
    include("Ireland_2004.jl")
    get_solution(Ireland_2004)
    Ireland_2004 = nothing




    include("../models/QUEST3_2009.jl")
    moments = get_moments(QUEST3_2009, derivatives = false)

    @test isapprox(moments[:non_stochastic_steady_state]([:E_BGYN,:E_LUCYN,:E_LIGSN,:E_VL,:E_LCY],:),[2.4,4.99556,-3.68888,19.0693,-0.538115],rtol = 1e-5)

    var_dec = get_var_decomp(QUEST3_2009)

    @test isapprox(var_dec(:E_BGYN,:) * 100, [3.00, 0.23, 0.54, 0.01, 0.08, 0.38, 0.14, 0.10, 83.60, 3.28, 1.11, 3.36, 0.01, 1.56, 0.17, 0.17, 0.00, 2.24, 0.01], rtol = 1e-3)

    model = QUEST3_2009

    observables = [:outputgap, :inflation, :interest]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)
    
    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1, max_range = 1e-5), x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = false), model.parameter_values)
        
    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-5), x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-5)
            break
        end
    end

    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-5)

    write_to_dynare_file(QUEST3_2009)
    translate_dynare_file("QUEST3_2009.mod") # fix BGADJ1 = 0.001BGADJ2;
    include("QUEST3_2009.jl")
    get_solution(QUEST3_2009)
    QUEST3_2009 = nothing




    include("../models/GNSS_2010.jl")
    moments = get_moments(GNSS_2010, derivatives = false)

    @test isapprox(moments[:non_stochastic_steady_state]([:x,:C,:Y,:D,:BE,:BH,:B],:),exp.([0.182322,0.1305,0.273583,1.04257,0.674026,0.144027,1.13688]),rtol = 1e-5)

    var_dec = get_var_decomp(GNSS_2010)

    @test isapprox(var_dec(:B,:) * 100, [42.97, 19.31, 11.70,  0.36,  4.45,  1.41,  0.70,  0.00,  0.61,  12.54, 2.85, 2.72,  0.38],rtol = 1e-3)

    model = GNSS_2010

    observables = [:C,:Y,:D,:BE]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.forward_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-5)
            break
        end
    end

    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-5)

    write_to_dynare_file(GNSS_2010)
    translate_dynare_file("GNSS_2010.mod")
    include("GNSS_2010.jl")
    get_solution(GNSS_2010)
    GNSS_2010 = nothing


    include("../models/Gali_Monacelli_2005_CITR.jl")
    moments = get_moments(Gali_Monacelli_2005_CITR, derivatives = false)
    
    @test isapprox(moments[:standard_deviation]([:a,:r,:s,:pi],:),[2.2942, 0.4943, 2.7590, 0.3295],atol = 1e-4)
    
    var_dec = get_var_decomp(Gali_Monacelli_2005_CITR)
    
    @test isapprox(var_dec(:pih,:) * 100, [62.06, 37.94],rtol = 1e-3)
    @test isapprox(var_dec(:x,:) * 100, [56.22, 43.78],rtol = 1e-3)
    
    model = Gali_Monacelli_2005_CITR

    observables = [:pi]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

    for i in 1:100        
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences worked after $i iterations")
            @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
            break
        end
    end

    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

    write_to_dynare_file(Gali_Monacelli_2005_CITR)
    translate_dynare_file("Gali_Monacelli_2005_CITR.mod")
    # include("Gali_Monacelli_2005_CITR.jl") # cannot do this because the model contains Gamma
    # get_solution(Gali_Monacelli_2005_CITR)
    Gali_Monacelli_2005_CITR = nothing
    
    
    include("../models/Ascari_Sbordone_2014.jl")
    moments = get_moments(Ascari_Sbordone_2014, derivatives = false)
    
    @test isapprox(log.(moments[:non_stochastic_steady_state]([:i,:y,:psi,:phi],:)),[-4.59512, -1.09861, 1.25138, 1.35674],rtol = 1e-4)
    
    var_dec = get_var_decomp(Ascari_Sbordone_2014)
    
    @test isapprox(var_dec(:w,:) * 100, [2.02, 95.18, 2.80],rtol = 1e-3)
    @test isapprox(var_dec(:y,:) * 100, [0.47, 99.41, 0.12],rtol = 1e-3)
    
    model = Ascari_Sbordone_2014

    observables = [:y,:psi]

    Random.seed!(1)
    simulated_data = simulate(model)

    get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)
    
    # SS(model, parameters = [:alpha => 0.1, :trend_inflation => 1.5, :var_rho => 0.01]) # avoid the NaN error for finitediff in tests

    back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)
    
    # use forward_cdm so that parameter values stay positive. they would return NaN otherwise
    # fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
    
    # if !isfinite(ℒ.norm(fin_grad))
        for i in 1:100        
            local fin_grad = FiniteDifferences.grad(FiniteDifferences.forward_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
            if isfinite(ℒ.norm(fin_grad))
                println("Finite differences worked after $i iterations")
                @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-4)
                break
            end
        end
    # end

    # @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-4)

    write_to_dynare_file(Ascari_Sbordone_2014)
    translate_dynare_file("Ascari_Sbordone_2014.mod")
    include("Ascari_Sbordone_2014.jl")
    get_solution(Ascari_Sbordone_2014)
    Ascari_Sbordone_2014 = nothing
        
# end

include("../models/SGU_2003_debt_premium.jl")
moments = get_moments(SGU_2003_debt_premium, derivatives = false)

@test isapprox(moments[:non_stochastic_steady_state][4:8],[0.7442,exp(0.00739062),exp(-1.07949),exp(1.22309),exp(-3.21888)],rtol = 1e-5)

corrr = get_correlation(SGU_2003_debt_premium)

@test isapprox(corrr(:i,:r),0.0114,rtol = 1e-3)
@test isapprox(corrr(:i,:k),0.4645,rtol = 1e-3)

# if test_higher_order
    mean_2nd = get_mean(SGU_2003_debt_premium, algorithm = :pruned_second_order, derivatives = false)
    autocorr_3rd = get_autocorrelation(SGU_2003_debt_premium, algorithm = :pruned_third_order)

    @test isapprox(mean_2nd(:r),exp(-3.2194),rtol = 1e-3)
    @test isapprox(autocorr_3rd(:c,5),0.4784,rtol = 1e-3)
# end

model = SGU_2003_debt_premium

observables = [:r]

Random.seed!(1)
simulated_data = simulate(model)

get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

# fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

for i in 1:100        
    local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
    if isfinite(ℒ.norm(fin_grad))
        println("Finite differences worked after $i iterations")
        @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
        break
    end
end

# @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

write_to_dynare_file(SGU_2003_debt_premium)
translate_dynare_file("SGU_2003_debt_premium.mod")
include("SGU_2003_debt_premium.jl")
get_solution(SGU_2003_debt_premium)
SGU_2003_debt_premium = nothing



include("../models/JQ_2012_RBC.jl")
moments = get_moments(JQ_2012_RBC, derivatives = false)

@test isapprox(moments[:non_stochastic_steady_state][1:6],[1.01158,3.63583,0.811166,0.114801,0.251989,10.0795],rtol = 1e-5)

var_dec = get_variance_decomposition(JQ_2012_RBC)

@test isapprox(var_dec(:R,:) * 100, [67.43,32.57],rtol = 1e-4)
@test isapprox(var_dec(:k,:) * 100, [17.02,82.98],rtol = 1e-4)
@test isapprox(var_dec(:c,:) * 100, [13.12,86.88],rtol = 1e-4)

# if test_higher_order
    mean_2nd = get_mean(JQ_2012_RBC, algorithm = :pruned_second_order, derivatives = false)
    autocorr_3rd = get_autocorrelation(JQ_2012_RBC, algorithm = :pruned_third_order)

    @test isapprox(mean_2nd(:k),10.0762,rtol = 1e-3)
    @test isapprox(autocorr_3rd(:r,5),0.2927,rtol = 1e-3)
# end

model = JQ_2012_RBC

observables = [:R,:k]

Random.seed!(1)
simulated_data = simulate(model)

get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

# fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

for i in 1:100        
    local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
    if isfinite(ℒ.norm(fin_grad))
        println("Finite differences worked after $i iterations")
        @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
        break
    end
end

# @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

write_to_dynare_file(JQ_2012_RBC)
translate_dynare_file("JQ_2012_RBC.mod")
include("JQ_2012_RBC.jl")
get_solution(JQ_2012_RBC)
JQ_2012_RBC = nothing





include("../models/Ghironi_Melitz_2005.jl")
moments = get_moments(Ghironi_Melitz_2005, derivatives = false)

@test isapprox(moments[:non_stochastic_steady_state]([:Nd,:Nd̄,:Ne,:Nē,:Nx,:Nx̄],:),[7.50695, 7.50695,0.192486, 0.192486, 1.57988, 1.57988],rtol = 1e-5)

var_dec = get_var_decomp(Ghironi_Melitz_2005)

@test isapprox(var_dec(:z̃x,:) * 100, [96.05, 3.95],rtol = 1e-3)
@test isapprox(var_dec(:C,:) * 100, [99.86, 0.14],rtol = 1e-3)
@test isapprox(var_dec(:Nx,:) * 100, [29.27, 70.73],rtol = 1e-3)


# if test_higher_order
    mean_2nd = get_mean(Ghironi_Melitz_2005, algorithm = :pruned_second_order, derivatives = false)
    autocorr_3rd = get_autocorrelation(Ghironi_Melitz_2005, algorithm = :pruned_third_order)

    @test isapprox(mean_2nd(:w),3.1434,rtol = 1e-3)
    @test isapprox(autocorr_3rd(:C,5),0.9178,rtol = 1e-3)
# end

model = Ghironi_Melitz_2005

observables = [:C,:Nx]

Random.seed!(1)
simulated_data = simulate(model)

get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values, verbose = true)

back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

# fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

for i in 1:100        
    local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
    if isfinite(ℒ.norm(fin_grad))
        println("Finite differences worked after $i iterations")
        @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
        break
    end
end

# @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

write_to_dynare_file(Ghironi_Melitz_2005)
translate_dynare_file("Ghironi_Melitz_2005.mod")
include("Ghironi_Melitz_2005.jl")
get_solution(Ghironi_Melitz_2005)
Ghironi_Melitz_2005 = nothing






include("../models/Gali_2015_chapter_3_nonlinear.jl")
moments = get_moments(Gali_2015_chapter_3_nonlinear, derivatives = false)

@test isapprox(moments[:non_stochastic_steady_state]([:C,:N,:M_real,:Y],:),[0.95058, 0.934655, 0.915236, 0.95058],rtol = 1e-5)

var_dec = get_var_decomp(Gali_2015_chapter_3_nonlinear)

@test isapprox(var_dec(:C,:) * 100, [27.53, 0.72,  71.76],rtol = 1e-3)
@test isapprox(var_dec(:R,:) * 100, [15.37, 0.23, 84.40],rtol = 1e-3)
@test isapprox(var_dec(:Y,:) * 100, [27.53, 0.72, 71.76],rtol = 1e-3)

# if test_higher_order
    mean_2nd = get_mean(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order, derivatives = false)

    @test isapprox(mean_2nd(:C),0.9156,rtol = 1e-3)
# end

model = Gali_2015_chapter_3_nonlinear

observables = [:N,:Y]

Random.seed!(1)
simulated_data = simulate(model)

get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values)

back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

# fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

for i in 1:100        
    local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
    if isfinite(ℒ.norm(fin_grad))
        println("Finite differences worked after $i iterations")
        @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
        break
    end
end

# @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

write_to_dynare_file(Gali_2015_chapter_3_nonlinear)
translate_dynare_file("Gali_2015_chapter_3_nonlinear.mod")
include("Gali_2015_chapter_3_nonlinear.jl")
get_solution(Gali_2015_chapter_3_nonlinear)
Gali_2015_chapter_3_nonlinear = nothing




include("../models/Caldara_et_al_2012.jl")
moments = get_moments(Caldara_et_al_2012, derivatives = false)

@test isapprox(moments[:non_stochastic_steady_state]([:Rᵏ,:V,:c,:i],:),[0.00908174, 0.687139, 0.724731, 0.18689],rtol = 1e-5)

@test isapprox(moments[:standard_deviation]([:Rᵏ,:V,:c,:i],:),[0.0022, 0.0061, 0.0566, 0.0528],atol = 1e-4)

# if test_higher_order
    autocorr_3rd = get_autocorrelation(Caldara_et_al_2012, algorithm = :pruned_third_order)

    @test isapprox(autocorr_3rd(:c,5),0.9424,rtol = 1e-3)
# end

model = Caldara_et_al_2012

observables = [:c]

Random.seed!(1)
simulated_data = simulate(model)

get_loglikelihood(model, simulated_data(observables, :, :simulate), model.parameter_values)

back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

# fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

for i in 1:100        
    local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
    if isfinite(ℒ.norm(fin_grad))
        println("Finite differences worked after $i iterations")
        @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-3)
        break
    end
end

# @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-3)

write_to_dynare_file(Caldara_et_al_2012)
translate_dynare_file("Caldara_et_al_2012.mod")
include("Caldara_et_al_2012.jl")
get_solution(Caldara_et_al_2012)
Caldara_et_al_2012 = nothing





include("../models/Aguiar_Gopinath_2007.jl")
moments = get_moments(Aguiar_Gopinath_2007, derivatives = false)

@test isapprox(moments[:non_stochastic_steady_state]([:b,:c,:u,:ul],:),[0.0645177, 0.496156, -1.66644, -1.597],rtol = 1e-5)

var_dec = get_var_decomp(Aguiar_Gopinath_2007)

@test isapprox(var_dec(:c,:) * 100, [0.02, 99.98],rtol = 1e-3)
@test isapprox(var_dec(:i_y,:) * 100, [0.31, 99.69],rtol = 1e-3)

# if test_higher_order
    mean_2nd = get_mean(Aguiar_Gopinath_2007, algorithm = :pruned_second_order, derivatives = false)
    autocorr_3rd = get_autocorrelation(Aguiar_Gopinath_2007, algorithm = :pruned_third_order)
    
    @test isapprox(mean_2nd(:k),2.5946,rtol = 1e-3)
    @test isapprox(autocorr_3rd(:c,5),0.9299,rtol = 1e-3)
# end

model = Aguiar_Gopinath_2007

observables = [:c,:u]

Random.seed!(1)
simulated_data = simulate(model)

get_loglikelihood(model, simulated_data(observables, :, :simulate), verbose = true, model.parameter_values)

back_grad = Zygote.gradient(x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

# fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x, verbose = true), model.parameter_values)

for i in 1:100        
    local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1, max_range = 1e-4),x-> get_loglikelihood(model, simulated_data(observables, :, :simulate), x), model.parameter_values)
    if isfinite(ℒ.norm(fin_grad))
        println("Finite differences worked after $i iterations")
        @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)
        break
    end
end

# @test isapprox(back_grad[1], fin_grad[1], rtol = 1e-6)

write_to_dynare_file(Aguiar_Gopinath_2007)
translate_dynare_file("Aguiar_Gopinath_2007.mod")
include("Aguiar_Gopinath_2007.jl")
get_solution(Aguiar_Gopinath_2007)
Aguiar_Gopinath_2007 = nothing
