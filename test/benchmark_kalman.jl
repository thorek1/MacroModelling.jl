using ForwardDiff, MacroModelling, BenchmarkTools, Zygote

include("models/SW07.jl")

data = simulate(m, levels = true, periods = 200)[:,:,1]

observables = [:c,:k,:y,:r,:inve,:pinf]

calculate_kalman_filter_loglikelihood(m,data(observables),observables)

forw_grad = ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x),Float64.(m.parameter_values))

rev_grad = Zygote.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x),Float64.(m.parameter_values))[1]


@benchmark calculate_kalman_filter_loglikelihood(m,data(observables),observables)

@benchmark ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x),Float64.(m.parameter_values))

@benchmark Zygote.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x),Float64.(m.parameter_values))


@profview kk = calculate_kalman_filter_loglikelihood(m,data(observables),observables)
@profview ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x),Float64.(m.parameter_values))

@profview Zygote.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x),Float64.(m.parameter_values))

@profview for i in 1:50 Zygote.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x),Float64.(m.parameter_values)) end

@time ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x),Float64.(m.parameter_values))

using Octavian