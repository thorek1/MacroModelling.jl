using Revise
using MacroModelling
using BenchmarkTools
using DelimitedFiles
using AxisKeys

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))

model = Smets_Wouters_2007

# Same SW07 data preparation used in test/test_sw07_estimation.jl
raw_data, raw_header = readdlm(joinpath(@__DIR__, "..", "test", "data", "usmodel.csv"), ',', Float64, '\n'; header = true)
variable_names = Symbol.(strip.(vec(raw_header)))
data = KeyedArray(raw_data', Variable = variable_names, Time = 1:size(raw_data, 1))

observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]
sample_idx = 47:230
data = data(observables_old, sample_idx)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data = rekey(data, :Variable => observables)

llh_data = data(observables)
known_parameters = copy(model.parameter_values)
new_parameters = known_parameters .+ 0.001

function clear_nsss_cache!(m)
    while length(m.caches.solver_cache) > 1
        pop!(m.caches.solver_cache)
    end
    return nothing
end

clear_nsss_cache!(model)

function evaluate_llh(m, data, parameters)
    return get_loglikelihood(
        m,
        data,
        parameters;
        presample_periods = 4,
        initial_covariance = :diagonal,
        quadratic_matrix_equation_algorithm = :doubling,
        filter = :kalman,
    )
end

function setup_known_to_new_transition!(m, data, known_params)
    clear_nsss_cache!(m)
    evaluate_llh(m, data, known_params)
    return nothing
end

# Warm-up compile and ensure LLHs are finite before benchmarking.
llh_known = evaluate_llh(
    model,
    llh_data,
    known_parameters,
)
llh_new = evaluate_llh(
    model,
    llh_data,
    new_parameters,
)
println("Warm-up known LLH: ", llh_known)
println("Warm-up new LLH: ", llh_new)

trial = @benchmark evaluate_llh(
    $model,
    $llh_data,
    $new_parameters,
) setup = setup_known_to_new_transition!($model, $llh_data, $known_parameters)

@profview_allocs for _ in 1:10000
    setup_known_to_new_transition!(model, llh_data, known_parameters)
    evaluate_llh(model, llh_data, new_parameters)
end

@profview for _ in 1:1000
    setup_known_to_new_transition!(model, llh_data, known_parameters)
    evaluate_llh(model, llh_data, new_parameters)
end

println(trial)
println("Minimum time: ", minimum(trial).time, " ns")
println("Minimum memory: ", minimum(trial).memory, " bytes")
