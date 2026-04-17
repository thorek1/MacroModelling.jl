# using Revise
test_set = ENV["TEST_SET"]
using Preferences: set_preferences!
set_preferences!("MacroModelling", "dispatch_doctor_mode" => test_set in ["estimate_sw07", "estimation", "1st_order_inversion_estimation", "pruned_2nd_order_estimation", "2nd_order_estimation", "pruned_3rd_order_estimation", "3rd_order_estimation", "estimation_pigeons", "1st_order_inversion_estimation_pigeons", "2nd_order_estimation_pigeons", "pruned_2nd_order_estimation_pigeons", "3rd_order_estimation_pigeons", "pruned_3rd_order_estimation_pigeons", "gali_pruned_2nd_order_estimation", "rrule_robustness"
] ? "disable" : "error")
set_preferences!("MacroModelling", "dispatch_doctor_union_limit" => 4)

println("Running test set: $test_set")
println("Threads used: ", Threads.nthreads())

if test_set == "jet"
    include("test_jet.jl")
elseif test_set == "estimate_sw07"
    include("test_sw07_estimation.jl")
elseif test_set == "estimation"
    include("test_estimation.jl")
elseif test_set == "1st_order_inversion_estimation"
    include("test_1st_order_inversion_filter_estimation.jl")
elseif test_set == "2nd_order_estimation"
    include("test_2nd_order_estimation.jl")
elseif test_set == "pruned_2nd_order_estimation"
    include("test_pruned_2nd_order_estimation.jl")
elseif test_set == "3rd_order_estimation"
    include("test_3rd_order_estimation.jl")
elseif test_set == "pruned_3rd_order_estimation"
    include("test_pruned_3rd_order_estimation.jl")
elseif test_set == "estimation_pigeons"
    include("test_estimation_pigeons.jl")
elseif test_set == "1st_order_inversion_estimation_pigeons"
    include("test_1st_order_inversion_filter_estimation_pigeons.jl")
elseif test_set == "2nd_order_estimation_pigeons"
    include("test_2nd_order_estimation_pigeons.jl")
elseif test_set == "pruned_2nd_order_estimation_pigeons"
    include("test_pruned_2nd_order_estimation_pigeons.jl")
elseif test_set == "3rd_order_estimation_pigeons"
    include("test_3rd_order_estimation_pigeons.jl")
elseif test_set == "pruned_3rd_order_estimation_pigeons"
    include("test_pruned_3rd_order_estimation_pigeons.jl")
elseif test_set == "plots_1"
    include("test_plots_1.jl")
elseif test_set == "plots_2"
    include("test_plots_2.jl")
elseif test_set == "plots_3"
    include("test_plots_3.jl")
elseif test_set == "plots_4"
    include("test_plots_4.jl")
elseif test_set == "plots_5"
    include("test_plots_5.jl")
elseif test_set == "higher_order_1"
    include("test_higher_order_1.jl")
elseif test_set == "higher_order_2"
    include("test_higher_order_2.jl")
elseif test_set == "higher_order_3"
    include("test_higher_order_3.jl")
elseif test_set == "basic"
    include("test_basic.jl")
elseif test_set == "gali_pruned_2nd_order_estimation"
    include("test_gali_pruned_2nd_order_estimation.jl")
elseif test_set == "rrule_robustness"
    include("test_rrule_robustness.jl")
elseif test_set == "update_equations"
    include("test_update_equations.jl")
end
