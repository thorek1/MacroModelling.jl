using MacroModelling
import MacroModelling: get_relevant_steady_state_and_state_update
include(joinpath(@__DIR__, "..", "test", "models", "Caldara_et_al_2012_estim.jl"))
m = Caldara_et_al_2012_estim
res = get_relevant_steady_state_and_state_update(Val(:third_order), m.parameter_values, m; estimation=true)
println(length(res))
for (i,x) in enumerate(res)
    sz = x isa AbstractArray ? string(size(x)) : "-"
    println(i,": ", typeof(x), " size=", sz)
end
