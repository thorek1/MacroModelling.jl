using Test
using MacroModelling

if VERSION < v"1.13"
    using JET
end

@testset verbose = true "Static checking (JET.jl)" begin
    if VERSION < v"1.13"
        JET.test_package(MacroModelling; target_defined_modules = true, toplevel_logger = nothing)
    end
end
