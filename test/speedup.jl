using Requires
@time_imports using MacroModelling
@time using MacroModelling
# 22 seconds with precompilation @model @parameters get_SS get_solution
# 22 seconds with precompilation @model @parameters get_SS
# 20 seconds with precompilation @model @parameters
# 19 seconds without precompilation


# using SnoopCompile

@model RBC begin
    1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
    z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
end 

@parameters RBC silent = true begin
    δ = 0.02
    α = 0.5
end

get_SS(RBC)

@time get_solution(RBC)
# with precompilation 2.224324 seconds (1.10 M allocations: 46.159 MiB, 0.67% gc time, 99.92% compilation time)
# 2.505074 seconds (5.66 M allocations: 264.101 MiB, 1.49% gc time, 99.93% compilation time)

# @time get_SS(RBC)
# with precomp 1.596658 seconds (9.10 M allocations: 499.542 MiB, 1.91% gc time, 99.63% compilation time: 68% of which was recompilation)
# 5.689448 seconds (48.08 M allocations: 2.510 GiB, 7.24% gc time, 99.91% compilation time)

@profview get_SS(RBC)



@time begin   
    @model RBC begin
        1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
        z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
    end 
end
# with precomp 0.472191 seconds (2.88 M allocations: 157.095 MiB, 3.84% gc time, 99.48% compilation time)
# 0.732525 seconds (4.49 M allocations: 248.918 MiB, 5.24% gc time, 99.67% compilation time)

@profview begin
    @model RBC begin
        1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
        z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
    end
end


tinf = @snoopi_deep begin
    @model RBC begin
        1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
        z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
    end 
end

staleinstances(tinf)

# with precomp 3.083087 seconds (4.09 M allocations: 212.054 MiB, 0.84% gc time, 86.85% compilation time)
# 5.451956 seconds (23.42 M allocations: 1.151 GiB, 2.63% gc time, 92.39% compilation time: 1% of which was recompilation)
@time begin
    @parameters RBC silent = true begin
        δ = 0.02
        α = 0.5
    end
end

tinf2 = @snoopi_deep begin
    @parameters RBC silent = true begin
        δ = 0.02
        α = 0.5
    end
end

@profview begin
    @parameters RBC silent = true begin
        δ = 0.02
        α = 0.5
    end
end

# @model RBC begin
#     1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
#     c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
#     z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
# end

# @parameters RBC silent = true begin
#     δ = 0.02
#     α = 0.5
# end
using SnoopCompile
using SnoopCompileCore
invalidations = @snoopr begin
    using MacroModelling
end

trees = SnoopCompile.invalidation_trees(invalidations);

@show length(SnoopCompile.uinvalidated(invalidations)) # show total invalidations

show(trees[end-3])

# Count number of children (number of invalidations per invalidated method)
n_invalidations = map(trees) do methinvs
    SnoopCompile.countchildren(methinvs)
end

tinf = @snoopi_deep begin
    @model RBC begin
        1  /  c[0] = (0.95 /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
        z[0] = 0.2 * z[-1] + 0.01 * eps_z[x]
    end 
end

staleinstances(tinf)
itrigs = inference_triggers(tinf)


tinf2 = @snoopi_deep begin
    @parameters RBC silent = true begin
        δ = 0.02
        α = 0.5
    end
end

staleinstances(tinf2)
itrigs = inference_triggers(tinf2)