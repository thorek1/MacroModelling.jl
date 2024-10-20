using MacroModelling
using StatsPlots

include("../models/Smets_Wouters_2007.jl")

# include("../models/NAWM_EAUS_2008.jl")

# get_solution(NAWM_EAUS_2008,verbose = true)

get_solution(Smets_Wouters_2007)#, verbose = true)

plot_solution(Smets_Wouters_2007, :r, 
                variables = [:pinfobs, 
                            # :robs, 
                            :ygap], 
                # verbose = true, 
                # parameters =[:curvp => 25 , :curvw => 25], 
                σ = 5,
                plots_per_page = 3,
                save_plots=true,
                algorithm = [#:first_order, 
                :pruned_second_order, :pruned_third_order])

plot_solution(Smets_Wouters_2007, :y, 
                variables = [:pinfobs, 
                            # :robs, 
                            :ygap], 
                # verbose = true, 
                # parameters =[:curvp => 10 , :curvw => 10], 
                # σ = 3,
                algorithm = [:first_order, :pruned_second_order, :pruned_third_order])


vars = get_variables(Smets_Wouters_2007)
states = get_state_variables(Smets_Wouters_2007)
states|>println

output_gap = [Float64[], Float64[], Float64[]]
inflation = [Float64[], Float64[], Float64[]]

n = 15
shock_size_range = range(-5,5,n)

for i in shock_size_range
    irfs = get_irf(Smets_Wouters_2007, 
                    shocks = :epinf,
                    # algorithm = :pruned_second_order, 
                    shock_size = i,
                    # levels = false,
                    generalised_irf = true)([:pinfobs,:ygap],:,:epinf)

    push!(output_gap[1], i >= 0 ? minimum(irfs(:ygap,:,:)) : maximum(irfs(:ygap,:,:)))
    push!(inflation[1], i < 0 ? minimum(irfs(:pinfobs,:,:)) : maximum(irfs(:pinfobs,:,:)))

    irfs = get_irf(Smets_Wouters_2007, 
                    shocks = :epinf,
                    algorithm = :pruned_second_order, 
                    shock_size = i,
                    # levels = false,
                    generalised_irf = true)([:pinfobs,:ygap],:,:epinf)

    push!(output_gap[2], i >= 0 ? minimum(irfs(:ygap,:,:)) : maximum(irfs(:ygap,:,:)))
    push!(inflation[2], i < 0 ? minimum(irfs(:pinfobs,:,:)) : maximum(irfs(:pinfobs,:,:)))


    irfs = get_irf(Smets_Wouters_2007, 
                    shocks = :epinf,
                    algorithm = :pruned_third_order, 
                    shock_size = i,
                    # levels = false,
                    generalised_irf = true)([:pinfobs,:ygap],:,:epinf)

    push!(output_gap[3], i >= 0 ? minimum(irfs(:ygap,:,:)) : maximum(irfs(:ygap,:,:)))
    push!(inflation[3], i < 0 ? minimum(irfs(:pinfobs,:,:)) : maximum(irfs(:pinfobs,:,:)))

end

p1 = plot(shock_size_range, output_gap[1] .- output_gap[1][n÷2+n%2], label = "1st order", title = "Output Gap")
plot!(p1,shock_size_range, output_gap[2] .- output_gap[2][n÷2+n%2], label = "pruned 2nd order", title = "Output Gap")
plot!(p1,shock_size_range, output_gap[3] .- output_gap[3][n÷2+n%2], label = "pruned 3rd order", title = "Output Gap")

p2 = plot(shock_size_range, inflation[1] .- inflation[1][n÷2+n%2], label = "1st order", title = "Inflation")
plot!(p2,shock_size_range, inflation[2] .- inflation[2][n÷2+n%2], label = "pruned 2nd order", title = "Inflation")
plot!(p2,shock_size_range, inflation[3] .- inflation[3][n÷2+n%2], label = "pruned 3rd order", title = "Inflation")

plot(p1, p2,layout = (2))

savefig("cost_push_shock.png")



output_gap = [Float64[], Float64[], Float64[]]
inflation = [Float64[], Float64[], Float64[]]

shock = :em
n = 15
shock_size_range = range(-5,5,n)

for i in shock_size_range
    irfs = get_irf(Smets_Wouters_2007, 
                    shocks = shock,
                    # algorithm = :pruned_second_order, 
                    shock_size = i,
                    # levels = false,
                    generalised_irf = true)([:pinfobs,:ygap],:,shock)

    push!(output_gap[1], i >= 0 ? minimum(irfs(:ygap,:,:)) : maximum(irfs(:ygap,:,:)))
    push!(inflation[1], i < 0 ? minimum(irfs(:pinfobs,:,:)) : maximum(irfs(:pinfobs,:,:)))

    irfs = get_irf(Smets_Wouters_2007, 
                    shocks = shock,
                    algorithm = :pruned_second_order, 
                    shock_size = i,
                    # levels = false,
                    generalised_irf = true)([:pinfobs,:ygap],:,shock)

    push!(output_gap[2], i >= 0 ? minimum(irfs(:ygap,:,:)) : maximum(irfs(:ygap,:,:)))
    push!(inflation[2], i < 0 ? minimum(irfs(:pinfobs,:,:)) : maximum(irfs(:pinfobs,:,:)))


    irfs = get_irf(Smets_Wouters_2007, 
                    shocks = shock,
                    algorithm = :pruned_third_order, 
                    shock_size = i,
                    # levels = false,
                    generalised_irf = true)([:pinfobs,:ygap],:,shock)

    push!(output_gap[3], i >= 0 ? minimum(irfs(:ygap,:,:)) : maximum(irfs(:ygap,:,:)))
    push!(inflation[3], i < 0 ? minimum(irfs(:pinfobs,:,:)) : maximum(irfs(:pinfobs,:,:)))

end

p1 = plot(shock_size_range, output_gap[1] .- output_gap[1][n÷2+n%2], label = "1st order", title = "Output Gap")
plot!(p1,shock_size_range, output_gap[2] .- output_gap[2][n÷2+n%2], label = "pruned 2nd order", title = "Output Gap")
plot!(p1,shock_size_range, output_gap[3] .- output_gap[3][n÷2+n%2], label = "pruned 3rd order", title = "Output Gap")

p2 = plot(shock_size_range, inflation[1] .- inflation[1][n÷2+n%2], label = "1st order", title = "Inflation")
plot!(p2,shock_size_range, inflation[2] .- inflation[2][n÷2+n%2], label = "pruned 2nd order", title = "Inflation")
plot!(p2,shock_size_range, inflation[3] .- inflation[3][n÷2+n%2], label = "pruned 3rd order", title = "Inflation")

plot(p1, p2,layout = (2))

savefig("monetary_policy_shock.png")

