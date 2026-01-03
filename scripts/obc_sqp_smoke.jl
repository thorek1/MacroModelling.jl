include(joinpath(@__DIR__, "..", "src", "MacroModelling.jl"))
using .MacroModelling

include(joinpath(@__DIR__, "..", "models", "Gali_2015_chapter_3_obc.jl"))

function main()
    m = Gali_2015_chapter_3_obc

    MacroModelling.solve!(m; dynamics = true, algorithm = :first_order, obc = true)

    state_update, _ = MacroModelling.parse_algorithm_to_state_update(:first_order, m, true)
    reference_steady_state, _, _ = MacroModelling.get_relevant_steady_states(m, :first_order)
    initial_state = zeros(m.timings.nVars)

    obc_shock_idx = contains.(string.(m.timings.exo), "ᵒᵇᶜ")
    periods_per_shock = m.max_obc_horizon + 1
    num_shocks = sum(obc_shock_idx) ÷ periods_per_shock

    present_shocks = zeros(m.timings.nExo)
    p = (initial_state, state_update, reference_steady_state, m, :first_order, m.max_obc_horizon, present_shocks)

    x0 = zeros(num_shocks * periods_per_shock)

    shock_idx = findfirst(==(:eps_a), m.timings.exo)
    shock_idx === nothing && (shock_idx = findfirst(==(:eps_z), m.timings.exo))
    shock_idx === nothing && (shock_idx = findfirst(==(:eps_nu), m.timings.exo))
    if shock_idx === nothing
        non_obc_idx = findall(x -> !occursin("ᵒᵇᶜ", x), string.(m.timings.exo))
        shock_idx = first(non_obc_idx)
    end
    @assert shock_idx !== nothing

    max_viol = 0.0
    for scale in (-1.0, -2.5, -5.0, -10.0)
        fill!(present_shocks, 0.0)
        present_shocks[shock_idx] = scale
        viol = m.obc_violation_function(x0, p)
        max_viol = maximum(viol)
        if max_viol > 0.0
            break
        end
    end

    @assert max_viol > 0.0

    x, solved = MacroModelling.obc_sqp_solve(x0, p; max_iter = 200, tol = 1e-6)

    viol = m.obc_violation_function(x, p)
    max_viol = maximum(viol)

    @assert solved
    @assert max_viol <= 1e-6

    println("OBC SQP solved: ", solved, " max_viol: ", max_viol)
end

main()
