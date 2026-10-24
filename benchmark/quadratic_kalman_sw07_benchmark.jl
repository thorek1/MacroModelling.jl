include("/private/tmp/claude-501/-Users-thorekockerols-GitHub-MacroModelling-jl/c2294a5c-2537-47e2-8f6d-68d07bd438d9/scratchpad/pfenv/qkf_filter.jl")
cd("/Users/thorekockerols/GitHub/nonlinearisties")
include("/Users/thorekockerols/GitHub/nonlinearisties/sw07_common.jl")

m    = SW07_MODEL
obs  = SW07_OBSERVABLES
data = SW07_DATA(obs)
pars = sw07_full_parameters(SW07_INITIAL_FREE_PARAMETERS)
println("model=", m.model_name, "  observables=", obs)
println("data ", size(data), "  algorithm=", SW07_ALGORITHM)

# --- inversion filter at pruned second order (the reference) ---
t0 = time(); inv2 = get_loglikelihood(m, data, pars; algorithm = :pruned_second_order,
                                      filter = :inversion, presample_periods = 4); t_inv = time()-t0
println("\ninversion  pruned_2nd = ", round(inv2, digits=3), "   [", round(t_inv, digits=3), " s]")

# --- quadratic Kalman filter ---
opts = MacroModelling.merge_calculation_options()
MacroModelling.solve!(m, parameters = pars, algorithm = :pruned_second_order, dynamics = true, opts = opts)
_,_,𝐒,_,_ = MacroModelling.get_relevant_steady_state_and_state_update(Val(:pruned_second_order), pars, m, opts = opts)
println("max|S2| = ", round(maximum(abs, Matrix(𝐒[2])), digits=3))
ssn = m.constants.post_complete_parameters.SS_and_pars_names
oi  = convert(Vector{Int}, indexin(obs, ssn))
NSSS = get_steady_state(m, parameters = pars, derivatives = false)
Y = collect(data) .- [NSSS(v) for v in obs]
t0 = time(); sys = build_qkf(m, 𝐒[1], 𝐒[2], oi); t_build = time()-t0
println("augmented dim nz = ", sys.nz, "   [build ", round(t_build, digits=2), " s]")
for mev in (1e-3, 1e-4, 1e-5, 1e-6)
    t0 = time(); q = run_qkf(sys, Y; me_var = mev, presample = 4); t1 = time()-t0
    println("  QKF ME var=", rpad(mev,7), " = ", rpad(round(q, digits=3),12), " [", round(t1, digits=2), " s]")
end
