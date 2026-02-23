using MacroModelling, Random, CSV, DataFrames, AxisKeys, Zygote, ForwardDiff, LinearAlgebra

function run_case(f, name)
    print("CASE ", name, " ... ")
    try
        out = f()
        println("PASS", out === nothing ? "" : " | " * string(out))
        return true
    catch err
        println("FAIL | ", sprint(showerror, err))
        return false
    end
end

results = Dict{String,Bool}()

include(joinpath(@__DIR__, "..", "models", "FS2000.jl"))
dat_fs = CSV.read(joinpath(@__DIR__, "..", "test", "data", "FS2000_data.csv"), DataFrame)
data_fs = KeyedArray(permutedims(Matrix(dat_fs)), Variable = Symbol.("log_" .* names(dat_fs)), Time = 1:size(dat_fs,1))
data_fs = log.(data_fs)
obs_fs = sort(Symbol.("log_" .* names(dat_fs)))
data_fs = data_fs(obs_fs, :)
p_fs = copy(FS2000.parameter_values)

results["fs2000_kalman_primal"] = run_case("fs2000_kalman_primal") do
    llh = get_loglikelihood(FS2000, data_fs, p_fs; filter = :kalman)
    "llh=$(llh)"
end
results["fs2000_kalman_fd"] = run_case("fs2000_kalman_fd") do
    g = ForwardDiff.gradient(x -> get_loglikelihood(FS2000, data_fs, x; filter = :kalman), p_fs)
    "grad_len=$(length(g)), norm=$(norm(g))"
end
results["fs2000_kalman_zyg"] = run_case("fs2000_kalman_zyg") do
    g = Zygote.gradient(x -> get_loglikelihood(FS2000, data_fs, x; filter = :kalman), p_fs)[1]
    "grad_len=$(length(g)), norm=$(norm(g))"
end

results["fs2000_inversion_primal"] = run_case("fs2000_inversion_primal") do
    llh = get_loglikelihood(FS2000, data_fs, p_fs; filter = :inversion)
    "llh=$(llh)"
end
results["fs2000_inversion_fd"] = run_case("fs2000_inversion_fd") do
    g = ForwardDiff.gradient(x -> get_loglikelihood(FS2000, data_fs, x; filter = :inversion), p_fs)
    "grad_len=$(length(g)), norm=$(norm(g))"
end
results["fs2000_inversion_zyg"] = run_case("fs2000_inversion_zyg") do
    g = Zygote.gradient(x -> get_loglikelihood(FS2000, data_fs, x; filter = :inversion), p_fs)[1]
    "grad_len=$(length(g)), norm=$(norm(g))"
end

results["fs2000_second_primal"] = run_case("fs2000_second_primal") do
    llh = get_loglikelihood(FS2000, data_fs, p_fs; algorithm = :second_order)
    "llh=$(llh)"
end
results["fs2000_second_zyg"] = run_case("fs2000_second_zyg") do
    g = Zygote.gradient(x -> get_loglikelihood(FS2000, data_fs, x; algorithm = :second_order), p_fs)[1]
    "grad_len=$(length(g)), norm=$(norm(g))"
end

results["fs2000_pruned2_primal"] = run_case("fs2000_pruned2_primal") do
    llh = get_loglikelihood(FS2000, data_fs, p_fs; algorithm = :pruned_second_order)
    "llh=$(llh)"
end
results["fs2000_pruned2_zyg"] = run_case("fs2000_pruned2_zyg") do
    g = Zygote.gradient(x -> get_loglikelihood(FS2000, data_fs, x; algorithm = :pruned_second_order), p_fs)[1]
    "grad_len=$(length(g)), norm=$(norm(g))"
end

dat_sw = CSV.read(joinpath(@__DIR__, "..", "test", "data", "usmodel.csv"), DataFrame)
data_sw = KeyedArray(permutedims(Matrix(dat_sw)), Variable = Symbol.(strip.(names(dat_sw))), Time = 1:size(dat_sw,1))
obs_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]
obs_sw = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data_sw = rekey(data_sw(obs_old, 47:230), :Variable => obs_sw)

function sw07_combined_params(all_params, fixed)
    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = all_params
    ctou, clandaw, cg, curvp, curvw = fixed
    [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]
end

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007_linear.jl"))
fixed_lin = Smets_Wouters_2007_linear.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007_linear.constants.post_complete_parameters.parameters)]
idx_est_lin = indexin([:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa], Smets_Wouters_2007_linear.constants.post_complete_parameters.parameters)
p_est_lin = copy(Smets_Wouters_2007_linear.parameter_values[idx_est_lin])

results["sw07_linear_primal"] = run_case("sw07_linear_primal") do
    llh = get_loglikelihood(Smets_Wouters_2007_linear, data_sw(obs_sw), sw07_combined_params(p_est_lin, fixed_lin); presample_periods = 4, initial_covariance = :diagonal, filter = :kalman)
    "llh=$(llh)"
end
results["sw07_linear_zyg"] = run_case("sw07_linear_zyg") do
    g = Zygote.gradient(x -> get_loglikelihood(Smets_Wouters_2007_linear, data_sw(obs_sw), sw07_combined_params(x, fixed_lin); presample_periods = 4, initial_covariance = :diagonal, filter = :kalman), p_est_lin)[1]
    "grad_len=$(length(g)), norm=$(norm(g))"
end

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))
fixed_nl = Smets_Wouters_2007.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007.constants.post_complete_parameters.parameters)]
idx_est_nl = indexin([:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa], Smets_Wouters_2007.constants.post_complete_parameters.parameters)
p_est_nl = copy(Smets_Wouters_2007.parameter_values[idx_est_nl])

results["sw07_nonlinear_primal"] = run_case("sw07_nonlinear_primal") do
    llh = get_loglikelihood(Smets_Wouters_2007, data_sw(obs_sw), sw07_combined_params(p_est_nl, fixed_nl); presample_periods = 4, initial_covariance = :diagonal, filter = :kalman)
    "llh=$(llh)"
end
results["sw07_nonlinear_zyg"] = run_case("sw07_nonlinear_zyg") do
    g = Zygote.gradient(x -> get_loglikelihood(Smets_Wouters_2007, data_sw(obs_sw), sw07_combined_params(x, fixed_nl); presample_periods = 4, initial_covariance = :diagonal, filter = :kalman), p_est_nl)[1]
    "grad_len=$(length(g)), norm=$(norm(g))"
end

include(joinpath(@__DIR__, "..", "test", "models", "Caldara_et_al_2012_estim.jl"))
dat_us = CSV.read(joinpath(@__DIR__, "..", "test", "data", "usmodel.csv"), DataFrame)
data_us = KeyedArray(permutedims(Matrix(dat_us)), Variable = Symbol.(strip.(names(dat_us))), Time = 1:size(dat_us,1))
data_cal = data_us([:dy], 75:230)
p_cal = copy(Caldara_et_al_2012_estim.parameter_values)

results["caldara_third_primal"] = run_case("caldara_third_primal") do
    llh = get_loglikelihood(Caldara_et_al_2012_estim, data_cal, p_cal; algorithm = :third_order, on_failure_loglikelihood = -Inf)
    "llh=$(llh)"
end
results["caldara_third_zyg"] = run_case("caldara_third_zyg") do
    g = Zygote.gradient(x -> get_loglikelihood(Caldara_et_al_2012_estim, data_cal, x; algorithm = :third_order, on_failure_loglikelihood = -Inf), p_cal)[1]
    "grad_len=$(length(g)), norm=$(norm(g))"
end

results["caldara_pruned3_primal"] = run_case("caldara_pruned3_primal") do
    llh = get_loglikelihood(Caldara_et_al_2012_estim, data_cal, p_cal; algorithm = :pruned_third_order, on_failure_loglikelihood = -Inf)
    "llh=$(llh)"
end
results["caldara_pruned3_zyg"] = run_case("caldara_pruned3_zyg") do
    g = Zygote.gradient(x -> get_loglikelihood(Caldara_et_al_2012_estim, data_cal, x; algorithm = :pruned_third_order, on_failure_loglikelihood = -Inf), p_cal)[1]
    "grad_len=$(length(g)), norm=$(norm(g))"
end

npass = count(values(results))
ntot = length(results)
println("SUMMARY: ", npass, "/", ntot, " cases passed")
for (k, v) in sort(collect(results); by = first)
    println(" - ", k, " => ", v ? "PASS" : "FAIL")
end