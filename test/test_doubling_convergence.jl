# Test that all doubling algorithm paths converge with the SW07 model.
# Uses verbose=true to surface solver convergence messages.
# Exercises: primal solve, get_moments (1st/2nd/3rd order), get_loglikelihood, AD gradient.

using MacroModelling
using LinearAlgebra
using DelimitedFiles, AxisKeys

println("="^80)
println("  SW07 Doubling Convergence Test")
println("="^80)

# ─── Load model ───────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))
m = Smets_Wouters_2007

# ─── 1. Primal solve (first order) ───────────────────────────────────────────
println("\n", "─"^60)
println("1. PRIMAL SOLVE (first_order) with lyapunov_algorithm=:doubling")
println("─"^60)
moments_1st = get_moments(m,
    algorithm = :first_order,
    verbose = true,
    lyapunov_algorithm = :doubling,
    sylvester_algorithm = :doubling)
println("  ✓ get_moments first_order completed")
println("  Keys: ", keys(moments_1st))

# ─── 2. Second order moments ─────────────────────────────────────────────────
println("\n", "─"^60)
println("2. GET_MOMENTS (pruned_second_order)")
println("─"^60)
moments_2nd = get_moments(m,
    algorithm = :pruned_second_order,
    verbose = true,
    lyapunov_algorithm = :doubling,
    sylvester_algorithm = :doubling)
println("  ✓ get_moments pruned_second_order completed")

# ─── 3. Third order moments ──────────────────────────────────────────────────
println("\n", "─"^60)
println("3. GET_MOMENTS (pruned_third_order)")
println("─"^60)
moments_3rd = get_moments(m,
    algorithm = :pruned_third_order,
    verbose = true,
    lyapunov_algorithm = :doubling,
    sylvester_algorithm = :doubling)
println("  ✓ get_moments pruned_third_order completed")

# ─── 4. Standard deviation (wrapper) ─────────────────────────────────────────
println("\n", "─"^60)
println("4. GET_STANDARD_DEVIATION")
println("─"^60)
stds = get_standard_deviation(m,
    algorithm = :first_order,
    verbose = true,
    lyapunov_algorithm = :doubling)
println("  ✓ get_standard_deviation completed")

# ─── 5. Log-likelihood (Kalman filter) ───────────────────────────────────────
println("\n", "─"^60)
println("5. GET_LOGLIKELIHOOD (Kalman filter)")
println("─"^60)

# Load data
dat, header = readdlm(joinpath(@__DIR__, "..", "test", "data", "usmodel.csv"), ',', header = true)
dat = Float64.(dat)
names_csv = vec(Symbol.(strip.(header)))
data_raw = KeyedArray(dat', Variable = names_csv, Time = axes(dat, 1))

observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]
sample_idx = 47:230
data_sub = data_raw(observables_old, sample_idx)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data_sub = rekey(data_sub, :Variable => observables)

llh = get_loglikelihood(m, data_sub, m.parameter_values,
    presample_periods = 4,
    initial_covariance = :theoretical,
    filter = :kalman,
    verbose = true,
    lyapunov_algorithm = :doubling,
    sylvester_algorithm = :doubling)
println("  ✓ get_loglikelihood (kalman) = $llh")
@assert isfinite(llh) "Log-likelihood is not finite!"

# ─── 6. Log-likelihood with inversion filter ─────────────────────────────────
println("\n", "─"^60)
println("6. GET_LOGLIKELIHOOD (inversion filter)")
println("─"^60)

llh_inv = get_loglikelihood(m, data_sub, m.parameter_values,
    presample_periods = 4,
    initial_covariance = :theoretical,
    filter = :inversion,
    verbose = true,
    lyapunov_algorithm = :doubling,
    sylvester_algorithm = :doubling)
println("  ✓ get_loglikelihood (inversion) = $llh_inv")
@assert isfinite(llh_inv) "Log-likelihood (inversion) is not finite!"

# ─── 7. AD gradient of log-likelihood (Zygote) ───────────────────────────────
println("\n", "─"^60)
println("7. AD GRADIENT (Zygote) of get_loglikelihood")
println("─"^60)

import Zygote

function loglik_wrapper(params)
    get_loglikelihood(m, data_sub, params,
        presample_periods = 4,
        initial_covariance = :diagonal,
        filter = :kalman,
        verbose = false,
        lyapunov_algorithm = :doubling,
        sylvester_algorithm = :doubling)
end

# First evaluate primal to warm up
llh_primal = loglik_wrapper(m.parameter_values)
println("  Primal llh = $llh_primal")

# Now compute gradient
grad = Zygote.gradient(loglik_wrapper, m.parameter_values)
grad_vec = grad[1]
n_finite = count(isfinite, grad_vec)
println("  ✓ Zygote gradient computed: $n_finite / $(length(grad_vec)) components finite")
println("  grad norm = ", norm(grad_vec[isfinite.(grad_vec)]))
@assert n_finite > 0 "No finite gradient components!"

# ─── 8. ForwardDiff gradient ─────────────────────────────────────────────────
println("\n", "─"^60)
println("8. AD GRADIENT (ForwardDiff) of get_loglikelihood")
println("─"^60)

import ForwardDiff

grad_fd = ForwardDiff.gradient(loglik_wrapper, m.parameter_values)
n_finite_fd = count(isfinite, grad_fd)
println("  ✓ ForwardDiff gradient computed: $n_finite_fd / $(length(grad_fd)) components finite")
println("  grad norm = ", norm(grad_fd[isfinite.(grad_fd)]))
@assert n_finite_fd > 0 "No finite gradient components (ForwardDiff)!"

# ─── Summary ─────────────────────────────────────────────────────────────────
println("\n", "="^80)
println("  ALL DOUBLING CONVERGENCE TESTS PASSED ✓")
println("="^80)
