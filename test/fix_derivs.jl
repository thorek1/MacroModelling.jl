using Revise
# using Pkg; Pkg.activate(".");
using MacroModelling
using FiniteDifferences
using Zygote
using ForwardDiff
import MacroModelling: solve_sylvester_equation, solve_lyapunov_equation
using Test
using LinearAlgebra

# test on large problem (green premium), 
# propagate error messages (solved)

include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")

parameters = copy(m.parameter_values)

alg = :gmres
alg2 = :doubling

μdiff = get_mean(m,parameters = parameters)

μdiff2 = get_mean(m,parameters = parameters, algorithm = :pruned_second_order, sylvester_algorithm = alg)

σdiff = get_std(m,parameters = parameters)

σdiff2 = get_std(m,parameters = parameters, algorithm = :pruned_second_order, sylvester_algorithm = alg)

σdiff3 = get_std(m,parameters = parameters, algorithm = :pruned_third_order, sylvester_algorithm = alg, lyapunov_algorithm = alg2)


μfinitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_mean(m; parameters = x, derivatives = false)), 
        parameters)[1]

@test isapprox(μfinitediff, μdiff[:,2:end], rtol = 1e-6)


μ2finitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_mean(m; parameters = x, derivatives = false, algorithm = :pruned_second_order, sylvester_algorithm = alg2)), 
        parameters)[1]

@test isapprox(μ2finitediff, μdiff2[:,2:end], rtol = 1e-6)

σfinitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_std(m; parameters = x, derivatives = false)), 
        parameters)[1]

@test isapprox(σfinitediff, σdiff[:,2:end], rtol = 1e-6)

norm(σfinitediff-σdiff[:,2:end]) / max(norm(σfinitediff), norm(σdiff[:,2:end]))

σfinitediff - σdiff[:,2:end]

σ2finitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_std(m; parameters = x, derivatives = false, algorithm = :pruned_second_order, sylvester_algorithm = alg2)), 
        parameters)[1]

@test isapprox(σ2finitediff, σdiff2[:,2:end], rtol = 1e-6)

norm(σ2finitediff-σdiff2[:,2:end]) / max(norm(σ2finitediff), norm(σdiff2[:,2:end]))


σ3finitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_std(m; parameters = x, derivatives = false, algorithm = :pruned_third_order, sylvester_algorithm = alg, lyapunov_algorithm = alg2)), 
        parameters)[1]

@test isapprox(σ3finitediff, σdiff3[:,2:end], rtol = 1e-6)

norm(σ3finitediff-σdiff3[:,2:end]) / max(norm(σ3finitediff), norm(σdiff3[:,2:end]))






μfinitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_mean(m; parameters = x, derivatives = false)), 
        parameters)[1]

isapprox(μfinitediff, μdiff[:,2:end], rtol = 1e-6)


μ2finitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_mean(m; parameters = x, derivatives = false, algorithm = :pruned_second_order, sylvester_algorithm = alg)), 
        parameters)[1]

isapprox(μ2finitediff, μdiff2[:,2:end], rtol = 1e-6)
μ2finitediff - μdiff2[:,2:end]

norm(μ2finitediff-μdiff2[:,2:end]) / max(norm(μ2finitediff), norm(μdiff2[:,2:end]))


σfinitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_std(m; parameters = x, derivatives = false)), 
        parameters)[1]

isapprox(σfinitediff, σdiff[:,2:end], rtol = 1e-6)
σfinitediff - σdiff[:,2:end]

norm(σfinitediff-σdiff[:,2:end]) / max(norm(σfinitediff), norm(σdiff[:,2:end]))




σ2finitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
        x -> collect(get_std(m; parameters = x, derivatives = false, algorithm = :pruned_second_order)), 
        parameters)[1]

isapprox(σ2finitediff, σdiff2[:,2:end], rtol = 1e-6)
# σ2finitediff - σdiff2[:,2:end]

norm(σ2finitediff-σdiff2[:,2:end]) / max(norm(σ2finitediff), norm(σdiff2[:,2:end]))



σ3finitediff = FiniteDifferences.jacobian(central_fdm(3,1), 
        x -> collect(get_std(m; parameters = x, derivatives = false, algorithm = :pruned_third_order)), 
        parameters)[1]

isapprox(σ3finitediff, σdiff3[:,2:end], rtol = 1e-6)

# σ3finitediff ./ σdiff3[:,2:end]

norm(σ3finitediff-σdiff3[:,2:end]) / max(norm(σ3finitediff), norm(σdiff3[:,2:end]))



using LinearAlgebra

A = randn(10,5)

inv(svd(A))
inv(qr(A))
aa = svd(A)
rank(A)
rank(aa)