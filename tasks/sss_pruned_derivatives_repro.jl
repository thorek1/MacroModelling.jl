using MacroModelling
using LinearAlgebra
using SparseArrays
import ChainRulesCore: NoTangent, rrule

include(joinpath(@__DIR__, "..", "test", "models", "RBC_CME_calibration_equations_and_parameter_definitions.jl"))

function summarize_diff(name, finite, analytic)
    diff = finite .- analytic
    max_abs, idx = findmax(abs.(diff))
    row, col = Tuple(idx)
    println("=== ", name, " ===")
    println("size: ", size(finite))
    println("max abs diff: ", max_abs, " at (row=", row, ", col=", col, ")")
    println("finite value:   ", finite[row, col])
    println("analytic value: ", analytic[row, col])
    println("finite first row:   ", finite[1, :])
    println("analytic first row: ", analytic[1, :])
    println()
end

function jacobian_central_4th(f, x::Vector{Float64})
    y0 = f(x)
    J = zeros(length(y0), length(x))
    for j in eachindex(x)
        h = max(cbrt(eps(Float64)) * max(abs(x[j]), 1.0), 1e-6)
        xpp = copy(x)
        xp = copy(x)
        xm = copy(x)
        xmm = copy(x)
        xpp[j] += 2h
        xp[j] += h
        xm[j] -= h
        xmm[j] -= 2h
        J[:, j] .= (-f(xpp) .+ 8f(xp) .- 8f(xm) .+ f(xmm)) ./ (12h)
    end
    return J
end

parameters = copy(m.parameter_values)

SSSdiff2 = Matrix(get_SSS(m))
SSSdiff2p = Matrix(get_SSS(m, algorithm = :pruned_second_order))
SSSdiff3 = Matrix(get_SSS(m, algorithm = :third_order))
SSSdiff3p = Matrix(get_SSS(m, algorithm = :pruned_third_order))

SSS2finitediff = jacobian_central_4th(
    x -> collect(get_SSS(m; parameters = x, derivatives = false)),
    parameters,
)

SSS2pfinitediff = jacobian_central_4th(
    x -> collect(get_SSS(m; parameters = x, derivatives = false, algorithm = :pruned_second_order)),
    parameters,
)

SSS3finitediff = jacobian_central_4th(
    x -> collect(get_SSS(m; parameters = x, derivatives = false, algorithm = :third_order)),
    parameters,
)

SSS3pfinitediff = jacobian_central_4th(
    x -> collect(get_SSS(m; parameters = x, derivatives = false, algorithm = :pruned_third_order)),
    parameters,
)

summarize_diff("second_order", SSS2finitediff, SSSdiff2[:, 2:end])
summarize_diff("pruned_second_order", SSS2pfinitediff, SSSdiff2p[:, 2:end])
summarize_diff("third_order", SSS3finitediff, SSSdiff3[:, 2:end])
summarize_diff("pruned_third_order", SSS3pfinitediff, SSSdiff3p[:, 2:end])

common, common_pb = rrule(MacroModelling._prepare_stochastic_steady_state_base_terms, parameters, m)
SSSstates = common[9]
SSSstates_finitediff = jacobian_central_4th(
    x -> MacroModelling._prepare_stochastic_steady_state_base_terms(x, m)[9],
    parameters,
)
SSSstates_analytic = zeros(length(SSSstates), length(parameters))
for i in eachindex(SSSstates)
    seed = zeros(length(SSSstates))
    seed[i] = 1.0
    SSSstates_analytic[i, :] .= common_pb((
        NoTangent(),
        zeros(length(common[2])),
        zeros(length(common[3])),
        NoTangent(),
        zeros(size(common[5])),
        spzeros(size(common[6])...),
        zeros(size(common[7])),
        zeros(size(common[8])),
        seed,
        NoTangent(),
    ))[2]
end
summarize_diff("common_SSSstates", SSSstates_finitediff, SSSstates_analytic)
