using MacroModelling;

@testset "Model without shocks" begin
    @model m begin
        K[0] = (1 - δ) * K[-1] + I[0]
        Z[0] = (1 - ρ) * μ + ρ * Z[-1] 
        I[1]  = ((ρ + δ - Z[0])/(1 - δ))  + ((1 + ρ)/(1 - δ)) * I[0]
    end

    @parameters m begin
        ρ = 0.05
        δ = 0.10
        μ = .17
        σ = .2
    end

    m_ss = get_steady_state(m)
    @test isapprox(m_ss(:,:Steady_state),[1/7.5,1/.75,.17],rtol = eps(Float32))

    m_sol = get_solution(m) 
    @test isapprox(m_sol(:,:K),[1/.75,.9,.04975124378109454],rtol = eps(Float32))
end

get_irf(m, initial_state = init)

plot_irf(m, initial_state = init, shocks = :none, save_plots = true, save_plots_path = "~/Downloads", save_plots_format = :png)

plot(m, initial_state = init)
m.timings.nExo




using MacroModelling;

@model m begin
    Z[0] = (1 - ρ) * μ + ρ * Z[-1]
    I[1]  = ((ρ + δ - Z[0])/(1 - δ))  + ((1 + ρ)/(1 - δ)) * I[0]
end
# Model: m
# Variables: 2
# Shocks: 0
# Parameters: 3
# Auxiliary variables: 0

@parameters m begin
    ρ = 0.05
    δ = 0.10
    μ = .17
    σ = .2
end
m_ss = get_steady_state(m)
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Variables_and_calibrated_parameters ∈ 2-element Vector{Symbol}
# →   Steady_state_and_∂steady_state∂parameter ∈ 4-element Vector{Symbol}
# And data, 2×4 Matrix{Float64}:
#         (:Steady_state)  (:ρ)      (:δ)      (:μ)
#   (:I)   0.133333        -7.55556  -7.55556   6.66667
#   (:Z)   0.17             0.0       0.0       1.0

m.SS_solve_func
# RuntimeGeneratedFunction(#=in MacroModelling=#, #=using MacroModelling=#, :((parameters, initial_guess, 𝓂)->begin
# 
# 
#           ρ = parameters[1]
#           δ = parameters[2]
#           μ = parameters[3]
# 
#           Z = μ
#           I = ((Z - δ) - ρ) / (δ + ρ)
#           SS_init_guess = [I, Z]
#           𝓂.SS_init_guess = if typeof(SS_init_guess) == Vector{Float64}
#                   SS_init_guess
#               else
#                   ℱ.value.(SS_init_guess)
#               end
#           return ComponentVector([I, Z], Axis([sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]))
#       end))

m_sol = get_solution(m) 
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Steady_state__States__Shocks ∈ 2-element Vector{Symbol}
# →   Variable ∈ 2-element Vector{Symbol}
# And data, 2×2 adjoint(::Matrix{Float64}) with eltype Float64:
#                    (:I)        (:Z)
#   (:Steady_state)   0.133333    0.17
#   (:Z₍₋₁₎)          0.0497512   0.05

init = m_ss(:,:Steady_state) |> collect
init[2] *= 1.5

plot_irf(m, initial_state = init, shocks = :none)

# , save_plots = true, save_plots_path = "~/Downloads", save_plots_format = :png)