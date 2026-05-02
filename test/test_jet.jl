using Test
using MacroModelling

if VERSION < v"1.13"
    using JET
end

# Build a minimal RBC model to use as the analysis target.
@model m begin
    y[0] = A[0] * k[-1]^alpha
    1 / c[0] = beta * 1 / c[1] * (alpha * A[1] * k[0]^(alpha - 1) + (1 - delta))
    1 / c[0] = beta * 1 / c[1] * (R[0] / Pi[+1])
    R[0] * beta = (Pi[0] / Pibar)^phi_pi
    A[0] * k[-1]^alpha = c[0] + k[0] - (1 - delta * z_delta[0]) * k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
end

@parameters m begin
    alpha = 0.157
    beta = 0.999
    delta = 0.0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = 0.9
    std_eps = 0.0068
    rho_z_delta = 0.9
    std_z_delta = 0.005
end

# Incremental JET analysis of the package.
# The test is split into chunks so that CI logs reveal which section causes
# a timeout/OOM when the full test_package call is too expensive.
# Each chunk exercises a progressively larger portion of the public API.
# Once all chunks pass individually, the final chunk attempts test_package
# on the whole module.

@testset verbose = true "Static checking (JET.jl)" begin
    if VERSION < v"1.13"
        # ── Chunk 1: Steady state (structures, parser, nsss_solver) ──
        @info "JET chunk 1/7: get_SS"
        @testset "Chunk 1 – Steady state" begin
            @test_call target_modules = (MacroModelling,) MacroModelling.get_SS(m)
        end

        # ── Chunk 2: First-order perturbation solution ──
        @info "JET chunk 2/7: get_solution (first order)"
        @testset "Chunk 2 – First-order solution" begin
            @test_call target_modules = (MacroModelling,) MacroModelling.get_solution(m)
        end

        # ── Chunk 3: IRFs (impulse response functions) ──
        @info "JET chunk 3/7: get_irf"
        @testset "Chunk 3 – IRFs" begin
            @test_call target_modules = (MacroModelling,) MacroModelling.get_irf(m)
        end

        # ── Chunk 4: Moments ──
        @info "JET chunk 4/7: get_moments"
        @testset "Chunk 4 – Moments" begin
            @test_call target_modules = (MacroModelling,) MacroModelling.get_moments(m; mean = true)
        end

        # ── Chunk 5: Simulation ──
        @info "JET chunk 5/7: simulate"
        @testset "Chunk 5 – Simulation" begin
            @test_call target_modules = (MacroModelling,) MacroModelling.simulate(m)
        end

        # ── Chunk 6: Higher-order perturbation ──
        @info "JET chunk 6/7: get_solution (second order)"
        @testset "Chunk 6 – Second-order solution" begin
            @test_call target_modules = (MacroModelling,) MacroModelling.get_solution(m; algorithm = :second_order)
        end

        # ── Chunk 7: Full package analysis ──
        @info "JET chunk 7/7: test_package (full)"
        @testset "Chunk 7 – Full package" begin
            JET.test_package(MacroModelling; target_modules = (MacroModelling,), toplevel_logger = nothing)
        end
    end
end
