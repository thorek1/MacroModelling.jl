using MacroModelling, Test, ForwardDiff
import LinearAlgebra as LA

function convergence_persistent_system(rho; third = false)
    cons = (post_model_macro = (nVars = 1, nPast_not_future_and_mixed = 1,
                               nExo = 1, past_not_future_and_mixed_idx = [1]),)
    S1 = reshape([rho, zero(rho), 0.01one(rho)], 1, 3)
    S2 = zeros(typeof(rho), 1, 6)
    S2[1, MacroModelling.compressed_pair_index(3, 3, 3)] = 0.002
    matrices = [S1, S2]
    third && push!(matrices, zeros(typeof(rho), 1, 10))
    return MacroModelling.build_ivashchenko_kalman_system_from_constants(
        cons, matrices, [1], third ? :third_order : :second_order)
end
@testset "Persistent stationary moments" begin
    for third in (false, true)
        rho = 0.999
        sys = convergence_persistent_system(rho; third)
        ws = MacroModelling.ivashchenko_kalman_workspace(sys, Float64)
        mean, covariance, solved = MacroModelling.ivashchenko_stationary_initialization(
            sys, [0.0], :theoretical, ws)
        @test solved
        @test mean[1] ≈ 0.001 / (1-rho) rtol=1e-8
        @test covariance[1] ≈ (0.01^2 + 2 * 0.001^2) / (1-rho^2) rtol=1e-8
    end
end

@testset "A unit root requires an explicit prior" begin
    sys = convergence_persistent_system(1.0)
    ws = MacroModelling.ivashchenko_kalman_workspace(sys, Float64)
    _, _, solved = MacroModelling.ivashchenko_stationary_initialization(
        sys, [0.0], :theoretical, ws)
    @test !solved
    _, _, solved = MacroModelling.ivashchenko_stationary_initialization(
        sys, [0.0], :diagonal, ws)
    @test solved
end

function convergence_polynomial_system(theta, third)
    cons = (post_model_macro = (nVars = 2, nPast_not_future_and_mixed = 2,
                               nExo = 1, past_not_future_and_mixed_idx = [1,2]),)
    S1 = [theta[1] 0.1 0.0 0.04; 0.0 0.6 0.0 0.03]
    S2 = zeros(eltype(theta), 2, 10)
    S2[1, MacroModelling.compressed_pair_index(1,1,4)] = theta[2]
    S2[1, MacroModelling.compressed_pair_index(4,4,4)] = 0.002
    S2[2, MacroModelling.compressed_pair_index(4,1,4)] = 0.003
    solution = [S1,S2]
    if third
        S3 = zeros(eltype(theta), 2, 20)
        S3[1, MacroModelling.compressed_triple_index(1,1,1,4)] = theta[3]
        S3[2, MacroModelling.compressed_triple_index(4,4,1,4)] = 0.002
        push!(solution, S3)
    end
    sys = MacroModelling.build_ivashchenko_kalman_system_from_constants(cons, solution, [1],
        third ? :third_order : :second_order)
    return sys, solution
end
function convergence_likelihood(theta, third, closure; record=false)
    sys, = convergence_polynomial_system(theta, third)
    MacroModelling.ivashchenko_filter_pass(sys, reshape([0.01,-0.02,0.0,0.03],1,:),
        zeros(eltype(theta),2); gaussian_closure=closure, measurement_error=1e-4, record)
end
@testset "Accelerated stationary adjoint" begin
    for third in (false,true), closure in (third ? (:exact,:linearized) : (:exact,:linearized,:diagonal))
        theta = [0.95,0.008,0.005]
        sys, solution = convergence_polynomial_system(theta, third)
        ll,tape = convergence_likelihood(theta,third,closure;record=true)
        @test isfinite(ll)
        @test any(it.accelerated for it in tape.initialization_tape.iterations)
        @test ll ≈ first(convergence_likelihood(theta,third,closure)) atol=1e-10
        solution_bar, = MacroModelling.ivashchenko_filter_pullback(sys,tape,solution,1.0)
        function solution_scalar(p)
            _, matrices = convergence_polynomial_system(p,third)
            sum(sum(b .* s) for (b,s) in zip(solution_bar,matrices))
        end
        reverse = ForwardDiff.gradient(solution_scalar,theta)
        forward = ForwardDiff.gradient(p -> first(convergence_likelihood(p,third,closure)),theta)
        @test reverse ≈ forward rtol=1e-6 atol=1e-7
        finite_difference = map(eachindex(theta)) do index
            plus, minus = copy(theta), copy(theta)
            plus[index] += 1e-6
            minus[index] -= 1e-6
            (first(convergence_likelihood(plus, third, closure)) -
             first(convergence_likelihood(minus, third, closure))) / 2e-6
        end
        @test reverse ≈ finite_difference rtol=1e-5 atol=1e-6
    end
end

function convergence_square_shocks(a, q, c, target; start=0.0, max_iter=1000)
    J = LA.Diagonal([true])
    args = (Val(:LagrangeNewton), [start], zeros(1))
    if c === nothing
        return MacroModelling.find_shocks(args..., zeros(1,1), J, reshape([a],1,1),
            reshape([q],1,1), [target]; max_iter)
    end
    return MacroModelling.find_shocks(args..., zeros(1), zeros(1,1), zeros(1,1), zeros(1,1),
        J, reshape([a],1,1), reshape([q],1,1), reshape([c],1,1), [target]; max_iter)
end
@testset "Square shock roots" begin
    for c in (nothing, 0.1)
        x,matched = convergence_square_shocks(1.0,0.1,c,0.0;max_iter=5)
        @test matched
        @test x == [0.0]
        # Starting at the exact root must not use b as the initial residual.
        target = 2.0 + 0.1*4 + (c === nothing ? 0.0 : c*8)
        x,matched = convergence_square_shocks(1.0,0.1,c,target;start=2.0,max_iter=1)
        @test matched
        @test x ≈ [2.0]
    end
end

@testset "Zero-root derivatives and underidentified shocks" begin
    for cubic in (nothing, 0.1)
        derivative = ForwardDiff.derivative(0.0) do target
            scalar_type = typeof(target)
            J = LA.Diagonal([true])
            A = fill(scalar_type(2), 1, 1)
            Q = fill(scalar_type(0.1), 1, 1)
            args = (Val(:LagrangeNewton), zeros(scalar_type, 1), zeros(scalar_type, 1))
            if cubic === nothing
                x, matched = MacroModelling.find_shocks(args..., zeros(scalar_type, 1, 1),
                    J, A, Q, [target])
            else
                x, matched = MacroModelling.find_shocks(args..., zeros(scalar_type, 1),
                    zeros(scalar_type, 1, 1), zeros(scalar_type, 1, 1),
                    zeros(scalar_type, 1, 1), J, A, Q,
                    fill(scalar_type(cubic), 1, 1), [target])
            end
            @test matched
            x[1]
        end
        @test derivative ≈ 0.5
    end
    for cubic in (false, true), target in (0.0, 2.0)
        J = LA.Diagonal([true,true])
        args = (Val(:LagrangeNewton), [target,0.0], zeros(3))
        if cubic
            x, matched = MacroModelling.find_shocks(args..., zeros(4), zeros(3,2),
                zeros(4,2), zeros(4,2), J, [1.0 2.0], zeros(1,3), zeros(1,4), [target])
        else
            x, matched = MacroModelling.find_shocks(args..., zeros(3,2),
                J, [1.0 2.0], zeros(1,3), [target])
        end
        @test matched
        @test x ≈ [target/5,2target/5]
    end
    # Backtracking must recover from a first Newton step near 100, although
    # the smallest root is near one.
    x, matched = convergence_square_shocks(0.01, 1.0, nothing, 1.0)
    @test matched
    @test x[1] ≈ (-0.01 + sqrt(4.0001)) / 2
    # A quadratic with no real preimage must remain a visible failure.
    x, matched = convergence_square_shocks(0.0, 1.0, nothing, -1.0; max_iter=20)
    @test !matched
end
