@testset "Traub nonlinear solver" begin
    func_calls = Ref(0)
    jac_calls = Ref(0)

    function f!(out, x, _)
        func_calls[] += 1
        out[1] = x[1]^3 - 1
        return nothing
    end

    function jac!(J, x, _)
        jac_calls[] += 1
        J[1,1] = 3 * x[1]^2
        return nothing
    end

    func_buffer = zeros(Float64, 1)
    jac_buffer = zeros(Float64, 1, 1)

    prob = MacroModelling.𝒮.LinearProblem(jac_buffer, func_buffer)
    lu_cache = MacroModelling.𝒮.init(prob, MacroModelling.𝒮.LUFactorization())
    chol_cache = MacroModelling.𝒮.init(prob, MacroModelling.𝒮.CholeskyFactorization())

    fnj = MacroModelling.function_and_jacobian(f!, func_buffer, jac!, jac_buffer, chol_cache, lu_cache)

    # Neutral solver parameter placeholders (values are not used by the Traub step)
    neutral = 1.0
    damping = 0.5
    n_damping_params = 13
    param_values = vcat([neutral, 0.9, neutral, neutral, neutral, neutral],
                        fill(damping, n_damping_params),
                        [neutral, 0, 0.0, 2])
    params = MacroModelling.solver_parameters(param_values...)

    initial_guess = [1.2]
    lbs = fill(-Inf, 1)
    ubs = fill(Inf, 1)
    params_and_solved_vars = Float64[]

    sol, _ = MacroModelling.traub(fnj, initial_guess, params_and_solved_vars, lbs, ubs, params)

    fnj.func(fnj.func_buffer, sol, params_and_solved_vars)
    @test isapprox(fnj.func_buffer[1], 0.0; atol = 1e-10)
    @test func_calls[] >= 2
    @test jac_calls[] >= 1
end
