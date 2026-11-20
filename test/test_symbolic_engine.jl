using Test
using MacroModelling
import MacroModelling: to_symbolic, from_symbolic, simplify_symbolic, differentiate, solve_for_variable,
                       variable_is_redundant, solve_system_for_variables, substitute_symbolic, symbolic_sparse_jacobian,
                       build_symbolic_function, symbolic_number, SymbolicNode, normcdf, normpdf, norminvcdf
using SparseArrays

# Evaluate a Julia expression with the given symbol values.
function evaluate_expression(ex, values::Dict{Symbol, Float64})
    ex isa Number && return Float64(ex)
    ex isa Symbol && return values[ex]
    body = MacroModelling.replace_symbols(ex, Dict{Symbol, Any}(k => v for (k, v) in values))
    return Float64(Core.eval(MacroModelling, body))
end

@testset verbose = true "Native symbolic engine" begin
    point = Dict(:x => 1.3, :y => 0.7, :α => 0.36, :β => 0.99, :δ => 0.025, :z => 0.2)

    @testset "Canonical simplification" begin
        @test simplify_symbolic(:(x - x)) == 0
        @test simplify_symbolic(:(2x + 3x)) == :(5x)
        @test simplify_symbolic(:(x * x / x)) == :x
        @test simplify_symbolic(:(exp(log(x)))) == :x
        @test simplify_symbolic(:(log(exp(x + y)))) == simplify_symbolic(:(x + y))
        @test simplify_symbolic(:((x * y) ^ 2 / y ^ 2)) == :(x ^ 2)
        @test simplify_symbolic(:(1 / 2 + 1 / 2)) == 1
        @test simplify_symbolic(:(2.0 * 3)) == 6.0
        @test to_symbolic(:(x + y)) === to_symbolic(:(y + x))
        @test simplify_symbolic(:((α - 1) / (1 - α))) == -1
        @test simplify_symbolic(:((x - y) ^ 2 / (y - x))) == simplify_symbolic(:(y - x))
        for ex in (:(x ^ α * y ^ (1 - α) - δ * x / y), :(1 / (1 + x) - exp(-z) * sqrt(y)), :(normcdf(x - y) + log(β / x)))
            @test evaluate_expression(simplify_symbolic(ex), point) ≈ evaluate_expression(ex, point)
            @test evaluate_expression(Meta.parse(string(simplify_symbolic(ex))), point) ≈ evaluate_expression(ex, point)
        end
    end

    @testset "Derivatives" begin
        h = 1e-6
        for ex in (:(x ^ α * y ^ (1 - α)), :(log(x) / (1 + y ^ 2)), :(exp(x * y) - sqrt(x)), :(normcdf(x) * norminvcdf(y)),
                   :(x ^ y), :(max(x, y) * x), :(erfc(x) + tanh(y * x)))
            d = from_symbolic(differentiate(to_symbolic(ex), :x))
            up = merge(point, Dict(:x => point[:x] + h))
            down = merge(point, Dict(:x => point[:x] - h))
            numerical = (evaluate_expression(ex, up) - evaluate_expression(ex, down)) / (2h)
            @test evaluate_expression(d, point) ≈ numerical rtol = 1e-6
        end
        @test from_symbolic(differentiate(to_symbolic(:(3x ^ 2 + y)), :x)) == :(6x)
        @test from_symbolic(differentiate(to_symbolic(:(y * z)), :x)) == 0
    end

    @testset "Closed-form solutions" begin
        steady_state_capital = :(1 / β - (α * k ^ (α - 1) + 1 - δ))
        solution = solve_for_variable(to_symbolic(steady_state_capital), :k)
        @test solution !== nothing
        k = evaluate_expression(from_symbolic(solution), point)
        @test k ≈ ((1 / point[:β] - 1 + point[:δ]) / point[:α]) ^ (1 / (point[:α] - 1))

        # linear with the variable in several terms
        solution = solve_for_variable(to_symbolic(:(c - (w * l - c * τ))), :c)
        @test from_symbolic(solution) == simplify_symbolic(:(w * l / (1 + τ)))

        # rational in the variable
        solution = solve_for_variable(to_symbolic(:(y - x / (1 + x))), :x)
        @test evaluate_expression(from_symbolic(solution), point) ≈ point[:y] / (1 - point[:y])

        # two powers of the variable
        solution = solve_for_variable(to_symbolic(:(s * k ^ α - δ * k)), :k)
        @test evaluate_expression(from_symbolic(solution), merge(point, Dict(:s => 0.2))) ≈ (0.2 / point[:δ]) ^ (1 / (1 - point[:α]))

        # inverse functions
        @test from_symbolic(solve_for_variable(to_symbolic(:(log(x) - y)), :x)) == :(exp(y))
        @test from_symbolic(solve_for_variable(to_symbolic(:(normcdf(x) - y)), :x)) == :(norminvcdf(y))

        # even powers are only inverted for a variable known to be positive
        @test solve_for_variable(to_symbolic(:(x ^ 2 - y)), :x) === nothing
        @test from_symbolic(solve_for_variable(to_symbolic(:(x ^ 2 - y)), :x, positive = true)) == :(sqrt(y))

        # a single power of the variable has the root zero, a negative power has none
        @test from_symbolic(solve_for_variable(to_symbolic(:(0 - x)), :x)) == 0
        @test solve_for_variable(to_symbolic(:(y / x)), :x) === nothing
        @test from_symbolic(solve_for_variable(to_symbolic(:(a - ρ * a)), :a)) == 0
        @test solve_for_variable(to_symbolic(:(1 / c - β * R / c)), :c) === nothing

        # the variable enters only through one shared subexpression
        @test from_symbolic(solve_for_variable(to_symbolic(:(log(m) - ((1 - ρ) * log(mst) + ρ * log(m)))), :m)) == :mst

        # solutions are combined over a common denominator when that is simpler
        @test from_symbolic(solve_for_variable(to_symbolic(:(x - (c * b * x + p / (c ^ 2 * s) + x) / (c * b + 1))), :p)) == 0

        # no closed form
        @test solve_for_variable(to_symbolic(:(x + exp(x) - y)), :x) === nothing

        # redundancy: the variable only enters as a common factor
        @test variable_is_redundant(to_symbolic(:(z - ρ * z)), :z)
        @test variable_is_redundant(to_symbolic(:(1 / c - β * R / c)), :c)
        @test !variable_is_redundant(to_symbolic(:(c - w)), :c)
        @test variable_is_redundant(to_symbolic(:(x - (b * x + p + x) / (b + 1))), :x)
        @test variable_is_redundant(to_symbolic(:(c ^ -σ - β * c ^ -σ * R)), :c)
        @test !variable_is_redundant(to_symbolic(:(c ^ -σ - c ^ -α)), :c)

        system = solve_system_for_variables(SymbolicNode[to_symbolic(:(x + y - 3)), to_symbolic(:(x - y - 1))], [:x, :y])
        @test from_symbolic(system[:x]) == 2
        @test from_symbolic(system[:y]) == 1

        @test substitute_symbolic(:(x * y + x), :x, 2) == simplify_symbolic(:(2y + 2))
    end

    @testset "Code generation" begin
        exprs = SymbolicNode[to_symbolic(:(input_a ^ 2 * input_b + log(input_a))), to_symbolic(:(exp(input_b) * input_a ^ 2 * input_b))]
        f! = build_symbolic_function(exprs, [[:input_a], [:input_b]])
        out = zeros(2)
        f!(out, [2.0], [0.5])
        @test out ≈ [4 * 0.5 + log(2.0), exp(0.5) * 4 * 0.5]

        # outside the domain the generated code returns NaN instead of throwing
        f!(out, [-2.0], [0.5])
        @test isnan(out[1])

        jacobian = symbolic_sparse_jacobian(exprs, [:input_a, :input_b])
        @test nnz(jacobian) == 4
        j! = build_symbolic_function(jacobian, [[:input_a, :input_b]])
        buffer = similar(jacobian, Float64)
        j!(buffer, [2.0, 0.5])
        @test Matrix(buffer) ≈ [2 * 2.0 * 0.5 + 1 / 2.0  4.0; exp(0.5) * 2 * 2.0 * 0.5  exp(0.5) * 4 * (1 + 0.5)]

        dense! = build_symbolic_function(Matrix(jacobian), [[:input_a, :input_b]])
        dense = fill(NaN, 2, 2)
        dense!(dense, [2.0, 0.5])
        @test dense ≈ Matrix(buffer)
    end

    @testset "Block alias elimination" begin
        # x = a * y, y = b * z + x / 2, z^2 = y + c: x and y are eliminated, z stays
        eqs = Any[:(x - a * y), :(y - (b * z + x / 2)), :(z ^ 2 - (y + c))]
        remaining_eqs, remaining_vars, remaining_idx, eliminated =
            MacroModelling.eliminate_block_aliases(eqs, [:x, :y, :z], [1, 2, 3])
        # after x is gone, y has the parameter coefficient 1 - a / 2 in the second
        # equation, so y is taken from the third equation (constant coefficient)
        @test remaining_vars == [:z]
        @test remaining_idx == [2]
        @test first.(eliminated) == [:x, :y]

        sol = Dict(:a => 0.5, :b => 2.0, :c => 1.0)
        # positive root of the reduced equation by bisection
        g(zv) = evaluate_expression(remaining_eqs[1], merge(sol, Dict(:z => zv)))
        lo, hi = 0.0, 10.0
        for _ in 1:200
            mid = (lo + hi) / 2
            sign(g(mid)) == sign(g(lo)) ? (lo = mid) : (hi = mid)
        end
        sol[:z] = (lo + hi) / 2
        for (v, rhs, _) in Iterators.reverse(eliminated)
            sol[v] = evaluate_expression(rhs, sol)
        end
        for eq in eqs
            @test abs(evaluate_expression(eq, sol)) < 1e-10
        end

        # an excluded (bounded) variable and a solution with a domain hazard both stay in the block
        kept = MacroModelling.eliminate_block_aliases(Any[:(x - y ^ 0.3), :(y + x ^ 2 - 5)], [:x, :y], [1, 2], excluded_vars = Set([:y]))
        @test isempty(kept[4])
        @test kept[2] == [:x, :y]

        # ss_symbolic_mode = :full removes aliases from numerical blocks; the steady state is unchanged
        model_source = read(joinpath(@__DIR__, "..", "models", "Baxter_King_1993.jl"), String)
        full_source = replace(model_source, "@parameters Baxter_King_1993 begin" => "@parameters Baxter_King_1993 ss_symbolic_mode = :full begin")
        @test full_source != model_source
        block_sizes(m) = [length(split(d, ",")) for (t, d) in zip(m.constants.nsss_solver.step_types, m.constants.nsss_solver.descriptions) if t == 0x02]
        include_string(@__MODULE__, model_source)
        default_ss = get_steady_state(Baxter_King_1993, derivatives = false)
        default_blocks = block_sizes(Baxter_King_1993)
        include_string(@__MODULE__, full_source)
        full_ss = get_steady_state(Baxter_King_1993, derivatives = false)
        @test sum(block_sizes(Baxter_King_1993)) < sum(default_blocks)
        @test isapprox(collect(full_ss), collect(default_ss), rtol = 1e-10)
    end
end
