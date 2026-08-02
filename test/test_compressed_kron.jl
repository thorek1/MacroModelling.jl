using Test
using LinearAlgebra
using SparseArrays
using Random
using MacroModelling

Random.seed!(1234)

function pair_extractor(n)
    rows = Int[]
    cols = Int[]
    vals = Int[]
    row = 0
    for i in 1:n
        for j in 1:i
            row += 1
            push!(rows, row)
            push!(cols, (i - 1) * n + j)
            push!(vals, 1)
            if i != j
                push!(rows, row)
                push!(cols, (j - 1) * n + i)
                push!(vals, 1)
            end
        end
    end
    return sparse(rows, cols, vals, n * (n + 1) ÷ 2, n^2)
end

function triple_extractor(n)
    rows = Int[]
    cols = Int[]
    vals = Int[]
    row = 0
    for i in 1:n
        for j in 1:i
            for k in 1:j
                row += 1
                indices = unique(((i, j, k), (i, k, j), (j, i, k),
                                  (j, k, i), (k, i, j), (k, j, i)))
                for (a, b, c) in indices
                    push!(rows, row)
                    push!(cols, ((a - 1) * n + b - 1) * n + c)
                    push!(vals, 1)
                end
            end
        end
    end
    return sparse(rows, cols, vals, n * (n + 1) * (n + 2) ÷ 6, n^3)
end

@testset "compressed Kronecker vectors" begin
    for n in 0:5
        a = collect(1.0:n)
        b = collect(2.0:2.0:2n)
        c = [-Float64(i) for i in 1:n]
        U₂ = pair_extractor(n)
        U₃ = triple_extractor(n)

        pair = MacroModelling.compressed_kron²(a, b)
        triple = MacroModelling.compressed_kron³(a, b, c)
        @test pair ≈ U₂ * kron(a, b)
        @test triple ≈ U₃ * kron(kron(a, b), c)

        pair_out = similar(pair)
        triple_out = similar(triple)
        @test MacroModelling.compressed_kron²!(pair_out, a, b) === pair_out
        @test MacroModelling.compressed_kron³!(triple_out, a, b, c) === triple_out
        @test pair_out ≈ pair
        @test triple_out ≈ triple

        pair_power_out = similar(pair)
        triple_power_out = similar(triple)
        @test MacroModelling.compressed_kron²_power!(pair_power_out, a) === pair_power_out
        @test MacroModelling.compressed_kron³_power!(triple_power_out, a) === triple_power_out
        @test pair_power_out ≈ U₂ * kron(a, a)
        @test triple_power_out ≈ U₃ * kron(kron(a, a), a)
        @test MacroModelling.compressed_kron²_power(a) ≈ pair_power_out
        @test MacroModelling.compressed_kron³_power(a) ≈ triple_power_out

        same_triple_out = similar(triple)
        @test MacroModelling.compressed_kron³_same!(same_triple_out, a) === same_triple_out
        @test same_triple_out ≈ U₃ * kron(kron(a, a), a)
    end

    a = [1.0, -2.0, 3.0]
    @test MacroModelling.compressed_kron²(a, a) ≈ pair_extractor(3) * kron(a, a)
    @test MacroModelling.compressed_kron³(a, a, a) ≈ triple_extractor(3) * kron(kron(a, a), a)
end

@testset "compressed Kronecker argument order" begin
    # The pair and triple products are fully symmetric in their vector
    # arguments, so call sites are free to pass them in whatever order the
    # available method wants. `find_shocks` and the inversion filter rely on
    # this when they write `compressed_kron²!(buf, x, J)`.
    for n in (1, 3, 6)
        a = randn(n)
        b = randn(n)
        c = randn(n)
        @test MacroModelling.compressed_kron²(a, b) == MacroModelling.compressed_kron²(b, a)
        for perm in ((a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a))
            @test MacroModelling.compressed_kron³(a, b, c) ≈ MacroModelling.compressed_kron³(perm...)
        end
    end

    # And the vector-against-matrix form is the column-wise vector form.
    n = 4
    a = randn(n)
    B = randn(n, 3)
    out = MacroModelling.compressed_kron²(a, B)
    for j in axes(B, 2)
        @test out[:, j] ≈ MacroModelling.compressed_kron²(a, B[:, j])
    end
end

@testset "compressed power derivative weights" begin
    # d/dx compressed_kron²_power(x) = 2·compressed_kron²(x, dx) and
    # d/dx compressed_kron³_power(x) = 3·compressed_kron³(x, x, dx). These are
    # the factors that turn the forward Taylor weights 1/2 and 1/6 into 1 and
    # 1/2 in every Jacobian and pullback built on the compressed basis.
    for n in (3, 5, 8)
        x  = randn(n)
        dx = randn(n)
        h  = 1e-6
        fd² = (MacroModelling.compressed_kron²_power(x .+ h .* dx) .-
               MacroModelling.compressed_kron²_power(x .- h .* dx)) ./ (2h)
        fd³ = (MacroModelling.compressed_kron³_power(x .+ h .* dx) .-
               MacroModelling.compressed_kron³_power(x .- h .* dx)) ./ (2h)
        @test 2 .* MacroModelling.compressed_kron²(x, dx) ≈ fd² rtol = 1e-6
        @test 3 .* MacroModelling.compressed_kron³(x, x, dx) ≈ fd³ rtol = 1e-6
    end
end

@testset "column-wise compressed Kronecker" begin
    # The particle filters carry one column per particle; the `_columns!`
    # helpers must agree with the vector kernels column by column.
    n, N = 5, 7
    A = randn(n, N)
    B = randn(n, N)
    pair²  = zeros(n * (n + 1) ÷ 2, N)
    pairAB = zeros(n * (n + 1) ÷ 2, N)
    cube   = zeros(n * (n + 1) * (n + 2) ÷ 6, N)
    MacroModelling.compressed_kron²_power_columns!(pair², A)
    MacroModelling.compressed_kron²_columns!(pairAB, A, B)
    MacroModelling.compressed_kron³_power_columns!(cube, A)
    for j in 1:N
        @test pair²[:, j]  ≈ MacroModelling.compressed_kron²_power(A[:, j])
        @test pairAB[:, j] ≈ MacroModelling.compressed_kron²(A[:, j], B[:, j])
        @test cube[:, j]   ≈ MacroModelling.compressed_kron³_power(A[:, j])
    end
end

@testset "compressed Kronecker power edge cases" begin
    empty = Float64[]
    @test isempty(MacroModelling.compressed_kron²_power(empty))
    @test isempty(MacroModelling.compressed_kron³_power(empty))
    @test MacroModelling.compressed_kron²_power!(Float64[], empty) isa Vector{Float64}
    @test MacroModelling.compressed_kron³_power!(Float64[], empty) isa Vector{Float64}
end

@testset "compressed Kronecker analytical VJPs" begin
    n = 4
    a = [0.4, -1.1, 0.7, 1.3]
    b = [-0.3, 0.8, 1.2, -0.5]
    c = [0.9, -0.6, 0.2, 1.4]
    pair_cotangent = randn(n * (n + 1) ÷ 2)
    triple_cotangent = randn(n * (n + 1) * (n + 2) ÷ 6)
    da = zeros(n); db = zeros(n)
    MacroModelling.compressed_kron²_vjp!(da, db, pair_cotangent, a, b)
    eps_fd = 1e-7
    @test dot(da, c) ≈ (dot(pair_cotangent, MacroModelling.compressed_kron²(a .+ eps_fd .* c, b)) -
                        dot(pair_cotangent, MacroModelling.compressed_kron²(a .- eps_fd .* c, b))) / (2eps_fd)
    @test dot(db, c) ≈ (dot(pair_cotangent, MacroModelling.compressed_kron²(a, b .+ eps_fd .* c)) -
                        dot(pair_cotangent, MacroModelling.compressed_kron²(a, b .- eps_fd .* c))) / (2eps_fd)

    d3a = zeros(n); d3b = zeros(n); d3c = zeros(n)
    MacroModelling.compressed_kron³_vjp!(d3a, d3b, d3c, triple_cotangent, a, b, c)
    @test dot(d3a, c) ≈ (dot(triple_cotangent, MacroModelling.compressed_kron³(a .+ eps_fd .* c, b, c)) -
                         dot(triple_cotangent, MacroModelling.compressed_kron³(a .- eps_fd .* c, b, c))) / (2eps_fd)

    same_vjp = zeros(n)
    MacroModelling.compressed_kron³_power_vjp!(same_vjp, triple_cotangent, a)
    @test dot(same_vjp, c) ≈ (dot(triple_cotangent, MacroModelling.compressed_kron³(a .+ eps_fd .* c, a .+ eps_fd .* c, a .+ eps_fd .* c)) -
                              dot(triple_cotangent, MacroModelling.compressed_kron³(a .- eps_fd .* c, a .- eps_fd .* c, a .- eps_fd .* c))) / (2eps_fd)

    pair_power_vjp = zeros(n)
    MacroModelling.compressed_kron²_power_vjp!(pair_power_vjp, pair_cotangent, a)
    @test dot(pair_power_vjp, c) ≈ (dot(pair_cotangent, MacroModelling.compressed_kron²(a .+ eps_fd .* c, a .+ eps_fd .* c)) -
                                    dot(pair_cotangent, MacroModelling.compressed_kron²(a .- eps_fd .* c, a .- eps_fd .* c))) / (2eps_fd)

    identity_cotangent = randn(n * (n + 1) * (n + 2) ÷ 6, n)
    identity_vjp = zeros(n)
    MacroModelling.compressed_kron³_identity_vjp!(identity_vjp, identity_cotangent, a)
    @test dot(identity_vjp, c) ≈ (sum(identity_cotangent .* MacroModelling.compressed_kron³(a .+ eps_fd .* c, a .+ eps_fd .* c, Matrix{Float64}(I, n, n))) -
                                  sum(identity_cotangent .* MacroModelling.compressed_kron³(a .- eps_fd .* c, a .- eps_fd .* c, Matrix{Float64}(I, n, n)))) / (2eps_fd)
end

@testset "compressed state-update equivalence" begin
    n = 5
    m = 3
    U₂ = pair_extractor(n)
    U₃ = triple_extractor(n)
    S₂ = randn(m, size(U₂, 1))
    S₃ = randn(m, size(U₃, 1))
    a = randn(n)
    b = randn(n)

    @test S₂ * MacroModelling.compressed_kron²(a, a) / 2 ≈
          (S₂ * U₂) * kron(a, a) / 2
    @test S₂ * MacroModelling.compressed_kron²(a, b) ≈
          (S₂ * U₂) * kron(a, b)
    @test S₃ * MacroModelling.compressed_kron³(a, a, a) / 6 ≈
          (S₃ * U₃) * kron(kron(a, a), a) / 6
end

@testset "compressed directional derivative multiplicities" begin
    n = 5
    a = randn(n)
    da = randn(n)
    a₀ = randn(n)
    a₂ = randn(n)
    da₂ = randn(n)
    U₂ = pair_extractor(n)
    U₃ = triple_extractor(n)

    # The two symmetric pair permutations reduce to one compressed kernel.
    full_pair_derivative = U₂ * (kron(da, a) + kron(a, da)) / 2
    @test MacroModelling.compressed_kron²(da, a) ≈ full_pair_derivative

    # The three symmetric cubic permutations reduce to one kernel with 1/2
    # after the Taylor coefficient 1/6 is applied.
    full_triple_derivative = U₃ * (
        kron(kron(da, a), a) + kron(kron(a, da), a) + kron(kron(a, a), da)) / 6
    @test MacroModelling.compressed_kron³(da, a, a) / 2 ≈ full_triple_derivative

    # A mixed pair has no extra Taylor factor: both distinct directional
    # terms remain present in compressed coordinates.
    full_mixed_pair_derivative = U₂ * (kron(da, a₂) + kron(a₀, da₂))
    @test MacroModelling.compressed_kron²(da, a₂) +
          MacroModelling.compressed_kron²(a₀, da₂) ≈ full_mixed_pair_derivative
end

@testset "cached compressed cubic row maps" begin
    n_state = 3
    n_exo = 2
    state = randn(n_state)
    shock = randn(n_exo)
    state_vol = [state; 1.0]
    shock_offset = length(state_vol)
    augmented = [state_vol; shock]
    full_compressed = MacroModelling.compressed_kron³_power(augmented)

    shock_state_state_indices = sort!([MacroModelling.compressed_triple_index(shock_offset + q, i, j)
                                       for q in 1:n_exo for i in 1:length(state_vol) for j in 1:i])
    shock_state_state_rows = MacroModelling.compressed_shock_state_state_rows(
        shock_state_state_indices, shock_offset, length(state_vol), n_exo)
    shock_state_state = MacroModelling.compressed_triple_shock_state_state(
        shock, state_vol, shock_offset, shock_state_state_indices;
        index_rows = shock_state_state_rows)
    @test shock_state_state ≈ full_compressed[shock_state_state_indices] / 3

    shock_shock_state_indices = sort!([MacroModelling.compressed_triple_index(shock_offset + i,
                                                                               shock_offset + j,
                                                                               k)
                                      for i in 1:n_exo for j in 1:i for k in 1:length(state_vol)])
    shock_shock_state_rows = MacroModelling.compressed_shock_shock_state_rows(
        shock_shock_state_indices, shock_offset, length(state_vol), n_exo)
    shock_shock_state = MacroModelling.compressed_triple_shock_shock_state(
        shock, state_vol, shock_offset, shock_shock_state_indices;
        index_rows = shock_shock_state_rows)
    @test shock_shock_state ≈ full_compressed[shock_shock_state_indices] / 3

    state_to_pair = zeros(Float64, length(shock_shock_state_indices), n_exo * (n_exo + 1) ÷ 2)
    @test MacroModelling.compressed_triple_state_to_pair!(
        state_to_pair, state_vol, length(augmented), shock_offset, n_exo,
        shock_shock_state_indices; index_rows = shock_shock_state_rows) === state_to_pair
    @test state_to_pair ≈ MacroModelling.compressed_triple_state_to_pair(
        state_vol, length(augmented), shock_offset, n_exo, shock_shock_state_indices;
        index_rows = shock_shock_state_rows)

    state_pair_to_shock = zeros(Float64, length(shock_state_state_indices), n_exo)
    @test MacroModelling.compressed_triple_state_pair_to_shock!(
        state_pair_to_shock, MacroModelling.compressed_kron²_power(state_vol),
        length(augmented), shock_offset, n_exo, shock_state_state_indices;
        index_rows = shock_state_state_rows) === state_pair_to_shock
    @test state_pair_to_shock ≈ MacroModelling.compressed_triple_state_pair_to_shock(
        MacroModelling.compressed_kron²_power(state_vol), length(augmented), shock_offset,
        n_exo, shock_state_state_indices; index_rows = shock_state_state_rows)
end

# The model's `third_order_indices` caches these sets, and a handful of pullback
# entry points rebuild them when they are called without the model in scope. Both
# go through the same two builders, so what has to hold is that those builders
# agree with the sorted set plus the binary-search row map the callers index with.
@testset "compressed cubic index-map builders" begin
    for (n_state, n_exo) in ((1, 1), (3, 2), (5, 4), (12, 7))
        shock_offset = n_state

        indices, rows = MacroModelling.compressed_shock_state_state_index_map(n_state, n_exo)
        @test issorted(indices)
        @test allunique(indices)
        @test indices == sort!([MacroModelling.compressed_triple_index(shock_offset + q, i, j)
                                for q in 1:n_exo for i in 1:n_state for j in 1:i])
        @test rows == MacroModelling.compressed_shock_state_state_rows(indices, shock_offset,
                                                                      n_state, n_exo)

        indices, rows = MacroModelling.compressed_shock_shock_state_index_map(n_state, n_exo)
        @test issorted(indices)
        @test allunique(indices)
        @test indices == sort!([MacroModelling.compressed_triple_index(shock_offset + i,
                                                                       shock_offset + j, k)
                                for i in 1:n_exo for j in 1:i for k in 1:n_state])
        @test rows == MacroModelling.compressed_shock_shock_state_rows(indices, shock_offset,
                                                                      n_state, n_exo)
    end
end

@testset "compressed transition static audit" begin
    transition_files = [
        "src/MacroModelling.jl",
        "src/filter/find_shocks.jl",
        "src/filter/inversion.jl",
        "src/filter/particle.jl",
        "src/get_functions.jl",
        "src/occasionally_binding_constraints.jl",
        "src/steady_state/stochastic_steady_state.jl",
    ]

    strip_line_comments(source) = join((first(split(line, "#"; limit = 2)) for line in split(source, '\n')), '\n')
    active_sources = Dict(path => strip_line_comments(read(joinpath(@__DIR__, "..", path), String))
                          for path in transition_files)

    for (path, source) in active_sources
        if path != "src/get_functions.jl"
            @test !occursin(r"second_order_solution\s*\*\s*.*𝐔₂", source)
            @test !occursin(r"third_order_solution\s*\*\s*.*𝐔₃", source)
        end
        @test !occursin("ℒ.kron(aug_state, aug_state)", source)
        @test !occursin("ℒ.kron(aug_state₁, aug_state₁)", source)
        @test !occursin("ℒ.kron(state_vol, state_vol)", source)
        @test !occursin("ℒ.kron(state¹⁻_vol, state¹⁻_vol)", source)
        @test !occursin("ℒ.kron(ℒ.kron(", source)
        @test !occursin("compressed_pair_index(", source)
        @test !occursin("compressed_pair_indices(", source)
        @test !occursin("compressed_triple_indices(", source)
        if path != "src/filter/particle.jl"
            @test !occursin("searchsortedfirst(", source)
        end
    end

    # The only active U₂/U₃ conversions in the audited transition set are the
    # documented full-coordinate public get_solution outputs.
    public_solution = active_sources["src/get_functions.jl"]
    @test count("𝐔₂", public_solution) == 2
    @test count("𝐔₃", public_solution) == 2

    # Strip the two historical block-commented implementations before auditing
    # reverse-mode code. Solver and moments rrules may still use full tensors,
    # but active transition pullbacks must not construct same-state full k rons.
    rrule_source = replace(read(joinpath(@__DIR__, "..", "src/rrules.jl"), String), r"(?s)#=.*?=#" => "")
    rrule_source = strip_line_comments(rrule_source)
    @test !occursin("ℒ.kron(aug_state, aug_state)", rrule_source)
    @test !occursin("ℒ.kron(state_vol, state_vol)", rrule_source)
    @test !occursin("ℒ.kron(ℒ.kron(", rrule_source)
    @test !occursin("compressed_pair_index(", rrule_source)
    @test !occursin("compressed_pair_indices(", rrule_source)
    @test !occursin("compressed_triple_indices(", rrule_source)
    @test !occursin("searchsortedfirst(", rrule_source)
end
