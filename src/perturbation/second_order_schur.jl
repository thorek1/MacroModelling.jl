# State-block Schur solver for the second-order perturbation solution.
#
# The second-order Sylvester equation is  X = A X B + C  with
#   A = Ā E₊ᵀ        (Ā = A[:, i₊], only the forward-looking columns are nonzero),
#   B = 𝐔₂ (s⊗s + 𝛔) 𝐂₂ (compressed pairs of [past states; 1; shocks]),
#   s = [hx 0 hu; 0 1 0; 0 0 0].
#
# In the full (uncompressed, symmetric) pair space the equation splits into blocks:
#   (x,x):  Y = Ā₊ Y (hx⊗hx) + C₊          Stein equation on the n₊ × n₋² future block,
#   (x,1):  Y = Ā₊ Y hx + C₊               Stein equation on the n₊ × n₋ future block,
#   (1,1):  (I - Ā₊) Y = Ā₊ Σᵢ Y(eᵢ,eᵢ) + C₊,
#   every pair with a shock is explicit once the (x,x) block is known.
# The two Stein equations are solved on complex Schur forms of Ā₊ and hx
# (LAPACK), with BLAS products for the Kronecker structure. The full n × n
# Sylvester operator is never formed.

@inline second_order_pair_index(i::Int, k::Int) = i ≥ k ? (i * (i - 1)) ÷ 2 + k : (k * (k - 1)) ÷ 2 + i

"""
    kron_right(Y, F, G)

Return `Y * kron(F, G)` without forming the Kronecker product. The columns of
`Y` are indexed as `(c - 1) * size(G, 1) + d`, with `c` a row of `F` and `d` a
row of `G`.
"""
function kron_right(Y::AbstractMatrix, F::AbstractMatrix, G::AbstractMatrix)
    p = size(Y, 1)
    mF, kF = size(F)
    mG, kG = size(G)
    @assert size(Y, 2) == mF * mG
    R = promote_type(eltype(Y), eltype(F), eltype(G))
    Yd = Y isa Matrix{R} ? Y : Matrix{R}(Y)
    Y3 = reshape(Yd, p, mG, mF)
    T1 = Array{R}(undef, p, kG, mF)
    @inbounds for c in 1:mF
        ℒ.mul!(view(T1, :, :, c), view(Y3, :, :, c), G)
    end
    out = Matrix{R}(undef, p, kG * kF)
    ℒ.mul!(reshape(out, p * kG, kF), reshape(T1, p * kG, mF), F)
    return out
end

# Solve (I - λ T) y = y in place for upper triangular T (column oriented).
@inline function stein_backsubstitute!(y::AbstractVector{ComplexF64}, T::Matrix{ComplexF64}, λ::ComplexF64)
    p = length(y)
    @inbounds for j in p:-1:1
        yj = y[j] / (1 - λ * T[j, j])
        y[j] = yj
        λyj = λ * yj
        for i in 1:j-1
            y[i] += λyj * T[i, j]
        end
    end
    return y
end

# Triangular kernels on complex Schur forms (Tm, Th upper triangular).

# Ŷ = μ Tm Ŷ Th + R̂ for Ŷ (p × m), in place.
function kron1_triangular!(Ŷ::AbstractMatrix{ComplexF64}, Tm::Matrix{ComplexF64}, Th::Matrix{ComplexF64},
                            R̂::AbstractMatrix{ComplexF64}, μ::ComplexF64)
    p, m = size(Ŷ)
    Tm_up = ℒ.UpperTriangular(Tm)
    w = zeros(ComplexF64, p)
    @inbounds for b in 1:m
        y = view(Ŷ, :, b)
        if b > 1
            ℒ.mul!(w, view(Ŷ, :, 1:b - 1), view(Th, 1:b - 1, b))
            ℒ.lmul!(Tm_up, w)
            y .= view(R̂, :, b) .+ μ .* w
        else
            copyto!(y, view(R̂, :, b))
        end
        stein_backsubstitute!(y, Tm, μ * Th[b, b])
    end
    return Ŷ
end

# Ŷ = μ Tm Ŷ (Th ⊗ Th) + R̂ for Ŷ (p × m²), in place. Ŷ must be a Matrix.
function kron2_triangular!(Ŷ::Matrix{ComplexF64}, Tm::Matrix{ComplexF64}, Th::Matrix{ComplexF64},
                            R̂::AbstractMatrix{ComplexF64}, μ::ComplexF64)
    p = size(Ŷ, 1)
    m = size(Th, 1)
    Tm_up = ℒ.UpperTriangular(Tm)
    Ŷ3 = reshape(Ŷ, p, m, m)
    Ŷmat = reshape(Ŷ, p * m, m)   # an Array, so its column views stay BLAS-compatible
    P = zeros(ComplexF64, p, m)
    PT = zeros(ComplexF64, p, m)
    RHS = zeros(ComplexF64, p, m)
    @inbounds for a in 1:m
        # P[:, d] = Σ_{c<a} Ŷ(c, d) Th[c, a]
        if a > 1
            ℒ.mul!(vec(P), view(Ŷmat, :, 1:a - 1), view(Th, 1:a - 1, a))
        else
            fill!(P, zero(ComplexF64))
        end
        ℒ.mul!(PT, P, Th)
        ℒ.mul!(RHS, Tm_up, PT)
        RHS .*= μ
        RHS .+= view(R̂, :, (a - 1) * m + 1:a * m)
        kron1_triangular!(view(Ŷ3, :, :, a), Tm, Th, RHS, μ * Th[a, a])
    end
    return Ŷ
end

# Ŷ = μ Tm Ŷ (Th ⊗ Th ⊗ Th) + R̂ for Ŷ (p × m³), in place.
function kron3_triangular!(Ŷ::Matrix{ComplexF64}, Tm::Matrix{ComplexF64}, Th::Matrix{ComplexF64},
                            R̂::AbstractMatrix{ComplexF64}, μ::ComplexF64)
    p = size(Ŷ, 1)
    m = size(Th, 1)
    m2 = m * m
    Tm_up = ℒ.UpperTriangular(Tm)
    Ŷmat = reshape(Ŷ, p * m2, m)
    P = zeros(ComplexF64, p, m2)
    RHS = zeros(ComplexF64, p, m2)
    Yb = zeros(ComplexF64, p, m2)
    @inbounds for a in 1:m
        if a > 1
            ℒ.mul!(vec(P), view(Ŷmat, :, 1:a - 1), view(Th, 1:a - 1, a))
            ℒ.mul!(RHS, Tm_up, kron_right(P, Th, Th))
            RHS .*= μ
        else
            fill!(RHS, zero(ComplexF64))
        end
        RHS .+= view(R̂, :, (a - 1) * m2 + 1:a * m2)
        fill!(Yb, zero(ComplexF64))
        kron2_triangular!(Yb, Tm, Th, RHS, μ * Th[a, a])
        copyto!(view(Ŷ, :, (a - 1) * m2 + 1:a * m2), Yb)
    end
    return Ŷ
end

"""
    kron_right3(Y, F, G, H)

Return `Y * kron(F, kron(G, H))` without forming the Kronecker product.
"""
function kron_right3(Y::AbstractMatrix, F::AbstractMatrix, G::AbstractMatrix, H::AbstractMatrix)
    p = size(Y, 1)
    mF, kF = size(F)
    mG, kG = size(G)
    mH, kH = size(H)
    @assert size(Y, 2) == mF * mG * mH
    R = promote_type(eltype(Y), eltype(F), eltype(G), eltype(H))
    Yd = Y isa Matrix{R} ? Y : Matrix{R}(Y)
    T1 = Array{R}(undef, p, kG * kH, mF)
    @inbounds for c in 1:mF
        T1[:, :, c] = kron_right(Yd[:, (c - 1) * mG * mH + 1:c * mG * mH], G, H)
    end
    out = Matrix{R}(undef, p, kF * kG * kH)
    ℒ.mul!(reshape(out, p * kG * kH, kF), reshape(T1, p * kG * kH, mF), F)
    return out
end

"""
    solve_kron_stein(M, H, R)

Solve `Y = M * Y * kron(H, H) + R` for `Y` (size p × m²) on complex Schur forms
of `M` (p × p) and `H` (m × m). Returns `(Y, relative_residual)`.
"""
function solve_kron_stein(M::Matrix{Float64}, H::Matrix{Float64}, R::Matrix{Float64};
                    schur_M = nothing, schur_H = nothing)
    p = size(M, 1)
    m = size(H, 1)
    if m == 0 || p == 0
        return zeros(p, m * m), 0.0
    end
    SM = schur_M === nothing ? ℒ.schur(complex(M)) : schur_M
    SH = schur_H === nothing ? ℒ.schur(complex(H)) : schur_H
    U, Tm = SM.Z, SM.T
    Q, Th = SH.Z, SH.T
    R̂ = kron_right(U' * R, Q, Q)
    Ŷ = zeros(ComplexF64, p, m * m)
    kron2_triangular!(Ŷ, Tm, Th, R̂, one(ComplexF64))
    Y = real.(U * kron_right(Ŷ, Q', Q'))
    return Y, kron_stein_probe_residual(Y, M, H, R)
end

"""
    solve_kron3_stein(M, H, R)

Solve `Y = M * Y * kron(H, H, H) + R` for `Y` (size p × m³) on complex Schur
forms. Returns `(Y, relative_residual)`.
"""
function solve_kron3_stein(M::Matrix{Float64}, H::Matrix{Float64}, R::Matrix{Float64};
                    schur_M = nothing, schur_H = nothing)
    p = size(M, 1)
    m = size(H, 1)
    if m == 0 || p == 0
        return zeros(p, m^3), 0.0
    end
    SM = schur_M === nothing ? ℒ.schur(complex(M)) : schur_M
    SH = schur_H === nothing ? ℒ.schur(complex(H)) : schur_H
    U, Tm = SM.Z, SM.T
    Q, Th = SH.Z, SH.T
    R̂ = kron_right3(U' * R, Q, Q, Q)
    Ŷ = zeros(ComplexF64, p, m^3)
    kron3_triangular!(Ŷ, Tm, Th, R̂, one(ComplexF64))
    Qa = Matrix(Q')
    Y = real.(U * kron_right3(Ŷ, Qa, Qa, Qa))
    # probe residual with x ⊗ y ⊗ z
    worst = 0.0
    for probe in 1:2
        x = [sin(1.3 * i + probe) for i in 1:m]
        y = [cos(0.7 * i + 2 * probe) for i in 1:m]
        z = [sin(0.4 * i + 3 * probe) for i in 1:m]
        v = kron(x, kron(y, z))
        Kv = kron(H * x, kron(H * y, H * z))
        Yv = Y * v
        r = Yv - M * (Y * Kv) - R * v
        worst = max(worst, ℒ.norm(r) / max(ℒ.norm(Yv), ℒ.norm(R * v), eps()))
    end
    return Y, worst
end

# Relative residual of Y = M Y kron(H, H) + R on two rank-one probes x ⊗ y,
# using kron(H, H) (x ⊗ y) = (H x) ⊗ (H y). Costs O(p m²) instead of a full
# residual.
function kron_stein_probe_residual(Y::Matrix{Float64}, M::Matrix{Float64}, H::Matrix{Float64}, R::Matrix{Float64})
    m = size(H, 1)
    worst = 0.0
    for probe in 1:2
        x = [sin(1.3 * i + probe) for i in 1:m]
        y = [cos(0.7 * i + 2 * probe) for i in 1:m]
        v = kron(x, y)
        Kv = kron(H * x, H * y)
        Yv = Y * v
        r = Yv - M * (Y * Kv) - R * v
        scale = max(ℒ.norm(Yv), ℒ.norm(R * v), eps())
        worst = max(worst, ℒ.norm(r) / scale)
    end
    return worst
end

"""
    solve_stein(M, H, R)

Solve `Y = M * Y * H + R` for `Y` (size p × m) on complex Schur forms.
Returns `(Y, relative_residual)`.
"""
function solve_stein(M::Matrix{Float64}, H::Matrix{Float64}, R::Matrix{Float64};
                    schur_M = nothing, schur_H = nothing)
    p = size(M, 1)
    m = size(H, 1)
    if m == 0 || p == 0
        return zeros(p, m), 0.0
    end
    SM = schur_M === nothing ? ℒ.schur(complex(M)) : schur_M
    SH = schur_H === nothing ? ℒ.schur(complex(H)) : schur_H
    U, Tm = SM.Z, SM.T
    Q, Th = SH.Z, SH.T
    R̂ = U' * R * Q
    Ŷ = zeros(ComplexF64, p, m)
    kron1_triangular!(Ŷ, Tm, Th, R̂, one(ComplexF64))
    Y = real.(U * Ŷ * Q')
    res = Y - M * Y * H - R
    scale = max(ℒ.norm(Y), ℒ.norm(R), eps())
    return Y, ℒ.norm(res) / scale
end

"""
    second_order_schur_solve(Ā, s, C, i₊, n₋, nₑ; tol)

Solve the compressed second-order equation `X = A X B + C` (see the file
header) for `X` (n × compressed pairs). `Ā = A[:, i₊]`, `s = 𝐒₁₋╱𝟏ₑ`.
Returns `(X, solved, Z)` with `Z = X[i₊, :] * B` (used by the pullback).
"""
function second_order_schur_solve(Ā::Matrix{Float64}, s::AbstractMatrix{Float64}, C::Matrix{Float64},
                                    i₊::AbstractVector{Int}, n₋::Int, nₑ::Int; tol::Float64 = 1e-10,
                                    schur_M = nothing, schur_H = nothing)
    pidx = second_order_pair_index
    nₑ₋ = n₋ + 1 + nₑ
    n₊ = length(i₊)
    c0 = n₋ + 1
    nc = size(C, 2)
    @assert nc == nₑ₋ * (nₑ₋ + 1) ÷ 2

    Ā₊ = Ā[i₊, :]
    C₊ = C[i₊, :]
    sd = Matrix{Float64}(s)
    hx = sd[1:n₋, 1:n₋]
    SE = [collect(1:n₋); collect(n₋ + 2:nₑ₋)]
    nSE = length(SE)
    G = sd[1:n₋, SE]

    # (x,x) block
    Cxx = Matrix{Float64}(undef, n₊, n₋ * n₋)
    @inbounds for a in 1:n₋, b in 1:n₋
        copyto!(view(Cxx, :, (a - 1) * n₋ + b), view(C₊, :, pidx(a, b)))
    end
    SM = schur_M === nothing ? ℒ.schur(complex(Ā₊)) : schur_M
    SH = schur_H === nothing ? ℒ.schur(complex(hx)) : schur_H
    Yxx, res_xx = solve_kron_stein(Ā₊, hx, Cxx, schur_M = SM, schur_H = SH)

    # (x,1) block
    Cx1 = C₊[:, [pidx(c0, a) for a in 1:n₋]]
    Yx1, res_x1 = solve_stein(Ā₊, hx, Cx1, schur_M = SM, schur_H = SH)

    if !(res_xx < tol) || !(res_x1 < tol)
        return C, false, zeros(0, 0)
    end

    W = kron_right(Yxx, G, G)
    W1 = Yx1 * G

    pos(i) = i ≤ n₋ ? i : i - 1

    Z = zeros(n₊, nc)
    @inbounds for i in 1:nₑ₋, k in 1:i
        j = pidx(i, k)
        if i == c0 && k == c0
            continue
        elseif i == c0 || k == c0
            o = i == c0 ? k : i
            copyto!(view(Z, :, j), view(W1, :, pos(o)))
        else
            copyto!(view(Z, :, j), view(W, :, (pos(i) - 1) * nSE + pos(k)))
        end
    end

    # (1,1) block: (I - Ā₊) Y₁₁ = Ā₊ Σ Y(e,e) + C₊₁₁, Y(e,e) = Ā₊ Z(e,e) + C₊(e,e)
    zee = zeros(n₊)
    cee = zeros(n₊)
    @inbounds for e in n₋ + 2:nₑ₋
        j = pidx(e, e)
        zee .+= view(Z, :, j)
        cee .+= view(C₊, :, j)
    end
    Σee = Ā₊ * zee + cee
    j11 = pidx(c0, c0)
    lhs = -Ā₊
    @inbounds for i in 1:n₊
        lhs[i, i] += 1
    end
    lu11 = ℒ.lu!(lhs, check = false)
    if !ℒ.issuccess(lu11)
        return C, false, zeros(0, 0)
    end
    Y11 = lu11 \ (Ā₊ * Σee + view(C₊, :, j11))
    Z[:, j11] .= Y11 .+ Σee

    X = Ā * Z
    X .+= C
    return X, all(isfinite, X), Z
end

"""
    second_order_schur_adjoint(Ā, s, Gc, i₊, n₋, nₑ; tol)

Solve the adjoint compressed equation `∂C = Aᵀ ∂C Bᵀ + Gc` with the same block
structure as `second_order_schur_solve`. Returns `(∂C, solved)`.
"""
function second_order_schur_adjoint(Ā::Matrix{Float64}, s::AbstractMatrix{Float64}, Gc::AbstractMatrix{Float64},
                                    i₊::AbstractVector{Int}, n₋::Int, nₑ::Int; tol::Float64 = 1e-10)
    pidx = second_order_pair_index
    nₑ₋ = n₋ + 1 + nₑ
    n₊ = length(i₊)
    c0 = n₋ + 1
    nc = size(Gc, 2)
    @assert nc == nₑ₋ * (nₑ₋ + 1) ÷ 2

    Ā₊t = Matrix(transpose(Ā[i₊, :]))
    sd = Matrix{Float64}(s)
    hxt = Matrix(transpose(sd[1:n₋, 1:n₋]))
    hut = Matrix(transpose(sd[1:n₋, n₋ + 2:nₑ₋]))
    SE = [collect(1:n₋); collect(n₋ + 2:nₑ₋)]
    nSE = length(SE)
    Gt = Matrix(transpose(sd[1:n₋, SE]))

    # The compressed adjoint equals the full-space adjoint with the symmetric
    # forcing Gs (off-diagonal pairs split in half), read back with weight 2
    # on off-diagonal pairs.
    Gs = Matrix{Float64}(Gc)
    @inbounds for i in 1:nₑ₋, k in 1:i - 1
        view(Gs, :, pidx(i, k)) .*= 0.5
    end
    F = transpose(Ā) * Gs   # n₊ × nc

    j11 = pidx(c0, c0)
    lhs = -Ā₊t
    @inbounds for i in 1:n₊
        lhs[i, i] += 1
    end
    lu11 = ℒ.lu!(lhs, check = false)
    if !ℒ.issuccess(lu11)
        return Gc, false
    end
    Q11 = lu11 \ F[:, j11]
    AQ11 = Ā₊t * Q11

    # Known Q on (SE × SE) pairs that contain a shock
    Qk = zeros(n₊, nSE * nSE)
    @inbounds for α in 1:nSE, β in 1:nSE
        i = SE[α]
        k = SE[β]
        (i ≤ n₋ && k ≤ n₋) && continue
        col = view(Qk, :, (α - 1) * nSE + β)
        copyto!(col, view(F, :, pidx(i, k)))
        if i == k
            col .+= AQ11
        end
    end
    rest = kron_right(Qk, Gt, Gt)   # n₊ × n₋²

    Fxx = Matrix{Float64}(undef, n₊, n₋ * n₋)
    @inbounds for a in 1:n₋, b in 1:n₋
        copyto!(view(Fxx, :, (a - 1) * n₋ + b), view(F, :, pidx(a, b)))
    end
    SM = ℒ.schur(complex(Ā₊t))
    SH = ℒ.schur(complex(hxt))
    Qxx, res_xx = solve_kron_stein(Ā₊t, hxt, Ā₊t * rest + Fxx, schur_M = SM, schur_H = SH)

    QE1 = F[:, [pidx(c0, e) for e in n₋ + 2:nₑ₋]]
    Fx1 = F[:, [pidx(c0, a) for a in 1:n₋]]
    QE1hu = QE1 * hut
    Qx1, res_x1 = solve_stein(Ā₊t, hxt, Ā₊t * QE1hu + Fx1, schur_M = SM, schur_H = SH)

    if !(res_xx < tol) || !(res_x1 < tol)
        return Gc, false
    end

    Vxx = kron_right(Qxx, hxt, hxt)
    Vxx .+= rest
    Vx1 = Qx1 * hxt
    Vx1 .+= QE1hu

    ∂C = Matrix{Float64}(Gc)
    @inbounds for a in 1:n₋, b in 1:a
        j = pidx(a, b)
        f = a == b ? 1.0 : 2.0
        for (r, ir) in enumerate(i₊)
            ∂C[ir, j] += f * Vxx[r, (a - 1) * n₋ + b]
        end
    end
    @inbounds for a in 1:n₋
        j = pidx(c0, a)
        for (r, ir) in enumerate(i₊)
            ∂C[ir, j] += 2.0 * Vx1[r, a]
        end
    end
    @inbounds for (r, ir) in enumerate(i₊)
        ∂C[ir, j11] += Q11[r]
    end
    @inbounds for e in n₋ + 2:nₑ₋
        j = pidx(e, e)
        for (r, ir) in enumerate(i₊)
            ∂C[ir, j] += Q11[r]
        end
    end
    return ∂C, all(isfinite, ∂C)
end

"""
    kron_right_pullback!(∂F, ∂G, L, Y, F, G)

Add the gradients of `⟨L, Y * kron(F, G)⟩` with respect to `F` and `G` to
`∂F` and `∂G` (they may be the same array). Same column layout as `kron_right`.
"""
function kron_right_pullback!(∂F::AbstractMatrix, ∂G::AbstractMatrix, L::AbstractMatrix,
                                Y::AbstractMatrix, F::AbstractMatrix, G::AbstractMatrix)
    p = size(Y, 1)
    mF, kF = size(F)
    mG, kG = size(G)
    R = promote_type(eltype(Y), eltype(L), eltype(F), eltype(G))
    Yd = Y isa Matrix{R} ? Y : Matrix{R}(Y)
    Ld = L isa Matrix{R} ? L : Matrix{R}(L)
    Y3 = reshape(Yd, p, mG, mF)
    L3 = reshape(Ld, p, kG, kF)
    Gt = Matrix{R}(transpose(G))
    # ∂F[c, a] = Σ_{p,d} Y[p, (c,d)] Σ_b L[p, (a,b)] G[d, b]
    M = Array{R}(undef, p, mG, kF)
    @inbounds for a in 1:kF
        ℒ.mul!(view(M, :, :, a), view(L3, :, :, a), Gt)
    end
    ℒ.mul!(∂F, transpose(reshape(Yd, p * mG, mF)), reshape(M, p * mG, kF), true, true)
    # ∂G[d, b] = Σ_{p,c} Y[p, (c,d)] Σ_a L[p, (a,b)] F[c, a]
    N3 = reshape(reshape(Ld, p * kG, kF) * transpose(F), p, kG, mF)
    @inbounds for c in 1:mF
        ℒ.mul!(∂G, transpose(view(Y3, :, :, c)), view(N3, :, :, c), true, true)
    end
    return ∂F, ∂G
end

"""
    second_order_hessian_forcing(∇₂, M)

Return `∇₂ * compressed_kron²(M)` as a dense matrix, built equation by
equation: row `r` is the lower triangle of `M' H_r M`, with `H_r` the
symmetric Hessian of equation `r` on its own support. `∇₂` uses the
compressed pair columns `(i, k)`, `i ≥ k`.
"""
function second_order_hessian_forcing(∇₂::SparseMatrixCSC{Float64}, M::AbstractMatrix{Float64})
    n = size(∇₂, 1)
    N, q = size(M)
    @assert size(∇₂, 2) == N * (N + 1) ÷ 2
    nc = q * (q + 1) ÷ 2
    Md = Matrix{Float64}(M)
    ∇₂t = sparse(transpose(∇₂))
    outT = zeros(nc, n)
    slot = zeros(Int, N)
    supp = Int[]
    us = Int[]
    vs = Int[]
    Cr = zeros(q, q)
    @inbounds for r in 1:n
        rng = ∇₂t.colptr[r]:∇₂t.colptr[r + 1] - 1
        isempty(rng) && continue
        empty!(supp); empty!(us); empty!(vs)
        for idx in rng
            j = ∇₂t.rowval[idx]
            i = (isqrt(8 * j - 7) + 1) ÷ 2
            k = j - (i * (i - 1)) ÷ 2
            push!(us, i); push!(vs, k)
            if slot[i] == 0
                push!(supp, i); slot[i] = length(supp)
            end
            if slot[k] == 0
                push!(supp, k); slot[k] = length(supp)
            end
        end
        s = length(supp)
        Hr = zeros(s, s)
        for (t_, idx) in enumerate(rng)
            a = slot[us[t_]]
            b = slot[vs[t_]]
            v = ∇₂t.nzval[idx]
            Hr[a, b] = v
            Hr[b, a] = v
        end
        Ms = Md[supp, :]
        T = Hr * Ms
        ℒ.mul!(Cr, transpose(Ms), T)
        col = view(outT, :, r)
        for i in 1:q
            base = (i * (i - 1)) ÷ 2
            for k in 1:i
                col[base + k] = Cr[k, i]
            end
        end
        for i in supp
            slot[i] = 0
        end
    end
    return Matrix(transpose(outT))
end

"""
    schur_perturbation_solve_dual(order, Ā, s, C, i₊, n₋, nₑ; tol)

State-block Schur solve of the second (`Val(2)`) or third (`Val(3)`) order
equation for AD number types. ForwardDiffExt adds the method for
`ForwardDiff.Dual`; this fallback returns `(C, false)`, so the caller uses
the generic Sylvester solver.
"""
schur_perturbation_solve_dual(::Val, Ā, s, C, i₊, n₋::Int, nₑ::Int; tol::Float64 = 1e-10) = (C, false)

"""
    second_order_schur_dZ(X₊, G, dG, n₋, nₑ)

Directional derivative of `Z = X₊ B(s)` in the state rows `G = s[1:n₋, SE]`
of `s`, along `dG`, with `X₊` fixed (compressed pair columns).
"""
function second_order_schur_dZ(X₊::Matrix{Float64}, G::Matrix{Float64}, dG::Matrix{Float64}, n₋::Int, nₑ::Int)
    pidx = second_order_pair_index
    nₑ₋ = n₋ + 1 + nₑ
    c0 = n₋ + 1
    p = size(X₊, 1)
    nSE = nₑ₋ - 1
    nc = nₑ₋ * (nₑ₋ + 1) ÷ 2
    Yxx = Matrix{Float64}(undef, p, n₋ * n₋)
    @inbounds for a in 1:n₋, b in 1:n₋
        copyto!(view(Yxx, :, (a - 1) * n₋ + b), view(X₊, :, pidx(a, b)))
    end
    Yx1 = X₊[:, [pidx(c0, a) for a in 1:n₋]]
    T = kron_right(Yxx, dG, G)
    dW1 = Yx1 * dG
    pos(i) = i ≤ n₋ ? i : i - 1
    dZ = zeros(p, nc)
    @inbounds for i in 1:nₑ₋, k in 1:i
        j = pidx(i, k)
        if i == c0 && k == c0
            continue
        elseif i == c0 || k == c0
            o = i == c0 ? k : i
            copyto!(view(dZ, :, j), view(dW1, :, pos(o)))
        else
            α = pos(i); β = pos(k)
            view(dZ, :, j) .= view(T, :, (α - 1) * nSE + β) .+ view(T, :, (β - 1) * nSE + α)
        end
    end
    return dZ
end
