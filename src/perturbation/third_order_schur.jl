# State-block Schur solver for the third-order perturbation solution.
#
# Same idea as second_order_schur.jl. The third-order equation is X = A X B + C
# with B = 𝐔₃ (s⊗s⊗s + Σ_placements s⊗𝛔) 𝐂₃ (compressed triples of
# [past states; 1; shocks]). In the full symmetric triple space the number of
# "1" slots of a column is kept by s⊗s⊗s, and the 𝛔 terms feed a column with
# two "1" slots from columns with a shock pair (eᵢ, eᵢ). So:
#   (x,x,x):  Y = Ā₊ Y (hx⊗hx⊗hx) + C₊                 triple Stein equation,
#   (x,x,1):  Y = Ā₊ Y (hx⊗hx) + C₊                    pair Stein equation,
#   (x,1,1):  Y = Ā₊ (Y + Q) hx + C₊,  Q = Σᵢ X₊(x,eᵢ,eᵢ)   Stein equation,
#   (1,1,1):  (I - Ā₊) Y = 3 Ā₊ Σᵢ X₊(1,eᵢ,eᵢ) + C₊,
# and every column with a shock is explicit.

@inline function third_order_triple_index(i::Int, k::Int, l::Int)
    # sort descending
    if i < k; i, k = k, i; end
    if k < l; k, l = l, k; end
    if i < k; i, k = k, i; end
    return (i * (i - 1) * (i + 1)) ÷ 6 + (k * (k - 1)) ÷ 2 + l
end

"""
    third_order_schur_solve(Ā, s, C, i₊, n₋, nₑ; tol, max_bytes)

Solve the compressed third-order equation `X = A X B + C` (see the file
header). Returns `(X, solved, Z)` with `Z = X[i₊, :] * B`. Gives up (solved =
false) when the dense (x,x,x) block would need more than `max_bytes`.
"""
function third_order_schur_solve(Ā::Matrix{Float64}, s::AbstractMatrix{Float64}, C::Matrix{Float64},
                                    i₊::AbstractVector{Int}, n₋::Int, nₑ::Int;
                                    tol::Float64 = 1e-10, max_bytes::Real = 2.0e9)
    tidx = third_order_triple_index
    nₑ₋ = n₋ + 1 + nₑ
    n₊ = length(i₊)
    c0 = n₋ + 1
    nc = size(C, 2)
    @assert nc == nₑ₋ * (nₑ₋ + 1) * (nₑ₋ + 2) ÷ 6

    # complex Ŷ, R̂ and real Y, R of the (x,x,x) block
    if 48.0 * n₊ * n₋^3 > max_bytes
        return C, false, zeros(0, 0)
    end

    Ā₊ = Ā[i₊, :]
    C₊ = C[i₊, :]
    sd = Matrix{Float64}(s)
    hx = sd[1:n₋, 1:n₋]
    SE = [collect(1:n₋); collect(n₋ + 2:nₑ₋)]
    nSE = length(SE)
    G = sd[1:n₋, SE]
    E = collect(n₋ + 2:nₑ₋)

    SM = ℒ.schur(complex(Ā₊))
    SH = ℒ.schur(complex(hx))

    # (x,x,x)
    Cxxx = Matrix{Float64}(undef, n₊, n₋^3)
    @inbounds for a in 1:n₋, b in 1:n₋, c in 1:n₋
        copyto!(view(Cxxx, :, ((a - 1) * n₋ + (b - 1)) * n₋ + c), view(C₊, :, tidx(a, b, c)))
    end
    Y0, res0 = solve_kron3_stein(Ā₊, hx, Cxxx, schur_M = SM, schur_H = SH)
    Cxxx = nothing

    # (x,x,1)
    Cxx1 = Matrix{Float64}(undef, n₊, n₋^2)
    @inbounds for a in 1:n₋, b in 1:n₋
        copyto!(view(Cxx1, :, (a - 1) * n₋ + b), view(C₊, :, tidx(a, b, c0)))
    end
    Y1, res1 = solve_kron_stein(Ā₊, hx, Cxx1, schur_M = SM, schur_H = SH)

    if !(res0 < tol) || !(res1 < tol)
        @debug "third_order_schur_solve residuals" res0 res1 tol
        return C, false, zeros(0, 0)
    end

    Z = zeros(n₊, nc)

    # columns without a "1": Z(α,β,γ) = Y0 (G⊗G⊗G), α ≥ β ≥ γ over SE positions
    Y0r = reshape(Y0, n₊ * n₋^2, n₋)
    Vα = zeros(n₊ * n₋^2)
    Uαβ = zeros(n₊ * n₋, nSE)
    tmp = zeros(n₊, nSE)
    @inbounds for α in 1:nSE
        ℒ.mul!(Vα, Y0r, view(G, :, α))
        ℒ.mul!(view(Uαβ, :, 1:α), reshape(Vα, n₊ * n₋, n₋), view(G, :, 1:α))
        for β in 1:α
            Uβ = reshape(view(Uαβ, :, β), n₊, n₋)
            ℒ.mul!(view(tmp, :, 1:β), Uβ, view(G, :, 1:β))
            for γ in 1:β
                copyto!(view(Z, :, tidx(SE[α], SE[β], SE[γ])), view(tmp, :, γ))
            end
        end
    end
    Y0 = nothing; Y0r = nothing

    # columns with one "1": Z(α,β,1) = Y1 (G⊗G)
    W1 = kron_right(Y1, G, G)
    @inbounds for α in 1:nSE, β in 1:α
        copyto!(view(Z, :, tidx(SE[α], SE[β], c0)), view(W1, :, (α - 1) * nSE + β))
    end

    # Q[:, d] = Σᵢ X₊(d, eᵢ, eᵢ), d ∈ states
    zq = zeros(n₊, n₋)
    cq = zeros(n₊, n₋)
    @inbounds for d in 1:n₋, e in E
        j = tidx(d, e, e)
        view(zq, :, d) .+= view(Z, :, j)
        view(cq, :, d) .+= view(C₊, :, j)
    end
    Q = Ā₊ * zq + cq

    # (x,1,1)
    Cx11 = C₊[:, [tidx(a, c0, c0) for a in 1:n₋]]
    Y2, res2 = solve_stein(Ā₊, hx, Cx11 + Ā₊ * (Q * hx), schur_M = SM, schur_H = SH)
    if !(res2 < tol)
        return C, false, zeros(0, 0)
    end
    W2 = (Y2 + Q) * G
    @inbounds for α in 1:nSE
        copyto!(view(Z, :, tidx(SE[α], c0, c0)), view(W2, :, α))
    end

    # (1,1,1)
    z1 = zeros(n₊)
    c1 = zeros(n₊)
    @inbounds for e in E
        j = tidx(c0, e, e)
        z1 .+= view(Z, :, j)
        c1 .+= view(C₊, :, j)
    end
    q1 = Ā₊ * z1 + c1
    j111 = tidx(c0, c0, c0)
    lhs = -Ā₊
    @inbounds for i in 1:n₊
        lhs[i, i] += 1
    end
    lu111 = ℒ.lu!(lhs, check = false)
    if !ℒ.issuccess(lu111)
        return C, false, zeros(0, 0)
    end
    Y3 = lu111 \ (3 .* (Ā₊ * q1) .+ view(C₊, :, j111))
    Z[:, j111] .= Y3 .+ 3 .* q1

    X = Ā * Z
    X .+= C
    return X, all(isfinite, X), Z
end
