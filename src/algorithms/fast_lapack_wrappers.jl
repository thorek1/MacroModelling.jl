@stable default_mode = "disable" begin

# Old way (≤v0.1.42): Q = qr(A)  — allocates a new QR factorisation object each call
function factorize_qr!(qr_mat::AbstractMatrix,
                       qr_factors::AbstractMatrix{R},
                       qr_ws::FastLapackInterface.QRWs{R};
                       use_fastlapack_qr::Bool = true) where {R <: AbstractFloat}
    if use_fastlapack_qr && R <: Union{Float32, Float64}
        copyto!(qr_factors, qr_mat)
        ℒ.LAPACK.geqrf!(qr_ws, qr_factors; resize = true)
        return qr_factors
    else
        copyto!(qr_factors, qr_mat)
        return ℒ.qr!(qr_factors)
    end
end

# Old way (≤v0.1.42): dest = Q' * src  — allocates intermediate Q.Q' and result
function apply_qr_transpose_left!(dest::AbstractMatrix{R},
                                  src::AbstractMatrix,
                                  Q::AbstractMatrix{R},
                                  qr_orm_ws,
                                  qr_orm_dims::NTuple{3, Int},
                                  qr_ws;
                                  use_fastlapack_qr::Bool = true) where {R <: AbstractFloat}
    orm_dims = (size(Q, 1), size(Q, 2), size(src, 2))
    if qr_orm_dims != orm_dims
        qr_orm_ws = FastLapackInterface.QROrmWs(qr_ws, 'L', 'T', Q, src)
        qr_orm_dims = orm_dims
    end

    copyto!(dest, src)
    ℒ.LAPACK.ormqr!(qr_orm_ws, 'L', 'T', Q, dest)
    return qr_orm_ws, qr_orm_dims
end

# Fallback: dest = Q' * src  (uses standard mul! when FastLapackInterface is not active)
function apply_qr_transpose_left!(dest::AbstractMatrix{R},
                                  src::AbstractMatrix,
                                  Q::ℒ.QRCompactWY,
                                  qr_orm_ws,
                                  qr_orm_dims::NTuple{3, Int},
                                  qr_ws;
                                  use_fastlapack_qr::Bool = true) where {R <: AbstractFloat}
    ℒ.mul!(dest, Q.Q', src) # dest = Q' * src
    return qr_orm_ws, qr_orm_dims
end

# Old way (≤v0.1.42): F = lu(A)  — allocates a new LU factorisation object each call
function factorize_lu!(A::AbstractMatrix{R},
                       lu_ws,
                       lu_dims::NTuple{2, Int};
                       use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    if use_fastlapack_lu && R <: Union{Float32, Float64}
        dims = (size(A, 1), size(A, 2))
        if lu_dims != dims
            lu_ws = FastLapackInterface.LUWs(A)
            lu_dims = dims
        end
        _, _, info = ℒ.LAPACK.getrf!(lu_ws, A; resize = true)
        return lu_ws, lu_dims, info == 0, nothing
    else
        lu = ℒ.lu!(A, check = false)
        return lu_ws, lu_dims, ℒ.issuccess(lu), lu
    end
end

# Old way (≤v0.1.42): X = A \ B  — solves A * X = B, allocates result
function solve_lu_left!(A::AbstractMatrix{R},
                        B::AbstractVecOrMat{R},
                        lu_ws,
                        lu;
                        use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    # B ← A \ B  (overwrites B in-place)
    if use_fastlapack_lu && R <: Union{Float32, Float64}
        ℒ.LAPACK.getrs!(lu_ws, 'N', A, B)
    else
        ℒ.ldiv!(lu, B) # B = A \ B
    end
    return B
end

# B ← A \ B  (Nothing-dispatch variant, always uses LAPACK)
function solve_lu_left!(A::AbstractMatrix{R},
                        B::AbstractVecOrMat{R},
                        lu_ws,
                        lu::Nothing;
                        use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    ℒ.LAPACK.getrs!(lu_ws, 'N', A, B) # B = A \ B
    return B
end

# Old way (≤v0.1.42): X = B / A  — solves X * A = B, allocates result
function solve_lu_right!(A::AbstractMatrix{R},
                         B::AbstractMatrix{R},
                         lu_ws,
                         lu,
                         rhs_t::AbstractMatrix{R};
                         use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    # B ← B / A  (overwrites B in-place)
    if use_fastlapack_lu && R <: Union{Float32, Float64}
        rhs_t_dims = (size(B, 2), size(B, 1))
        @assert size(rhs_t) == rhs_t_dims

        copyto!(rhs_t, transpose(B))
        ℒ.LAPACK.getrs!(lu_ws, 'T', A, rhs_t)
        copyto!(B, transpose(rhs_t))
    else
        ℒ.rdiv!(B, lu) # B = B / A
    end
    return B
end

# B ← B / A  (Nothing-dispatch variant, always uses LAPACK)
function solve_lu_right!(A::AbstractMatrix{R},
                         B::AbstractMatrix{R},
                         lu_ws,
                         lu::Nothing,
                         rhs_t::AbstractMatrix{R};
                         use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    rhs_t_dims = (size(B, 2), size(B, 1))
    @assert size(rhs_t) == rhs_t_dims

    copyto!(rhs_t, transpose(B))
    ℒ.LAPACK.getrs!(lu_ws, 'T', A, rhs_t)
    copyto!(B, transpose(rhs_t))
    return B
end

# Old way (≤v0.1.42): S = schur(D, E); ordschur!(S, eigenselect)  — allocates Schur object
# Returns (qz_ws, qz_dims, schdcmp, schur_ok, has_unit_root_eigenvalues).
# has_unit_root_eigenvalues is true when any generalized eigenvalue has |λ| ∈ [1-tol, 1+tol].
function factorize_generalized_schur!(D::AbstractMatrix{R},
                                      E::AbstractMatrix{R},
                                      qz_ws,
                                      qz_dims::NTuple{2, Int},
                                      eigenselect::AbstractVector{Bool};
                                      use_fastlapack_schur::Bool = true,
                                      unit_root_tol::Float64 = 1e-8) where {R <: AbstractFloat}
    if use_fastlapack_schur && R <: Union{Float32, Float64}
        dims = (size(D, 1), size(D, 2))
        if qz_dims != dims
            qz_ws = FastLapackInterface.GeneralizedSchurWs(D)
            qz_dims = dims
        end

        try
            # FastLapackInterface.ed selects abs(lambda)^2 >= criterium.
            # Nudging just inside the unit circle avoids LAPACK reordering
            # failures for exactly unit-root blocks while preserving the
            # exterior subspace used by the QME Schur extraction.
            S, T, α, β, _, Z = ℒ.LAPACK.gges!(qz_ws, 'N', 'V', D, E;
                                              select = FastLapackInterface.ed,
                                              criterium = (1.0 - sqrt(eps(Float64)))^2,
                                              resize = true)
            has_ur = detect_unit_roots(α, β, unit_root_tol)
            return qz_ws, qz_dims, (S = S, T = T, Z = Z), true, has_ur
        catch
            return qz_ws, qz_dims, nothing, false, false
        end
    else
        schdcmp = try
            ℒ.schur!(D, E)
        catch
            return qz_ws, qz_dims, nothing, false, false
        end

        @. eigenselect = abs(schdcmp.β / schdcmp.α) < 1

        try
            ℒ.ordschur!(schdcmp, eigenselect)
        catch
            return qz_ws, qz_dims, nothing, false, false
        end

        has_ur = detect_unit_roots(schdcmp.α, schdcmp.β, unit_root_tol)
        return qz_ws, qz_dims, schdcmp, true, has_ur
    end
end

# Detect unit root eigenvalues from generalized Schur eigenvalue vectors.
# Returns true if any |α[i]/β[i]| is within tol of 1.0.
function detect_unit_roots(α::AbstractVector, β::AbstractVector, tol::Float64)::Bool
    for i in eachindex(α, β)
        βi = abs(β[i])
        βi == 0 && continue
        eig_mag = abs(α[i]) / βi
        if abs(eig_mag - 1) ≤ tol
            return true
        end
    end
    return false
end

end # dispatch_doctor
