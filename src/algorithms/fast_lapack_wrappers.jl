@stable default_mode = "disable" begin

function factorize_qr!(qr_mat::AbstractMatrix,
                       qr_factors::AbstractMatrix{R},
                       qr_ws::FastLapackInterface.QRWs{R};
                       use_fastlapack_qr::Bool = true) where {R <: Union{Float32, Float64}}
    copyto!(qr_factors, qr_mat)
    ℒ.LAPACK.geqrf!(qr_ws, qr_factors; resize = true)
    return qr_factors
end

function factorize_qr!(qr_mat::AbstractMatrix,
                       qr_factors::AbstractMatrix{R},
                       ::Nothing;
                       use_fastlapack_qr::Bool = true) where {R <: AbstractFloat}
    copyto!(qr_factors, qr_mat)
    return ℒ.qr!(qr_factors)
end

function apply_qr_transpose_left!(dest::AbstractMatrix{R},
                                  src::AbstractMatrix,
                                  Q::StridedMatrix{R},
                                  qr_orm_ws::FastLapackInterface.QROrmWs{R},
                                  qr_orm_dims::NTuple{3, Int},
                                  qr_ws::FastLapackInterface.QRWs{R};
                                  use_fastlapack_qr::Bool = true) where {R <: Union{Float32, Float64}}
    orm_dims = (size(Q, 1), size(Q, 2), size(src, 2))
    local_qr_orm_ws = qr_orm_dims == orm_dims ? qr_orm_ws : FastLapackInterface.QROrmWs(qr_ws, 'L', 'T', Q, src)
    copyto!(dest, src)
    ℒ.LAPACK.ormqr!(local_qr_orm_ws, 'L', 'T', Q, dest)
    return nothing
end

function apply_qr_transpose_left!(dest::AbstractMatrix{R},
                                  src::AbstractMatrix,
                                  Q,
                                  ::Nothing,
                                  qr_orm_dims::NTuple{3, Int},
                                  ::Nothing;
                                  use_fastlapack_qr::Bool = true) where {R <: AbstractFloat}
    ℒ.mul!(dest, Q.Q', src)
    return nothing
end

function factorize_lu!(A::AbstractMatrix{R},
                       lu_ws::FastLapackInterface.LUWs,
                       lu_dims::NTuple{2, Int};
                       use_fastlapack_lu::Bool = true) where {R <: Union{Float32, Float64}}
    dims = (size(A, 1), size(A, 2))
    if lu_dims != dims
        nothing
    end
    _, _, info = ℒ.LAPACK.getrf!(lu_ws, A; resize = true)
    return info == 0, nothing
end

function factorize_lu!(A::AbstractMatrix{R},
                       ::Nothing,
                       lu_dims::NTuple{2, Int};
                       use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    lu = ℒ.lu!(A, check = false)
    return ℒ.issuccess(lu), lu
end

function solve_lu_left!(A::AbstractMatrix{R},
                        B::AbstractVecOrMat{R},
                        lu_ws::FastLapackInterface.LUWs,
                        ::Nothing;
                        use_fastlapack_lu::Bool = true) where {R <: Union{Float32, Float64}}
    ℒ.LAPACK.getrs!(lu_ws, 'N', A, B)
    return nothing
end

function solve_lu_left!(A::AbstractMatrix{R},
                        B::AbstractVecOrMat{R},
                        ::Nothing,
                        lu::ℒ.LU;
                        use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    ℒ.ldiv!(lu, B)
    return nothing
end

function solve_lu_right!(A::AbstractMatrix{R},
                         B::AbstractMatrix{R},
                         lu_ws::FastLapackInterface.LUWs,
                         ::Nothing,
                         rhs_t::AbstractMatrix{R};
                         use_fastlapack_lu::Bool = true) where {R <: Union{Float32, Float64}}
    rhs_t_dims = (size(B, 2), size(B, 1))
    @assert size(rhs_t) == rhs_t_dims

    copyto!(rhs_t, transpose(B))
    ℒ.LAPACK.getrs!(lu_ws, 'T', A, rhs_t)
    copyto!(B, transpose(rhs_t))
    return nothing
end

function solve_lu_right!(A::AbstractMatrix{R},
                         B::AbstractMatrix{R},
                         ::Nothing,
                         lu::ℒ.LU,
                         rhs_t::AbstractMatrix{R};
                         use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    ℒ.rdiv!(B, lu)
    return nothing
end

function factorize_generalized_schur!(D::AbstractMatrix{R},
                                      E::AbstractMatrix{R},
                                      qz_ws,
                                      qz_dims::NTuple{2, Int},
                                      eigenselect::AbstractVector{Bool};
                                      use_fastlapack_schur::Bool = true) where {R <: AbstractFloat}
    if use_fastlapack_schur && R <: Union{Float32, Float64}
        dims = (size(D, 1), size(D, 2))
        if qz_dims != dims
            qz_ws = FastLapackInterface.GeneralizedSchurWs(D)
            qz_dims = dims
        end

        try
            S, T, _, _, _, Z = ℒ.LAPACK.gges!(qz_ws, 'V', 'V', D, E;
                                              select = FastLapackInterface.ed,
                                              criterium = 1.0,
                                              resize = true)
            return qz_ws, qz_dims, (S = S, T = T, Z = Z), true
        catch
            return qz_ws, qz_dims, nothing, false
        end
    else
        schdcmp = try
            ℒ.schur!(D, E)
        catch
            return qz_ws, qz_dims, nothing, false
        end

        @. eigenselect = abs(schdcmp.β / schdcmp.α) < 1

        try
            ℒ.ordschur!(schdcmp, eigenselect)
        catch
            return qz_ws, qz_dims, nothing, false
        end

        return qz_ws, qz_dims, schdcmp, true
    end
end

end # dispatch_doctor
