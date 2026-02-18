@stable default_mode = "disable" begin

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

function apply_qr_transpose_left!(dest::AbstractMatrix{R},
                                  src::AbstractMatrix,
                                  Q,
                                  qr_orm_ws,
                                  qr_orm_dims::NTuple{3, Int},
                                  qr_ws;
                                  use_fastlapack_qr::Bool = true) where {R <: AbstractFloat}
    if use_fastlapack_qr && R <: Union{Float32, Float64}
        orm_dims = (size(Q, 1), size(Q, 2), size(src, 2))
        if qr_orm_dims != orm_dims
            qr_orm_ws = FastLapackInterface.QROrmWs(qr_ws, 'L', 'T', Q, src)
            qr_orm_dims = orm_dims
        end

        copyto!(dest, src)
        ℒ.LAPACK.ormqr!(qr_orm_ws, 'L', 'T', Q, dest)
        return qr_orm_ws, qr_orm_dims
    else
        ℒ.mul!(dest, Q.Q', src)
        return qr_orm_ws, qr_orm_dims
    end
end

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

function solve_lu_left!(A::AbstractMatrix{R},
                        B::AbstractVecOrMat{R},
                        lu_ws,
                        lu;
                        use_fastlapack_lu::Bool = true) where {R <: AbstractFloat}
    if use_fastlapack_lu && R <: Union{Float32, Float64}
        ℒ.LAPACK.getrs!(lu_ws, 'N', A, B)
    else
        ℒ.ldiv!(lu, B)
    end
    return B
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
