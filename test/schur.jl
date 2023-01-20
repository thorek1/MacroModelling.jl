using LinearAlgebra

function generalized_schur(A::AbstractMatrix, B::AbstractMatrix)
    n, _ = size(A)
    Q = Matrix{Float64}(I, n, n)
    Z = Matrix{Float64}(I, n, n)
    T = copy(A)
    for i = 1:n-1
        for k = i+1:n
            if abs(B[k,k]) < eps(Float64)
                c = 1
                s = 0
            elseif abs(B[i,i]) < eps(Float64)
                c = 0
                s = 1
            else
                if abs(B[i,i]) > abs(B[k,k])
                    tau = -B[k,k]/B[i,i]
                    s = 1/sqrt(1 + tau^2)
                    c = s*tau
                else
                    tau = -B[i,i]/B[k,k]
                    c = 1/sqrt(1 + tau^2)
                    s = c*tau
                end
            end
            G = [c s; -conj(s) c]
            T[i:k, i:i+1] = G * T[i:k, i:i+1]
            B[i:k, i:i+1] = G * B[i:k, i:i+1]
            Q[:, i:k] = Q[:, i:k] * G'
            Z[i:k, i:i+1] = G' * Z[i:k, i:i+1]
        end
    end
    return (Q, T, Z)
end



function Schur(A, B)
    eig_A = eigen(A)
    eig_B = eigen(B)
    eig_A_real, eig_A_imag = [real(eig_A.values), imag(eig_A.values)]
    eig_B_real, eig_B_imag = [real(eig_B.values), imag(eig_B.values)]
    Q = Matrix([complex(eig_A_real) complex(eig_A_imag)])
    T = Q' * (A * Q)
    Z = Q' * B * Matrix([complex(eig_B_real) complex(eig_B_imag)])
    return (T, Z)
end
A = randn(10,10)
B = randn(10,10)


res = Schur(A,B)
res_orig = schur(A,B)
using LinearAlgebra


using LinearAlgebra

function QR(A)
    m, n = size(A)
    Q = Matrix{Float64}(I, m, m)
    for i in 1:(n - (m == n))
        H = Matrix{Float64}(I, m, m)
        H[i:end, i:end] .= make_householder(A[i:end, i])
        Q *= H
        A *= H
    end
    return Q, A
end

function make_householder(a)
    v = a / (a[1] + copysign(norm(a), a[1]))
    v[1] = 1
    # H = Matrix{Float64}(I, length(a), length(a))
    H = -(2 * inv(v * v')) * (v * v')
    return H
end

# task 1: show qr decomp of wp example
a = [12 -51   4;
     6 167 -68;
    -4  24 -41]

    q, r = qr(a)
    Q, R = QR(a)
    Q
    q
    R
    r
println("q:")
println(round.(q, 6))
println("r:")
println(round.(r, 6))

# task 2: use qr decomp for polynomial regression example
function polyfit(x, y, n)
    return lsqr(x.^(0:n), y)
end

function lsqr(a, b)
    q, r = qr(a)
    m, n = size(r)
    return r[1:n, :] \ (q' * b)[1:n]
end

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321]

println("\npolyfit:")
println(polyfit(x, y, 2))

qr_givens(A)



using LinearAlgebra

function givens_rotation(A)
    num_rows, num_cols = size(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = Matrix{Float64}(I, num_rows, num_cols)
    R = copy(A)

    # Iterate over lower triangular matrix.
    idx = findall(tril(ones(num_rows, num_cols), -1) .* R .!= 0)

    for rowcol in idx
        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        r = hypot(R[rowcol[2], rowcol[2]], R[rowcol[1], rowcol[2]])
        c = R[rowcol[2], rowcol[2]]/r
        s = -R[rowcol[1], rowcol[2]]/r

        G = Matrix{Float64}(I, num_rows, num_cols)
        G[rowcol[2], rowcol[2]] = c
        G[rowcol[1], rowcol[1]] = c
        G[rowcol[1], rowcol[2]] = s
        G[rowcol[2], rowcol[1]] = -s

        R = G * R
        Q = Q * G'
    end

    return (Q, R)
end


A = rand(6,6)


num_rows, num_cols = size(A)

# Initialize orthogonal matrix Q and upper triangular matrix R.
Q = Matrix{Float64}(I, num_rows, num_cols)
R = copy(A)

# Iterate over lower triangular matrix.
idx = findall(tril(ones(num_rows, num_cols), -1) .* R .!= 0)

using BenchmarkTools
@benchmark q,r = givens_rotation(A)
q
r
q * r
@benchmark Q,R = qr(A)
Q * R ≈ q * r



using GenericSchur
import GenericSchur: ggschur!, schur

schur!(A,B)
GenericSchur.schur(A,B)


using LinearAlgebra
schur(A,B)
schur(A)




# This file is part of GenericSchur.jl, released under the MIT "Expat" license
# Portions derived from LAPACK, see below.

# Schur decomposition (QZ algorithm) for generalized eigen-problems

# Developer notes:
# We don't currently implement GeneralizedHessenberg types so anyone else
# using internal functions is doomed to confusion.

using LinearAlgebra: givensAlgorithm
# export ggschur!, schur!
# decomposition is A = Q S Z', B = Q  Tmat Z'

function schur!(A::StridedMatrix{T}, B::StridedMatrix{T}; kwargs...
                ) where {T<:AbstractFloat}
    ggschur!(A, B; kwargs...)
end

# ggschur! is similar to LAPACK zgges:
# Q is Vsl, Z is Vsr
# Tmat overwrites B
# S overwrites A

function ggschur!(A::StridedMatrix{T}, B::StridedMatrix{T};
                  wantQ::Bool=true, wantZ::Bool=true,
                  scale::Bool=true,
                  kwargs...) where T <: Complex

    n,_ = size(A)
    nb,_ = size(B)
    if nb != n
        throw(ArgumentError("matrices must have the same sizes"))
    end

    if scale
        scaleA, cscale, anrm = _scale!(A)
        scaleB, cscaleb, bnrm = _scale!(B)
    else
        scaleA = false
        scaleB = false
    end
    # maybe balance here

    bqr = qr!(B)
    # apply bqr.Q to A
    lmul!(bqr.Q', A)
    if wantQ
        Q = Matrix(bqr.Q)
    else
        Q = Matrix{T}(undef, 0, 0)
    end
    if wantZ
        Z = Matrix{T}(I, n, n)
    else
        Z =  Matrix{T}(undef, 0, 0)
    end

    # materializing R may waste memory; can we rely on storage in modified B?
    A, B, Q, Z = _hessenberg!(A, bqr.R, Q, Z)
    α, β, S, Tmat, Q, Z = _gqz!(A, B, Q, Z, true; kwargs...)

    # TODO if balancing, unbalance Q, Z

    if scaleA
        safescale!(S, cscale, anrm)
        # checkme: is this always correct?
        α = diag(S)
    end
    if scaleB
        safescale!(Tmat, cscaleb, bnrm)
        # checkme: is this always correct?
        β = diag(Tmat)
    end

    return GeneralizedSchur(S, Tmat, α, β, Q, Z)
end

# B temporarily loses triangularity, so is not specially typed.
# compare to zgghrd
function _hessenberg!(A::StridedMatrix{T}, B::StridedMatrix{T}, Q, Z;
                      ilo=1, ihi=size(A,1)
                      ) where T
    n, _ = size(A)
    wantQ = !isempty(Q)
    wantZ = !isempty(Z)

    triu!(B)

    for jc = ilo:ihi-2
        for jr = ihi:-1:jc+2
            # rotate rows jr-1,jr to null A[jr,jc]
            Gq,r = givens(A,jr-1,jr,jc)
            lmul!(Gq, A)
            lmul!(Gq, B)
            if wantQ
                rmul!(Q, Gq')
            end
            # rotate cols jr,jr-1 to null B[jr,jr-1]
            Gz,r = givens(B',jr,jr-1,jr)
            rmul!(A,Gz')
            rmul!(B,Gz')
            if wantZ
                rmul!(Z, Gz')
            end
        end
    end

    return triu!(A,-1), triu!(B), Q, Z
end

# single-shift QZ algo
#
# translated from LAPACK's zhgeqz
# LAPACK is released under a BSD license, and is
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
#
# H is Hessenberg
# B is upper-triangular
# This is an internal routine so we don't check.
function _gqz!(H::StridedMatrix{T}, B::StridedMatrix{T}, Q, Z, wantSchur;
               debug = false,
               ilo=1, ihi=size(H,1), maxiter=100*(ihi-ilo+1)
               ) where {T <: AbstractFloat}
    n, _ = size(H)
    wantQ = !isempty(Q)
    wantZ = !isempty(Z)

    if ilo > 1 || ihi < n
        @warn "almost surely not working for nontrivial ilo,ihi"
    end
    α = Vector{T}(undef, n)
    β = Vector{T}(undef, n)

    RT = real(T)
    ulp = eps(RT)
    safmin = safemin(RT)
    anorm = norm(view(H,ilo:ihi,ilo:ihi),2) # Frobenius
    bnorm = norm(view(B,ilo:ihi,ilo:ihi),2) # Frobenius
    atol = max(safmin, ulp * anorm)
    btol = max(safmin, ulp * bnorm)
    ascale = one(RT) / max(safmin, anorm)
    bscale = one(RT) / max(safmin, bnorm)
    half = 1 / RT(2)

    # TODO: if ihi < n, deal with ihi+1:n

    ifirst = ilo
    ilast = ihi

    if wantSchur
        ifirstm, ilastm = 1,n
    else
        ifirstm, ilastm = ilo,ihi
    end

    shiftcount = 0
    eshift = zero(T)

    niter = 0
    while true
        niter += 1
        if niter > maxiter
            @warn "convergence failure; factorization incomplete at ilast=$ilast"
            break
        end

        # warning: _istart, ifirst, ilast are as in LAPACK, not GLA
        # note zlartg(f,g,c,s,r) -> (c,s,r) = givensAlgorithm(f,g)

        # Split H if possible
        # Use 2 tests:
        # 1: H[j,j-1] == 0 || j=ilo
        # 2: B[j,j] = 0

        null_Btail = false # branch 50 in zhgeqz
        deflate_H = false  # branch 60 in zhgeqz
        ns_Bblock_found = false # branch 70 in zhgeqz

        if ilast == ilo
            deflate_H = true
        elseif abs1(H[ilast,ilast-1]) < atol
            H[ilast, ilast-1] = 0
            deflate_H = true
        end
        # For now, we follow LAPACK quite closely.
        # br is a branch indicator corresponding to goto targets
        br = deflate_H ? :br60 : :br_none
        if !deflate_H
            # deflate_H is only tentatively false at this point
            if abs(B[ilast,ilast]) <= btol
                B[ilast,ilast] = 0
                null_Btail = true
                br = :br50
            else
                # general case
                for j=ilast-1:-1:ilo
                    # Test 1
                    split_test1 = false
                    if j==ilo
                        split_test1 = true
                    else
                        if abs1(H[j,j-1]) <= atol
                            H[j,j-1] = 0
                            split_test1 = true
                        end
                    end
                    # Test 2
                    split_test2 = abs(B[j,j]) < btol
                    if split_test2
                        B[j,j] = 0
                        # Test 1a: check for 2 consec. small subdiags in H
                        split_test1a = !split_test1 &&
                            (abs1(H[j,j-1]) * (ascale * abs1(H[j+1,j]))
                             <= abs1(H[j,j]) * (ascale * atol))
                        # if both tests pass, split 1x1 block off top
                        # if remaining leading diagonal elt vanishes, iterate
                        if split_test1 || split_test1a
                            # @mydebug println("splitting 1x1 at top (j=$j)")
                            for jch=j:ilast-1
                                G,r = givens(H[jch,jch],H[jch+1,jch],jch,jch+1)
                                H[jch,jch] = r
                                H[jch+1,jch] = 0
                                lmul!(G,view(H,:,jch+1:ilastm))
                                lmul!(G,view(B,:,jch+1:ilastm))
                                if wantQ
                                    rmul!(Q, G')
                                end
                                if split_test1a
                                    H[jch,jch-1] *= G.c
                                end
                                split_test1a = false
                                if abs1(B[jch+1, jch+1]) > btol
                                    if jch+1 >= ilast
                                        deflate_H = true
                                        br = :br60
                                        break
                                    else
                                        ifirst = jch+1
                                        ns_Bblock_found = true
                                        br = :br70
                                        break
                                    end
                                end
                                B[jch+1, jch+1] = 0
                            end # jch loop
                            if !ns_Bblock_found && !deflate_H
                                null_Btail = true
                                br = :br50
                            end
                            break
                        else
                            # only test 2 passed: chase 0 to B[ilast,ilast]
                            # then process as above
                            # @mydebug println("chasing diag 0 in B from $j to $ilast")
                            for jch=j:ilast-1
                                G,r = givens(B[jch,jch+1],B[jch+1,jch+1],
                                             jch,jch+1)
                                B[jch,jch+1] = r
                                B[jch+1,jch+1] = 0
                                if jch < ilastm-1
                                    lmul!(G,view(B,:,jch+2:ilastm))
                                end
                                lmul!(G,view(H,:,jch-1:ilastm))
                                if wantQ
                                    rmul!(Q,G')
                                end
                                G,r = givens(conj(H[jch+1,jch]),conj(H[jch+1,jch-1]),
                                             jch,jch-1)
                                H[jch+1,jch] = r'
                                H[jch+1,jch-1] = 0
                                rmul!(view(H,ifirstm:jch,:),G')
                                rmul!(view(B,ifirstm:jch+1,:),G')
                                if wantZ
                                    rmul!(Z,G')
                                end
                            end # jch loop
                            null_Btail = true
                            br = :br50
                        end # if split_test1 || split_test1a
                    elseif split_test1 # but not split_test2, i.e. tiny B[j,j]
                        ifirst = j
                        ns_Bblock_found = true
                        br = :br70
                        break
                    end # if B[j,j] tiny
                    # neither test passed; try next j
                end # j loop
                # @mydebug println("after j loop branch $br")
                if br == :br_none
                    error("dev error: drop-through assumed impossible")
                end
            end # B[ilast,ilast] is/is-not tiny branches
            if null_Btail # br == :br50
                # B[ilast,ilast] is 0; clear H[ilast,ilast-1] to split
                G,r = givens(conj(H[ilast,ilast]),conj(H[ilast,ilast-1]),ilast,ilast-1)
                H[ilast,ilast] = r'
                H[ilast,ilast-1] = 0
                rmul!(view(H,ifirstm:ilast-1,:),G')
                rmul!(view(B,ifirstm:ilast-1,:),G')
                if wantZ
                    rmul!(Z,G')
                end
                deflate_H = true
                br = :br60
            end
        end # if trivial split
        if deflate_H
            # H[ilast, ilast-1] == 0: standardize B, set α,β
            absb = abs(B[ilast,ilast])
            # @mydebug println("deflation ilast=$ilast, |B_tail|=$absb")
            if absb > safmin
                signbc = conj(B[ilast,ilast] / absb)
                B[ilast,ilast] = absb
                if wantSchur
                    B[ifirstm:ilast-1,ilast] .*= signbc
                    H[ifirstm:ilast,ilast] .*= signbc
                else
                    H[ilast,ilast] *= signbc
                end
                if wantZ
                    Z[:,ilast] .*= signbc
                end
            else
                B[ilast,ilast] = 0
            end
            α[ilast] = H[ilast,ilast]
            β[ilast] = B[ilast,ilast]
            ilast -= 1
            if ilast < ilo
                # we are done; exit outer loop
                break
            end
            shiftcount = 0
            eshift = zero(T)
            if !wantSchur
                ilastm = ilast
                if ifirstm > ilast
                    ifirstm = ilo
                end
            end
            continue # iteration loop
        end # deflation branch

        @assert ns_Bblock_found

        # QZ step
        # This iteration involves block ifirst:ilast, assumed nonempty
        if !wantSchur
            ifirstm = ifirst
        end

        # Compute shift
        shiftcount += 1
        # At this point ifirst < ilast and diagonal elts of B in that block
        # are all nontrivial.
        if shiftcount % 10 !=  0
            # Wilkinson shift, i.e. eigenvalue of last 2x2 block of
            # A B⁻¹ nearest to last elt

            # factor B as U D where U has unit diagonal, compute A D⁻¹ U⁻¹
            b11 = (bscale * B[ilast-1,ilast-1])
            b22 = (bscale * B[ilast,ilast])
            u12 = (bscale * B[ilast-1,ilast]) / b22
            ad11 = (ascale * H[ilast-1,ilast-1]) / b11
            ad21 = (ascale * H[ilast,ilast-1]) / b11
            ad12 = (ascale * H[ilast-1,ilast]) / b22
            ad22 = (ascale * H[ilast,ilast]) / b22
            abi22 = ad22 - u12 * ad21
            t1 = half * (ad11 + abi22)
            rtdisc = sqrt(t1^2 + ad12 * ad21 - ad11 * ad22)
            t2 = real(t1 - abi22) * real(rtdisc) + imag(t1-abi22) * imag(rtdisc)
            shift =  (t2 <= 0) ? (t1 + rtdisc) : (t1 - rtdisc)
            if isnan(shift)
                shift = zero(T)
                @warn "logic error in generalized Schur, check results"
                # @mydebug println("NaN encountered bscale=$bscale " *
                #                  "b11=$(B[ilast-1,ilast-1]) " *
                #                  "b22=$(B[ilast,ilast])")
            end
        else
            # exceptional shift
            # "Chosen for no particularly good reason" (LAPACK)
            eshift += (ascale * H[ilast, ilast-1]) / (bscale * B[ilast-1, ilast-1])
            if isnan(eshift)
                @warn "logic error in generalized Schur, check results"
                # @mydebug println("NaN encountered bscale=$bscale " *
                #                  "b11=$(B[ilast-1,ilast-1]) ")
                eshift = zero(T)
            end
            shift = eshift
        end

        # check for two consecutive small subdiagonals
        local f1
        gotit = false
        for j=ilast-1:-1:ifirst+1
            _istart = j
            f1 = ascale * H[j,j] - shift * (bscale * B[j,j])
            t1 = abs1(f1)
            t2 = ascale * abs1(H[j+1,j])
            tr = max(t1,t2)
            if tr < one(RT) && tr != zero(RT)
                t1 /= tr
                t2 /= tr
            end
            if abs1(H[j,j-1]) * t2 <= t1 * atol
                gotit = true
                break
            end
        end
        if !gotit
            _istart = ifirst
            f1 = (ascale * H[ifirst, ifirst]
                  - shift * (bscale * B[ifirst, ifirst]))
        end

        # qz sweep
        # @mydebug println("QZ sweep range $(_istart):$ilast shift=$shift")
        g2 = ascale*H[_istart+1, _istart]
        c,s,t3 = givensAlgorithm(f1, g2)
        for j=_istart:ilast-1
            if j > _istart
                    c,s,r = givensAlgorithm(H[j,j-1], H[j+1,j-1])
                    H[j,j-1] = r
                    H[j+1,j-1] = 0
            end
            G = Givens(j,j+1,T(c),s)
            lmul!(G, view(H,:,j:ilastm))
            lmul!(G, view(B,:,j:ilastm))
            if wantQ
                rmul!(Q, G')
            end
            c,s,r = givensAlgorithm(B[j+1,j+1], B[j+1,j])
            B[j+1,j+1] = r
            B[j+1,j] = 0
            G = Givens(j,j+1,T(c),s)
            rmul!(view(H,ifirstm:min(j+2,ilast),:),G)
            rmul!(view(B,ifirstm:j,:),G)
            if wantZ
                rmul!(Z, G)
            end
        end
    end # iteration loop
    # @mydebug println("ggschur! done in $niter iters")
    # TODO if ilo > 1, deal with 1:ilo-1

    return α, β, H, B, Q, Z
end

struct GeneralizedSchur{Ty,M<:AbstractMatrix,A<:AbstractVector,B<:AbstractVector{Ty}}# <: Factorization{Ty}
    S::M
    T::M
    α::A
    β::B
    Q::M
    Z::M
    # function GeneralizedSchur{Ty,M,A,B}(S::AbstractMatrix{Ty}, T::AbstractMatrix{Ty},
    #                                     alpha::AbstractVector, beta::AbstractVector{Ty},
    #                                     Q::AbstractMatrix{Ty}, Z::AbstractMatrix{Ty}) where {Ty,M,A,B}
    #     new{Ty,M,A,B}(S, T, alpha, beta, Q, Z)
    # end
end


function eigvecs(S::GeneralizedSchur{Complex{T}}, left=false) where {T <: AbstractFloat}
    left && throw(ArgumentError("not implemented"))
    v = _geigvecs(S.S, S.T, S.Z)
    # CHECKME: Euclidean norm differs from LAPACK, so wait for upstream.
    # _enormalize!(v)
    return v
end


# right eigenvectors
# translated from LAPACK::ztgevc
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _geigvecs(S::StridedMatrix{T}, P::StridedMatrix{T},
                    Z::StridedMatrix{T}=Matrix{T}(undef,0,0)
                    ) where {T <: Complex}
    n = size(S,1)
    RT = real(T)
    ulp = eps(RT)
    safmin = safemin(RT)
    smallnum = safmin * (n / ulp)
    bigx = one(RT) / smallnum
    bignum = one(RT) / (n * safmin)

    vectors = Matrix{T}(undef,n,n)
    v = zeros(T,n)
    if size(Z,1) > 0
        v2 = zeros(T,n)
    end

    # We use the 1-norms of the strictly upper part of S, P columns
    # to avoid overflow
    anorm = abs1(S[1,1])
    bnorm = abs1(P[1,1])
    snorms = zeros(RT,n)
    pnorms = zeros(RT,n)
    @inbounds for j=2:n
        for i=1:j-1
            snorms[j] += abs1(S[i,j])
            pnorms[j] += abs1(P[i,j])
        end
        anorm = max(anorm,snorms[j] + abs1(S[j,j]))
        bnorm = max(bnorm,pnorms[j] + abs1(P[j,j]))
    end
    ascale = one(RT) / max(anorm, safmin)
    bscale = one(RT) / max(bnorm, safmin)

    idx = n+1
    for ki=n:-1:1
        idx -= 1
        if abs1(S[ki,ki]) <= safmin && abs(real(P[ki,ki])) <= safmin
            # singular pencil; return unit eigenvector
            vectors[:,idx] .= zero(T)
            vectors[idx,idx] = one(T)
        else
            # compute coeffs a,b in (a A - b B) x = 0
            t = 1 / max(abs1(S[ki,ki]) * ascale,
                        abs(real(P[ki,ki])) * bscale, safmin)
            sα = (t * S[ki,ki]) * ascale
            sβ = (t * real(P[ki,ki])) * bscale
            acoeff = sβ * ascale
            bcoeff = sα * bscale
            # scale to avoid underflow

            lsa = abs(sβ) >= safmin && abs(acoeff) < smallnum
            lsb = abs1(sα) >= safmin && abs1(bcoeff) < smallnum
            s = one(RT)
            if lsa
                s = (smallnum / abs(sβ)) * min(anorm, bigx)
            end
            if lsb
                s = max(s, (smallnum / abs1(sα)) * min(bnorm, bigx))
            end
            if lsa || lsb
                s = min(s, 1 / (safmin * max( one(RT), abs(acoeff),
                                              abs1(bcoeff))))
                if lsa
                    acoeff = ascale * (s * sβ)
                else
                    acoeff *= s
                end
                if lsb
                    bcoeff = bscale * (s * sα)
                else
                    bcoeff *= s
                end
            end
            aca = abs(acoeff)
            acb = abs1(bcoeff)
            xmax = one(RT)
            v .= zero(T)
            dmin = max(ulp * aca * anorm, ulp * acb * bnorm, safmin)

            # triangular solve of (a A - b B) x = 0, columnwise
            # v[1:j-1] contains sums w
            # v[j+1:ki] contains x

            v[1:ki-1] .= acoeff * S[1:ki-1,ki] - bcoeff * P[1:ki-1,ki]
            v[ki] = one(T)

            for j=ki-1:-1:1
                # form x[j] = -v[j] / d
                # with scaling and perturbation
                d = acoeff * S[j,j] - bcoeff * P[j,j]
                if abs1(d) <= dmin
                    d = complex(dmin)
                end
                if abs1(d) < one(RT)
                    if abs1(v[j]) >= bignum * abs1(d)
                        t = 1 / abs1(v[j])
                        v[1:ki] *= t
                    end
                end
                v[j] = -v[j] / d
                if j > 1
                    # w = w + x[j] * (a S[:,j] - b P[:,j]) with scaling

                    if abs1(v[j]) > one(RT)
                        t = 1 / abs1(v[j])
                        if aca * snorms[j] + acb * pnorms[j] >= bignum * t
                            v[1:ki] *= t
                        end
                    end
                    ca = acoeff * v[j]
                    cb = bcoeff * v[j]
                    v[1:j-1] += ca * S[1:j-1,j] - cb * P[1:j-1,j]
                end
            end # for j (solve loop)
            if size(Z,1) > 0
                mul!(v2, Z, v)
                v, v2 = v2, v
                iend = n
            else
                iend = ki
            end

            xmax = zero(RT)
            for jr=1:iend
                xmax = max(xmax, abs1(v[jr]))
            end
            if xmax > safmin
                t = 1 / xmax
                vectors[1:iend,idx] .= t * v[1:iend]
            else
                iend = 0
            end
            vectors[iend+1:n,idx] .= zero(T)
        end # nonsingular branch
    end # index loop

    return vectors
end # function

"""
    canonicalize!(S::GeneralizedSchur)

rescale the fields of `S` so that `S.β` and the diagonal of `S.T` are real.
"""
function canonicalize!(S::GeneralizedSchur{Ty}) where Ty
    n = size(S.S,1)
    sf = safemin(real(Ty))
    for k=1:n
        scale = abs(S.T[k,k])
        if scale > sf
            t1 = conj(S.T[k,k] / scale)
            t2 = S.T[k,k] / scale
            S.T[k,k+1:n] .*= t1
            S.S[k,k:n] .*= t1
            if !(S.Q === nothing) && !isempty(S.Q)
                S.Q[:,k] .*= t2
            end
        else
            S.T[k,k] = zero(Ty)
        end
        S.α[k] = S.S[k,k]
        S.β[k] = S.T[k,k]
    end
    S
end

function _scale!(A::AbstractArray{T}) where {T}
    smlnum = sqrt(safemin(real(T))) / eps(real(T))
    bignum = 1 / smlnum
    anrm = norm(A,Inf)
    scaleA = false
    cscale = one(real(T))
    if anrm > 0 && anrm < smlnum
        scaleA = true
        cscale = smlnum
    elseif anrm > bignum
        scaleA = true
        cscale = bignum
    end
    scaleA && safescale!(A, anrm, cscale)
    scaleA, cscale, anrm
end


function safemin(T)
    sfmin = floatmin(T)
    small = one(T) / floatmax(T)
    if small >= sfmin
        sfmin = small * (one(T) + eps(T))
    end
    sfmin
end

abs1(z) = abs(z)
abs1(z::T) where {T <: Complex} = abs(real(z))+abs(imag(z))


using LinearAlgebra: norm, qr!, lmul!, I, triu!, givens, rmul!, Givens

A = rand(5,5)
B = rand(5,5)

schur!(complex(A),complex(B))