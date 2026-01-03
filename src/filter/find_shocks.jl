# Algorithms
# - LagrangeNewton: fast, but no guarantee of convergence to global minimum
# - COBYLA: best known chances of convergence to global minimum; ok speed for third order; lower tol on optimality conditions (1e-7)
# - SLSQP: relatively slow and not guaranteed to converge to global minimum

@stable default_mode = "disable" begin
function find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 1000,
                    tol::Float64 = 1e-13) # will fail for higher or lower precision
    x = copy(initial_guess)
    
    λ = zeros(size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]

    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 

    norm2 = 1.0
    
    Δnorm = 1e12

    x̂ = copy(shock_independent)

    x̄ = zeros(size(𝐒ⁱ,2))

    ∂x = zero(𝐒ⁱ)
    
    fxλ = zeros(length(xλ))
    
    fxλp = zeros(length(xλ), length(xλ))

    tmp = zeros(size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    lI = -2 * vec(ℒ.I(size(𝐒ⁱ, 2)))

    @inbounds for i in 1:max_iter
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
        ℒ.axpby!(1, 𝐒ⁱ, 2, ∂x)

        ℒ.mul!(x̄, ∂x', λ)
        
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ,2))
        copyto!(fxλ, size(𝐒ⁱ,2) + 1, x̂, 1, size(shock_independent,1))
        
        # fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * λ - 2 * x
                # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x))]

        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ)
        ℒ.axpby!(1, lI, 2, tmp)

        fxλp[1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        fxλp[1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = ∂x'

        ℒ.rmul!(∂x, -1)
        fxλp[size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = ∂x

        # fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))'
        #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        
        # f̂xλp = ℒ.lu(fxλp, check = false)

        # if !ℒ.issuccess(f̂xλp)
        #     return x, false
        # end

        try
            f̂xλp = ℒ.factorize(fxλp)
            ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        catch
            # ℒ.svd(fxλp)
            # println("factorization fails")
            return x, false
        end
        
        if !all(isfinite,Δxλ) break end
        
        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ
    
        # x = xλ[1:size(𝐒ⁱ, 2)]
        copyto!(x, 1, xλ, 1, size(𝐒ⁱ,2))

        # λ = xλ[size(𝐒ⁱ, 2)+1:end]
        copyto!(λ, 1, xλ, size(𝐒ⁱ,2) + 1, length(λ))

        ℒ.kron!(kron_buffer, x, x)

        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)

        ℒ.mul!(x̂, 𝐒ⁱ, x, 1, 1)

        norm2 = ℒ.norm(x̂)

        ℒ.axpby!(1, shock_independent, -1, x̂)

        if ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
            # println("LagrangeNewton: $i, Tol reached, $x")
            break
        end

        # if i > 500 && ℒ.norm(Δxλ) > 1e-11 && ℒ.norm(Δxλ) > Δnorm
        #     # println("LagrangeNewton: $i, Norm increase")
        #     return x, false
        # end
        # # if i == max_iter
        #     println("LagrangeNewton: $i, Max iter reached")
            # println(ℒ.norm(Δxλ) / ℒ.norm(xλ))
        # end
    end

    # println(λ)
    # println("Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
    # println(ℒ.norm(Δxλ))
    # println(ℒ.norm(Δxλ) / ℒ.norm(xλ))
    # if !(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol))
    #     println("Find shocks failed. Norm 1: $(ℒ.norm(x̂) / max(norm1,norm2)); Norm 2: $(ℒ.norm(Δxλ) / ℒ.norm(xλ))")
    # end

    return x, ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
end

end # dispatch_doctor

function rrule(::typeof(find_shocks),
                ::Val{:LagrangeNewton},
                initial_guess::Vector{Float64},
                kron_buffer::Vector{Float64},
                kron_buffer2::AbstractMatrix{Float64},
                J::ℒ.Diagonal{Bool, Vector{Bool}},
                𝐒ⁱ::AbstractMatrix{Float64},
                𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                shock_independent::Vector{Float64};
                max_iter::Int = 1000,
                tol::Float64 = 1e-13)

    x, matched = find_shocks(Val(:LagrangeNewton),
                            initial_guess,
                            kron_buffer,
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent,
                            max_iter = max_iter,
                            tol = tol)

    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x)

    λ = tmp' \ x * 2

    fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
    -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

    ℒ.kron!(kron_buffer, x, x)

    xλ = ℒ.kron(x,λ)


    ∂shock_independent = similar(shock_independent)

    # ∂𝐒ⁱ = similar(𝐒ⁱ)

    # ∂𝐒ⁱ²ᵉ = similar(𝐒ⁱ²ᵉ)

    function find_shocks_pullback(∂x)
        ∂x = vcat(∂x[1], zero(λ))

        S = -fXλp' \ ∂x

        copyto!(∂shock_independent, S[length(initial_guess)+1:end])
        
        # copyto!(∂𝐒ⁱ, ℒ.kron(S[1:length(initial_guess)], λ) - ℒ.kron(x, S[length(initial_guess)+1:end]))
        ∂𝐒ⁱ = S[1:length(initial_guess)] * λ' - S[length(initial_guess)+1:end] * x'
        
        # copyto!(∂𝐒ⁱ²ᵉ, 2 * ℒ.kron(S[1:length(initial_guess)], xλ) - ℒ.kron(kron_buffer, S[length(initial_guess)+1:end]))
        ∂𝐒ⁱ²ᵉ = 2 * S[1:length(initial_guess)] * xλ' - S[length(initial_guess)+1:end] * kron_buffer'

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒ⁱ, ∂𝐒ⁱ²ᵉ, ∂shock_independent, NoTangent(), NoTangent()
    end

    return (x, matched), find_shocks_pullback
end


@stable default_mode = "disable" begin

function find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    kron_buffer4::AbstractMatrix{Float64},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 1000,
                    tol::Float64 = 1e-13) # will fail for higher or lower precision
    x = copy(initial_guess)

    λ = zeros(size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]

    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 

    norm2 = 1.0
    
    Δnorm = 1e12

    x̂ = copy(shock_independent)

    x̄ = zeros(size(𝐒ⁱ,2))

    ∂x = zero(𝐒ⁱ)

    ∂x̂ = zero(𝐒ⁱ)
    
    fxλ = zeros(length(xλ))
    
    fxλp = zeros(length(xλ), length(xλ))

    tmp = zeros(size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    tmp2 = zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    II = sparse(ℒ.I(length(x)^2))

    lI = -2 * vec(ℒ.I(size(𝐒ⁱ, 2)))
    
    @inbounds for i in 1:max_iter
        ℒ.kron!(kron_buffer2, J, x)
        ℒ.kron!(kron_buffer3, J, kron_buffer)

        copy!(∂x, 𝐒ⁱ)
        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)

        ℒ.mul!(∂x, 𝐒ⁱ³ᵉ, kron_buffer3, 3, 1)

        ℒ.mul!(x̄, ∂x', λ)
        
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ,2))
        copyto!(fxλ, size(𝐒ⁱ,2) + 1, x̂, 1, size(shock_independent,1))
        # fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
                # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]
        
        x_kron_II!(kron_buffer4, x)
        # ℒ.kron!(kron_buffer4, II, x)
        ℒ.mul!(tmp2, 𝐒ⁱ³ᵉ, kron_buffer4)
        ℒ.mul!(tmp, tmp2', λ)
        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ, 2, 6)
        ℒ.axpy!(1,lI,tmp)

        fxλp[1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        
        fxλp[1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = ∂x'

        ℒ.rmul!(∂x, -1)
        fxλp[size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = ∂x
        # fXλp = [reshape((2 * 𝐒ⁱ²ᵉ + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))'
        #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        
        try
            f̂xλp = ℒ.factorize(fxλp)
            ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        catch
            # ℒ.svd(fxλp)
            # println("factorization fails")
            return x, false
        end
        
        if !all(isfinite,Δxλ) break end
        
        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ
    
        # x = xλ[1:size(𝐒ⁱ, 2)]
        copyto!(x, 1, xλ, 1, size(𝐒ⁱ,2))

        # λ = xλ[size(𝐒ⁱ, 2)+1:end]
        copyto!(λ, 1, xλ, size(𝐒ⁱ,2) + 1, length(λ))

        ℒ.kron!(kron_buffer, x, x)

        ℒ.kron!(kron_buffer², x, kron_buffer)

        ℒ.mul!(x̂, 𝐒ⁱ, x)

        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

        ℒ.mul!(x̂, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

        norm2 = ℒ.norm(x̂)

        ℒ.axpby!(1, shock_independent, -1, x̂)

        if ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
            # println("LagrangeNewton: $i, Tol: $(ℒ.norm(Δxλ) / ℒ.norm(xλ)) reached, x: $x")
            break
        end

        # if i > 500 && ℒ.norm(Δxλ) > 1e-11 && ℒ.norm(Δxλ) > Δnorm
        #     # println(ℒ.norm(Δxλ))
        #     # println(ℒ.norm(x̂) / max(norm1,norm2))
        #     # println("LagrangeNewton: $i, Norm increase")
        #     return x, false
        # end
        # if i == max_iter
        #     println("LagrangeNewton: $i, Max iter reached")
        #     # println(ℒ.norm(Δxλ))
        # end
    end

    # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
    # println("LagrangeNewton: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
    # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")

    # println(ℒ.norm(x))
    # println(x)
    # println(λ)
    # println([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
    # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))])
    # println(fxλp)
    # println(reshape(tmp, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2)))
    # println([reshape((2 * 𝐒ⁱ²ᵉ - 2 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))'
    #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))])
    # println(fxλp)
    # println("Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
    # println(ℒ.norm(Δxλ))
    # println(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < tol)

    # if !(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol))
    #     println("Find shocks failed. Norm 1: $(ℒ.norm(x̂) / max(norm1,norm2)); Norm 2: $(ℒ.norm(Δxλ) / ℒ.norm(xλ))")
    # end

    return x, ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
end


end # dispatch_doctor


function rrule(::typeof(find_shocks),
                ::Val{:LagrangeNewton},
                initial_guess::Vector{Float64},
                kron_buffer::Vector{Float64},
                kron_buffer²::Vector{Float64},
                kron_buffer2::AbstractMatrix{Float64},
                kron_buffer3::AbstractMatrix{Float64},
                kron_buffer4::AbstractMatrix{Float64},
                J::ℒ.Diagonal{Bool, Vector{Bool}},
                𝐒ⁱ::AbstractMatrix{Float64},
                𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                shock_independent::Vector{Float64};
                max_iter::Int = 1000,
                tol::Float64 = 1e-13)

    x, matched = find_shocks(Val(:LagrangeNewton),
                            initial_guess,
                            kron_buffer,
                            kron_buffer²,
                            kron_buffer2,
                            kron_buffer3,
                            kron_buffer4,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            𝐒ⁱ³ᵉ,
                            shock_independent,
                            max_iter = max_iter,
                            tol = tol)

    ℒ.kron!(kron_buffer, x, x)

    ℒ.kron!(kron_buffer², x, kron_buffer)

    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer)

    λ = tmp' \ x * 2

    fXλp = [reshape((2 * 𝐒ⁱ²ᵉ + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
    -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

    xλ = ℒ.kron(x,λ)

    xxλ = ℒ.kron(x,xλ)

    function find_shocks_pullback(∂x)
        ∂x = vcat(∂x[1], zero(λ))

        S = -fXλp' \ ∂x

        ∂shock_independent = S[length(initial_guess)+1:end]
        
        ∂𝐒ⁱ = ℒ.kron(S[1:length(initial_guess)], λ) - ℒ.kron(x, S[length(initial_guess)+1:end])

        ∂𝐒ⁱ²ᵉ = 2 * ℒ.kron(S[1:length(initial_guess)], xλ) - ℒ.kron(kron_buffer, S[length(initial_guess)+1:end])
        
        ∂𝐒ⁱ³ᵉ = 3 * ℒ.kron(S[1:length(initial_guess)], xxλ) - ℒ.kron(kron_buffer²,S[length(initial_guess)+1:end])

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  ∂𝐒ⁱ, ∂𝐒ⁱ²ᵉ, ∂𝐒ⁱ³ᵉ, ∂shock_independent, NoTangent(), NoTangent()
    end

    return (x, matched), find_shocks_pullback
end


# SQP (Sequential Quadratic Programming) implementation using analytical derivatives only
# Solves: min ||x||² subject to c(x) = shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) = 0
# Uses L1 exact penalty merit function for globalization with backtracking line search

@stable default_mode = "disable" begin

function find_shocks(::Val{:SQP},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 1000,
                    tol::Float64 = 1e-13)
    
    n = size(𝐒ⁱ, 2)  # number of shocks
    m = size(𝐒ⁱ, 1)  # number of constraints
    
    x = copy(initial_guess)
    λ = zeros(m)
    
    # Preallocate buffers
    ∇f = zeros(n)           # gradient of objective
    c = zeros(m)            # constraint values
    A = zeros(m, n)         # constraint Jacobian
    H = zeros(n, n)         # Hessian of Lagrangian
    
    # KKT system: [H  A'] [d]   = [-∇f]
    #             [A  0 ] [λ_new]   [-c ]
    KKT = zeros(n + m, n + m)
    rhs = zeros(n + m)
    sol = zeros(n + m)
    
    d = zeros(n)            # step direction
    
    # Merit function parameter
    μ = 1.0
    
    norm1 = ℒ.norm(shock_independent)
    
    for iter in 1:max_iter
        # Compute gradient: ∇f = 2x
        copyto!(∇f, x)
        ℒ.rmul!(∇f, 2)
        
        # Compute constraint: c = shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)
        ℒ.kron!(kron_buffer, x, x)
        ℒ.mul!(c, 𝐒ⁱ, x)
        ℒ.mul!(c, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)
        ℒ.axpby!(1, shock_independent, -1, c)
        
        # Compute constraint Jacobian: A = -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * kron(I, x))
        ℒ.kron!(kron_buffer2, J, x)
        ℒ.mul!(A, 𝐒ⁱ²ᵉ, kron_buffer2)
        ℒ.axpby!(1, 𝐒ⁱ, 2, A)
        ℒ.rmul!(A, -1)
        
        # Compute Hessian of Lagrangian: H = 2I - reshape(2 * 𝐒ⁱ²ᵉ' * λ, n, n)
        # For stability, use H = 2I (BFGS-like simplification if needed)
        # Full analytical Hessian:
        fill!(H, 0.0)
        for i in 1:n
            H[i, i] = 2.0
        end
        # Add second-order term from constraint: -∑ᵢ λᵢ ∇²cᵢ(x)
        # ∇²cᵢ(x) comes from 𝐒ⁱ²ᵉ * kron(x,x), which gives 2*𝐒ⁱ²ᵉ reshaped
        # H -= reshape(2 * 𝐒ⁱ²ᵉ' * λ, n, n)
        tmp_vec = 𝐒ⁱ²ᵉ' * λ
        for j in 1:n
            for i in 1:n
                H[i, j] -= 2.0 * tmp_vec[(j-1)*n + i]
            end
        end
        
        # Build KKT system
        KKT[1:n, 1:n] .= H
        KKT[1:n, n+1:end] .= A'
        KKT[n+1:end, 1:n] .= A
        KKT[n+1:end, n+1:end] .= 0
        
        rhs[1:n] .= -∇f
        rhs[n+1:end] .= -c
        
        # Solve KKT system
        KKT_fact = try
            ℒ.factorize(KKT)
        catch
            return x, false
        end
        
        try
            ℒ.ldiv!(sol, KKT_fact, rhs)
        catch
            return x, false
        end
        
        if !all(isfinite, sol)
            return x, false
        end
        
        copyto!(d, 1, sol, 1, n)
        λ_new = sol[n+1:end]
        
        # Update penalty parameter: μ ≥ ||λ_new||_∞ + δ
        μ = max(μ, ℒ.norm(λ_new, Inf) + 0.1)
        
        # Merit function: φ(x) = f(x) + μ * ||c(x)||₁
        # Current merit value
        f_curr = sum(abs2, x)
        merit_curr = f_curr + μ * ℒ.norm(c, 1)
        
        # Directional derivative of merit function (for Armijo condition)
        # D_d φ = ∇f'd - μ * ||c||₁ (at a feasible descent direction)
        directional_deriv = ℒ.dot(∇f, d) - μ * ℒ.norm(c, 1)
        
        # Backtracking line search
        α = 1.0
        β = 0.5        # step reduction factor  
        c_armijo = 1e-4  # Armijo constant
        
        x_trial = copy(x)
        c_trial = zeros(m)
        
        line_search_success = false
        for ls_iter in 1:20
            x_trial .= x .+ α .* d
            
            # Evaluate constraint at trial point
            ℒ.kron!(kron_buffer, x_trial, x_trial)
            ℒ.mul!(c_trial, 𝐒ⁱ, x_trial)
            ℒ.mul!(c_trial, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)
            ℒ.axpby!(1, shock_independent, -1, c_trial)
            
            f_trial = sum(abs2, x_trial)
            merit_trial = f_trial + μ * ℒ.norm(c_trial, 1)
            
            # Armijo condition
            if merit_trial <= merit_curr + c_armijo * α * directional_deriv
                line_search_success = true
                break
            end
            
            α *= β
        end
        
        if !line_search_success
            # Accept step anyway if it's very small (near convergence)
            if ℒ.norm(d) < sqrt(tol)
                copyto!(x, x_trial)
                copyto!(λ, λ_new)
            else
                return x, false
            end
        else
            copyto!(x, x_trial)
            copyto!(λ, λ_new)
        end
        
        # Check convergence
        # Recompute constraint at new x
        ℒ.kron!(kron_buffer, x, x)
        ℒ.mul!(c, 𝐒ⁱ, x)
        ℒ.mul!(c, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)
        norm2 = ℒ.norm(c)
        ℒ.axpby!(1, shock_independent, -1, c)
        
        constraint_norm = ℒ.norm(c) / max(norm1, norm2)
        step_norm = ℒ.norm(d) / max(ℒ.norm(x), 1.0)
        
        if constraint_norm < tol && step_norm < sqrt(tol)
            return x, true
        end
    end
    
    # Check final convergence
    ℒ.kron!(kron_buffer, x, x)
    ℒ.mul!(c, 𝐒ⁱ, x)
    ℒ.mul!(c, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)
    norm2 = ℒ.norm(c)
    ℒ.axpby!(1, shock_independent, -1, c)
    
    return x, ℒ.norm(c) / max(norm1, norm2) < tol
end

end # dispatch_doctor


# rrule for SQP solver (second order)
function rrule(::typeof(find_shocks),
                ::Val{:SQP},
                initial_guess::Vector{Float64},
                kron_buffer::Vector{Float64},
                kron_buffer2::AbstractMatrix{Float64},
                J::ℒ.Diagonal{Bool, Vector{Bool}},
                𝐒ⁱ::AbstractMatrix{Float64},
                𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                shock_independent::Vector{Float64};
                max_iter::Int = 1000,
                tol::Float64 = 1e-13)

    x, matched = find_shocks(Val(:SQP),
                            initial_guess,
                            kron_buffer,
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent,
                            max_iter = max_iter,
                            tol = tol)

    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x)

    λ = tmp' \ x * 2

    fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
    -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

    ℒ.kron!(kron_buffer, x, x)

    xλ = ℒ.kron(x,λ)


    ∂shock_independent = similar(shock_independent)

    function find_shocks_sqp_pullback(∂x)
        ∂x = vcat(∂x[1], zero(λ))

        S = -fXλp' \ ∂x

        copyto!(∂shock_independent, S[length(initial_guess)+1:end])
        
        ∂𝐒ⁱ = S[1:length(initial_guess)] * λ' - S[length(initial_guess)+1:end] * x'
        
        ∂𝐒ⁱ²ᵉ = 2 * S[1:length(initial_guess)] * xλ' - S[length(initial_guess)+1:end] * kron_buffer'

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂𝐒ⁱ, ∂𝐒ⁱ²ᵉ, ∂shock_independent, NoTangent(), NoTangent()
    end

    return (x, matched), find_shocks_sqp_pullback
end


# SQP solver for third order case
@stable default_mode = "disable" begin

function find_shocks(::Val{:SQP},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    kron_buffer4::AbstractMatrix{Float64},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 1000,
                    tol::Float64 = 1e-13)
    
    n = size(𝐒ⁱ, 2)  # number of shocks
    m = size(𝐒ⁱ, 1)  # number of constraints
    
    x = copy(initial_guess)
    λ = zeros(m)
    
    # Preallocate buffers
    ∇f = zeros(n)           # gradient of objective
    c = zeros(m)            # constraint values
    A = zeros(m, n)         # constraint Jacobian
    H = zeros(n, n)         # Hessian of Lagrangian
    
    # KKT system
    KKT = zeros(n + m, n + m)
    rhs = zeros(n + m)
    sol = zeros(n + m)
    
    d = zeros(n)
    
    # Sparse identity for kron operations
    II = sparse(ℒ.I(n^2))
    
    # Merit function parameter
    μ = 1.0
    
    norm1 = ℒ.norm(shock_independent)
    
    tmp2 = zeros(m, n^2)
    
    for iter in 1:max_iter
        # Compute gradient: ∇f = 2x
        copyto!(∇f, x)
        ℒ.rmul!(∇f, 2)
        
        # Compute kron products
        ℒ.kron!(kron_buffer, x, x)
        ℒ.kron!(kron_buffer², x, kron_buffer)
        
        # Compute constraint: c = shock_independent - 𝐒ⁱ*x - 𝐒ⁱ²ᵉ*kron(x,x) - 𝐒ⁱ³ᵉ*kron(x,kron(x,x))
        ℒ.mul!(c, 𝐒ⁱ, x)
        ℒ.mul!(c, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)
        ℒ.mul!(c, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)
        ℒ.axpby!(1, shock_independent, -1, c)
        
        # Compute constraint Jacobian: A = -(𝐒ⁱ + 2*𝐒ⁱ²ᵉ*kron(I,x) + 3*𝐒ⁱ³ᵉ*kron(I,kron(x,x)))
        ℒ.kron!(kron_buffer2, J, x)
        ℒ.kron!(kron_buffer3, J, kron_buffer)
        
        copy!(A, 𝐒ⁱ)
        ℒ.mul!(A, 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)
        ℒ.mul!(A, 𝐒ⁱ³ᵉ, kron_buffer3, 3, 1)
        ℒ.rmul!(A, -1)
        
        # Compute Hessian of Lagrangian
        # H = 2I - reshape((2*𝐒ⁱ²ᵉ + 6*𝐒ⁱ³ᵉ*kron(I,kron(I,x)))'*λ, n, n)
        fill!(H, 0.0)
        for i in 1:n
            H[i, i] = 2.0
        end
        
        # Second order contribution: -reshape(2*𝐒ⁱ²ᵉ'*λ, n, n)
        tmp_vec = 𝐒ⁱ²ᵉ' * λ
        for j in 1:n
            for i in 1:n
                H[i, j] -= 2.0 * tmp_vec[(j-1)*n + i]
            end
        end
        
        # Third order contribution: -reshape(6*𝐒ⁱ³ᵉ*kron(I,kron(I,x))'*λ, n, n)
        x_kron_II!(kron_buffer4, x)
        ℒ.mul!(tmp2, 𝐒ⁱ³ᵉ, kron_buffer4)
        tmp_vec3 = tmp2' * λ
        for j in 1:n
            for i in 1:n
                H[i, j] -= 6.0 * tmp_vec3[(j-1)*n + i]
            end
        end
        
        # Build KKT system
        KKT[1:n, 1:n] .= H
        KKT[1:n, n+1:end] .= A'
        KKT[n+1:end, 1:n] .= A
        KKT[n+1:end, n+1:end] .= 0
        
        rhs[1:n] .= -∇f
        rhs[n+1:end] .= -c
        
        # Solve KKT system
        KKT_fact = try
            ℒ.factorize(KKT)
        catch
            return x, false
        end
        
        try
            ℒ.ldiv!(sol, KKT_fact, rhs)
        catch
            return x, false
        end
        
        if !all(isfinite, sol)
            return x, false
        end
        
        copyto!(d, 1, sol, 1, n)
        λ_new = sol[n+1:end]
        
        # Update penalty parameter
        μ = max(μ, ℒ.norm(λ_new, Inf) + 0.1)
        
        # Merit function value
        f_curr = sum(abs2, x)
        merit_curr = f_curr + μ * ℒ.norm(c, 1)
        
        # Directional derivative
        directional_deriv = ℒ.dot(∇f, d) - μ * ℒ.norm(c, 1)
        
        # Backtracking line search
        α = 1.0
        β = 0.5
        c_armijo = 1e-4
        
        x_trial = copy(x)
        c_trial = zeros(m)
        
        line_search_success = false
        for ls_iter in 1:20
            x_trial .= x .+ α .* d
            
            ℒ.kron!(kron_buffer, x_trial, x_trial)
            ℒ.kron!(kron_buffer², x_trial, kron_buffer)
            ℒ.mul!(c_trial, 𝐒ⁱ, x_trial)
            ℒ.mul!(c_trial, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)
            ℒ.mul!(c_trial, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)
            ℒ.axpby!(1, shock_independent, -1, c_trial)
            
            f_trial = sum(abs2, x_trial)
            merit_trial = f_trial + μ * ℒ.norm(c_trial, 1)
            
            if merit_trial <= merit_curr + c_armijo * α * directional_deriv
                line_search_success = true
                break
            end
            
            α *= β
        end
        
        if !line_search_success
            if ℒ.norm(d) < sqrt(tol)
                copyto!(x, x_trial)
                copyto!(λ, λ_new)
            else
                return x, false
            end
        else
            copyto!(x, x_trial)
            copyto!(λ, λ_new)
        end
        
        # Check convergence
        ℒ.kron!(kron_buffer, x, x)
        ℒ.kron!(kron_buffer², x, kron_buffer)
        ℒ.mul!(c, 𝐒ⁱ, x)
        ℒ.mul!(c, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)
        ℒ.mul!(c, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)
        norm2 = ℒ.norm(c)
        ℒ.axpby!(1, shock_independent, -1, c)
        
        constraint_norm = ℒ.norm(c) / max(norm1, norm2)
        step_norm = ℒ.norm(d) / max(ℒ.norm(x), 1.0)
        
        if constraint_norm < tol && step_norm < sqrt(tol)
            return x, true
        end
    end
    
    # Final convergence check
    ℒ.kron!(kron_buffer, x, x)
    ℒ.kron!(kron_buffer², x, kron_buffer)
    ℒ.mul!(c, 𝐒ⁱ, x)
    ℒ.mul!(c, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)
    ℒ.mul!(c, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)
    norm2 = ℒ.norm(c)
    ℒ.axpby!(1, shock_independent, -1, c)
    
    return x, ℒ.norm(c) / max(norm1, norm2) < tol
end

end # dispatch_doctor


# rrule for SQP solver (third order)
function rrule(::typeof(find_shocks),
                ::Val{:SQP},
                initial_guess::Vector{Float64},
                kron_buffer::Vector{Float64},
                kron_buffer²::Vector{Float64},
                kron_buffer2::AbstractMatrix{Float64},
                kron_buffer3::AbstractMatrix{Float64},
                kron_buffer4::AbstractMatrix{Float64},
                J::ℒ.Diagonal{Bool, Vector{Bool}},
                𝐒ⁱ::AbstractMatrix{Float64},
                𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                shock_independent::Vector{Float64};
                max_iter::Int = 1000,
                tol::Float64 = 1e-13)

    x, matched = find_shocks(Val(:SQP),
                            initial_guess,
                            kron_buffer,
                            kron_buffer²,
                            kron_buffer2,
                            kron_buffer3,
                            kron_buffer4,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            𝐒ⁱ³ᵉ,
                            shock_independent,
                            max_iter = max_iter,
                            tol = tol)

    ℒ.kron!(kron_buffer, x, x)

    ℒ.kron!(kron_buffer², x, kron_buffer)

    tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer)

    λ = tmp' \ x * 2

    fXλp = [reshape((2 * 𝐒ⁱ²ᵉ + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
    -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

    xλ = ℒ.kron(x,λ)

    xxλ = ℒ.kron(x,xλ)

    function find_shocks_sqp_pullback(∂x)
        ∂x = vcat(∂x[1], zero(λ))

        S = -fXλp' \ ∂x

        ∂shock_independent = S[length(initial_guess)+1:end]
        
        ∂𝐒ⁱ = ℒ.kron(S[1:length(initial_guess)], λ) - ℒ.kron(x, S[length(initial_guess)+1:end])

        ∂𝐒ⁱ²ᵉ = 2 * ℒ.kron(S[1:length(initial_guess)], xλ) - ℒ.kron(kron_buffer, S[length(initial_guess)+1:end])
        
        ∂𝐒ⁱ³ᵉ = 3 * ℒ.kron(S[1:length(initial_guess)], xxλ) - ℒ.kron(kron_buffer²,S[length(initial_guess)+1:end])

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),  ∂𝐒ⁱ, ∂𝐒ⁱ²ᵉ, ∂𝐒ⁱ³ᵉ, ∂shock_independent, NoTangent(), NoTangent()
    end

    return (x, matched), find_shocks_sqp_pullback
end


# @stable default_mode = "disable" begin

# function find_shocks(::Val{:SLSQP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         if length(grad) > 0
#             copy!(grad, X)

#             ℒ.rmul!(grad, 2)
#             # grad .= 2 .* X
#         end
        
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         if length(jac) > 0
#             ℒ.kron!(kron_buffer2, J, x)

#             copy!(jac', 𝐒ⁱ)

#             ℒ.mul!(jac', 𝐒ⁱ²ᵉ, kron_buffer2, -2, -1)
#             # jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))'
#         end

#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X)
#     end
    
#     opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = try 
#         NLopt.optimize(opt, initial_guess)
#     catch
#         return initial_guess, false
#     end

#     ℒ.kron!(kron_buffer, x, x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron_buffer

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         # NLopt.MAXEVAL_REACHED,
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println(ℒ.norm(x))
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end



# function find_shocks(::Val{:SLSQP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         if length(grad) > 0
#             copy!(grad, X)

#             ℒ.rmul!(grad, 2)
#             # grad .= 2 .* X
#         end
        
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.kron!(kron_buffer², x, kron_buffer)

#         if length(jac) > 0
#             ℒ.kron!(kron_buffer2, J, x)

#             ℒ.kron!(kron_buffer3, J, kron_buffer)

#             copy!(jac', 𝐒ⁱ)

#             ℒ.mul!(jac', 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)

#             ℒ.mul!(jac', 𝐒ⁱ³ᵉ, kron_buffer3, -3, -1)
#             # jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(J, x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(J, ℒ.kron(x,x)))'
#         end

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.mul!(res, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * ℒ.kron!(kron_buffer, x, x) - 𝐒ⁱ³ᵉ * ℒ.kron!(kron_buffer², x, kron_buffer)
#     end
    
#     opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = try 
#         NLopt.optimize(opt, initial_guess)
#     catch
#         return initial_guess, false
#     end

#     # println("SLSQP - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         # NLopt.MAXEVAL_REACHED,
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println(ℒ.norm(x))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end






# function find_shocks(::Val{:COBYLA},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 10000,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X)
#     end

#     opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = NLopt.optimize(opt, initial_guess)

#     # println("COBYLA - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println("COBYLA: $(opt.numevals)")

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end



# function find_shocks(::Val{:COBYLA},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 10000,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.kron!(kron_buffer², x, kron_buffer)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.mul!(res, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X) - 𝐒ⁱ³ᵉ * ℒ.kron(X, ℒ.kron(X,X))
#     end

#     opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = NLopt.optimize(opt, initial_guess)

#     # println("COBYLA - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])
#     # println(ℒ.norm(x))
#     # println(x)
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("COBYLA: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("COBYLA: $(opt.numevals)")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")

#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end

# end # dispatch_doctor





# function find_shocks(::Val{:MadNLP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     model = JuMP.Model(MadNLP.Optimizer)

#     JuMP.set_silent(model)

#     JuMP.set_optimizer_attribute(model, "tol", tol)

#     JuMP.@variable(model, x[1:length(initial_guess)])

#     JuMP.set_start_value.(x, initial_guess)

#     JuMP.@objective(model, Min, sum(abs2,x))

#     JuMP.@constraint(model, 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)) .== shock_independent)

#     JuMP.optimize!(model)

#     x = JuMP.value.(x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println(ℒ.norm(y - shock_independent) / max(norm1,norm2))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:Ipopt},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     model = JuMP.Model(Ipopt.Optimizer)

#     JuMP.set_silent(model)

#     JuMP.set_optimizer_attribute(model, "tol", tol)

#     JuMP.@variable(model, x[1:length(initial_guess)])

#     JuMP.set_start_value.(x, initial_guess)

#     JuMP.@objective(model, Min, sum(abs2,x))

#     JuMP.@constraint(model, 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)) .== shock_independent)

#     JuMP.optimize!(model)

#     x = JuMP.value.(x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println(ℒ.norm(y - shock_independent) / max(norm1,norm2))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end


# function find_shocks(::Val{:newton},
#     kron_buffer::Vector{Float64},
#     kron_buffer2::AbstractMatrix{Float64},
#     J::ℒ.Diagonal{Bool, Vector{Bool}},
#     𝐒ⁱ::AbstractMatrix{Float64},
#     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#     shock_independent::Vector{Float64};
#     tol::Float64 = 1e-13) # will fail for higher or lower precision

#     nExo = Int(sqrt(length(kron_buffer)))

#     x = zeros(nExo)

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

#     Δx = zeros(nExo)

#     ∂x = zero(𝐒ⁱ)

#     Ĵ = ℒ.I(nExo)*2

#     max_iter = 1000

# 	norm1 = 1

# 	norm2 = ℒ.norm(shock_independent)

#     for i in 1:max_iter
#         ℒ.kron!(kron_buffer, x, x)
#         ℒ.kron!(kron_buffer2, Ĵ, x)
        
#         ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
#         ℒ.axpy!(1, 𝐒ⁱ, ∂x)
#         # ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(nExo), x))

#         ∂x̂ = try 
#             ℒ.factorize(∂x)
#         catch
#             return x, false
#         end 

#         ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)
#         ℒ.mul!(x̄, 𝐒ⁱ, x)
#         ℒ.axpy!(1, x̄, x̂)
# 				norm1 = ℒ.norm(x̂)
#         ℒ.axpby!(1, shock_independent, -1, x̂)
#         try 
#             ℒ.ldiv!(Δx, ∂x̂, x̂)
#         catch
#             return x, false
#         end
#         # Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
#         # println(ℒ.norm(Δx))
#         if i > 6 && (ℒ.norm(x̂) / max(norm1,norm2) < tol)
#             # println(i)
#             break
#         end
        
#         ℒ.axpy!(1, Δx, x)
#         # x += Δx

#         if !all(isfinite.(x))
#             return x, false
#         end
#     end

#     return x, ℒ.norm(x̂) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:newton},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision

#     nExo = Int(sqrt(length(kron_buffer)))

#     x = zeros(nExo)

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

#     Δx = zeros(nExo)

#     ∂x = zero(𝐒ⁱ)

#     Ĵ = ℒ.I(nExo)*2

#     max_iter = 1000

#     norm1 = 1

# 	norm2 = ℒ.norm(shock_independent)

#     for i in 1:max_iter
#         ℒ.kron!(kron_buffer, x, x)
#         ℒ.kron!(kron_buffer², x, kron_buffer)
#         ℒ.kron!(kron_buffer2, Ĵ, x)
#         ℒ.kron!(kron_buffer3, Ĵ, kron_buffer)
        
#         ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
#         ℒ.mul!(∂x, 𝐒ⁱ³ᵉ, kron_buffer3, 1 ,1)
#         ℒ.axpy!(1, 𝐒ⁱ, ∂x)
#         # ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(nExo), x) + 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(nExo), ℒ.kron(x,x)))

#         ∂x̂ = try 
#             ℒ.factorize(∂x)
#         catch
#             return x, false
#         end 
							
#         ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)
#         ℒ.mul!(x̂, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)
#         ℒ.mul!(x̄, 𝐒ⁱ, x)
#         ℒ.axpy!(1, x̄, x̂)
# 				norm1 = ℒ.norm(x̂)
#         ℒ.axpby!(1, shock_independent, -1, x̂)
#         try 
#             ℒ.ldiv!(Δx, ∂x̂, x̂)
#         catch
#             return x, false
#         end
#         # Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
#         # println(ℒ.norm(Δx))
#         if i > 6 && (ℒ.norm(x̂) / max(norm1,norm2)) < tol
#             # println("Iters: $i Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
#             break
#         end
        
#         ℒ.axpy!(1, Δx, x)
#         # x += Δx

#         if !all(isfinite.(x))
#             return x, false
#         end
#     end

#     # println("Iters: $max_iter Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
#     return x, ℒ.norm(x̂) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGS},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function optim_fun(x::Vector{S}, grad::Vector{S}) where S <: Float64
#         if length(grad) > 0
#             grad .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#         end

#         return sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end
    

#     opt = NLopt.Opt(NLopt.:LD_LBFGS, size(𝐒ⁱ,2))
                    
#     opt.min_objective = optim_fun

#     opt.xtol_abs = eps()
#     opt.ftol_abs = eps()
#     opt.maxeval = 10000

#     (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:LBFGS},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = eps()) # will fail for higher or lower precision

#     function optim_fun(x::Vector{S}, grad::Vector{S}) where S <: Float64
#         if length(grad) > 0
#             grad .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#         end

#         return sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     opt = NLopt.Opt(NLopt.:LD_LBFGS, size(𝐒ⁱ,2))
                    
#     opt.min_objective = optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     opt.maxeval = 10000

#     (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGSjl},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     sol = Optim.optimize(f,g!,
#         # X -> sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X)))),
#                         zeros(size(𝐒ⁱ,2)), 
#                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)))#; 
#                         # autodiff = :forward)

#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGSjl},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end

#     sol = Optim.optimize(f,g!,
#     # X -> sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X))),
#                         zeros(size(𝐒ⁱ,2)), 
#                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)))#; 
#                         # autodiff = :forward)

#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end


# function find_shocks(::Val{:speedmapping},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     sol = speedmapping(zeros(size(𝐒ⁱ,2)), f = f, g! = g!, tol = tol, maps_limit = 10000, stabilize = false)
# println(sol)
#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:speedmapping},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision
#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end

#     sol = speedmapping(zeros(size(𝐒ⁱ,2)), f = f, g! = g!, tol = tol, maps_limit = 10000, stabilize = false)

#     x = sol.minimizer
    
#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end