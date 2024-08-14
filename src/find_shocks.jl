

function find_shocks(::Val{:LagrangeNewton},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-14) # will fail for higher or lower precision
    X = zeros(size(𝐒ⁱ, 2))
    λ = zeros(size(𝐒ⁱ, 1))
    
    Xλ = [  X
            λ   ]
    
    norm1 = ℒ.norm(shock_independent) 
    
    maxiter = 100
    for i in 1:maxiter
        fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]))' * Xλ[size(𝐒ⁱ, 2)+1:end] - 2 * Xλ[1:size(𝐒ⁱ, 2)]
                shock_independent - 𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] - 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)])]

        fXλp = [reshape((2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.I(length(X))))' * Xλ[size(𝐒ⁱ, 2)+1:end], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]))'
                -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ΔXλ = fXλp \ fXλ
    
        Xλ -= ΔXλ
    
        norm2 = ℒ.norm(𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] + 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]))
    
        # println(ℒ.norm(ΔXλ))
        if ℒ.norm(shock_independent - (𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] + 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]))) / max(norm1,norm2) < eps() && ℒ.norm(ΔXλ) < tol
            println(i)
            break
        end
        if i == maxiter
            println(ℒ.norm(ΔXλ))
        end
    end
              
    x = Xλ[1:size(𝐒ⁱ, 2)]
                    
    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x)

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end




function find_shocks(::Val{:LagrangeNewton},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-14) # will fail for higher or lower precision
    X = zeros(size(𝐒ⁱ, 2))
    λ = zeros(size(𝐒ⁱ, 1))
    
    Xλ = [  X
            λ   ]
    
    norm1 = ℒ.norm(shock_independent) 
    
    maxiter = 500

    for i in 1:maxiter
        fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))' * Xλ[size(𝐒ⁱ, 2)+1:end] - 2 * Xλ[1:size(𝐒ⁱ, 2)]
                shock_independent - 𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] - 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]) - 𝐒ⁱ³ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)]))]

        fXλp = [reshape((2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.I(length(X))) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(ℒ.I(length(X)),Xλ[1:size(𝐒ⁱ, 2)])))' * Xλ[size(𝐒ⁱ, 2)+1:end], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))'
                -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ΔXλ = fXλp \ fXλ
    
        Xλ -= ΔXλ
    
        norm2 = ℒ.norm(𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] + 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]) + 𝐒ⁱ³ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))
    
        # println(ℒ.norm(ΔXλ))
        if ℒ.norm(shock_independent - (𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] + 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]) + 𝐒ⁱ³ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))) / max(norm1,norm2) < eps() && ℒ.norm(ΔXλ) < tol
            println(i)
            break
        end
        if i == maxiter
            println(ℒ.norm(ΔXλ))
            println(ℒ.norm(shock_independent - (𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] + 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]) + 𝐒ⁱ³ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))) / max(norm1,norm2))
        end
    end
              
    x = Xλ[1:size(𝐒ⁱ, 2)]
                    
    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end


function find_shocks(::Val{:Newton},
    kron_buffer::Vector{Float64},
    kron_buffer2::AbstractMatrix{Float64},
    J::AbstractMatrix{Float64},
    𝐒ⁱ::AbstractMatrix{Float64},
    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
    shock_independent::Vector{Float64};
    tol::Float64 = 1e-14) # will fail for higher or lower precision

    nExo = Int(sqrt(length(kron_buffer)))

    x = zeros(nExo)

    x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

    x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

    x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

    Δx = zeros(nExo)

    ∂x = zero(𝐒ⁱ)

    Ĵ = ℒ.I(nExo)*2

    max_iter = 1000

	norm1 = 1

	norm2 = ℒ.norm(shock_independent)

    for i in 1:max_iter
        ℒ.kron!(kron_buffer, x, x)
        ℒ.kron!(kron_buffer2, Ĵ, x)
        
        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
        ℒ.axpy!(1, 𝐒ⁱ, ∂x)
        # ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(nExo), x))

        ∂x̂ = try 
            ℒ.factorize(∂x)
        catch
            return x, false
        end 

        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)
        ℒ.mul!(x̄, 𝐒ⁱ, x)
        ℒ.axpy!(1, x̄, x̂)
				norm1 = ℒ.norm(x̂)
        ℒ.axpby!(1, shock_independent, -1, x̂)
        try 
            ℒ.ldiv!(Δx, ∂x̂, x̂)
        catch
            return x, false
        end
        # Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
        # println(ℒ.norm(Δx))
        if i > 6 && (ℒ.norm(x̂) / max(norm1,norm2) < tol)
            # println(i)
            break
        end
        
        ℒ.axpy!(1, Δx, x)
        # x += Δx

        if !all(isfinite.(x))
            return x, false
        end
    end

    return x, ℒ.norm(x̂) / max(norm1,norm2) < tol
end



function find_shocks(::Val{:Newton},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-14) # will fail for higher or lower precision

    nExo = Int(sqrt(length(kron_buffer)))

    x = zeros(nExo)

    x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

    x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

    x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

    Δx = zeros(nExo)

    ∂x = zero(𝐒ⁱ)

    Ĵ = ℒ.I(nExo)*2

    max_iter = 1000

    norm1 = 1

	norm2 = ℒ.norm(shock_independent)

    for i in 1:max_iter
        ℒ.kron!(kron_buffer, x, x)
        ℒ.kron!(kron_buffer², x, kron_buffer)
        ℒ.kron!(kron_buffer2, Ĵ, x)
        ℒ.kron!(kron_buffer3, Ĵ, kron_buffer)
        
        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
        ℒ.mul!(∂x, 𝐒ⁱ³ᵉ, kron_buffer3, 1 ,1)
        ℒ.axpy!(1, 𝐒ⁱ, ∂x)
        # ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(nExo), x) + 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(nExo), ℒ.kron(x,x)))

        ∂x̂ = try 
            ℒ.factorize(∂x)
        catch
            return x, false
        end 
							
        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)
        ℒ.mul!(x̂, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)
        ℒ.mul!(x̄, 𝐒ⁱ, x)
        ℒ.axpy!(1, x̄, x̂)
				norm1 = ℒ.norm(x̂)
        ℒ.axpby!(1, shock_independent, -1, x̂)
        try 
            ℒ.ldiv!(Δx, ∂x̂, x̂)
        catch
            return x, false
        end
        # Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
        # println(ℒ.norm(Δx))
        if i > 6 && (ℒ.norm(x̂) / max(norm1,norm2)) < tol
            # println("Iters: $i Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
            break
        end
        
        ℒ.axpy!(1, Δx, x)
        # x += Δx

        if !all(isfinite.(x))
            return x, false
        end
    end

    # println("Iters: $max_iter Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
    return x, ℒ.norm(x̂) / max(norm1,norm2) < tol
end




function find_shocks(::Val{:SLSQP},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision
    function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
        if length(grad) > 0
            grad .= 2 .* X
        end
        
        sum(abs2, X)
    end

    function constraint_optim(res::Vector{S}, X::Vector{S}, jac::Matrix{S}) where S <: Float64
        if length(jac) > 0
            # jac .= 𝒟.jacobian(x -> shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x), backend, X)'
            jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), X))'
        end

        res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)
    end
    
    # opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
    opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
    opt.min_objective = objective_optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = 500

    NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

    (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end



function find_shocks(::Val{:SLSQP},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision
    function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
        if length(grad) > 0
            grad .= 2 .* X
        end
        
        sum(abs2, X)
    end

    function constraint_optim(res::Vector{S}, X::Vector{S}, jac::Matrix{S}) where S <: Float64
        if length(jac) > 0
            # jac .= 𝒟.jacobian(x -> shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x)), backend, X)'
            jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), X) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(X)),kron(X,X)))'
        end

        res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))
    end

    # opt = NLopt.Opt(NLopt.:LN_COBYLA, T.nExo)
    opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
    opt.min_objective = objective_optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = 500

    NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

    (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end






function find_shocks(::Val{:COBYLA},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision
    function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
        if length(grad) > 0
            grad .= 2 .* X
        end
        
        sum(abs2, X)
    end

    function constraint_optim(res::Vector{S}, X::Vector{S}, jac::Matrix{S}) where S <: Float64
        if length(jac) > 0
            # jac .= 𝒟.jacobian(x -> shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x), backend, X)'
            jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), X))'
        end

        res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)
    end
    
    opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
    opt.min_objective = objective_optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = 500

    NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

    (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end



function find_shocks(::Val{:COBYLA},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision
    function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
        if length(grad) > 0
            grad .= 2 .* X
        end
        
        sum(abs2, X)
    end

    function constraint_optim(res::Vector{S}, X::Vector{S}, jac::Matrix{S}) where S <: Float64
        if length(jac) > 0
            # jac .= 𝒟.jacobian(x -> shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x)), backend, X)'
            jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), X) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(X)),kron(X,X)))'
        end

        res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))
    end

    opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
    opt.min_objective = objective_optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = 500

    NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

    (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end



function find_shocks(::Val{:LBFGS},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision

    function optim_fun(x::Vector{S}, grad::Vector{S}) where S <: Float64
        if length(grad) > 0
            grad .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
        end

        return sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
    end
    

    opt = NLopt.Opt(NLopt.:LD_LBFGS, size(𝐒ⁱ,2))
                    
    opt.min_objective = optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = 10000

    (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end




function find_shocks(::Val{:LBFGS},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = eps()) # will fail for higher or lower precision

    function optim_fun(x::Vector{S}, grad::Vector{S}) where S <: Float64
        if length(grad) > 0
            grad .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
        end

        return sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
    end

    opt = NLopt.Opt(NLopt.:LD_LBFGS, size(𝐒ⁱ,2))
                    
    opt.min_objective = optim_fun

    # opt.xtol_abs = eps()
    # opt.ftol_abs = eps()
    opt.maxeval = 10000

    (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end



function find_shocks(::Val{:LBFGSjl},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision

    function f(X)
        sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))))
    end

    function g!(G, x)
        G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
    end

    sol = Optim.optimize(f,g!,
        # X -> sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X)))),
                        zeros(size(𝐒ⁱ,2)), 
                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)))#; 
                        # autodiff = :forward)

    x = sol.minimizer

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end



function find_shocks(::Val{:LBFGSjl},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision

    function f(X)
        sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)))
    end

    function g!(G, x)
        G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
    end

    sol = Optim.optimize(f,g!,
    # X -> sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X))),
                        zeros(size(𝐒ⁱ,2)), 
                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)))#; 
                        # autodiff = :forward)

    x = sol.minimizer

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end


function find_shocks(::Val{:speedmapping},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision

    function f(X)
        sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))))
    end

    function g!(G, x)
        G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
    end

    sol = speedmapping(zeros(size(𝐒ⁱ,2)), f = f, g! = g!, tol = tol, maps_limit = 10000, stabilize = false)
println(sol)
    x = sol.minimizer

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end




function find_shocks(::Val{:speedmapping},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    tol::Float64 = 1e-15) # will fail for higher or lower precision
    function f(X)
        sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)))
    end

    function g!(G, x)
        G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
    end

    sol = speedmapping(zeros(size(𝐒ⁱ,2)), f = f, g! = g!, tol = tol, maps_limit = 10000, stabilize = false)

    x = sol.minimizer
    
    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
end

