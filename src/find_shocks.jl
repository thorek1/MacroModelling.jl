function find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 1000,
                    tol::Float64 = 1e-14) # will fail for higher or lower precision
    x = initial_guess

    λ = zeros(size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]
    
    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 
    
    norm2 = 1.0
    
    Δnorm = 1e12
    
    x̂ = copy(shock_independent)

    x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

    Ĵ = ℒ.I(length(x)) * 2

    ∂x = zero(𝐒ⁱ)
    
    fxλ = zeros(size(𝐒ⁱ²ᵉ,1) + size(shock_independent,1))

    fxλp = zeros(size(𝐒ⁱ, 2) + size(𝐒ⁱ, 1), size(𝐒ⁱ, 2) + size(𝐒ⁱ, 1))

    tmp = zeros(size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    for i in 1:max_iter
        ℒ.kron!(kron_buffer2, Ĵ, x)

        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpy!(1, 𝐒ⁱ, ∂x)
        
        ℒ.mul!(x̄, ∂x', λ)
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ²ᵉ,1))
        copyto!(fxλ, size(𝐒ⁱ²ᵉ,1) + 1, x̂, 1, size(shock_independent,1))
        # copyto!(fxλp, ∂x)
        # fxλ = [x̄#∂x' * λ - 2 * x
        #        x̂]

        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ)
        ℒ.rmul!(tmp, 2)

        # Define the destination and source ranges
        # dest_range = CartesianIndices((1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)))
        
        # Perform the copy using copyto!
        # copyto!(fxλp, dest_range, tmp, CartesianIndices(tmp))

        fxλp[1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp#2 * 𝐒ⁱ²ᵉ' * λ
        fxλp[1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] .= ∂x'

        ℒ.rmul!(∂x, -1)
        fxλp[size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] .= ∂x
        # fxλp = [reshape((2 * 𝐒ⁱ²ᵉ)' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  ∂x'
        #         -∂x  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        # println(size(fxλp))
        f̂xλp = try 
            ℒ.factorize(fxλp)
        catch
            return x, false
        end

        # Δxλ = fxλp \ fxλ
        ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
    
        if !all(isfinite,Δxλ) break end
        
        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ
    
        x = xλ[1:size(𝐒ⁱ, 2)]

        λ = xλ[size(𝐒ⁱ, 2)+1:end]

        ℒ.kron!(kron_buffer, x, x)

        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)

        ℒ.mul!(x̄, 𝐒ⁱ, x)

        ℒ.axpy!(1, x̄, x̂)
        # tmp2 = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron!(kron_buffer, x, x)

        norm2 = ℒ.norm(x̂)

        ℒ.axpby!(1, shock_independent, -1, x̂)
        # ℒ.axpby!(1, shock_independent, -1, x̂)

        if ℒ.norm(x̂) / max(norm1,norm2) < eps() && ℒ.norm(Δxλ) / ℒ.norm(xλ) < tol
            # println("LagrangeNewton: $i, Tol reached")
            break
        end

        if i > 5 && ℒ.norm(Δxλ) > 1e-12 && ℒ.norm(Δxλ) > Δnorm
            # println("LagrangeNewton: $i, Norm increase")
            break
        end
        # if i == max_iter
            # println("LagrangeNewton: $i, Max iter reached")
        #     # println(ℒ.norm(Δxλ))
        # end
    end
         
    # println(ℒ.norm(x̂))         

    # y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x)

    # norm1 = ℒ.norm(y)

	# norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
    # println(ℒ.norm(Δxλ))
    # println(ℒ.norm(Δxλ) / ℒ.norm(xλ))
    return x, ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < tol
end




function find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 1000,
                    tol::Float64 = 1e-14) # will fail for higher or lower precision
    X = initial_guess
    λ = zeros(size(𝐒ⁱ, 1))
    
    Xλ = [  X
            λ   ]

    ΔXλ = copy(Xλ)

    norm1 = ℒ.norm(shock_independent) 

    Δnorm = 1e12

    for i in 1:max_iter
        fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))' * Xλ[size(𝐒ⁱ, 2)+1:end] - 2 * Xλ[1:size(𝐒ⁱ, 2)]
                shock_independent - 𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] - 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]) - 𝐒ⁱ³ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)]))]

        fXλp = [reshape((2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.I(length(X))) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(ℒ.I(length(X)),Xλ[1:size(𝐒ⁱ, 2)])))' * Xλ[size(𝐒ⁱ, 2)+1:end], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))'
                -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), Xλ[1:size(𝐒ⁱ, 2)]) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
    
        ΔXλ = fXλp \ fXλ
  
        if !all(isfinite,ΔXλ) break end

        Xλ -= ΔXλ
    
        norm2 = ℒ.norm(𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] + 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]) + 𝐒ⁱ³ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))
    
        # println(ℒ.norm(ΔXλ))
        if ℒ.norm(shock_independent - (𝐒ⁱ * Xλ[1:size(𝐒ⁱ, 2)] + 𝐒ⁱ²ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)],Xλ[1:size(𝐒ⁱ, 2)]) + 𝐒ⁱ³ᵉ * ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], ℒ.kron(Xλ[1:size(𝐒ⁱ, 2)], Xλ[1:size(𝐒ⁱ, 2)])))) / max(norm1,norm2) < eps() && ℒ.norm(ΔXλ) < tol
            # println("LagrangeNewton: $i, Tol reached")
            break
        end

        if i > 5 && ℒ.norm(ΔXλ) > 1e-12 && ℒ.norm(ΔXλ) > Δnorm
            # println("LagrangeNewton: $i, Norm increase")
            break
        end
        # if i == max_iter
            # println("LagrangeNewton: $i, Max iter reached")
            # println(ℒ.norm(ΔXλ))
        # end
    end

    x = Xλ[1:size(𝐒ⁱ, 2)]
                    
    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && ℒ.norm(ΔXλ) < tol
end





function find_shocks(::Val{:SLSQP},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 500,
                    tol::Float64 = 1e-14) # will fail for higher or lower precision
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

        res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X)
    end
    
    # opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
    opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
    opt.min_objective = objective_optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = max_iter

    NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

    (minf,x,ret) = try 
        NLopt.optimize(opt, initial_guess)
    catch
        return initial_guess, false
    end

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x)

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    solved = ret ∈ Symbol.([
        NLopt.SUCCESS,
        NLopt.STOPVAL_REACHED,
        NLopt.FTOL_REACHED,
        NLopt.XTOL_REACHED,
        NLopt.ROUNDOFF_LIMITED,
    ])

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
end



function find_shocks(::Val{:SLSQP},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 500,
                    tol::Float64 = 1e-14) # will fail for higher or lower precision
    function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
        if length(grad) > 0
            grad .= 2 .* X
        end
        
        sum(abs2, X)
    end

    function constraint_optim(res::Vector{S}, X::Vector{S}, jac::Matrix{S}) where S <: Float64
        if length(jac) > 0
            # jac .= 𝒟.jacobian(x -> shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x)), backend, X)'
            jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), X) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(X,X)))'
        end

        res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X) - 𝐒ⁱ³ᵉ * ℒ.kron(X, ℒ.kron(X,X))
    end

    # opt = NLopt.Opt(NLopt.:LN_COBYLA, T.nExo)
    opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
    opt.min_objective = objective_optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = max_iter

    NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

    (minf,x,ret) = try 
        NLopt.optimize(opt, initial_guess)
    catch
        return initial_guess, false
    end

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    solved = ret ∈ Symbol.([
        NLopt.SUCCESS,
        NLopt.STOPVAL_REACHED,
        NLopt.FTOL_REACHED,
        NLopt.XTOL_REACHED,
        NLopt.ROUNDOFF_LIMITED,
    ])

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
end






function find_shocks(::Val{:COBYLA},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 10000,
                    tol::Float64 = 1e-14) # will fail for higher or lower precision
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

        res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X)
    end
    
    opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
    opt.min_objective = objective_optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = max_iter

    NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

    (minf,x,ret) = NLopt.optimize(opt, initial_guess)

    # println("COBYLA - retcode: $ret, nevals: $(opt.numevals)")

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x)

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    solved = ret ∈ Symbol.([
        NLopt.SUCCESS,
        NLopt.STOPVAL_REACHED,
        NLopt.FTOL_REACHED,
        NLopt.XTOL_REACHED,
        NLopt.ROUNDOFF_LIMITED,
    ])

    # println("COBYLA: $(opt.numevals)")

    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
end



function find_shocks(::Val{:COBYLA},
                    initial_guess::Vector{Float64},
                    kron_buffer::Vector{Float64},
                    kron_buffer²::Vector{Float64},
                    kron_buffer2::AbstractMatrix{Float64},
                    kron_buffer3::AbstractMatrix{Float64},
                    J::AbstractMatrix{Float64},
                    𝐒ⁱ::AbstractMatrix{Float64},
                    𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
                    𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
                    shock_independent::Vector{Float64};
                    max_iter::Int = 10000,
                    tol::Float64 = 1e-14) # will fail for higher or lower precision
    function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
        if length(grad) > 0
            grad .= 2 .* X
        end
        
        sum(abs2, X)
    end

    function constraint_optim(res::Vector{S}, X::Vector{S}, jac::Matrix{S}) where S <: Float64
        if length(jac) > 0
            # jac .= 𝒟.jacobian(x -> shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x)), backend, X)'
            jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(X)), X) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(X)), ℒ.kron(X,X)))'
        end

        res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X) - 𝐒ⁱ³ᵉ * ℒ.kron(X, ℒ.kron(X,X))
    end

    opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
    opt.min_objective = objective_optim_fun

    opt.xtol_abs = eps()
    opt.ftol_abs = eps()
    opt.maxeval = max_iter

    NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

    (minf,x,ret) = NLopt.optimize(opt, initial_guess)

    # println("COBYLA - retcode: $ret, nevals: $(opt.numevals)")

    y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

    norm1 = ℒ.norm(y)

	norm2 = ℒ.norm(shock_independent)

    solved = ret ∈ Symbol.([
        NLopt.SUCCESS,
        NLopt.STOPVAL_REACHED,
        NLopt.FTOL_REACHED,
        NLopt.XTOL_REACHED,
        NLopt.ROUNDOFF_LIMITED,
    ])

    # println("COBYLA: $(opt.numevals)")
    # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")

    return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
end





# function find_shocks(::Val{:Newton},
#     kron_buffer::Vector{Float64},
#     kron_buffer2::AbstractMatrix{Float64},
#     J::AbstractMatrix{Float64},
#     𝐒ⁱ::AbstractMatrix{Float64},
#     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#     shock_independent::Vector{Float64};
#     tol::Float64 = 1e-14) # will fail for higher or lower precision

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



# function find_shocks(::Val{:Newton},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::AbstractMatrix{Float64},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-14) # will fail for higher or lower precision

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
#                     J::AbstractMatrix{Float64},
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
#                     J::AbstractMatrix{Float64},
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
#                     J::AbstractMatrix{Float64},
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
#                     J::AbstractMatrix{Float64},
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
#                     J::AbstractMatrix{Float64},
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
#                     J::AbstractMatrix{Float64},
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

