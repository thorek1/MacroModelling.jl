function levenberg_marquardt(f::Function, 
    initial_guess::Array{T,1}, 
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1},
    parameters::solver_parameters;
    tol::Tolerances = Tolerances()
    ) where {T <: AbstractFloat}
    # issues with optimization: https://www.gurobi.com/documentation/8.1/refman/numerics_gurobi_guidelines.html

    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    iterations = 250
    
    ϕ̄ = parameters.ϕ̄
    ϕ̂ = parameters.ϕ̂
    μ̄¹ = parameters.μ̄¹
    μ̄² = parameters.μ̄²
    p̄¹ = parameters.p̄¹
    p̄² = parameters.p̄²
    ρ = parameters.ρ
    ρ¹ = parameters.ρ¹
    ρ² = parameters.ρ²
    ρ³ = parameters.ρ³
    ν = parameters.ν
    λ¹ = parameters.λ¹
    λ² = parameters.λ²
    λ̂¹ = parameters.λ̂¹
    λ̂² = parameters.λ̂²
    λ̅¹ = parameters.λ̅¹
    λ̅² = parameters.λ̅²
    λ̂̅¹ = parameters.λ̂̅¹
    λ̂̅² = parameters.λ̂̅²
    transformation_level = parameters.transformation_level
    shift = parameters.shift
    backtracking_order = parameters.backtracking_order

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert all(lower_bounds .< upper_bounds)
    @assert backtracking_order ∈ [2,3] "Backtracking order can only be quadratic (2) or cubic (3)."

    max_linesearch_iterations = 600

    function f̂(x) 
        f(undo_transform(x,transformation_level))  
        # f(undo_transform(x,transformation_level,shift))  
    end

    upper_bounds  = transform(upper_bounds,transformation_level)
    # upper_bounds  = transform(upper_bounds,transformation_level,shift)
    lower_bounds  = transform(lower_bounds,transformation_level)
    # lower_bounds  = transform(lower_bounds,transformation_level,shift)

    current_guess = copy(transform(initial_guess,transformation_level))
    # current_guess = copy(transform(initial_guess,transformation_level,shift))
    previous_guess = similar(current_guess)
    guess_update = similar(current_guess)

    ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))
    ∇̂ = similar(∇)

    prep = 𝒟.prepare_jacobian(f̂, backend, current_guess)

    largest_step = T(1.0)
    largest_residual = T(1.0)
    largest_relative_step = T(1.0)

    μ¹ = μ̄¹
    μ² = μ̄²

    p¹ = p̄¹
    p² = p̄²

    grad_iter = 0
    func_iter = 0

	for iter in 1:iterations
        # make the jacobian and f calls nonallocating
        𝒟.jacobian!(f̂, ∇, prep, backend, current_guess)
        grad_iter += 1

        previous_guess .= current_guess

        # ∇̂ .= ∇' * ∇
        ℒ.mul!(∇̂, ∇', ∇)

        μ¹s = μ¹ * sum(abs2, f̂(current_guess))^p¹
        func_iter += 1

        for i in 1:size(∇̂,1)
            ∇̂[i,i] += μ¹s
            ∇̂[i,i] += μ² * ∇̂[i,i]^p²
        end
        # ∇̂ .+= μ¹ * sum(abs2, f̂(current_guess))^p¹ * ℒ.I + μ² * ℒ.Diagonal(∇̂).^p²

        if !all(isfinite,∇̂)
            largest_relative_step = 1.0
            largest_residual = 1.0
            break
        end

        ∇̄ = ℒ.cholesky!(∇̂, check = false)

        if !ℒ.issuccess(∇̄)
            largest_relative_step = 1.0
            largest_residual = 1.0
            break
        end

        ℒ.mul!(guess_update, ∇', f̂(current_guess))
        ℒ.ldiv!(∇̄, guess_update)
        ℒ.axpy!(-1, guess_update, current_guess)
        # current_guess .-= ∇̄ \ ∇' * f̂(current_guess)

        minmax!(current_guess, lower_bounds, upper_bounds)

        P = sum(abs2, f̂(previous_guess))
        P̃ = P
        
        P̋ = sum(abs2, f̂(current_guess))

        func_iter += 3

        α = 1.0
        ᾱ = 1.0

        ν̂ = ν

        guess_update .= current_guess - previous_guess
        g = f̂(previous_guess)' * ∇ * guess_update
        U = sum(abs2,guess_update)
        func_iter += 1

        if P̋ > ρ * P 
            linesearch_iterations = 0
            while P̋ > (1 + ν̂ - ρ¹ * α^2) * P̃ + ρ² * α^2 * g - ρ³ * α^2 * U && linesearch_iterations < max_linesearch_iterations
                if backtracking_order == 2
                    # Quadratic backtracking line search
                    α̂ = -g * α^2 / (2 * (P̋ - P̃ - g * α))
                elseif backtracking_order == 3
                    # Cubic backtracking line search
                    a = (ᾱ^2 * (P̋ - P̃ - g * α) - α^2 * (P - P̃ - g * ᾱ)) / (ᾱ^2 * α^2 * (α - ᾱ))
                    b = (α^3 * (P - P̃ - g * ᾱ) - ᾱ^3 * (P̋ - P̃ - g * α)) / (ᾱ^2 * α^2 * (α - ᾱ))

                    if isapprox(a, zero(a), atol=eps())
                        α̂ = g / (2 * b)
                    else
                        # discriminant
                        d = max(b^2 - 3 * a * g, 0)
                        # quadratic equation root
                        α̂ = (sqrt(d) - b) / (3 * a)
                    end

                    ᾱ = α
                end

                α̂ = min(α̂, ϕ̄ * α)
                α = max(α̂, ϕ̂ * α)
                
                copy!(current_guess, previous_guess)
                ℒ.axpy!(α, guess_update, current_guess)
                # current_guess .= previous_guess + α * guess_update
                minmax!(current_guess, lower_bounds, upper_bounds)
                
                P = P̋

                P̋ = sum(abs2, f̂(current_guess))
                func_iter += 1

                ν̂ *= α

                linesearch_iterations += 1
            end

            μ¹ *= λ̅¹
            μ² *= λ̅²

            p¹ *= λ̂̅¹
            p² *= λ̂̅²
        else
            μ¹ = min(μ¹ / λ¹, μ̄¹)
            μ² = min(μ² / λ², μ̄²)

            p¹ = min(p¹ / λ̂¹, p̄¹)
            p² = min(p² / λ̂², p̄²)
        end

        best_previous_guess = undo_transform(previous_guess,transformation_level)
        best_current_guess = undo_transform(current_guess,transformation_level)

        largest_step = ℒ.norm(best_previous_guess - best_current_guess) # maximum(abs, previous_guess - current_guess)
        largest_relative_step = largest_step / max(ℒ.norm(best_previous_guess), ℒ.norm(best_current_guess)) # maximum(abs, (previous_guess - current_guess) ./ previous_guess)
        
        largest_residual = ℒ.norm(f̂(current_guess)) # maximum(abs, f(undo_transform(current_guess,transformation_level)))
        # largest_residual = maximum(abs, f(undo_transform(current_guess,transformation_level,shift)))

        # allow for norm increases (in both measures) as this can lead to the solution
        
        if largest_residual <= ftol || largest_step <= xtol || largest_relative_step <= rel_xtol
            # println("LM Iteration: $iter; xtol ($xtol): $largest_step; ftol ($ftol): $largest_residual; rel_xtol ($rel_xtol): $largest_relative_step")
            break
            # else
            # end
        end
    end

    best_guess = undo_transform(current_guess,transformation_level)

    return best_guess, (grad_iter, func_iter, largest_relative_step, largest_residual)#, f(best_guess))
end


function newton(f::Function, 
    initial_guess::Array{T,1}, 
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1},
    parameters::solver_parameters;
    tol::Tolerances = Tolerances()
    ) where {T <: AbstractFloat}
    # issues with optimization: https://www.gurobi.com/documentation/8.1/refman/numerics_gurobi_guidelines.html

    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    iterations = 250
    # transformation_level = 0 # parameters.transformation_level

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert all(lower_bounds .< upper_bounds)

    # function f̂(x) 
    #     f(undo_transform(x,transformation_level))  
    # end

    # upper_bounds  = transform(upper_bounds,transformation_level)
    # lower_bounds  = transform(lower_bounds,transformation_level)

    # new_guess = copy(transform(initial_guess,transformation_level))

    new_guess = copy(initial_guess)

    new_residuals = f(new_guess)

    ∇ = Array{T,2}(undef, length(new_guess), length(new_guess))

    prep = 𝒟.prepare_jacobian(f, backend, new_guess)

    # largest_step = zero(T) + 1
    # largest_residual = zero(T) + 1

    rel_xtol_reached = 1.0
    rel_ftol_reached = 1.0
    new_residuals_norm = 1.0
    guess_update_norm = 1.0
    # init_residuals_norm = ℒ.norm(new_residuals)
    iters = [0,0]
    # resnorm = 1.0
    # relresnorm = 1.0

	for iter in 1:iterations
    # while iter < iterations
        𝒟.jacobian!(f, ∇, prep, backend, new_guess)

        # old_residuals_norm = ℒ.norm(new_residuals)

        # old_residuals = copy(new_residuals)

        new_residuals = f(new_guess)

        if !all(isfinite,new_residuals) 
            # println("GN not finite after $iter iteration; - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")  # rel_ftol: $rel_ftol_reached; 
            rel_xtol_reached = 1.0
            rel_ftol_reached = 1.0
            new_residuals_norm = 1.0
            break 
        end
        
        if new_residuals_norm < ftol || rel_xtol_reached < rel_xtol || guess_update_norm < xtol # || rel_ftol_reached < rel_ftol
            new_guess_norm = ℒ.norm(new_guess)

            old_residuals_norm = new_residuals_norm

            new_residuals_norm = ℒ.norm(new_residuals)
        
            ∇̂ = ℒ.lu!(∇, check = false)
        
            ℒ.ldiv!(∇̂, new_residuals)

            guess_update = new_residuals
    
            guess_update_norm = ℒ.norm(guess_update)
    
            ℒ.axpy!(-1, guess_update, new_guess)
    
            iters[1] += 1
            iters[2] += 1

            # println("GN worked with $(iter+1) iterations - xtol ($xtol): $guess_update_norm; ftol ($ftol): $new_residuals_norm; rel_xtol ($rel_xtol): $rel_xtol_reached")# rel_ftol: $rel_ftol_reached")
            break
        end

        new_guess_norm = ℒ.norm(new_guess)

        old_residuals_norm = new_residuals_norm

        new_residuals_norm = ℒ.norm(new_residuals)
        
        if iter > 5 && ℒ.norm(rel_xtol_reached) > sqrt(rel_xtol) && new_residuals_norm > old_residuals_norm
            # println("GN: $iter, Norm increase")
            break
        end
        # if resnorm < ftol # && iter > 4
        #     println("GN worked with $iter iterations - norm: $resnorm; relative norm: $relresnorm")
        #     return undo_transform(new_guess,transformation_level), (iter, zero(T), zero(T), resnorm) # f(undo_transform(new_guess,transformation_level)))
        # end

        ∇̂ = ℒ.lu!(∇, check = false)
        
        if !ℒ.issuccess(∇̂)
            # println("GN factorisation failed after $iter iterations; - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")  # rel_ftol: $rel_ftol_reached; 
            rel_xtol_reached = 1.0
            rel_ftol_reached = 1.0
            new_residuals_norm = 1.0
            # iters = [iter,iter]
            break
        end

        # ∇̂ = try 
        #     ℒ.factorize(∇)
        # catch
        #     # println("GN fact failed after $iter iteration; - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")  # rel_ftol: $rel_ftol_reached; 
        #     rel_xtol_reached = 1.0
        #     rel_ftol_reached = 1.0
        #     new_residuals_norm = 1.0
        #     break
        #     # ℒ.svd(fxλp)
        #     # return undo_transform(new_guess,transformation_level), (iter, largest_step, largest_residual, f(undo_transform(new_guess,transformation_level)))
        # end

        # rel_ftol_reached = ℒ.norm(∇̂' \ new_residuals) / new_residuals_norm

        ℒ.ldiv!(∇̂, new_residuals)

        guess_update = new_residuals

        guess_update_norm = ℒ.norm(guess_update)

        ℒ.axpy!(-1, guess_update, new_guess)

        if !all(isfinite,new_guess) 
            # println("GN not finite after $iter iteration; - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")  # rel_ftol: $rel_ftol_reached; 
            rel_xtol_reached = 1.0
            rel_ftol_reached = 1.0
            new_residuals_norm = 1.0
            # iters = [iter,iter]
            break 
        end
        
        minmax!(new_guess, lower_bounds, upper_bounds)

        rel_xtol_reached = guess_update_norm / max(new_guess_norm, ℒ.norm(new_guess))
        # rel_ftol_reached = new_residuals_norm / max(eps(),init_residuals_norm)
        
        iters[1] += 1
        iters[2] += 1

        # println("GN: $(iters[1]) iterations - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")
    end

    # if iters[1] == iterations
    #     println("GN failed to converge - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")#; rel_ftol: $rel_ftol_reached")
    # else
    #     println("GN converged after $(iters[1]) iterations - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")
    # end

    # best_guess = undo_transform(new_guess,transformation_level)
    
    return new_guess, (iters[1], iters[2], rel_xtol_reached, new_residuals_norm)
end



function minmax!(x::Vector{Float64},lb::Vector{Float64},ub::Vector{Float64})
    @inbounds for i in eachindex(x)
        x[i] = max(lb[i], min(x[i], ub[i]))
    end
end



# transformation of NSSS problem
function transform(x::Vector{T}, option::Int, shift::AbstractFloat)::Vector{T} where T <: Real
    if option == 4
        return asinh.(asinh.(asinh.(asinh.(x .+ shift))))
    elseif option == 3
        return asinh.(asinh.(asinh.(x .+ shift)))
    elseif option == 2
        return asinh.(asinh.(x .+ shift))
    elseif option == 1
        return asinh.(x .+ shift)
    else # if option == 0
        return x .+ shift
    end
end

function transform(x::Vector{T}, option::Int)::Vector{T} where T <: Real
    if option == 4
        return asinh.(asinh.(asinh.(asinh.(x))))
    elseif option == 3
        return asinh.(asinh.(asinh.(x)))
    elseif option == 2
        return asinh.(asinh.(x))
    elseif option == 1
        return asinh.(x)
    else # if option == 0
        return x
    end
end

function undo_transform(x::Vector{T}, option::Int, shift::AbstractFloat)::Vector{T} where T <: Real
    if option == 4
        return sinh.(sinh.(sinh.(sinh.(x)))) .- shift
    elseif option == 3
        return sinh.(sinh.(sinh.(x))) .- shift
    elseif option == 2
        return sinh.(sinh.(x)) .- shift
    elseif option == 1
        return sinh.(x) .- shift
    else # if option == 0
        return x .- shift
    end
end

function undo_transform(x::Vector{T}, option::Int)::Vector{T} where T <: Real
    if option == 4
        return sinh.(sinh.(sinh.(sinh.(x))))
    elseif option == 3
        return sinh.(sinh.(sinh.(x)))
    elseif option == 2
        return sinh.(sinh.(x))
    elseif option == 1
        return sinh.(x)
    else # if option == 0
        return x
    end
end
