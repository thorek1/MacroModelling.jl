@stable default_mode = "disable" begin

function levenberg_marquardt(
    fnj::function_and_jacobian,
    # f::Function, 
    initial_guess::Array{T,1}, 
    parameters_and_solved_vars::Array{T,1},
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1},
    parameters::solver_parameters;
    tol::Tolerances = Tolerances()
    )::Tuple{Vector{T}, Tuple{Int, Int, T, T}} where {T <: AbstractFloat}
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

    # Get pre-allocated workspace vectors and buffers
    ws = fnj.workspace
    u_bounds = ws.u_bounds
    l_bounds = ws.l_bounds
    current_guess = ws.current_guess
    current_guess_untransformed = ws.current_guess_untransformed
    previous_guess = ws.previous_guess
    previous_guess_untransformed = ws.previous_guess_untransformed
    guess_update = ws.guess_update
    factor = ws.factor
    best_previous_guess = ws.best_previous_guess
    best_current_guess = ws.best_current_guess
    
    # Initialize from input bounds and guess
    copy!(u_bounds, upper_bounds)
    copy!(l_bounds, lower_bounds)
    copy!(current_guess, initial_guess)

    for _ in 1:transformation_level
        u_bounds .= asinh.(u_bounds)
        l_bounds .= asinh.(l_bounds)
        current_guess .= asinh.(current_guess)
    end

    sol_cache = ws.chol_buffer

    copy!(current_guess_untransformed, current_guess)
    ∇ = ws.jac_buffer
    ∇̂ = sol_cache.A
    # ∇̄ = similar(fnj.jac_buffer)

    # ∇̂ = choose_matrix_format(∇' * ∇, multithreaded = false)
    
    # if ∇̂ isa SparseMatrixCSC
    #     prob = 𝒮.LinearProblem(∇̂, guess_update, 𝒮.CHOLMODFactorization())
    #     sol_cache = 𝒮.init(prob, 𝒮.CHOLMODFactorization())
    # else
        # X = ℒ.Symmetric(∇̂, :U)
        # prob = 𝒮.LinearProblem(X, guess_update, 𝒮.CholeskyFactorization)
        # prob = 𝒮.LinearProblem(∇̂, guess_update, 𝒮.CholeskyFactorization())
        # sol_cache = 𝒮.init(prob, 𝒮.CholeskyFactorization())
    # end
    
    # prep = 𝒟.prepare_jacobian(f̂, backend, current_guess)

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
        copy!(current_guess_untransformed, current_guess)
        
        if transformation_level > 0
            factor .= 1
            for _ in 1:transformation_level
                factor .*= cosh.(current_guess_untransformed)
                current_guess_untransformed .= sinh.(current_guess_untransformed)
            end
        end

        fnj.jac(∇, current_guess_untransformed::Vector{T}, parameters_and_solved_vars::Vector{T})
        # 𝒟.jacobian!(f̂, ∇, prep, backend, current_guess)

        if transformation_level > 0
            scale_columns!(∇, factor)
            # if ∇ isa SparseMatrixCSC
            #     # ∇̄ = ∇ .* factor'
            #     copy!(∇̄.nzval, ∇.nzval)
            #     @inbounds for j in 1:size(∇, 2)
            #         col_start = ∇̄.colptr[j]
            #         col_end = ∇̄.colptr[j+1] - 1
            #         for k in col_start:col_end
            #             ∇̄.nzval[k] *= factor[j]
            #         end
            #     end
            # else
            #     # ℒ.mul!(∇̄, ∇, factor')
            #     @. ∇̄ = ∇ * factor'
            #     # ∇ .*= factor'
            # end
        end

        grad_iter += 1

        previous_guess .= current_guess

        # ∇̂ .= ∇' * ∇
        if ∇̂ isa SparseMatrixCSC
            ∇̂ = ∇' * ∇
        else
            ℒ.mul!(∇̂, ∇', ∇)
        end

        fnj.func(ws.func_buffer::Vector{T}, current_guess_untransformed::Vector{T}, parameters_and_solved_vars::Vector{T})

        copy!(factor, ws.func_buffer)

        μ¹s = μ¹ * ℒ.dot(factor, factor)^p¹
        # μ¹s = μ¹ * sum(abs2, f̂(current_guess))^p¹
        func_iter += 1
        
        update_∇̂!(∇̂, μ¹s, μ², p²)
        # @inbounds for i in 1:size(∇̂,1) # fix allocs here
        #     ∇̂[i,i] += μ¹s
        #     ∇̂[i,i] += μ² * ∇̂[i,i]^p²
        # end
        # ∇̂ .+= μ¹ * sum(abs2, f̂(current_guess))^p¹ * ℒ.I + μ² * ℒ.Diagonal(∇̂).^p²

        finn = has_nonfinite(∇̂)

        if finn
            largest_relative_step = 1.0
            largest_residual = 1.0
            break
        end

        # fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)

        ℒ.mul!(guess_update, ∇', factor)

        # X = ℒ.Symmetric(∇̂, :U)
        # sol_cache.A = X
        sol_cache.A = ∇̂
        sol_cache.b = guess_update
        sol = 𝒮.solve!(sol_cache)
        copy!(guess_update, sol_cache.u)

        if !(𝒮.SciMLBase.successful_retcode(sol.retcode) || sol.retcode == 𝒮.SciMLBase.ReturnCode.Default || isfinite(sum(guess_update)))
            largest_relative_step = 1.0
            largest_residual = 1.0
            break
        end

        ℒ.axpy!(-1, guess_update, current_guess)
        # current_guess .-= ∇̄ \ ∇' * f̂(current_guess)

        minmax!(current_guess, l_bounds, u_bounds)

        copy!(previous_guess_untransformed, previous_guess)

        for _ in 1:transformation_level
            previous_guess_untransformed .= sinh.(previous_guess_untransformed)
        end
        
        # fnj.func(fnj.func_buffer, previous_guess_untransformed, parameters_and_solved_vars)

        P = ℒ.dot(factor, factor)
        # P = sum(abs2, f̂(previous_guess))
        P̃ = P
        
        copy!(current_guess_untransformed, current_guess)

        for _ in 1:transformation_level
            current_guess_untransformed .= sinh.(current_guess_untransformed)
        end

        fnj.func(ws.func_buffer::Vector{T}, current_guess_untransformed::Vector{T}, parameters_and_solved_vars::Vector{T})
      
        P̋ = ℒ.dot(ws.func_buffer, ws.func_buffer)
        # P̋ = sum(abs2, f̂(current_guess))

        func_iter += 3

        α = 1.0
        ᾱ = 1.0
        α̂ = 1.0

        ν̂ = ν

        # guess_update .= current_guess - previous_guess
        guess_update .= current_guess
        guess_update .-= previous_guess

        # fnj.func(fnj.func_buffer, previous_guess_untransformed, parameters_and_solved_vars)

        # g = factor' * ∇̄ * guess_update
        g = ℒ.dot(factor, ∇, guess_update)
        # g = f̂(previous_guess)' * ∇ * guess_update
        U = sum(abs2,guess_update)
        func_iter += 1

        if P̋ > ρ * P 
            linesearch_iterations = 0

            cond = condition_P̋(P̋, ν̂, ρ¹, α, P̃, ρ², g, ρ³, U)

            while cond && linesearch_iterations < max_linesearch_iterations # fix allocs here
                if backtracking_order == 2
                    # Quadratic backtracking line search

                    α̂ = update_α̂(g, α, P̋, P̃)
                    # α̂ = -g * α^2 / (2 * (P̋ - P̃ - g * α)) # fix allocs here

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

                α̂, α = minmax_α(α̂, ϕ̄, α, ϕ̂)
                # α̂ = min(α̂, ϕ̄ * α)
                # α = max(α̂, ϕ̂ * α)
                
                copy!(current_guess, previous_guess)
                ℒ.axpy!(α, guess_update, current_guess)
                # current_guess .= previous_guess + α * guess_update
                minmax!(current_guess, l_bounds, u_bounds)
                
                P = P̋

                copy!(current_guess_untransformed, current_guess)

                for _ in 1:transformation_level
                    current_guess_untransformed .= sinh.(current_guess_untransformed)
                end

                fnj.func(ws.func_buffer::Vector{T}, current_guess_untransformed::Vector{T}, parameters_and_solved_vars::Vector{T})

                P̋ = ℒ.dot(ws.func_buffer, ws.func_buffer)
                # P̋ = sum(abs2, f̂(current_guess))
                func_iter += 1

                ν̂ *= α

                linesearch_iterations += 1

                cond = condition_P̋(P̋, ν̂, ρ¹, α, P̃, ρ², g, ρ³, U)
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

        for _ in 1:transformation_level
            best_previous_guess .= sinh.(previous_guess)
            best_current_guess .= sinh.(current_guess)
        end

        # best_previous_guess = undo_transform(previous_guess, transformation_level)
        # best_current_guess = undo_transform(current_guess, transformation_level)

        @. factor = best_previous_guess - best_current_guess
        largest_step = ℒ.norm(factor) # maximum(abs, previous_guess - current_guess)
        largest_relative_step = largest_step / max(ℒ.norm(best_previous_guess), ℒ.norm(best_current_guess)) # maximum(abs, (previous_guess - current_guess) ./ previous_guess)
        
        copy!(current_guess_untransformed, current_guess)

        for _ in 1:transformation_level
            current_guess_untransformed .= sinh.(current_guess_untransformed)
        end

        fnj.func(ws.func_buffer::Vector{T}, current_guess_untransformed::Vector{T}, parameters_and_solved_vars::Vector{T})

        largest_residual = ℒ.norm(ws.func_buffer)    
        # largest_residual = ℒ.norm(f̂(current_guess)) # maximum(abs, f(undo_transform(current_guess,transformation_level)))
        # largest_residual = maximum(abs, f(undo_transform(current_guess,transformation_level,shift)))

        # allow for norm increases (in both measures) as this can lead to the solution
        
        if largest_residual <= ftol || largest_step <= xtol || largest_relative_step <= rel_xtol
            # println("LM Iteration: $iter; xtol ($xtol): $largest_step; ftol ($ftol): $largest_residual; rel_xtol ($rel_xtol): $largest_relative_step")
            break
            # else
            # end
        end
    end
    
    for _ in 1:transformation_level
        best_current_guess .= sinh.(current_guess)
    end

    return best_current_guess, (grad_iter, func_iter, largest_relative_step, largest_residual)#, f(best_guess))
end

function scale_columns!(A::AbstractMatrix{T}, v::AbstractVector{T}) where T
    @inbounds for j in 1:size(A, 2)
        for i in 1:size(A, 1)
            A[i, j] *= v[j]
        end
    end
    return A
end

function scale_columns!(A::SparseMatrixCSC{T}, v::AbstractVector{T}) where T
    @inbounds for j in 1:size(A, 2)
        scale = v[j]
        col_start = A.colptr[j]
        col_end = A.colptr[j+1] - 1
        for k in col_start:col_end
            A.nzval[k] *= scale
        end
    end
    return A
end

function update_∇̂!(∇̂::AbstractMatrix{T}, μ¹s::T, μ²::T, p²::T) where T <: Real
    n = size(∇̂, 1)                # hoist size lookup
    @inbounds for i in 1:n
        x = ∇̂[i,i]                # read once
        x += μ¹s
        x += μ² * (x^p²)          # scalar pow, no array allocation
        ∇̂[i,i] = x               # write back
    end
    return nothing
end


function minmax_α(α̂::T, ϕ̄::T, α::T, ϕ̂::T)::Tuple{T,T} where T <: Real
    α̂ = min(α̂, ϕ̄ * α)
    α = max(α̂, ϕ̂ * α)
    return α̂, α
end

function condition_P̋(P̋::T, ν̂::T, ρ¹::T, α::T, P̃::T, ρ²::T, g::T, ρ³::T, U::T)::Bool where T <: Real
    cond  = (1 + ν̂ - ρ¹ * α^2) * P̃ + ρ² * α^2 * g - ρ³ * α^2 * U
    return P̋ > cond
end

function update_α̂(g::T, α::T, P̋::T, P̃::T)::T where T <: Real
    return -g * α^2 / (2 * (P̋ - P̃ - g * α))
end

function has_nonfinite(A::AbstractArray)
    @inbounds for x in A
        if !isfinite(x)
            return true
        end
    end
    return false
end


function newton(
    # f::Function, 
    fnj::function_and_jacobian, 
    initial_guess::Array{T,1}, 
    parameters_and_solved_vars::Array{T,1},
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1},
    parameters::solver_parameters;
    tol::Tolerances = Tolerances()
    )::Tuple{Vector{T}, Tuple{Int, Int, T, T}} where {T <: AbstractFloat}
    # issues with optimization: https://www.gurobi.com/documentation/8.1/refman/numerics_gurobi_guidelines.html

    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    iterations = 250
    transformation_level = 0 # parameters.transformation_level

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    # @assert all(lower_bounds .< upper_bounds)

    # Get pre-allocated workspace vectors and buffers
    ws = fnj.workspace
    new_guess = ws.current_guess
    copy!(new_guess, initial_guess)
    
    guess_update = ws.lu_buffer.b

    fnj.func(ws.func_buffer, new_guess, parameters_and_solved_vars)

    new_residuals = ws.func_buffer
    # new_residuals = f(new_guess)

    ∇ = ws.jac_buffer

    sol_cache = ws.lu_buffer

    rel_xtol_reached = 1.0
    rel_ftol_reached = 1.0
    new_residuals_norm = 1.0
    guess_update_norm = 1.0
    
    grad_iter = 0
    func_iter = 0
    
    for iter in 1:iterations
    
        if ∇ isa SparseMatrixCSC
            ∇.nzval .= 0
        else
            ∇ .= 0
        end

        fnj.jac(∇, new_guess, parameters_and_solved_vars)

        fnj.func(new_residuals, new_guess, parameters_and_solved_vars)

        finn = has_nonfinite(new_residuals)

        if finn
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
            
            # sol_cache.A = ∇
            # copy!(sol_cache.A, ∇)
            sol_cache.A = ∇
            # sol_cache.A = sol_cache.alg isa 𝒮.FastLUFactorization ? copy(∇) : ∇
            sol_cache.b = new_residuals
            sol = 𝒮.solve!(sol_cache)
            if sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sol.retcode)
                rel_xtol_reached = typemax(T)
                new_residuals_norm = typemax(T)
                break
            end
            guess_update .= sol_cache.u
            if has_nonfinite(guess_update)
                rel_xtol_reached = typemax(T)
                new_residuals_norm = typemax(T)
                break
            end
            # new_residuals .= guess_update
            copy!(new_residuals, guess_update)

            guess_update_norm = ℒ.norm(new_residuals)
            ℒ.axpy!(-1, new_residuals, new_guess)

            # guess_update_norm = ℒ.norm(sol_cache.u)
    
            # ℒ.axpy!(-1, sol_cache.u, new_guess)
    
            grad_iter += 1
            func_iter += 1

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

        # sol_cache.A = ∇

        # sol_cache.b = new_residuals
        # 𝒮.solve!(sol_cache)
        # copy!(guess_update, sol_cache.u)

        # copy!(sol_cache.A, ∇)
        sol_cache.A = ∇
        # sol_cache.A = sol_cache.alg isa 𝒮.FastLUFactorization ? copy(∇) : ∇
        sol_cache.b = new_residuals
        sol = 𝒮.solve!(sol_cache)
        if sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sol.retcode)
            rel_xtol_reached = typemax(T)
            new_residuals_norm = typemax(T)
            break
        end
        # guess_update .= sol_cache.u
        copy!(guess_update, sol_cache.u)
        
        if has_nonfinite(guess_update)
            rel_xtol_reached = typemax(T)
            new_residuals_norm = typemax(T)
            break
        end
        # new_residuals .= guess_update
        copy!(new_residuals, guess_update)

        guess_update_norm = ℒ.norm(new_residuals)
        ℒ.axpy!(-1, new_residuals, new_guess)

        # guess_update_norm = ℒ.norm(guess_update)

        # ℒ.axpy!(-1, guess_update, new_guess)

        finn = has_nonfinite(new_guess)

        if finn
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
        
        grad_iter += 1
        func_iter += 1

        # println("GN: $grad_iter iterations - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")
    end

    # if grad_iter == iterations
    #     println("GN failed to converge - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")#; rel_ftol: $rel_ftol_reached")
    # else
    #     println("GN converged after $grad_iter iterations - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")
    # end

    # best_guess = undo_transform(new_guess,transformation_level)
    
    return new_guess, (grad_iter, func_iter, rel_xtol_reached, new_residuals_norm)
end



function minmax!(x::Vector{Float64},lb::Vector{Float64},ub::Vector{Float64})
    @inbounds for i in eachindex(x)
        x[i] = max(lb[i], min(x[i], ub[i]))
    end
end



# transformation of NSSS problem
# function transform(x::Vector{T}, option::Int, shift::AbstractFloat)::Vector{T} where T <: Real
#     if option == 4
#         return asinh.(asinh.(asinh.(asinh.(x .+ shift))))
#     elseif option == 3
#         return asinh.(asinh.(asinh.(x .+ shift)))
#     elseif option == 2
#         return asinh.(asinh.(x .+ shift))
#     elseif option == 1
#         return asinh.(x .+ shift)
#     else # if option == 0
#         return x .+ shift
#     end
# end

# function transform(x::Vector{T}, option::Int)::Vector{T} where T <: Real
#     if option == 4
#         return asinh.(asinh.(asinh.(asinh.(x))))
#     elseif option == 3
#         return asinh.(asinh.(asinh.(x)))
#     elseif option == 2
#         return asinh.(asinh.(x))
#     elseif option == 1
#         return asinh.(x)
#     else # if option == 0
#         return x
#     end
# end

# function undo_transform(x::Vector{T}, option::R, shift::AbstractFloat)::Vector{T} where {T,R}
#     if option == 4
#         return sinh.(sinh.(sinh.(sinh.(x)))) .- shift
#     elseif option == 3
#         return sinh.(sinh.(sinh.(x))) .- shift
#     elseif option == 2
#         return sinh.(sinh.(x)) .- shift
#     elseif option == 1
#         return sinh.(x) .- shift
#     else # if option == 0
#         return x .- shift
#     end
# end

# function undo_transform(x::Vector{T}, option::R)::Vector{T} where {T,R}
#     if option == 4
#         return sinh.(sinh.(sinh.(sinh.(x))))
#     elseif option == 3
#         return sinh.(sinh.(sinh.(x)))
#     elseif option == 2
#         return sinh.(sinh.(x))
#     elseif option == 1
#         return sinh.(x)
#     else # if option == 0
#         return x
#     end
# end

# function undo_transform(x::T, option::R)::T where {T,R}

#     x = ifelse(option == 1, 
#             sinh(x), 
#             ifelse(option == 2, 
#                 sinh(sinh(x)), 
#                 ifelse(option == 3, 
#                     sinh(sinh(sinh(x))), 
#                     ifelse(option == 4, 
#                         sinh(sinh(sinh(sinh(x)))), 
#                         x
#                     )
#                 )
#             )
#         )
#     return x
# end

end # dispatch_doctor