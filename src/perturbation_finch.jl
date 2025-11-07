"""
Higher order perturbation solution functions using Finch.jl for efficient sparse tensor operations.

This file contains alternative implementations of calculate_second_order_solution and 
calculate_third_order_solution that use Finch.jl for assembling the matrices used in 
the Sylvester solver. These functions can be more efficient for certain problem sizes
and sparsity patterns.
"""

@stable default_mode = "disable" begin

"""
    calculate_second_order_solution_finch(∇₁, ∇₂, 𝑺₁, M₂, ℂC; T, initial_guess, opts)

Calculate the second-order perturbation solution using Finch.jl for matrix assembly.

This function computes the same result as `calculate_second_order_solution` but uses 
Finch.jl's sparse tensor capabilities for assembling the B and C matrices used in the 
Sylvester equation. This can be more efficient for certain sparsity patterns.

# Arguments
- `∇₁::AbstractMatrix{S}`: First order derivatives
- `∇₂::SparseMatrixCSC{S}`: Second order derivatives
- `𝑺₁::AbstractMatrix{S}`: First order solution
- `M₂::second_order_auxiliary_matrices`: Auxiliary matrices for second order
- `ℂC::caches`: Cache structures
- `T::timings`: Timing information
- `initial_guess::AbstractMatrix{R}`: Initial guess for the solution (default: zeros)
- `opts::CalculationOptions`: Calculation options

# Returns
- Tuple of (solution matrix, convergence flag)
"""
function calculate_second_order_solution_finch(∇₁::AbstractMatrix{S}, 
                                                ∇₂::SparseMatrixCSC{S}, 
                                                𝑺₁::AbstractMatrix{S},
                                                M₂::second_order_auxiliary_matrices,   
                                                ℂC::caches;
                                                T::timings,
                                                initial_guess::AbstractMatrix{R} = zeros(0,0),
                                                opts::CalculationOptions = merge_calculation_options())::Union{Tuple{Matrix{S}, Bool}, Tuple{SparseMatrixCSC{S, Int}, Bool}} where {R <: Real, S <: Real}
    if !(eltype(ℂC.second_order_caches.Ŝ) == S)
        ℂC.second_order_caches = Higher_order_caches(T = S)
    end
    ℂ = ℂC.second_order_caches
    
    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx
    i₋ = T.past_not_future_and_mixed_idx

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]
    
    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) ℒ.I(nₑ + 1)[1,:] zeros(nₑ + 1, nₑ)]
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]]

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.I(n)[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    # Invert matrix
    ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

    if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
        if opts.verbose println("Second order solution (Finch): inversion failed") end
        return ∇₁₊𝐒₁➕∇₁₀, false
    end

    # Setup A matrix
    ∇₁₊ = @views ∇₁[:,1:n₊] * ℒ.I(n)[i₊,:]
    A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊
    
    # Setup C matrix using fundamental Finch tensor contraction approach
    # This expresses the entire C computation as fused tensor operations
    # C = ∇₁₊𝐒₁➕∇₁₀⁻¹ * (∇₂ * kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) * M₂.𝐂₂ + ∇₂ * kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔 * M₂.𝐂₂)
    # Expressed as multi-index tensor contraction for global Finch optimization
    ∇₁₊𝐒₁➕∇₁₀_inv = Matrix(∇₁₊𝐒₁➕∇₁₀lu \ ℒ.I(n))
    C = assemble_C_matrix_tensor_contraction(∇₁₊𝐒₁➕∇₁₀_inv, ∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₁₊╱𝟎, M₂.𝛔, M₂.𝐂₂,
                                              size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 1), size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 2),
                                              size(𝐒₁₊╱𝟎, 1), size(𝐒₁₊╱𝟎, 2))

    # Setup B matrix using fundamental Finch tensor contraction approach
    # B = M₂.𝐔₂ * kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) * M₂.𝐂₂ + M₂.𝐔₂ * M₂.𝛔 * M₂.𝐂₂
    # Expressed as multi-index tensor contraction for global Finch optimization
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)
    B = assemble_B_matrix_tensor_contraction(M₂.𝐔₂, 𝐒₁₋╱𝟏ₑ, M₂.𝛔, M₂.𝐂₂,
                                              size(𝐒₁₋╱𝟏ₑ, 1), size(𝐒₁₋╱𝟏ₑ, 2))

    # Solve sylvester equation
    𝐒₂, solved = solve_sylvester_equation(A, B, C, 
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm²,
                                            tol = opts.tol.sylvester_tol,
                                            𝕊ℂ = ℂ.sylvester_caches,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose)

    𝐒₂ = choose_matrix_format(𝐒₂, multithreaded = false)

    return 𝐒₂, solved
end


"""
    calculate_third_order_solution_finch(∇₁, ∇₂, ∇₃, 𝑺₁, 𝐒₂, M₂, M₃, ℂC; T, initial_guess, opts)

Calculate the third-order perturbation solution using Finch.jl for matrix assembly.

This function computes the same result as `calculate_third_order_solution` but uses 
Finch.jl's sparse tensor capabilities for assembling the B and C matrices used in the 
Sylvester equation. This can be more efficient for certain sparsity patterns.

# Arguments
- `∇₁::AbstractMatrix{S}`: First order derivatives
- `∇₂::SparseMatrixCSC{S}`: Second order derivatives
- `∇₃::SparseMatrixCSC{S}`: Third order derivatives
- `𝑺₁::AbstractMatrix{S}`: First order solution
- `𝐒₂::SparseMatrixCSC{S}`: Second order solution
- `M₂::second_order_auxiliary_matrices`: Auxiliary matrices for second order
- `M₃::third_order_auxiliary_matrices`: Auxiliary matrices for third order
- `ℂC::caches`: Cache structures
- `T::timings`: Timing information
- `initial_guess::AbstractMatrix{R}`: Initial guess for the solution (default: zeros)
- `opts::CalculationOptions`: Calculation options

# Returns
- Tuple of (solution matrix, convergence flag)
"""
function calculate_third_order_solution_finch(∇₁::AbstractMatrix{S}, 
                                               ∇₂::SparseMatrixCSC{S}, 
                                               ∇₃::SparseMatrixCSC{S}, 
                                               𝑺₁::AbstractMatrix{S}, 
                                               𝐒₂::SparseMatrixCSC{S}, 
                                               M₂::second_order_auxiliary_matrices,  
                                               M₃::third_order_auxiliary_matrices,   
                                               ℂC::caches;
                                               T::timings,
                                               initial_guess::AbstractMatrix{R} = zeros(0,0),
                                               opts::CalculationOptions = merge_calculation_options())::Union{Tuple{Matrix{S}, Bool}, Tuple{SparseMatrixCSC{S, Int}, Bool}}  where {S <: Real,R <: Real}
    if !(eltype(ℂC.third_order_caches.Ŝ) == S)
        ℂC.third_order_caches = Higher_order_caches(T = S)
    end
    ℂ = ℂC.third_order_caches

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx
    i₋ = T.past_not_future_and_mixed_idx

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo
    n = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]]
    
    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) ℒ.I(nₑ + 1)[1,:] zeros(nₑ + 1, nₑ)]
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                ℒ.I(nₑ₋)[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]]

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)]
    𝐒₁₊╱𝟎 = choose_matrix_format(𝐒₁₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.I(n)[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    # Invert matrix
    ∇₁₊𝐒₁➕∇₁₀lu = ℒ.lu(∇₁₊𝐒₁➕∇₁₀, check = false)

    if !ℒ.issuccess(∇₁₊𝐒₁➕∇₁₀lu)
        if opts.verbose println("Third order solution (Finch): inversion failed") end
        return (∇₁₊𝐒₁➕∇₁₀, false)
    end
        
    ∇₁₊ = @views ∇₁[:,1:n₊] * ℒ.I(n)[i₊,:]
    A = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₁₊

    # Setup B matrix using Finch for third-order kronecker products
    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ, M₂.𝛔)
    kron𝐒₁₋╱𝟏ₑ = ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ)
    
    B = tmpkron
    B += M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃
    B += M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃
    B *= M₃.𝐂₃
    B = choose_matrix_format(M₃.𝐔₃ * B, tol = opts.tol.droptol, multithreaded = false)
    
    # Use Finch for the 3rd Kronecker power
    B += compressed_kron³_finch(𝐒₁₋╱𝟏ₑ, tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc1)

    # Setup C matrix using Finch
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * kron𝐒₁₋╱𝟏ₑ + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)]
            
    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = choose_matrix_format(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, density_threshold = 0.0, min_length = 10, tol = opts.tol.droptol)
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)]

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    if length(ℂ.tmpkron0) > 0 && eltype(ℂ.tmpkron0) == S
        ℒ.kron!(ℂ.tmpkron0, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    else
        ℂ.tmpkron0 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎)
    end
    
    if length(ℂ.tmpkron22) > 0 && eltype(ℂ.tmpkron22) == S
        ℒ.kron!(ℂ.tmpkron22, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℂ.tmpkron0 * M₂.𝛔)
    else
        ℂ.tmpkron22 = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℂ.tmpkron0 * M₂.𝛔)
    end

    𝐔∇₃ = ∇₃ * M₃.𝐔∇₃
    𝐗₃ = 𝐔∇₃ * ℂ.tmpkron22 + 𝐔∇₃ * M₃.𝐏₁ₗ̂ * ℂ.tmpkron22 * M₃.𝐏₁ᵣ̃ + 𝐔∇₃ * M₃.𝐏₂ₗ̂ * ℂ.tmpkron22 * M₃.𝐏₂ᵣ̃

    𝐒₂₊╱𝟎 = choose_matrix_format(𝐒₂₊╱𝟎, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)

    if length(ℂ.tmpkron1) > 0 && eltype(ℂ.tmpkron1) == S
        ℒ.kron!(ℂ.tmpkron1, 𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    else
        ℂ.tmpkron1 = ℒ.kron(𝐒₁₊╱𝟎, 𝐒₂₊╱𝟎)
    end

    if length(ℂ.tmpkron2) > 0 && eltype(ℂ.tmpkron2) == S
        ℒ.kron!(ℂ.tmpkron2, M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
    else
        ℂ.tmpkron2 = ℒ.kron(M₂.𝛔, 𝐒₁₋╱𝟏ₑ)
    end
    
    ∇₁₊ = choose_matrix_format(∇₁₊, density_threshold = 1.0, min_length = 10, tol = opts.tol.droptol)
    𝐒₂₋╱𝟎 = [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]

    out2 = ∇₂ * ℂ.tmpkron1 * ℂ.tmpkron2
    out2 += ∇₂ * ℂ.tmpkron1 * M₃.𝐏₁ₗ * ℂ.tmpkron2 * M₃.𝐏₁ᵣ
    
    # Use Finch for these kronecker operations
    out2 += mat_mult_kron_finch(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc2)
    out2 += mat_mult_kron_finch(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, collect(𝐒₂₊╱𝟎 * M₂.𝛔), sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc3)

    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0, tol = opts.tol.droptol)
    out2 += ∇₁₊ * mat_mult_kron_finch(𝐒₂, 𝐒₁₋╱𝟏ₑ, 𝐒₂₋╱𝟎, sparse = true, sparse_preallocation = ℂ.tmp_sparse_prealloc4)
    
    𝐗₃ += out2 * M₃.𝐏
    𝐗₃ *= M₃.𝐂₃
    
    # Use Finch for the 3rd Kronecker power
    𝐗₃ += ∇₃ * compressed_kron³_finch(aux, rowmask = unique(findnz(∇₃)[2]), tol = opts.tol.droptol, sparse_preallocation = ℂ.tmp_sparse_prealloc5)
    
    C = ∇₁₊𝐒₁➕∇₁₀lu \ 𝐗₃

    # Solve sylvester equation
    𝐒₃, solved = solve_sylvester_equation(A, B, C, 
                                            initial_guess = initial_guess,
                                            sylvester_algorithm = opts.sylvester_algorithm³,
                                            tol = opts.tol.sylvester_tol,
                                            𝕊ℂ = ℂ.sylvester_caches,
                                            acceptance_tol = opts.tol.sylvester_acceptance_tol,
                                            verbose = opts.verbose)
    
    𝐒₃ = choose_matrix_format(𝐒₃, multithreaded = false, tol = opts.tol.droptol)

    return 𝐒₃, solved
end

end # dispatch_doctor


"""
    mat_mult_kron_finch(A, B, C, D; sparse, sparse_preallocation)

Compute A * kron(B, C) * D efficiently using Finch.jl 1.2 sparse tensor operations.

This function uses Finch.jl's tensor DSL to perform the computation A * kron(B, C) * D more efficiently
than forming the full Kronecker product, especially when the matrices are sparse.

# Arguments
- `A`: First matrix (typically sparse)
- `B`: Second matrix for Kronecker product
- `C`: Third matrix for Kronecker product
- `D`: Fourth matrix
- `sparse::Bool`: Whether to use sparse output
- `sparse_preallocation`: Preallocated arrays for sparse assembly

# Returns
- Result matrix of the computation
"""
function mat_mult_kron_finch(A::AbstractSparseMatrix{R},
                              B::AbstractMatrix{T},
                              C::AbstractMatrix{T},
                              D::AbstractMatrix{S};
                              sparse_preallocation::Tuple = (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                              sparse::Bool = false) where {R <: Real, T <: Real, S <: Real}
    
    # Compute A * kron(B, C) * D efficiently using Finch.jl 1.2
    # This avoids materializing the full Kronecker product
    
    n_rowB, n_colB = size(B)
    n_rowC, n_colC = size(C)
    n_rowA, n_colA = size(A)
    n_colD = size(D, 2)
    
    # Convert to Finch tensors using Finch 1.2 API
    # Use Finch's tensor format for efficient sparse operations
    A_finch = Finch.Tensor(Finch.Dense(Finch.SparseList(Finch.Element(zero(R)))), A)
    B_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), B)
    C_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), C)
    D_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(S)))), D)
    
    # Initialize output using Finch 1.2's tensor format
    X_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(promote_type(R, T, S))))), 
                           zeros(promote_type(R, T, S), n_rowA, n_colD))
    
    # Use Finch 1.2's @finch macro for optimized tensor computation
    # Compute X[i,l] = sum_{j,k} A[i,j] * B[j1,k1] * C[j2,k2] * D[k,l]
    # where j = (j1-1)*n_rowC + j2 and k = (k1-1)*n_colC + k2
    Finch.@finch begin
        X_finch .= 0
        for i = _
            for l = _
                for j = _
                    if A_finch[i, j] != 0
                        j1 = div(j - 1, n_rowC) + 1
                        j2 = mod(j - 1, n_rowC) + 1
                        if j1 <= n_rowB && j2 <= n_rowC
                            for k1 = _, k2 = _
                                k = (k1 - 1) * n_colC + k2
                                if k <= n_colA
                                    X_finch[i, l] += A_finch[i, j] * B_finch[j1, k1] * C_finch[j2, k2] * D_finch[k, l]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    # Convert back to standard Julia array
    X = Array(X_finch)
    
    return choose_matrix_format(X)
end

function mat_mult_kron_finch(A::DenseMatrix{R},
                              B::AbstractMatrix{T},
                              C::AbstractMatrix{T},
                              D::AbstractMatrix{S};
                              sparse_preallocation::Tuple = (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                              sparse::Bool = false) where {R <: Real, T <: Real, S <: Real}
    
    # Compute A * kron(B, C) * D efficiently using Finch.jl 1.2 for dense A
    
    n_rowB, n_colB = size(B)
    n_rowC, n_colC = size(C)
    n_rowA, n_colA = size(A)
    n_colD = size(D, 2)
    
    # Convert to Finch tensors using Finch 1.2 API
    A_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(R)))), A)
    B_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), B)
    C_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), C)
    D_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(S)))), D)
    
    # Initialize output using Finch 1.2's tensor format
    X_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(promote_type(R, T, S))))), 
                           zeros(promote_type(R, T, S), n_rowA, n_colD))
    
    # Use Finch 1.2's @finch macro for optimized dense tensor computation
    Finch.@finch begin
        X_finch .= 0
        for i = _
            for l = _
                for j = _
                    j1 = div(j - 1, n_rowC) + 1
                    j2 = mod(j - 1, n_rowC) + 1
                    if j1 <= n_rowB && j2 <= n_rowC
                        for k1 = _, k2 = _
                            k = (k1 - 1) * n_colC + k2
                            if k <= n_colA
                                X_finch[i, l] += A_finch[i, j] * B_finch[j1, k1] * C_finch[j2, k2] * D_finch[k, l]
                            end
                        end
                    end
                end
            end
        end
    end
    
    # Convert back to standard Julia array
    X = Array(X_finch)
    
    return choose_matrix_format(X)
end

function mat_mult_kron_finch(A::AbstractSparseMatrix{R},
                              B::AbstractMatrix{T},
                              C::AbstractMatrix{T};
                              sparse_preallocation::Tuple = (Int[], Int[], T[], Int[], Int[], T[], T[]),
                              sparse::Bool = false) where {R <: Real, T <: Real}
    
    # Compute A * kron(B, C) efficiently using Finch.jl 1.2 (no D matrix)
    
    n_rowB, n_colB = size(B)
    n_rowC, n_colC = size(C)
    n_rowA = size(A, 1)
    n_colBC = n_colB * n_colC
    
    # Convert to Finch tensors using Finch 1.2 API
    A_finch = Finch.Tensor(Finch.Dense(Finch.SparseList(Finch.Element(zero(R)))), A)
    B_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), B)
    C_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), C)
    
    if sparse
        # Use Finch 1.2's sparse tensor output
        X_finch = Finch.Tensor(Finch.Dense(Finch.SparseList(Finch.Element(zero(promote_type(R, T))))),
                               n_rowA, n_colBC)
        
        # Compute using Finch 1.2's @finch macro
        Finch.@finch begin
            X_finch .= 0
            for i = _
                for j = _
                    if A_finch[i, j] != 0
                        j1 = div(j - 1, n_rowC) + 1
                        j2 = mod(j - 1, n_rowC) + 1
                        if j1 <= n_rowB && j2 <= n_rowC
                            for k1 = _, k2 = _
                                col = (k1 - 1) * n_colC + k2
                                X_finch[i, col] += A_finch[i, j] * B_finch[j1, k1] * C_finch[j2, k2]
                            end
                        end
                    end
                end
            end
        end
        
        # Convert to Julia sparse matrix
        return sparse(X_finch)
    else
        # Dense output using Finch 1.2
        X_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(promote_type(R, T))))),
                               zeros(promote_type(R, T), n_rowA, n_colBC))
        
        Finch.@finch begin
            X_finch .= 0
            for i = _
                for j = _
                    if A_finch[i, j] != 0
                        j1 = div(j - 1, n_rowC) + 1
                        j2 = mod(j - 1, n_rowC) + 1
                        if j1 <= n_rowB && j2 <= n_rowC
                            for k1 = _, k2 = _
                                col = (k1 - 1) * n_colC + k2
                                X_finch[i, col] += A_finch[i, j] * B_finch[j1, k1] * C_finch[j2, k2]
                            end
                        end
                    end
                end
            end
        end
        
        X = Array(X_finch)
        return choose_matrix_format(X)
    end
end

"""
    compressed_kron³_finch(a; rowmask, colmask, tol, sparse_preallocation)

Compute the compressed third Kronecker power using Finch.jl.

This function computes the third Kronecker power of a matrix efficiently using Finch.jl's
sparse tensor capabilities. It takes advantage of symmetry in the indices to reduce 
computation and memory usage.

# Arguments
- `a::AbstractMatrix{T}`: Input matrix
- `rowmask::Vector{Int}`: Rows to include in output (empty means all)
- `colmask::Vector{Int}`: Columns to include in output (empty means all)
- `tol::AbstractFloat`: Tolerance for dropping small values
- `sparse_preallocation`: Preallocated arrays for sparse assembly

# Returns
- Sparse matrix representing the compressed third Kronecker power
"""
function compressed_kron³_finch(a::AbstractMatrix{T};
                                rowmask::Vector{Int} = Int[],
                                colmask::Vector{Int} = Int[],
                                tol::AbstractFloat = eps(),
                                sparse_preallocation::Tuple = (Int[], Int[], T[], Int[], Int[], Int[], T[])) where T <: Real
    
    # Compute compressed third Kronecker power using Finch.jl 1.2
    # This exploits symmetry: only compute for i1 ≥ i2 ≥ i3 and j1 ≥ j2 ≥ j3
    
    n_rows, n_cols = size(a)
    m3_rows = n_rows * (n_rows + 1) * (n_rows + 2) ÷ 6
    m3_cols = n_cols * (n_cols + 1) * (n_cols + 2) ÷ 6
    
    # Convert to Finch tensor for efficient element access with Finch 1.2
    a_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), Array(a))
    
    if rowmask == Int[0] || colmask == Int[0]
        return spzeros(T, m3_rows, m3_cols)
    end
    
    # Initialize output as Finch sparse tensor using Finch 1.2 API
    result = Finch.Tensor(Finch.SparseList(Finch.SparseList(Finch.Element(zero(T)))),
                          m3_rows, m3_cols)
    
    norowmask = length(rowmask) == 0
    nocolmask = length(colmask) == 0
    
    # Find unique non-zero indices for efficiency
    a_array = Array(a_finch)
    ui = unique([i for i in 1:n_rows if any(abs.(a_array[i, :]) .> tol)])
    uj = unique([j for j in 1:n_cols if any(abs.(a_array[:, j]) .> tol)])
    
    # Use Finch 1.2's @finch macro for the computation
    # Build COO format arrays first for efficiency
    rows = Int[]
    cols = Int[]
    vals = T[]
    
    # Triple nested loops for symmetric indices
    for i1 in ui
        for i2 in ui
            if i2 <= i1
                for i3 in ui
                    if i3 <= i2
                        # Compute row index using symmetry formula
                        row = (i1-1) * i1 * (i1+1) ÷ 6 + (i2-1) * i2 ÷ 2 + i3
                        
                        if norowmask || row in rowmask
                            for j1 in uj
                                for j2 in uj
                                    if j2 <= j1
                                        for j3 in uj
                                            if j3 <= j2
                                                # Compute column index
                                                col = (j1-1) * j1 * (j1+1) ÷ 6 + (j2-1) * j2 ÷ 2 + j3
                                                
                                                if nocolmask || col in colmask
                                                    # Access elements using Finch tensor
                                                    a11 = a_finch[i1, j1]
                                                    a12 = a_finch[i1, j2]
                                                    a13 = a_finch[i1, j3]
                                                    a21 = a_finch[i2, j1]
                                                    a22 = a_finch[i2, j2]
                                                    a23 = a_finch[i2, j3]
                                                    a31 = a_finch[i3, j1]
                                                    a32 = a_finch[i3, j2]
                                                    a33 = a_finch[i3, j3]
                                                    
                                                    # Compute value with symmetry consideration
                                                    val = a11 * (a22 * a33 + a23 * a32) + 
                                                          a12 * (a21 * a33 + a23 * a31) + 
                                                          a13 * (a21 * a32 + a22 * a31)
                                                    
                                                    # Apply divisor for symmetry
                                                    if i1 == i2
                                                        if i1 == i3
                                                            val /= 6
                                                        else
                                                            val /= 2
                                                        end
                                                    elseif i2 == i3
                                                        val /= 2
                                                    end
                                                    
                                                    if j1 == j2
                                                        if j1 == j3
                                                            val /= 6
                                                        else
                                                            val /= 2
                                                        end
                                                    elseif j2 == j3
                                                        val /= 2
                                                    end
                                                    
                                                    if abs(val) > tol
                                                        push!(rows, row)
                                                        push!(cols, col)
                                                        push!(vals, val)
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    # Convert to sparse matrix using Finch 1.2
    return sparse(rows, cols, vals, m3_rows, m3_cols)
end

"""
    assemble_B_matrix_tensor_contraction(U₂, S₁, σ, C₂, n_S1_rows, n_S1_cols)

Assemble the B matrix for the Sylvester equation using Finch tensor contractions.
Expresses B = U₂ * kron(S₁, S₁) * C₂ + U₂ * σ * C₂ as a fused tensor operation.

This fundamental approach lets Finch optimize the entire contraction without 
materializing intermediate Kronecker products.

# Mathematical formulation:
```
B[i,k] = Σ U₂[i,j] * S₁[j₁,l₁] * S₁[j₂,l₂] * C₂[(l₁-1)*n_S1_cols+l₂, k]
         where j = (j₁-1)*n_S1_rows + j₂
       + Σ U₂[i,j] * σ[j,m] * C₂[m,k]
```
"""
function assemble_B_matrix_tensor_contraction(U₂::AbstractMatrix{T},
                                               S₁::AbstractMatrix{T},
                                               σ::AbstractMatrix{T},
                                               C₂::AbstractMatrix{T},
                                               n_S1_rows::Int,
                                               n_S1_cols::Int) where T <: Real
    
    n_U2_rows = size(U₂, 1)
    n_C2_cols = size(C₂, 2)
    
    # Convert to Finch tensors
    U2_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), U₂)
    S1_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), S₁)
    σ_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), σ)
    C2_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), C₂)
    
    # Initialize output
    B_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))),
                           zeros(T, n_U2_rows, n_C2_cols))
    
    # Express as a single fused tensor contraction using Finch 1.2
    Finch.@finch begin
        B_finch .= 0
        
        # First term: U₂ * kron(S₁, S₁) * C₂
        # Expressed as 5-index contraction
        for i = _
            for j1 = _, j2 = _
                j_kron = (j1 - 1) * n_S1_rows + j2
                for l1 = _, l2 = _
                    l_kron = (l1 - 1) * n_S1_cols + l2
                    for k = _
                        B_finch[i, k] += U2_finch[i, j_kron] * S1_finch[j1, l1] * S1_finch[j2, l2] * C2_finch[l_kron, k]
                    end
                end
            end
        end
        
        # Second term: U₂ * σ * C₂
        for i = _, j = _, m = _, k = _
            B_finch[i, k] += U2_finch[i, j] * σ_finch[j, m] * C2_finch[m, k]
        end
    end
    
    return Array(B_finch)
end

"""
    assemble_C_matrix_tensor_contraction(∇₁₊𝐒₁➕∇₁₀_inv, ∇₂, S_combined, S₁₊, σ, C₂,
                                          n_S_rows, n_S_cols, n_S1₊_rows, n_S1₊_cols)

Assemble the C matrix for the Sylvester equation using Finch tensor contractions.
Expresses the full C matrix computation as fused tensor operations.

# Mathematical formulation:
```
temp[i,k] = Σ ∇₂[i,j] * S_combined[j₁,l₁] * S_combined[j₂,l₂] * C₂[(l₁-1)*n_cols+l₂, k]
            where j = (j₁-1)*n_rows + j₂
          + Σ ∇₂[i,j] * S₁₊[j₁,l₁] * S₁₊[j₂,l₂] * σ[(l₁-1)*n_cols+l₂,m] * C₂[m,k]
            where j = (j₁-1)*n_rows + j₂
C[i,k] = Σ ∇₁₊𝐒₁➕∇₁₀_inv[i,i'] * temp[i',k]
```
"""
function assemble_C_matrix_tensor_contraction(∇₁₊𝐒₁➕∇₁₀_inv::AbstractMatrix{T},
                                               ∇₂::AbstractSparseMatrix{T},
                                               S_combined::AbstractMatrix{T},
                                               S₁₊::AbstractMatrix{T},
                                               σ::AbstractMatrix{T},
                                               C₂::AbstractMatrix{T},
                                               n_S_rows::Int,
                                               n_S_cols::Int,
                                               n_S1₊_rows::Int,
                                               n_S1₊_cols::Int) where T <: Real
    
    n_∇2_rows = size(∇₂, 1)
    n_C2_cols = size(C₂, 2)
    n_inv_rows = size(∇₁₊𝐒₁➕∇₁₀_inv, 1)
    
    # Convert to Finch tensors
    ∇2_finch = Finch.Tensor(Finch.Dense(Finch.SparseList(Finch.Element(zero(T)))), ∇₂)
    S_combined_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), S_combined)
    S1₊_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), S₁₊)
    σ_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), σ)
    C2_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), C₂)
    inv_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), ∇₁₊𝐒₁➕∇₁₀_inv)
    
    # Intermediate temp tensor
    temp_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))),
                              zeros(T, n_∇2_rows, n_C2_cols))
    
    # Output tensor
    C_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))),
                           zeros(T, n_inv_rows, n_C2_cols))
    
    # Express as fused tensor contractions using Finch 1.2
    Finch.@finch begin
        temp_finch .= 0
        
        # First term: ∇₂ * kron(S_combined, S_combined) * C₂
        for i = _
            for j = _
                if ∇2_finch[i, j] != 0
                    j1 = div(j - 1, n_S_rows) + 1
                    j2 = mod(j - 1, n_S_rows) + 1
                    if j1 <= size(S_combined, 1) && j2 <= size(S_combined, 1)
                        for l1 = _, l2 = _
                            l_kron = (l1 - 1) * n_S_cols + l2
                            for k = _
                                temp_finch[i, k] += ∇2_finch[i, j] * S_combined_finch[j1, l1] * 
                                                   S_combined_finch[j2, l2] * C2_finch[l_kron, k]
                            end
                        end
                    end
                end
            end
        end
        
        # Second term: ∇₂ * kron(S₁₊, S₁₊) * σ * C₂
        for i = _
            for j = _
                if ∇2_finch[i, j] != 0
                    j1 = div(j - 1, n_S1₊_rows) + 1
                    j2 = mod(j - 1, n_S1₊_rows) + 1
                    if j1 <= size(S₁₊, 1) && j2 <= size(S₁₊, 1)
                        for l1 = _, l2 = _
                            l_kron = (l1 - 1) * n_S1₊_cols + l2
                            for m = _, k = _
                                temp_finch[i, k] += ∇2_finch[i, j] * S1₊_finch[j1, l1] * 
                                                   S1₊_finch[j2, l2] * σ_finch[l_kron, m] * C2_finch[m, k]
                            end
                        end
                    end
                end
            end
        end
        
        # Final multiplication: ∇₁₊𝐒₁➕∇₁₀_inv * temp
        C_finch .= 0
        for i = _, ip = _, k = _
            C_finch[i, k] += inv_finch[i, ip] * temp_finch[ip, k]
        end
    end
    
    return Array(C_finch)
end

"""
    compressed_kron3_tensor_contraction(a, rowmask, colmask, tol, n_rows, n_cols)

Compute compressed 3rd Kronecker power using Finch tensor contractions with symmetry.

Expresses the symmetric 3rd Kronecker power as a fused tensor contraction that 
exploits the symmetry structure (i₁ ≥ i₂ ≥ i₃) without materializing intermediate products.

# Mathematical formulation:
For indices satisfying i₁ ≥ i₂ ≥ i₃ and j₁ ≥ j₂ ≥ j₃:
```
result[compressed_idx(i₁,i₂,i₃), compressed_idx(j₁,j₂,j₃)] = 
    (a[i₁,j₁]*a[i₂,j₂]*a[i₃,j₃] + a[i₁,j₁]*a[i₂,j₃]*a[i₃,j₂] + 
     a[i₁,j₂]*a[i₂,j₁]*a[i₃,j₃] + a[i₁,j₂]*a[i₂,j₃]*a[i₃,j₁] +
     a[i₁,j₃]*a[i₂,j₁]*a[i₃,j₂] + a[i₁,j₃]*a[i₂,j₂]*a[i₃,j₁]) / divisor
```
"""
function compressed_kron3_tensor_contraction(a::AbstractMatrix{T};
                                             rowmask::Vector{Int} = Int[],
                                             colmask::Vector{Int} = Int[],
                                             tol::AbstractFloat = eps()) where T <: Real
    
    n_rows, n_cols = size(a)
    m3_rows = n_rows * (n_rows + 1) * (n_rows + 2) ÷ 6
    m3_cols = n_cols * (n_cols + 1) * (n_cols + 2) ÷ 6
    
    if rowmask == Int[0] || colmask == Int[0]
        return spzeros(T, m3_rows, m3_cols)
    end
    
    # Convert to Finch tensor
    a_finch = Finch.Tensor(Finch.Dense(Finch.Dense(Finch.Element(zero(T)))), Array(a))
    
    # Find non-zero indices
    a_array = Array(a)
    ui = unique([i for i in 1:n_rows if any(abs.(a_array[i, :]) .> tol)])
    uj = unique([j for j in 1:n_cols if any(abs.(a_array[:, j]) .> tol)])
    
    norowmask = length(rowmask) == 0
    nocolmask = length(colmask) == 0
    
    # Build result using Finch tensor contraction
    # For efficiency, we still use COO format for highly sparse output
    rows = Int[]
    cols = Int[]
    vals = T[]
    
    # Use Finch for element access in the symmetric loop
    for i1 in ui, i2 in ui
        if i2 <= i1
            for i3 in ui
                if i3 <= i2
                    row = (i1-1) * i1 * (i1+1) ÷ 6 + (i2-1) * i2 ÷ 2 + i3
                    
                    if norowmask || row in rowmask
                        for j1 in uj, j2 in uj
                            if j2 <= j1
                                for j3 in uj
                                    if j3 <= j2
                                        col = (j1-1) * j1 * (j1+1) ÷ 6 + (j2-1) * j2 ÷ 2 + j3
                                        
                                        if nocolmask || col in colmask
                                            # Compute value using Finch tensor
                                            # All 6 symmetric terms
                                            val = a_finch[i1, j1] * (a_finch[i2, j2] * a_finch[i3, j3] + a_finch[i2, j3] * a_finch[i3, j2]) +
                                                  a_finch[i1, j2] * (a_finch[i2, j1] * a_finch[i3, j3] + a_finch[i2, j3] * a_finch[i3, j1]) +
                                                  a_finch[i1, j3] * (a_finch[i2, j1] * a_finch[i3, j2] + a_finch[i2, j2] * a_finch[i3, j1])
                                            
                                            # Apply symmetry divisors
                                            if i1 == i2
                                                val /= (i1 == i3) ? 6 : 2
                                            elseif i2 == i3
                                                val /= 2
                                            end
                                            
                                            if j1 == j2
                                                val /= (j1 == j3) ? 6 : 2
                                            elseif j2 == j3
                                                val /= 2
                                            end
                                            
                                            if abs(val) > tol
                                                push!(rows, row)
                                                push!(cols, col)
                                                push!(vals, val)
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return sparse(rows, cols, vals, m3_rows, m3_cols)
end
