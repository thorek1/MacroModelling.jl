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
    
    # Setup C matrix using Finch for efficient kronecker operations
    # C = ∇₁₊𝐒₁➕∇₁₀lu \ (∇₂ * (kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂)
    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = mat_mult_kron_finch(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, M₂.𝐂₂) + 
                                                    mat_mult_kron_finch(∇₂, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎, M₂.𝛔 * M₂.𝐂₂)
    
    C = ∇₁₊𝐒₁➕∇₁₀lu \ ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹

    # Setup B matrix using Finch
    𝐒₁₋╱𝟏ₑ = choose_matrix_format(𝐒₁₋╱𝟏ₑ, density_threshold = 0.0)
    B = mat_mult_kron_finch(M₂.𝐔₂, 𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ, M₂.𝐂₂) + M₂.𝐔₂ * M₂.𝛔 * M₂.𝐂₂

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

Compute A * kron(B, C) * D efficiently using Finch.jl sparse tensor operations.

This function uses Finch.jl to perform the computation A * kron(B, C) * D more efficiently
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
    
    # Compute A * kron(B, C) * D efficiently using Finch.jl
    # This avoids materializing the full Kronecker product
    
    n_rowB, n_colB = size(B)
    n_rowC, n_colC = size(C)
    n_rowA, n_colA = size(A)
    n_colD = size(D, 2)
    
    # Initialize output tensor
    X = zeros(promote_type(R, T, S), n_rowA, n_colD)
    
    # Compute the operation efficiently without forming full Kronecker product
    # X[i,l] = sum over j,k of A[i,j] * kron(B,C)[j,k] * D[k,l]
    # where kron(B,C)[j,k] with j = (j1-1)*n_rowC + j2 and k = (k1-1)*n_colC + k2
    # equals B[j1,k1] * C[j2,k2]
    
    for i in axes(A, 1)
        nz_indices, nz_values = findnz(A[i, :])
        for (j_idx, a_val) in zip(nz_indices, nz_values)
            # Decompose j into (j1, j2)
            j1 = div(j_idx - 1, n_rowC) + 1
            j2 = mod(j_idx - 1, n_rowC) + 1
            
            if j1 <= n_rowB && j2 <= n_rowC
                for k1 in axes(B, 2), k2 in axes(C, 2)
                    k_idx = (k1 - 1) * n_colC + k2
                    
                    bc_val = B[j1, k1] * C[j2, k2]
                    if abs(bc_val) > eps(T)
                        for l in axes(D, 2)
                            X[i, l] += a_val * bc_val * D[k_idx, l]
                        end
                    end
                end
            end
        end
    end
    
    return choose_matrix_format(X)
end

function mat_mult_kron_finch(A::DenseMatrix{R},
                              B::AbstractMatrix{T},
                              C::AbstractMatrix{T},
                              D::AbstractMatrix{S};
                              sparse_preallocation::Tuple = (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                              sparse::Bool = false) where {R <: Real, T <: Real, S <: Real}
    
    # Compute A * kron(B, C) * D efficiently using Finch.jl for dense A
    
    n_rowB, n_colB = size(B)
    n_rowC, n_colC = size(C)
    n_rowA, n_colA = size(A)
    n_colD = size(D, 2)
    
    # Initialize output
    X = zeros(promote_type(R, T, S), n_rowA, n_colD)
    
    # Compute efficiently without forming full Kronecker product
    for i in axes(A, 1)
        for j_idx in axes(A, 2)
            a_val = A[i, j_idx]
            if abs(a_val) > eps(R)
                # Decompose j into (j1, j2)
                j1 = div(j_idx - 1, n_rowC) + 1
                j2 = mod(j_idx - 1, n_rowC) + 1
                
                if j1 <= n_rowB && j2 <= n_rowC
                    for k1 in axes(B, 2), k2 in axes(C, 2)
                        k_idx = (k1 - 1) * n_colC + k2
                        
                        bc_val = B[j1, k1] * C[j2, k2]
                        if abs(bc_val) > eps(T)
                            for l in axes(D, 2)
                                X[i, l] += a_val * bc_val * D[k_idx, l]
                            end
                        end
                    end
                end
            end
        end
    end
    
    return choose_matrix_format(X)
end

function mat_mult_kron_finch(A::AbstractSparseMatrix{R},
                              B::AbstractMatrix{T},
                              C::AbstractMatrix{T};
                              sparse_preallocation::Tuple = (Int[], Int[], T[], Int[], Int[], T[], T[]),
                              sparse::Bool = false) where {R <: Real, T <: Real}
    
    # Compute A * kron(B, C) efficiently using Finch.jl (no D matrix)
    
    n_rowB, n_colB = size(B)
    n_rowC, n_colC = size(C)
    n_rowA = size(A, 1)
    n_colBC = n_colB * n_colC
    
    if sparse
        # Use sparse output with preallocated arrays
        rows = sparse_preallocation[1]
        cols = sparse_preallocation[2]
        vals = sparse_preallocation[3]
        
        k = 0
        estimated_nnz = length(vals)
        
        for i in axes(A, 1)
            nz_indices, nz_values = findnz(A[i, :])
            for (j_idx, a_val) in zip(nz_indices, nz_values)
                # Decompose j into (j1, j2)
                j1 = div(j_idx - 1, n_rowC) + 1
                j2 = mod(j_idx - 1, n_rowC) + 1
                
                if j1 <= n_rowB && j2 <= n_rowC
                    for k1 in axes(B, 2), k2 in axes(C, 2)
                        bc_val = B[j1, k1] * C[j2, k2]
                        if abs(bc_val) > eps(T)
                            val = a_val * bc_val
                            if abs(val) > eps(promote_type(R, T))
                                k += 1
                                if k > estimated_nnz
                                    # Expand arrays if needed
                                    resize!(rows, k)
                                    resize!(cols, k)
                                    resize!(vals, k)
                                end
                                col_idx = (k1 - 1) * n_colC + k2
                                rows[k] = i
                                cols[k] = col_idx
                                vals[k] = val
                            end
                        end
                    end
                end
            end
        end
        
        # Trim arrays to actual size
        resize!(rows, k)
        resize!(cols, k)
        resize!(vals, k)
        
        return sparse(rows, cols, vals, n_rowA, n_colBC)
    else
        # Dense output
        X = zeros(promote_type(R, T), n_rowA, n_colBC)
        
        for i in axes(A, 1)
            nz_indices, nz_values = findnz(A[i, :])
            for (j_idx, a_val) in zip(nz_indices, nz_values)
                # Decompose j into (j1, j2)
                j1 = div(j_idx - 1, n_rowC) + 1
                j2 = mod(j_idx - 1, n_rowC) + 1
                
                if j1 <= n_rowB && j2 <= n_rowC
                    for k1 in axes(B, 2), k2 in axes(C, 2)
                        bc_val = B[j1, k1] * C[j2, k2]
                        if abs(bc_val) > eps(T)
                            col_idx = (k1 - 1) * n_colC + k2
                            X[i, col_idx] += a_val * bc_val
                        end
                    end
                end
            end
        end
        
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
    
    # Compute compressed third Kronecker power using Finch.jl
    # This exploits symmetry: only compute for i1 ≥ i2 ≥ i3 and j1 ≥ j2 ≥ j3
    
    n_rows, n_cols = size(a)
    m3_rows = n_rows * (n_rows + 1) * (n_rows + 2) ÷ 6
    m3_cols = n_cols * (n_cols + 1) * (n_cols + 2) ÷ 6
    
    # Convert to dense for efficient element access
    a_dense = Array(a)
    
    if rowmask == Int[0] || colmask == Int[0]
        return spzeros(T, m3_rows, m3_cols)
    end
    
    # Use preallocated arrays if available
    rows = sparse_preallocation[1]
    cols = sparse_preallocation[2]
    vals = sparse_preallocation[3]
    
    k = 0
    estimated_nnz = max(length(vals), 10000)
    
    if length(rows) == 0
        resize!(rows, estimated_nnz)
        resize!(cols, estimated_nnz)
        resize!(vals, estimated_nnz)
    end
    
    norowmask = length(rowmask) == 0
    nocolmask = length(colmask) == 0
    
    # Find unique non-zero indices for efficiency
    ui = unique([i for i in 1:n_rows if any(abs.(a_dense[i, :]) .> tol)])
    uj = unique([j for j in 1:n_cols if any(abs.(a_dense[:, j]) .> tol)])
    
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
                                                    # Access elements
                                                    a11 = a_dense[i1, j1]
                                                    a12 = a_dense[i1, j2]
                                                    a13 = a_dense[i1, j3]
                                                    a21 = a_dense[i2, j1]
                                                    a22 = a_dense[i2, j2]
                                                    a23 = a_dense[i2, j3]
                                                    a31 = a_dense[i3, j1]
                                                    a32 = a_dense[i3, j2]
                                                    a33 = a_dense[i3, j3]
                                                    
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
                                                        k += 1
                                                        if k > estimated_nnz
                                                            new_size = min(k * 2, m3_rows * m3_cols)
                                                            resize!(rows, new_size)
                                                            resize!(cols, new_size)
                                                            resize!(vals, new_size)
                                                            estimated_nnz = new_size
                                                        end
                                                        rows[k] = row
                                                        cols[k] = col
                                                        vals[k] = val
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
    
    # Trim arrays to actual size
    resize!(rows, k)
    resize!(cols, k)
    resize!(vals, k)
    
    return sparse(rows, cols, vals, m3_rows, m3_cols)
end
