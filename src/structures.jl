# struct FWrap#{F}
#     f#::F
# end
# (F::FWrap)(x...) = F.f(x...)

# struct MyFunction{T, S}
#     f
# end

# # Define a method for the custom type that takes in an input of type T
# # and returns an output of type S
# function (mf::MyFunction{T, S})(x::T) where {T, S}
#     return mf.f(x)::S
# end

# # Example function to be wrapped
# function square(x)
#     return x^2
# end

# # Wrap the square function in the custom type with input and output type specialization
# my_square = MyFunction{Int, Int}(square)

# # Test the wrapped function
# println(my_square(2.0))  # Output: 4


# struct FWrap{T, S}
#     f#::Function
# end

# # Define a method for the custom type that takes in an input of type T
# # and returns an output of type S
# function (mf::FWrap{T, S})(x::T) where {T, S}
#     return mf.f(x...)::S
# end

# # Example function to be wrapped
# function square(x::Int)
#     return x^2
# end

# # Wrap the square function in the custom type with input and output type specialization
# my_square = MyFunction{Int, Int}(square)

# # Test the wrapped function
# println(my_square(2))  # Output: 4
# using SparseArrays

# function model_jacobian(x,y,z) sparse(rand(2,2)) end


# my_jac = FWrap{Tuple{Vector{Float64}, Vector{Number}, Vector{Float64}}, SparseMatrixCSC{Float64,Int64}}(model_jacobian)

# my_jac(([1.0], Number[1.0], [.9]))

# typeof(my_jac)

# using ForwardDiff

# Type alias for revision history entries
const RevisionEntry = NamedTuple{(:timestamp, :action, :equation_index, :old_equation, :new_equation), Tuple{Dates.DateTime, Symbol, Union{Int, Nothing}, Union{Expr, Nothing}, Union{Expr, Nothing}}}

# =============================================================================
# STRUCTURES OVERVIEW
# =============================================================================
#
# The model struct (ℳ) contains several sub-structs organized by purpose:
#
# 1. CONSTANTS (𝓂.constants) - Model structure information that never changes
#    after the @model and @parameters macros are processed:
#    - post_model_macro: Variable/shock counts, indices, timing info
#    - post_parameters_macro: Parameter configuration from @parameters
#    - post_complete_parameters: Computed indices, display names, etc.
#    - second_order_indices: Perturbation auxiliary matrices and computational constants
#    - third_order_indices: Third order perturbation auxiliary matrices and computational constants
#
# 2. WORKSPACES (𝓂.workspaces) - Pre-allocated temporary buffers that are 
#    reused across function calls to avoid repeated allocations:
#    - qme: Quadratic matrix equation solver workspace
#    - sylvester_*: Sylvester equation solver workspaces  
#    - lyapunov_*: Lyapunov equation solver workspaces
#    - second_order/third_order: Higher order perturbation workspaces
#    - find_shocks: Conditional forecast shock finding workspace
#    - inversion: Inversion filter workspace
#    - kalman: Kalman filter workspace
#
# 3. CACHES (𝓂.caches) - Stored computation results that can be reused:
#    - non_stochastic_steady_state: NSSS solution values
#    - jacobian/hessian/third_order_derivatives: Perturbation derivatives
#    - first_order_solution_matrix/second_order_solution/etc.: Solved policy matrices
#    - outdated: Flags indicating which caches need recomputation
#
# 4. FUNCTIONS (𝓂.functions) - Compiled model functions:
#    - NSSS_solve/check: Steady state solvers
#    - jacobian/hessian/third_order_derivatives: Derivative functions
#    - state_update functions: Policy function evaluators
#
# Data Flow:
#   @model macro → post_model_macro (constants)
#   @parameters macro → post_parameters_macro, post_complete_parameters (constants)
#   solve!() → populates caches using workspaces, guided by constants
#   get_irf/simulate/etc → reads from caches, may trigger solve!() if outdated
#
# =============================================================================


mutable struct equations
    original::Vector{Expr}
    dynamic::Vector{Expr}
    steady_state::Vector{Expr}
    steady_state_aux::Vector{Expr}
    obc_violation::Vector{Expr}
    calibration::Vector{Expr}
    calibration_no_var::Vector{Expr}
    calibration_parameters::Vector{Symbol}
    calibration_original::Vector{Expr}
end

struct post_model_macro
    max_obc_horizon::Int
    # present_only::Vector{Symbol}
    # future_not_past::Vector{Symbol}
    # past_not_future::Vector{Symbol}
    # mixed::Vector{Symbol}
    future_not_past_and_mixed::Vector{Symbol}
    past_not_future_and_mixed::Vector{Symbol}
    # present_but_not_only::Vector{Symbol}
    # mixed_in_past::Vector{Symbol}
    # not_mixed_in_past::Vector{Symbol}
    # mixed_in_future::Vector{Symbol}

    var::Vector{Symbol}

    parameters_in_equations::Vector{Symbol}

    exo::Vector{Symbol}
    exo_past::Vector{Symbol}
    exo_present::Vector{Symbol}
    exo_future::Vector{Symbol}

    aux::Vector{Symbol}
    aux_present::Vector{Symbol}
    aux_future::Vector{Symbol}
    aux_past::Vector{Symbol}

    ➕_vars::Vector{Symbol}
    
    nPresent_only::Int
    nMixed::Int
    nFuture_not_past_and_mixed::Int
    nPast_not_future_and_mixed::Int
    # nPresent_but_not_only::Int
    nVars::Int
    nExo::Int
    present_only_idx::Vector{Int}
    present_but_not_only_idx::Vector{Int}
    future_not_past_and_mixed_idx::Vector{Int}
    not_mixed_in_past_idx::Vector{Int}
    past_not_future_and_mixed_idx::Vector{Int}
    mixed_in_past_idx::Vector{Int}
    mixed_in_future_idx::Vector{Int}
    past_not_future_idx::Vector{Int}
    reorder::Vector{Int}
    dynamic_order::Vector{Int}
    vars_in_ss_equations::Vector{Symbol}
    vars_in_ss_equations_no_aux::Vector{Symbol}

    dyn_var_future_list::Vector{Set{Symbol}}
    dyn_var_present_list::Vector{Set{Symbol}}
    dyn_var_past_list::Vector{Set{Symbol}}
    dyn_ss_list::Vector{Set{Symbol}}
    dyn_exo_list::Vector{Set{Symbol}}

    dyn_future_list::Vector{Set{Symbol}}
    dyn_present_list::Vector{Set{Symbol}}
    dyn_past_list::Vector{Set{Symbol}}

    var_list_aux_SS::Vector{Set{Symbol}}
    ss_list_aux_SS::Vector{Set{Symbol}}
    par_list_aux_SS::Vector{Set{Symbol}}
    var_future_list_aux_SS::Vector{Set{Symbol}}
    var_present_list_aux_SS::Vector{Set{Symbol}}
    var_past_list_aux_SS::Vector{Set{Symbol}}
    ss_equations_with_aux_variables::Vector{Int}
end

struct symbolics
    ss_equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    # dyn_equations_future::Vector{SPyPyC.Sym{PythonCall.Core.Py}}

    # dyn_shift_var_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_shift_var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_shift_var_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    # dyn_shift2_var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    # dyn_var_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_var_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_ss_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_exo_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    # dyn_exo_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_exo_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_exo_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}} 

    # dyn_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    var_present_list_aux_SS::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_past_list_aux_SS::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_future_list_aux_SS::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    ss_list_aux_SS::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_list_aux_SS::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dynamic_variables_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dynamic_variables_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    par_list_aux_SS::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    calibration_equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    calibration_equations_parameters::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    # parameters::Vector{SPyPyC.Sym{PythonCall.Core.Py}}

    # var_present::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    # var_past::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    # var_future::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    vars_in_ss_equations::Set{SPyPyC.Sym{PythonCall.Core.Py}}

    ss_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    par_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    var_redundant_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # var_redundant_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # var_solved_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # var_solved_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
end

struct moments_substate_indices
    I_plus_s_s::SparseMatrixCSC{Float64, Int}
    e_es::SparseMatrixCSC{Float64, Int}
    e_ss::SparseMatrixCSC{Float64, Int}
    ss_s::SparseMatrixCSC{Float64, Int}
    s_s::SparseMatrixCSC{Float64, Int}
end

struct moments_dependency_kron_indices
    kron_s_s::BitVector
    kron_s_e::BitVector
    kron_s_v::BitVector
end

"""
Second-order perturbation auxiliary matrices and index caches.

These are computed once when the model structure is known and reused across solutions.
Contains three categories of data:

1. **Auxiliary matrices** (𝛔, 𝐂₂, 𝐔₂, 𝐔∇₂): Sparse integer matrices for second-order
   perturbation solution. Populated by `create_second_order_auxiliary_matrices` during
   `write_functions_mapping!`.

2. **Computational constants** (s_in_s⁺, kron_*, etc.): BitVectors and index arrays indicating
   which elements are non-zero in Kronecker products and state vectors. Populated by
   `ensure_computational_constants!` on first use.

3. **Moment computation constants** (e4, vec_Iₑ, etc.): Pre-computed values for efficient
   moment calculations. Populated by `ensure_moments_constants!` on first use.
"""
mutable struct second_order_indices
    # =========================================================================
    # AUXILIARY MATRICES (for perturbation solution)
    # Filled by create_second_order_auxiliary_matrices (MacroModelling.jl)
    # Triggered by: write_functions_mapping! ← solve!
    # =========================================================================
    𝛔::SparseMatrixCSC{Int}              # Commutation matrix
    𝐂₂::SparseMatrixCSC{Int}             # Duplication matrix for 2nd order
    𝐔₂::SparseMatrixCSC{Int}             # Unique elements selector for 2nd order
    𝐔∇₂::SparseMatrixCSC{Int}            # Gradient unique elements selector

    # =========================================================================
    # COMPUTATIONAL CONSTANTS (for efficient sparse operations)
    # Filled by ensure_computational_constants! (options_and_caches.jl)
    # Triggered by: solve!, calculate_* (perturbation.jl), inversion filter
    # =========================================================================
    # State vector element membership (BitVectors for fast indexing)
    s_in_s⁺::BitVector                   # States in augmented state vector
    s_in_s::BitVector                    # States in state vector
    kron_s⁺_s⁺::BitVector                # Non-zeros in kron(s⁺, s⁺)
    kron_s⁺_s::BitVector                 # Non-zeros in kron(s⁺, s)
    e_in_s⁺::BitVector                   # Shocks in augmented state
    v_in_s⁺::BitVector                   # Volatility in augmented state
    kron_s_s::BitVector                  # Non-zeros in kron(s, s)
    kron_e_e::BitVector                  # Non-zeros in kron(e, e)
    kron_v_v::BitVector                  # Non-zeros in kron(v, v)
    kron_s_e::BitVector                  # Non-zeros in kron(s, e)
    kron_e_s::BitVector                  # Non-zeros in kron(e, s)
    kron_s⁺_s⁺_s⁺::BitVector             # Non-zeros in kron(s⁺, kron(s⁺, s⁺))
    kron_s_s⁺_s⁺::BitVector              # Non-zeros in kron(kron(s⁺, s⁺), s)
    # Index arrays for solution matrix slicing
    shockvar_idxs::Vector{Int}           # Shock-variable cross indices
    shock_idxs::Vector{Int}              # Shock indices in state
    shock_idxs2::Vector{Int}             # Shock² indices in kron state
    shock²_idxs::Vector{Int}             # Pure shock² product indices
    var_vol²_idxs::Vector{Int}           # Variable × volatility² indices

    # =========================================================================
    # CONDITIONAL FORECAST CONSTANTS
    # Filled by ensure_conditional_forecast_constants! (options_and_caches.jl)
    # Triggered by: get_conditional_forecast, find_shocks
    # =========================================================================
    var²_idxs::Vector{Int}               # Variable² indices
    shockvar²_idxs::Vector{Int}          # Shock × variable² indices

    # =========================================================================
    # MOMENT COMPUTATION CONSTANTS (model-constant values for moments.jl)
    # Filled by ensure_moments_constants! (options_and_caches.jl)
    # Triggered by: calculate_mean, calculate_*_order_moments*
    # =========================================================================
    kron_states::BitVector               # Kronecker product state mask
    I_plus_s_s::SparseMatrixCSC{Float64, Int}  # I + kron(s,s) selector
    e4::Vector{Float64}                  # 4th moment of standard normal (E[ε⁴])
    # Cached derived values from e4 and nExo (computed once, reused)
    vec_Iₑ::Vector{Float64}              # vec(I(nExo)) for moment calculations
    e4_nᵉ²_nᵉ²::Matrix{Float64}          # reshape(e4, nᵉ², nᵉ²)
    e4_nᵉ_nᵉ³::Matrix{Float64}           # reshape(e4, nᵉ, nᵉ³)
    e4_minus_vecIₑ_outer::Matrix{Float64} # e4_nᵉ²_nᵉ² - vec_Iₑ * vec_Iₑ'
end

"""
Third-order perturbation auxiliary matrices and computational constants.

These are computed once when the model structure is known and reused across solutions.
Contains three categories of data:

1. **Auxiliary matrices** (𝐂₃, 𝐔₃, 𝐏*, etc.): Sparse integer matrices for third-order
   perturbation solution. Populated by `create_third_order_auxiliary_matrices` during
   `write_functions_mapping!`.

2. **Computational constants** (var_vol³_idxs, shock_idxs*, etc.): Index arrays for slicing
   solution matrices. Populated by `ensure_conditional_forecast_constants!`.

3. **Moment computation constants** (e6, substate_indices, etc.): Pre-computed values
   for efficient third-order moment calculations. Populated by `ensure_moments_constants!`.
"""
mutable struct third_order_indices
    # =========================================================================
    # AUXILIARY MATRICES (for perturbation solution)
    # Filled by create_third_order_auxiliary_matrices (MacroModelling.jl)
    # Triggered by: write_functions_mapping! ← solve!
    # =========================================================================
    𝐂₃::SparseMatrixCSC{Int}             # Duplication matrix for 3rd order
    𝐔₃::SparseMatrixCSC{Int}             # Unique elements selector for 3rd order
    𝐈₃::Dict{Vector{Int}, Int}           # Index mapping for 3rd order terms
    𝐂∇₃::SparseMatrixCSC{Int}            # Gradient duplication matrix
    𝐔∇₃::SparseMatrixCSC{Int}            # Gradient unique selector
    𝐏::SparseMatrixCSC{Int}              # Permutation matrix
    𝐏₁ₗ::SparseMatrixCSC{Int}            # Left permutation 1
    𝐏₁ᵣ::SparseMatrixCSC{Int}            # Right permutation 1
    𝐏₁ₗ̂::SparseMatrixCSC{Int}            # Modified left permutation 1
    𝐏₂ₗ̂::SparseMatrixCSC{Int}            # Modified left permutation 2
    𝐏₁ₗ̄::SparseMatrixCSC{Int}            # Alternative left permutation 1
    𝐏₂ₗ̄::SparseMatrixCSC{Int}            # Alternative left permutation 2
    𝐏₁ᵣ̃::SparseMatrixCSC{Int}            # Alternative right permutation 1
    𝐏₂ᵣ̃::SparseMatrixCSC{Int}            # Alternative right permutation 2
    𝐒𝐏::SparseMatrixCSC{Int}             # Combined selection-permutation

    # =========================================================================
    # CONDITIONAL FORECAST CONSTANTS
    # Filled by ensure_conditional_forecast_constants! (options_and_caches.jl)
    # Triggered by: get_conditional_forecast, find_shocks
    # =========================================================================
    var_vol³_idxs::Vector{Int}           # Variable × volatility³ indices
    shock_idxs2::Vector{Int}             # Shock² indices (3rd order context)
    shock_idxs3::Vector{Int}             # Shock³ indices
    shock³_idxs::Vector{Int}             # Pure shock³ product indices
    shockvar1_idxs::Vector{Int}          # Shock × var indices (position 1)
    shockvar2_idxs::Vector{Int}          # Shock × var indices (position 2)
    shockvar3_idxs::Vector{Int}          # Shock × var indices (position 3)
    shockvar³2_idxs::Vector{Int}         # Shock × var³ indices (2nd variant)
    shockvar³_idxs::Vector{Int}          # Shock × var³ indices

    # =========================================================================
    # MOMENT COMPUTATION CONSTANTS
    # Filled by ensure_moments_constants! (options_and_caches.jl)
    # Triggered by: calculate_*_order_moments*
    # =========================================================================
    e6::Vector{Float64}                  # 6th moment of standard normal (E[ε⁶])
    kron_e_v::BitVector                  # Kronecker shock × volatility mask
    # Cached derived value from e6 and nExo (computed once, reused)
    e6_nᵉ³_nᵉ³::Matrix{Float64}          # reshape(e6, nᵉ³, nᵉ³)

    # =========================================================================
    # MOMENT SUBSTATE CONSTANTS (Dict constants for autocorrelation)
    # Filled by ensure_moments_substate_indices! (options_and_caches.jl)
    # Triggered by: calculate_third_order_moments_with_autocorrelation
    # =========================================================================
    substate_indices::Dict{Int, moments_substate_indices}
    dependency_kron_indices::Dict{Tuple{Vararg{Symbol}}, moments_dependency_kron_indices}
end


"""
Pre-allocated workspace vectors and buffers for nonlinear solvers (Levenberg-Marquardt and Newton).
All vectors are the same size as the problem dimension (number of unknowns).

Used by `levenberg_marquardt` and `newton` in nonlinear_solver.jl.
Avoids per-call allocations for temporary arrays.
"""
mutable struct nonlinear_solver_workspace{T <: Real}
    # Function and Jacobian evaluation buffers
    func_buffer::Vector{T}
    jac_buffer::AbstractMatrix{T}
    chol_buffer::𝒮.LinearCache
    lu_buffer::𝒮.LinearCache
    
    # Guess vectors
    current_guess::Vector{T}
    previous_guess::Vector{T}
    guess_update::Vector{T}
    
    # Untransformed guess vectors (for coordinate transformation)
    current_guess_untransformed::Vector{T}
    previous_guess_untransformed::Vector{T}
    
    # Output vectors
    best_previous_guess::Vector{T}
    best_current_guess::Vector{T}
    
    # Multipurpose factor vector (transformation jacobian diagonal / temp storage)
    factor::Vector{T}
    
    # Transformed bounds (for Levenberg-Marquardt with coordinate transformation)
    u_bounds::Vector{T}
    l_bounds::Vector{T}
end


mutable struct function_and_jacobian{T <: Real}
    func::Function
    jac::Function
    workspace::nonlinear_solver_workspace{T}
end


mutable struct krylov_workspace{G <: AbstractFloat}
    gmres::GmresWorkspace{G,G,Vector{G}}
    dqgmres::DqgmresWorkspace{G,G,Vector{G}}
    bicgstab::BicgstabWorkspace{G,G,Vector{G}}
end

"""
Workspace for the Sylvester equation solver (A * X * B + C = X).

All buffer fields are initialized to 0-dimensional objects and lazily resized on first use.
Sylvester has two independent dimensions: n (rows of A/C) and m (cols of B/C).
"""
mutable struct sylvester_workspace{G <: AbstractFloat, H <: Real}
    # Dimensions (stored for reallocation checks)
    n::Int  # rows of A, rows of C
    m::Int  # cols of B, cols of C
    
    # Krylov method buffers (lazily allocated, n×m)
    tmp::Matrix{G}
    𝐗::Matrix{G}
    𝐂::Matrix{G}
    
    # Doubling algorithm working matrices (lazily allocated)
    𝐀::Matrix{G}      # n×n copy of A
    𝐀¹::Matrix{G}     # n×n for A²
    𝐁::Matrix{G}      # m×m copy of B
    𝐁¹::Matrix{G}     # m×m for B²
    𝐂_dbl::Matrix{G}  # n×m iteration buffer
    𝐂¹::Matrix{G}     # n×m iteration buffer
    𝐂B::Matrix{G}     # n×m temporary for C*B multiplication
    
    # Krylov solver state (lazily allocated)
    krylov_workspace::krylov_workspace{G}
    
    # ForwardDiff partials buffers (for forward-mode AD)
    P̃::Matrix{H}       # For sylvester equation partials
    Ã_fd::Matrix{H}    # Temporary for ForwardDiff partials of A
    B̃_fd::Matrix{H}    # Temporary for ForwardDiff partials of B
    C̃_fd::Matrix{H}    # Temporary for ForwardDiff partials of C
end


"""
Pre-allocated workspace matrices for the quadratic matrix equation doubling algorithm.
All matrices are square with dimension n = size(A,1) = size(B,1) = size(C,1).

Used by `solve_quadratic_matrix_equation` with `Val{:doubling}` in quadratic_matrix_equation.jl.
Also used by stochastic steady state calculations in `calculate_second_order_stochastic_steady_state`
and `calculate_third_order_stochastic_steady_state`.
Avoids per-call allocations for temporary matrices in the iterative doubling algorithm.

Fields:
- `E`, `F`: Working matrices for the doubling recurrence
- `X`, `Y`: Current iteration solution matrices
- `X_new`, `Y_new`, `E_new`, `F_new`: Next iteration matrices  
- `temp1`, `temp2`, `temp3`: Temporary matrices for intermediate computations
- `B̄`: Copy of B for LU factorization (modified in-place)
- `AXX`: Temporary for residual computation (A * X² + B * X + C)
- `I_n`: Pre-computed identity matrix for QME doubling (UniformScaling)
- `I_nPast`: Pre-computed identity matrix for stochastic steady state (UniformScaling)
"""
mutable struct qme_workspace{T <: Real, R <: Real}
    # Doubling algorithm working matrices
    E::Matrix{T}
    F::Matrix{T}
    X::Matrix{T}
    Y::Matrix{T}
    X_new::Matrix{T}
    Y_new::Matrix{T}
    E_new::Matrix{T}
    F_new::Matrix{T}
    
    # Temporary matrices for intermediate operations
    temp1::Matrix{T}
    temp2::Matrix{T}
    temp3::Matrix{T}
    
    # LU factorization buffer
    B̄::Matrix{T}
    
    # Residual computation buffer
    AXX::Matrix{T}
    
    # Sylvester workspace for ForwardDiff path
    sylvester_ws::sylvester_workspace{T, R}
    
    # ForwardDiff partials buffers (for forward-mode AD)
    X̃::Matrix{R}               # For QME solution partials
    X̃_first_order::Matrix{R}   # For first order solution partials
    p_tmp::Matrix{R}            # For calculate_first_order_solution
    ∂SS_and_pars::Matrix{R}     # For NSSS partials in get_NSSS_and_parameters
    
    # Pre-computed identity matrices (Diagonal{Bool} - supports indexing for schur algorithm)
    I_n::ℒ.Diagonal{Bool, Vector{Bool}}       # Identity for QME doubling (dimension n = nVars - nPresent_only)
    I_nPast::ℒ.Diagonal{Bool, Vector{Bool}}   # Identity for schur & stochastic steady state (dimension nPast_not_future_and_mixed)
end


"""
Pre-allocated workspace matrices for the Lyapunov equation solver.
Solves: A * X * A' + C = X using the doubling algorithm or Krylov methods.

Used by `solve_lyapunov_equation` in lyapunov.jl.
Avoids per-call allocations for temporary matrices.

Fields for doubling algorithm:
- `𝐂`: Current solution iterate
- `𝐂¹`: Next solution iterate  
- `𝐀`: Copy of A for iteration (will be squared)
- `𝐂A`: Temporary for C * A'
- `𝐀²`: Temporary for A * A

Fields for Krylov methods (bicgstab, gmres):
- `tmp̄`: Temporary matrix for linear operator
- `𝐗`: Reshape buffer for solution vector
- `b`: RHS vector for Krylov solver
- `krylov_solver`: Pre-allocated Krylov solver state

All buffer fields are initialized to 0-dimensional objects and lazily resized on first use.
"""
mutable struct lyapunov_workspace{T <: Real, R <: Real}
    # Dimension (stored for reallocation checks)
    n::Int
    
    # Doubling algorithm working matrices (lazily allocated)
    𝐂::Matrix{T}
    𝐂¹::Matrix{T}
    𝐀::Matrix{T}
    𝐂A::Matrix{T}
    𝐀²::Matrix{T}
    
    # Krylov method buffers (lazily allocated)
    tmp̄::Matrix{T}
    𝐗::Matrix{T}
    b::Vector{T}
    
    # Krylov solver state (lazily allocated, can be reused across calls)
    bicgstab_workspace::Krylov.BicgstabWorkspace{T, T, Vector{T}}
    gmres_workspace::Krylov.GmresWorkspace{T, T, Vector{T}}
    
    # ForwardDiff partials buffers (for forward-mode AD)
    P̃::Matrix{R}       # For lyapunov equation partials
    Ã_fd::Matrix{R}    # Temporary for ForwardDiff partials of A
    C̃_fd::Matrix{R}    # Temporary for ForwardDiff partials of C
end


struct ss_solve_block
    ss_problem::function_and_jacobian
    extended_ss_problem::function_and_jacobian
end

mutable struct non_stochastic_steady_state
    solve_blocks_in_place::Vector{ss_solve_block}
    dependencies::Any
end

"""
Tracks which cache elements are outdated and need recalculation.

When parameters change (via `𝓂.parameter_values = ...`), all fields are set to `true` (outdated).
When a cache is computed (e.g., by `solve!()`), its corresponding field is set to `false` (up to date).

This enables lazy evaluation: caches are only recomputed when actually needed AND outdated.
"""
mutable struct outdated_caches
    # Non-stochastic steady state
    non_stochastic_steady_state::Bool
    # Perturbation derivative buffers
    jacobian::Bool
    hessian::Bool
    third_order_derivatives::Bool
    # Perturbation solution buffers
    first_order_solution::Bool
    second_order_solution::Bool
    pruned_second_order_solution::Bool
    third_order_solution::Bool
    pruned_third_order_solution::Bool
end


"""
Stored computation results that can be reused across function calls.

Caches store the final outputs of expensive computations (steady state, perturbation solutions).
They are invalidated when parameters change (tracked by `outdated` flags) and recomputed
lazily when needed by get_* functions.

Purpose: Avoid recomputation when the same result is needed multiple times.

Fields:
- `outdated`: Flags indicating which caches need recomputation (see [`outdated_caches`](@ref))
- Perturbation derivatives (`jacobian`, `hessian`, `third_order_derivatives`): 
  Model derivative matrices evaluated at steady state
- Perturbation solutions (`first_order_solution_matrix`, `second_order_solution`, etc.):
  Policy function coefficient matrices
- `non_stochastic_steady_state`: NSSS solution values
- `solver_cache`: Recent solver guesses for warm-starting

Relationship to other structs:
- Caches are computed using `constants` (for dimensions/structure) and `workspaces` (for temporary buffers)
- Caches are read by get_* functions (get_irf, simulate, etc.)
- Caches are invalidated when `parameter_values` changes
"""
mutable struct caches
    # =========================================================================
    # CACHE INVALIDATION FLAGS
    # =========================================================================
    outdated::outdated_caches
    
    # =========================================================================
    # PERTURBATION DERIVATIVE CACHES
    # Computed by model derivative functions, used by perturbation solvers
    # =========================================================================
    jacobian::AbstractMatrix{<: Real}                      # ∇f at SS
    jacobian_parameters::AbstractMatrix{<: Real}           # ∂∇f/∂θ
    jacobian_SS_and_pars::AbstractMatrix{<: Real}          # ∂∇f/∂(SS,θ)
    hessian::AbstractMatrix{<: Real}                       # ∇²f at SS
    hessian_parameters::AbstractMatrix{<: Real}            # ∂∇²f/∂θ
    hessian_SS_and_pars::AbstractMatrix{<: Real}           # ∂∇²f/∂(SS,θ)
    third_order_derivatives::AbstractMatrix{<: Real}       # ∇³f at SS
    third_order_derivatives_parameters::AbstractMatrix{<: Real}  # ∂∇³f/∂θ
    third_order_derivatives_SS_and_pars::AbstractMatrix{<: Real} # ∂∇³f/∂(SS,θ)
    
    # =========================================================================
    # PERTURBATION SOLUTION CACHES
    # Policy function coefficient matrices (𝐒₁, 𝐒₂, 𝐒₃)
    # =========================================================================
    first_order_solution_matrix::Matrix{<: Real}           # 𝐒₁ - first order policy
    qme_solution::Matrix{<: Real}                          # Quadratic matrix eqn solution
    second_order_stochastic_steady_state::Vector{<: Real}  # E[x] deviation from NSSS (2nd)
    second_order_solution::AbstractMatrix{<: Real}         # 𝐒₂ - second order policy
    pruned_second_order_stochastic_steady_state::Vector{<: Real}  # Pruned 2nd order SSS
    third_order_stochastic_steady_state::Vector{<: Real}   # E[x] deviation (3rd)
    third_order_solution::AbstractMatrix{<: Real}          # 𝐒₃ - third order policy
    pruned_third_order_stochastic_steady_state::Vector{<: Real}   # Pruned 3rd order SSS
    
    # =========================================================================
    # STEADY STATE CACHES
    # =========================================================================
    non_stochastic_steady_state::Vector{<: Real}           # NSSS values
    solver_cache::CircularBuffer{Vector{Vector{Float64}}}  # Recent solver guesses
    ∂equations_∂parameters::AbstractMatrix{<: Real}        # SS sensitivity to params
    ∂equations_∂SS_and_pars::AbstractMatrix{<: Real}       # SS Jacobian
end


# Structs for perturbation derivative functions (used for AD)
struct jacobian_functions
    f::Function                     # The main jacobian function
    f_parameters::Function          # Derivative w.r.t. parameters
    f_SS_and_pars::Function         # Derivative w.r.t. steady state and parameters
end

struct hessian_functions
    f::Function                     # The main hessian function
    f_parameters::Function          # Derivative w.r.t. parameters
    f_SS_and_pars::Function         # Derivative w.r.t. steady state and parameters
end

struct third_order_derivatives_functions
    f::Function                     # The main third order derivatives function
    f_parameters::Function          # Derivative w.r.t. parameters
    f_SS_and_pars::Function         # Derivative w.r.t. steady state and parameters
end

mutable struct model_functions
    # NSSS-related functions
    NSSS_solve::Function
    NSSS_check::Function
    NSSS_custom::Union{Nothing, Function}
    NSSS_∂equations_∂parameters::Function
    NSSS_∂equations_∂SS_and_pars::Function
    # Perturbation derivative functions
    jacobian::jacobian_functions
    hessian::hessian_functions
    third_order_derivatives::third_order_derivatives_functions
    # State update functions for perturbation solutions
    first_order_state_update::Function
    first_order_state_update_obc::Function
    second_order_state_update::Function
    second_order_state_update_obc::Function
    pruned_second_order_state_update::Function
    pruned_second_order_state_update_obc::Function
    third_order_state_update::Function
    third_order_state_update_obc::Function
    pruned_third_order_state_update::Function
    pruned_third_order_state_update_obc::Function
    # OBC-related functions
    obc_violation::Function
    # Whether all functions have been written/compiled
    functions_written::Bool
end


"""
Workspace for find_shocks used in conditional forecasts.

Contains pre-allocated Kronecker product buffers that depend only on n_exo (number of shocks).
All buffers are initialized to 0-dimensional and lazily resized on first use via ensure_find_shocks_buffers!.

Used by find_shocks_conditional_forecast (filter/find_shocks.jl).
"""
mutable struct find_shocks_workspace{T <: Real}
    # Dimension (for reallocation checks)
    n_exo::Int
    
    # 2nd order buffers
    kron_buffer::Vector{T}   # n_exo^2 - for ℒ.kron(x, x)
    kron_buffer2::Matrix{T}  # n_exo^2 × n_exo - for ℒ.kron(J, x), where J = I(n_exo)
    
    # 3rd order buffers
    kron_buffer²::Vector{T}  # n_exo^3 - for ℒ.kron(x, kron_buffer)
    kron_buffer3::Matrix{T}  # n_exo^3 × n_exo - for ℒ.kron(J, kron_buffer)
    kron_buffer4::Matrix{T}  # n_exo^3 × n_exo^2 - for ℒ.kron(kron(J,J), x)
end


"""
Workspace for inversion filter computations.
Contains pre-allocated buffers for state-related kronecker products and state vectors.
Buffers are lazily allocated and resized as needed via ensure_inversion_buffers!.
"""
mutable struct inversion_workspace{T <: Real}
    # Dimensions (for reallocation checks)
    n_exo::Int
    n_past::Int
    
    # Shock-related kron buffers (2nd order)
    kron_buffer::Vector{T}           # n_exo^2
    kron_buffer2::Matrix{T}          # (n_exo^2, n_exo) - ℒ.kron(J, x) where J = I(n_exo)
    
    # Shock-related kron buffers (3rd order)
    kron_buffer²::Vector{T}          # n_exo^3
    kron_buffer3::Matrix{T}          # (n_exo^3, n_exo)
    kron_buffer4::Matrix{T}          # (n_exo^3, n_exo^2)
    
    # State-related kron buffers
    kron_buffer_state::Matrix{T}     # (n_exo * (n_past+1), n_exo) - ℒ.kron(J, state_vol) where J = I(n_exo)
    kronstate_vol::Vector{T}         # (n_past+1)^2 - ℒ.kron(state_vol, state_vol)
    kronaug_state::Vector{T}         # (n_past+1+n_exo)^2
    kron_kron_aug_state::Vector{T}   # (n_past+1+n_exo)^3 (3rd order)
    
    # State vectors
    state_vol::Vector{T}             # n_past+1
    aug_state₁::Vector{T}            # n_past+1+n_exo
    aug_state₂::Vector{T}            # n_past+1+n_exo
    
    # Pullback buffers (for reverse-mode AD in rrule)
    ∂_tmp1::Matrix{T}                # (n_exo, n_past + n_exo)
    ∂_tmp2::Matrix{T}                # (n_past, n_past + n_exo)
    ∂_tmp3::Vector{T}                # n_past + n_exo
    ∂𝐒t⁻::Matrix{T}                  # (n_past, n_past + n_exo)
    ∂data::Matrix{T}                 # (n_past, n_periods)
    # Pullback buffers for pruned second order inversion filter
    ∂𝐒ⁱ²ᵉtmp::Matrix{T}              # (n_exo, n_exo * n_obs)
    ∂𝐒ⁱ²ᵉtmp2::Matrix{T}             # (n_obs, n_exo^2)
    kronSλ::Vector{T}                # n_obs * n_exo
    kronxS::Vector{T}                # n_exo * n_obs
end


"""  
Workspace for Kalman filter computations.
Contains pre-allocated buffers for state estimates, covariances, and matrix operations.
Buffers are lazily allocated and resized as needed via ensure_kalman_buffers!.
"""
mutable struct kalman_workspace{T <: Real}
    # Dimensions (for reallocation checks)
    n_obs::Int
    n_states::Int
    
    # State and observation vectors
    u::Vector{T}             # n_states - state estimate
    z::Vector{T}             # n_obs - predicted observation
    ztmp::Vector{T}          # n_obs - temp for observation
    utmp::Vector{T}          # n_states - temp for state
    
    # Matrix buffers
    Ctmp::Matrix{T}          # (n_obs, n_states) - C*P buffer
    F::Matrix{T}             # (n_obs, n_obs) - innovation covariance
    K::Matrix{T}             # (n_states, n_obs) - Kalman gain
    tmp::Matrix{T}           # (n_states, n_states) - temp for P
    Ptmp::Matrix{T}          # (n_states, n_states) - temp for P
end


mutable struct higher_order_workspace{F <: Real, G <: AbstractFloat, H <: Real}
    tmpkron0::SparseMatrixCSC{F, Int}
    tmpkron1::SparseMatrixCSC{F, Int}
    tmpkron11::SparseMatrixCSC{F, Int}
    tmpkron12::SparseMatrixCSC{F, Int}
    tmpkron2::SparseMatrixCSC{F, Int}
    tmpkron22::SparseMatrixCSC{F, Int}
    tmp_sparse_prealloc1::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc2::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc3::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc4::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc5::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc6::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    Ŝ::Matrix{F}
    sylvester_workspace::sylvester_workspace{G, H}
    # Pullback gradient buffers (lazily allocated, used in rrule pullback functions)
    # Second order pullback buffers
    ∂∇₂::Matrix{F}
    ∂∇₁::Matrix{F}
    ∂𝐒₁::Matrix{F}
    ∂spinv::Matrix{F}
    ∂𝐒₁₋╱𝟏ₑ::Matrix{F}
    ∂𝐒₁₊╱𝟎::Matrix{F}
    ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋::Matrix{F}
    # Third order pullback buffers (only dense matrices)
    ∂∇₁_3rd::Matrix{F}  # separate from 2nd order since dimensions differ
    ∂𝐒₁_3rd::Matrix{F}  # separate from 2nd order since dimensions differ
    ∂spinv_3rd::Matrix{F}  # separate from 2nd order since dimensions differ
    # ForwardDiff partials buffers for stochastic steady state (accessed via model struct)
    ∂x_second_order::Matrix{H}     # For second order SSS partials
    ∂x_third_order::Matrix{H}      # For third order SSS partials
end


"""
Pre-allocated workspaces for temporary buffers across all computational routines.

Workspaces are reused across function calls to avoid repeated heap allocations.
They contain no persistent state - all values are overwritten on each use.
Most workspaces are lazily resized when dimensions change (via ensure_*_buffers! functions).

Purpose: Speed up computation by eliminating allocation overhead in hot loops.

Fields:
- `second_order/third_order`: Higher-order perturbation solution workspaces
- `custom_steady_state_buffer`: Buffer for custom steady state evaluation
- `qme`: Quadratic matrix equation solver workspace
- `lyapunov_*`: Lyapunov equation solver workspaces (1st, 2nd, 3rd order)
- `sylvester_*`: Sylvester equation solver workspace
- `find_shocks`: Conditional forecast shock finding workspace
- `inversion`: Inversion filter workspace
- `kalman`: Kalman filter workspace

Relationship to other structs:
- Workspaces are sized based on values from `constants` (variable counts, etc.)
- Workspaces are used when computing values stored in `caches`
- Workspaces do NOT store results - those go in `caches`
"""
mutable struct workspaces
    # Higher-order perturbation solution workspaces
    second_order::higher_order_workspace        # Kronecker products, sparse preallocs
    third_order::higher_order_workspace         # Separate workspace for 3rd order
    # Steady state buffer
    custom_steady_state_buffer::Vector{Float64} # For custom SS function evaluation
    # Matrix equation solver workspaces
    qme::qme_workspace{Float64, Float64}                 # Quadratic matrix equation (1st order)
    lyapunov_1st_order::lyapunov_workspace{Float64, Float64}  # Covariance (1st order moments)
    lyapunov_2nd_order::lyapunov_workspace{Float64, Float64}  # Covariance (2nd order moments)
    lyapunov_3rd_order::lyapunov_workspace{Float64, Float64}  # Covariance (3rd order moments)
    sylvester_1st_order::sylvester_workspace{Float64, Float64} # Sylvester equation
    # Filter workspaces
    find_shocks::find_shocks_workspace{Float64}  # Conditional forecast shock finding
    inversion::inversion_workspace{Float64}      # Inversion filter
    kalman::kalman_workspace{Float64}            # Kalman filter
end



struct post_parameters_macro
    parameters_as_function_of_parameters::Vector{Symbol}
    precompile::Bool
    simplify::Bool
    symbolic::Bool
    guess::Dict{Symbol, Float64}
    ss_calib_list::Vector{Set{Symbol}}
    par_calib_list::Vector{Set{Symbol}}
    # ss_no_var_calib_list::Vector{Set{Symbol}}
    # par_no_var_calib_list::Vector{Set{Symbol}}
    bounds::Dict{Symbol,Tuple{Float64,Float64}}
    ss_solver_parameters_algorithm::Symbol
    ss_solver_parameters_maxtime::Float64
end

struct post_complete_parameters{S <: Union{Symbol, String}}
    parameters::Vector{Symbol}
    missing_parameters::Vector{Symbol}
    dyn_var_future_idx::Vector{Int}
    dyn_var_present_idx::Vector{Int}
    dyn_var_past_idx::Vector{Int}
    dyn_ss_idx::Vector{Int}
    # shocks_ss::Vector{Int}
    diag_nVars::ℒ.Diagonal{Bool, Vector{Bool}}
    var_axis::Vector{S}
    calib_axis::Vector{S}
    exo_axis_plain::Vector{S}
    exo_axis_with_subscript::Vector{S}
    # var_has_curly::Bool
    # exo_has_curly::Bool
    SS_and_pars_names::Vector{Symbol}
    # all_variables::Vector{Symbol}
    # NSSS_labels::Vector{Symbol}
    # aux_indices::Vector{Int}
    # processed_all_variables::Vector{Symbol}
    full_NSSS_display::Vector{S}
    steady_state_expand_matrix::SparseMatrixCSC{Float64, Int}
    custom_ss_expand_matrix::SparseMatrixCSC{Float64, Int}
    vars_in_ss_equations::Vector{Symbol}
    vars_in_ss_equations_with_aux::Vector{Symbol}
    SS_and_pars_names_lead_lag::Vector{Symbol}
    # SS_and_pars_names_no_exo::Vector{Symbol}
    SS_and_pars_no_exo_idx::Vector{Int}
    vars_idx_excluding_aux_obc::Vector{Int}
    vars_idx_excluding_obc::Vector{Int}
    initialized::Bool
    dyn_index::UnitRange{Int}
    reverse_dynamic_order::Vector{Int}
    comb::Vector{Int}
    future_not_past_and_mixed_in_comb::Vector{Int}
    past_not_future_and_mixed_in_comb::Vector{Int}
    Ir::ℒ.Diagonal{Bool, Vector{Bool}}
    nabla_zero_cols::UnitRange{Int}
    nabla_minus_cols::UnitRange{Int}
    nabla_e_start::Int
    expand_future::Matrix{Bool}
    expand_past::Matrix{Bool}
end

"""
Model structure constants that are fixed once @model and @parameters macros are processed.

These values describe the model's structure (variable counts, indices, auxiliary matrices)
and never change after model definition. They are computed once and reused across all
subsequent operations (solving, simulation, estimation).

Fields:
- `post_model_macro`: Variable/shock counts, indices, timing info from @model
- `post_parameters_macro`: Parameter configuration from @parameters
- `post_complete_parameters`: Computed indices, display axes, expansion matrices
- `second_order`: Auxiliary matrices and computational constants for 2nd order perturbation
- `third_order`: Auxiliary matrices and computational constants for 3rd order perturbation

Relationship to other structs:
- Constants are populated during macro expansion and early solve!() calls
- Workspaces use constants to determine buffer sizes
- Caches store results computed using constants
"""
mutable struct constants#{F <: Real, G <: AbstractFloat}
    # Model structure from @model macro (variable counts, indices, timing)
    post_model_macro::post_model_macro
    # Parameter configuration from @parameters macro
    post_parameters_macro::post_parameters_macro
    # Derived indices and display info (computed after both macros)
    post_complete_parameters::post_complete_parameters
    # Second-order perturbation auxiliary matrices and indices
    second_order::second_order_indices
    # Third-order perturbation auxiliary matrices and indices
    third_order::third_order_indices
end

mutable struct solver_parameters
    # xtol::Float64 
    # ftol::Float64 
    # rel_xtol::Float64 
    # iterations::Int
    ϕ̄::Float64    
    ϕ̂::Float64    
    μ̄¹::Float64   
    μ̄²::Float64   
    p̄¹::Float64   
    p̄²::Float64   
    ρ::Float64    
    ρ¹::Float64   
    ρ²::Float64   
    ρ³::Float64   
    ν::Float64    
    λ¹::Float64   
    λ²::Float64   
    λ̂¹::Float64   
    λ̂²::Float64   
    λ̅¹::Float64   
    λ̅²::Float64   
    λ̂̅¹::Float64   
    λ̂̅²::Float64   
    starting_value::Float64
    transformation_level::Int
    shift::Float64 
    backtracking_order::Int
end

"""
Counters for steady state and perturbation solves.

Each counter tracks total attempts and failed attempts, with separate tallies
for estimation routines. Perturbation counters are split by order (1, 2, 3).
"""
mutable struct SolveCounters
    # Steady state solve counters (total = all attempts, failed = only failures)
    ss_solves_total::Int
    ss_solves_failed::Int
    ss_solves_total_estimation::Int
    ss_solves_failed_estimation::Int
    
    # First order perturbation solve counters
    first_order_solves_total::Int
    first_order_solves_failed::Int
    first_order_solves_total_estimation::Int
    first_order_solves_failed_estimation::Int
    
    # Second order perturbation solve counters
    second_order_solves_total::Int
    second_order_solves_failed::Int
    second_order_solves_total_estimation::Int
    second_order_solves_failed_estimation::Int
    
    # Third order perturbation solve counters
    third_order_solves_total::Int
    third_order_solves_failed::Int
    third_order_solves_total_estimation::Int
    third_order_solves_failed_estimation::Int
end


"""
The main model struct containing all model data, organized into four categories:

1. **User-facing data**:
   - `model_name`: Name of the model (from @model macro)
   - `parameter_values`: Current parameter values (can be modified by user)
   - `equations`: Original and transformed model equations

2. **Constants** (`constants`): Model structure information that never changes
   - Variable/shock counts, indices, timing info
   - Perturbation auxiliary matrices
   - See [`constants`](@ref) for details

3. **Workspaces** (`workspaces`): Pre-allocated temporary buffers
   - Reused across function calls to avoid allocations
   - No persistent state - overwritten on each use
   - See [`workspaces`](@ref) for details

4. **Caches** (`caches`): Stored computation results
   - Perturbation solutions, steady state values
   - Invalidated when parameters change
   - See [`caches`](@ref) for details

5. **Functions** (`functions`): Compiled model functions
   - Steady state solvers, derivative evaluators
   - Policy function evaluators

Typical usage:
```julia
@model MyModel begin ... end
@parameters MyModel begin ... end
# MyModel is now a fully initialized ℳ struct
get_irf(MyModel)  # Uses constants + workspaces → caches → returns IRF
```
"""
mutable struct ℳ
    # =========================================================================
    # USER-FACING MODEL DATA
    # =========================================================================
    model_name::Any                           # Model identifier
    parameter_values::Vector{Float64}         # Current parameter values (mutable)

    # =========================================================================
    # STEADY STATE SOLVER INFRASTRUCTURE
    # =========================================================================
    NSSS::non_stochastic_steady_state         # Steady state solver blocks

    # =========================================================================
    # MODEL EQUATIONS (various representations)
    # =========================================================================
    equations::equations                      # Original, dynamic, and SS equations

    # =========================================================================
    # COMPUTATION INFRASTRUCTURE (see struct documentation for details)
    # =========================================================================
    caches::caches                            # Stored results (invalidated on param change)
    constants::constants                      # Model structure (never changes after init)
    workspaces::workspaces                    # Temporary buffers (reused, no state)
    functions::model_functions                # Compiled model functions

    counters::SolveCounters                   # Solve counters (steady state and perturbation)

    revision_history::Vector{RevisionEntry}   # History of equation modifications
end


# =============================================================================
# STRUCT CONSTRUCTORS
# =============================================================================
# Constructors for all structs defined above. Organized by category:
# 1. Index structs (second_order_indices, third_order_indices)
# 2. Workspace structs (nonlinear_solver_workspace, krylov_workspace, etc.)
# 3. Cache structs (outdated_caches, caches)
# 4. Function structs (model_functions)
# 5. Top-level structs (workspaces, constants, SolveCounters)
# =============================================================================

"""
    Second_order_indices()

Create an empty `second_order_indices` struct with all fields initialized to empty/zero values.
These will be lazily populated by various ensure_*! functions as needed.

See [`second_order_indices`](@ref) for field documentation.
"""
function Second_order_indices()
    empty_sparse_int = SparseMatrixCSC{Int, Int64}(ℒ.I, 0, 0)
    empty_sparse_float = spzeros(Float64, 0, 0)
    empty_matrix_float = Matrix{Float64}(undef, 0, 0)
    return second_order_indices(
        # Auxiliary matrices (𝛔, 𝐂₂, 𝐔₂, 𝐔∇₂)
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        # Computational index caches (BitVectors)
        BitVector(),         # s_in_s⁺
        BitVector(),         # s_in_s
        BitVector(),         # kron_s⁺_s⁺
        BitVector(),         # kron_s⁺_s
        BitVector(),         # e_in_s⁺
        BitVector(),         # v_in_s⁺
        BitVector(),         # kron_s_s
        BitVector(),         # kron_e_e
        BitVector(),         # kron_v_v
        BitVector(),         # kron_s_e
        BitVector(),         # kron_e_s
        BitVector(),         # kron_s⁺_s⁺_s⁺
        BitVector(),         # kron_s_s⁺_s⁺
        # Index arrays
        Int[],               # shockvar_idxs
        Int[],               # shock_idxs
        Int[],               # shock_idxs2
        Int[],               # shock²_idxs
        Int[],               # var_vol²_idxs
        # Conditional forecast indices
        Int[],               # var²_idxs
        Int[],               # shockvar²_idxs
        # Moment computation caches
        BitVector(),         # kron_states
        empty_sparse_float,  # I_plus_s_s
        Float64[],           # e4
        Float64[],           # vec_Iₑ
        empty_matrix_float,  # e4_nᵉ²_nᵉ²
        empty_matrix_float,  # e4_nᵉ_nᵉ³
        empty_matrix_float,  # e4_minus_vecIₑ_outer
    )
end

"""
    Third_order_indices()

Create an empty `third_order_indices` struct with all fields initialized to empty/zero values.
These will be lazily populated by various ensure_*! functions as needed.

See [`third_order_indices`](@ref) for field documentation.
"""
function Third_order_indices()
    empty_sparse_int = SparseMatrixCSC{Int, Int64}(ℒ.I, 0, 0)
    empty_matrix_float = Matrix{Float64}(undef, 0, 0)
    return third_order_indices(
        # Auxiliary matrices (𝐂₃, 𝐔₃, 𝐈₃, 𝐂∇₃, 𝐔∇₃, 𝐏, 𝐏₁ₗ, 𝐏₁ᵣ, ...)
        empty_sparse_int,    # 𝐂₃
        empty_sparse_int,    # 𝐔₃
        Dict{Vector{Int}, Int}(),  # 𝐈₃
        empty_sparse_int,    # 𝐂∇₃
        empty_sparse_int,    # 𝐔∇₃
        empty_sparse_int,    # 𝐏
        empty_sparse_int,    # 𝐏₁ₗ
        empty_sparse_int,    # 𝐏₁ᵣ
        empty_sparse_int,    # 𝐏₁ₗ̂
        empty_sparse_int,    # 𝐏₂ₗ̂
        empty_sparse_int,    # 𝐏₁ₗ̄
        empty_sparse_int,    # 𝐏₂ₗ̄
        empty_sparse_int,    # 𝐏₁ᵣ̃
        empty_sparse_int,    # 𝐏₂ᵣ̃
        empty_sparse_int,    # 𝐒𝐏
        # Conditional forecast index caches
        Int[],               # var_vol³_idxs
        Int[],               # shock_idxs2
        Int[],               # shock_idxs3
        Int[],               # shock³_idxs
        Int[],               # shockvar1_idxs
        Int[],               # shockvar2_idxs
        Int[],               # shockvar3_idxs
        Int[],               # shockvar³2_idxs
        Int[],               # shockvar³_idxs
        # Moment computation caches
        Float64[],           # e6
        BitVector(),         # kron_e_v
        empty_matrix_float,  # e6_nᵉ³_nᵉ³
        # Substate index Dict caches
        Dict{Int, moments_substate_indices}(),
        Dict{Tuple{Vararg{Symbol}}, moments_dependency_kron_indices}(),
    )
end


"""
Create a workspace for nonlinear solvers (Levenberg-Marquardt and Newton).

# Arguments
- `func_buffer::Vector{T}`: Pre-allocated buffer for function evaluation
- `jac_buffer::AbstractMatrix{T}`: Pre-allocated buffer for Jacobian
- `chol_buffer::LinearCache`: Pre-allocated Cholesky factorization cache
- `lu_buffer::LinearCache`: Pre-allocated LU factorization cache
"""
function Nonlinear_solver_workspace(func_buffer::Vector{T}, jac_buffer::AbstractMatrix{T}, 
                                    chol_buffer::𝒮.LinearCache, lu_buffer::𝒮.LinearCache) where T <: Real
    n = length(func_buffer)
    nonlinear_solver_workspace(
        func_buffer,          # func_buffer
        jac_buffer,           # jac_buffer
        chol_buffer,          # chol_buffer
        lu_buffer,            # lu_buffer
        zeros(T, n),          # current_guess
        zeros(T, n),          # previous_guess
        zeros(T, n),          # guess_update
        zeros(T, n),          # current_guess_untransformed
        zeros(T, n),          # previous_guess_untransformed
        zeros(T, n),          # best_previous_guess
        zeros(T, n),          # best_current_guess
        zeros(T, n),          # factor
        zeros(T, n),          # u_bounds
        zeros(T, n),          # l_bounds
    )
end


function Krylov_workspace(;S::Type = Float64)
    krylov_workspace(  GmresWorkspace(0,0,Vector{S}),
                    DqgmresWorkspace(0,0,Vector{S}),
                    BicgstabWorkspace(0,0,Vector{S}))
end

function Sylvester_workspace(;S::Type = Float64, T::Type = Float64)
    sylvester_workspace(
        0, 0,                   # n, m dimensions
        zeros(S,0,0),           # tmp (Krylov)
        zeros(S,0,0),           # 𝐗 (Krylov)
        zeros(S,0,0),           # 𝐂 (Krylov)
        zeros(S,0,0),           # 𝐀 (doubling)
        zeros(S,0,0),           # 𝐀¹ (doubling)
        zeros(S,0,0),           # 𝐁 (doubling)
        zeros(S,0,0),           # 𝐁¹ (doubling)
        zeros(S,0,0),           # 𝐂_dbl (doubling)
        zeros(S,0,0),           # 𝐂¹ (doubling)
        zeros(S,0,0),           # 𝐂B (doubling)
        Krylov_workspace(S = S),
        # ForwardDiff partials buffers
        zeros(T,0,0),           # P̃
        zeros(T,0,0),           # Ã_fd
        zeros(T,0,0),           # B̃_fd
        zeros(T,0,0))           # C̃_fd
end

"""
    Find_shocks_workspace(;T::Type = Float64)

Create a workspace for find_shocks conditional forecast with lazy buffer allocation.
All buffers are initialized to 0-dimensional objects and resized on-demand via ensure_find_shocks_buffers!.
"""
function Find_shocks_workspace(;T::Type = Float64)
    find_shocks_workspace{T}(
        0,                      # n_exo dimension
        zeros(T,0),             # kron_buffer (n_exo^2)
        zeros(T,0,0),           # kron_buffer2 (n_exo × n_exo)
        zeros(T,0),             # kron_buffer² (n_exo^3)
        zeros(T,0,0),           # kron_buffer3 (n_exo × n_exo^2)
        zeros(T,0,0))           # kron_buffer4 (n_exo^2 × n_exo)
end

function Higher_order_workspace(;T::Type = Float64, S::Type = Float64)
    higher_order_workspace(spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        zeros(T,0,0),
                        Sylvester_workspace(S = S),
                        # Second order pullback gradient buffers (lazily allocated)
                        zeros(T,0,0),  # ∂∇₂
                        zeros(T,0,0),  # ∂∇₁
                        zeros(T,0,0),  # ∂𝐒₁
                        zeros(T,0,0),  # ∂spinv
                        zeros(T,0,0),  # ∂𝐒₁₋╱𝟏ₑ
                        zeros(T,0,0),  # ∂𝐒₁₊╱𝟎
                        zeros(T,0,0),  # ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋
                        # Third order pullback gradient buffers (only dense matrices)
                        zeros(T,0,0),  # ∂∇₁_3rd
                        zeros(T,0,0),  # ∂𝐒₁_3rd
                        zeros(T,0,0),  # ∂spinv_3rd
                        # ForwardDiff partials buffers for stochastic steady state (accessed via model struct)
                        zeros(S,0,0),  # ∂x_second_order
                        zeros(S,0,0))  # ∂x_third_order
end

"""
    Qme_workspace(n::Int; T::Type = Float64)

Create a pre-allocated workspace for the quadratic matrix equation doubling algorithm.
`n` is the dimension of the square matrices (nVars - nPresent_only).
"""
function Qme_workspace(n::Int; T::Type = Float64, S::Type = Float64, nPast::Int = 0)
    qme_workspace(  zeros(T, n, n),  # E
                    zeros(T, n, n),  # F
                    zeros(T, n, n),  # X
                    zeros(T, n, n),  # Y
                    zeros(T, n, n),  # X_new
                    zeros(T, n, n),  # Y_new
                    zeros(T, n, n),  # E_new
                    zeros(T, n, n),  # F_new
                    zeros(T, n, n),  # temp1
                    zeros(T, n, n),  # temp2
                    zeros(T, n, n),  # temp3
                    zeros(T, n, n),  # B̄
                    zeros(T, n, n),  # AXX
                    Sylvester_workspace(S = T),  # sylvester_ws
                    # ForwardDiff partials buffers
                    zeros(S, 0, 0),  # X̃
                    zeros(S, 0, 0),  # X̃_first_order
                    zeros(S, 0, 0),  # p_tmp
                    zeros(S, 0, 0),  # ∂SS_and_pars
                    # Pre-computed identity matrices (Diagonal{Bool} - supports indexing)
                    ℒ.I(n),             # I_n
                    ℒ.I(nPast))         # I_nPast
end

"""
    Lyapunov_workspace(n::Int; T::Type = Float64)

Create a workspace for the Lyapunov equation solver with lazy buffer allocation.
`n` is the dimension of the square matrices.
Buffers are initialized to 0-dimensional objects and resized on-demand when the corresponding algorithm is used.
"""
function Lyapunov_workspace(n::Int; T::Type = Float64)
    lyapunov_workspace{T, T}(
        n,                      # dimension
        zeros(T, 0, 0),         # 𝐂 (doubling)
        zeros(T, 0, 0),         # 𝐂¹ (doubling)
        zeros(T, 0, 0),         # 𝐀 (doubling)
        zeros(T, 0, 0),         # 𝐂A (doubling)
        zeros(T, 0, 0),         # 𝐀² (doubling)
        zeros(T, 0, 0),         # tmp̄ (Krylov)
        zeros(T, 0, 0),         # 𝐗 (Krylov)
        zeros(T, 0),            # b (Krylov)
        Krylov.BicgstabWorkspace(0, 0, Vector{T}),  # bicgstab_workspace
        Krylov.GmresWorkspace(0, 0, Vector{T}; memory = 20),  # gmres_workspace
        # ForwardDiff partials buffers
        zeros(T, 0, 0),         # P̃
        zeros(T, 0, 0),         # Ã_fd
        zeros(T, 0, 0)          # C̃_fd
    )
end

"""
    Inversion_workspace(;T::Type = Float64)

Create a workspace for inversion filter computations with lazy buffer allocation.
All buffers are initialized to 0-dimensional objects and resized on-demand via ensure_inversion_buffers!.
"""
function Inversion_workspace(;T::Type = Float64)
    inversion_workspace{T}(
        0, 0,                   # n_exo, n_past dimensions
        zeros(T, 0),            # kron_buffer (n_exo^2)
        zeros(T, 0, 0),         # kron_buffer2 (n_exo^2 × n_exo)
        zeros(T, 0),            # kron_buffer² (n_exo^3)
        zeros(T, 0, 0),         # kron_buffer3 (n_exo^3 × n_exo)
        zeros(T, 0, 0),         # kron_buffer4 (n_exo^3 × n_exo^2)
        zeros(T, 0, 0),         # kron_buffer_state (n_exo × n_past+1)
        zeros(T, 0),            # kronstate_vol ((n_past+1)^2)
        zeros(T, 0),            # kronaug_state ((n_past+1+n_exo)^2)
        zeros(T, 0),            # kron_kron_aug_state ((n_past+1+n_exo)^3)
        zeros(T, 0),            # state_vol (n_past+1)
        zeros(T, 0),            # aug_state₁ (n_past+1+n_exo)
        zeros(T, 0),            # aug_state₂ (n_past+1+n_exo)
        # Pullback buffers (for reverse-mode AD)
        zeros(T, 0, 0),         # ∂_tmp1 (n_exo × n_past+n_exo)
        zeros(T, 0, 0),         # ∂_tmp2 (n_past × n_past+n_exo)
        zeros(T, 0),            # ∂_tmp3 (n_past+n_exo)
        zeros(T, 0, 0),         # ∂𝐒t⁻ (n_past × n_past+n_exo)
        zeros(T, 0, 0),         # ∂data (n_past × n_periods)
        # Pullback buffers for pruned second order
        zeros(T, 0, 0),         # ∂𝐒ⁱ²ᵉtmp (n_exo × n_exo*n_obs)
        zeros(T, 0, 0),         # ∂𝐒ⁱ²ᵉtmp2 (n_obs × n_exo^2)
        zeros(T, 0),            # kronSλ (n_obs * n_exo)
        zeros(T, 0))            # kronxS (n_exo * n_obs)
end


"""
    Kalman_workspace(;T::Type = Float64)

Create a workspace for Kalman filter computations with lazy buffer allocation.
All buffers are initialized to 0-dimensional objects and resized on-demand via ensure_kalman_buffers!.
"""
function Kalman_workspace(;T::Type = Float64)
    kalman_workspace{T}(
        0, 0,                   # n_obs, n_states dimensions
        zeros(T, 0),            # u (n_states)
        zeros(T, 0),            # z (n_obs)
        zeros(T, 0),            # ztmp (n_obs)
        zeros(T, 0),            # utmp (n_states)
        zeros(T, 0, 0),         # Ctmp (n_obs × n_states)
        zeros(T, 0, 0),         # F (n_obs × n_obs)
        zeros(T, 0, 0),         # K (n_states × n_obs)
        zeros(T, 0, 0),         # tmp (n_states × n_states)
        zeros(T, 0, 0))         # Ptmp (n_states × n_states)
end


"""
    Functions()

Create a default `model_functions` instance with placeholder functions.
All function fields are initialized to identity or no-op functions,
and `functions_written` is set to `false`.
"""
Functions() = model_functions(
    (x->x),                                                    # NSSS_solve
    (x->x),                                                    # NSSS_check
    nothing,                                                   # NSSS_custom
    (x->x),                                                    # NSSS_∂equations_∂parameters
    (x->x),                                                    # NSSS_∂equations_∂SS_and_pars
    jacobian_functions(x->x, x->x, x->x),                      # jacobian
    hessian_functions(x->x, x->x, x->x),                       # hessian
    third_order_derivatives_functions(x->x, x->x, x->x),       # third_order_derivatives
    (x,y)->nothing,                                            # first_order_state_update
    (x,y)->nothing,                                            # first_order_state_update_obc
    (x,y)->nothing,                                            # second_order_state_update
    (x,y)->nothing,                                            # second_order_state_update_obc
    (x,y)->nothing,                                            # pruned_second_order_state_update
    (x,y)->nothing,                                            # pruned_second_order_state_update_obc
    (x,y)->nothing,                                            # third_order_state_update
    (x,y)->nothing,                                            # third_order_state_update_obc
    (x,y)->nothing,                                            # pruned_third_order_state_update
    (x,y)->nothing,                                            # pruned_third_order_state_update_obc
    x->x,                                                      # obc_violation
    false                                                      # functions_written
)


function Workspaces(;T::Type = Float64, S::Type = Float64)
    workspaces(Higher_order_workspace(T = T, S = S),
                Higher_order_workspace(T = T, S = S),
                Float64[],
                Qme_workspace(0, T = T),  # Initialize with size 0, will be resized when needed
                Lyapunov_workspace(0, T = T),  # 1st order - will be resized
                Lyapunov_workspace(0, T = T),  # 2nd order - will be resized
                Lyapunov_workspace(0, T = T),  # 3rd order - will be resized
                Sylvester_workspace(S = S),  # 1st order sylvester - will be resized
                Find_shocks_workspace(T = T),  # conditional forecast - will be resized
                Inversion_workspace(T = T),  # inversion filter - will be resized
                Kalman_workspace(T = T))  # Kalman filter - will be resized
end


"""
    outdated_caches()

Create an `outdated_caches` struct with all fields set to `true` (all caches outdated).
This is the default state when parameters change.

See [`outdated_caches`](@ref) for field documentation.
"""
outdated_caches() = outdated_caches(true, true, true, true, true, true, true, true, true)


"""
    Caches()

Create a default `caches` struct with empty arrays and all caches marked as outdated.
All cache buffers are initialized to 0-dimensional objects and will be resized as needed.

See [`caches`](@ref) for field documentation.
"""
Caches() = caches(
    outdated_caches(),
    zeros(0,0), zeros(0,0), zeros(0,0),  # jacobian, jacobian_parameters, jacobian_SS_and_pars
    zeros(0,0), zeros(0,0), zeros(0,0),  # hessian, hessian_parameters, hessian_SS_and_pars
    zeros(0,0), zeros(0,0), zeros(0,0),  # third_order_derivatives, third_order_derivatives_parameters, third_order_derivatives_SS_and_pars
    zeros(0,0), zeros(0,0),              # first_order_solution_matrix, qme_solution
    Float64[], SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), Float64[],  # 2nd order
    Float64[], SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), Float64[],  # 3rd order
    Float64[],                           # non_stochastic_steady_state
    CircularBuffer{Vector{Vector{Float64}}}(500),  # solver_cache
    zeros(0,0), zeros(0,0),              # ∂equations_∂parameters, ∂equations_∂SS_and_pars
)


"""
    SolveCounters()

Create a `SolveCounters` struct with all counters initialized to zero.

See [`SolveCounters`](@ref) for field documentation.
"""
SolveCounters() = SolveCounters(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


function Constants(model_struct; T::Type = Float64, S::Type = Float64)
    constants( model_struct,
            post_parameters_macro(
                Symbol[],
                false,
                true,
                false,
                Dict{Symbol, Float64}(),
                Set{Symbol}[],
                Set{Symbol}[],
                # Set{Symbol}[],
                # Set{Symbol}[],
                Dict{Symbol,Tuple{Float64,Float64}}(),
                :ESCH,
                120.0
                ),
            post_complete_parameters{Symbol}(
                Symbol[],
                Symbol[],
                Int[],
                Int[],
                Int[],
                Int[],
                # Int[],
                ℒ.I(0),
                Symbol[],
                Symbol[],
                Symbol[],
                Symbol[],
                # false,
                # false,
                Symbol[],
                # Symbol[],
                # Symbol[],
                # Int[],
                # Symbol[],
                Symbol[],
                spzeros(Float64, 0, 0),
                spzeros(Float64, 0, 0),
                Symbol[],
                Symbol[],
                Symbol[],
                # Symbol[],
                Int[],
                Int[],
                Int[],
                false,
                1:0,
                Int[],
                Int[],
                Int[],
                Int[],
                ℒ.I(0),
                1:0,
                1:0,
                1,
                zeros(Bool, 0, 0),
                zeros(Bool, 0, 0)),
            Second_order_indices(),
            Third_order_indices())
end
