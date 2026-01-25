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


mutable struct equations
    original::Vector{Expr}
    dynamic::Vector{Expr}
    steady_state::Vector{Expr}
    steady_state_aux::Vector{Expr}
    obc_violation::Vector{Expr}
    calibration::Vector{Expr}
    calibration_no_var::Vector{Expr}
    calibration_parameters::Vector{Symbol}
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

struct moments_substate_cache
    I_plus_s_s::SparseMatrixCSC{Float64, Int}
    e_es::SparseMatrixCSC{Float64, Int}
    e_ss::SparseMatrixCSC{Float64, Int}
    ss_s::SparseMatrixCSC{Float64, Int}
    s_s::SparseMatrixCSC{Float64, Int}
end

struct moments_dependency_kron_cache
    kron_s_s::BitVector
    kron_s_e::BitVector
    kron_s_v::BitVector
end

mutable struct second_order
    𝛔::SparseMatrixCSC{Int}
    𝐂₂::SparseMatrixCSC{Int}
    𝐔₂::SparseMatrixCSC{Int}
    𝐔∇₂::SparseMatrixCSC{Int}

    s_in_s⁺::BitVector
    s_in_s::BitVector
    kron_s⁺_s⁺::BitVector
    kron_s⁺_s::BitVector
    e_in_s⁺::BitVector
    v_in_s⁺::BitVector
    kron_s_s::BitVector
    kron_e_e::BitVector
    kron_v_v::BitVector
    kron_s_e::BitVector
    kron_e_s::BitVector
    shockvar_idxs::Vector{Int}
    shock_idxs::Vector{Int}
    shock_idxs2::Vector{Int}
    shock²_idxs::Vector{Int}
    var_vol²_idxs::Vector{Int}
    var²_idxs::Vector{Int}
    shockvar²_idxs::Vector{Int}

    kron_states::BitVector
    I_plus_s_s::SparseMatrixCSC{Float64, Int}
    e4::Vector{Float64}
end

mutable struct third_order
    𝐂₃::SparseMatrixCSC{Int}
    𝐔₃::SparseMatrixCSC{Int}
    𝐈₃::Dict{Vector{Int}, Int}

    𝐂∇₃::SparseMatrixCSC{Int}
    𝐔∇₃::SparseMatrixCSC{Int}

    𝐏::SparseMatrixCSC{Int}

    𝐏₁ₗ::SparseMatrixCSC{Int}
    𝐏₁ᵣ::SparseMatrixCSC{Int}

    𝐏₁ₗ̂::SparseMatrixCSC{Int}
    𝐏₂ₗ̂::SparseMatrixCSC{Int}

    𝐏₁ₗ̄::SparseMatrixCSC{Int}
    𝐏₂ₗ̄::SparseMatrixCSC{Int}

    𝐏₁ᵣ̃::SparseMatrixCSC{Int}
    𝐏₂ᵣ̃::SparseMatrixCSC{Int}

    𝐒𝐏::SparseMatrixCSC{Int}

    var_vol³_idxs::Vector{Int}
    shock_idxs2::Vector{Int}
    shock_idxs3::Vector{Int}
    shock³_idxs::Vector{Int}
    shockvar1_idxs::Vector{Int}
    shockvar2_idxs::Vector{Int}
    shockvar3_idxs::Vector{Int}
    shockvar³2_idxs::Vector{Int}
    shockvar³_idxs::Vector{Int}

    e6::Vector{Float64}
    kron_e_v::BitVector
    substate_cache::Dict{Int, moments_substate_cache}
    dependency_kron_cache::Dict{Tuple{Vararg{Symbol}}, moments_dependency_kron_cache}
end


mutable struct function_and_jacobian
    func::Function
    # func_aux::Function
    func_buffer::Vector{<:Real}
    # func_aux_buffer::Vector{<:Real}
    jac::Function
    jac_buffer::AbstractMatrix{<:Real}
    chol_buffer::𝒮.LinearCache
    lu_buffer::𝒮.LinearCache
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
When parameters change, all fields are set to `true` (outdated).
When a cache is computed, its corresponding field is set to `false` (up to date).
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

mutable struct caches
    # Perturbation derivative buffers
    jacobian::AbstractMatrix{<: Real}
    jacobian_parameters::AbstractMatrix{<: Real}
    jacobian_SS_and_pars::AbstractMatrix{<: Real}
    hessian::AbstractMatrix{<: Real}
    hessian_parameters::AbstractMatrix{<: Real}
    hessian_SS_and_pars::AbstractMatrix{<: Real}
    third_order_derivatives::AbstractMatrix{<: Real}
    third_order_derivatives_parameters::AbstractMatrix{<: Real}
    third_order_derivatives_SS_and_pars::AbstractMatrix{<: Real}
    # Perturbation solution buffers
    first_order_solution_matrix::Matrix{<: Real}
    qme_solution::Matrix{<: Real}
    second_order_stochastic_steady_state::Vector{<: Real}
    second_order_solution::AbstractMatrix{<: Real}
    pruned_second_order_stochastic_steady_state::Vector{<: Real}
    third_order_stochastic_steady_state::Vector{<: Real}
    third_order_solution::AbstractMatrix{<: Real}
    pruned_third_order_stochastic_steady_state::Vector{<: Real}
    # Non-stochastic steady state solution and solver caches
    non_stochastic_steady_state::Vector{<: Real}
    solver_cache::CircularBuffer{Vector{Vector{Float64}}}
    ∂equations_∂parameters::AbstractMatrix{<: Real}
    ∂equations_∂SS_and_pars::AbstractMatrix{<: Real}
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
end

mutable struct solution
    outdated::outdated_caches
    functions_written::Bool
end

mutable struct krylov_workspace{G <: AbstractFloat}
    gmres::GmresWorkspace{G,G,Vector{G}}
    dqgmres::DqgmresWorkspace{G,G,Vector{G}}
    bicgstab::BicgstabWorkspace{G,G,Vector{G}}
end

mutable struct sylvester_workspace{G <: AbstractFloat}
    tmp::Matrix{G}
    𝐗::Matrix{G}
    𝐂::Matrix{G}
    krylov_workspace::krylov_workspace{G}
end


mutable struct higher_order_workspace{F <: Real, G <: AbstractFloat}
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
    sylvester_workspace::sylvester_workspace{G}
end

mutable struct workspaces
    second_order::higher_order_workspace
    third_order::higher_order_workspace
    custom_steady_state_buffer::Vector{Float64}
end



struct post_parameters_macro
    parameters_as_function_of_parameters::Vector{Symbol}
    precompile::Bool
    simplify::Bool
    guess::Dict{Symbol, Float64}
    ss_calib_list::Vector{Set{Symbol}}
    par_calib_list::Vector{Set{Symbol}}
    # ss_no_var_calib_list::Vector{Set{Symbol}}
    # par_no_var_calib_list::Vector{Set{Symbol}}
    bounds::Dict{Symbol,Tuple{Float64,Float64}}
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

mutable struct constants#{F <: Real, G <: AbstractFloat}
    # Model structure information (constant for a given model after @model macro)
    post_model_macro::post_model_macro
    post_parameters_macro::post_parameters_macro
    post_complete_parameters::post_complete_parameters
    second_order::second_order
    third_order::third_order
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

mutable struct ℳ
    model_name::Any
    # SS_optimizer
    parameter_values::Vector{Float64}

    # ss
    # dynamic_variables::Vector{Symbol}
    # dyn_ss_past::Vector{Symbol}
    # dyn_ss_present::Vector{Symbol}
    # dyn_ss_future::Vector{Symbol}

    # var_present::Vector{Symbol}
    # var_future::Vector{Symbol}
    # var_past::Vector{Symbol}

    # exo_list::Vector{Set{Symbol}}
    # var_list::Vector{Set{Symbol}}
    # dynamic_variables_list::Vector{Set{Symbol}}
    # dynamic_variables_future_list::Vector{Set{Symbol}}


    # ss_list::Vector{Set{Symbol}}

    # var_solved_list
    # var_solved_calib_list
    # var_redundant_list
    # var_redundant_calib_list

    # par_list::Vector{Set{Symbol}}
    # var_future_list::Vector{Set{Symbol}}
    # var_present_list::Vector{Set{Symbol}}
    # var_past_list::Vector{Set{Symbol}}

    # dyn_shift_var_future_list::Vector{Set{Symbol}}
    # dyn_shift_var_present_list::Vector{Set{Symbol}}
    # dyn_shift_var_past_list::Vector{Set{Symbol}}

    # dyn_shift2_var_past_list::Vector{Set{Symbol}}


    # dyn_exo_future_list::Vector{Set{Symbol}}
    # dyn_exo_present_list::Vector{Set{Symbol}}
    # dyn_exo_past_list::Vector{Set{Symbol}} 

    # non_linear_solved_vars
    # non_linear_solved_vals
    # solved_sub_vals
    # solved_sub_values

    NSSS::non_stochastic_steady_state

    # ss_equations::Vector{Expr}
    # t_future_equations 
    # t_past_equations 
    # t_present_equations 
    # dyn_equations_future::Vector{Expr}

    equations::equations

    caches::caches

    # model_jacobian::Tuple{Vector{Function}, SparseMatrixCSC{Float64}}
    # model_jacobian::Tuple{Vector{Function}, Vector{Int}, Matrix{<: Real}}
    # # model_jacobian_parameters::Function
    # model_jacobian_SS_and_pars_vars::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}
    # # model_jacobian::FWrap{Tuple{Vector{Float64}, Vector{Number}, Vector{Float64}}, SparseMatrixCSC{Float64}}#{typeof(model_jacobian)}
    # model_hessian::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}
    # model_hessian_SS_and_pars_vars::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}
    # model_third_order_derivatives::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}
    # model_third_order_derivatives_SS_and_pars_vars::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}

    constants::constants
    workspaces::workspaces

    # obc_shock_bounds::Vector{Tuple{Symbol, Bool, Float64}}
    functions::model_functions

    solution::solution
    # symbolics::symbolics

    # estimation_helper::Dict{Vector{Symbol}, timings}
end
