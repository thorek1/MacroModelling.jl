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


struct timings
    present_only::Vector{Symbol}
    future_not_past::Vector{Symbol}
    past_not_future::Vector{Symbol}
    mixed::Vector{Symbol}
    future_not_past_and_mixed::Vector{Symbol}
    past_not_future_and_mixed::Vector{Symbol}
    present_but_not_only::Vector{Symbol}
    mixed_in_past::Vector{Symbol}
    not_mixed_in_past::Vector{Symbol}
    mixed_in_future::Vector{Symbol}
    exo::Vector{Symbol}
    var::Vector{Symbol}
    aux::Vector{Symbol}
    exo_present::Vector{Symbol}
    nPresent_only::Int
    nMixed::Int
    nFuture_not_past_and_mixed::Int
    nPast_not_future_and_mixed::Int
    nPresent_but_not_only::Int
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
end

struct symbolics
    ss_equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    dyn_equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    # dyn_equations_future::Vector{SPyPyC.Sym{PythonCall.Core.Py}}

    # dyn_shift_var_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_shift_var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_shift_var_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    # dyn_shift2_var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    dyn_var_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_var_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_ss_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_exo_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    # dyn_exo_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_exo_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dyn_exo_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}} 

    dyn_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    var_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    ss_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dynamic_variables_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # dynamic_variables_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    par_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    calibration_equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    calibration_equations_parameters::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    # parameters::Vector{SPyPyC.Sym{PythonCall.Core.Py}}

    # var_present::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    # var_past::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    # var_future::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    vars_in_ss_equations::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    var::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    ➕_vars::Set{SPyPyC.Sym{PythonCall.Core.Py}}

    ss_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    par_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    var_redundant_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # var_redundant_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # var_solved_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    # var_solved_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
end

struct auxilliary_indices
    dyn_var_future_idx::Vector{Int}
    dyn_var_present_idx::Vector{Int}
    dyn_var_past_idx::Vector{Int}
    dyn_ss_idx::Vector{Int}
    shocks_ss::Vector{Int}
end

struct second_order_auxilliary_matrices
    𝛔::SparseMatrixCSC{Int}
    𝐂₂::SparseMatrixCSC{Int}
    𝐔₂::SparseMatrixCSC{Int}
    𝐔∇₂::SparseMatrixCSC{Int}
end

struct third_order_auxilliary_matrices
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
end


struct perturbation_solution
    solution_matrix::Matrix{Float64}
    state_update::Function
    state_update_obc::Function
end

struct second_order_perturbation_solution
    # solution_matrix::SparseMatrixCSC{Float64}
    stochastic_steady_state::Vector{Float64}
    state_update::Function
    state_update_obc::Function
end

struct third_order_perturbation_solution
    # solution_matrix::SparseMatrixCSC{Float64}
    stochastic_steady_state::Vector{Float64}
    state_update::Function
    state_update_obc::Function
end


mutable struct perturbation
    first_order::perturbation_solution
    second_order::second_order_perturbation_solution
    pruned_second_order::second_order_perturbation_solution
    third_order::third_order_perturbation_solution
    pruned_third_order::third_order_perturbation_solution
    qme_solution::Matrix{Float64}
    second_order_solution::AbstractMatrix{Float64}
    third_order_solution::AbstractMatrix{Float64}
    auxilliary_indices::auxilliary_indices
    second_order_auxilliary_matrices::second_order_auxilliary_matrices
    third_order_auxilliary_matrices::third_order_auxilliary_matrices
end

mutable struct function_and_jacobian
    func::Function
    func_aux::Function
    func_buffer::Vector{<:Real}
    func_aux_buffer::Vector{<:Real}
    jac::Function
    jac_buffer::AbstractMatrix{<:Real}
end

struct ss_solve_block
    ss_problem::function_and_jacobian
    extended_ss_problem::function_and_jacobian
end

mutable struct solution
    perturbation::perturbation
    non_stochastic_steady_state::Vector{Float64}
    # algorithms::Set{Symbol}
    outdated_algorithms::Set{Symbol}
    outdated_NSSS::Bool
    functions_written::Bool
    # valid_steady_state_solution
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
    exo::Vector{Symbol}
    parameters_in_equations::Vector{Symbol}
    parameters_as_function_of_parameters::Vector{Symbol}
    parameters::Vector{Symbol}
    parameter_values::Vector{Float64}

    guess::Dict{Symbol, Float64}

    # ss
    # dynamic_variables::Vector{Symbol}
    # dyn_ss_past::Vector{Symbol}
    # dyn_ss_present::Vector{Symbol}
    # dyn_ss_future::Vector{Symbol}

    aux::Vector{Symbol}
    aux_present::Vector{Symbol}
    aux_future::Vector{Symbol}
    aux_past::Vector{Symbol}

    exo_future::Vector{Symbol}
    exo_present::Vector{Symbol}
    exo_past::Vector{Symbol}

    vars_in_ss_equations::Vector{Symbol}
    var::Vector{Symbol}
    # var_present::Vector{Symbol}
    # var_future::Vector{Symbol}
    # var_past::Vector{Symbol}

    # exo_list::Vector{Set{Symbol}}
    # var_list::Vector{Set{Symbol}}
    # dynamic_variables_list::Vector{Set{Symbol}}
    # dynamic_variables_future_list::Vector{Set{Symbol}}

    ss_calib_list::Vector{Set{Symbol}}
    par_calib_list::Vector{Set{Symbol}}

    ss_no_var_calib_list::Vector{Set{Symbol}}
    par_no_var_calib_list::Vector{Set{Symbol}}

    # ss_list::Vector{Set{Symbol}}

    ss_aux_equations::Vector{Expr}
    var_list_aux_SS::Vector{Set{Symbol}}
    ss_list_aux_SS::Vector{Set{Symbol}}
    par_list_aux_SS::Vector{Set{Symbol}}
    var_future_list_aux_SS::Vector{Set{Symbol}}
    var_present_list_aux_SS::Vector{Set{Symbol}}
    var_past_list_aux_SS::Vector{Set{Symbol}}

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

    dyn_var_future_list::Vector{Set{Symbol}}
    dyn_var_present_list::Vector{Set{Symbol}}
    dyn_var_past_list::Vector{Set{Symbol}}
    dyn_ss_list::Vector{Set{Symbol}}
    dyn_exo_list::Vector{Set{Symbol}}

    # dyn_exo_future_list::Vector{Set{Symbol}}
    # dyn_exo_present_list::Vector{Set{Symbol}}
    # dyn_exo_past_list::Vector{Set{Symbol}} 

    dyn_future_list::Vector{Set{Symbol}}
    dyn_present_list::Vector{Set{Symbol}}
    dyn_past_list::Vector{Set{Symbol}}

    solved_vars::Vector#{Union{Symbol,Vector{Symbol}}}
    solved_vals::Vector#{Union{Float64,Expr,Int,Vector{Union{Float64,Expr,Int}}}}
    # non_linear_solved_vars
    # non_linear_solved_vals
    # solved_sub_vals
    # solved_sub_values
    ss_solve_blocks::Vector#{RuntimeGeneratedFunction}
    ss_solve_blocks_in_place::Vector{ss_solve_block}
    # Vector{Tuple{
    #     Tuple{
    #         Tuple{Vector{Float64}, RuntimeGeneratedFunctions.RuntimeGeneratedFunction}, 
    #         Tuple{AbstractMatrix{Float64}, RuntimeGeneratedFunctions.RuntimeGeneratedFunction}
    #         }, 
    #     Tuple{
    #         Tuple{Vector{Float64}, RuntimeGeneratedFunctions.RuntimeGeneratedFunction}, 
    #         Tuple{AbstractMatrix{Float64}, RuntimeGeneratedFunctions.RuntimeGeneratedFunction}
    #         }
    #     }
    # }
    # ss_solve_blocks_no_transform::Vector#{RuntimeGeneratedFunction}
    #ss_solve_blocks_optim::Vector#{RuntimeGeneratedFunction}
    # SS_init_guess::Vector{Real}
    NSSS_solver_cache::CircularBuffer{Vector{Vector{Float64}}}
    SS_solve_func::Function
    SS_calib_func::Function
    SS_check_func::Function
    ∂SS_equations_∂parameters::Tuple{AbstractMatrix{<: Real}, Function}
    ∂SS_equations_∂SS_and_pars::Tuple{AbstractMatrix{<: Real}, Function}
    # nonlinear_solution_helper
    SS_dependencies::Any

    ➕_vars::Vector{Symbol}
    # ss_equations::Vector{Expr}
    ss_equations_with_aux_variables::Vector{Int}
    # t_future_equations 
    # t_past_equations 
    # t_present_equations 
    dyn_equations::Vector{Expr}
    ss_equations::Vector{Expr}
    # dyn_equations_future::Vector{Expr}
    original_equations::Vector{Expr}

    calibration_equations_no_var::Vector{Expr}

    calibration_equations::Vector{Expr}
    calibration_equations_parameters::Vector{Symbol}

    bounds::Dict{Symbol,Tuple{Float64,Float64}}

    jacobian::Tuple{AbstractMatrix{<: Real},Function}
    jacobian_parameters::Tuple{AbstractMatrix{<: Real},Function}
    jacobian_SS_and_pars::Tuple{AbstractMatrix{<: Real},Function}
    hessian::Tuple{AbstractMatrix{<: Real},Function}
    hessian_parameters::Tuple{AbstractMatrix{<: Real},Function}
    hessian_SS_and_pars::Tuple{AbstractMatrix{<: Real},Function}
    third_order_derivatives::Tuple{AbstractMatrix{<: Real},Function}
    third_order_derivatives_parameters::Tuple{AbstractMatrix{<: Real},Function}
    third_order_derivatives_SS_and_pars::Tuple{AbstractMatrix{<: Real},Function}

    # model_jacobian::Tuple{Vector{Function}, SparseMatrixCSC{Float64}}
    model_jacobian::Tuple{Vector{Function}, Vector{Int}, Matrix{<: Real}}
    # model_jacobian_parameters::Function
    model_jacobian_SS_and_pars_vars::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}
    # model_jacobian::FWrap{Tuple{Vector{Float64}, Vector{Number}, Vector{Float64}}, SparseMatrixCSC{Float64}}#{typeof(model_jacobian)}
    model_hessian::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}
    model_hessian_SS_and_pars_vars::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}
    model_third_order_derivatives::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}
    model_third_order_derivatives_SS_and_pars_vars::Tuple{Vector{Function}, SparseMatrixCSC{<: Real}}

    timings::timings

    caches::caches

    obc_violation_equations::Vector{Expr}
    # obc_shock_bounds::Vector{Tuple{Symbol, Bool, Float64}}
    max_obc_horizon::Int
    obc_violation_function::Function

    solver_parameters::Vector{solver_parameters}

    solution::solution
    # symbolics::symbolics

    estimation_helper::Dict{Vector{Symbol}, timings}
end
