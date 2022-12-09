
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
    ss_equations::Vector{Sym}
    # dyn_equations::Vector{Sym}
    # dyn_equations_future::Vector{Sym}
    
    # dyn_shift_var_present_list::Vector{Set{Sym}}
    # dyn_shift_var_past_list::Vector{Set{Sym}}
    # dyn_shift_var_future_list::Vector{Set{Sym}}

    # dyn_shift2_var_past_list::Vector{Set{Sym}}

    # dyn_var_present_list::Vector{Set{Sym}}
    # dyn_var_past_list::Vector{Set{Sym}}
    # dyn_var_future_list::Vector{Set{Sym}}
    # dyn_ss_list::Vector{Set{Sym}}
    # dyn_exo_list::Vector{Set{Sym}}

    var_present_list::Vector{Set{Sym}}
    var_past_list::Vector{Set{Sym}}
    var_future_list::Vector{Set{Sym}}
    ss_list::Vector{Set{Sym}}
    var_list::Vector{Set{Sym}}
    # dynamic_variables_list::Vector{Set{Sym}}
    # dynamic_variables_future_list::Vector{Set{Sym}}

    par_list::Vector{Set{Sym}}

    calibration_equations::Vector{Sym}
    calibration_equations_parameters::Vector{Sym}
    # parameters::Vector{Sym}

    # var_present::Set{Sym}
    # var_past::Set{Sym}
    # var_future::Set{Sym}
    var::Set{Sym}
    nonnegativity_auxilliary_vars::Set{Sym}

    ss_calib_list::Vector{Set{Sym}}
    par_calib_list::Vector{Set{Sym}}

    var_redundant_list::Vector{Set{Sym}}
    # var_redundant_calib_list::Vector{Set{Sym}}
    # var_solved_list::Vector{Set{Sym}}
    # var_solved_calib_list::Vector{Set{Sym}}
end


struct perturbation_solution
    solution_matrix::AbstractMatrix{Float64}
    state_update::Function
end

struct higher_order_perturbation_solution
    solution_matrix::AbstractMatrix{Float64}
    stochastic_steady_state::Vector{Float64}
    state_update::Function
end

mutable struct perturbation
    first_order::perturbation_solution
    linear_time_iteration::perturbation_solution
    second_order::higher_order_perturbation_solution
    third_order::higher_order_perturbation_solution
end


mutable struct solution
    perturbation::perturbation
    non_stochastic_steady_state::ComponentVector{Float64}
    algorithms::Set{Symbol}
    outdated_algorithms::Set{Symbol}
    outdated_NSSS::Bool
    functions_written::Bool
    valid_steady_state_solution
end



mutable struct ℳ
    model_name
    SS_optimizer
    exo::Vector{Symbol}
    par::Vector{Symbol}
    parameters::Vector{Symbol}
    parameter_values::Vector{Number}
    ss
    dynamic_variables::Vector{Symbol}
    dyn_ss_past::Vector{Symbol}
    dyn_ss_present::Vector{Symbol}
    dyn_ss_future::Vector{Symbol}

    aux::Vector{Symbol}
    aux_present::Vector{Symbol}
    aux_future::Vector{Symbol}
    aux_past::Vector{Symbol}

    exo_future::Vector{Symbol}
    exo_present::Vector{Symbol}
    exo_past::Vector{Symbol}

    var::Vector{Symbol}
    var_present::Vector{Symbol}
    var_future::Vector{Symbol}
    var_past::Vector{Symbol}

    exo_list::Vector{Set{Symbol}}
    var_list::Vector{Set{Symbol}}
    dynamic_variables_list::Vector{Set{Symbol}}
    dynamic_variables_future_list::Vector{Set{Symbol}}

    ss_calib_list::Vector{Set{Symbol}}
    par_calib_list::Vector{Set{Symbol}}

    ss_no_var_calib_list::Vector{Set{Symbol}}
    par_no_var_calib_list::Vector{Set{Symbol}}

    ss_list::Vector{Set{Symbol}}

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

    par_list::Vector{Set{Symbol}}
    var_future_list::Vector{Set{Symbol}}
    var_present_list::Vector{Set{Symbol}}
    var_past_list::Vector{Set{Symbol}}

    dyn_shift_var_future_list::Vector{Set{Symbol}}
    dyn_shift_var_present_list::Vector{Set{Symbol}}
    dyn_shift_var_past_list::Vector{Set{Symbol}}

    dyn_shift2_var_past_list::Vector{Set{Symbol}}

    dyn_var_future_list::Vector{Set{Symbol}}
    dyn_var_present_list::Vector{Set{Symbol}}
    dyn_var_past_list::Vector{Set{Symbol}}
    dyn_ss_list::Vector{Set{Symbol}}
    dyn_exo_list::Vector{Set{Symbol}}

    solved_vars::Vector#{Union{Symbol,Vector{Symbol}}}
    solved_vals::Vector#{Union{Float64,Expr,Int,Vector{Union{Float64,Expr,Int}}}}
    non_linear_solved_vars
    non_linear_solved_vals
    # solved_sub_vals
    # solved_sub_values
    ss_solve_blocks
    ss_solve_blocks_optim
    SS_init_guess::Vector{Real}
    SS_solve_func
    nonlinear_solution_helper
    SS_dependencies

    nonnegativity_auxilliary_vars::Vector{Symbol}
    ss_equations::Vector{Expr}
    t_future_equations 
    t_past_equations 
    t_present_equations 
    dyn_equations::Vector{Expr}
    dyn_equations_future::Vector{Expr}
    equations::Vector{Expr}

    calibration_equations_no_var::Vector{Expr}

    calibration_equations::Vector{Expr}
    calibration_equations_parameters::Vector{Symbol}

    bounds⁺::Vector{Symbol}

    bounded_vars::Vector{Symbol}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}

    model_function::Function

    timings::timings
    solution::solution
    # symbolics::symbolics
    
end

