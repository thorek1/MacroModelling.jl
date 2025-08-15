using MacroModelling


@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    k[ss] / q[ss] = 2.5 | α
    β = 0.95
end

steady_state = SS(RBC, derivatives = false)

get_non_stochastic_steady_state_residuals(RBC, steady_state)


RBC.dyn_equations

RBC.ss_equations
unknown_vars = union(
    reduce(union, RBC.dyn_future_list) |> collect |> sort,
    reduce(union, RBC.dyn_present_list) |> collect |> sort
)

known_vars = union(
    reduce(union, RBC.dyn_past_list) |> collect |> sort,
    reduce(union, RBC.dyn_ss_list) |> collect |> sort,
    reduce(union, RBC.dyn_exo_list) |> collect |> sort
)

stochastic_steady_state = unknown_vars = union(
    reduce(union, RBC.dyn_var_future_list),
    reduce(union, RBC.dyn_var_present_list),
    reduce(union, RBC.dyn_var_past_list)
)|> collect |> sort

known_vars = union(
    reduce(union, RBC.dyn_ss_list),
    reduce(union, RBC.dyn_exo_list)
) |> collect |> sort

function write_ss_check_function!(𝓂::ℳ;
                                    cse = true,
                                    skipzeros = true, 
                                    density_threshold::Float64 = .1,
                                    nnz_parallel_threshold::Int = 1000000,
                                    min_length::Int = 10000)
    unknowns = union(setdiff(𝓂.vars_in_ss_equations, 𝓂.➕_vars), 𝓂.calibration_equations_parameters)

    ss_equations = vcat(𝓂.ss_equations, 𝓂.calibration_equations)



    np = length(𝓂.parameters)
    nu = length(unknowns)
    # nc = length(𝓂.calibration_equations_no_var)

    Symbolics.@variables 𝔓[1:np] 𝔘[1:nu]# ℭ[1:nc]

    parameter_dict = Dict{Symbol, Symbol}()
    back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
    calib_vars = Symbol[]
    calib_expr = []


    for (i,v) in enumerate(𝓂.parameters)
        push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
    end

    for (i,v) in enumerate(unknowns)
        push!(parameter_dict, v => :($(Symbol("𝔘_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔘_$i"))), @__MODULE__) => 𝔘[i])
    end

    for (i,v) in enumerate(𝓂.calibration_equations_no_var)
        push!(calib_vars, v.args[1])
        push!(calib_expr, v.args[2])
        # push!(parameter_dict, v.args[1] => :($(Symbol("ℭ_$i"))))
        # push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("ℭ_$i"))), @__MODULE__) => ℭ[i])
    end

    calib_replacements = Dict{Symbol,Any}()
    for (i,x) in enumerate(calib_vars)
        replacement = Dict(x => calib_expr[i])
        for ii in i+1:length(calib_vars)
            calib_expr[ii] = replace_symbols(calib_expr[ii], replacement)
        end
        push!(calib_replacements, x => calib_expr[i])
    end


    ss_equations_sub = ss_equations |> 
        x -> replace_symbols.(x, Ref(calib_replacements)) |> 
        x -> replace_symbols.(x, Ref(parameter_dict)) |> 
        x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
        x -> Symbolics.substitute.(x, Ref(back_to_array_dict))


    lennz = length(ss_equations_sub)

    if lennz > nnz_parallel_threshold
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, func_exprs = Symbolics.build_function(ss_equations_sub, 𝔓, 𝔘,
                                                cse = cse, 
                                                skipzeros = skipzeros,
                                                # nanmath = false, 
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}


    𝓂.SS_check_func = func_exprs



    ∂SS_equations_∂parameters = Symbolics.sparsejacobian(ss_equations_sub, 𝔓) # nϵ x nx

    lennz = nnz(∂SS_equations_∂parameters)

    if (lennz / length(∂SS_equations_∂parameters) > density_threshold) || (length(∂SS_equations_∂parameters) < min_length)
        derivatives_mat = convert(Matrix, ∂SS_equations_∂parameters)
        buffer = zeros(Float64, size(∂SS_equations_∂parameters))
    else
        derivatives_mat = ∂SS_equations_∂parameters
        buffer = similar(∂SS_equations_∂parameters, Float64)
        buffer.nzval .= 0
    end

    if lennz > nnz_parallel_threshold
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end
    
    _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔓, 𝔘, 
                                                cse = cse, 
                                                skipzeros = skipzeros,
                                                # nanmath = false, 
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    𝓂.∂SS_equations_∂parameters = buffer, func_exprs



    ∂SS_equations_∂SS_and_pars = Symbolics.sparsejacobian(ss_equations_sub, 𝔘) # nϵ x nx

    lennz = nnz(∂SS_equations_∂SS_and_pars)

    if (lennz / length(∂SS_equations_∂SS_and_pars) > density_threshold) || (length(∂SS_equations_∂SS_and_pars) < min_length)
        derivatives_mat = convert(Matrix, ∂SS_equations_∂SS_and_pars)
        buffer = zeros(Float64, size(∂SS_equations_∂SS_and_pars))
    else
        derivatives_mat = ∂SS_equations_∂SS_and_pars
        buffer = similar(∂SS_equations_∂SS_and_pars, Float64)
        buffer.nzval .= 0
    end

    if lennz > nnz_parallel_threshold
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔓, 𝔘, 
                                                cse = cse, 
                                                skipzeros = skipzeros, 
                                                # nanmath = false,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    𝓂.∂SS_equations_∂SS_and_pars = buffer, func_exprs

    return nothing
end
