using Revise
using MacroModelling
using Zygote, ForwardDiff, FiniteDifferences
using BenchmarkTools
using LinearAlgebra

include("../models/Smets_Wouters_2007.jl")

model = Smets_Wouters_2007

params = deepcopy(model.parameter_values)
param_idx = 1

# MacroModelling.DEFAULT_SOLVER_PARAMETERS[7]
# MacroModelling.solver_parameters(6.8658210317889115, 3.054280631509596, 9.239560890529688, 5.0330393159601705, 4.619974181880515, 2.130665389110862, 13.395678237998878, 8.95412704048986, 16.67031860308238, 4.1686309854116175, 7.193385978766233, 6.284359482297452, 1.6025436780830082, 4.080789181245917, 11.237586964445232, 0.9812514892088027, 10.182504561803604, 2.2723756926184744, 5.580529028552923, 4.761189900509761, 1, 0.0, 2)

popfirst!(MacroModelling.DEFAULT_SOLVER_PARAMETERS)
pushfirst!(MacroModelling.DEFAULT_SOLVER_PARAMETERS, MacroModelling.DEFAULT_SOLVER_PARAMETERS[3]);

MacroModelling.clear_solution_caches!(model, :first_order)
get_statistics(model, params, non_stochastic_steady_state = :all, verbose = true)

out_bench = @benchmark get_statistics(model, params, non_stochastic_steady_state = :all) setup = MacroModelling.clear_solution_caches!(model, :first_order)

@profview for i in 1:10000 
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_statistics(model, params, non_stochastic_steady_state = :all)
end

@profview_allocs for i in 1:10000
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_statistics(model, params, non_stochastic_steady_state = :all)
end


# first order solution
MacroModelling.clear_solution_caches!(model, :first_order)
get_solution(model, params)[2]

out_bench = @benchmark get_solution(model, params) setup = MacroModelling.clear_solution_caches!(model, :first_order)

@profview for i in 1:5000
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params)
    get_solution(model, params .+ 0.001)
end

@profview_allocs for i in 1:5000
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params)
    get_solution(model, params .+ 0.001)
end


# Gradients
# Zygote
MacroModelling.clear_solution_caches!(model, :first_order)
Zygote.gradient(x->norm(get_solution(model, x)[2]),params)


out_bench = @benchmark Zygote.gradient(x->norm(get_solution(model, x)),params) setup = MacroModelling.clear_solution_caches!(model, :first_order)


@profview for i in 1:1000
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    Zygote.gradient(x->norm(get_solution(model, x)),params)
end

@profview_allocs for i in 1:1000
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    Zygote.gradient(x->norm(get_solution(model, x)),params)
end

# ForwardDiff
MacroModelling.clear_solution_caches!(model, :first_order)
first_order_one_param = x -> begin
    perturbed = convert.(eltype(x),copy(params))
    perturbed[param_idx] = x
    get_solution(model, perturbed)[2]
end

ForwardDiff.derivative(first_order_one_param, params[param_idx])



out_bench = @benchmark ForwardDiff.derivative(first_order_one_param, params[param_idx]) setup = MacroModelling.clear_solution_caches!(model, :first_order)


@profview for i in 1:1000
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    ForwardDiff.derivative(first_order_one_param, params[param_idx])
end

@profview_allocs for i in 1:1000
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    ForwardDiff.derivative(first_order_one_param, params[param_idx])
end

# FiniteDifferences
MacroModelling.clear_solution_caches!(model, :first_order)
FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params)


out_bench = @benchmark FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params) setup = MacroModelling.clear_solution_caches!(model, :first_order)


@profview for i in 1:100
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params)
end

@profview_allocs for i in 1:100
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params)
end


# second order solution
MacroModelling.clear_solution_caches!(model, :first_order)
get_solution(model, params, algorithm = :second_order)[3] * model.constants.second_order.𝐔₂ |> norm

out_bench = @benchmark get_solution(model, params, algorithm = :second_order) setup = MacroModelling.clear_solution_caches!(model, :second_order)

@profview for i in 1:500
    MacroModelling.clear_solution_caches!(model, :second_order)
    get_solution(model, params)
    get_solution(model, params .+ 0.001, algorithm = :second_order)
end

@profview_allocs for i in 1:500
    MacroModelling.clear_solution_caches!(model, :second_order)
    get_solution(model, params)
    get_solution(model, params .+ 0.001, algorithm = :second_order)
end


# Gradients
# Zygote
MacroModelling.clear_solution_caches!(model, :second_order)
Zygote.gradient(x->norm(get_solution(model, x, algorithm = :second_order)[3] * model.constants.second_order.𝐔₂),params)[1]


out_bench = @benchmark Zygote.gradient(x->norm(get_solution(model, x, algorithm = :second_order)),params) setup = MacroModelling.clear_solution_caches!(model, :second_order)


@profview for i in 1:100
    MacroModelling.clear_solution_caches!(model, :second_order)
    get_solution(model, params .+ 0.001)
    Zygote.gradient(x->norm(get_solution(model, x, algorithm = :second_order)[3]),params)[1]
end

@profview_allocs for i in 1:100
    MacroModelling.clear_solution_caches!(model, :second_order)
    get_solution(model, params .+ 0.001)
    Zygote.gradient(x->norm(get_solution(model, x, algorithm = :second_order)[3]),params)[1]
end

# ForwardDiff
MacroModelling.clear_solution_caches!(model, :second_order)
second_order_one_param = x -> begin
    perturbed = convert.(eltype(x),copy(params))
    perturbed[param_idx] = x
    get_solution(model, perturbed, algorithm = :second_order)[3] * model.constants.second_order.𝐔₂
end

ForwardDiff.derivative(second_order_one_param, params[param_idx])


out_bench = @benchmark ForwardDiff.derivative(second_order_one_param, params[param_idx]) setup = MacroModelling.clear_solution_caches!(model, :second_order)


@profview for i in 1:100
        MacroModelling.clear_solution_caches!(model, :second_order)
        get_solution(model, params .+ 0.001)
        ForwardDiff.derivative(second_order_one_param, params[param_idx])
    end

@profview_allocs for i in 1:100
    MacroModelling.clear_solution_caches!(model, :second_order)
    get_solution(model, params .+ 0.001)
    ForwardDiff.derivative(second_order_one_param, params[param_idx])
end

# FiniteDifferences
MacroModelling.clear_solution_caches!(model, :first_order)
FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params)


out_bench = @benchmark FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params) setup = MacroModelling.clear_solution_caches!(model, :first_order)


@profview for i in 1:100
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params)
end

@profview_allocs for i in 1:100
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params)
end


# third order solution
include("../models/FS2000.jl")
model = FS2000

params = deepcopy(model.parameter_values)
param_idx = 1

MacroModelling.clear_solution_caches!(model, :first_order)
get_solution(model, params, algorithm = :third_order)[4] * model.constants.third_order.𝐔₃ |> norm

out_bench = @benchmark get_solution(model, params, algorithm = :third_order) setup = MacroModelling.clear_solution_caches!(model, :third_order)

@profview for i in 1:10
    MacroModelling.clear_solution_caches!(model, :third_order)
    get_solution(model, params)
    get_solution(model, params .+ 0.001, algorithm = :third_order)
end

@profview_allocs for i in 1:10
    MacroModelling.clear_solution_caches!(model, :third_order)
    get_solution(model, params)
    get_solution(model, params .+ 0.001, algorithm = :third_order)
end


# Gradients
# Zygote
MacroModelling.clear_solution_caches!(model, :third_order)
zyg_grad = Zygote.gradient(x->norm(get_solution(model, x, algorithm = :third_order)[4] * model.constants.third_order.𝐔₃),params)[1]


out_bench = @benchmark Zygote.gradient(x->norm(get_solution(model, x, algorithm = :third_order)[4]),params) setup = MacroModelling.clear_solution_caches!(model, :third_order)


@profview for i in 1:100
    MacroModelling.clear_solution_caches!(model, :third_order)
    get_solution(model, params .+ 0.001)
    Zygote.gradient(x->norm(get_solution(model, x, algorithm = :third_order)[4]),params)[1]
end

@profview_allocs for i in 1:100
    MacroModelling.clear_solution_caches!(model, :third_order)
    get_solution(model, params .+ 0.001)
    Zygote.gradient(x->norm(get_solution(model, x, algorithm = :third_order)[4]),params)[1]
end

# FiniteDifferences
MacroModelling.clear_solution_caches!(model, :first_order)
fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1),x->norm(get_solution(model, x, algorithm = :third_order)[4] * model.constants.third_order.𝐔₃),params)[1]

isapprox(zyg_grad,fin_grad)
zyg_grad - fin_grad
norm(zyg_grad - fin_grad)/max(norm(zyg_grad), norm(fin_grad))

out_bench = @benchmark FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params) setup = MacroModelling.clear_solution_caches!(model, :first_order)


@profview for i in 1:100
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params)
end

@profview_allocs for i in 1:100
    MacroModelling.clear_solution_caches!(model, :first_order)
    get_solution(model, params .+ 0.001)
    FiniteDifferences.grad(FiniteDifferences.central_fdm(2,1),x->norm(get_solution(model, x)),params)
end

# ForwardDiff
MacroModelling.clear_solution_caches!(model, :third_order)
third_order_one_param = x -> begin
    perturbed = convert.(eltype(x),copy(params))
    perturbed[param_idx] = x
    norm(get_solution(model, perturbed, algorithm = :third_order)[4])
    # get_solution(model, perturbed, algorithm = :third_order)[4] * model.constants.third_order.𝐔₃
end

ForwardDiff.derivative(third_order_one_param, params[param_idx])


out_bench = @benchmark ForwardDiff.derivative(third_order_one_param, params[param_idx]) setup = MacroModelling.clear_solution_caches!(model, :third_order)


@profview for i in 1:100
        MacroModelling.clear_solution_caches!(model, :third_order)
        get_solution(model, params .+ 0.001)
        ForwardDiff.derivative(third_order_one_param, params[param_idx])
    end

@profview_allocs for i in 1:100
    MacroModelling.clear_solution_caches!(model, :third_order)
    get_solution(model, params .+ 0.001)
    ForwardDiff.derivative(third_order_one_param, params[param_idx])
end
