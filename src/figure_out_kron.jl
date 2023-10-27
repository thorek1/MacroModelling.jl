using MacroModelling
using MatrixEquations, BenchmarkTools, ThreadedSparseArrays
import MacroModelling: parse_variables_input_to_index, calculate_covariance, solve_matrix_equation_AD, write_functions_mapping!, multiplicate, generateSumVectors, product_moments, solve_matrix_equation_forward, calculate_second_order_moments, determine_efficient_order, calculate_third_order_solution, get_symbols, match_pattern,calculate_quadratic_iteration_solution, calculate_linear_time_iteration_solution, A_mult_kron_power_3_B, mat_mult_kron
import LinearAlgebra as ℒ
import RecursiveFactorization as RF
import SpeedMapping: speedmapping


include("../models/RBC_baseline.jl")

include("../test/models/GNSS_2010.jl")

𝓂 = GNSS_2010
max_perturbation_order = 1
import  Symbolics

future_varss  = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₁₎$")))
present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎$")))
past_varss    = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₋₁₎$")))
shock_varss   = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₓ₎$")))
ss_varss      = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₛₛ₎$")))

sort!(future_varss  ,by = x->replace(string(x),r"₍₁₎$"=>"")) #sort by name without time index because otherwise eps_zᴸ⁽⁻¹⁾₍₋₁₎ comes before eps_z₍₋₁₎
sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))
sort!(past_varss    ,by = x->replace(string(x),r"₍₋₁₎$"=>""))
sort!(shock_varss   ,by = x->replace(string(x),r"₍ₓ₎$"=>""))
sort!(ss_varss      ,by = x->replace(string(x),r"₍ₛₛ₎$"=>""))

steady_state = []
for (i, var) in enumerate(ss_varss)
    push!(steady_state,:($var = X̄[$i]))
    # ii += 1
end

ii = 1

alll = []
for var in future_varss
    push!(alll,:($var = X[$ii]))
    ii += 1
end

for var in present_varss
    push!(alll,:($var = X[$ii]))
    ii += 1
end

for var in past_varss
    push!(alll,:($var = X[$ii]))
    ii += 1
end

for var in shock_varss
    push!(alll,:($var = X[$ii]))
    ii += 1
end


# paras = []
# push!(paras,:((;$(vcat(𝓂.parameters,𝓂.calibration_equations_parameters)...)) = params))

paras = []
for (i, parss) in enumerate(vcat(𝓂.parameters,𝓂.calibration_equations_parameters))
    push!(paras,:($parss = params[$i]))
end

# # watch out with naming of parameters in model and functions
# mod_func2 = :(function model_function_uni_redux(X::Vector, params::Vector{Number}, X̄::Vector)
#     $(alll...)
#     $(paras...)
# 	$(𝓂.calibration_equations_no_var...)
#     $(steady_state...)
#     [$(𝓂.dyn_equations...)]
# end)


# 𝓂.model_function = @RuntimeGeneratedFunction(mod_func2)
# 𝓂.model_function = eval(mod_func2)

dyn_future_list = collect(reduce(union, 𝓂.dyn_future_list))
dyn_present_list = collect(reduce(union, 𝓂.dyn_present_list))
dyn_past_list = collect(reduce(union, 𝓂.dyn_past_list))
dyn_exo_list = collect(reduce(union,𝓂.dyn_exo_list))

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))

vars_raw = [dyn_future_list[indexin(sort(future),future)]...,
        dyn_present_list[indexin(sort(present),present)]...,
        dyn_past_list[indexin(sort(past),past)]...,
        dyn_exo_list[indexin(sort(exo),exo)]...]

# overwrite SymPyCall names
eval(:(Symbolics.@variables $(reduce(union,get_symbols.(𝓂.dyn_equations))...)))

vars = eval(:(Symbolics.@variables $(vars_raw...)))

eqs = Symbolics.parse_expr_to_symbolic.(𝓂.dyn_equations,(@__MODULE__,))

# second_order_idxs = []
# third_order_idxs = []
# if max_perturbation_order >= 2 
    nk = length(vars_raw)
    second_order_idxs = [nk * (i-1) + k for i in 1:nk for k in 1:i]
    # if max_perturbation_order == 3
        third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]
    # end
# end

tasks_per_thread = 1 # customize this as needed. More tasks have more overhead, but better
                # load balancing

chunk_size = max(1, length(vars) ÷ (tasks_per_thread * Threads.nthreads()))
data_chunks = Iterators.partition(vars, chunk_size) # partition your data into chunks that
                                            # individual tasks will deal with
#See also ChunkSplitters.jl and SplittablesBase.jl for partitioning data
full_data_chunks = [[i,eqs,vars,max_perturbation_order,second_order_idxs,third_order_idxs] for i in data_chunks]
typeof(full_data_chunks)

function take_symbolic_derivatives(all_inputs::Vector)
    var_chunk, eqs, vars, max_perturbation_order, second_order_idxs, third_order_idxs = all_inputs

    # Initialize storage for derivatives and indices
    first_order, second_order, third_order = [], [], []
    row1, row2, row3 = Int[], Int[], Int[]
    column1, column2, column3 = Int[], Int[], Int[]

    # Compute derivatives for each variable in the chunk
    for var1 in var_chunk
        c1 = Int(indexin(var1, vars)...)

        # Check each equation for the presence of the variable
        for (r, eq) in enumerate(eqs)
            if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
                deriv_first = Symbolics.derivative(eq, var1)
                push!(first_order, Symbolics.toexpr(deriv_first))
                push!(row1, r)
                push!(column1, c1)

                # Compute second order derivatives if required
                if max_perturbation_order >= 2 
                    for (c2, var2) in enumerate(vars)
                        if (((c1 - 1) * length(vars) + c2) ∈ second_order_idxs) && 
                            (Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first)))
                            deriv_second = Symbolics.derivative(deriv_first, var2)
                            push!(second_order, Symbolics.toexpr(deriv_second))
                            push!(row2, r)
                            push!(column2, Int.(indexin([(c1 - 1) * length(vars) + c2], second_order_idxs))...)

                            # Compute third order derivatives if required
                            if max_perturbation_order == 3
                                for (c3, var3) in enumerate(vars)
                                    if (((c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3) ∈ third_order_idxs) && 
                                        (Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second)))
                                        deriv_third = Symbolics.derivative(deriv_second, var3)
                                        push!(third_order, Symbolics.toexpr(deriv_third))
                                        push!(row3, r)
                                        push!(column3, Int.(indexin([(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3], third_order_idxs))...)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return first_order, second_order, third_order, row1, row2, row3, column1, column2, column3
end

import ThreadsX
ThreadsX.mapi(take_symbolic_derivatives, full_data_chunks)

using FLoops, MicroCollections, BangBang
@floop for chunk in full_data_chunks
    out = take_symbolic_derivatives(chunk)
    @reduce(states = append!!(EmptyVector(), out))
end



tasks = map(full_data_chunks) do chunk
    # Each chunk of your data gets its own spawned task that does its own local, sequential work
    # and then returns the result
    Threads.@spawn begin
        take_symbolic_derivatives(chunk)
    end
end


states = fetch.(tasks)

first_order =   vcat([i[1] for i in states]...)
second_order =  vcat([i[2] for i in states]...)
third_order =   vcat([i[3] for i in states]...)

row1 =  vcat([i[4] for i in states]...)
row2 =  vcat([i[5] for i in states]...)
row3 =  vcat([i[6] for i in states]...)

column1 =   vcat([i[7] for i in states]...)
column2 =   vcat([i[8] for i in states]...)
column3 =   vcat([i[9] for i in states]...)














# Remove redundant variables in non stochastic steady state problem:      2.518 seconds
# Set up non stochastic steady state problem:     2.127 seconds
# Take symbolic derivatives up to first order:    6.124 seconds
# Find non stochastic steady state:       0.212 seconds


# no threads
# Remove redundant variables in non stochastic steady state problem:      2.568 seconds
# Set up non stochastic steady state problem:     2.125 seconds
# Take symbolic derivatives up to first order:    4.841 seconds
# Find non stochastic steady state:       0.195 seconds

get_SSS(RBC_baseline,algorithm = :third_order)
get_SSS(RBC_baseline,algorithm = :third_order)



include("../test/models/GNSS_2010.jl")
include("../test/models/SW03.jl")

@benchmark get_solution(GNSS_2010, algorithm = :second_third_order)

get_solution(GNSS_2010, algorithm = :pruned_third_order)

include("../test/models/NAWM_EAUS_2008.jl")
#no threads
# Take symbolic derivatives up to first order:    3.437 seconds

get_solution(NAWM_EAUS_2008,algorithm = :pruned_second_order)

@benchmark get_solution(NAWM_EAUS_2008, algorithm = :pruned_second_order)

@profview get_solution(NAWM_EAUS_2008, algorithm = :pruned_second_order)

@profview for i in 1:5 get_solution(m, algorithm = :pruned_second_order) end

@profview for i in 1:5 get_solution(m, algorithm = :pruned_third_order) end

@benchmark get_solution(m, algorithm = :first_order)

@benchmark get_solution(m, algorithm = :pruned_second_order)

@benchmark get_solution(m, algorithm = :pruned_third_order)

get_SSS(m,algorithm = :pruned_third_order)

get_std(m,algorithm = :pruned_third_order)

@profview for i in 1:10 get_solution(m,algorithm = :pruned_third_order) end
@benchmark get_solution(GNSS_2010,algorithm = :pruned_third_order)
@benchmark get_irf(m,algorithm = :pruned_third_order, shocks = :eta_G, variables = :C)
get_shocks(m)

m = GNSS_2010

m = RBC_baseline


m = green_premium_recalib




𝓂 = m
write_functions_mapping!(𝓂, 3)
parameters = 𝓂.parameter_values
verbose = true
silent = false
T = 𝓂.timings
tol =eps()
M₂ = 𝓂.solution.perturbation.second_order_auxilliary_matrices;
M₃ = 𝓂.solution.perturbation.third_order_auxilliary_matrices;



nₑ₋ = 𝓂.timings.nPast_not_future_and_mixed + 𝓂.timings.nVars + 𝓂.timings.nFuture_not_past_and_mixed + 𝓂.timings.nExo

# setup compression matrices
colls2 = [nₑ₋ * (i-1) + k for i in 1:nₑ₋ for k in 1:i]
𝐂₂ = sparse(colls2, 1:length(colls2), 1)
𝐔₂ = 𝐂₂' * sparse([i <= k ? (k - 1) * nₑ₋ + i : (i - 1) * nₑ₋ + k for k in 1:nₑ₋ for i in 1:nₑ₋], 1:nₑ₋^2, 1)

findnz(𝐂₂)
colls3 = [nₑ₋^2 * (i-1) + nₑ₋ * (k-1) + l for i in 1:nₑ₋ for k in 1:i for l in 1:k]
𝐂∇₃ = sparse(colls3, 1:length(colls3) , 1)

sparse([1,1],[1,10],ones(2),1,18915) * 𝐔₂ |> findnz


SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)

∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix

𝑺₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

∇₂ * 𝐔₂ |>findnz
∇₂ * 𝐂₂ |>findnz
# ([14, 9, 32, 14, 14, 17, 9, 32, 9, 32  …  23, 18, 19, 24, 18, 18, 19, 24, 7, 85], [1, 6, 6, 7, 28, 36, 39, 39, 45, 45  …  17578, 17670, 17670, 17719, 17727, 17766, 17766, 17766, 17912, 18528], [9.758936959857042e-5, -511182.664806759, 0.0, -0.000596781277575951, 4423.239672386465, 4228.784468072789, -383521.366610366, 0.0, -328661.57573967846, 0.0  …  0.016104591640058362, -277.43682068319697, -1.9096958759619143, -0.3020994557063705, 1.9096958759619143, 2.6625924677156942, 0.025103167950409996, 0.015746616704267007, -1.0, -1424.3149435714465])
# 105×18915 SparseArrays.SparseMatrixCSC{Float64, Int64} with 218 stored entries:
# ⎡⠇⠀⠀⠀⠀⠶⠒⠂⠈⠍⠀⠀⠀⠀⠰⠖⠀⠂⠐⠺⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠫⠤⠤⠊⠂⠂⠄⠀⠘⠨⠿⠂⠀⠘⠭⠄⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠒⠎⠀⠀⠀⠀⎤
# ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⢀⠀⡀⠀⠀⠠⠀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠐⠂⠀⠀⠀⡄⠀⠀⠀⠀⠀⠄⠀⢀⠀⠀⢀⠀⠀⠀⠄⠀⠠⠄⠀⠀⠀⠀⠄⠀⎦
# 105×37636 SparseArrays.SparseMatrixCSC{Float64, Int64} with 370 stored entries:
# ⎡⠯⠶⠔⠾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠴⠔⠒⠒⠀⠈⠧⠛⠀⠀⠀⠀⠀⠀⠖⠂⠒⠀⠒⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠥⠤⠛⠂⠄⠘⠿⠂⠘⠥⠀⠀⠀⠀⠉⠀⠀⠐⠎⠀⠀⎤
# ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠀⡀⢀⠀⠀⠀⢠⠀⢄⠀⠀⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠒⠀⠀⡄⠀⠀⠠⠀⡀⢀⠀⠠⠀⠄⠀⠠⠀⎦

# ∇₂ * 𝐔₂

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
    
    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];


    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    # ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = - ∇₂ * sparse(ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 
    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = -(mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + mat_mult_kron(∇₂, 𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 

    X = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
    droptol!(X,tol)

    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    B = spinv * ∇₁₊
    droptol!(B,tol)

    C = (M₂.𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + M₂.𝐔₂ * M₂.𝛔) * M₂.𝐂₂
    droptol!(C,tol)

    r1,c1,v1 = findnz(B)
    r2,c2,v2 = findnz(C)
    r3,c3,v3 = findnz(X)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    push!(coordinates,(r2,c2))
    push!(coordinates,(r3,c3))
    
    values = vcat(v1, v2, v3)

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(B))
    push!(dimensions,size(C))
    push!(dimensions,size(X))

    solver = length(X.nzval) / length(X) < .1 ? :sylvester : :gmres

    𝐒₂, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = solver, sparse_output = true)

    𝐒₂ *= M₂.𝐔₂



∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

∇₃ |> findnz
# ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  5, 5, 7, 5, 5, 5, 7, 5, 5, 7], [1, 2, 3, 7, 18, 19, 20, 24, 35, 36  …  3888, 3890, 3941, 3952, 3958, 3986, 3989, 3990, 3992, 3992], [50.48505194503706, -0.6283447768657043, -0.3141723884328518, 0.01873475423775321, -0.6283447768657043, -0.18446548770428417, 0.18446548770428464, -0.011000061446283651, -0.3141723884328518, 0.18446548770428464  …  -0.06410256410256433, 0.0038225694831487155, -0.0019084573972820997, -0.06410256410256433, 0.0019112847415743552, 0.0038225694831487155, -0.0019084573972820997, 0.0019112847415743552, -0.0002849347303432456, 0.0005505165569082998])


# ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  5, 5, 7, 5, 5, 5, 7, 5, 5, 7], [1, 2, 3, 7, 18, 19, 20, 24, 35, 36  …  3888, 3890, 3941, 3952, 3958, 3986, 3989, 3990, 3992, 3992], [50.48505194503706, -0.6283447768657043, -0.3141723884328518, 0.01873475423775321, -0.6283447768657043, -0.18446548770428406, 0.18446548770428464, -0.011000061446283651, -0.3141723884328518, 0.18446548770428464  …  -0.06410256410256433, 0.0038225694831487155, -0.0019084573972820997, -0.06410256410256433, 0.0019112847415743552, 0.0038225694831487155, -0.0019084573972820997, 0.0019112847415743552, -0.0002849347303432456, 0.0005505165569082998])


∇₃ * 𝐂∇₃ |> findnz
# ([1, 1, 1, 1, 1, 1, 1, 1, 2, 1  …  6, 6, 5, 5, 5, 5, 7, 5, 5, 7], [1, 2, 3, 4, 5, 6, 7, 20, 20, 57  …  120, 256, 322, 491, 529, 554, 557, 558, 560, 560], [50.48505194503706, -0.6283447768657043, -0.18446548770428417, -0.43323369470773176, -0.3141723884328518, 0.18446548770428464, 0.10830842367693383, -50.48505194503706, -0.0, 0.01873475423775321  …  348.28953472649823, -36.00000000000012, 2.1499353995462793, 0.06410256410256439, -0.06410256410256433, 0.0038225694831487155, -0.0019084573972820997, 0.0019112847415743552, -0.0002849347303432456, 0.0005505165569082998])
droptol!(∇₃,eps())
# 105×7301384 SparseArrays.SparseMatrixCSC{Float64, Int64} with 857 stored entries:
# ⎡⠋⠖⠔⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠴⠔⠒⠒⠀⠈⠧⠋⠀⠀⠀⠀⠀⠀⠖⠂⠒⠀⠒⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠥⠤⠛⠂⠄⠀⠿⠂⠈⠥⠀⠀⠀⠀⠉⠀⠀⠐⠆⠀⠀⎤
# ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠀⡀⢀⠀⠀⠀⢠⠀⢄⠀⠀⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠒⠀⠀⡄⠀⠀⠠⠀⡀⢀⠀⠠⠀⠄⠀⠠⠀⎦

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]


    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    B = spinv * ∇₁₊
    droptol!(B,tol)

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋

    # 𝐗₃ = -∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux)
    𝐗₃ = -A_mult_kron_power_3_B(∇₃, aux)

    tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)
    out = - ∇₃ * tmpkron - ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron * M₃.𝐏₁ᵣ̃ - ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron * M₃.𝐏₂ᵣ̃
    𝐗₃ += out
    
    # tmp𝐗₃ = -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
    tmp𝐗₃ = -mat_mult_kron(∇₂, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)

    tmpkron1 = -∇₂ *  ℒ.kron(𝐒₁₊╱𝟎,𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔,𝐒₁₋╱𝟏ₑ)
    out2 = tmpkron1 * tmpkron2 +  tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
    
    𝐗₃ += (tmp𝐗₃ + out2 + -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * M₂.𝛔)) * M₃.𝐏# |> findnz
    
    𝐗₃ += @views -∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]) * M₃.𝐏
    droptol!(𝐗₃,tol)
    
    X = spinv * 𝐗₃ * M₃.𝐂₃
    droptol!(X,tol)
    
    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
    
    C = M₃.𝐔₃ * tmpkron + M₃.𝐔₃ * M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐔₃ * M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃
    C += M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ)) # no speed up here from A_mult_kron_power_3_B
    C *= M₃.𝐂₃
    droptol!(C,tol)

    r1,c1,v1 = findnz(B)
    r2,c2,v2 = findnz(C)
    r3,c3,v3 = findnz(X)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    push!(coordinates,(r2,c2))
    push!(coordinates,(r3,c3))
    
    values = vcat(v1, v2, v3)

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(B))
    push!(dimensions,size(C))
    push!(dimensions,size(X))

    𝐒₃, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres, sparse_output = true)

    𝐒₃ *= M₃.𝐔₃

# 105×29791 SparseArrays.SparseMatrixCSC{Float64, Int64} with 83430 stored entries:
# ⎡⣿⣿⣿⠀⠀⠀⠀⠠⠀⠀⠀⠀⣿⣿⣿⠀⠀⠠⠀⠀⠀⢸⣿⢸⡇⠀⠀⠤⠀⠀⠀⣿⣿⣿⠀⠀⠀⠀⠀⠠⠄⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⠀⠀⠀⠀⠄⠀⢸⣿⣿⣿⡇⣿⎤
# ⎣⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⠀⠀⠀⠀⠀⠀⢸⣿⢸⡇⠀⠀⠀⠀⠀⠀⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⡇⣿⎦


function A_mult_kron_power_3_B(A::AbstractArray{R},B::AbstractArray{T}; tol::AbstractFloat = eps()) where {R <: Real, T <: Real}
    n_row = size(B,1)
    n_col = size(B,2)

    B̄ = collect(B)

    vals = T[]
    rows = Int[]
    cols = Int[]

    for row in 1:size(A,1)
        idx_mat, vals_mat = A[row,:] |> findnz

        if length(vals_mat) == 0 continue end

        for col in 1:size(B,2)^3
            col_1, col_3 = divrem((col - 1) % (n_col^2), n_col) .+ 1
            col_2 = ((col - 1) ÷ (n_col^2)) + 1

            mult_val = 0.0

            for (i,idx) in enumerate(idx_mat)
                i_1, i_3 = divrem((idx - 1) % (n_row^2), n_row) .+ 1
                i_2 = ((idx - 1) ÷ (n_row^2)) + 1
                mult_val += vals_mat[i] * B̄[i_1,col_1] * B̄[i_2,col_2] * B̄[i_3,col_3]
            end

            if abs(mult_val) > tol
                push!(vals,mult_val)
                push!(rows,row)
                push!(cols,col)
            end
        end
    end

    sparse(rows,cols,vals,size(A,1),size(B,2)^3)
end

M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ))


# tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)
# out = - ∇₃ * tmpkron - ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron * M₃.𝐏₁ᵣ̃ - ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron * M₃.𝐏₂ᵣ̃
# 𝐗₃ += out

tmp𝐗₃ = -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
29*841

droptol!(tmp𝐗₃,eps())
tmpp = -mat_mult_kron(∇₂,⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)

isapprox(tmpp,tmp𝐗₃,rtol = 1e-10)

function mat_mult_kron(A::AbstractArray{T},B::AbstractArray{T},C::AbstractArray{T}; tol::AbstractFloat = eps()) where T <: Real
    n_rowB = size(B,1)
    n_colB = size(B,2)

    n_rowC = size(C,1)
    n_colC = size(C,2)

    B̄ = collect(B)
    C̄ = collect(C)

    vals = T[]
    rows = Int[]
    cols = Int[]

    for row in 1:size(A,1)
        idx_mat, vals_mat = A[row,:] |> findnz

        if length(vals_mat) == 0 continue end

        for col in 1:(n_colB*n_colC)
            col_1, col_2 = divrem((col - 1) % (n_colB*n_colC), n_colC) .+ 1

            mult_val = 0.0

            for (i,idx) in enumerate(idx_mat)
                i_1, i_2 = divrem((idx - 1) % (n_rowB*n_rowC), n_rowC) .+ 1
                
                mult_val += vals_mat[i] * B̄[i_1,col_1] * C̄[i_2,col_2]
            end

            if abs(mult_val) > tol
                push!(vals,mult_val)
                push!(rows,row)
                push!(cols,col)
            end
        end
    end

    sparse(rows,cols,vals,size(A,1),n_colB*n_colC)
end

n_colB = size(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,2)
n_colC = size(⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎,2)
col = 900

((col - 1) ÷ (n_colB*n_colC)) + 1
((col - 1) % (n_colB*n_colC)) + 1

col_1, col_2 = divrem((col - 1) % (n_colB*n_colC), n_colC) .+ 1


∇₂[1,:]' * kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋[:,1],⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎[:,1])


@benchmark tmp𝐗₃ = -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
# 29*841

droptol!(tmp𝐗₃,eps())
@benchmark tmpp = -mat_mult_kron(∇₂,⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)

@profview tmpp = -mat_mult_kron(∇₂,⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)
tmpp[1,1]
tmp𝐗₃[1,1]
isapprox(tmpp,tmp𝐗₃,rtol = 1e-15)




# kron(aux,aux)
# ℒ.kron(ℒ.kron(aux, aux), aux)[:,1]

# # first column of kronecker product
# ℒ.kron(ℒ.kron(aux[:,1],aux[:,1]),aux[:,1])[1843]

# ∇₃[1,:] * ℒ.kron(ℒ.kron(aux[:,1],aux[:,1]),aux[:,1])'



# # first row of third order derivatives matrix
# idx_mat, vals_mat = ∇₃[1,:] |> findnz
# idx_kron, vals_kron = aux[:,1] |> findnz




# (∇₃*ℒ.kron(ℒ.kron(aux, aux), aux))[:,2]


function A_mult_kron_power_3_B(A::AbstractArray{T},B::AbstractArray{T}; tol::AbstractFloat = eps()) where T <: Real
    n_row = size(B,1)
    n_col = size(B,2)

    B̄ = collect(B)

    vals = T[]
    rows = Int[]
    cols = Int[]

    for row in 1:size(A,1)
        idx_mat, vals_mat = A[row,:] |> findnz

        if length(vals_mat) == 0 continue end

        for col in 1:size(B,2)^3
            col_1, col_3 = divrem((col - 1) % (n_col^2), n_col) .+ 1
            col_2 = ((col - 1) ÷ (n_col^2)) + 1

            mult_val = 0.0

            for (i,idx) in enumerate(idx_mat)
                i_1, i_3 = divrem((idx - 1) % (n_row^2), n_row) .+ 1
                i_2 = ((idx - 1) ÷ (n_row^2)) + 1
                mult_val += vals_mat[i] * B̄[i_1,col_1] * B̄[i_2,col_2] * B̄[i_3,col_3]
            end

            if abs(mult_val) > tol
                push!(vals,mult_val)
                push!(rows,row)
                push!(cols,col)
            end
        end
    end

    sparse(rows,cols,vals,size(A,1),size(B,2)^3)
end




function A_mult_kron_power_3_B_multithreaded(A::AbstractArray{T},B::AbstractArray{T}) where T <: Real
    n_row = size(B,1)
    n_col = size(B,2)

    B̄ = collect(B)

    sparse_init() = [T[], Int[], Int[]]
    Polyester.@batch per=thread threadlocal= sparse_init() for row in 1:size(A,1)
        idx_mat, vals_mat = A[row,:] |> findnz

        if length(vals_mat) == 0 continue end

        for col in 1:size(B,2)^3
            col_1 = ((col - 1) % (n_col^2) ÷ n_col) + 1
            col_2 = ((col - 1) ÷ (n_col^2)) + 1
            col_3 = ((col - 1) % n_col) + 1

            mult_val = 0.0

            for (i,idx) in enumerate(idx_mat)
                i_1, i_3 = divrem((idx - 1) % (n_row^2), n_row) .+ 1
                i_2 = ((idx - 1) ÷ (n_row^2)) + 1
                mult_val += vals_mat[i] * B̄[i_1,col_1] * B̄[i_2,col_2] * B̄[i_3,col_3] 
            end

            if abs(mult_val) > eps()
                push!(threadlocal[1],mult_val)
                push!(threadlocal[2],row)
                push!(threadlocal[3],col)
            end
        end
    end
    
    sparse(Int.(threadlocal[1][2]),Int.(threadlocal[1][3]),T.(threadlocal[1][1]),size(A,1),size(B,2)^3)
end


using BenchmarkTools
@benchmark A_mult_kron_power_3_B(∇₃,aux)
@benchmark A_mult_kron_power_3_B_multithreaded(∇₃,aux)
@benchmark ∇₃*ℒ.kron(ℒ.kron(aux, aux), aux)


@profview for i in 1:10 A_mult_kron_power_3_B(∇₃,aux) end
@profview for i in 1:10 A_mult_kron_power_3_B_multithreaded(∇₃,aux) end



idx = 20
n_row = 3
i_1 = ((idx - 1) % (n_row^2) ÷ n_row) + 1
i_2 = ((idx - 1) ÷ (n_row^2)) + 1
i_3 = ((idx - 1) % n_row) + 1


temp = (idx - 1)
i_1, i_3 = divrem(temp % (n_row^2), n_row) .+1
i_2 = (temp ÷ (n_row^2)) + 1



manual_sparse = sparse(rows,cols,vals,size(∇₃,1),size(aux,2)^3)


isapprox(∇₃*ℒ.kron(ℒ.kron(aux, aux), aux), manual_sparse, rtol = 1e-15)


return sparse(final_rows, final_cols, vals, size(A,1) * size(B,1), size(A,1) * size(B,1))




using Kronecker
aux ⊗ 2

@benchmark 𝐗₃ = -∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux)
𝐗₃ |> collect
𝐗₃ = -∇₃ * aux ⊗ 3