using MacroModelling
# include("../models/FS2000.jl")
# include("../models/Caldara_et_al_2012.jl")
include("../models/Ascari_Sbordone_2014.jl")

# include("../test/models/RBC_CME.jl")

# Backus_Kehoe_Kydland_1992
# Iacoviello_2005_linear
# Ghironi_Melitz_2005

FS2000.SS_solve_func
FS2000.ss_solve_blocks

# l = (m * n * psi) / ((n - 1.0) * (psi - 1.0))
# P = (m ^ 2 * n ^ alp * psi * exp(➕₇)) / (bet * ((-alp - n * psi * (alp - 1.0)) + n * (alp - 1.0) + psi * (alp - 1.0) + 1.0) * ➕₉ ^ alp)
# k = ➕₆ ^ (1 / (alp - 1))
# c = (bet * ((-alp - n * psi * (alp - 1.0)) + n * (alp - 1.0) + psi * (alp - 1.0) + 1.0) * ➕₉ ^ alp * exp(➕₃)) / (m * n ^ alp * psi)
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2895 =#
# ➕₂ = min(1.0e12, max(eps(), k))
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2897 =#
# return [(((c - k * (1 - del) * exp(➕₁)) + k) - ➕₂ ^ alp * n ^ (1 - alp) * exp(➕₃))

# SS(FS2000, derivatives = false, verbose = true, parameters = :alp => .3)
# (:P)            0.993250554567553
# (:R)            1.007250755287009
# (:W)            2.7185621689742425
# (:c)            1.006996669068534
# (:d)            0.8608478832599568
# (:dA)           1.0085362275720395
# (:e)            1.0
# (:gp_obs)       0.9917343300675393
# (:gy_obs)       1.0085362275720395
# (:k)           18.982191719109345
# (:l)            0.8610478832599567
# (:log_gp_obs)  -0.008300019997333728
# (:log_gy_obs)   0.008499999999999985
# (:m)            1.0002
# (:n)            0.3167291493594366
# (:y)            1.3558767746137372
import BlockTriangularForm
import SymPyPythonCall as SPyPyC
import Subscripts: super, sub
import MacroTools: postwalk
import MacroModelling: create_symbols_eqs!, remove_redundant_SS_vars!, solve_steady_state!, get_symbols, simplify, make_equation_rebust_to_domain_errors
𝓂 = Ascari_Sbordone_2014
# 𝓂 = FS2000

symbolics = create_symbols_eqs!(𝓂)

remove_redundant_SS_vars!(𝓂, symbolics) 

# solve_steady_state!(𝓂, true, symbolics, verbose = true)


symbolic_SS = true
Symbolics = symbolics
verbose = false


unknowns = union(Symbolics.vars_in_ss_equations,Symbolics.calibration_equations_parameters)

# @assert length(unknowns) <= length(Symbolics.ss_equations) + length(Symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

incidence_matrix = fill(0,length(unknowns),length(unknowns))

eq_list = vcat(union.(setdiff.(union.(Symbolics.var_list,
                                    Symbolics.ss_list),
                                Symbolics.var_redundant_list),
                        Symbolics.par_list),
                union.(Symbolics.ss_calib_list,
                        Symbolics.par_calib_list))


for i in 1:length(unknowns)
    for k in 1:length(unknowns)
        incidence_matrix[i,k] = collect(unknowns)[i] ∈ collect(eq_list)[k]
    end
end

Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence_matrix))
R̂ = []
for i in 1:n_blocks
    [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1] - 1]
end
push!(R̂,1)

vars = hcat(P, R̂)'
eqs = hcat(Q, R̂)'

# @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations for: " * repr([collect(Symbol.(unknowns))[vars[1,eqs[1,:] .< 0]]...]) # repr([vcat(Symbolics.ss_equations,Symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
# @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant euqations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now." # repr([vcat(Symbolics.ss_equations,Symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])


ss_equations = vcat(Symbolics.ss_equations,Symbolics.calibration_equations)# .|> SPyPyC.Sym
# println(ss_equations)

SS_solve_func = []

atoms_in_equations = Set{Symbol}()
atoms_in_equations_list = []
relevant_pars_across = Symbol[]
NSSS_solver_cache_init_tmp = []

min_max_errors = []

# n_block = 1


nb = n_blocks

while nb > 0 
    if length(eqs[:,eqs[2,:] .== nb]) == 2
        var_to_solve_for = collect(unknowns)[vars[:,vars[2,:] .== nb][1]]

        eq_to_solve = ss_equations[eqs[:,eqs[2,:] .== nb][1]]

        # eliminate min/max from equations if solving for variables inside min/max. set to the variable we solve for automatically
        parsed_eq_to_solve_for = eq_to_solve |> string |> Meta.parse

        minmax_fixed_eqs = postwalk(x -> 
            x isa Expr ?
                x.head == :call ? 
                    x.args[1] ∈ [:Max,:Min] ?
                        Symbol(var_to_solve_for) ∈ get_symbols(x.args[2]) ?
                            x.args[2] :
                        Symbol(var_to_solve_for) ∈ get_symbols(x.args[3]) ?
                            x.args[3] :
                        x :
                    x :
                x :
            x,
        parsed_eq_to_solve_for)

        if parsed_eq_to_solve_for != minmax_fixed_eqs
            [push!(atoms_in_equations, a) for a in setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs))]
            push!(min_max_errors,:(solution_error += abs($parsed_eq_to_solve_for)))
            eq_to_solve = eval(minmax_fixed_eqs)
        end

        soll = try SPyPyC.solve(eq_to_solve,var_to_solve_for)
        catch
        end
        
        if isnothing(soll)
            # println("Could not solve single variables case symbolically.")
            println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
            # solve numerically
            continue
        # elseif PythonCall.pyconvert(Bool,soll[1].is_number)

        elseif soll[1].is_number == true
            # ss_equations = ss_equations.subs(var_to_solve,soll[1])
            ss_equations = [eq.subs(var_to_solve_for,soll[1]) for eq in ss_equations]
            
            push!(𝓂.solved_vars,Symbol(var_to_solve_for))
            push!(𝓂.solved_vals,Meta.parse(string(soll[1])))

            if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = max(eps(),$(𝓂.solved_vals[end]))))
            else
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
            end

            push!(atoms_in_equations_list,[])
        else

            push!(𝓂.solved_vars,Symbol(var_to_solve_for))
            push!(𝓂.solved_vals,Meta.parse(string(soll[1])))
            
            # atoms = reduce(union,soll[1].atoms())

            [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
            push!(atoms_in_equations_list, Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)),Symbol.(soll[1].atoms()))))

            # println(atoms_in_equations)
            # push!(atoms_in_equations, soll[1].atoms())

            if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = min(max($(𝓂.lower_bounds[indexin([𝓂.solved_vars[end]],𝓂.bounded_vars)][1]),$(𝓂.solved_vals[end])),$(𝓂.upper_bounds[indexin([𝓂.solved_vars[end]],𝓂.bounded_vars)][1]))))
            else
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
            end
        end
        nb -= 1
    else 
        break
    end
end


# multi var solve

        # push!(single_eqs,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
        # solve symbolically
    # else
vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== nb][1,:]]

eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== nb][1,:]]

eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== nb][1,:]







# write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)


➕_vars = Symbol[]
unique_➕_vars = Union{Symbol,Expr}[]

vars_to_exclude = [Symbol.(vars_to_solve),Symbol[]]

rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_rebust_to_domain_errors(Meta.parse.(string.(eqs_to_solve)), vars_to_exclude, 𝓂.bounds, ➕_vars, unique_➕_vars)


push!(𝓂.solved_vars, Symbol.(vars_to_solve))
push!(𝓂.solved_vals, rewritten_eqs)


syms_in_eqs = Set{Symbol}()

for i in vcat(ss_and_aux_equations_dep, ss_and_aux_equations, rewritten_eqs)
    push!(syms_in_eqs, get_symbols(i)...)
end

setdiff!(syms_in_eqs,➕_vars)

syms_in_eqs2 = Set{Symbol}()

for i in ss_and_aux_equations
    push!(syms_in_eqs2, get_symbols(i)...)
end

union!(syms_in_eqs, intersect(syms_in_eqs2, ➕_vars))

push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

calib_pars = Expr[]
calib_pars_input = Symbol[]
relevant_pars = union(intersect(reduce(union, vcat(𝓂.par_list_aux_SS, 𝓂.par_calib_list)[eq_idx_in_block_to_solve]), syms_in_eqs),intersect(syms_in_eqs, ➕_vars))

union!(relevant_pars_across, relevant_pars)

iii = 1
for parss in union(𝓂.parameters, 𝓂.parameters_as_function_of_parameters)
    if :($parss) ∈ relevant_pars
        push!(calib_pars, :($parss = parameters_and_solved_vars[$iii]))
        push!(calib_pars_input, :($parss))
        iii += 1
    end
end

guess = Expr[]
result = Expr[]

sorted_vars = sort(Symbol.(vars_to_solve))

for (i, parss) in enumerate(sorted_vars) 
    push!(guess,:($parss = guess[$i]))
    push!(result,:($parss = sol[$i]))
end

# separate out auxilliary variables (nonnegativity)
# nnaux = []
# nnaux_linear = []
# nnaux_error = []
# push!(nnaux_error, :(aux_error = 0))
solved_vals = Expr[]
partially_solved_block = Expr[]

other_vrs_eliminated_by_sympy = Set{Symbol}()

for (i,val) in enumerate(𝓂.solved_vals[end])
    if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
        val = vcat(𝓂.ss_aux_equations, 𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
        # push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
        push!(other_vrs_eliminated_by_sympy, val.args[2])
        # push!(nnaux_linear,:($val))
        # push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
    end
end



for (i,val) in enumerate(rewritten_eqs)
    push!(solved_vals, postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
end

# if length(nnaux) > 1
#     all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

#     nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
    
#     inc_matrix = fill(0,length(all_symbols),length(all_symbols))

#     for i in 1:length(all_symbols)
#         for k in 1:length(nn_symbols)
#             inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
#         end
#     end

#     QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

#     nnaux = nnaux[QQ]
#     nnaux_linear = nnaux_linear[QQ]
# end

other_vars = Expr[]
other_vars_input = Symbol[]
other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, ➕_vars),
                                    sort(𝓂.solved_vars[end]) ),
                            union(syms_in_eqs, other_vrs_eliminated_by_sympy ) )
                            # union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

for var in other_vrs
    push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
    push!(other_vars_input,:($(var)))
    iii += 1
end

# solved_vals[end] = Expr(:call, :+, solved_vals[end], ss_and_aux_equations_error_dep...)

funcs = :(function block(parameters_and_solved_vars::Vector, guess::Vector)
        $(guess...) 
        $(calib_pars...) # add those variables which were previously solved and are used in the equations
        $(other_vars...) # take only those that appear in equations - DONE

        $(ss_and_aux_equations_dep...)
        # return [$(solved_vals...),$(nnaux_linear...)]
        return [$(solved_vals...)]
    end)

push!(NSSS_solver_cache_init_tmp,fill(0.897,length(sorted_vars)))
push!(NSSS_solver_cache_init_tmp,[Inf])

# WARNING: infinite bounds are transformed to 1e12
lbs = Float64[]
ubs = Float64[]

limit_boundaries = 1e12

for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
    if haskey(𝓂.bounds,i)
        push!(lbs,𝓂.bounds[i][1])
        push!(ubs,𝓂.bounds[i][2])
    else
        push!(lbs,-limit_boundaries)
        push!(ubs, limit_boundaries)
    end
end

push!(SS_solve_func,ss_and_aux_equations...)

push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...), $(other_vars_input...)]))

push!(SS_solve_func,:(lbs = [$(lbs...)]))
push!(SS_solve_func,:(ubs = [$(ubs...)]))
        
n_block = length(𝓂.ss_solve_blocks) + 1   
    
push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

if VERSION >= v"1.9"
    push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())))
else
    push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver())))
end

push!(SS_solve_func,:(solution = block_solver_AD(params_and_solved_vars,
                                                        $(n_block), 
                                                        𝓂.ss_solve_blocks[$(n_block)], 
                                                        # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                        # f, 
                                                        inits,
                                                        lbs, 
                                                        ubs,
                                                        solver_parameters,
                                                        # fail_fast_solvers_only = fail_fast_solvers_only,
                                                        cold_start,
                                                        verbose)))
                                                        
push!(SS_solve_func,:(iters += solution[2][2])) 
push!(SS_solve_func,:(solution_error += solution[2][1])) 

if length(ss_and_aux_equations_error) > 0
    push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, ss_and_aux_equations_error...))))
end

push!(SS_solve_func,:(sol = solution[1]))

push!(SS_solve_func,:($(result...)))   

push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))


# sort!(vars_to_solve, by = Symbol)
# sort!(eqs_to_solve, by = string)



# bounds = Dict{Symbol,Tuple{Float64,Float64}}()
# ➕_vars = Symbol[]
# unique_➕_vars = Union{Symbol,Expr}[]

# vars_to_exclude = [Symbol.(vars_to_solve),Symbol[]]

# rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = MacroModelling.make_equation_rebust_to_domain_errors(Meta.parse.(string.(eqs_to_solve)), vars_to_exclude, bounds, ➕_vars, unique_➕_vars)


# push!(𝓂.solved_vars, Symbol.(vars_to_solve))
# push!(𝓂.solved_vals, rewritten_eqs)


# syms_in_eqs = Set{Symbol}()

# for i in vcat(ss_and_aux_equations_dep,rewritten_eqs)
#     push!(syms_in_eqs, get_symbols(i)...)
# end

# setdiff!(syms_in_eqs,➕_vars)

# syms_in_eqs2 = Set{Symbol}()

# for i in ss_and_aux_equations
#     push!(syms_in_eqs2, get_symbols(i)...)
# end

# union!(syms_in_eqs, intersect(syms_in_eqs2, ➕_vars))

# push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

# calib_pars = Expr[]
# calib_pars_input = Symbol[]
# relevant_pars = union(intersect(reduce(union, vcat(𝓂.par_list_aux_SS, 𝓂.par_calib_list)[eq_idx_in_block_to_solve]), syms_in_eqs),intersect(syms_in_eqs, ➕_vars))

# union!(relevant_pars_across, relevant_pars)

# iii = 1
# for parss in union(𝓂.parameters, 𝓂.parameters_as_function_of_parameters)
#     if :($parss) ∈ relevant_pars
#         push!(calib_pars, :($parss = parameters_and_solved_vars[$iii]))
#         push!(calib_pars_input, :($parss))
#         iii += 1
#     end
# end

# guess = Expr[]
# result = Expr[]

# sorted_vars = sort(Symbol.(vars_to_solve))

# for (i, parss) in enumerate(sorted_vars) 
#     push!(guess,:($parss = guess[$i]))
#     push!(result,:($parss = sol[$i]))
# end

# # separate out auxilliary variables (nonnegativity)
# # nnaux = []
# # nnaux_linear = []
# # nnaux_error = []
# # push!(nnaux_error, :(aux_error = 0))
# solved_vals = Expr[]
# partially_solved_block = Expr[]

# other_vrs_eliminated_by_sympy = Set{Symbol}()

# for (i,val) in enumerate(𝓂.solved_vals[end])
#     if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
#         val = vcat(𝓂.ss_aux_equations, 𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
#         # push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
#         push!(other_vrs_eliminated_by_sympy, val.args[2])
#         # push!(nnaux_linear,:($val))
#         # push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
#     end
# end



# for (i,val) in enumerate(rewritten_eqs)
#     push!(solved_vals, postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
# end

# # if length(nnaux) > 1
# #     all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

# #     nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
    
# #     inc_matrix = fill(0,length(all_symbols),length(all_symbols))

# #     for i in 1:length(all_symbols)
# #         for k in 1:length(nn_symbols)
# #             inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
# #         end
# #     end

# #     QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

# #     nnaux = nnaux[QQ]
# #     nnaux_linear = nnaux_linear[QQ]
# # end

# other_vars = Expr[]
# other_vars_input = Symbol[]
# other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, ➕_vars),
#                                     sort(𝓂.solved_vars[end]) ),
#                             union(syms_in_eqs, other_vrs_eliminated_by_sympy ) )
#                             # union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

# for var in other_vrs
#     push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
#     push!(other_vars_input,:($(var)))
#     iii += 1
# end

# solved_vals[end] = Expr(:call, :+, solved_vals[end], ss_and_aux_equations_error_dep...)

# funcs = :(function block(parameters_and_solved_vars::Vector, guess::Vector)
#         $(guess...) 
#         $(calib_pars...) # add those variables which were previously solved and are used in the equations
#         $(other_vars...) # take only those that appear in equations - DONE

#         $(ss_and_aux_equations_dep...)
#         # return [$(solved_vals...),$(nnaux_linear...)]
#         return [$(solved_vals...)]
#     end)

# push!(NSSS_solver_cache_init_tmp,fill(0.7688,length(sorted_vars)))
# push!(NSSS_solver_cache_init_tmp,[Inf])

# # WARNING: infinite bounds are transformed to 1e12
# lbs = Float64[]
# ubs = Float64[]

# limit_boundaries = 1e12

# for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
#     if haskey(bounds,i)
#         push!(lbs,bounds[i][1])
#         push!(ubs,bounds[i][2])
#     else
#         push!(lbs,-limit_boundaries)
#         push!(ubs, limit_boundaries)
#     end
# end

# push!(SS_solve_func,ss_and_aux_equations...)

# push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...), $(other_vars_input...)]))

# push!(SS_solve_func,:(lbs = [$(lbs...)]))
# push!(SS_solve_func,:(ubs = [$(ubs...)]))
        
# n_block = length(𝓂.ss_solve_blocks) + 1   
    
# push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

# if VERSION >= v"1.9"
#     push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())))
# else
#     push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver())))
# end

# push!(SS_solve_func,:(solution = block_solver_AD(params_and_solved_vars,
#                                                         $(n_block), 
#                                                         𝓂.ss_solve_blocks[$(n_block)], 
#                                                         # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
#                                                         # f, 
#                                                         inits,
#                                                         lbs, 
#                                                         ubs,
#                                                         solver_parameters,
#                                                         # fail_fast_solvers_only = fail_fast_solvers_only,
#                                                         cold_start,
#                                                         verbose)))
                                                        
# push!(SS_solve_func,:(iters += solution[2][2])) 
# push!(SS_solve_func,:(solution_error += solution[2][1])) 

# if length(ss_and_aux_equations_error) > 0
#     push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, ss_and_aux_equations_error...))))
# end

# push!(SS_solve_func,:(sol = solution[1]))

# push!(SS_solve_func,:($(result...)))   

# push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
# push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

# push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))





# numerical_sol = false

# P = (m * ➕₆ ^ (1 / (alp - 1))) / (((-del + ➕₆ ^ (1 / (alp - 1.0)) * ➕₅ ^ (1.0 - alp) * ➕₇ ^ alp) - exp(➕₄)) + 1.0)
# n = (l * (psi - 1.0)) / ((l * psi - l) - m * psi)
# k = exp(➕₄) / ➕₆ ^ (1 / (alp - 1))
# c = (((-del + ➕₆ ^ (1 / (alp - 1)) * ➕₅ ^ (1 - alp) * ➕₇ ^ alp) - exp(➕₄)) + 1.0) / ➕₆ ^ (1 / (alp - 1))
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2895 =#
# ➕₁ = min(1.0e12, max(eps(), k))
# ➕₂ = min(1.0e12, max(eps(), n))
# #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2897 =#
# return [((-bet * ➕₁ ^ alp * ➕₂ ^ (1 - alp) * (1 - alp) * exp(➕₃)) / (l * m) + 1 / P) + abs(➕₅ - (l * (1.0 - psi)) / (-l * psi + l + m * psi)) + abs(➕₆ - ➕₅ ^ (1.0 - alp) / ((del + exp(➕₄)) - 1.0)) + abs(➕₇ - ➕₆ ^ (-1 / (alp - 1.0)))]

# # write some code that tries solving first all equations for all variables and then solves for one variable less and one equation less. note that the subset of equations and variables to solve for must satisfy the incidence_matrix_subset. that is solving the subset of equations for the subset of variables must be possible. if not, then solve numerically. if it is possible, then solve symbolically and then check if the solution is valid for the other equations. 
incidence_matrix_subset = incidence_matrix[vars[:,vars[2,:] .== nb][1,:], eqs[:,eqs[2,:] .== nb][1,:]]
# var_combo = [1,2,3,5]
# eq_combo = [1,2,3,5]

# incidence_matrix_subset[var_combo,eq_combo]
# incidence_matrix_subset[setdiff(1:length(eqs_to_solve),var_combo),setdiff(1:length(eqs_to_solve),eq_combo)]

# var_indices_to_select_from = findall([sum(incidence_matrix_subset[:,eq_combo],dims = 2)...] .> 0)

# var_indices_in_remaining_eqs = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),eq_combo)],dims = 2)...] .> 0) 

# soll = try SPyPyC.solve(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
# catch
# end

# soll[1] |> collect
# soll[2] |> collect

import Combinatorics: combinations



function partial_solve(eqs_to_solve, vars_to_solve, incidence_matrix_subset)
    for n in length(eqs_to_solve)-1:-1:2
        for eq_combo in combinations(1:length(eqs_to_solve), n)
            var_indices_to_select_from = findall([sum(incidence_matrix_subset[:,eq_combo],dims = 2)...] .> 0)

            var_indices_in_remaining_eqs = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),eq_combo)],dims = 2)...] .> 0) 

            for var_combo in combinations(var_indices_to_select_from, n)
                remaining_vars_in_remaining_eqs = setdiff(var_indices_in_remaining_eqs, var_combo)
                # println("Solving for: ",vars_to_solve[var_combo]," in: ",eqs_to_solve[eq_combo])
                if length(remaining_vars_in_remaining_eqs) == length(eqs_to_solve) - n # not sure whether this condition needs to be there. could be because if the last remaining vars not solved for in the block is not present in the remaining block he will not be able to solve it for the same reasons he wasnt able to solve the unpartitioned block
                    soll = try SPyPyC.solve(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
                    catch
                    end
                    
                    if !(isnothing(soll) || length(soll) == 0)
                        soll_collected = soll isa Dict ? collect(values(soll)) : collect(soll[1])
                        
                        return (vars_to_solve[setdiff(1:length(eqs_to_solve),var_combo)],
                                vars_to_solve[var_combo],
                                eqs_to_solve[setdiff(1:length(eqs_to_solve),eq_combo)],
                                soll_collected)
                    end
                end
            end
        end
    end
end


pv = sortperm(vars_to_solve, by = Symbol)
pe = sortperm(eqs_to_solve, by = string)

sort!(vars_to_solve, by = Symbol)
sort!(eqs_to_solve, by = string)

solved_system = partial_solve(eqs_to_solve, vars_to_solve, incidence_matrix_subset[pv,pe])

# !isnothing(solved_system) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "LambertW"))

# solved_system = partial_solve(eqs_to_solve, vars_to_solve, incidence_matrix_subset)

# any(contains.(string.(solved_system[4]),"LambertW"))



function make_equation_rebust_to_domain_errors(eqs::Vector{Expr}, 
    vars_to_exclude::Vector{Vector{Symbol}}, 
    bounds::Dict{Symbol,Tuple{Float64,Float64}}, 
    ➕_vars::Vector{Symbol}, 
    unique_➕_vars::Vector{Union{Symbol,Expr}}, 
    precompile::Bool = false)
ss_and_aux_equations = Expr[]
ss_and_aux_equations_dep = Expr[]
ss_and_aux_equations_error = Expr[]
ss_and_aux_equations_error_dep = Expr[]
rewritten_eqs = Expr[]
# write down ss equations including nonnegativity auxilliary variables
# find nonegative variables, parameters, or terms
for eq in eqs 
rewritten_eq = postwalk(x -> 
x isa Expr ? 
# x.head == :(=) ? 
#     Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
#         x.head == :ref ?
#             occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
#     x : 
x.head == :call ?
x.args[1] == :* ?
x.args[2] isa Int ?
x.args[3] isa Int ?
x :
Expr(:call, :*, x.args[3:end]..., x.args[2]) : # 2beta => beta * 2 
x :
x.args[1] ∈ [:^] ?
!(x.args[3] isa Int) ?
x.args[2] isa Symbol ? # nonnegative parameters 
x.args[2] ∈ vars_to_exclude[1] ?
begin
bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
x 
end :
begin
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if x.args[2] in vars_to_exclude[1]
push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end

push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end

:($(replacement) ^ $(x.args[3]))
end :
x.args[2].head == :call ? # nonnegative expressions
begin
if precompile
replacement = x.args[2]
else
replacement = simplify(x.args[2])
end

if !(replacement isa Int) # check if the nonnegative term is just a constant
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
    push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
    push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
    push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
    push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end

push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end
end

:($(replacement) ^ $(x.args[3]))
end :
x :
x :
x.args[2] isa Float64 ?
x :
x.args[1] ∈ [:log] ?
!(x.args[3] isa Int) ?
x.args[2] isa Symbol ? # nonnegative parameters 
x.args[2] ∈ vars_to_exclude[1] ?
begin
bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
x 
end :
begin
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if x.args[2] in vars_to_exclude[1]
push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) 
push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end

push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end

:($(Expr(:call, x.args[1], replacement)))
end :
x.args[2].head == :call ? # nonnegative expressions
begin
if precompile
replacement = x.args[2]
else
replacement = simplify(x.args[2])
end

if !(replacement isa Int) # check if the nonnegative term is just a constant
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
    push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
    push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
    push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
    push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end

push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end
end

:($(Expr(:call, x.args[1], replacement)))
end :
x :
x :
x.args[1] ∈ [:norminvcdf, :norminv, :qnorm] ?
x.args[2] isa Symbol ? # nonnegative parameters 
x.args[2] ∈ vars_to_exclude[1] ?
begin
bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1-eps())) : (eps(), 1 - eps())
x 
end :
begin
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if x.args[2] in vars_to_exclude[1]
push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2])))))) 
push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end

push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end

:($(Expr(:call, x.args[1], replacement)))
end :
x.args[2].head == :call ? # nonnegative expressions
begin
if precompile
replacement = x.args[2]
else
replacement = simplify(x.args[2])
end

if !(replacement isa Int) # check if the nonnegative term is just a constant
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end

push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end
end

:($(Expr(:call, x.args[1], replacement)))
end :
x :
x.args[1] ∈ [:exp] ?
x.args[2] isa Symbol ? # have exp terms bound so they dont go to Inf
x.args[2] ∈ vars_to_exclude[1] ?
begin
bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12), min(bounds[x.args[2]][2], 700)) : (-1e12, 700)
x 
end :
begin
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if x.args[2] in vars_to_exclude[1]
push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2])))))) 
push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2])))))) 
push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end

push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end

:($(Expr(:call, x.args[1], replacement)))
end :
x.args[2].head == :call ? # have exp terms bound so they dont go to Inf
begin
if precompile
replacement = x.args[2]
else
replacement = simplify(x.args[2])
end

if !(replacement isa Int) # check if the nonnegative term is just a constant
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2]))))))
push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2]))))))
push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end


push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end
end

:($(Expr(:call, x.args[1], replacement)))
end :
x :
x.args[1] ∈ [:erfcinv] ?
x.args[2] isa Symbol ? # nonnegative parameters 
x.args[2] ∈ vars_to_exclude[1] ?
begin
bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 2 - eps())) : (eps(), 2 - eps())
x 
end :
begin
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if x.args[2] in vars_to_exclude[1]
push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2])))))) 
push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end


push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end

:($(Expr(:call, x.args[1], replacement)))
end :
x.args[2].head == :call ? # nonnegative expressions
begin
if precompile
replacement = x.args[2]
else
replacement = simplify(x.args[2])
end

if !(replacement isa Int) # check if the nonnegative term is just a constant
if x.args[2] ∈ unique_➕_vars
➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
replacement = Symbol("➕" * sub(string(➕_vars_idx)))
else
push!(unique_➕_vars,x.args[2])

if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
else
push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2]))))))
push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
end

push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
replacement = Symbol("➕" * sub(string(length(➕_vars))))
end
end

:($(Expr(:call, x.args[1], replacement)))
end :
x :
x :
x :
x,
eq)
push!(rewritten_eqs,rewritten_eq)
end

vars_to_exclude_from_block = vcat(vars_to_exclude...)

found_new_dependecy = true

while found_new_dependecy
found_new_dependecy = false

for ssauxdep in ss_and_aux_equations_dep
push!(vars_to_exclude_from_block, ssauxdep.args[2])
end

for (iii, ssaux) in enumerate(ss_and_aux_equations)
if !isempty(intersect(get_symbols(ssaux), vars_to_exclude_from_block))
found_new_dependecy = true
push!(vars_to_exclude_from_block, ssaux.args[2])
push!(ss_and_aux_equations_dep, ssaux)
push!(ss_and_aux_equations_error_dep, ss_and_aux_equations_error[iii])
deleteat!(ss_and_aux_equations, iii)
deleteat!(ss_and_aux_equations_error, iii)
end
end
end

return rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep
end




bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    ➕_vars = Symbol[]
    unique_➕_vars = Union{Symbol,Expr}[]

    vars_to_exclude = [Symbol.(solved_system[1]),Symbol.(solved_system[2])]

    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_rebust_to_domain_errors(Meta.parse.(string.(solved_system[3])), vars_to_exclude, bounds, ➕_vars, unique_➕_vars)

    vars_to_exclude = [Symbol.(vcat(solved_system[1])),Symbol[]]

    rewritten_eqs2, ss_and_aux_equations2, ss_and_aux_equations_dep2, ss_and_aux_equations_error2, ss_and_aux_equations_error_dep2 = make_equation_rebust_to_domain_errors(Meta.parse.(string.(solved_system[4])), vars_to_exclude, bounds, ➕_vars, unique_➕_vars)

    push!(𝓂.solved_vars, Symbol.(vcat(solved_system[1], solved_system[2])))
    push!(𝓂.solved_vals, vcat(rewritten_eqs, rewritten_eqs2))

    syms_in_eqs = Set{Symbol}()

    for i in vcat(rewritten_eqs, rewritten_eqs2)
        push!(syms_in_eqs, get_symbols(i)...)
    end

    setdiff!(syms_in_eqs,➕_vars)

    syms_in_eqs2 = Set{Symbol}()

    for i in vcat(ss_and_aux_equations, ss_and_aux_equations2)
        push!(syms_in_eqs2, get_symbols(i)...)
    end

    union!(syms_in_eqs, intersect(syms_in_eqs2, ➕_vars))

    calib_pars = Expr[]
    calib_pars_input = Symbol[]
    relevant_pars = union(intersect(reduce(union, vcat(𝓂.par_list_aux_SS, 𝓂.par_calib_list)[eq_idx_in_block_to_solve]), syms_in_eqs),intersect(syms_in_eqs, ➕_vars))
    
    union!(relevant_pars_across, relevant_pars)

    iii = 1
    for parss in union(𝓂.parameters, 𝓂.parameters_as_function_of_parameters)
        if :($parss) ∈ relevant_pars
            push!(calib_pars, :($parss = parameters_and_solved_vars[$iii]))
            push!(calib_pars_input, :($parss))
            iii += 1
        end
    end

    guess = Expr[]
    result = Expr[]

    sorted_vars = sort(Symbol.(solved_system[1]))

    for (i, parss) in enumerate(sorted_vars) 
        push!(guess,:($parss = guess[$i]))
        push!(result,:($parss = sol[$i]))
    end

    # separate out auxilliary variables (nonnegativity)
    # nnaux = []
    # nnaux_linear = []
    # nnaux_error = []
    # push!(nnaux_error, :(aux_error = 0))
    solved_vals = Expr[]
    partially_solved_block = Expr[]

    other_vrs_eliminated_by_sympy = Set{Symbol}()

    for (i,val) in enumerate(𝓂.solved_vals[end])
        if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
            val = vcat(𝓂.ss_aux_equations, 𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
            # push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
            push!(other_vrs_eliminated_by_sympy, val.args[2])
            # push!(nnaux_linear,:($val))
            # push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
        end
    end



    for (var,val) in Dict(Symbol.(solved_system[2]) .=> rewritten_eqs2)
        push!(partially_solved_block, :($var = $(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))))
    end

    for (i,val) in enumerate(rewritten_eqs)
        push!(solved_vals, postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
    end

    # if length(nnaux) > 1
    #     all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

    #     nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
        
    #     inc_matrix = fill(0,length(all_symbols),length(all_symbols))

    #     for i in 1:length(all_symbols)
    #         for k in 1:length(nn_symbols)
    #             inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
    #         end
    #     end

    #     QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

    #     nnaux = nnaux[QQ]
    #     nnaux_linear = nnaux_linear[QQ]
    # end

    other_vars = Expr[]
    other_vars_input = Symbol[]
    other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, ➕_vars),
                                        sort(𝓂.solved_vars[end]) ),
                                union(syms_in_eqs, other_vrs_eliminated_by_sympy ) )
                                # union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

    for var in other_vrs
        push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
        push!(other_vars_input,:($(var)))
        iii += 1
    end

    solved_vals[end] = Expr(:call, :+, solved_vals[end], ss_and_aux_equations_error_dep2...)

    funcs = :(function block(parameters_and_solved_vars::Vector, guess::Vector)
            $(guess...) 
            $(calib_pars...) # add those variables which were previously solved and are used in the equations
            $(other_vars...) # take only those that appear in equations - DONE

            $(ss_and_aux_equations_dep2...)

            $(partially_solved_block...) # add those variables which were previously solved and are used in the equations

            $(ss_and_aux_equations_dep...)
            # return [$(solved_vals...),$(nnaux_linear...)]
            return [$(solved_vals...)]
        end)

    push!(NSSS_solver_cache_init_tmp,fill(0.7688,length(sorted_vars)))
    push!(NSSS_solver_cache_init_tmp,[Inf])

    # WARNING: infinite bounds are transformed to 1e12
    lbs = Float64[]
    ubs = Float64[]

    limit_boundaries = 1e12

    for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
        if haskey(bounds,i)
            push!(lbs,bounds[i][1])
            push!(ubs,bounds[i][2])
        else
            push!(lbs,-limit_boundaries)
            push!(ubs, limit_boundaries)
        end
    end

    push!(SS_solve_func,ss_and_aux_equations...)
    push!(SS_solve_func,ss_and_aux_equations2...)

    push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...), $(other_vars_input...)]))

    push!(SS_solve_func,:(lbs = [$(lbs...)]))
    push!(SS_solve_func,:(ubs = [$(ubs...)]))
            
    n_block = length(𝓂.ss_solve_blocks) + 1   
        
    push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

    if VERSION >= v"1.9"
        push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())))
    else
        push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver())))
    end

    push!(SS_solve_func,:(solution = block_solver_AD(params_and_solved_vars,
                                                            $(n_block), 
                                                            𝓂.ss_solve_blocks[$(n_block)], 
                                                            # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                            # f, 
                                                            inits,
                                                            lbs, 
                                                            ubs,
                                                            solver_parameters,
                                                            # fail_fast_solvers_only = fail_fast_solvers_only,
                                                            cold_start,
                                                            verbose)))
                                                            
    push!(SS_solve_func,:(iters += solution[2][2])) 
    push!(SS_solve_func,:(solution_error += solution[2][1])) 
    push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, ss_and_aux_equations_error..., ss_and_aux_equations_error2...))))
    push!(SS_solve_func,:(sol = solution[1]))

    push!(SS_solve_func,:($(result...)))   
    push!(SS_solve_func,:($(ss_and_aux_equations_dep2...)))  
    push!(SS_solve_func,:($(partially_solved_block...)))  

    push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
    push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))


push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))



    # Solving for: [[1, 2, 3, 4]] in: [[1, 2, 3, 4]]
    # Solving for: [[1, 2, 3, 5]] in: [[1, 2, 3, 4]]
    # Solving for: [[1, 2, 4, 5]] in: [[1, 2, 3, 4]]
    # Solving for: [[1, 3, 4, 5]] in: [[1, 2, 3, 4]]
    # Solving for: [[2, 3, 4, 5]] in: [[1, 2, 3, 4]]
    # Solving for: [[1, 2, 3, 4]] in: [[1, 2, 3, 5]]
    # Solving for: [[1, 2, 3, 5]] in: [[1, 2, 3, 5]]
    # Solving for: [[1, 2, 4, 5]] in: [[1, 2, 3, 5]]

# varc = [1, 2, 4, 5]
# eqsc = [1, 2, 3, 4]



# indices_to_select_from = findall([sum(incidence_matrix_subset[:,eqsc],dims = 2)...] .> 0)

# indices_that_need_to_be_there = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),eqsc)],dims = 2)...] .> 0) 

# # # for var_combo in combinations(indices_to_select_from, n)
#     unmatched_vars = setdiff(indices_that_need_to_be_there, varc)

# #     if length(unmatched_vars) <= length(eqs_to_solve) - n


# incidence_matrix_subset[varc,varc]
# incidence_matrix_subset[setdiff(1:size(incidence_matrix_subset,1), varc), setdiff(1:size(incidence_matrix_subset,1), eqsc)]

# soll[1] |> collect
solved_system[1]
solved_system[2]
solved_system[3]
solved_system[4]

        
# soll = solved_system[3]



# atoms = reduce(union,map(x->x.atoms(),collect(soll)))
# # println(atoms)
# [push!(atoms_in_equations, Symbol(a)) for a in atoms];

# for (k, vars) in enumerate(vars_to_solve)
#     push!(𝓂.solved_vars,Symbol(vars))
#     push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

#     push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
#     # push!(atoms_in_equations_list, Set(Symbol.(soll[vars].atoms()))) # to be fixed
#     push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
# end



function write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve)
    push!(𝓂.solved_vars,Symbol.(vars_to_solve))
    push!(𝓂.solved_vals,Meta.parse.(string.(eqs_to_solve)))

    syms_in_eqs = Set()

    for i in eqs_to_solve
        push!(syms_in_eqs, Symbol.(SPyPyC.:↓(SPyPyC.free_symbols(i)))...)
    end

    push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

    calib_pars = []
    calib_pars_input = []
    relevant_pars = reduce(union,vcat(𝓂.par_list_aux_SS,𝓂.par_calib_list)[eq_idx_in_block_to_solve])
    relevant_pars_across = union(relevant_pars_across,relevant_pars)
    
    iii = 1
    for parss in union(𝓂.parameters,𝓂.parameters_as_function_of_parameters)
        if :($parss) ∈ relevant_pars
            push!(calib_pars,:($parss = parameters_and_solved_vars[$iii]))
            push!(calib_pars_input,:($parss))
            iii += 1
        end
    end

    guess = []
    result = []
    sorted_vars = sort(𝓂.solved_vars[end])
    for (i, parss) in enumerate(sorted_vars) 
        push!(guess,:($parss = guess[$i]))
        push!(result,:($parss = sol[$i]))
    end


    # separate out auxilliary variables (nonnegativity)
    nnaux = []
    nnaux_linear = []
    nnaux_error = []
    push!(nnaux_error, :(aux_error = 0))
    solved_vals = []
    
    other_vrs_eliminated_by_sympy = Set()

    for (i,val) in enumerate(𝓂.solved_vals[end])
        if typeof(val) ∈ [Symbol,Float64,Int]
            push!(solved_vals,val)
        else
            if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
                val = vcat(𝓂.ss_aux_equations,𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
                push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                push!(other_vrs_eliminated_by_sympy, val.args[2])
                push!(nnaux_linear,:($val))
                push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
            else
                push!(solved_vals,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
            end
        end
    end

    # sort nnaux vars so that they enter in right order. avoid using a variable before it is declared
    if length(nnaux) > 1
        all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

        nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
        
        inc_matrix = fill(0,length(all_symbols),length(all_symbols))

        for i in 1:length(all_symbols)
            for k in 1:length(nn_symbols)
                inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
            end
        end

        QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

        nnaux = nnaux[QQ]
        nnaux_linear = nnaux_linear[QQ]
    end

    other_vars = []
    other_vars_input = []
    other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars),
                                        sort(𝓂.solved_vars[end]) ),
                            union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

    for var in other_vrs
        push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
        push!(other_vars_input,:($(var)))
        iii += 1
    end

    funcs = :(function block(parameters_and_solved_vars::Vector, guess::Vector)
            $(guess...) 
            $(calib_pars...) # add those variables which were previously solved and are used in the equations
            $(other_vars...) # take only those that appear in equations - DONE
            
            return [$(solved_vals...),$(nnaux_linear...)]
        end)

    push!(NSSS_solver_cache_init_tmp,fill(0.7688,length(sorted_vars)))
    push!(NSSS_solver_cache_init_tmp,[Inf])

    # WARNING: infinite bounds are transformed to 1e12
    lbs = []
    ubs = []
    
    limit_boundaries = 1e12

    for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
        if i ∈ 𝓂.bounded_vars
            push!(lbs,𝓂.lower_bounds[i .== 𝓂.bounded_vars][1] == -Inf ? -limit_boundaries+rand() : 𝓂.lower_bounds[i .== 𝓂.bounded_vars][1])
            push!(ubs,𝓂.upper_bounds[i .== 𝓂.bounded_vars][1] ==  Inf ?  limit_boundaries-rand() : 𝓂.upper_bounds[i .== 𝓂.bounded_vars][1])
        else
            push!(lbs,-limit_boundaries+rand())
            push!(ubs,limit_boundaries+rand())
        end
    end

    push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...),$(other_vars_input...)]))

    push!(SS_solve_func,:(lbs = [$(lbs...)]))
    push!(SS_solve_func,:(ubs = [$(ubs...)]))

    n_block = length(𝓂.ss_solve_blocks) + 1

    push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

    if VERSION >= v"1.9"
        push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())))
    else
        push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver())))
    end

    push!(SS_solve_func,:(solution = block_solver_AD(params_and_solved_vars,
                                                            $(n_block), 
                                                            𝓂.ss_solve_blocks[$(n_block)], 
                                                            # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                            # f, 
                                                            inits,
                                                            lbs, 
                                                            ubs,
                                                            solver_parameters,
                                                            # fail_fast_solvers_only = fail_fast_solvers_only,
                                                            cold_start,
                                                            verbose)))
    
    push!(SS_solve_func,:(iters += solution[2][2])) 
    push!(SS_solve_func,:(solution_error += solution[2][1])) 
    push!(SS_solve_func,:(sol = solution[1]))
    
    push!(SS_solve_func,:($(result...)))   
    
    push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
    push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

    push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
end




# if symbolic_SS
# soll = try SPyPyC.solve(eqs_to_solve,vars_to_solve)
# # soll = try solve(SPyPyC.Sym(eqs_to_solve),var_order)#,check=false,force = true,manual=true)
# catch
# end

# if isnothing(soll)
#     if verbose
#         println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
#     end
#     numerical_sol = true
#     # continue
# elseif length(soll) == 0
#     if verbose
#         println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
#     end
#     numerical_sol = true
#     # continue
# # elseif length(intersect(vars_to_solve,reduce(union,map(x->x.atoms(),collect(soll[1]))))) > 0
# elseif length(intersect((union(SPyPyC.free_symbols.(soll[1])...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
# # elseif length(intersect(union(SPyPyC.free_symbols.(soll[1])...),vars_to_solve)) > 0
#     if verbose
#         println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
#     end
#     numerical_sol = true
#     # println("Could not solve for: ",intersect(var_list,reduce(union,map(x->x.atoms(),solll)))...)
#     # break_ind = true
#     # break
# else
#     if verbose
#         println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.")
#     end
#     # relevant_pars = reduce(union,vcat(𝓂.par_list,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
#     # relevant_pars = reduce(union,map(x->x.atoms(),collect(soll[1])))
#     atoms = reduce(union,map(x->x.atoms(),collect(soll[1])))
#     # println(atoms)
#     [push!(atoms_in_equations, Symbol(a)) for a in atoms]
    
#     for (k, vars) in enumerate(vars_to_solve)
#         push!(𝓂.solved_vars,Symbol(vars))
#         push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

#         push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
#         # push!(atoms_in_equations_list, Set(Symbol.(soll[vars].atoms()))) # to be fixed
#         push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
#     end
# end

n = nb
# end
            
        # try symbolically and use numerical if it does not work
        # if numerical_sol || !symbolic_SS
            # if !symbolic_SS && verbose
            #     println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
            # end
            push!(𝓂.solved_vars,Symbol.(vcat(solved_system[1],solved_system[2])))
            push!(𝓂.solved_vals,Meta.parse.(string.(vcat(solved_system[3],solved_system[4]))))

            syms_in_eqs = Set()

            for i in eqs_to_solve
                # push!(syms_in_eqs, Symbol.(PythonCall.pystr.(i.atoms()))...)
                push!(syms_in_eqs, Symbol.(SPyPyC.:↓(SPyPyC.free_symbols(i)))...)
            end

            push!(atoms_in_equations_list,setdiff(syms_in_eqs, 𝓂.solved_vars[end]))

            calib_pars = []
            calib_pars_input = []
            relevant_pars = reduce(union,vcat(𝓂.par_list_aux_SS,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
            relevant_pars_across = union(relevant_pars_across,relevant_pars)
            
            iii = 1
            for parss in union(𝓂.parameters,𝓂.parameters_as_function_of_parameters)
                # valss   = 𝓂.parameter_values[i]
                if :($parss) ∈ relevant_pars
                    push!(calib_pars,:($parss = parameters_and_solved_vars[$iii]))
                    push!(calib_pars_input,:($parss))
                    iii += 1
                end
            end

            guess = []
            result = []
            sorted_vars = sort(solved_system[1])
            # sorted_vars = sort(setdiff(𝓂.solved_vars[end],𝓂.➕_vars))
            for (i, parss) in enumerate(sorted_vars) 
                push!(guess,:($parss = guess[$i]))
                # push!(guess,:($parss = undo_transformer(guess[$i])))
                push!(result,:($parss = sol[$i]))
            end
            
            # separate out auxilliary variables (nonnegativity)
            nnaux = []
            nnaux_linear = []
            nnaux_error = []
            push!(nnaux_error, :(aux_error = 0))
            solved_vals = []
            partially_solved_block = []

            eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

            other_vrs_eliminated_by_sympy = Set()

            for (i,val) in enumerate(𝓂.solved_vals[end])
                if eq_idx_in_block_to_solve[i] ∈ 𝓂.ss_equations_with_aux_variables
                    val = vcat(𝓂.ss_aux_equations,𝓂.calibration_equations)[eq_idx_in_block_to_solve[i]]
                    push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                    push!(other_vrs_eliminated_by_sympy, val.args[2])
                    push!(nnaux_linear,:($val))
                    push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
                end
            end

            for (var,val) in Dict(solved_system[2] .=> solved_system[4])
                push!(partially_solved_block,:($var = $(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))))
            end

            for (i,val) in enumerate(solved_system[3])
                push!(solved_vals,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
            end

            if length(nnaux) > 1
                all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

                nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
                
                inc_matrix = fill(0,length(all_symbols),length(all_symbols))

                for i in 1:length(all_symbols)
                    for k in 1:length(nn_symbols)
                        inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
                    end
                end

                QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

                nnaux = nnaux[QQ]
                nnaux_linear = nnaux_linear[QQ]
            end

            other_vars = []
            other_vars_input = []
            # other_vars_inverse = []
            other_vrs = intersect( setdiff( union(𝓂.var, 𝓂.calibration_equations_parameters, 𝓂.➕_vars),
                                                sort(𝓂.solved_vars[end]) ),
                                    union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

            for var in other_vrs
                # var_idx = findfirst(x -> x == var, union(𝓂.var,𝓂.calibration_equations_parameters))
                push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
                push!(other_vars_input,:($(var)))
                iii += 1
                # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
            end

            funcs = :(function block(parameters_and_solved_vars::Vector, guess::Vector)
                    $(guess...) 
                    $(calib_pars...) # add those variables which were previously solved and are used in the equations
                    $(other_vars...) # take only those that appear in equations - DONE

                    $(partially_solved_block...) # add those variables which were previously solved and are used in the equations

                    return [$(solved_vals...),$(nnaux_linear...)]
                end)

            push!(NSSS_solver_cache_init_tmp,fill(0.7688,length(sorted_vars)))
            push!(NSSS_solver_cache_init_tmp,[Inf])

            # WARNING: infinite bounds are transformed to 1e12
            lbs = []
            ubs = []
            
            limit_boundaries = 1e12

            for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
                if i ∈ 𝓂.bounded_vars
                    push!(lbs,𝓂.lower_bounds[i .== 𝓂.bounded_vars][1] == -Inf ? -limit_boundaries+rand() : 𝓂.lower_bounds[i .== 𝓂.bounded_vars][1])
                    push!(ubs,𝓂.upper_bounds[i .== 𝓂.bounded_vars][1] ==  Inf ?  limit_boundaries-rand() : 𝓂.upper_bounds[i .== 𝓂.bounded_vars][1])
                else
                    push!(lbs,-limit_boundaries+rand())
                    push!(ubs,limit_boundaries+rand())
                end
            end

            push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...),$(other_vars_input...)]))

            push!(SS_solve_func,:(lbs = [$(lbs...)]))
            push!(SS_solve_func,:(ubs = [$(ubs...)]))
                            
            push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

            if VERSION >= v"1.9"
                push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())))
            else
                push!(SS_solve_func,:(block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[$(n_block)]; linear_solver = ℐ.DirectLinearSolver())))
            end

            push!(SS_solve_func,:(solution = block_solver_AD(params_and_solved_vars,
                                                                    $(n_block), 
                                                                    𝓂.ss_solve_blocks[$(n_block)], 
                                                                    # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                                    # f, 
                                                                    inits,
                                                                    lbs, 
                                                                    ubs,
                                                                    solver_parameters,
                                                                    # fail_fast_solvers_only = fail_fast_solvers_only,
                                                                    cold_start,
                                                                    verbose)))
                                                                    
            push!(SS_solve_func,:(iters += solution[2][2])) 
            push!(SS_solve_func,:(solution_error += solution[2][1])) 
            push!(SS_solve_func,:(sol = solution[1]))
            
            push!(SS_solve_func,:($(result...)))   
            push!(SS_solve_func,:($(partially_solved_block...)))  
            
            push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
            push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

            push!(𝓂.ss_solve_blocks,@RuntimeGeneratedFunction(funcs))
            
            n_block += 1
#         end
#     end
#     n -= 1
# end





import ImplicitDifferentiation as ℐ
import MacroModelling: block_solver, solver_parameters, levenberg_marquardt
import AbstractDifferentiation as 𝒜
import ForwardDiff as ℱ
# import Diffractor: DiffractorForwardBackend
𝒷 = 𝒜.ForwardDiffBackend

solver_params = solver_parameters(eps(), eps(), 100, 2.9912988764832833, 0.8725, 0.0027, 0.028948770826150612, 8.04, 4.076413176215408, 0.06375413238034794, 0.24284340766769424, 0.5634017580097571, 0.009549630552246828, 0.6342888355132347, 0.5275522227754195, 1.0, 0.06178989216048817, 0.5234277812131813, 0.422, 0.011209254402846185, 0.5047, 0.6020757011698457, 1, 0.0, 2)

cold_start = true
verbose = true
parameters = 𝓂.parameter_values


alp = parameters[1]
bet = parameters[2]
gam = parameters[3]
mst = parameters[4]
psi = parameters[6]
del = parameters[7]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3240 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3241 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3242 =#
NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3243 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3244 =#
iters = 0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3245 =#
m = mst
e = 1.0
➕₃ = min(700, max(-1.0e12, -alp * gam))
➕₄ = min(700, max(-1.0e12, gam))
params_and_solved_vars = [alp, bet, psi, del, m, ➕₃, ➕₄]
lbs = [-1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12]
ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
inits = [max.(lbs[1:(length(lbs)-length(params_and_solved_vars))], min.(ubs[1:(length(lbs)-length(params_and_solved_vars))], .37)), params_and_solved_vars]

𝓂.ss_solve_blocks[1](params_and_solved_vars,[0.8610478832599567])

function ss_solve_blocks(parameters_and_solved_vars, guess)
l = guess[1]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2888 =#
alp = parameters_and_solved_vars[1]
bet = parameters_and_solved_vars[2]
psi = parameters_and_solved_vars[3]
del = parameters_and_solved_vars[4]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2889 =#
m = parameters_and_solved_vars[5]
➕₃ = parameters_and_solved_vars[6]
➕₄ = parameters_and_solved_vars[7]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2891 =#
➕₅ = min(1.0e12, max(eps(), (l * (1.0 - psi)) / (-l * psi + l + m * psi)))
➕₆ = min(1.0e12, max(eps(), ➕₅ ^ (1.0 - alp) / ((del + exp(➕₄)) - 1.0)))
➕₇ = min(1.0e12, max(eps(), ➕₆ ^ (-1 / (alp - 1.0))))
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2893 =#
P = (m * ➕₆ ^ (1 / (alp - 1))) / (((-del + ➕₆ ^ (1 / (alp - 1.0)) * ➕₅ ^ (1.0 - alp) * ➕₇ ^ alp) - exp(➕₄)) + 1.0)
n = (l * (psi - 1.0)) / ((l * psi - l) - m * psi)
k = exp(➕₄) / ➕₆ ^ (1 / (alp - 1))
c = (((-del + ➕₆ ^ (1 / (alp - 1)) * ➕₅ ^ (1 - alp) * ➕₇ ^ alp) - exp(➕₄)) + 1.0) / ➕₆ ^ (1 / (alp - 1))
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2895 =#
➕₁ = min(1.0e12, max(eps(), k))
➕₂ = min(1.0e12, max(eps(), n))
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2897 =#
return [((-bet * ➕₁ ^ alp * ➕₂ ^ (1 - alp) * (1 - alp) * exp(➕₃)) / (l * m) + 1 / P) + abs(➕₅ - (l * (1.0 - psi)) / (-l * psi + l + m * psi)) + abs(➕₆ - ➕₅ ^ (1.0 - alp) / ((del + exp(➕₄)) - 1.0)) + abs(➕₇ - ➕₆ ^ (-1 / (alp - 1.0)))]
end

ss_solve_blocks(params_and_solved_vars, [.37])


parameters_and_solved_vars = params_and_solved_vars

l = 0.8610478832599567
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2888 =#
alp = parameters_and_solved_vars[1]
bet = parameters_and_solved_vars[2]
psi = parameters_and_solved_vars[3]
del = parameters_and_solved_vars[4]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2889 =#
m = parameters_and_solved_vars[5]
➕₃ = parameters_and_solved_vars[6]
➕₄ = parameters_and_solved_vars[7]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2891 =#
➕₅ = min(1.0e12, max(eps(), (l * (1.0 - psi)) / (-l * psi + l + m * psi)))
➕₆ = min(1.0e12, max(eps(), ➕₅ ^ (1.0 - alp) / ((del + exp(➕₄)) - 1.0)))
➕₇ = min(1.0e12, max(eps(), ➕₆ ^ (-1 / (alp - 1.0))))
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2893 =#
P = (m * ➕₆ ^ (1 / (alp - 1))) / (((-del + ➕₆ ^ (1 / (alp - 1.0)) * ➕₅ ^ (1.0 - alp) * ➕₇ ^ alp) - exp(➕₄)) + 1.0)
n = (l * (psi - 1.0)) / ((l * psi - l) - m * psi)
k = exp(➕₄) / ➕₆ ^ (1 / (alp - 1))
c = (((-del + ➕₆ ^ (1 / (alp - 1)) * ➕₅ ^ (1 - alp) * ➕₇ ^ alp) - exp(➕₄)) + 1.0) / ➕₆ ^ (1 / (alp - 1))
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2895 =#
➕₁ = min(1.0e12, max(eps(), k))
➕₂ = min(1.0e12, max(eps(), n))

((-bet * ➕₁ ^ alp * ➕₂ ^ (1 - alp) * (1 - alp) * exp(➕₃)) / (l * m) + 1 / P)



block_solver_AD = ℐ.ImplicitFunction(block_solver, ss_solve_blocks; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷());
solution = block_solver_AD(params_and_solved_vars, 1, ss_solve_blocks, inits, lbs, ubs, solver_params, cold_start, verbose)


block_solver_AD = ℐ.ImplicitFunction(block_solver, 𝓂.ss_solve_blocks[1]; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷());
solution = block_solver_AD(params_and_solved_vars, 1, 𝓂.ss_solve_blocks[1], inits, lbs, ubs, solver_params, cold_start, verbose)
iters += (solution[2])[2]
solution_error += (solution[2])[1]
solution_error += abs(➕₃ - -alp * gam) + abs(➕₄ - gam)
sol = solution[1]
l = sol[1]
➕₅ = min(1.0e12, max(eps(), (l * (1.0 - psi)) / (-l * psi + l + m * psi)))
➕₆ = min(1.0e12, max(eps(), ➕₅ ^ (1.0 - alp) / ((del + exp(➕₄)) - 1.0)))
➕₇ = min(1.0e12, max(eps(), ➕₆ ^ (-1 / (alp - 1.0))))
P = (m * ➕₆ ^ (1 / (alp - 1))) / (((-del + ➕₆ ^ (1 / (alp - 1.0)) * ➕₅ ^ (1.0 - alp) * ➕₇ ^ alp) - exp(➕₄)) + 1.0)
n = (l * (psi - 1.0)) / ((l * psi - l) - m * psi)
k = exp(➕₄) / ➕₆ ^ (1 / (alp - 1))
c = (((-del + ➕₆ ^ (1 / (alp - 1)) * ➕₅ ^ (1 - alp) * ➕₇ ^ alp) - exp(➕₄)) + 1.0) / ➕₆ ^ (1 / (alp - 1))





alp = parameters[1]
bet = parameters[2]
gam = parameters[3]
mst = parameters[4]
psi = parameters[6]
del = parameters[7]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2608 =#
mst = min(max(mst, 1.1920928955078125e-7), 1.0000000000009352e12)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2609 =#
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2610 =#
NSSS_solver_cache_tmp = []
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2611 =#
solution_error = 0.0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2612 =#
iters = 0
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2613 =#
➕₄ = min(max(-9.999999999998646e11, -alp * gam), 700.0)
➕₅ = min(max(-9.999999999996294e11, -gam), 700.0)
e = 1.0
➕₂ = min(max(-9.999999999994171e11, -alp * gam), 700.0)
➕₃ = min(max(-9.999999999999353e11, -gam), 700.0)
m = mst

function ss_solve_block(parameters_and_solved_vars::Vector, guess::Vector)
    n = guess[1]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2329 =#
    alp = parameters_and_solved_vars[1]
    bet = parameters_and_solved_vars[2]
    gam = parameters_and_solved_vars[3]
    psi = parameters_and_solved_vars[4]
    del = parameters_and_solved_vars[5]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2330 =#
    m = parameters_and_solved_vars[6]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2332 =#
    l = (m * n * psi) / ((n - 1.0) * (psi - 1.0))
    P = (m ^ 2 * n ^ alp * psi * exp(alp * gam)) / (bet * ((-alp - n * psi * (alp - 1.0)) + n * (alp - 1.0) + psi * (alp - 1.0) + 1.0) * (((n ^ (alp - 1.0) * (bet * (del - 1.0) + exp(gam)) * exp(gam * (alp - 1.0))) / (alp * bet)) ^ (1 / (alp - 1.0))) ^ alp)
    k = ((n ^ (alp - 1.0) * ((bet * del - bet) + exp(gam)) * exp(gam * (alp - 1.0))) / (alp * bet)) ^ (1 / (alp - 1))
    c = (bet * ((-alp - n * psi * (alp - 1.0)) + n * (alp - 1.0) + psi * (alp - 1.0) + 1.0) * (((n ^ (alp - 1.0) * (bet * (del - 1.0) + exp(gam)) * exp(gam * (alp - 1.0))) / (alp * bet)) ^ (1 / (alp - 1.0))) ^ alp * exp(-alp * gam)) / (m * n ^ alp * psi)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2334 =#
    return [((c - k * (1 - del) * exp(-gam)) + k) - k ^ alp * n ^ (1 - alp) * exp(-alp * gam)]
end

params_and_solved_vars = [alp, bet, gam, psi, del, m]#, ➕₂, ➕₃, ➕₄, ➕₅]
ss_solve_block(params_and_solved_vars, [-0.316729])


SS(FS2000, derivatives  = false)(:l)

import ImplicitDifferentiation as ℐ
import MacroModelling: block_solver, solver_parameters, levenberg_marquardt
import AbstractDifferentiation as 𝒜
import ForwardDiff as ℱ
# import Diffractor: DiffractorForwardBackend
𝒷 = 𝒜.ForwardDiffBackend

solver_params = solver_parameters(eps(), eps(), 100, 2.9912988764832833, 0.8725, 0.0027, 0.028948770826150612, 8.04, 4.076413176215408, 0.06375413238034794, 0.24284340766769424, 0.5634017580097571, 0.009549630552246828, 0.6342888355132347, 0.5275522227754195, 1.0, 0.06178989216048817, 0.5234277812131813, 0.422, 0.011209254402846185, 0.5047, 0.6020757011698457, 1, 0.0, 2)

guess = [.89]

params_and_solved_vars = [alp, bet, psi, del, m, ➕₂, ➕₃, ➕₄, ➕₅]

# block_solver(params_and_solved_vars, guess)

lbs = [-1.1920928955078125e7, -9.999999999992286e11, 1.1920928955078125e-7, -9.999999999991228e11, 1.1920928955078125e-7, -9.999999999995709e11, -9.999999999995663e11, -9.999999999997446e11, -9.999999999991472e11, 1.1920928955078125e-7]
ubs = [1.0000000000005587e12, 1.0000000000006997e12, 1.0000000000006309e12, 1.0000000000009012e12, 1.0000000000004235e12, 1.0000000000004601e12, 1.0000000000009324e12, 1.0000000000003635e12, 1.0000000000004766e12, 1.0000000000004362e12]
inits = [max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], guess)), params_and_solved_vars]
block_solver_AD = ℐ.ImplicitFunction(block_solver, ss_solve_block; linear_solver = ℐ.DirectLinearSolver(), conditions_backend = 𝒷())
solution = block_solver_AD(params_and_solved_vars, 1, ss_solve_block, inits, lbs, ubs, solver_params, false, false)

@benchmark solution = block_solver_AD(params_and_solved_vars, 1, ss_solve_block, inits, lbs, ubs, solver_params, false, false)

ss_solve_block(params_and_solved_vars, guess)

using BenchmarkTools
l = guess[1]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/write_up_symbolic_solve.jl:459 =#
alp = parameters_and_solved_vars[1]
bet = parameters_and_solved_vars[2]
psi = parameters_and_solved_vars[3]
del = parameters_and_solved_vars[4]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/write_up_symbolic_solve.jl:460 =#
m = parameters_and_solved_vars[5]
➕₂ = parameters_and_solved_vars[6]
➕₃ = parameters_and_solved_vars[7]
➕₄ = parameters_and_solved_vars[8]
➕₅ = parameters_and_solved_vars[9]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/write_up_symbolic_solve.jl:462 =#
n = l*(psi - 1)/(l*psi - l - m*psi)

P = -m/(del*((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) - (-l*(psi - 1)/(-l*psi + l + m*psi))^(1 - alp)*(((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1)))^alp*exp(➕₄) - ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) + ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1)))

c = -del*((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) + (-l*(psi - 1)/(-l*psi + l + m*psi))^(1 - alp)*(((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1)))^alp*exp(➕₄) + ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) - ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))

k = ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))

#= /Users/thorekockerols/GitHub/MacroModelling.jl/src/write_up_symbolic_solve.jl:464 =#
[P*bet*k^alp*n^(1 - alp)*(1 - alp)*exp(➕₄)/m]


l = P*bet*k^alp*n^(1 - alp)*(1 - alp)*exp(➕₄)/m

n = l*(psi - 1)/(l*psi - l - m*psi)

P = -m/(del*((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) - (-l*(psi - 1)/(-l*psi + l + m*psi))^(1 - alp)*(((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1)))^alp*exp(➕₄) - ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) + ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1)))

c = -del*((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) + (-l*(psi - 1)/(-l*psi + l + m*psi))^(1 - alp)*(((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1)))^alp*exp(➕₄) + ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) - ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))

k = ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))






tbp = :(P = -m/(del*((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) - (-l*(psi - 1)/(-l*psi + l + m*psi))^(1 - alp)*(((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1)))^alp*exp(➕₄) - ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))*exp(➕₅) + ((-l*(psi - 1)/(-l*psi + l + m*psi))^(alp - 1)*(bet*(del - 1)*exp(➕₃) + 1)*exp(-➕₂)/(alp*bet))^(1/(alp - 1))))


tbp = :((m ^ 2 * n ^ alp * psi * exp(alp * gam)) / (bet * ((-alp - n * psi * (alp - 1.0)) + n * (alp - 1.0) + psi * (alp - 1.0) + 1.0) * (((n ^ (alp - 1.0) * (bet * (del - 1.0) + exp(gam)) * exp(gam * (alp - 1.0))) / (alp * bet)) ^ (1 / (alp - 1.0))) ^ alp))

tbp = [
:((m * n * psi) / ((n - 1.0) * (psi - 1.0)))
:((m ^ 2 * n ^ alp * psi * exp(alp * gam)) / (bet * ((-alp - n * psi * (alp - 1.0)) + n * (alp - 1.0) + psi * (alp - 1.0) + 1.0) * (((n ^ (alp - 1.0) * (bet * (del - 1.0) + exp(gam)) * exp(gam * (alp - 1.0))) / (alp * bet)) ^ (1 / (alp - 1.0))) ^ alp))
:(((n ^ (alp - 1.0) * ((bet * del - bet) + exp(gam)) * exp(gam * (alp - 1.0))) / (alp * bet)) ^ (1 / (alp - 1)))
:((bet * ((-alp - n * psi * (alp - 1.0)) + n * (alp - 1.0) + psi * (alp - 1.0) + 1.0) * (((n ^ (alp - 1.0) * (bet * (del - 1.0) + exp(gam)) * exp(gam * (alp - 1.0))) / (alp * bet)) ^ (1 / (alp - 1.0))) ^ alp * exp(-alp * gam)) / (m * n ^ alp * psi))
]



precomp = false


function make_equation_rebust_to_domain_errors(eqs::Vector{Expr}, 
                                                vars_to_exclude::Vector{Vector{Symbol}}, 
                                                bounds::Dict{Symbol,Tuple{Float64,Float64}}, 
                                                ➕_vars::Vector{Symbol}, 
                                                unique_➕_vars::Vector{Union{Symbol,Expr}}, 
                                                precompile::Bool = false)
    ss_and_aux_equations = Expr[]
    ss_and_aux_equations_dep = Expr[]
    ss_and_aux_equations_error = Expr[]
    ss_and_aux_equations_error_dep = Expr[]
    rewritten_eqs = Expr[]
    # write down ss equations including nonnegativity auxilliary variables
    # find nonegative variables, parameters, or terms
    for eq in eqs 
        rewritten_eq = postwalk(x -> 
            x isa Expr ? 
                # x.head == :(=) ? 
                #     Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                #         x.head == :ref ?
                #             occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
                #     x : 
                x.head == :call ?
                    x.args[1] == :* ?
                        x.args[2] isa Int ?
                            x.args[3] isa Int ?
                                x :
                            Expr(:call, :*, x.args[3:end]..., x.args[2]) : # 2beta => beta * 2 
                        x :
                    x.args[1] ∈ [:^] ?
                        !(x.args[3] isa Int) ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                x.args[2] ∈ vars_to_exclude[1] ?
                                    begin
                                        bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
                                        x 
                                    end :
                                begin
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                    else
                                        push!(unique_➕_vars,x.args[2])

                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end

                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                    end

                                    :($(replacement) ^ $(x.args[3]))
                                end :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if x.args[2] ∈ unique_➕_vars
                                            ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                            replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                        else
                                            push!(unique_➕_vars,x.args[2])

                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end

                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                        end
                                    end

                                    :($(replacement) ^ $(x.args[3]))
                                end :
                            x :
                        x :
                    x.args[2] isa Float64 ?
                        x :
                    x.args[1] ∈ [:log] ?
                        !(x.args[3] isa Int) ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                x.args[2] ∈ vars_to_exclude[1] ?
                                    begin
                                        bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
                                        x 
                                    end :
                                begin
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                    else
                                        push!(unique_➕_vars,x.args[2])

                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) 
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end

                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if x.args[2] ∈ unique_➕_vars
                                            ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                            replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                        else
                                            push!(unique_➕_vars,x.args[2])

                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2]))))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end

                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x :
                    x.args[1] ∈ [:norminvcdf, :norminv, :qnorm] ?
                        x.args[2] isa Symbol ? # nonnegative parameters 
                            x.args[2] ∈ vars_to_exclude[1] ?
                            begin
                                bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1-eps())) : (eps(), 1 - eps())
                                x 
                            end :
                            begin
                                if x.args[2] ∈ unique_➕_vars
                                    ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                    replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                else
                                    push!(unique_➕_vars,x.args[2])

                                    if x.args[2] in vars_to_exclude[1]
                                        push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2]))))))
                                        push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    else
                                        push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2])))))) 
                                        push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    end

                                    push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                    replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                end
                            
                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x.args[2].head == :call ? # nonnegative expressions
                            begin
                                if precompile
                                    replacement = x.args[2]
                                else
                                    replacement = simplify(x.args[2])
                                end

                                if !(replacement isa Int) # check if the nonnegative term is just a constant
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                    else
                                        push!(unique_➕_vars,x.args[2])

                                        if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                            push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end

                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                    end
                                end

                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x :
                    x.args[1] ∈ [:exp] ?
                        x.args[2] isa Symbol ? # have exp terms bound so they dont go to Inf
                            x.args[2] ∈ vars_to_exclude[1] ?
                            begin
                                bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12), min(bounds[x.args[2]][2], 700)) : (-1e12, 700)
                                x 
                            end :
                            begin
                                if x.args[2] ∈ unique_➕_vars
                                    ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                    replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                else
                                    push!(unique_➕_vars,x.args[2])
                                    
                                    if x.args[2] in vars_to_exclude[1]
                                        push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2])))))) 
                                        push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    else
                                        push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2])))))) 
                                        push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    end
                                    
                                    push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                    replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                end
                            
                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x.args[2].head == :call ? # have exp terms bound so they dont go to Inf
                            begin
                                if precompile
                                    replacement = x.args[2]
                                else
                                    replacement = simplify(x.args[2])
                                end

                                if !(replacement isa Int) # check if the nonnegative term is just a constant
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                    else
                                        push!(unique_➕_vars,x.args[2])
                                        
                                        if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                            push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end

                                        
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                    end
                                end

                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x :
                    x.args[1] ∈ [:erfcinv] ?
                        x.args[2] isa Symbol ? # nonnegative parameters 
                            x.args[2] ∈ vars_to_exclude[1] ?
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 2 - eps())) : (eps(), 2 - eps())
                                    x 
                                end :
                            begin
                                if x.args[2] ∈ unique_➕_vars
                                    ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                    replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                else
                                    push!(unique_➕_vars,x.args[2])

                                    if x.args[2] in vars_to_exclude[1]
                                        push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2]))))))
                                        push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    else
                                        push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2])))))) 
                                        push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    end

                                    
                                    push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                    replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                end
                            
                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x.args[2].head == :call ? # nonnegative expressions
                            begin
                                if precompile
                                    replacement = x.args[2]
                                else
                                    replacement = simplify(x.args[2])
                                end

                                if !(replacement isa Int) # check if the nonnegative term is just a constant
                                    if x.args[2] ∈ unique_➕_vars
                                        ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                        replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                    else
                                        push!(unique_➕_vars,x.args[2])

                                        if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                            push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations_dep, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2]))))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end
                                        
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                    end
                                end

                                :($(Expr(:call, x.args[1], replacement)))
                            end :
                        x :
                    x :
                x :
            x,
        eq)
        push!(rewritten_eqs,rewritten_eq)
    end

    vars_to_exclude_from_block = vcat(vars_to_exclude...)

    found_new_dependecy = true

    while found_new_dependecy
        found_new_dependecy = false

        for ssauxdep in ss_and_aux_equations_dep
            push!(vars_to_exclude_from_block, ssauxdep.args[2])
        end

        for (iii, ssaux) in enumerate(ss_and_aux_equations)
            if !isempty(intersect(get_symbols(ssaux), vars_to_exclude_from_block))
                found_new_dependecy = true
                push!(vars_to_exclude_from_block, ssaux.args[2])
                push!(ss_and_aux_equations_dep, ssaux)
                push!(ss_and_aux_equations_error_dep, ss_and_aux_equations_error[iii])
                deleteat!(ss_and_aux_equations, iii)
                deleteat!(ss_and_aux_equations_error, iii)
            end
        end
    end

    return rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep
end
# ss_and_aux_equations[1].args[2]

bounds = Dict{Symbol,Tuple{Float64,Float64}}()
➕_vars = Symbol[]
unique_➕_vars = Union{Symbol,Expr}[]
vars_to_exclude = [Symbol.(vcat(solved_system[1])),Symbol.(vcat(solved_system[2]))]

rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_rebust_to_domain_errors(Meta.parse.(string.(solved_system[3])), vars_to_exclude, bounds, ➕_vars, unique_➕_vars)


vars_to_exclude = [Symbol.(vcat(solved_system[1])),Symbol[]]

rewritten_eqs2, ss_and_aux_equations2, ss_and_aux_equations_dep2, ss_and_aux_equations_error2, ss_and_aux_equations_error_dep2 = make_equation_rebust_to_domain_errors(Meta.parse.(string.(solved_system[4])), vars_to_exclude, bounds, ➕_vars, unique_➕_vars)

















bounds = []
lbs = []
ubs = []
➕_vars = []
unique_➕_vars = []

rewritten_eq, ss_and_aux_equations, ss_and_aux_equations_error = make_equation_rebust_to_domain_errors(tbp[2], [:n], bounds, lbs, ubs, ➕_vars, unique_➕_vars)


ss_and_aux_equations

cleaned_eqs

Expr(:call,:(=), :➕ₑ, Expr(:call,:+, ss_and_aux_equations_error...))

push!(ss_and_aux_equations,unblock(eqs))








eqs = postwalk(x -> 
    x isa Expr ? 
        # x.head == :(=) ? 
        #     Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
        #         x.head == :ref ?
        #             occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
        #     x : 
        x.head == :call ?
            x.args[1] == :* ?
                x.args[2] isa Int ?
                    x.args[3] isa Int ?
                        x :
                    Expr(:call, :*, x.args[3:end]..., x.args[2]) : # 2beta => beta * 2 
                x :
            x.args[1] ∈ [:^] ?
                !(x.args[3] isa Int) ?
                    x.args[2] isa Symbol ? # nonnegative parameters 
                            begin
                                if x.args[2] ∈ unique_➕_vars
                                    ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                    replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                else
                                    push!(unique_➕_vars,x.args[2])
                                    # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                    # push!(lower_bounds,eps(Float32))
                                    # push!(upper_bounds,1e12+rand())
                                    push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the cond_var_decomp
                                    push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    # push!(ss_eq_aux_ind,length(ss_and_aux_equations))

                                    # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                    replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                end

                                :($(replacement) ^ $(x.args[3]))
                            end :
                        x :
                    x.args[2].head == :ref ?
                        x :
                    x.args[2].head == :call ? # nonnegative expressions
                        begin
                            if precompile
                                replacement = x.args[2]
                            else
                                replacement = simplify(x.args[2])
                            end

                            if !(replacement isa Int) # check if the nonnegative term is just a constant
                                if x.args[2] ∈ unique_➕_vars
                                    ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                    replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                else
                                    push!(unique_➕_vars,x.args[2])
                                    # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                    # push!(lower_bounds,eps(Float32))
                                    # push!(upper_bounds,1e12+rand())
                                    push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the cond_var_decomp
                                    push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    # push!(ss_eq_aux_ind,length(ss_and_aux_equations))

                                    # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                    replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                end
                            end

                            :($(replacement) ^ $(x.args[3]))
                        end :
                    x :
                x :
            x.args[2] isa Float64 ?
                x :
            x.args[1] ∈ [:log] ?
                !(x.args[3] isa Int) ?
                    x.args[2] isa Symbol ? # nonnegative parameters 
                        begin
                            if x.args[2] ∈ unique_➕_vars
                                ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                            else
                                push!(unique_➕_vars,x.args[2])
                                # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                # push!(lower_bounds,eps(Float32))
                                # push!(upper_bounds,1e12+rand())
                                push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the cond_var_decomp
                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                # push!(ss_eq_aux_ind,length(ss_and_aux_equations))

                                # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                replacement = Symbol("➕" * sub(string(length(➕_vars))))
                            end

                            :($(Expr(:call, x.args[1], replacement)))
                        end :
                    x.args[2].head == :ref ?
                        x :
                    x.args[2].head == :call ? # nonnegative expressions
                        begin
                            if precompile
                                replacement = x.args[2]
                            else
                                replacement = simplify(x.args[2])
                            end

                            if !(replacement isa Int) # check if the nonnegative term is just a constant
                                if x.args[2] ∈ unique_➕_vars
                                    ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                    replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                                else
                                    push!(unique_➕_vars,x.args[2])
                                    # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                    # push!(lower_bounds,eps(Float32))
                                    # push!(upper_bounds,1e12+rand())
                                    push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the cond_var_decomp
                                    push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                    # push!(ss_eq_aux_ind,length(ss_and_aux_equations))

                                    # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                    replacement = Symbol("➕" * sub(string(length(➕_vars))))
                                end
                            end

                            :($(Expr(:call, x.args[1], replacement)))
                        end :
                    x :
                x :
            x.args[1] ∈ [:norminvcdf, :norminv, :qnorm] ?
                x.args[2] isa Symbol ? # nonnegative parameters 
                    begin
                        if x.args[2] ∈ unique_➕_vars
                            ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                            replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                        else
                            push!(unique_➕_vars,x.args[2])
                            # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                            # push!(lower_bounds,eps(Float32))
                            # push!(upper_bounds,1e12+rand())
                            push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the cond_var_decomp
                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                            # push!(ss_eq_aux_ind,length(ss_and_aux_equations))

                            # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                            replacement = Symbol("➕" * sub(string(length(➕_vars))))
                        end

                        :($(Expr(:call, x.args[1], replacement)))
                    end :
                x.args[2].head == :ref ?
                    x :
                x.args[2].head == :call ? # nonnegative expressions
                    begin
                        if precompile
                            replacement = x.args[2]
                        else
                            replacement = simplify(x.args[2])
                        end

                        if !(replacement isa Int) # check if the nonnegative term is just a constant
                            if x.args[2] ∈ unique_➕_vars
                                ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                            else
                                push!(unique_➕_vars,x.args[2])
                                # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                # push!(lower_bounds,eps())
                                # push!(upper_bounds,1-eps())

                                push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1-eps(),max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the code
                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                # push!(ss_eq_aux_ind,length(ss_and_aux_equations))
                                
                                # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                replacement = Symbol("➕" * sub(string(length(➕_vars))))
                            end
                        end
                        :($(Expr(:call, x.args[1], replacement)))
                    end :
                x :
            x.args[1] ∈ [:exp] ?
                x.args[2] isa Symbol ? # have exp terms bound so they dont go to Inf
                    begin
                        if x.args[2] ∈ unique_➕_vars
                            ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                            replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                        else
                            push!(unique_➕_vars,x.args[2])
                            # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                            # push!(lower_bounds,eps(Float32))
                            # push!(upper_bounds,1e12+rand())
                            push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the cond_var_decomp
                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                            # push!(ss_eq_aux_ind,length(ss_and_aux_equations))

                            # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                            replacement = Symbol("➕" * sub(string(length(➕_vars))))
                        end

                        :($(Expr(:call, x.args[1], replacement)))
                    end :
                x.args[2].head == :ref ?
                    x :
                x.args[2].head == :call ? # have exp terms bound so they dont go to Inf
                    begin
                        if precompile
                            replacement = x.args[2]
                        else
                            replacement = simplify(x.args[2])
                        end

                        if !(replacement isa Int) # check if the nonnegative term is just a constant
                            if x.args[2] ∈ unique_➕_vars
                                ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                            else
                                push!(unique_➕_vars,x.args[2])
                                # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                # push!(lower_bounds,-1e12+rand())
                                # push!(upper_bounds,700)
                                
                                push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(700,max(-1e12,$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the code
                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                push!(ss_eq_aux_ind,length(ss_and_aux_equations))
                                
                                # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                replacement = Symbol("➕" * sub(string(length(➕_vars))))
                            end
                        end
                        :($(Expr(:call, x.args[1], replacement)))
                    end :
                x :
            x.args[1] ∈ [:erfcinv] ?
                x.args[2] isa Symbol ? # nonnegative parameters
                    begin
                        if x.args[2] ∈ unique_➕_vars
                            ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                            replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                        else
                            push!(unique_➕_vars,x.args[2])
                            # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                            # push!(lower_bounds,eps(Float32))
                            # push!(upper_bounds,1e12+rand())
                            push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(1e12,max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the cond_var_decomp
                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                            # push!(ss_eq_aux_ind,length(ss_and_aux_equations))

                            # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                            replacement = Symbol("➕" * sub(string(length(➕_vars))))
                        end

                        :($(Expr(:call, x.args[1], replacement)))
                    end :
                x.args[2].head == :ref ?
                    x :
                x.args[2].head == :call ? # nonnegative expressions
                    begin
                        if precompile
                            replacement = x.args[2]
                        else
                            replacement = simplify(x.args[2])
                        end

                        if !(replacement isa Int) # check if the nonnegative term is just a constant
                            if x.args[2] ∈ unique_➕_vars
                                ➕_vars_idx = findfirst([x.args[2]] .== unique_➕_vars)
                                replacement = Symbol("➕" * sub(string(➕_vars_idx)))
                            else
                                push!(unique_➕_vars,x.args[2])
                                # push!(bounded_vars,:($(Symbol("➕" * sub(string(length(➕_vars)+1))))))
                                # push!(lower_bounds,eps())
                                # push!(upper_bounds,2-eps())
                                push!(ss_and_aux_equations, Expr(:call,:(=), :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), :(min(2-eps(),max(eps(),$(x.args[2])))))) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier in the code
                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                push!(ss_eq_aux_ind,length(ss_and_aux_equations))
                                
                                # push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                replacement = Symbol("➕" * sub(string(length(➕_vars))))
                            end
                        end
                        :($(Expr(:call, x.args[1], replacement)))
                    end :
                x :
            x :
        x :
    x,
tbpi)