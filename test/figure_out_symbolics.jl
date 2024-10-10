using Revise
using MacroModelling
# using Groebner
# using Symbolics
# using Nemo

@model m begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * (eps_z[x-2] + eps_z[x+2] + eps_z_s[x])
    ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[ss]) / 4
    ZZ_avg_fut[0] = (A[0] + A[1] + A[2] + A[ss]) / 4
    log_ZZ_avg[0] = log(ZZ_avg[0]/ZZ_avg[ss])
    c_normlogpdf[0]= normlogpdf(c[0])
    c_norminvcdf[0]= norminvcdf(c[0]-1)
end


@parameters m verbose = true begin
    k[ss] / (4 * y[ss]) = cap_share | alpha
    cap_share = 1.66
    # alpha = .157

    beta | log(R[ss]) = R_ss - 1
    R_ss = 1.0035
    # beta = .999

    c[ss]/y[ss] = 1 - I_K_ratio | delta
    # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
    I_K_ratio = .15
    # delta = .0226

    Pibar | Pi[ss] = Pi_ss
    Pi_ss = R_ss - Pi_real
    Pi_real = 1/1000
    # Pibar = 1.0008

    phi_pi = 1.5
    rhoz = 9 / 10
    std_eps = .0068
    rho_z_delta = rhoz
    std_z_delta = .005

    0 < alpha < 1
    0 < beta < 1
end

m.ss_aux_equations

m.var_redundant_list

SS(m)
𝓂 = m

module sym_wrangle_space
    import Symbolics
    import Nemo
end

import Symbolics

var_present_list = 𝓂.var_present_list_aux_SS
var_past_list = 𝓂.var_past_list_aux_SS
var_future_list = 𝓂.var_future_list_aux_SS
ss_list = 𝓂.ss_list_aux_SS
var_list = 𝓂.var_list_aux_SS

var_redundant_list = [Set{Symbol}() for _ in 1:length(𝓂.ss_aux_equations)]

# check variables which appear in two time periods. they might be redundant in steady state
redundant_vars = intersect.(
    union.(
        intersect.(var_future_list,var_present_list),
        intersect.(var_future_list,var_past_list),
        intersect.(var_present_list,var_past_list),
        intersect.(ss_list,var_present_list),
        intersect.(ss_list,var_past_list),
        intersect.(ss_list,var_future_list)
    ),
var_list)

redundant_idx = getindex(1:length(redundant_vars), (length.(redundant_vars) .> 0) .& (length.(var_list) .> 1))

for i in redundant_idx
    for var_to_solve_for in redundant_vars[i]   
        parsed_expr = Symbolics.parse_expr_to_symbolic((𝓂.ss_aux_equations[i]), sym_wrangle_space) |> Symbolics.expand

        @eval sym_wrangle_space begin
            Symbolics.@variables $var_to_solve_for
            
            solved_expr = try Symbolics.symbolic_solve($parsed_expr, $var_to_solve_for)
            catch
                nothing
            end

            if isnothing(solved_expr)
                solved_expr = try Symbolics.symbolic_solve(1/$parsed_expr, $var_to_solve_for)
                catch
                    nothing
                end
            end

            if !isnothing(solved_expr)
                solved_expr = Symbolics.fast_substitute(solved_expr, Dict(1//0 => 0, -1//0 => 0))

                if solved_expr isa Vector{Int} && solved_expr == [0] # take out variable if it is redundant from that equation only
                    # push!(var_redundant_list[i], var_to_solve_for)

                    tmp = Symbolics.fast_substitute($parsed_expr, Dict($var_to_solve_for => 1)) |> 
                    string |> 
                    Meta.parse
                end
            end
        end

        solved_expr = sym_wrangle_space.solved_expr 

        if solved_expr isa Vector{Int} && solved_expr == [0] # take out variable if it is redundant from that equation only
            push!(var_redundant_list[i], var_to_solve_for)

            𝓂.ss_aux_equations[i] = sym_wrangle_space.tmp
        end
    end
end

copy!(𝓂.var_redundant_list, var_redundant_list)















import BlockTriangularForm
import Symbolics
import Nemo

𝓂 = m

ss_equations = 𝓂.ss_aux_equations
dyn_equations = 𝓂.dyn_equations
dyn_var_present_list = 𝓂.dyn_var_present_list
dyn_var_past_list = 𝓂.dyn_var_past_list
dyn_var_future_list = 𝓂.dyn_var_future_list
dyn_exo_list = 𝓂.dyn_exo_list

dyn_future_list = 𝓂.dyn_future_list
dyn_present_list = 𝓂.dyn_present_list
dyn_past_list = 𝓂.dyn_past_list

var_present_list = 𝓂.var_present_list_aux_SS
var_past_list = 𝓂.var_past_list_aux_SS
var_future_list = 𝓂.var_future_list_aux_SS
ss_list = 𝓂.ss_list_aux_SS
var_list = 𝓂.var_list_aux_SS

par_list = 𝓂.par_list_aux_SS

calibration_equations = 𝓂.calibration_equations
calibration_equations_parameters = 𝓂.calibration_equations_parameters

vars_in_ss_equations = 𝓂.vars_in_ss_equations
var = 𝓂.var
➕_vars = 𝓂.➕_vars
ss_calib_list = 𝓂.ss_calib_list
par_calib_list = 𝓂.par_calib_list

var_redundant_list = 𝓂.var_redundant_list

# write_ss_check_function!(𝓂)

unknowns = union(vars_in_ss_equations, calibration_equations_parameters)

@assert length(unknowns) <= length(ss_equations) + length(calibration_equations) "Unable to solve steady state. More unknowns than equations."

incidence_matrix = fill(0,length(unknowns),length(unknowns))

eq_list = vcat(union.(setdiff.(union.(var_list,
                                    ss_list),
                                var_redundant_list),
                        par_list),
                union.(ss_calib_list,
                        par_calib_list))

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

# @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations for: " * repr([collect(Symbol.(unknowns))[vars[1,eqs[1,:] .< 0]]...]) # repr([vcat(ss_equations,calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
@assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant euqations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now." # repr([vcat(ss_equations,calibration_equations)[-eqs[1,eqs[1,:].<0]]...])

n = n_blocks

ss_equations = vcat(ss_equations,calibration_equations)# .|> SPyPyC.Sym
# println(ss_equations)

SS_solve_func = []

atoms_in_equations = Set{Symbol}()
atoms_in_equations_list = []
relevant_pars_across = Symbol[]
NSSS_solver_cache_init_tmp = []

min_max_errors = []

unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()

# while n > 0 
#     if length(eqs[:,eqs[2,:] .== n]) == 2
        var_to_solve_for = collect(unknowns)[vars[:,vars[2,:] .== n][1]]

        eq_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1]]

        # eliminate min/max from equations if solving for variables inside min/max. set to the variable we solve for automatically
        parsed_eq_to_solve_for =  Symbolics.parse_expr_to_symbolic(eq_to_solve,sym_wrangle_space) #string |> Meta.parse

        # minmax_fixed_eqs = postwalk(x -> 
        #     x isa Expr ?
        #         x.head == :call ? 
        #             x.args[1] ∈ [:Max,:Min] ?
        #                 Symbol(var_to_solve_for) ∈ get_symbols(x.args[2]) ?
        #                     x.args[2] :
        #                 Symbol(var_to_solve_for) ∈ get_symbols(x.args[3]) ?
        #                     x.args[3] :
        #                 x :
        #             x :
        #         x :
        #     x,
        # parsed_eq_to_solve_for)

        # if parsed_eq_to_solve_for != minmax_fixed_eqs
        #     [push!(atoms_in_equations, a) for a in setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs))]
        #     push!(min_max_errors,:(solution_error += abs($parsed_eq_to_solve_for)))
        #     push!(SS_solve_func, :(if solution_error > 1e-12 if verbose println("Failed for min max terms in equations with error $solution_error") end; return [0.0], (solution_error,1) end))
        #     eq_to_solve = eval(minmax_fixed_eqs)
        # end
        
        eval(:(Symbolics.@variables $var_to_solve_for))
            
        soll = try Symbolics.symbolic_solve(parsed_eq_to_solve_for, eval(var_to_solve_for))
        catch
            nothing
        end

        # if avoid_solve && count_ops(Meta.parse(string(eq_to_solve))) > 15
        #     soll = nothing
        # else
        #     soll = try SPyPyC.solve(eq_to_solve,var_to_solve_for)
        #     catch
        #     end
        # end
        
        if isnothing(soll) || isempty(soll)
            println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
            
            eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

            write_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)
            # write_domain_safe_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
        elseif soll[1].is_number == true
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
            
            [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
            push!(atoms_in_equations_list, Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)),Symbol.(soll[1].atoms()))))

            if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars)
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = min(max($(𝓂.bounds[𝓂.solved_vars[end]][1]), $(𝓂.solved_vals[end])), $(𝓂.bounds[𝓂.solved_vars[end]][2]))))
                push!(SS_solve_func,:(solution_error += $(Expr(:call,:abs, Expr(:call, :-, 𝓂.solved_vars[end], 𝓂.solved_vals[end])))))
                push!(SS_solve_func, :(if solution_error > 1e-12 if verbose println("Failed for analytical aux variables with error $solution_error") end; return [0.0], (solution_error,1) end))

                unique_➕_eqs[𝓂.solved_vals[end]] = 𝓂.solved_vars[end]
            else
                vars_to_exclude = [vcat(Symbol.(var_to_solve_for), 𝓂.➕_vars), Symbol[]]
                
                rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_rebust_to_domain_errors([𝓂.solved_vals[end]], vars_to_exclude, 𝓂.bounds, 𝓂.➕_vars, unique_➕_eqs)

                if length(vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)) > 0
                    push!(SS_solve_func,vcat(ss_and_aux_equations, ss_and_aux_equations_dep)...)
                    push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)...))))
                    push!(SS_solve_func, :(if solution_error > 1e-12 if verbose println("Failed for analytical variables with error $solution_error") end; return [0.0], (solution_error,1) end))
                end
                
                push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(rewritten_eqs[1])))
            end

            if haskey(𝓂.bounds, 𝓂.solved_vars[end]) && 𝓂.solved_vars[end] ∉ 𝓂.➕_vars
                push!(SS_solve_func,:(solution_error += abs(min(max($(𝓂.bounds[𝓂.solved_vars[end]][1]), $(𝓂.solved_vars[end])), $(𝓂.bounds[𝓂.solved_vars[end]][2])) - $(𝓂.solved_vars[end]))))
                push!(SS_solve_func, :(if solution_error > 1e-12 if verbose println("Failed for bounded variables with error $solution_error") end; return [0.0], (solution_error,1) end))
            end
        end
    else
        vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]

        eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

        numerical_sol = false
        
        if symbolic_SS
            if avoid_solve && count_ops(Meta.parse(string(eqs_to_solve))) > 15
                soll = nothing
            else
                soll = try SPyPyC.solve(eqs_to_solve,vars_to_solve)
                catch
                end
            end

            if isnothing(soll) || length(soll) == 0 || length(intersect((union(SPyPyC.free_symbols.(soll[1])...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
                if verbose println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.") end

                numerical_sol = true
            else
                if verbose println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.") end
                
                atoms = reduce(union,map(x->x.atoms(),collect(soll[1])))

                for a in atoms push!(atoms_in_equations, Symbol(a)) end
                
                for (k, vars) in enumerate(vars_to_solve)
                    push!(𝓂.solved_vars,Symbol(vars))
                    push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

                    push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                end
            end
        end

        eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

        incidence_matrix_subset = incidence_matrix[vars[:,vars[2,:] .== n][1,:], eq_idx_in_block_to_solve]

        # try symbolically and use numerical if it does not work
        if numerical_sol || !symbolic_SS
            pv = sortperm(vars_to_solve, by = Symbol)
            pe = sortperm(eqs_to_solve, by = string)

            if length(pe) > 5
                write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)
                # write_domain_safe_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)
            else
                solved_system = partial_solve(eqs_to_solve[pe], vars_to_solve[pv], incidence_matrix_subset[pv,pe], avoid_solve = avoid_solve)
                
                # if !isnothing(solved_system) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "LambertW")) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "Heaviside")) 
                #     write_reduced_block_solution!(𝓂, SS_solve_func, solved_system, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, 
                #     𝓂.➕_vars, unique_➕_eqs)  
                # else
                    write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)  
                    # write_domain_safe_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
                # end
            end

            if !symbolic_SS && verbose
                println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
            end
        end
    end
    n -= 1
end






















eval(:(Symbolics.@variables $var_to_solve_for))

parsed_expr = Symbolics.parse_expr_to_symbolic(ss_equations[2], Main) |>expand

(-R*beta) / (Pi*c) + 1 / c
parsed_expr*c
# solved_expr = Symbolics.symbolic_solve(parsed_expr, eval(var_to_solve_for))
solved_expr =  Symbolics.symbolic_solve(1/parsed_expr, c)
solved_expr isa Vector{<: Real}
solved_expr =  Symbolics.derivative(parsed_expr, c)

tosymbol(parsed_expr)

solved_expr = Symbolics.fast_substitute(solved_expr, Dict(1//0 => 0, -1//0 => 0))





x = collect(redundant_vars[redundant_idx[3]])[1]

Symbolics.@variables $x

Symbolics.@variables k

c

for i in redundant_idx
    for var_to_solve_for in redundant_vars[i]            
        if avoid_solve && count_ops(Meta.parse(string(ss_equations[i]))) > 15
            soll = nothing
        else
            soll = try SPyPyC.solve(ss_equations[i],var_to_solve_for)
            catch
            end
        end

        if isnothing(soll)
            continue
        end
        
        if length(soll) == 0 || soll == SPyPyC.Sym[0] # take out variable if it is redundant from that euation only
            push!(var_redundant_list[i],var_to_solve_for)
            ss_equations[i] = ss_equations[i].subs(var_to_solve_for,1).replace(SPyPyC.Sym(ℯ),exp(1)) # replace euler constant as it is not translated to julia properly
        end

    end
end







import MacroModelling: convert_to_ss_equation, get_symbols

test1 = :(pi[0] ^ -1 * pi[-1] ^ gamma_p)

test1 = :((-epsilon_b*(➕₉^(-sigma_c))) / R_f + beta*epsilon_b*(➕₄^(-sigma_c)))

test1 = :(Pi[0] / Pibar)
test1 = :(ZZ_avg[0] / ZZ_avg[ss])
:(c[0] - 1)
:(y - A * k ^ alpha)
test1 = :(1 / c - ((beta * 1) / c) * (alpha * A * k ^ (alpha - 1) + (1 - delta)))
:(1 / c - ((beta * 1) / c) * (R / Pi))
:(Pi / Pibar)
:(R * beta - ➕₁ ^ phi_pi)
:(A * k ^ alpha - ((c + k) - (1 - delta * z_delta) * k))
:(z_delta - ((1 - rho_z_delta) + rho_z_delta * z_delta + 0))
:(A - ((1 - rhoz) + rhoz * A + std_eps * (0 + 0 + 0)))
:(ZZ_avg - (A + A + A + A) / 4)
:(ZZ_avg_fut - (A + A + A + A) / 4)
:(log_ZZ_avg - log(1))
:(c_normlogpdf - normlogpdf(c))
:(c - 1)


using Symbolics
# using Groebner
using Nemo





module sym_wrangle_space
import Symbolics
parse_t = Symbolics.parse_expr_to_symbolic(:(1 / c - ((beta * 1) / c) * (alpha * A * k ^ (alpha - 1) + (1 - delta))), sym_wrangle_space) 
end

import MacroModelling: convert_to_ss_equation, get_symbols
import Symbolics
import Nemo
import Groebner

# derivative(parsed_t,c)
Symbolics.@variables a,b,d,e,c, A,x,y,k, alpha, beta, delta, epsilon_b

test1 = :((-x * a ^ -b) / d + e * x * c ^ -b)


Symbolics.symbolic_solve((-x * a ^ -b) / d + e * x * c ^ -b, x)

Symbolics.symbolic_solve(1/(x * a ^ b + x * c ^ d), b)

Symbolics.symbolic_solve(x * a ^ b + x * c, x)

Symbolics.symbolic_solve(x * a ^ b + x, x)

test1 = :((-epsilon_b * ➕₉ ^ -sigma_c) / R_f + beta * epsilon_b * ➕₄ ^ -sigma_c)
test1 = :(-(epsilon_b * alp ) / R_f + beta * epsilon_b * ➕₉)


import Symbolics as Smb
using Nemo
using Groebner
Smb.@variables x, a, b
Smb.symbolic_solve(x * a ^ b + x, x)

import SymPyPythonCall as Smbpy
Smbpy.@syms xx aa bb
Smbpy.solve(xx * aa ^ bb + xx, xx)



parsed_t = test1 |> convert_to_ss_equation |> x->Symbolics.parse_expr_to_symbolic(x,Main) |> Symbolics.expand

Symbolics.symbolic_solve(1/parsed_t, epsilon_b)


Symbolics.symbolic_solve(1 / c - ((beta * 1) / c) * (alpha * A * k ^ (alpha - 1) + (1 - delta)),c)

Symbolics.symbolic_solve(1 / c - ((beta * 1) / c) * (alpha * A * k ^ (alpha - 1) + (1 - delta)),k)

Symbolics.derivative( 1 / c - ((beta * 1) / c) * (alpha * A * k ^ (alpha - 1) + (1 - delta)),c)

Symbolics.simplify(Symbolics.symbolic_solve(1 / c - ((beta * 1) / c) * (alpha * A * k ^ (alpha - 1) + (1 - delta)),c), expand = true, simplify_fractions = false)


Symbolics.symbolic_solve(0~ 1 / c - ((beta * 1) / c) * (alpha * A * k ^ (alpha - 1) + (1 - delta)),c)[1]|>simplify

Symbolics.symbolic_solve(0 ~ parsed_t,k)


found_roots = symbolic_solve(1/x^2 ~ 1/y^2 - 2/x^3 * (x-y), x)
Symbolics.symbolic_solve(parsed_t,[A])

test1 |> convert_to_ss_equation |> x->Symbolics.parse_expr_to_symbolic(x,Main) |> Symbolics.simplify |> Symbolics.latexify |> string |> Meta.parse

ex_ss = convert_to_ss_equation(test1)
parsed_expr = Symbolics.parse_expr_to_symbolic(ex_ss,Main)

Meta.parse(string(parsed_expr))

for x in get_symbols(ex_ss)
    Symbolics.@variables $x
end

parsed = ex_ss |> eval |> string |> Meta.parse

postwalk(x ->   x isa Expr ? 
                    x.args[1] == :conjugate ? 
                        x.args[2] : 
                    x : 
                x, parsed)






using Symbolics, Nemo

@variables c beta alpha A k delta

expr = 1 / c - ((beta * 1) / c) * (alpha * A * k ^ (alpha - 1) + (1 - delta))

expr = expand(beta * (alpha * A * k ^ (alpha - 1) + (1 - delta)))

exx = symbolic_solve(expand(expr), c)
Symbolics.fast_substitute(exx, Dict(1//0=>0, -1//0=>0))

exx = substitute(exx,Dict(-(1//0)=>0))

symbolic_solve(expand(expr), c)

1//0


expr = expand((x + b)*(x^2 + 2x + 1)*(x^2 - a))

symbolic_solve(expr, x)

symbolic_solve(expr, x, dropmultiplicity=false)



symbolic_solve(a+x^(b-1), x)








@model Smets_Wouters_2003 begin
    -q[0] + beta * ((1 - tau) * q[1] + epsilon_b[1] * (r_k[1] * z[1] - psi^-1 * r_k[ss] * (-1 + exp(psi * (-1 + z[1])))) * (C[1] - h * C[0])^(-sigma_c)) = 0
    -q_f[0] + beta * ((1 - tau) * q_f[1] + epsilon_b[1] * (r_k_f[1] * z_f[1] - psi^-1 * r_k_f[ss] * (-1 + exp(psi * (-1 + z_f[1])))) * (C_f[1] - h * C_f[0])^(-sigma_c)) = 0
    -r_k[0] + alpha * epsilon_a[0] * mc[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^(-1 + alpha) = 0
    -r_k_f[0] + alpha * epsilon_a[0] * mc_f[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^(-1 + alpha) = 0
    -G[0] + T[0] = 0
    -G[0] + G_bar * epsilon_G[0] = 0
    -G_f[0] + T_f[0] = 0
    -G_f[0] + G_bar * epsilon_G[0] = 0
    -L[0] + nu_w[0]^-1 * L_s[0] = 0
    -L_s_f[0] + L_f[0] * (W_i_f[0] * W_f[0]^-1)^(lambda_w^-1 * (-1 - lambda_w)) = 0
    L_s_f[0] - L_f[0] = 0
    L_s_f[0] + lambda_w^-1 * L_f[0] * W_f[0]^-1 * (-1 - lambda_w) * (-W_disutil_f[0] + W_i_f[0]) * (W_i_f[0] * W_f[0]^-1)^(-1 + lambda_w^-1 * (-1 - lambda_w)) = 0
    Pi_ws_f[0] - L_s_f[0] * (-W_disutil_f[0] + W_i_f[0]) = 0
    Pi_ps_f[0] - Y_f[0] * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) = 0
    -Q[0] + epsilon_b[0]^-1 * q[0] * (C[0] - h * C[-1])^(sigma_c) = 0
    -Q_f[0] + epsilon_b[0]^-1 * q_f[0] * (C_f[0] - h * C_f[-1])^(sigma_c) = 0
    -W[0] + epsilon_a[0] * mc[0] * (1 - alpha) * L[0]^(-alpha) * (K[-1] * z[0])^alpha = 0
    -W_f[0] + epsilon_a[0] * mc_f[0] * (1 - alpha) * L_f[0]^(-alpha) * (K_f[-1] * z_f[0])^alpha = 0
    -Y_f[0] + Y_s_f[0] = 0
    Y_s[0] - nu_p[0] * Y[0] = 0
    -Y_s_f[0] + Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) = 0
    beta * epsilon_b[1] * (C_f[1] - h * C_f[0])^(-sigma_c) - epsilon_b[0] * R_f[0]^-1 * (C_f[0] - h * C_f[-1])^(-sigma_c) = 0
    beta * epsilon_b[1] * pi[1]^-1 * (C[1] - h * C[0])^(-sigma_c) - epsilon_b[0] * R[0]^-1 * (C[0] - h * C[-1])^(-sigma_c) = 0
    Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) - lambda_p^-1 * Y_f[0] * (1 + lambda_p) * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-1 - lambda_p^-1 * (1 + lambda_p)) = 0
    epsilon_b[0] * W_disutil_f[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) - omega * epsilon_b[0] * epsilon_L[0] * L_s_f[0]^sigma_l = 0
    -1 + xi_p * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1) + (1 - xi_p) * pi_star[0]^(-lambda_p^-1) = 0
    -1 + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1) + xi_w * (W[-1] * W[0]^-1)^(-lambda_w^-1) * (pi[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1) = 0
    -Phi - Y_s[0] + epsilon_a[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^alpha = 0
    -Phi - Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_a[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^alpha = 0
    eta_b[exo] - log(epsilon_b[0]) + rho_b * log(epsilon_b[-1]) = 0
    -eta_L[exo] - log(epsilon_L[0]) + rho_L * log(epsilon_L[-1]) = 0
    eta_I[exo] - log(epsilon_I[0]) + rho_I * log(epsilon_I[-1]) = 0
    eta_w[exo] - f_1[0] + f_2[0] = 0
    eta_a[exo] - log(epsilon_a[0]) + rho_a * log(epsilon_a[-1]) = 0
    eta_p[exo] - g_1[0] + g_2[0] * (1 + lambda_p) = 0
    eta_G[exo] - log(epsilon_G[0]) + rho_G * log(epsilon_G[-1]) = 0
    -f_1[0] + beta * xi_w * f_1[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1) + epsilon_b[0] * w_star[0] * L[0] * (1 + lambda_w)^-1 * (C[0] - h * C[-1])^(-sigma_c) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)) = 0
    -f_2[0] + beta * xi_w * f_2[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) + omega * epsilon_b[0] * epsilon_L[0] * (L[0] * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)))^(1 + sigma_l) = 0
    -g_1[0] + beta * xi_p * pi_star[0] * g_1[1] * pi_star[1]^-1 * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1) + epsilon_b[0] * pi_star[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c) = 0
    -g_2[0] + beta * xi_p * g_2[1] * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_b[0] * mc[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c) = 0
    -nu_w[0] + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)) + xi_w * nu_w[-1] * (W[-1] * pi[0]^-1 * W[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w)) = 0
    -nu_p[0] + (1 - xi_p) * pi_star[0]^(-lambda_p^-1 * (1 + lambda_p)) + xi_p * nu_p[-1] * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p)) = 0
    -K[0] + K[-1] * (1 - tau) + I[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2) = 0
    -K_f[0] + K_f[-1] * (1 - tau) + I_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2) = 0
    U[0] - beta * U[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C[0] - h * C[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s[0]^(1 + sigma_l)) = 0
    U_f[0] - beta * U_f[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C_f[0] - h * C_f[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s_f[0]^(1 + sigma_l)) = 0
    -epsilon_b[0] * (C[0] - h * C[-1])^(-sigma_c) + q[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2 - varphi * I[-1]^-1 * epsilon_I[0] * I[0] * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])) + beta * varphi * I[0]^-2 * epsilon_I[1] * q[1] * I[1]^2 * (-1 + I[0]^-1 * epsilon_I[1] * I[1]) = 0
    -epsilon_b[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) + q_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2 - varphi * I_f[-1]^-1 * epsilon_I[0] * I_f[0] * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])) + beta * varphi * I_f[0]^-2 * epsilon_I[1] * q_f[1] * I_f[1]^2 * (-1 + I_f[0]^-1 * epsilon_I[1] * I_f[1]) = 0
    eta_pi[exo] - log(pi_obj[0]) + rho_pi_bar * log(pi_obj[-1]) + log(calibr_pi_obj) * (1 - rho_pi_bar) = 0
    -C[0] - I[0] - T[0] + Y[0] - psi^-1 * r_k[ss] * K[-1] * (-1 + exp(psi * (-1 + z[0]))) = 0
    -calibr_pi + eta_R[exo] - log(R[ss]^-1 * R[0]) + r_Delta_pi * (-log(pi[ss]^-1 * pi[-1]) + log(pi[ss]^-1 * pi[0])) + r_Delta_y * (-log(Y[ss]^-1 * Y[-1]) + log(Y[ss]^-1 * Y[0]) + log(Y_f[ss]^-1 * Y_f[-1]) - log(Y_f[ss]^-1 * Y_f[0])) + rho * log(R[ss]^-1 * R[-1]) + (1 - rho) * (log(pi_obj[0]) + r_pi * (-log(pi_obj[0]) + log(pi[ss]^-1 * pi[-1])) + r_Y * (log(Y[ss]^-1 * Y[0]) - log(Y_f[ss]^-1 * Y_f[0]))) = 0
    -C_f[0] - I_f[0] + Pi_ws_f[0] - T_f[0] + Y_f[0] + L_s_f[0] * W_disutil_f[0] - L_f[0] * W_f[0] - psi^-1 * r_k_f[ss] * K_f[-1] * (-1 + exp(psi * (-1 + z_f[0]))) = 0
    epsilon_b[0] * (K[-1] * r_k[0] - r_k[ss] * K[-1] * exp(psi * (-1 + z[0]))) * (C[0] - h * C[-1])^(-sigma_c) = 0
    epsilon_b[0] * (K_f[-1] * r_k_f[0] - r_k_f[ss] * K_f[-1] * exp(psi * (-1 + z_f[0]))) * (C_f[0] - h * C_f[-1])^(-sigma_c) = 0
end


@parameters Smets_Wouters_2003 verbose = true begin  
    calibr_pi_obj | 1 = pi_obj[ss]
    calibr_pi | pi[ss] = pi_obj[ss]
    # Phi | Y_s[ss] * .408 = Phi
    # Phi = .408 * Y_j[ss]
    # (Y_j[ss] + Phi) / Y_j[ss] = 1.408 -> Phi; | this seems problematic because of the parameter
    # lambda_p | .6 = C_f[ss] / Y_f[ss]
    # lambda_w | L[ss] = .33
    # G_bar | .18 = G[ss] / Y[ss]
    # calibr_pi_obj = 0
    # calibr_pi = 1
    lambda_p = .368
    G_bar = .362
    lambda_w = 0.5
    Phi = .819

    alpha = 0.3
    beta = 0.99
    gamma_w = 0.763
    gamma_p = 0.469
    h = 0.573
    omega = 1
    psi = 0.169
    r_pi = 1.684
    r_Y = 0.099
    r_Delta_pi = 0.14
    r_Delta_y = 0.159
    rho = 0.961
    rho_b = 0.855
    rho_L = 0.889
    rho_I = 0.927
    rho_a = 0.823
    rho_G = 0.949
    rho_pi_bar = 0.924
    sigma_c = 1.353
    sigma_l = 2.4
    tau = 0.025
    varphi = 6.771
    xi_w = 0.737
    xi_p = 0.908

    # Putting non-negative constraint on first block is enough
    # 0 < K
    # 0 < I
    # 0 < Y_s
    # 0 < q
    # 0 < r_k
    # 5 < f_1
    # 0 < L
    # 0 < W
    # 30 < g_1
    # 0 < z
    # 0 < mc
    # 0 < w_star
    # 5 < f_2
    # 0 < Y
    # 0 < g_2
    # 0 < C
end