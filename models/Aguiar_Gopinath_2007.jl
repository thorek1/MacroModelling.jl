@model Aguiar_Gopinath_2007 begin
	y[0] = (exp(g[0]) * l[0]) ^ alpha * exp(z[0]) * k[-1] ^ (1 - alpha)

	z[0] = rho_z * z[-1] + σᶻ * eps_z[x]

	g[0] = (1 - rho_g) * mu_g + rho_g * g[-1] + σᵍ * eps_g[x]

	# l_m[0] = 1 - l[0] # works only with this; fix the nonnegativity constraint

	u[0] = (c[0] ^ gamma * (1 - l[0]) ^ (1 - gamma)) ^ (1 - sigma) / (1 - sigma)

	uc[0] = (1 - sigma) * u[0] * gamma / c[0]

	ul[0] = (1 - sigma) * u[0] * ( - (1 - gamma)) / (1 - l[0])

	c[0] + k[0] * exp(g[0]) = y[0] + (1 - delta) * k[-1] - k[-1] * phi / 2 * (k[0] * exp(g[0]) / k[-1] - exp(mu_g)) ^ 2 - b[-1] + b[0] * exp(g[0]) * q[0]

	1 / q[0] = 1 + r_star + psi * (exp(b[0] - b_star) - 1)

	exp(g[0]) * uc[0] * (1 + phi * (k[0] * exp(g[0]) / k[-1] - exp(mu_g))) = beta * exp(g[0] * gamma * (1 - sigma)) * uc[1] * (1 - delta + (1 - alpha) * y[1] / k[0] - phi / 2 * (k[1] * exp(g[1]) * ( - (2 * (k[1] * exp(g[1]) / k[0] - exp(mu_g)))) / k[0] + (k[1] * exp(g[1]) / k[0] - exp(mu_g)) ^ 2))

	ul[0] + y[0] * alpha * uc[0] / l[0] = 0

	q[0] * exp(g[0]) * uc[0] = beta * exp(g[0] * gamma * (1 - sigma)) * uc[1]

	invest[0] = k[-1] * phi / 2 * (k[0] * exp(g[0]) / k[-1] - exp(mu_g)) ^ 2 + k[0] * exp(g[0]) - (1 - delta) * k[-1]

	c_y[0] = c[0] / y[0]

	i_y[0] = invest[0] / y[0]

	nx[0] = (b[-1] - b[0] * exp(g[0]) * q[0]) / y[0]

	delta_y[0] = g[-1] + log(y[0]) - log(y[-1])

end


@parameters Aguiar_Gopinath_2007 begin
	beta = 1 / 1.02

	gamma = 0.36

	b_share = 0.1

	b_share * y[ss] =  b_star | b_star

	1 + r_star = 1 / q[ss] | r_star

	psi = 0.001

	alpha = 0.68

	sigma = 2

	delta = 0.05

	phi = 4

	mu_g = log(1.0066)

	rho_z = 0.95

	rho_g = 0.01

	σᶻ = .01

	σᵍ = .0005
end


get_SS(Aguiar_Gopinath_2007, verbose = true)

std(Aguiar_Gopinath_2007)

get_stochastic_steady_state(Aguiar_Gopinath_2007)

plot(Aguiar_Gopinath_2007, algorithm = :third_order)

plot_solution(Aguiar_Gopinath_2007, :b,algorithm = [:first_order,:second_order,:third_order])


nnaux = Any[:(➕₃ = max(eps(), -(-(c ^ gamma) * ➕₂ ^ (1 - gamma)))), :(➕₁ = max(eps(), -(-l * exp(g))))]

findall([Aguiar_Gopinath_2007.ss_equations[1]] .== Aguiar_Gopinath_2007.ss_aux_equations)

findall(Aguiar_Gopinath_2007.ss_aux_equations,Aguiar_Gopinath_2007.ss_equations[1])
map(x->findall(x ,Aguiar_Gopinath_2007.ss_aux_equations),Aguiar_Gopinath_2007.ss_equations)
Aguiar_Gopinath_2007.ss_equations

Meta.parse.(string.( setdiff(setdiff(𝓂.ss_aux_equations, 𝓂.ss_equations), 𝓂.ss_equations_post_modification) ))


𝓂 = Aguiar_Gopinath_2007;
symbolics = create_symbols_eqs!(𝓂);
remove_redundant_SS_vars!(𝓂,symbolics)
# solve_steady_state!(𝓂, false, symbolics, verbose = true)
# write_functions_mapping!(𝓂, symbolics)

# JQ_2012.calibration_equations

using BlockTriangularForm, SparseArrays


unknowns = union(symbolics.var,symbolics.➕_vars,symbolics.calibration_equations_parameters)

    @assert length(unknowns) <= length(symbolics.ss_equations) + length(symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = fill(0,length(unknowns),length(unknowns))


    eq_list = vcat(union.(setdiff.(union.(symbolics.var_list,
                                           symbolics.ss_list),
                                    symbolics.var_redundant_list),
                            symbolics.par_list),
                    union.(symbolics.ss_calib_list,
                            symbolics.par_calib_list))


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

	# collect(unknowns)[[49,60]]
	# collect(unknowns)[[49,60]]
    n = n_blocks

    ss_equations = vcat(symbolics.ss_equations,symbolics.calibration_equations)# .|> SymPy.Sym
	# ss_equations[[40,63]]
	# ss_equations[40]
	# ss_equations[61]
	# vars[1,vars[2,:] .== eqs[2,eqs[1,:] .== 63]]
    # # println(ss_equations)
	# 61 in eqs[1,:]

	# eqs[1,:].|> abs |>unique
	# vars[1,:].|> abs |>unique

	# findall(vars[1,:] .== 27)

	# vars[:,37]
	# findall(eqs[2,:] .== 6)

	# ss_equations[eqs[:,6]]

	eq_not_used = ss_equations[setdiff(1:size(eqs,2),unique(abs.(eqs[1,:])))]
	eq_not_solved = ss_equations[abs.(eqs[1,eqs[1,:] .< 0])]

	var_not_solved = collect(unknowns)[vars[1,eqs[1,:] .< 0]]

	vars
	eqs
	ss_equations[4]
	eq_list[4]
	collect(unknowns)[8]






using BlockTriangularForm, SparseArrays, SymPy
import MacroTools: postwalk

𝓂 = Aguiar_Gopinath_2007;
symbolics = create_symbols_eqs!(𝓂);
remove_redundant_SS_vars!(𝓂,symbolics)

symbolic_SS = false
verbose = true



function get_symbols(ex)
    par = Set()
    postwalk(x ->   
    x isa Expr ? 
        x.head == :(=) ?
            for i in x.args
                i isa Symbol ? 
                    push!(par,i) :
                x
            end :
        x.head == :call ? 
            for i in 2:length(x.args)
                x.args[i] isa Symbol ? 
                    push!(par,x.args[i]) : 
                x
            end : 
        x : 
    x, ex)
    return par
end


	unknowns = union(symbolics.var,symbolics.➕_vars,symbolics.calibration_equations_parameters)

    @assert length(unknowns) <= length(symbolics.ss_equations) + length(symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = fill(0,length(unknowns),length(unknowns))

    eq_list = vcat(union.(setdiff.(union.(symbolics.var_list,
                                           symbolics.ss_list),
                                    symbolics.var_redundant_list),
                            symbolics.par_list),
                    union.(symbolics.ss_calib_list,
                            symbolics.par_calib_list))


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

    @assert length(unique(eqs)) == size(eqs,2) "Could not solve system of steady state and calibration equations for: " * repr([collect(Symbol.(unknowns))[vars[1,eqs[1,:] .< 0]]...]) # repr([vcat(symbolics.ss_equations,symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
    
    n = n_blocks

    ss_equations = vcat(symbolics.ss_equations,symbolics.calibration_equations) .|> SymPy.Sym
    # println(ss_equations)

    SS_solve_func = []

    atoms_in_equations = Set()
    atoms_in_equations_list = []
    relevant_pars_across = []
    NSSS_solver_cache_init_tmp = []

    n_block = 1

    while n > 0 
        if length(eqs[:,eqs[2,:] .== n]) == 2
            var_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1]]

            soll = try solve(ss_equations[eqs[:,eqs[2,:] .== n][1]],var_to_solve)
            catch
            end

            if isnothing(soll)
                # println("Could not solve single variables case symbolically.")
                println("Failed finding solution symbolically for: ",var_to_solve," in: ",ss_equations[eqs[:,eqs[2,:] .== n][1]])
                # solve numerically
                continue
            elseif soll[1].is_number
                # ss_equations = ss_equations.subs(var_to_solve,soll[1])
                ss_equations = [eq.subs(var_to_solve,soll[1]) for eq in ss_equations]
                
                push!(𝓂.solved_vars,Symbol(var_to_solve))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))

                if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = max(eps(),$(𝓂.solved_vals[end]))))
                else
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                end

                push!(atoms_in_equations_list,[])
            else

                push!(𝓂.solved_vars,Symbol(var_to_solve))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))
                
                # atoms = reduce(union,soll[1].atoms())
                [push!(atoms_in_equations, a) for a in soll[1].atoms()]
                push!(atoms_in_equations_list, Set(Symbol.(soll[1].atoms())))
                # println(atoms_in_equations)
                # push!(atoms_in_equations, soll[1].atoms())

                if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = max(eps(),$(𝓂.solved_vals[end]))))
                else
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                end
            end

            # push!(single_eqs,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
            # solve symbolically
        else

            vars_to_solve = collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]

            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false
            
            if symbolic_SS
                soll = try solve(SymPy.Sym(eqs_to_solve),vars_to_solve)
                # soll = try solve(SymPy.Sym(eqs_to_solve),var_order)#,check=false,force = true,manual=true)
                catch
                end

                # println(soll)
                if isnothing(soll)
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # continue
                elseif length(soll) == 0
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # continue
                elseif length(intersect(vars_to_solve,reduce(union,map(x->x.atoms(),collect(soll[1]))))) > 0
                    if verbose
                        println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.")
                    end
                    numerical_sol = true
                    # println("Could not solve for: ",intersect(var_list,reduce(union,map(x->x.atoms(),solll)))...)
                    # break_ind = true
                    # break
                else
                    if verbose
                        println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.")
                    end
                    # relevant_pars = reduce(union,vcat(𝓂.par_list,𝓂.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
                    # relevant_pars = reduce(union,map(x->x.atoms(),collect(soll[1])))
                    atoms = reduce(union,map(x->x.atoms(),collect(soll[1])))
                    # println(atoms)
                    [push!(atoms_in_equations, a) for a in atoms]
                    
                    for (k, vars) in enumerate(vars_to_solve)
                        push!(𝓂.solved_vars,Symbol(vars))
                        push!(𝓂.solved_vals,Meta.parse(string(soll[1][k]))) #using convert(Expr,x) leads to ugly expressions

                        push!(atoms_in_equations_list, Set(Symbol.(soll[1][k].atoms())))
                        push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                    end
                end


            end
                
            # try symbolically and use numerical if it does not work
            if numerical_sol || !symbolic_SS
                if !symbolic_SS && verbose
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
                end
                push!(𝓂.solved_vars,Symbol.(collect(unknowns)[vars[:,vars[2,:] .== n][1,:]]))
                push!(𝓂.solved_vals,Meta.parse.(string.(ss_equations[eqs[:,eqs[2,:] .== n][1,:]])))
                
                syms_in_eqs = Set(Symbol.(SymPy.Sym(ss_equations[eqs[:,eqs[2,:] .== n][1,:]]).atoms()))
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
                sorted_vars = sort(𝓂.solved_vars[end])
                # sorted_vars = sort(setdiff(𝓂.solved_vars[end],𝓂.➕_vars))
                for (i, parss) in enumerate(sorted_vars) 
                    push!(guess,:($parss = guess[$i]))
                    # push!(guess,:($parss = undo_transformer(guess[$i])))
                    push!(result,:($parss = sol[$i]))
                end

                other_vars = []
                other_vars_input = []
                # other_vars_inverse = []
                other_vrs = intersect(setdiff(union(𝓂.var,𝓂.calibration_equations_parameters,𝓂.➕_vars),sort(𝓂.solved_vars[end])),syms_in_eqs)
                
                for var in other_vrs
                    # var_idx = findfirst(x -> x == var, union(𝓂.var,𝓂.calibration_equations_parameters))
                    push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
                    push!(other_vars_input,:($(var)))
                    iii += 1
                    # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
                end
                
                # separate out auxilliary variables (nonnegativity)
                nnaux = []
                nnaux_linear = []
                nnaux_error = []
                push!(nnaux_error, :(aux_error = 0))
                solved_vals = []
                
                for val in 𝓂.solved_vals[end]
                    if val isa Symbol
                        push!(solved_vals,val)
                    else
                        if (val.args[1] == :+ && val.args[3] ∈ 𝓂.➕_vars) 
                            push!(nnaux,:($(val.args[3]) = max(eps(),-$(val.args[2]))))
                            push!(nnaux_linear,:($(val.args[3]) + $(val.args[2])))
                            push!(nnaux_error, :(aux_error += min(0.0,-$(val.args[2]))))
                        elseif (val.args[1] == :- && val.args[2] ∈ 𝓂.➕_vars) 
                            push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                            push!(nnaux_linear,:($(val.args[2]) - $(val.args[3])))
                            push!(nnaux_error, :(aux_error += min(0.0,$(val.args[3]))))
                        else
                            push!(solved_vals,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
                        end
                    end
                end

                # sort nnaux vars so that they enter in right order. avoid using a variable before it is declared
                if length(nnaux) > 1

                    nn_symbols = map(x->intersect(𝓂.➕_vars,x), get_symbols.(nnaux))

                    all_symbols = reduce(vcat,nn_symbols) |> Set

                    inc_matrix = fill(0,length(all_symbols),length(all_symbols))


                    for i in 1:length(all_symbols)
                        for k in 1:length(all_symbols)
                            inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
                        end
                    end

                    QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

                    nnaux = nnaux[QQ]
                    nnaux_linear = nnaux_linear[QQ]
                end


                funcs = :(function block(parameters_and_solved_vars::Vector{Float64}, guess::Vector{Float64}, transformer_option::Int)
                        # if guess isa Tuple guess = guess[1] end
                        guess = undo_transformer(guess, option = transformer_option) 
                        # println(guess)
                        $(guess...) 
                        $(calib_pars...) # add those variables which were previously solved and are used in the equations
                        $(other_vars...) # take only those that appear in equations - DONE

                        # $(aug_lag...)
                        # $(nnaux...)
                        # $(nnaux_linear...)
                        return [$(solved_vals...),$(nnaux_linear...)]
                    end)

                push!(NSSS_solver_cache_init_tmp,fill(.9,length(sorted_vars)))

                # WARNING: infinite bounds are transformed to 1e12
                lbs = []
                ubs = []
                
                limit_boundaries = 1e12

                for i in sorted_vars
                    if i ∈ 𝓂.bounded_vars
                        push!(lbs,𝓂.lower_bounds[i .== 𝓂.bounded_vars][1] == -Inf ? -limit_boundaries : 𝓂.lower_bounds[i .== 𝓂.bounded_vars][1])
                        push!(ubs,𝓂.upper_bounds[i .== 𝓂.bounded_vars][1] ==  Inf ?  limit_boundaries : 𝓂.upper_bounds[i .== 𝓂.bounded_vars][1])
                    else
                        push!(lbs,-limit_boundaries)
                        push!(ubs,limit_boundaries)
                    end
                end
                push!(SS_solve_func,:(lbs = [$(lbs...)]))
                push!(SS_solve_func,:(ubs = [$(ubs...)]))
				
                push!(SS_solve_func,:(inits = max.(lbs,min.(ubs, closest_solution[$(n_block)]))))
                push!(SS_solve_func,:(block_solver_RD = block_solver_AD([$(calib_pars_input...),$(other_vars_input...)],
                                                                        $(n_block), 
                                                                        𝓂.ss_solve_blocks[$(n_block)], 
                                                                        # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                                        # f, 
                                                                        inits,
                                                                        lbs, 
                                                                        ubs,
                                                                        fail_fast_solvers_only = fail_fast_solvers_only,
                                                                        verbose = verbose)))
                
                push!(SS_solve_func,:(solution = block_solver_RD([$(calib_pars_input...),$(other_vars_input...)])))

                push!(SS_solve_func,:(solution_error += sum(abs2,𝓂.ss_solve_blocks[$(n_block)]([$(calib_pars_input...),$(other_vars_input...)],solution,0))))
                push!(SS_solve_func,:(sol = solution))

                # push!(SS_solve_func,:(println(sol))) 

                push!(SS_solve_func,:($(result...)))   
                push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))

                n_block += 1
            end
        end
        n -= 1
    end
