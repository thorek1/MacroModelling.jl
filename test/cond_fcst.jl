using MacroModelling
using SparseArrays, AxisKeys
# import LinearAlgebra as ℒ
# import MacroModelling: Symbol_input, ℳ



include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags_numsolve.jl")

get_irf(m)

using SparseArrays
conditions = spzeros(17,2)
conditions[4,1] = .0002
conditions[6,2] = .0001

shocks = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,2),Variables = [:eps_z], Periods = 1:2)
shocks[1,1] = .05
shocks[1,2] = .05

Y = get_conditional_forecast(m,conditions, shocks = shocks)





function plot_conditional_forecast(𝓂::ℳ,
    conditions::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}};
    shocks::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}, Nothing} = nothing, 
    periods::Int = 40, 
    parameters = nothing,
    variables::Symbol_input = :all_including_auxilliary, 
    conditions_in_levels::Bool = false,
    levels::Bool = false,
    show_plots::Bool = true,
    save_plots::Bool = false,
    save_plots_format::Symbol = :pdf,
    save_plots_path::String = ".",
    plots_per_page::Int = 4,
    verbose = false)

    Y = get_conditional_forecast(𝓂,
                                conditions,
                                shocks = shocks, 
                                periods = periods, 
                                parameters = parameters,
                                variables = variables, 
                                conditions_in_levels = conditions_in_levels,
                                levels = levels,
                                verbose = verbose)
    
    periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    full_SS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    full_SS[indexin(𝓂.aux,full_SS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)
    
    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())
    
    reference_steady_state = [s ∈ 𝓂.exo_present ? 0 : NSSS[s] for s in full_SS]
    
    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)
                    
              
    if conditions isa SparseMatrixCSC{Float64}
        @assert length(full_SS) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(full_SS)) * " variables (including auxilliary variables): " * repr(full_SS)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(full_SS),periods)
        nzs = findnz(conditions)
        for i in 1:length(nzs[1])
            cond_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        conditions = cond_tmp
    elseif conditions isa Matrix{Union{Nothing,Float64}}
        @assert length(full_SS) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(full_SS)) * " variables (including auxilliary variables): " * repr(full_SS)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(full_SS),periods)
        cond_tmp[:,axes(conditions,2)] = conditions
        conditions = cond_tmp
    elseif conditions isa KeyedArray{Union{Nothing,Float64}} || conditions isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(conditions,1),full_SS)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(axiskeys(conditions,1),full_SS))
        
        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(full_SS),periods)
        cond_tmp[indexin(sort(axiskeys(conditions,1)),full_SS),axes(conditions,2)] .= conditions(sort(axiskeys(conditions,1)))
        conditions = cond_tmp
    end
    
    if shocks isa SparseMatrixCSC{Float64}
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        nzs = findnz(shocks)
        for i in 1:length(nzs[1])
            shocks_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(shocks,1),𝓂.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(axiskeys(shocks,1),𝓂.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[indexin(sort(axiskeys(shocks,1)),𝓂.exo),axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
    end
      
    # plots = []
    default(size=(700,500),
            # leg = false,
            # plot_titlefont = (10, fontt), 
            # titlefont = (10, fontt), 
            # guidefont = (8, fontt), 
            plot_titlefont = (10), 
            titlefont = (10), 
            guidefont = (8), 
            legendfontsize = 8, 
            # tickfont = (8, fontt),
            # tickfontfamily = fontt,
            tickfontsize = 8,
            # tickfontrotation = 9,
            # rotation = 90,
            # tickfontvalign = :center,
            # topmargin = 10mm,
            # rightmargin = 17mm, 
            framestyle = :box)


    # for i in 1:length(var_idx)
        n_subplots = length(var_idx)
        pp = []
        pane = 1
        plot_count = 1
        for i in 1:length(var_idx)
            if all(isapprox.(Y[i,:], 0, atol = eps(Float32)))
                n_subplots -= 1
            end
        end

        for i in 1:length(var_idx)
            SS = reference_steady_state[var_idx[i]]
            if !(all(isapprox.(Y[i,:],0,atol = eps(Float32))))
            # if !(plot_count ∈ unique(round.((1:𝓂.timings.timings.nVars)/plots_per_page))*plots_per_page)
                if !(plot_count % plots_per_page == 0)
                    plot_count += 1
                    if all((Y[i,:] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                        Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                        hline!([SS 0], color = :black, label = "") 
                                        Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)                             
                            end)
                        else
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                        Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                        hline!([SS 0], color = :black, label = "")                         
                            end)
                        end
                    else
                        cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        hline!([SS], color = :black, label = "")
                                        Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)   
                            end)
                        else
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        hline!([SS], color = :black, label = "")
                            end)
                        end

                    end
                else

                    plot_count = 1
                    if all((Y[i,:] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                    Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                    hline!([SS 0],color = :black,label = "")   
                                    Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)                            
                        end)
                    else
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                    Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                    hline!([SS 0],color = :black,label = "")                              
                        end)
                    end
                    else
                        cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        hline!([SS], color = :black, label = "")
                                        Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)  
                            end)
                        else 
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        hline!([SS], color = :black, label = "")
                            end)
                        end

                    end

                    shock_string = "Conditional forecast"
                    shock_name = "conditional_forecast"

                    ppp = Plots.plot(pp...)

                    p = Plots.plot(ppp,begin
                                                Plots.scatter(fill(0,1,1), 
                                                label = "Condition", 
                                                marker = :star8,
                                                markercolor = :black,
                                                linewidth = 0, 
                                                framestyle = :none, 
                                                legend = :inside)

                                                Plots.scatter!(fill(0,1,1), 
                                                label = "", 
                                                markerstrokewidth = 0, 
                                                framestyle = :none, 
                                                marker = :rect,
                                                markercolor = :white,
                                                legend = :inside)
                                            end, 
                                                layout = grid(2, 1, heights=[0.99, 0.01]),
                                                plot_title = "Model: "*𝓂.model_name*"        " * shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")
                    
                    if show_plots# & (length(pp) > 0)
                        display(p)
                    end

                    if save_plots# & (length(pp) > 0)
                        savefig(p, save_plots_path * "/conditional_fcst__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
                    end

                    pane += 1
                    pp = []
                end
            end
        end
        if length(pp) > 0

            shock_string = "Conditional forecast"
            shock_name = "conditional_forecast"

            # p = Plots.plot(pp...,plot_title = "Model: " * 𝓂.model_name * "        " * shock_string * "  (" * string(pane) * "/" * string(Int(ceil(n_subplots/plots_per_page))) * ")")

            ppp = Plots.plot(pp...)

            p = Plots.plot(ppp,begin
                                        Plots.scatter(fill(0,1,1), 
                                        label = "Condition", 
                                        marker = :star8,
                                        markercolor = :black,
                                        linewidth = 0, 
                                        framestyle = :none, 
                                        legend = :inside)

                                        Plots.scatter!(fill(0,1,1), 
                                        label = "", 
                                        markerstrokewidth = 0, 
                                        framestyle = :none, 
                                        marker = :rect,
                                        markercolor = :white,
                                        legend = :inside)
                                    end, 
                                        layout = grid(2, 1, heights=[0.99, 0.01]),
                                        plot_title = "Model: "*𝓂.model_name*"        " * shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")
            
            if show_plots
                display(p)
            end

            if save_plots
                savefig(p, save_plots_path * "/conditional_fcst__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end
end

using Plots, LaTeXStrings
import MacroModelling: Symbol_input,ℳ
plot_conditional_forecast(m, conditions, shocks = shocks)






full_SS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
   
full_SS[indexin(𝓂.aux,full_SS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

reference_steady_state = [s ∈ 𝓂.exo_present ? 0 : NSSS[s] for s in full_SS]

periods = 40
periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))
variables = :all_including_auxilliary
𝓂 = m
using Plots, LaTeXStrings
show_plots::Bool = true
save_plots::Bool = false
save_plots_format::Symbol = :pdf
save_plots_path::String = "."
plots_per_page::Int = 9
verbose = false


cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(full_SS),periods)
nzs = findnz(conditions)
for i in 1:length(nzs[1])
    cond_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
end
conditions = cond_tmp
cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
if length(cond_idx) > 0
    Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "Condition")
end
i = 4
SS = reference_steady_state[var_idx[i]]
Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
hline!([SS 0], color = :black, label = "")
Plots.scatter!([1],[1.0],label = "Condition")

    # plots = []
    default(size=(700,500),
            # leg = false,
            # plot_titlefont = (10, fontt), 
            # titlefont = (10, fontt), 
            # guidefont = (8, fontt), 
            plot_titlefont = (10), 
            titlefont = (10), 
            guidefont = (8), 
            legendfontsize = 8, 
            # tickfont = (8, fontt),
            # tickfontfamily = fontt,
            tickfontsize = 8,
            # tickfontrotation = 9,
            # rotation = 90,
            # tickfontvalign = :center,
            # topmargin = 10mm,
            # rightmargin = 17mm, 
            framestyle = :box)


            var_idx = parse_variables_input_to_index(variables, 𝓂.timings)


    # for i in 1:length(var_idx)
        n_subplots = length(var_idx)
        pp = []
        pane = 1
        plot_count = 1
        for i in 1:length(var_idx)
            if all(isapprox.(Y[i,:], 0, atol = eps(Float32)))
                n_subplots -= 1
            end
        end

        for i in 1:length(var_idx)
            SS = reference_steady_state[var_idx[i]]
            if !(all(isapprox.(Y[i,:],0,atol = eps(Float32))))
            # if !(plot_count ∈ unique(round.((1:𝓂.timings.timings.nVars)/plots_per_page))*plots_per_page)
                if !(plot_count % plots_per_page == 0)
                    plot_count += 1
                    if all((Y[i,:] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                        Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                        hline!([SS 0], color = :black, label = "") 
                                        Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)                             
                            end)
                        else
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                        Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                        hline!([SS 0], color = :black, label = "")                         
                            end)
                        end
                    else
                        cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        hline!([SS], color = :black, label = "")
                                        Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)   
                            end)
                        else
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        hline!([SS], color = :black, label = "")
                            end)
                        end

                    end
                else

                    plot_count = 1
                    if all((Y[i,:] .+ SS) .> eps(Float32)) & (SS > eps(Float32))
                        cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                    Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                    hline!([SS 0],color = :black,label = "")   
                                    Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)                            
                        end)
                    else
                        push!(pp,begin
                                    Plots.plot(1:periods, Y[i,:] .+ SS,title = string(𝓂.timings.var[var_idx[i]]),ylabel = "Level",label = "")
                                    Plots.plot!(twinx(),1:periods, 100*((Y[i,:] .+ SS) ./ SS .- 1), ylabel = L"\% \Delta", label = "")
                                    hline!([SS 0],color = :black,label = "")                              
                        end)
                    end
                    else
                        cond_idx = findall(conditions[var_idx[i],:] .!= nothing)
                        if length(cond_idx) > 0
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        hline!([SS], color = :black, label = "")
                                        Plots.scatter!(cond_idx,conditions[var_idx[i],cond_idx] .+ SS, label = "",marker = :star8, markercolor = :black)  
                            end)
                        else 
                            push!(pp,begin
                                        Plots.plot(1:periods, Y[i,:] .+ SS, title = string(𝓂.timings.var[var_idx[i]]), label = "", ylabel = "Level")#, rightmargin = 17mm)#,label = reshape(String.(𝓂.timings.solution.algorithm),1,:)
                                        hline!([SS], color = :black, label = "")
                            end)
                        end

                    end

                    shock_string = "Conditional forecast"
                    shock_name = "conditional_forecast"

                    ppp = Plots.plot(pp...)

                    p = Plots.plot(ppp,begin
                                                Plots.scatter(fill(0,1,1), 
                                                label = "Condition", 
                                                marker = :star8,
                                                markercolor = :black,
                                                linewidth = 0, 
                                                framestyle = :none, 
                                                legend = :inside)

                                                Plots.scatter!(fill(0,1,1), 
                                                label = "", 
                                                markerstrokewidth = 0, 
                                                framestyle = :none, 
                                                marker = :rect,
                                                markercolor = :white,
                                                legend = :inside)
                                            end, 
                                                layout = grid(2, 1, heights=[0.99, 0.01]),
                                                plot_title = "Model: "*𝓂.model_name*"        " * shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")
                    
                    if show_plots# & (length(pp) > 0)
                        display(p)
                    end

                    if save_plots# & (length(pp) > 0)
                        savefig(p, save_plots_path * "/conditional_fcst__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
                    end

                    pane += 1
                    pp = []
                end
            end
        end
        if length(pp) > 0

            shock_string = "Conditional forecast"
            shock_name = "conditional_forecast"

            # p = Plots.plot(pp...,plot_title = "Model: " * 𝓂.model_name * "        " * shock_string * "  (" * string(pane) * "/" * string(Int(ceil(n_subplots/plots_per_page))) * ")")

            ppp = Plots.plot(pp...)

            p = Plots.plot(ppp,begin
                                        Plots.scatter(fill(0,1,1), 
                                        label = "Condition", 
                                        marker = :star8,
                                        markercolor = :black,
                                        linewidth = 0, 
                                        framestyle = :none, 
                                        legend = :inside)

                                        Plots.scatter!(fill(0,1,1), 
                                        label = "", 
                                        markerstrokewidth = 0, 
                                        framestyle = :none, 
                                        marker = :rect,
                                        markercolor = :white,
                                        legend = :inside)
                                    end, 
                                        layout = grid(2, 1, heights=[0.99, 0.01]),
                                        plot_title = "Model: "*𝓂.model_name*"        " * shock_string *"  ("*string(pane)*"/"*string(Int(ceil(n_subplots/plots_per_page)))*")")
            
            if show_plots
                display(p)
            end

            if save_plots
                savefig(p, save_plots_path * "/conditional_fcst__" * 𝓂.model_name * "__" * shock_name * "__" * string(pane) * "." * string(save_plots_format))
            end
        end
    # end





𝓂 = m
C = @views 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

import LinearAlgebra as ℒ
findnz(conditions)
ℒ.det(C[6,1:2])
any(C[6,1:2] .!= 0)
any([0,0] .!= 0)
@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

get_solution(RBC_CME)

conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,3),Variables = [:c,:y], Periods = 1:3)
conditions[1,1] = .01
conditions[2,3] = .02

shocks = Matrix{Union{Nothing,Float64}}(undef,2,2)
shocks[2,2] = .05



conditions = Matrix{Union{Nothing,Float64}}(undef,7,2)
conditions[4,1] = .01
conditions[6,2] = .02

using SparseArrays
conditions = spzeros(7,2)
conditions[4,1] = .01
conditions[6,2] = .02



shocks = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),Variables = [:delta_eps], Periods = [1])
shocks[1,1] = .05

# using SparseArrays
# shocks = spzeros(2,1)
# shocks[1,1] = .05

get_conditional_forecast(RBC_CME,conditions, shocks = shocks)


std(RBC_CME)


𝓂 = RBC_CME
verbose = true
parameters = 𝓂.parameter_values

var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

solve!(𝓂, verbose = verbose)

write_parameters_input!(𝓂,parameters, verbose = verbose)

SS_and_pars, _ = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose)

∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)

𝑺₁ = calculate_first_order_solution(∇₁; T = 𝓂.timings)

# A = @views 𝑺₁[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[indexin(𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nVars),:]

C = @views 𝑺₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]


fcast = spzeros(size(C,1),5)
fcast[6,1] = .01
fcast[6,2] = .02
fcast[5,2] = .02

Y = zeros(size(C,1),size(fcast,2))
shocks = zeros(size(C,2),size(fcast,2))

shocks[:,1] = C[findnz(fcast[:,1])[1],:] \ collect(fcast[:,1] - A * zeros(size(A,1)))[findnz(fcast[:,1])[1]]
Y[:,1] = A * zeros(size(A,1)) + C * shocks[:,1]

for i in 2:size(fcast,2)
    shocks[:,i] = C[findnz(fcast[:,i])[1],:] \ (fcast[:,i] - A * Y[:,i-1])[findnz(fcast[:,i])[1]]
    Y[:,i] = A * Y[:,i-1] + C * shocks[:,i]
end


using AxisKeys
fcast = [nothing 1.0 0; .2 .3 nothing]
KeyedArray(fcast,Shock = 1:2, Period = 1:3)|>typeof

state_update = 𝓂.solution.perturbation.first_order.state_update
C = @views 𝑺₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]
shocks = spzeros(size(C,2),5)
shocks[1,1] = 1

fcast = spzeros(size(C,1),5)
fcast[6,1] = .01
fcast[5,1] = .03
fcast[6,2] = .02
fcast[5,2] = .02



Y = zeros(size(C,1),size(fcast,2))

cond_var_idx = findnz(fcast[:,1])[1]
free_shock_idx = setdiff(axes(C,2),findnz(shocks[:,1])[1])
if size(C[:,free_shock_idx],2) == length(cond_var_idx)
    @assert ℒ.det(C[cond_var_idx,free_shock_idx]) > eps(Float32) "Check restrictions in period 1."
end
@assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with more free shocks than conditioned variables. Period 1 has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."

shocks[free_shock_idx,1] = C[cond_var_idx,free_shock_idx] \ (fcast[cond_var_idx,1] - state_update(zeros(size(C,1)), collect(shocks[:,1]))[cond_var_idx])
Y[:,1] = state_update(zeros(size(C,1)), collect(shocks[:,1]))


for i in 2:size(fcast,2)
    cond_var_idx = findnz(fcast[:,i])[1]
    cond_shock_idx = findnz(shocks[:,i])[1]
    free_shock_idx = setdiff(axes(C,2),cond_shock_idx)

    if size(C[:,free_shock_idx],2) == length(cond_var_idx)
        @assert ℒ.det(C[cond_var_idx,free_shock_idx]) > eps(Float32) "Check restrictions in period " * i * "."
    end

    @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with more free shocks than conditioned variables. Period " * repr(i) * " has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."

    shocks[free_shock_idx,i] = C[cond_var_idx,free_shock_idx] \ (fcast[cond_var_idx,i] - state_update(Y[:,i-1], collect(shocks[:,i]))[cond_var_idx])

    Y[:,i] = state_update(Y[:,i-1], collect(shocks[:,i]))
end

Y

shocks

ℒ.lu(C[cond_var_idx,:]')
prod(ℒ.diag(C[cond_var_idx,:]))

ℒ.det(C[cond_var_idx,:])
ℒ.checksquare(C[cond_var_idx,:])

Matrix{Union{Nothing,Float64}}(undef, 2, 4)
using AxisKeys

conditions = zeros(7,5)
conditions[1,1] = .1

conditions = zeros(6,5)
conditions[1,1] = .1

conditions = spzeros(7,5)
conditions[1,1] = .1

conditions = spzeros(6,5)
conditions[1,1] = 0.0


conditions = Matrix{Union{Nothing,Float64}}(undef,7,5)
conditions[1,1] = .1

conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,5),Variables = [:c,:k], Periods= 1:5)
conditions[1,1] = .1
conditions[2,1] = .21



shocks = spzeros(2,5)
shocks[1,1] = 1



shocks = Matrix{Union{Nothing,Float64}}(undef,2,5)
shocks[1,1] = 1
shocks[2,1] = 1.2
shocks[1,2] = .9
shocks[2,2] = .2

Float64[shocks[1:2,1:2]...]

shocks = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,5),Variables = [:eps_z], Periods= 1:5)
shocks[1,1] = 1

# shocks = spzeros(2,5)
idxs = findnz(conditions)
cond_tmp[idxs[1],idxs[2]] .= idxs[3]

function get_conditional_forecast(𝓂::ℳ;
    conditions::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}},
    periods::Int = 40, 
    parameters = nothing,
    variables::Symbol_input = :all, 
    shocks::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}, Nothing} = nothing, 
    conditions_in_levels::Bool = false,
    levels::Bool = false,
    verbose = false)

    periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

    full_SS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))

    if conditions isa SparseMatrixCSC{Float64}
        @assert length(full_SS) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(full_SS)) * " variables (including auxilliary variables): " * repr(full_SS)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(full_SS),periods)
        cond_tmp[findnz(conditions)[1],findnz(conditions)[2]] .= findnz(conditions)[3]
        conditions = cond_tmp
    elseif conditions isa Matrix{Union{Nothing,Float64}}
        @assert length(full_SS) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(full_SS)) * " variables (including auxilliary variables): " * repr(full_SS)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(full_SS),periods)
        cond_tmp[:,axes(conditions,2)] = conditions
        conditions = cond_tmp
    elseif conditions isa KeyedArray{Union{Nothing,Float64}} || conditions isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(conditions,1),full_SS)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(axiskeys(conditions,1),full_SS))
        
        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(full_SS),periods)
        cond_tmp[indexin(sort(axiskeys(conditions,1)),full_SS),axes(conditions,2)] .= conditions(sort(axiskeys(conditions,1)))
        conditions = cond_tmp
    end

    if shocks isa SparseMatrixCSC{Float64}
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[findnz(shocks)[1],findnz(shocks)[2]] .= findnz(shocks)[3]
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        @assert length(setdiff(axiskeys(shocks,1),𝓂.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(axiskeys(shocks,1),𝓂.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[indexin(sort(axiskeys(shocks,1)),𝓂.exo),axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif shocks == nothing
        shocks = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
    end

    full_SS[indexin(𝓂.aux,full_SS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true)

    state_update = parse_algorithm_to_state_update(:first_order, 𝓂)

    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

    reference_steady_state = [s ∈ 𝓂.exo_present ? 0 : NSSS[s] for s in full_SS]

    if conditions_in_levels
        conditions .-= reference_steady_state
    end

    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    C = @views 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

    Y = zeros(size(C,1),periods)

    cond_var_idx = findall(conditions[:,1] .!= nothing)
    
    free_shock_idx = findall(shocks[:,1] .== nothing)

    if size(C[:,free_shock_idx],2) == length(cond_var_idx)
        @assert ℒ.det(C[cond_var_idx,free_shock_idx]) > eps(Float32) "Numerical stabiltiy issues for restrictions in period 1."
    elseif length(cond_var_idx) > 1
        lu_sol = try ℒ.lu(C[cond_var_idx,free_shock_idx]) catch end
        @assert isnothing(lu_sol) "Numerical stabiltiy issues for restrictions in period 1."
    end

    @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with more free shocks than conditioned variables. Period 1 has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."

    shocks[free_shock_idx,1] .= 0

    shocks[free_shock_idx,1] = C[cond_var_idx,free_shock_idx] \ (conditions[cond_var_idx,1] - state_update(zeros(size(C,1)), Float64[shocks[:,1]...])[cond_var_idx])

    Y[:,1] = state_update(zeros(size(C,1)), Float64[shocks[:,1]...])

    for i in 2:size(conditions,2)
        cond_var_idx = findall(conditions[:,i] .!= nothing)
        
        free_shock_idx = findall(shocks[:,i] .== nothing)
        shocks[free_shock_idx,i] .= 0

        if size(C[:,free_shock_idx],2) == length(cond_var_idx)
            @assert ℒ.det(C[cond_var_idx,free_shock_idx]) > eps(Float32) "Numerical stabiltiy issues for restrictions in period " * repr(i) * "."
        elseif length(cond_var_idx) > 1
            lu_sol = try ℒ.lu(C[cond_var_idx,free_shock_idx]) catch end
            @assert isnothing(lu_sol) "Numerical stabiltiy issues for restrictions in period " * repr(i) * "."
        end

        @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with more free shocks than conditioned variables. Period " * repr(i) * " has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."

        shocks[free_shock_idx,i] = C[cond_var_idx,free_shock_idx] \ (conditions[cond_var_idx,i] - state_update(Y[:,i-1], Float64[shocks[:,i]...])[cond_var_idx])

        Y[:,i] = state_update(Y[:,i-1], Float64[shocks[:,i]...])
    end

    return KeyedArray([levels ? (Y[var_idx,:] .+ reference_steady_state[var_idx]) : Y[var_idx,:]; convert(Matrix{Float64},shocks)];  Variables_and_shocks = [𝓂.timings.var[var_idx]; 𝓂.timings.exo], Periods = 1:periods)
end



conditions = zeros(7,5)
conditions[1,1] = .1

conditions = zeros(6,5)
conditions[1,1] = .1

conditions = spzeros(7,5)
conditions[1,1] = .01

conditions = spzeros(6,5)
conditions[1,1] = 0.0


conditions = Matrix{Union{Nothing,Float64}}(undef,7,5)
conditions[1,1] = .01

conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,5),Variables = [:c,:k], Periods= 1:5)
conditions[1,1] = .1
conditions[2,1] = .21



shocks = spzeros(2,5)
shocks[1,1] = 1



shocks = Matrix{Union{Nothing,Float64}}(undef,2,5)
shocks[1,2] = .1
# shocks[2,1] = 1.2
# shocks[1,2] = .9
# shocks[2,2] = .2

Float64[shocks[1:2,1:2]...]

shocks = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,5),Variables = [:eps_z], Periods= 1:5)
shocks[2,1] = 1


get_conditional_forecast(𝓂; conditions, shocks = shocks)


C[cond_var_idx,free_shock_idx]


convert(Matrix{Float64},shcks)
return KeyedArray(Y[var_idx,:,:];  Variables = T.var[var_idx], Periods = 1:periods, Shocks = shocks isa Symbol_input ? [T.exo[shock_idx]...] : [:Shock_matrix])
    

[Y;shocks]


    irfs =  irf(state_update, 
                initial_state, 
                𝓂.timings; 
                periods = periods, 
                shocks = shocks, 
                variables = variables, 
                negative_shock = negative_shock)
    if levels
        return irfs .+ reference_steady_state[var_idx]
    else
        return irfs
    end
end