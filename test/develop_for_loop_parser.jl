# using MacroModelling
import MacroModelling as MM
import MacroTools: postwalk, @capture, unblock

# TODOs
# parse lag indices
# nested indices

function replace_for_loop_indices(exxpr,index_variable,indices,concatenate)
    calls = []
    # println(exxpr)
    # println(typeof(exxpr))
    # println(index_variable)
    # println(typeof(index_variable))
    # println(indices)
    # println(concatenate)
    for idx in indices
        push!(calls, postwalk(x -> begin
            x isa Expr ?
                x.head == :ref ?
                    @capture(x, name_{index_}[time_]) ?
                        index == index_variable ?
                            :($(Expr(:ref, Symbol(string(name) * "◖" * string(idx) * "◗"),time))) :
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ MM.get_symbols(time) ?
                                :($(Expr(:ref, Expr(:curly,name,index), Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            x :
                        x :
                    @capture(x, name_[time_]) ?
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ MM.get_symbols(time) ?
                                :($(Expr(:ref, name, Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            x :
                        x :
                    x :
                @capture(x, name_{index_}) ?
                    index == index_variable ?
                        :($(Symbol(string(name) * "◖" * string(idx) * "◗"))) :
                    x :
                x :
            x
        end,
        exxpr))
    end
    # println(calls)
    if concatenate
        return :($(Expr(:call, :+, calls...)))
    else
        # return :($(Expr(:block,calls...)))
        return calls
    end
end


replace_for_loop_indices(:(phi{co} * S{co}[lag-1]),:lag,-4:0,true)
MM.get_symbols(:(lag))


function core_for_loop_removal_loop(arg)
    postwalk(x -> begin
                    x = unblock(x)
                    x isa Expr ?
                        x.head == :for ?
                            x.args[2] isa Array ?
                                length(x.args[2]) >= 1 ?
                                    [replace_for_loop_indices(X, Symbol(x.args[1].args[1]), eval(x.args[1].args[2]), false) for X in x.args[2]] :
                                #     begin 
                                #         xx = for X in x.args[2] 
                                #             yy = replace_for_loop_indices(X, 
                                #                 Symbol(x.args[1].args[1]), 
                                #                 eval(x.args[1].args[2]),
                                #                 false)
                                #             return yy
                                #         end
                                #         # println(xx)
                                #         xx
                                #         # println(Expr(x.args[1],x.args[2]...))
                                # #     for X in x.args[2]
                                # #         replace_for_loop_indices(unblock(X), 
                                # #                             Symbol(x.args[1].args[1]), 
                                # #                             eval(x.args[1].args[2]),
                                # #                             false)
                                # #     end
                                #     end :
                                x :
                            # begin
                            #     println(dump(x.args[2]))
                            #     x.args[2].head == :(=) || (x.args[2].head == :block && all([i isa Expr && i.head == :(=) for i in x.args[2].args]))
                            # end ?# || (x.args[2].head == :block && x.args[2].args[2].head == :call) ? #equations inside for loops
                            x.args[2].head ∉ [:(=), :block] ?
                                replace_for_loop_indices(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    eval(x.args[1].args[2]),
                                                    true) : # for loop part of equation
                            replace_for_loop_indices(unblock(x.args[2]), 
                                                Symbol(x.args[1].args[1]), 
                                                eval(x.args[1].args[2]),
                                                false) : # for loop part across equations
                        x :
                    x
                end,
    arg)
end


function parse_for_loops(equations_block)
    eqs = Expr[]
    for arg in equations_block.args
        # println(arg)
        if isa(arg,Expr)
            parsed_eqs = core_for_loop_removal_loop(arg)

            # println(parsed_eqs)
            if parsed_eqs isa Expr
                # if parsed_eqs.head == :for
                #     parsed_eqss = core_for_loop_removal_loop(parsed_eqs)
                #     println(parsed_eqss)
                # else
                    push!(eqs,unblock(parsed_eqs))
                # end
            elseif parsed_eqs isa Array
                for B in parsed_eqs
                    if B isa Array
                        for b in B
                            push!(eqs,unblock(b))
                        end
                    elseif B isa Expr
                        # println(B)
                        if B.head == :block
                            println(B)
                            for b in B.args
                                if b isa Expr
                                    push!(eqs,b)
                                end
                            end
                        end
                    else
                        push!(eqs,unblock(B))
                    end
                end
            end

        end
    end
    return Expr(:block,eqs...)
end





exxp = :(begin

    # for s in [:T,:NT]
    # g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

    # g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

    for s in [:T,:NT]
        for c ∈ [:EA,:US] y{c}{s}[0] end = for c ∈ [:EA,:US] C{c}{s}[0] + I{c}{s}[0] + alpha{s} * G{c}{s}[0] end
    end

    for s in [:T,:NT]
        for c ∈ [:EA,:US]
            # y{c}{s}[0] = C{c}{s}[0] + I{c}{s}[0] + alpha{s} * G{c}{s}[0] 
            y{c}{s}[0] = C{c}{s}[0] + I{c}{s}[0]
            # y{s}[0] = C{s}[0] + I{s}[0]
        end
    end

    for co ∈ [:EA,:US]

        Y{co}[0] = ((LAMBDA{co}*K{co}[-4]^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co}*Z{co}[-1]^(-nu{co}))^(-1/nu{co})

        (1-delta{co})*K{co}[-1] + S{co}[0] - K{co}[0]

        X{co}[0] = phi{co}*S{co}[0] + for lag ∈ -4:0 phi{co} * S{co}[lag+0] end
    end
end)


out = parse_for_loops(exxp);


exxxp = :(for s = [:T, :NT]
Any[:(y◖EA◗{s}[0] = C◖EA◗{s}[0] + I◖EA◗{s}[0]), :(y◖US◗{s}[0] = C◖US◗{s}[0] + I◖US◗{s}[0])]
end)


exxxp.args[2] |> unblock |> dump


exxxxxp = :(for s = [:T, :NT]
y◖EA◗{s}[0] = C◖EA◗{s}[0] + I◖EA◗{s}[0]
y◖US◗{s}[0] = C◖US◗{s}[0] + I◖US◗{s}[0]
end)

exxxxxp.args[2] |> unblock |> dump


dump(:(for s = [:T, :NT]
Any[:(y◖EA◗{s}[0] = C◖EA◗{s}[0] + I◖EA◗{s}[0]), :(y◖US◗{s}[0] = C◖US◗{s}[0] + I◖US◗{s}[0])]
end))


:(Any[quote
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:169 =#
Y◖EA◗[0] = ((LAMBDA◖EA◗ * K◖EA◗[-4] ^ theta◖EA◗ * N◖EA◗[0] ^ (1 - theta◖EA◗)) ^ -nu◖EA◗ + sigma◖EA◗ * Z◖EA◗[-1] ^ -nu◖EA◗) ^ (-1 / nu◖EA◗)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:171 =#
((1 - delta◖EA◗) * K◖EA◗[-1] + S◖EA◗[0]) - K◖EA◗[0]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:173 =#
X◖EA◗[0] = phi◖EA◗ * S◖EA◗[0] + (phi◖EA◗ * S◖EA◗[-4 + 0] + phi◖EA◗ * S◖EA◗[-3 + 0] + phi◖EA◗ * S◖EA◗[-2 + 0] + phi◖EA◗ * S◖EA◗[-1 + 0] + phi◖EA◗ * S◖EA◗[0 + 0])
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:174 =#
end, quote
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:169 =#
Y◖US◗[0] = ((LAMBDA◖US◗ * K◖US◗[-4] ^ theta◖US◗ * N◖US◗[0] ^ (1 - theta◖US◗)) ^ -nu◖US◗ + sigma◖US◗ * Z◖US◗[-1] ^ -nu◖US◗) ^ (-1 / nu◖US◗)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:171 =#
((1 - delta◖US◗) * K◖US◗[-1] + S◖US◗[0]) - K◖US◗[0]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:173 =#
X◖US◗[0] = phi◖US◗ * S◖US◗[0] + (phi◖US◗ * S◖US◗[-4 + 0] + phi◖US◗ * S◖US◗[-3 + 0] + phi◖US◗ * S◖US◗[-2 + 0] + phi◖US◗ * S◖US◗[-1 + 0] + phi◖US◗ * S◖US◗[0 + 0])
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:174 =#
end])

:(quote
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:169 =#
Y◖EA◗[0] = ((LAMBDA◖EA◗ * K◖EA◗[-4] ^ theta◖EA◗ * N◖EA◗[0] ^ (1 - theta◖EA◗)) ^ -nu◖EA◗ + sigma◖EA◗ * Z◖EA◗[-1] ^ -nu◖EA◗) ^ (-1 / nu◖EA◗)
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:171 =#
((1 - delta◖EA◗) * K◖EA◗[-1] + S◖EA◗[0]) - K◖EA◗[0]
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:173 =#
X◖EA◗[0] = phi◖EA◗ * S◖EA◗[0] + (phi◖EA◗ * S◖EA◗[-4 + 0] + phi◖EA◗ * S◖EA◗[-3 + 0] + phi◖EA◗ * S◖EA◗[-2 + 0] + phi◖EA◗ * S◖EA◗[-1 + 0] + phi◖EA◗ * S◖EA◗[0 + 0])
#= /Users/thorekockerols/GitHub/MacroModelling.jl/test/develop_for_loop_parser.jl:174 =#
end) |> unblock |> dump

dump(:(for s = [:T, :NT]
y◖EA◗{s}[0] = C◖EA◗{s}[0] + I◖EA◗{s}[0]
y◖US◗{s}[0] = C◖US◗{s}[0] + I◖US◗{s}[0]
end))



parse_for_loops(:(for s = [:T, :NT]
Any[:(y◖EA◗{s}[0] = C◖EA◗{s}[0] + I◖EA◗{s}[0]), :(y◖US◗{s}[0] = C◖US◗{s}[0] + I◖US◗{s}[0])]
end))

dump(:(for s = [:T, :NT]
Any[:(y◖EA◗{s}[0] = C◖EA◗{s}[0] + I◖EA◗{s}[0]), :(y◖US◗{s}[0] = C◖US◗{s}[0] + I◖US◗{s}[0])]
end))



dump(:(for s = [:T, :NT]
y◖EA◗{s}[0] = C◖EA◗{s}[0] + I◖EA◗{s}[0]
y◖US◗{s}[0] = C◖US◗{s}[0] + I◖US◗{s}[0]
end))
exxp |> unblock |> dump
exxp.args[2] |> dump
exxp.args[2].args[2] |> dump

Expr(:block,)


out|>unblock
dump(:(Any[:(y◖EA◗{s}[0] = C◖EA◗{s}[0] + I◖EA◗{s}[0]), :(y◖US◗{s}[0] = C◖US◗{s}[0] + I◖US◗{s}[0])]))