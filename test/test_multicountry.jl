using MacroModelling
import MacroModelling as MM
import MacroTools: postwalk, @capture, unblock



@model RBC_multicountry begin
    for co in countries

        Y{co}[0] = ((LAMBDA{co}*K{co}(-{J})^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co}*Z{co}[-1]^(-nu{co}))^(-1/nu{co});
        K{co}[0] = (1-delta{co})*K{co}[-1] + S{co}[0];
        X{co}[0] =
        for lag in (-J+1):0
                + phi{co}*S{co}({lag})
        end
        ;

        A{co}[0] = (1-eta{co})*A{co}[-1] + N{co}[0];
        L{co}[0] = 1 - alpha{co}*N{co}[0] - (1-alpha{co})*eta{co}*A{co}[-1];

        # Utility multiplied by gamma
        # U{co} = (C{co}^mu{co}*L{co}^(1-mu{co}))^gamma{co};

        # FOC with respect to consumption
        psi{co}*mu{co}/C{co}[0] * U{co}[0] = LGM[0];

        # FOC with respect to labor
        # NOTE: this condition is only valid for alpha = 1
        psi{co}*(1-mu{co})/L{co}[0] * U{co}[0] * (-alpha{co}) = - LGM[0] * (1-theta{co})/N{co}[0] * (LAMBDA{co}[0] * K{co}(-{J})^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co})*Y{co}[0]^(1+nu{co});

        # FOC with respect to capital
        for lag in 0:(J-1)
            +beta{co}^{lag}*LGM(+{lag})*phi{co}
        end
        for lag in 1:J
            -beta{co}^{lag}*LGM(+{lag})*phi{co}*(1-delta{co})
        end
        = beta{co}^{J}*LGM(+{J})*theta{co}/K{co}*(LAMBDA{co}(+{J})*K{co}^theta{co}*N{co}(+{J})^(1-theta{co}))^(-nu{co})*Y{co}(+{J})^(1+nu{co});

        # FOC with respect to stock of inventories
        LGM=beta{co}*LGM(+1)*(1+sigma{co}*Z{co}^(-nu{co}-1)*Y{co}(+1)^(1+nu{co}));

        # Shock process
        if co == countries[1]
            define alt_co = countries[2]
        else
            define alt_co = countries[1]
        end
        (LAMBDA{co}-1) = rho{co}{co}*(LAMBDA{co}(-1)-1) + rho{co}{alt_co}*(LAMBDA{alt_co}(-1)-1) + E{co};


        NX{co} = (Y{co} - (C{co} + X{co} + Z{co} - Z{co}(-1)))/Y{co};

    end

    # World ressource constraint
    for co in countries
    +C_@{co} + X_@{co} + Z_@{co} - Z_@{co}(-1)
    end
        =
    for co in countries
    +Y_@{co}
    end
        ;

end









countries = [:EA,:US]

exxp = :(@model RBC_baseline begin
	c[0] ^ (-σ) = β * c[1] ^ (-σ) * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)

	ψ * c[0] ^ σ / (1 - l[0]) = w[0]

	k[0] = (1 - δ) * k[-1] + i[0]

	# y[0] = c[0] + i[0] + g[0]
    for s in [:T,:NT]
    for c ∈ [:EA,:US] y{c}{s}[0] end = for c ∈ [:EA,:US] C{c}{s}[0] + I{c}{s}[0] + alpha{s} * G{c}{s}[0] end
    end
	y[0] = z[0] * k[-1] ^ α * l[0] ^ (1 - α)

	w[0] = y[0] * (1 - α) / l[0]

	r[0] = y[0] * α * 4 / k[-1]

	z[0] = (1 - ρᶻ) + ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

	g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

end)

exxp |> dump




exxp = :(begin
	c[0] ^ (-σ) = β * c[1] ^ (-σ) * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)

	ψ * c[0] ^ σ / (1 - l[0]) = w[0]

	k[0] = (1 - δ) * k[-1] + i[0]

	# y[0] = c[0] + i[0] + g[0]
    for s in [:T,:NT]
    for c ∈ [:EA,:US] y{c}{s}[0] end = for c ∈ [:EA,:US] C{c}{s}[0] + I{c}{s}[0] + alpha{s} * G{c}{s}[0] end
    end
	y[0] = z[0] * k[-1] ^ α * l[0] ^ (1 - α)

	w[0] = y[0] * (1 - α) / l[0]

	r[0] = y[0] * α * 4 / k[-1]

	z[0] = (1 - ρᶻ) + ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

	g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

end)



tt = :(J+1)

index_variable = :J
index_variable ∈ MM.get_symbols(tt)


function replace_for_loop_indices(exxpr,index_variable,indices,concatenate)
    calls = []
    println(exxpr)
    for idx in indices
        push!(calls, postwalk(x -> begin
            x isa Expr ?
                x.head == :ref ?
                    @capture(x, name_{index_}[time_]) ?
                        index == index_variable ?
                            :($(Expr(:ref, Symbol(string(name) * "◖" * string(idx) * "◗"),time))) :
                        x :
                    @capture(x, name_[time_]) ?
                        index_variable ∈ MM.get_symbols(time) ?
                            :($(Expr(:ref, name,Meta.parse(replace(string(time),string(index_variable) =>idx))))) :
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

    if concatenate
        return :($(Expr(:call, :+, calls...)))
    else
        return calls
    end
end

dump(:(K{c}[J-1]))
idx =1
index_var = :J
r"\bJ\b"
Meta.parse(replace(string(:(x + J-1)),string(index_var) =>idx))
eval(Meta.parse(replace(string(:(J-1)),r"\b"*string(index_var)*"\b" =>idx)))
@capture(:(K[J-1]), name_[time_])
replace_for_loop_indices(:(K{k}[J-1]),:J,4:-1:1,true)





function parse_for_loops(equations_block)
    eqs = Expr[]
    for arg in equations_block.args
        println(arg)
        if isa(arg,Expr)
            parsed_eqs = postwalk(x -> begin
                    x isa Expr ? 
                        x.head == :for ?
                            x.args[2].args[2].head == :(=) ?# || (x.args[2].head == :block && x.args[2].args[2].head == :call) ? #equations inside for loops
                                replace_for_loop_indices(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[1]), 
                                                        eval(x.args[1].args[2]),
                                                        # [i isa QuoteNode ? i.value : i for i in x.args[1].args[2].args], 
                                                        false) :
                            replace_for_loop_indices(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    eval(x.args[1].args[2]),
                                                    # [i isa QuoteNode ? i.value : i for i in x.args[1].args[2].args], 
                                                    true)  : # for loop part of equation
                        x :
                    x
                end,
            arg)

            if parsed_eqs isa Expr
                push!(eqs,parsed_eqs)
            else
                for B in parsed_eqs
                    if B.head == :block
                        for b in B.args
                            if b isa Expr
                                push!(eqs,b)
                            end
                        end
                    else
                        push!(eqs,B)
                    end
                end
            end

        end
    end
    return Expr(:block,eqs...)
end

:(1:4)|>eval
replace_for_loop_indices(:(K[J-1]),:J,1:4,true)

exxp = :(begin
    g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

    for s in [:T,:NT]
        for c ∈ [:EA,:US] y{c}{s}[0] end = for c ∈ [:EA,:US] C{c}{s}[0] + I{c}{s}[0] + alpha{s} * G{c}{s}[0] end
    end

    for co ∈ [:EA,:US]

        Y{co}[0] = ((LAMBDA{co}*K{co}[-4]^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co}*Z{co}[-1]^(-nu{co}))^(-1/nu{co})

        (1-delta{co})*K{co}[-1] + S{co}[0] - K{co}[0]

        X{co}[0] = phi{co}*S{co}[0] + for lag in -4:0 phi{co} * S{co}[lag+0] end
    end
end)



exxp = :(begin
    # g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

    # for s in [:T,:NT]
    #     for c ∈ [:EA,:US] y{c}{s}[0] end = for c ∈ [:EA,:US] C{c}{s}[0] + I{c}{s}[0] + alpha{s} * G{c}{s}[0] end
    # end

    # for co ∈ [:EA,:US]

        # Y{co}[0] = ((LAMBDA{co}*K{co}[-4]^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co}*Z{co}[-1]^(-nu{co}))^(-1/nu{co})

        # (1-delta{co})*K{co}[-1] + S{co}[0] - K{co}[0]

        X{co}[0] = phi{co}*S{co}[0] + for lag ∈ -4:0 phi{co} * S{co}[lag+0] end
    # end
end)




:((1-delta{co})*K{co}[-1] + S{co}[0] - K{co}[0]) |> dump

exxp |> dump
exxp.args[2] |> dump
exxp.args[2].args[2] |> dump

parse_for_loops(exxp)




function replace_for_loop_indices(exxpr,index_variable,indices,concatenate)
    calls = []
    for idx in indices
        push!(calls, postwalk(x -> begin
            x isa Expr ?
                x.head == :ref ?
                @capture(x, name_{index_}[time_]) ?
                        index == index_variable ?
                            :($(Expr(:ref, Symbol(string(name) * "◖" * string(idx) * "◗"),time))) :
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

    if concatenate
        return :($(Expr(:call, :+, calls...)))
    else
        return calls
    end
end

function parse_for_loops(equations_block)
    eqs = Expr[]
    for arg in equations_block.args
        if isa(arg,Expr)
            parsed_eqs = postwalk(x -> begin
                    x isa Expr ? 
                        x.head == :for ?
                            x.args[2].args[2].head == :(=) ?
                                replace_for_loop_indices(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[1]), 
                                                        [i.value for i  in x.args[1].args[2].args], 
                                                        false) :
                            replace_for_loop_indices(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    [i.value for i  in x.args[1].args[2].args], 
                                                    true)  :
                        x :
                    x
                end,
            arg)
            if parsed_eqs isa Expr
                push!(eqs,parsed_eqs)
            # elseif parsed_eqs isa Array
            #     [push!(eqs,b) for B in parsed_eqs for b in B.args if b isa Expr]
            else
                push!(eqs,parsed_eqs...)
            end
        end
    end
    return Expr(:block,eqs...)
end







function parse_for_loops(equations_block)
    eqs = Expr[]
    for arg in equations_block.args
        if isa(arg,Expr)
            parsed_eqs = postwalk(x -> begin
                    x isa Expr ? 
                        x.head == :for ?
                            x.args[2].args[2].head == :(=) ?
                                replace_for_loop_indices(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[1]), 
                                                        [i.value for i  in x.args[1].args[2].args], 
                                                        false) :
                            replace_for_loop_indices(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    [i.value for i  in x.args[1].args[2].args], 
                                                    true)  :
                        x :
                    x
                end,
            arg)
            if parsed_eqs isa Expr
                push!(eqs,parsed_eqs)
            # elseif parsed_eqs isa Array
            #     [push!(eqs,b) for B in parsed_eqs for b in B.args if b isa Expr]
            else
                push!(eqs,parsed_eqs...)
            end
        end
    end
    return Expr(:block,eqs...)
end






eqs = Expr[]
        parsed_eqs = parse_for_loops(arg)
        if parsed_eqs isa Expr
            push!(eqs,parsed_eqs)
        else
            push!(eqs,parsed_eqs...)
        end
    end
end

Expr(:block,eqs...)

parse_for_loops(exxp)

exxp|>dump

MacroTools.postwalk(x -> begin
    x isa Expr ? 
        # x.head == :(=) ?
        #     for arg in x.args
        #             arg isa Expr ?
                        x.head == :for ?
                            x.args[2].args[2] == :(=) ?
                                replace_for_loop_indices(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[1]), 
                                                        [i.value for i  in x.args[1].args[2].args],
                                                        false)  :
                            x :
                        x :
            # end :
        # x :
    x
end,
exxpr)
exxpr |> dump
exxpr.args[2].args[2]

exxpr.args[2] |>dump

exxp.args[4].args[8]
exxp.args[4].args[8] |> dump



for c ∈ [:EA,:US] y{c}[0] end = for c ∈ [:EA,:US] C{c}[0] + I{c}[0] + G{c}[0] end

y_₎₍EA₎₍_[0] + y_₎₍US₎₍_[0] = C_₎₍EA₎₍_[0] + I_₎₍EA₎₍_[0] + G_₎₍EA₎₍_[0] + C_₎₍US₎₍_[0] + I_₎₍US₎₍_[0] + G_₎₍US₎₍_[0]

parsed_expr = :(for c ∈ [:EA] y{c}[0] end = for c ∈ [:EA] C{c}[0] + I{c}[0] + G{c}[0] end)

transalted = :(y_EA_[0] = C_EA_[0] + I_EA_[0] + G_EA_[0])

exxpr = :(C{c}[0] + I{c}[0] + alpha{c} * G{c}[0])
exxpr = :(Y{c}[0])


replace_for_loop_indices(:(Y{c}[0] * alpha{c} + 3),:c,[:EA,:US,:CN])





MacroTools.postwalk(x -> begin
    if isa(x, Expr) && x.head == :for
        # Extract symbols and their values
        c_values = x.args[1].args[2].args
        println(c_values)
        symbols = [arg.args[1] for arg in x.args[2].args if isa(arg, Expr) && arg.head == :ref]

        println(symbols)
        # Create new expressions
        left_expr = Expr(:call, :+, [:( $(Symbol("$(s)_₎₍$(c)₎₍_"))[0]) for s in symbols, c in c_values]...)
        right_expr = Expr(:call, :+, [:( $(Symbol("$(s)_₎₍$(c)₎₍_"))[0]) for s in symbols, c in c_values]...)
        
        # Return new assignment expression
        return :( $left_expr = $right_expr )
    else
        return x
    end
end, parsed_expr)


MacroTools.postwalk(x -> begin
if isa(x, Expr) && x.head == :for
    # Extract symbols and their values
    c_values = x.args[1].args[2].args
    symbols = [arg.args[1] for arg in x.args[2].args[1].args]

    # Create new expressions
    left_expr = Expr(:call, :+, [:( $(Symbol("$(s)_₎₍$(c)₎₍_"))[0]) for s in symbols, c in c_values]...)
    right_expr = Expr(:call, :+, [:( $(Symbol("$(s)_₎₍$(c)₎₍_"))[0]) for s in symbols, c in c_values]...)
    
    # Return new assignment expression
    return :( $left_expr = $right_expr )
else
    return x
end
end, parsed_expr)

using MacroTools

function transform_expression(expr)
    return MacroTools.postwalk(x -> begin
        if isa(x, Expr) && x.head == :for
            # Extract symbols and their values
            c_values = x.args[1].args[2].args
            symbols = [arg.args[1] for arg in x.args[2].args[1].args]

            # Create new expressions
            left_expr = Expr(:call, :+, [:( $(Symbol("$(s)_₎₍$(c)₎₍_"))[0] for s in symbols, c in c_values]...)
            right_expr = Expr(:call, :+, [:( $(Symbol("$(s)_₎₍$(c)₎₍_"))[0] for s in symbols, c in c_values]...)
            
            # Return new assignment expression
            return :( $left_expr = $right_expr )
        else
            return x
        end
    end, expr)
end
    

parsed_expr |> dump

parsed_expr isa Expr
parsed_expr.head

parsed_expr.args[1].args[1]



function replace_index(x)
    if isa(x, Expr) && x.head == :ref && length(x.args) > 1
        v = x.args[1]
        if isa(v, Expr) && v.head == :curly
            varname = string(v.args[1], "_EA_")
            new_expr = Expr(x.head, Symbol(varname), x.args[2:end]...)
        end
    end
end


outt = MacroTools.postwalk(x -> begin
    x isa Expr ? 
        # x.head == :(=) ? 
            x.head == :for ?
                # println(x) :
                # x.head == :ref ?
                x.args[2].head == :block ?
                x isa Expr ?
                    # x.head == :ref ?
                        println(x) :
                    # MacroTools.@capture(x.args[2].args[2].args[2], name{index_}[name_]) ?
                    #     :($name) :
                    # x :
                    # for arg in x.args[2].args
                    #     MacroTools.@capture(arg, name{index_}) ?
                    #         # println(:($name)) 
                    #         :($name) :
                    #     arg
                    # end :
                    x :
                x :
            x :
        # x :
    x
end,
exxp);

parsed_expr |> dump

parsed_expr.args[1] |> dump

parsed_expr.args[1].args[2] |> dump
parsed_expr.args[1].args[2].args[2] |> dump




outt = MacroTools.@capture(:(C{c}[0]),name{index_}[time_])
x
y


time

if isa(x, Expr) && x.head == :curly && length(x.args) > 0 && x.args[1] == :c
    return :(_EA_), :(_US_)
elseif isa(x, Symbol) && occursin("{c}", String(x))
    replaced = replace(String(x), "{c}" => "")
    return Symbol(replaced * "_EA_"), Symbol(replaced * "_US_")
else
    return x
end


function convert_to_ss_equation(eq::Expr)
    postwalk(x -> 
        x isa Expr ? 
            x.head == :(=) ? 
                Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                    x.head == :ref ?
                        occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 :
                x.args[1] : 
            x.head == :call ?
                x.args[1] == :* ?
                    x.args[2] isa Int ?
                        x.args[3] isa Int ?
                            x :
                        :($(x.args[3]) * $(x.args[2])) : # avoid 2X syntax. doesnt work with sympy
                    x :
                x :
            unblock(x) : 
        x,
    eq)
end




transformed_expr = MacroTools.postwalk(x -> begin
if isa(x, Expr) && x.head == :ref && isa(x.args[2], Expr) && x.args[2].head == :curly
    if x.args[1] == :y
        return :($(Symbol("y_₎₍", x.args[2].args[2], "₎₍_"))[0])
    elseif x.args[1] == :C
        return :($(Symbol("C_₎₍", x.args[2].args[2], "₎₍_"))[0])
    elseif x.args[1] == :I
        return :($(Symbol("I_₎₍", x.args[2].args[2], "₎₍_"))[0])
    elseif x.args[1] == :G
        return :($(Symbol("G_₎₍", x.args[2].args[2], "₎₍_"))[0])
    end
end
return x
end, parsed_expr)