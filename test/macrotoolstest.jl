import MacroTools: postwalk
using MacroTools
using Subscripts



ex = :(begin
    R[0] * beta = C[1] / C[0]
    C[0] = w[0] * L[0] - B[0] + R[-1] * B[-1] + (1-v) * (Rk[-1] * Q[-1] * K[-1] - (R[-1] + mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1])) * (Q[-1] * K[-1] - N[-1])) - We 
    w[0] = C[0] / (1-L[0])
    K[0] = (1-delta) * K[-1] + I[0] 
    Q[0] = 1 + chi * (I[0] / K[-1] - delta)
    Y[0] = (A[0] * K[-1])^alpha * L[0]^(1-alpha)
    Rk[-1] = (alpha * Y[0] / K[-1] + Q[0] * (1-delta))/Q[-1]
    w[0] = (1-alpha) * Y[0] / L[0]
    N[0] = v * (Rk[-1] * Q[-1] * K[-1] - (R[-1] + mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1])) * (Q[-1] * K[-1] - N[-1])) + We 
    0 = (omegabar[0] * (1 - F[0]) + (1 - mu) * G[0]) * Rk[0] / R[0] * Q[0] * K[0] / N[0] - (Q[0] * K[0] / N[0] - 1)
    0 = (1 - (omegabar[0] * (1 - F[0]) + G[0])) * Rk[0] / R[0] + (1 - F[0]) / (1 - F[0] - omegabar[0] * mu * (normpdf((log(omegabar[0]) + sigma^2/2)  / sigma)/ omegabar[0]  / sigma)) * ((omegabar[0] * (1 - F[0]) + (1 - mu) * G[0]) * Rk[0] / R[0] - 1) 
    G[0] = normcdf(((log(omegabar[0])+sigma^2/2)/sigma) - sigma)
    F[0] = normcdf((log(omegabar[0])+sigma^2/2)/sigma)
    EFP[0] = (mu * G[0] * Rk[-1] * Q[-1] * K[-1] / (Q[-1] * K[-1] - N[-1]))
    Y[0] + walras[0] = C[0] + I[0] + EFP[0] * (Q[-1] * K[-1] - N[-1])
    B[0] = Q[0] * K[0] - N[0]  
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end)

Expr(:call,:log,:K)
dump(:((log(omegabar[0])+sigma^2/2)/sigma))

# dump(:((A[0] * K[-1])^alpha * L[0]^(1-alpha)))

# dump(:(K[-1]))

# Expr(:ref,:K,-1)




lower_bounds = []
upper_bounds = []
bounded_vars = []
nonnegativity_aux_vars = []
ss_and_aux_equations = []


for (i,arg) in enumerate(ex.args)
    if isa(arg,Expr)
        # find nonegative variables, parameters, or terms
        eqs = MacroTools.postwalk(x -> 
            x isa Expr ? 
                x.head == :(=) ? 
                    Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                        x.head == :ref ?
                            occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
                    x : 
                x.head == :call ?
                    x.args[1] == :* ?
                        x.args[2] isa Int ?
                            x.args[3] isa Int ?
                                x :
                            :($(x.args[3]) * $(x.args[2])) :
                        x :
                    x.args[1] ∈ [:^, :log] ?
                        x.args[2] isa Symbol ? # nonnegative parameters 
                            begin
                                if length(intersect(bounded_vars,[x.args[2]])) == 0
                                    push!(lower_bounds,eps())
                                    push!(upper_bounds,Inf)
                                    push!(bounded_vars,x.args[2]) 
                                end
                                x
                            end :
                        x.args[2].head == :ref ?
                            x.args[2].args[1] isa Symbol ? # nonnegative variables 
                                begin
                                    if length(intersect(bounded_vars,[x.args[2].args[1]])) == 0
                                        push!(lower_bounds,eps())
                                        push!(upper_bounds,Inf)
                                        push!(bounded_vars,x.args[2].args[1]) 
                                    end
                                    x
                                end :
                            x :
                        x.args[2].head == :call ? # nonnegative expressions
                            begin
                                push!(lower_bounds,eps())
                                push!(upper_bounds,Inf)
                                push!(bounded_vars,:($(Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))))))
                                push!(ss_and_aux_equations, Expr(:call,:-, :($(Expr(:ref,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))),0))), x.args[2])) # take position of equation in order to get name of vars which are being replaced and substitute accordingly or rewrite to have substitutuion earlier i the code
                                push!(nonnegativity_aux_vars,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)+1))))
                                :($(Expr(:ref,Symbol("nonnegativity_auxilliary" * sub(string(length(nonnegativity_aux_vars)))),0)) ^ $(x.args[3]))
                            end :
                        x :
                    x :
                x :
            x,
        ex.args[i])

        push!(ss_and_aux_equations,unblock(eqs))
    end
end


ss_aux_equations = []

# tag vars and pars in changed SS equations
var_list_aux_SS = []
ss_list_aux_SS = []
par_list_aux_SS = []

var_future_list_aux_SS = []
var_present_list_aux_SS = []
var_past_list_aux_SS = []

# # label all variables parameters and exogenous variables and timings for changed SS equations including nonnegativity auxilliary variables
for eq in ss_and_aux_equations
    var_tmp = Set()
    ss_tmp = Set()
    par_tmp = Set()
    var_future_tmp = Set()
    var_present_tmp = Set()
    var_past_tmp = Set()

    # label all variables parameters and exogenous variables and timings for individual equations
    postwalk(x -> 
        x isa Expr ? 
            x.head == :call ? 
                for i in 2:length(x.args)
                    x.args[i] isa Symbol ? 
                        occursin(r"^(ss|stst|steady|steadystate|steady_state|x|ex|exo|exogenous){1}$"i,string(x.args[i])) ? 
                            x :
                        push!(par_tmp,x.args[i]) : 
                    x
                end :
            x.head == :ref ? 
                x.args[2] isa Int ? 
                    x.args[2] == 0 ? 
                        push!(var_present_tmp,x.args[1]) : 
                    x.args[2] > 0 ? 
                        push!(var_future_tmp,x.args[1]) : 
                    x.args[2] < 0 ? 
                        push!(var_past_tmp,x.args[1]) : 
                    x :
                occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\-{1}\s{1}\d+$))"i,string(x.args[2])) ?
                    push!(var_past_tmp,x.args[1]) : 
                occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\+{1}\s{1}\d+$))"i,string(x.args[2])) ?
                    push!(var_future_tmp,x.args[1]) : 
                occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    push!(ss_tmp,x.args[1]) :
                x : 
            x :
        x,
    eq)

    var_tmp = union(var_future_tmp,var_present_tmp,var_past_tmp)
    
    push!(var_list_aux_SS,var_tmp)
    push!(ss_list_aux_SS,ss_tmp)
    push!(par_list_aux_SS,par_tmp)
    push!(var_future_list_aux_SS,var_future_tmp)
    push!(var_present_list_aux_SS,var_present_tmp)
    push!(var_past_list_aux_SS,var_past_tmp)


    # write down SS equations including nonnegativity auxilliary variables
    prs_ex = postwalk(x -> 
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
                        :($(x.args[3]) * $(x.args[2])) :
                    x :
                x :
            unblock(x) : 
        x,
    eq)
    
    push!(ss_aux_equations,unblock(prs_ex))

end



convert(Expr,ss_aux_equations[1])

var_list_aux_SS
ss_list_aux_SS
par_list_aux_SS
var_future_list_aux_SS
var_present_list_aux_SS
var_past_list_aux_SS


lower_bounds
upper_bounds
bounded_vars







using MacroModelling

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    # A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta)*k[-1]
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    c_normcdf[0]= normcdf(c[0])
    c_normpdf[0]= normpdf(c[0])
    c_norminvcdf[0]= norminvcdf(c[0]-1)
    c_norminv[0]= norminv(c[0]-1)
    c_qnorm[0]= qnorm(c[0]-1)
    c_dnorm[0]= dnorm(c[0])
    c_pnorm[0]= pnorm(c[0])
    c_normlogpdf[0]= normlogpdf(c[0])
    # c_norm[0]= cdf(Normal(),c[0])
    c_inv[0] = erfcinv(c[0])
    # c_binomlogpdf[0]= binomlogpdf(c[0])
    # A[0]=exp(z[0])
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
    # ZZ_avg[0] = (A[0] + A[-1] + A[-2] + A[-3]) / 4
    # A_annual[0] = (A[0] + A[-4] + A[-8] + A[-12]) / 4
    # y_avg[0] = log(y[0] / y[-4])
    # y_growth[0] = log(y[1] / y[2])
    # y_growthl[0] = log(y[0] / y[1])
    # y_growthl1[0] = log(y[-1] / y[0])
    # log(A[0]) = rhoz * log(A[-1]) + std_eps * eps_z[x]
end
RBC_CME.var_present_list_aux_SS
RBC_CME.bounded_vars
RBC_CME.ss_aux_equations

using SpecialFunctions

erfcinv(-1)