using Revise
import MacroTools: unblock, postwalk, @capture, flatten
import MacroModelling: parse_for_loops, write_out_for_loops, replace_indices, replace_indices_inside_for_loop, get_symbols

mm_call = :(
    a = 1;
    for co in [a] 
        log(R_t{co}[0] / R_ts{co}) = 0.8 * log(R_t{co}[-1] / R_ts{co}) + 0.3 * log(pi_cpi_t{co}[-1]) # Modified Taylor-rule

        K_t{co}[0] = (1 - delta) * K_t{co}[-1] + I_t{co}[0] # Law of motion for capital

        lambda_t{co}[0] = C_t{co}[0] ^ (-sig) # FOC wrt consumption

        1 = R_t{co}[0] * betta * lambda_t{co}[1] / lambda_t{co}[0] / pi_cpi_t{co}[1] # HH FOC wrt bonds

        1 = betta * lambda_t{co}[1] / lambda_t{co}[0] * (rk_t{co}[1] + (1 - delta) * PI_t{co}[1]) / PI_t{co}[0] # No arbitrage cond. plus FOC wrt investment/capital

        lambda_t{co}[0] * w_t{co}[0] = kappaN{co} * N_t{co}[0] ^ lab{co} # HH FOC wrt labor

        1 = (for se in 1:10 Psi_con{se}{co} * P_t{se}{co}[0] ^ (( - sigc{co}) / (1 - sigc{co})) end) ^ (( - (1 - sigc{co})) / sigc{co}) # Price index derived from opt. problem of cons.-bundler 

        pi_cpi_t{co}[0] = (for se in 1:10 Psi_con{se}{co} * (pi_ppi_t{se}{co}[0] * P_t{se}{co}[-1]) ^ (( - sigc{co}) / (1 - sigc{co})) end) ^ (( - (1 - sigc{co})) / sigc{co}) # CPI-inflation 

        PI_t{co}[0] = (for se in 1:10 Psi_inv{se}{co} * P_t{se}{co}[0] ^ (( - sigi{co}) / (1 - sigi{co})) end) ^ (( - (1 - sigi{co})) / sigi{co}) # Price index derived from opt. problem of inv.-bundler 

        w_t{co}[0] = (for se in 1:10 omega_N{se}{co} * w_t{se}{co}[0] ^ (( - upsi_N{co}) / (1 - upsi_N{co})) end) ^ (( - (1 - upsi_N{co})) / upsi_N{co}) # Wage index derived from opt. problem of labor-bundler

        rk_t{co}[0] = (for se in 1:10 omega_K{se}{co} * rk_t{se}{co}[0] ^ (( - upsi_K{co}) / (1 - upsi_K{co})) end) ^ (( - (1 - upsi_K{co})) / upsi_K{co}) # Rental rate index derived from opt. problem of capital-bundler

        Y_VA_t{co}[0] = C_t{co}[0] + I_t{co}[0] * PI_t{co}[0]

        Y_t{co}[0] = for se in 1:10 Y_t{se}{co}[0] end

        #### Consumption and investmend demand for different goods, relative prices ####
        for se in 1:10
            C_t{se}{co}[0] = C_t{co}[0] * Psi_con{se}{co} * (1 / P_t{se}{co}[0]) ^ (1 / (1 - sigc{co})) # Specific demand for sectoral consumption goods

            I_t{se}{co}[0] = I_t{co}[0] * Psi_inv{se}{co} * (PI_t{co}[0] / P_t{se}{co}[0]) ^ (1 / (1 - sigi{co})) # Specific demand for sectoral investment goods

            N_t{se}{co}[0] = N_t{co}[0] * omega_N{se}{co} * (w_t{co}[0] / w_t{se}{co}[0]) ^ (1 / (1 - upsi_N{co})) # Specific demand for labor types

            K_t{se}{co}[0] = K_t{co}[0] * omega_K{se}{co} * (rk_t{co}[1] / rk_t{se}{co}[1]) ^ (1 / (1 - upsi_K{co})) # Specific demand for capital types   

            Y_t{se}{co}[0] = C_t{se}{co}[0] + I_t{se}{co}[0] + Y_t{se}{co}[0] * kappap{se}{co} / 2 * (pi_ppi_t{se}{co}[1] - 1) ^ 2 + for se2 in 1:10 H{se2}_t{se}{co}[0] end

            Y_t{se}{co}[0] = epsi_t{co}[0] * epsi_t{se}{co}[0] * (1 - Pen_t{se}{co}[0]) * (N_t{se}{co}[0] ^ alphaN{se}{co} * K_t{se}{co}[-1] ^ (1 - alphaN{se}{co})) ^ alphaH{se}{co} * H_t{se}{co}[0] ^ (1 - alphaH{se}{co}) # Production technology of perfectly competitive firm

            EM_cost_t{se}{co}[0] = P_EM_t{co}[0] * (1 + shock_epsi_carb_int_t{co}[x]) * carb_int{se}{co} # Emissions costs

            mc_tild_t{se}{co}[0] = EM_cost_t{se}{co}[0] + mc_t{se}{co}[0] # Real marginal costs relevant for pricing (including emissions taxes)

            1 - thetap{se}{co} + mc_tild_t{se}{co}[0] * thetap{se}{co} / P_t{se}{co}[0] + (pi_ppi_t{se}{co}[1] - 1) * betta * lambda_t{co}[1] / lambda_t{co}[0] * kappap{se}{co} * pi_ppi_t{se}{co}[1] ^ 2 / pi_cpi_t{co}[1] * Y_t{se}{co}[1] / Y_t{se}{co}[0] = pi_ppi_t{se}{co}[0] * kappap{se}{co} * (pi_ppi_t{se}{co}[0] - 1) + (1 - thetap{se}{co}) * kappap{se}{co} / 2 * (pi_ppi_t{se}{co}[0] - 1) ^ 2 # Firm's pricing decision in case of nominal price rigidities

            w_t{se}{co}[0] = Y_t{se}{co}[0] * mc_t{se}{co}[0] * alphaN{se}{co} * alphaH{se}{co} / N_t{se}{co}[0] # FOC firm cost minimization -> labor

            rk_t{se}{co}[0] = Y_t{se}{co}[0] * mc_t{se}{co}[0] * (1 - alphaN{se}{co}) * alphaH{se}{co} / K_t{se}{co}[-1] # FOC firm cost minimization -> capital

            PH_t{se}{co}[0] = Y_t{se}{co}[0] * (1 - alphaH{se}{co}) * mc_t{se}{co}[0] / H_t{se}{co}[0] # FOC firm cost minimization -> intermediates

            PH_1_t{co}[0] = (for se2 in 1:10 Psi_1_1{co} * P_1_t{co}[0] ^ (( - sigh_1{co}) / (1 - sigh_1{co})) end) ^ (( - (1 - sigh_1{co})) / sigh_1{co}) # Price index derived from opt. problem of intermediate input bundler in sector se

            for se2 in 1:10 
                H{se}{se2}_t{co}[0] = H_t{se}{co}[0] * Psi{se}{se2}{co} * (PH_t{se}{co}[0] / P{se2}_t{co}[0]) ^ (1 / (1 - sigh{se}{co})) # Optimal amount of sector se2 output in bundle used in sector se 
            end

            log(epsi_t{se}{co}[0] / epsi_ts{se}{co}) = rho_eps{co} * log(epsi_t{se}{co}[-1] / epsi_ts{se}{co}) - shock_epsi_t{se}{co}[x] # Law of motion for sectoral TFP
        end
    end)





    out = postwalk(x -> begin
                    x = unblock(x)
                    x isa Expr ?
                        x.head == :for ?
                            x.args[2] isa Array ?
                                length(x.args[2]) >= 1 ?
                                    x.args[1].head == :block ?
                                        [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] :
                                    [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] :
                                x :
                            x.args[2].head ∉ [:(=), :block] ?
                                x.args[1].head == :block ?
                                    replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[2].args[1]), 
                                                        (x.args[1].args[2].args[2]),
                                                        true,
                                                        x.args[1].args[1].args[2].value) : # for loop part of equation
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    (x.args[1].args[2]),
                                                    true,
                                                    :+) : # for loop part of equation
                            x.args[1].head == :block ?
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[2].args[1]), 
                                                    (x.args[1].args[2].args[2]),
                                                    false,
                                                    x.args[1].args[1].args[2].value) : # for loop part of equation
                            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                Symbol(x.args[1].args[1]), 
                                                (x.args[1].args[2]),
                                                false,
                                                :+) :
                        x :
                    x
                end,
    arg)




mms_call = :(
    a = 1;
    #### Domestic aggregates ####
    for co in [a,b] 
        #### Consumption and investment demand for different goods, relative prices ####
        for se in 1:10
            #### Multi-country part ####
            for co2 in [a,b]
                for se2 in 1:10
                    H_t{se}{se2}{co}{co2}#[0] # = hb_hhh{se}{se2}{co}{co2} * (PHH_t{se}{se2}{co}[0] / P_t{se2}{co}{co2}[0]) ^ (1 / (1-sighh{se}{co})) * H_t{se}{se2}{co}[0]
                end

                # I_t{se}{co}{co2}[0] = hb_inv{se}{co}{co2} * (PI_t{se}{co}[0] / P_t{se}{co}{co2}[0]) ^ (1 / (1 - sigi{se}{co})) * I_t{se}{co}[0]

                # C_t{se}{co}{co2}[0] = hb_con{se}{co}{co2} * (PC_t{se}{co}[0] / P_t{se}{co}{co2}[0]) ^ (1 / (1 - sigc{se}{co})) * C_t{se}{co}[0]
            end
        end
    end
    )
    

    model_ex = parse_for_loops(mms_call) |> flatten
    model_ex.args[2]
    parsed_eqs = write_out_for_loops(mms_call.args[3]);
    parsed_eqs[1].args[4]

    replace_indices(parsed_eqs.args[1])
    

    aaa = :(for co = [a]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/fix_macros.jl:118 =#
    log(R_t{co}[0] / R_ts{co}) = 0.8 * log(R_t{co}[-1] / R_ts{co}) + 0.3 * log(pi_cpi_t{co}[-1])
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/fix_macros.jl:123 =#
    Any[:(C_t{1}{co}[0] = C_t{co}[0] * Psi_con{1}{co} * (1 / P_t{1}{co}[0]) ^ (1 / (1 - sigc{co}))), :(C_t{2}{co}[0] = C_t{co}[0] * Psi_con{2}{co} * (1 / P_t{2}{co}[0]) ^ (1 / (1 - sigc{co})))]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/fix_macros.jl:134 =#
end)

aaa |> typeof
aaa.head
aaa.args[1]
aaa.args[2] |> flatten
aaa.args[2].args[4] |> typeof



aaa = :(begin
begin
    #= REPL[393]:3 =#
    Y◖H◗[0] = ((LAMBDA◖H◗[0] * K◖H◗[-4] ^ theta◖H◗ * N◖H◗[0] ^ (1 - theta◖H◗)) ^ -nu◖H◗ + sigma◖H◗ * Z◖H◗[-1] ^ -nu◖H◗) ^ (-1 / nu◖H◗)
    #= REPL[393]:5 =#
    K◖H◗[0] = (1 - delta◖H◗) * K◖H◗[-1] + S◖H◗[0]
    #= REPL[393]:7 =#
    X◖H◗[0] = phi◖H◗ * S◖H◗[-3] + phi◖H◗ * S◖H◗[-2] + phi◖H◗ * S◖H◗[-1] + phi◖H◗ * S◖H◗[0]
    #= REPL[393]:9 =#
    A◖H◗[0] = (1 - eta◖H◗) * A◖H◗[-1] + N◖H◗[0]
    #= REPL[393]:11 =#
    L◖H◗[0] = (1 - alpha◖H◗ * N◖H◗[0]) - (1 - alpha◖H◗) * eta◖H◗ * A◖H◗[-1]
    #= REPL[393]:13 =#
    U◖H◗[0] = (C◖H◗[0] ^ mu◖H◗ * L◖H◗[0] ^ (1 - mu◖H◗)) ^ gamma◖H◗
    #= REPL[393]:15 =#
    ((psi◖H◗ * mu◖H◗) / C◖H◗[0]) * U◖H◗[0] = LGM[0]
    #= REPL[393]:17 =#
    ((psi◖H◗ * (1 - mu◖H◗)) / L◖H◗[0]) * U◖H◗[0] * -alpha◖H◗ = ((-(LGM[0]) * (1 - theta◖H◗)) / N◖H◗[0]) * (LAMBDA◖H◗[0] * K◖H◗[-4] ^ theta◖H◗ * N◖H◗[0] ^ (1 - theta◖H◗)) ^ -nu◖H◗ * Y◖H◗[0] ^ (1 + nu◖H◗)
    #= REPL[393]:19 =#
    (beta◖H◗ ^ 0 * LGM[0] * phi◖H◗ + beta◖H◗ ^ 1 * LGM[1] * phi◖H◗ + beta◖H◗ ^ 2 * LGM[2] * phi◖H◗ + beta◖H◗ ^ 3 * LGM[3] * phi◖H◗) + (-(beta◖H◗ ^ 1) * LGM[1] * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 2) * LGM[2] * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 3) * LGM[3] * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 4) * LGM[4] * phi◖H◗ * (1 - delta◖H◗)) = ((beta◖H◗ ^ 4 * LGM[4] * theta◖H◗) / K◖H◗[0]) * (LAMBDA◖H◗[4] * K◖H◗[0] ^ theta◖H◗ * N◖H◗[4] ^ (1 - theta◖H◗)) ^ -nu◖H◗ * Y◖H◗[4] ^ (1 + nu◖H◗)
    #= REPL[393]:26 =#
    LGM[0] = beta◖H◗ * LGM[1] * (1 + sigma◖H◗ * Z◖H◗[0] ^ (-nu◖H◗ - 1) * Y◖H◗[1] ^ (1 + nu◖H◗))
    #= REPL[393]:28 =#
    NX◖H◗[0] = (Y◖H◗[0] - ((C◖H◗[0] + X◖H◗[0] + Z◖H◗[0]) - Z◖H◗[-1])) / Y◖H◗[0]
    #= REPL[393]:29 =#
    #= REPL[393]:3 =#
    Y◖F◗[0] = ((LAMBDA◖F◗[0] * K◖F◗[-4] ^ theta◖F◗ * N◖F◗[0] ^ (1 - theta◖F◗)) ^ -nu◖F◗ + sigma◖F◗ * Z◖F◗[-1] ^ -nu◖F◗) ^ (-1 / nu◖F◗)
    #= REPL[393]:5 =#
    K◖F◗[0] = (1 - delta◖F◗) * K◖F◗[-1] + S◖F◗[0]
    #= REPL[393]:7 =#
    X◖F◗[0] = phi◖F◗ * S◖F◗[-3] + phi◖F◗ * S◖F◗[-2] + phi◖F◗ * S◖F◗[-1] + phi◖F◗ * S◖F◗[0]
    #= REPL[393]:9 =#
    A◖F◗[0] = (1 - eta◖F◗) * A◖F◗[-1] + N◖F◗[0]
    #= REPL[393]:11 =#
    L◖F◗[0] = (1 - alpha◖F◗ * N◖F◗[0]) - (1 - alpha◖F◗) * eta◖F◗ * A◖F◗[-1]
    #= REPL[393]:13 =#
    U◖F◗[0] = (C◖F◗[0] ^ mu◖F◗ * L◖F◗[0] ^ (1 - mu◖F◗)) ^ gamma◖F◗
    #= REPL[393]:15 =#
    ((psi◖F◗ * mu◖F◗) / C◖F◗[0]) * U◖F◗[0] = LGM[0]
    #= REPL[393]:17 =#
    ((psi◖F◗ * (1 - mu◖F◗)) / L◖F◗[0]) * U◖F◗[0] * -alpha◖F◗ = ((-(LGM[0]) * (1 - theta◖F◗)) / N◖F◗[0]) * (LAMBDA◖F◗[0] * K◖F◗[-4] ^ theta◖F◗ * N◖F◗[0] ^ (1 - theta◖F◗)) ^ -nu◖F◗ * Y◖F◗[0] ^ (1 + nu◖F◗)
    #= REPL[393]:19 =#
    (beta◖F◗ ^ 0 * LGM[0] * phi◖F◗ + beta◖F◗ ^ 1 * LGM[1] * phi◖F◗ + beta◖F◗ ^ 2 * LGM[2] * phi◖F◗ + beta◖F◗ ^ 3 * LGM[3] * phi◖F◗) + (-(beta◖F◗ ^ 1) * LGM[1] * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 2) * LGM[2] * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 3) * LGM[3] * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 4) * LGM[4] * phi◖F◗ * (1 - delta◖F◗)) = ((beta◖F◗ ^ 4 * LGM[4] * theta◖F◗) / K◖F◗[0]) * (LAMBDA◖F◗[4] * K◖F◗[0] ^ theta◖F◗ * N◖F◗[4] ^ (1 - theta◖F◗)) ^ -nu◖F◗ * Y◖F◗[4] ^ (1 + nu◖F◗)
    #= REPL[393]:26 =#
    LGM[0] = beta◖F◗ * LGM[1] * (1 + sigma◖F◗ * Z◖F◗[0] ^ (-nu◖F◗ - 1) * Y◖F◗[1] ^ (1 + nu◖F◗))
    #= REPL[393]:28 =#
    NX◖F◗[0] = (Y◖F◗[0] - ((C◖F◗[0] + X◖F◗[0] + Z◖F◗[0]) - Z◖F◗[-1])) / Y◖F◗[0]
    #= REPL[393]:29 =#
end
LAMBDA◖H◗[0] - 1 = rho◖H◗◖H◗ * (LAMBDA◖H◗[-1] - 1) + rho◖H◗◖F◗ * (LAMBDA◖F◗[-1] - 1) + Z_E◖H◗ * E◖H◗[x]
LAMBDA◖F◗[0] - 1 = rho◖F◗◖F◗ * (LAMBDA◖F◗[-1] - 1) + rho◖F◗◖H◗ * (LAMBDA◖H◗[-1] - 1) + Z_E◖F◗ * E◖F◗[x]
((C◖H◗[0] + X◖H◗[0] + Z◖H◗[0]) - Z◖H◗[-1]) + ((C◖F◗[0] + X◖F◗[0] + Z◖F◗[0]) - Z◖F◗[-1]) = Y◖H◗[0] + Y◖F◗[0]
end)

aaa.args[4] |> unblock

:(C_t{co}[0] * Psi_con{1}{co} * var"omega_N{1}◖a◗") |> dump

:(C_t{co}[0] * Psi_con{1}{co} * var"omega_N{1}◖a◗") |> replace_indices

model_ex|>dump
write_out_for_loops(mms_call.args[3])[1].args[6]
parsed_eqs = write_out_for_loops(mms_call.args[3])
parsed_eqs[1].args[6]

arg = parsed_eqs

eqs = Expr[]  # Initialize an empty array to collect expressions




postwalk(x -> begin
x = unblock(x)
x isa Expr ?
    x.head == :for ?
        x.args[2] isa Array ?
            length(x.args[2]) >= 1 ?
                x.args[1].head == :block ?
                    [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] :
                [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] :
            x :
        x.args[2].head ∉ [:(=), :block] ?
            x.args[1].head == :block ?
                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                    Symbol(x.args[1].args[2].args[1]), 
                                    (x.args[1].args[2].args[2]),
                                    true,
                                    x.args[1].args[1].args[2].value) : # for loop part of equation
            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                Symbol(x.args[1].args[1]), 
                                (x.args[1].args[2]),
                                true,
                                :+) : # for loop part of equation
        x.args[1].head == :block ?
            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                Symbol(x.args[1].args[2].args[1]), 
                                (x.args[1].args[2].args[2]),
                                false,
                                x.args[1].args[1].args[2].value) : # for loop part of equation
        replace_indices_inside_for_loop(unblock(x.args[2]), 
                            Symbol(x.args[1].args[1]), 
                            (x.args[1].args[2]),
                            false,
                            :+)[1] :
    x :
x
end,
mms_call)

mms_call.args[3].args[1].head
unblock(mms_call.args[3].args[2])
mms_call.args[3].args[1].args[1]
mms_call.args[3].args[1].args[2]

psd = replace_indices_inside_for_loop(unblock(mms_call.args[3].args[2]), 
                            Symbol(mms_call.args[3].args[1].args[1]), 
                            (mms_call.args[3].args[1].args[2]),
                            false,
                            :+)

postwalk(x -> begin
x = unblock(x)
x isa Expr ?
    x.head == :for ?
        x.args[2] isa Array ?
            length(x.args[2]) >= 1 ?
                x.args[1].head == :block ?
                    [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] :
                [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] :
            x :
        x.args[2].head ∉ [:(=), :block] ?
            x.args[1].head == :block ?
                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                    Symbol(x.args[1].args[2].args[1]), 
                                    (x.args[1].args[2].args[2]),
                                    true,
                                    x.args[1].args[1].args[2].value) : # for loop part of equation
            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                Symbol(x.args[1].args[1]), 
                                (x.args[1].args[2]),
                                true,
                                :+) : # for loop part of equation
        x.args[1].head == :block ?
            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                Symbol(x.args[1].args[2].args[1]), 
                                (x.args[1].args[2].args[2]),
                                false,
                                x.args[1].args[1].args[2].value) : # for loop part of equation
        replace_indices_inside_for_loop(unblock(x.args[2]), 
                            Symbol(x.args[1].args[1]), 
                            (x.args[1].args[2]),
                            false,
                            :+)[1] :
    x :
x
end,
psd[1].args[6])

psd[1].args[6].args[2]


replace_indices_inside_for_loop(unblock(psd[1].args[6].args[2].args[2]), 
                            Symbol(psd[1].args[6].args[1].args[1]), 
                            (psd[1].args[6].args[1].args[2]),
                            false,
                            :+)

exxpr = unblock(psd[1].args[6].args[2].args[2])
index_variable = Symbol(psd[1].args[6].args[1].args[1])
indices = (psd[1].args[6].args[1].args[2])
concatenate = false
operator = :+


@assert operator ∈ [:+,:*] "Only :+ and :* allowed as operators in for loops."
calls = []
indices = indices.args[1] == :(:) ? eval(indices) : [indices.args...]
# for idx in indices
idx = 1
    push!(calls, postwalk(x -> begin
        x isa Expr ?
            x.head == :ref ?
                @capture(x, name_{index_}name2_[time_]) ?
                    index == index_variable ?
                        :($(Expr(:ref, Symbol(string(name) * "◖" * string(idx) * "◗" * string(name2)),time))) :
                    time isa Expr || time isa Symbol ?
                        index_variable ∈ get_symbols(time) ?
                            :($(Expr(:ref, Expr(:curly,name,index), Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                        x :
                    x :
                @capture(x, name_[time_]) ?
                    time isa Expr || time isa Symbol ?
                        index_variable ∈ get_symbols(time) ?
                            :($(Expr(:ref, name, Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                        x :
                    x :
                x :
            @capture(x, name_{index_}name2_) ?
                index == index_variable ?
                    :($(Symbol(string(name) * "◖" * string(idx) * "◗" * string(name2)))) :
                x :
            x :
        @capture(x, name_) ?
            name == index_variable && idx isa Int ?
                :($idx) :
            x :
        x
    end,
    exxpr))
# end

exxpr |> string 
Meta.parse(replace(string(exxpr), "{" * string(index_variable) * "}" => "◖" * string(idx) * "◗"))

Symbol(replace(x, "{" => "◖", "}" => "◗"))

if arg isa Expr
    if arg.head == :block
        for b in arg.args
            if b isa Expr
                # If the result is an Expr, process and add to eqs
                push!(eqs, unblock(replace_indices(b)))
            # elseif b isa Array
            #     recurse(b)
            end
        end
    end
elseif arg isa Array
    # If the result is an Array, iterate and recurse
    for B in arg
        if B isa Expr
            if B.head == :block
                for b in B.args
                    if b isa Expr
                        # If the result is an Expr, process and add to eqs
                        push!(eqs, unblock(replace_indices(b)))
                    elseif b isa Array
                        for bb in b
                            if bb.head == :block
                                for bbb in bb.args
                                    if bbb isa Expr
                                        # If the result is an Expr, process and add to eqs
                                        println(replace_indices(bbb))
                                        push!(eqs, unblock(replace_indices(bbb)))
                                    elseif bbb isa Array
                                        for bbbb in bbb
                                            if bbbb.head ∈ [:block]
                                                for bbbbb in bbbb.args
                                                    if bbbbb isa Expr
                                                        # If the result is an Expr, process and add to eqs
                                                        push!(eqs, unblock(replace_indices(bbbbb)))
                                                    # elseif b isa Array
                                                    #     recurse(b)
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


eqs

arg[1].head
arg[1].args[6][1].args[6][1].head

eqs = []
# Define a helper recursive function
function recurse(arg)
        if arg isa Expr
            # if arg.head == :block
                for b in arg.args
                    if b isa Expr
                        # If the result is an Expr, process and add to eqs
                        push!(eqs, unblock(replace_indices(b)))
                    elseif b isa Array
                        recurse(b)
                    end
                end
            # end
        elseif arg isa Array
            # If the result is an Array, iterate and recurse
            for B in arg
                # println((B))
                recurse(B)
            end
        end
end

recurse(parsed_eqs)

function parse_for_loops(equations_block)
    eqs = Expr[]  # Initialize an empty array to collect expressions

    # Define a helper recursive function
    function recurse(arg)
            if arg isa Expr
                if arg.head == :block
                    for b in arg.args
                        if b isa Expr
                            # If the result is an Expr, process and add to eqs
                            push!(eqs, unblock(replace_indices(b)))
                        elseif b isa Array
                            recurse(b)
                        end
                    end
                end
            elseif arg isa Array
                # If the result is an Array, iterate and recurse
                for B in arg
                    println((B))
                    recurse(B)
                end
            end
    end

    for arg in equations_block.args
        if isa(arg,Expr)
            parsed_eqs = write_out_for_loops(arg)
            recurse(parsed_eqs)
        end
    end

    # Return the collected expressions as a block
    return Expr(:block, eqs...)
end




arg = mms_call

out = postwalk(x -> begin
x = unblock(x)
x isa Expr ?
    x.head == :for ?
        x.args[2] isa Array ?
            length(x.args[2]) >= 1 ?
                x.args[1].head == :block ?
                    begin # println(x.args[2])
                    [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] 
                end :
                begin # println(x.args[2])
                [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] end :
            x :
        x.args[2].head ∉ [:(=), :block] ?
            x.args[1].head == :block ?
                begin #println(x.args[2])
                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                    Symbol(x.args[1].args[2].args[1]), 
                                    (x.args[1].args[2].args[2]),
                                    true,
                                    x.args[1].args[1].args[2].value) end : # for loop part of equation
            
                            begin
                                # println(x.args[2])
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                Symbol(x.args[1].args[1]), 
                                (x.args[1].args[2]),
                                true,
                                :+) end : # for loop part of equation
        x.args[1].head == :block ?
            begin # println(x.args[2])
            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                Symbol(x.args[1].args[2].args[1]), 
                                (x.args[1].args[2].args[2]),
                                false,
                                x.args[1].args[1].args[2].value) end : # for loop part of equation
                            begin println(x.args[2]); println("###")
                            replace_indices_inside_for_loop(unblock(x.args[2]), 
                            Symbol(x.args[1].args[1]), 
                            (x.args[1].args[2]),
                            false,
                            :+) end :
    x :
x
end,
arg);


# out = postwalk(x -> begin
#     # x = unblock(x)
#     x isa Expr ?
#         push!(eqs,unblock(replace_indices(x))) :
#         # x isa Array ?
#         #     push!(eqs,unblock(replace_indices(x))) :
#         x
#     x end,
#     write_out_for_loops(mms_call.args[3])[1])


function flatten_expression(expr)
    # Initialize an empty array to collect top-level expressions
    collected_exprs = []

    # Recursive helper function to traverse and collect expressions
    function recurse(e)
        if e isa Expr
            if e.head == :quote
                # Unwrap the quote block and process its contents
                for sub in e.args
                    recurse(sub)
                end
            elseif e.head == :array
                # Unwrap the array and process each element
                for elem in e.args
                    recurse(elem)
                end
            else
                # For any other Expr, add it directly to collected expressions
                push!(collected_exprs, e)
            end
        elseif e isa AbstractArray
            # If it's any other kind of array, process each element
            for elem in e
                recurse(elem)
            end
        else
            # Literals and other non-Expr, non-Array elements are added directly
            push!(collected_exprs, e)
        end
    end

    # Start the recursion with the input expression
    recurse(expr)

    # Create a single quote block with all collected expressions
    # If there's only one expression, it's wrapped in a quote without a block
    if length(collected_exprs) == 1
        return Expr(:quote, collected_exprs[1])
    else
        return Expr(:quote, Expr(:block, collected_exprs...))
    end
end


parsed_eqs |> flatten_expression
    
eqs = []
function process_parsed_eqs(element)
    if element isa Expr
        new_args = [process_parsed_eqs(arg) for arg in element.args]
        new_expr = Expr(element.head, new_args...)
        push!(eqs, unblock(replace_indices(new_expr)))
    elseif element isa Array
        for item in element
            process_parsed_eqs(item)
        end
    else
        # If it's neither Expr nor Array, handle accordingly or ignore
    end
end

# Usage
process_parsed_eqs(parsed_eqs)
    
function process_parsed_eqs(element)
        if element isa Expr
            push!(eqs, unblock(replace_indices(element)))
        elseif element isa Array
            for item in element
                process_parsed_eqs(item)
            end
        end
    end

    process_parsed_eqs(parsed_eqs)


if parsed_eqs isa Expr
    push!(eqs,unblock(replace_indices(parsed_eqs)))
elseif parsed_eqs isa Array
    for B in parsed_eqs
        if B isa Array
            for b in B
                push!(eqs,unblock(replace_indices(b)))
            end
        elseif B isa Expr
            if B.head == :block
                for b in B.args
                    if b isa Expr
                        push!(eqs,replace_indices(b))
                    end
                end
            else
                push!(eqs,unblock(replace_indices(B)))
            end
        else
            push!(eqs,unblock(replace_indices(B)))
        end
    end
end


mms_call = :(
    a = 1;
    for co in [a] 
        log(R_t{co}[0] / R_ts{co}) = 0.8 * log(R_t{co}[-1] / R_ts{co}) + 0.3 * log(pi_cpi_t{co}[-1]) # Modified Taylor-rule

        w_t{co}[0] = (for se in 1:2 omega_N{se}{co} * w_t{se}{co}[0] ^ (( - upsi_N{co}) / (1 - upsi_N{co})) end) ^ (( - (1 - upsi_N{co})) / upsi_N{co}) # Wage index derived from opt. problem of labor-bundler
    end)


for arg in mms_call.args
    if isa(arg,Expr)
        # parsed_eqs = write_out_for_loops(arg)
        println((arg))
        println("##")
        # println((parsed_eqs))
    end
end

mms_call |> dump

arg = mms_call
mms_call.args[2] isa Array

mms_call.args[3]

out = postwalk(x -> begin
    x = unblock(x)
    x isa Expr ?
        x.head == :for ?
            x.args[2] isa Array ?
                length(x.args[2]) >= 1 ?
                    x.args[1].head == :block ?
                        [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] :
                    [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] :
                x :
            x.args[2].head ∉ [:(=), :block] ?
                x.args[1].head == :block ?
                    replace_indices_inside_for_loop(unblock(x.args[2]), 
                                        Symbol(x.args[1].args[2].args[1]), 
                                        (x.args[1].args[2].args[2]),
                                        true,
                                        x.args[1].args[1].args[2].value) : # for loop part of equation
                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                    Symbol(x.args[1].args[1]), 
                                    (x.args[1].args[2]),
                                    true,
                                    :+) : # for loop part of equation
            x.args[1].head == :block ?
                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                    Symbol(x.args[1].args[2].args[1]), 
                                    (x.args[1].args[2].args[2]),
                                    false,
                                    x.args[1].args[1].args[2].value) : # for loop part of equation
            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                Symbol(x.args[1].args[1]), 
                                (x.args[1].args[2]),
                                false,
                                :+) :
        x :
    x
    end,
arg)

arg = mms_call.args[3]
exxpr = unblock(arg.args[2])
index_variable = Symbol(arg.args[1].args[1])
indices = (arg.args[1].args[2])
concatenate = true
operator = :+

# function replace_indices_inside_for_loop(exxpr,index_variable,indices,concatenate, operator)
    @assert operator ∈ [:+,:*] "Only :+ and :* allowed as operators in for loops."
    calls = []
    indices = indices.args[1] == :(:) ? eval(indices) : [indices.args...]
    for idx in indices
        push!(calls, postwalk(x -> begin
            x isa Expr ?
                x.head == :ref ?
                    @capture(x, name_{index_}[time_]) ?
                        index == index_variable ?
                            :($(Expr(:ref, Symbol(string(name) * "◖" * string(idx) * "◗"),time))) :
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ get_symbols(time) ?
                                :($(Expr(:ref, Expr(:curly,name,index), Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            x :
                        x :
                    @capture(x, name_[time_]) ?
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ get_symbols(time) ?
                                :($(Expr(:ref, name, Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            x :
                        x :
                    x :
                @capture(x, name_{index_}) ?
                    index == index_variable ?
                        :($(Symbol(string(name) * "◖" * string(idx) * "◗"))) :
                    x :
                x :
            @capture(x, name_) ?
                name == index_variable && idx isa Int ?
                    :($idx) :
                x :
            x
        end,
        exxpr))
    end

    if concatenate
        return :($(Expr(:call, operator, calls...)))
    else
        return calls
    end
# end


