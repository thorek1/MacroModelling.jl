using Revise
import MacroTools


ex = :(for co in [H, F]
        # NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1]))/Y{co}[0]
        if co == H
            NX{F}{H}[0] = log(NX{F}[0] - NX{H}[0])
        # else
        #     NX{H}{F}[0] = log(NX{H}[0] - NX{F}[0]) 
        end
end)

ex |> dump


using Revise
using MacroModelling
include("Backus_Kehoe_Kydland_1992.jl")

Backus_Kehoe_Kydland_1992.ss_equations

:(nothing+0)|>dump


ex = :(begin
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:3 =#
    Y◖H◗[0] = ((LAMBDA◖H◗[0] * K◖H◗[-4] ^ theta◖H◗ * N◖H◗[0] ^ (1 - theta◖H◗)) ^ -nu◖H◗ + sigma◖H◗ * Z◖H◗[-1] ^ -nu◖H◗) ^ (-1 / nu◖H◗)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:5 =#
    K◖H◗[0] = (1 - delta◖H◗) * K◖H◗[-1] + S◖H◗[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:7 =#
    X◖H◗[0] = phi◖H◗ * S◖H◗[-3] + phi◖H◗ * S◖H◗[-2] + phi◖H◗ * S◖H◗[-1] + phi◖H◗ * S◖H◗[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:9 =#
    A◖H◗[0] = (1 - eta◖H◗) * A◖H◗[-1] + N◖H◗[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:11 =#
    L◖H◗[0] = (1 - alpha◖H◗ * N◖H◗[0]) - (1 - alpha◖H◗) * eta◖H◗ * A◖H◗[-1]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:13 =#
    U◖H◗[0] = (C◖H◗[0] ^ mu◖H◗ * L◖H◗[0] ^ (1 - mu◖H◗)) ^ gamma◖H◗
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:15 =#
    ((psi◖H◗ * mu◖H◗) / C◖H◗[0]) * U◖H◗[0] = LGM[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:17 =#
    ((psi◖H◗ * (1 - mu◖H◗)) / L◖H◗[0]) * U◖H◗[0] * -alpha◖H◗ = ((-(LGM[0]) * (1 - theta◖H◗)) / N◖H◗[0]) * (LAMBDA◖H◗[0] * K◖H◗[-4] ^ theta◖H◗ * N◖H◗[0] ^ (1 - theta◖H◗)) ^ -nu◖H◗ * Y◖H◗[0] ^ (1 + nu◖H◗)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:19 =#
    ((if 0 >= 0
                        beta◖H◗ ^ (0 + 1) * LGM[0] * phi◖H◗
                    elseif 0 > 5
                        beta◖H◗ ^ 0 * LGM[0] * phi◖H◗
                    else
                        beta◖H◗ ^ (0 + 2) * LGM[0] * phi◖H◗
                    end + if 1 >= 0
                        beta◖H◗ ^ (1 + 1) * LGM[1] * phi◖H◗
                    elseif 1 > 5
                        beta◖H◗ ^ 1 * LGM[1] * phi◖H◗
                    else
                        beta◖H◗ ^ (1 + 2) * LGM[1] * phi◖H◗
                    end + if 2 >= 0
                        beta◖H◗ ^ (2 + 1) * LGM[2] * phi◖H◗
                    elseif 2 > 5
                        beta◖H◗ ^ 2 * LGM[2] * phi◖H◗
                    else
                        beta◖H◗ ^ (2 + 2) * LGM[2] * phi◖H◗
                    end + if 3 >= 0
                        beta◖H◗ ^ (3 + 1) * LGM[3] * phi◖H◗
                    elseif 3 > 5
                        beta◖H◗ ^ 3 * LGM[3] * phi◖H◗
                    else
                        beta◖H◗ ^ (3 + 2) * LGM[3] * phi◖H◗
                    end) + (if 0 >= 1
                        beta◖H◗ ^ (0 + 1) * LGM[0] * phi◖H◗
                    elseif 0 > 5
                        beta◖H◗ ^ 0 * LGM[0] * phi◖H◗
                    else
                        beta◖H◗ ^ (0 + 2) * LGM[0] * phi◖H◗
                    end + if 1 >= 1
                        beta◖H◗ ^ (1 + 1) * LGM[1] * phi◖H◗
                    elseif 1 > 5
                        beta◖H◗ ^ 1 * LGM[1] * phi◖H◗
                    else
                        beta◖H◗ ^ (1 + 2) * LGM[1] * phi◖H◗
                    end + if 2 >= 1
                        beta◖H◗ ^ (2 + 1) * LGM[2] * phi◖H◗
                    elseif 2 > 5
                        beta◖H◗ ^ 2 * LGM[2] * phi◖H◗
                    else
                        beta◖H◗ ^ (2 + 2) * LGM[2] * phi◖H◗
                    end + if 3 >= 1
                        beta◖H◗ ^ (3 + 1) * LGM[3] * phi◖H◗
                    elseif 3 > 5
                        beta◖H◗ ^ 3 * LGM[3] * phi◖H◗
                    else
                        beta◖H◗ ^ (3 + 2) * LGM[3] * phi◖H◗
                    end) + (if 0 >= 2
                        beta◖H◗ ^ (0 + 1) * LGM[0] * phi◖H◗
                    elseif 0 > 5
                        beta◖H◗ ^ 0 * LGM[0] * phi◖H◗
                    else
                        beta◖H◗ ^ (0 + 2) * LGM[0] * phi◖H◗
                    end + if 1 >= 2
                        beta◖H◗ ^ (1 + 1) * LGM[1] * phi◖H◗
                    elseif 1 > 5
                        beta◖H◗ ^ 1 * LGM[1] * phi◖H◗
                    else
                        beta◖H◗ ^ (1 + 2) * LGM[1] * phi◖H◗
                    end + if 2 >= 2
                        beta◖H◗ ^ (2 + 1) * LGM[2] * phi◖H◗
                    elseif 2 > 5
                        beta◖H◗ ^ 2 * LGM[2] * phi◖H◗
                    else
                        beta◖H◗ ^ (2 + 2) * LGM[2] * phi◖H◗
                    end + if 3 >= 2
                        beta◖H◗ ^ (3 + 1) * LGM[3] * phi◖H◗
                    elseif 3 > 5
                        beta◖H◗ ^ 3 * LGM[3] * phi◖H◗
                    else
                        beta◖H◗ ^ (3 + 2) * LGM[3] * phi◖H◗
                    end)) + (-(beta◖H◗ ^ 1) * LGM[1] * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 2) * LGM[2] * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 3) * LGM[3] * phi◖H◗ * (1 - delta◖H◗) + -(beta◖H◗ ^ 4) * LGM[4] * phi◖H◗ * (1 - delta◖H◗)) = ((beta◖H◗ ^ 4 * LGM[4] * theta◖H◗) / K◖H◗[0]) * (LAMBDA◖H◗[4] * K◖H◗[0] ^ theta◖H◗ * N◖H◗[4] ^ (1 - theta◖H◗)) ^ -nu◖H◗ * Y◖H◗[4] ^ (1 + nu◖H◗)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:34 =#
    LGM[0] = beta◖H◗ * LGM[1] * (1 + sigma◖H◗ * Z◖H◗[0] ^ (-nu◖H◗ - 1) * Y◖H◗[1] ^ (1 + nu◖H◗))
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:36 =#
    NX◖H◗[0] = (Y◖H◗[0] - ((C◖H◗[0] + X◖H◗[0] + Z◖H◗[0]) - Z◖H◗[-1])) / Y◖H◗[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:42 =#
    if co == H
        NX◖F◗◖H◗[0] = log(NX◖F◗[0] - NX◖H◗[0])
    else
        NX◖H◗◖F◗[0] = log(NX◖H◗[0] - NX◖F◗[0])
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:43 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:3 =#
    Y◖F◗[0] = ((LAMBDA◖F◗[0] * K◖F◗[-4] ^ theta◖F◗ * N◖F◗[0] ^ (1 - theta◖F◗)) ^ -nu◖F◗ + sigma◖F◗ * Z◖F◗[-1] ^ -nu◖F◗) ^ (-1 / nu◖F◗)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:5 =#
    K◖F◗[0] = (1 - delta◖F◗) * K◖F◗[-1] + S◖F◗[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:7 =#
    X◖F◗[0] = phi◖F◗ * S◖F◗[-3] + phi◖F◗ * S◖F◗[-2] + phi◖F◗ * S◖F◗[-1] + phi◖F◗ * S◖F◗[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:9 =#
    A◖F◗[0] = (1 - eta◖F◗) * A◖F◗[-1] + N◖F◗[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:11 =#
    L◖F◗[0] = (1 - alpha◖F◗ * N◖F◗[0]) - (1 - alpha◖F◗) * eta◖F◗ * A◖F◗[-1]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:13 =#
    U◖F◗[0] = (C◖F◗[0] ^ mu◖F◗ * L◖F◗[0] ^ (1 - mu◖F◗)) ^ gamma◖F◗
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:15 =#
    ((psi◖F◗ * mu◖F◗) / C◖F◗[0]) * U◖F◗[0] = LGM[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:17 =#
    ((psi◖F◗ * (1 - mu◖F◗)) / L◖F◗[0]) * U◖F◗[0] * -alpha◖F◗ = ((-(LGM[0]) * (1 - theta◖F◗)) / N◖F◗[0]) * (LAMBDA◖F◗[0] * K◖F◗[-4] ^ theta◖F◗ * N◖F◗[0] ^ (1 - theta◖F◗)) ^ -nu◖F◗ * Y◖F◗[0] ^ (1 + nu◖F◗)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:19 =#
    ((if 0 >= 0
                        beta◖F◗ ^ (0 + 1) * LGM[0] * phi◖F◗
                    elseif 0 > 5
                        beta◖F◗ ^ 0 * LGM[0] * phi◖F◗
                    else
                        beta◖F◗ ^ (0 + 2) * LGM[0] * phi◖F◗
                    end + if 1 >= 0
                        beta◖F◗ ^ (1 + 1) * LGM[1] * phi◖F◗
                    elseif 1 > 5
                        beta◖F◗ ^ 1 * LGM[1] * phi◖F◗
                    else
                        beta◖F◗ ^ (1 + 2) * LGM[1] * phi◖F◗
                    end + if 2 >= 0
                        beta◖F◗ ^ (2 + 1) * LGM[2] * phi◖F◗
                    elseif 2 > 5
                        beta◖F◗ ^ 2 * LGM[2] * phi◖F◗
                    else
                        beta◖F◗ ^ (2 + 2) * LGM[2] * phi◖F◗
                    end + if 3 >= 0
                        beta◖F◗ ^ (3 + 1) * LGM[3] * phi◖F◗
                    elseif 3 > 5
                        beta◖F◗ ^ 3 * LGM[3] * phi◖F◗
                    else
                        beta◖F◗ ^ (3 + 2) * LGM[3] * phi◖F◗
                    end) + (if 0 >= 1
                        beta◖F◗ ^ (0 + 1) * LGM[0] * phi◖F◗
                    elseif 0 > 5
                        beta◖F◗ ^ 0 * LGM[0] * phi◖F◗
                    else
                        beta◖F◗ ^ (0 + 2) * LGM[0] * phi◖F◗
                    end + if 1 >= 1
                        beta◖F◗ ^ (1 + 1) * LGM[1] * phi◖F◗
                    elseif 1 > 5
                        beta◖F◗ ^ 1 * LGM[1] * phi◖F◗
                    else
                        beta◖F◗ ^ (1 + 2) * LGM[1] * phi◖F◗
                    end + if 2 >= 1
                        beta◖F◗ ^ (2 + 1) * LGM[2] * phi◖F◗
                    elseif 2 > 5
                        beta◖F◗ ^ 2 * LGM[2] * phi◖F◗
                    else
                        beta◖F◗ ^ (2 + 2) * LGM[2] * phi◖F◗
                    end + if 3 >= 1
                        beta◖F◗ ^ (3 + 1) * LGM[3] * phi◖F◗
                    elseif 3 > 5
                        beta◖F◗ ^ 3 * LGM[3] * phi◖F◗
                    else
                        beta◖F◗ ^ (3 + 2) * LGM[3] * phi◖F◗
                    end) + (if 0 >= 2
                        beta◖F◗ ^ (0 + 1) * LGM[0] * phi◖F◗
                    elseif 0 > 5
                        beta◖F◗ ^ 0 * LGM[0] * phi◖F◗
                    else
                        beta◖F◗ ^ (0 + 2) * LGM[0] * phi◖F◗
                    end + if 1 >= 2
                        beta◖F◗ ^ (1 + 1) * LGM[1] * phi◖F◗
                    elseif 1 > 5
                        beta◖F◗ ^ 1 * LGM[1] * phi◖F◗
                    else
                        beta◖F◗ ^ (1 + 2) * LGM[1] * phi◖F◗
                    end + if 2 >= 2
                        beta◖F◗ ^ (2 + 1) * LGM[2] * phi◖F◗
                    elseif 2 > 5
                        beta◖F◗ ^ 2 * LGM[2] * phi◖F◗
                    else
                        beta◖F◗ ^ (2 + 2) * LGM[2] * phi◖F◗
                    end + if 3 >= 2
                        beta◖F◗ ^ (3 + 1) * LGM[3] * phi◖F◗
                    elseif 3 > 5
                        beta◖F◗ ^ 3 * LGM[3] * phi◖F◗
                    else
                        beta◖F◗ ^ (3 + 2) * LGM[3] * phi◖F◗
                    end)) + (-(beta◖F◗ ^ 1) * LGM[1] * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 2) * LGM[2] * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 3) * LGM[3] * phi◖F◗ * (1 - delta◖F◗) + -(beta◖F◗ ^ 4) * LGM[4] * phi◖F◗ * (1 - delta◖F◗)) = ((beta◖F◗ ^ 4 * LGM[4] * theta◖F◗) / K◖F◗[0]) * (LAMBDA◖F◗[4] * K◖F◗[0] ^ theta◖F◗ * N◖F◗[4] ^ (1 - theta◖F◗)) ^ -nu◖F◗ * Y◖F◗[4] ^ (1 + nu◖F◗)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:34 =#
    LGM[0] = beta◖F◗ * LGM[1] * (1 + sigma◖F◗ * Z◖F◗[0] ^ (-nu◖F◗ - 1) * Y◖F◗[1] ^ (1 + nu◖F◗))
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:36 =#
    NX◖F◗[0] = (Y◖F◗[0] - ((C◖F◗[0] + X◖F◗[0] + Z◖F◗[0]) - Z◖F◗[-1])) / Y◖F◗[0]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:42 =#
    if co == H
        NX◖F◗◖H◗[0] = log(NX◖F◗[0] - NX◖H◗[0])
    else
        NX◖H◗◖F◗[0] = log(NX◖H◗[0] - NX◖F◗[0])
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/Backus_Kehoe_Kydland_1992.jl:43 =#
    LAMBDA◖H◗[0] - 1 = rho◖H◗◖H◗ * (LAMBDA◖H◗[-1] - 1) + rho◖H◗◖F◗ * (LAMBDA◖F◗[-1] - 1) + Z_E◖H◗ * E◖H◗[x]
    LAMBDA◖F◗[0] - 1 = rho◖F◗◖F◗ * (LAMBDA◖F◗[-1] - 1) + rho◖F◗◖H◗ * (LAMBDA◖H◗[-1] - 1) + Z_E◖F◗ * E◖F◗[x]
    ((C◖H◗[0] + X◖H◗[0] + Z◖H◗[0]) - Z◖H◗[-1]) + ((C◖F◗[0] + X◖F◗[0] + Z◖F◗[0]) - Z◖F◗[-1]) = Y◖H◗[0] + Y◖F◗[0]
end)

import MacroTools

function _condition_value(cond)
    if cond isa Bool
        return cond
    elseif cond isa Expr && cond.head == :call 
        a, b = cond.args[2], cond.args[3]

        if !((a isa Symbol && b isa Symbol) || (a isa Number && b isa Number))
            a = eval(a)
            b = eval(b)
        end

        if cond.args[1] == :(==)
            return a == b
        elseif cond.args[1] == :(!=)
            return a != b
        elseif cond.args[1] == :(<)
            return a < b
        elseif cond.args[1] == :(<=)
            return a <= b
        elseif cond.args[1] == :(>)
            return a > b
        elseif cond.args[1] == :(>=)
            return a >= b
        end
        # end
    end
    return nothing
end

function keep_true_branch(ex)
    MacroTools.prewalk(ex) do x
        if MacroTools.@capture(x, if cond_ body_ else body2_ end)
            cnd = _condition_value(cond)
            if cnd !== nothing
                cnd ? body : body2
            else
                x
            end
        elseif MacroTools.@capture(x, if cond_ body_ elseif cond2_ body2_ else body3_ end)
            cnd = _condition_value(cond)
            if cnd !== nothing
                cnd ? body : keep_true_branch(:(if $cond2 $body2 else $body3 end))
            else
                x
            end
        elseif MacroTools.@capture(x, if cond_ body_ end)
            begin 
                cnd = _condition_value(cond)
                cnd ? body : nothing
            end
        else
            x
        end
    end
end

keep_true_branch(ex) # |> dump


exx = :(if 2 >= 3
                               #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:156 =#
                               beta◖F◗ ^ (2 + 1) * LGM[2] * phi◖F◗
                           elseif #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:157 =# 6 > 6
                               #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:158 =#
                               beta◖F◗ ^ 2 * LGM[2] * phi◖F◗
                           elseif #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:157 =# 2 > 4
                               #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:158 =#
                               beta◖F◗ ^ 5 * LGM[2] * phi◖F◗
                           else
                               #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:160 =#
                               beta◖F◗ ^ (2 + 2) * LGM[2] * phi◖F◗
                           end)

                           exx |> dump
Expr(:block)
function extract_true_branch(ex::Expr)
    MacroTools.prewalk(ex) do node
        if node isa Expr && (node.head === :if || node.head === :elseif)
            cond, then_blk, else_blk = node.args
            cnd = _condition_value(MacroTools.unblock(cond))
            if cnd === true
                return then_blk |> MacroTools.unblock
            elseif cnd === false
                return else_blk |> MacroTools.unblock
            end
        end
        return node
    end
end


function evaluate_condition(cond)
    if cond isa Bool
        return cond
    elseif cond isa Expr && cond.head == :call 
        a, b = cond.args[2], cond.args[3]

        if typeof(a) ∉ [:Symbol, :Number]
            a = eval(a)
        end

        if typeof(b) ∉ [:Symbol, :Number]
            b = eval(b)
        end
        
        if cond.args[1] == :(==)
            return a == b
        elseif cond.args[1] == :(!=)
            return a != b
        elseif cond.args[1] == :(<)
            return a < b
        elseif cond.args[1] == :(<=)
            return a <= b
        elseif cond.args[1] == :(>)
            return a > b
        elseif cond.args[1] == :(>=)
            return a >= b
        end
        # end
    end
    return nothing
end

extract_true_branch(exx)


using MacroTools

"""
    extract_true_branch(ex::Expr) -> Expr

Top-down walk through `ex`, and whenever you hit an `if` whose
condition is known at compile time (`true` or `false`), replace
the entire `if` with the matching branch—and keep on walking
inside that branch to peel out any further nested `if`s.
"""
function extract_true_branch(ex::Expr)
    MacroTools.prewalk(ex) do node
        if node isa Expr && (node.head === :if || node.head === :elseif)
            cond, then_blk, else_blk = node.args
            val = _condition_value(MacroTools.unblock(cond))

            if val === true
                # recurse into the selected branch
                return extract_true_branch(MacroTools.unblock(then_blk))
            elseif val === false
                return extract_true_branch(MacroTools.unblock(else_blk))
            end
        end
        return node
    end
end

# Simple evaluator for literal comparisons
# function _condition_value(cond)
#     if cond === true
#         return true
#     elseif cond === false
#         return false
#     elseif cond isa Expr && cond.head === :call
#         op, a, b = cond.args
#         a = a isa Symbol ? eval(a) : a
#         b = b isa Symbol ? eval(b) : b

#         return op === :(==) ? a == b  :
#                op === :(!=) ? a != b :
#                op === :(<)  ? a <  b :
#                op === :(<=) ? a <= b :
#                op === :(>)  ? a >  b :
#                op === :(>=) ? a >= b :
#                nothing
#     else
#         return nothing
#     end
# end

extract_true_branch(exx)



exx = :(if 2 >= 3
                               #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:156 =#
                               beta◖F◗ ^ (2 + 1) * LGM[2] * phi◖F◗
                           elseif #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:157 =# 6^2 -6 > 6-2
                               #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:158 =#
                               beta◖F◗ ^ 2 * LGM[2] * phi◖F◗
                           elseif #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:157 =# 2 > 4
                               #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:158 =#
                               beta◖F◗ ^ 5 * LGM[2] * phi◖F◗
                           else
                               #= /Users/thorekockerols/GitHub/MacroModelling.jl/benchmark/test_if_else.jl:160 =#
                               beta◖F◗ ^ (2 + 2) * LGM[2] * phi◖F◗
                           end)

extract_true_branch(exx)
