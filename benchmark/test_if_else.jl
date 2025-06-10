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