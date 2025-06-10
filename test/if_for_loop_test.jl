using MacroModelling
using Test

@testset "if in for loop" begin
    expr = quote
        for co in [H, F]
            NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1]))/Y{co}[0]
            if co == H
                NX{H}d[0] = NX{H}[0] - NX{F}[0]
            else
                NX{F}d[0] = NX{F}[0] - NX{H}[0]
            end
        end
    end

    using MacroTools
    parsed = MacroTools.flatten(MacroTools.striplines(MacroModelling.parse_for_loops(expr)))

    @test parsed.head == :block
    @test length(parsed.args) == 4
    assigns = [sprint(show, x) for x in parsed.args]
    @test assigns[1] == ":(NX◖H◗[0] = (Y◖H◗[0] - ((C◖H◗[0] + X◖H◗[0] + Z◖H◗[0]) - Z◖H◗[-1])) / Y◖H◗[0])"
    @test assigns[2] == ":(NX◖H◗ * d[0] = NX◖H◗[0] - NX◖F◗[0])"
    @test assigns[3] == ":(NX◖F◗[0] = (Y◖F◗[0] - ((C◖F◗[0] + X◖F◗[0] + Z◖F◗[0]) - Z◖F◗[-1])) / Y◖F◗[0])"
    @test assigns[4] == ":(NX◖F◗ * d[0] = NX◖F◗[0] - NX◖H◗[0])"
end
