include("mini_parser.jl")
using .MiniParser
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
    parsed = MacroTools.flatten(MacroTools.striplines(MiniParser.parse_for_loops(expr)))

    @test parsed.head == :block
    @test length(parsed.args) == 4
    @test all(x->x isa Expr, parsed.args)
end

@testset "nested for loops" begin
    expr = quote
        for lag2 in 0:(4-2)
            for lag in 0:(4-1)
                if lag >= lag2
                    OUT{lag,lag2}[0] = lag
                else
                    OUT{lag,lag2}[0] = lag2
                end
            end
        end
    end
    using MacroTools
    parsed = MacroTools.flatten(MacroTools.striplines(MiniParser.parse_for_loops(expr)))
    @test parsed.head == :block
    @test length(parsed.args) == 12
end
