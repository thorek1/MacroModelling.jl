using Documenter
using MacroModelling 
import Turing, Plots, StatsPlots

makedocs(
    sitename = "MacroModelling.jl",
    authors = "Thore Kockerols",
    doctest = true,
    format = Documenter.HTML(),
    modules = [MacroModelling],
    pages = ["Introduction" => "index.md",
    "Tutorials" => ["Installation" => "tutorials/install.md",
                    "RBC" => "tutorials/rbc.md",
                    "Smets and Wouters (2003)" => "tutorials/sw03.md",
                    "Estimation" => "tutorials/estimation.md"],
    # "How-to guides" => "how_to.md",
    # "Model syntax" => "dsl.md",
    "API" => "api.md",
    "Index" => "call_index.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/thorek1/MacroModelling.jl.git"
)
