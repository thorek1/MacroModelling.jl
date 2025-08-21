using Documenter
using MacroModelling
import Turing, StatsPlots
using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:authoryear
)

# doctest(MacroModelling, fix=true)

makedocs(
    sitename = "MacroModelling.jl",
    authors = "Thore Kockerols",
    doctest = true,
    # doctest = false,
    # draft = true,
    format = Documenter.HTML(size_threshold = 204800*10),
    modules = [MacroModelling, StatsPlotsExt, TuringExt],
    pages = [
        "Introduction" => "index.md",
        "Tutorials" => [
            "Installation" => "tutorials/install.md",
            "Write your first simple model - RBC" => "tutorials/rbc.md",
            "Work with a more complex model - Smets and Wouters (2003)" => "tutorials/sw03.md",
            "Calibration / method of moments (for higher order perturbation solutions) - Gali (2015)" => "tutorials/calibration.md",
            "Estimate a model using gradient based samplers - Schorfheide (2000)" => "tutorials/estimation.md",
        ],
        "How-to guides" => [
            "Programmatic model writing using for-loops" => "how-to/loops.md",
            "Occasionally binding constraints" => "how-to/obc.md",
            # "how_to.md"
            ],
        # "Model syntax" => "dsl.md",
        "API" => "api.md",
        "Index" => "call_index.md",
    ],
    warnonly = true,
    plugins=[bib]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/thorek1/MacroModelling.jl.git")
