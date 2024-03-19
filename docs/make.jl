using Documenter, SciMLFisheries

makedocs(
    sitename="SciMLFisheries.jl",
    modules  = [SciMLFisheries],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["index.md"]
)

deploydocs(
    repo = "github.com/Jack-H-Buckner/SciMLFisheries.jl",
)