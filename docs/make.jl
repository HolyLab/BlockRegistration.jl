using Documenter, BlockRegistration

makedocs(
    modules = [BlockRegistration],
    clean = false,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "BlockRegistration.jl",
    authors = "Timothy E. Holy",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "cookbook.md",
        "details.md",
        "improving.md",  # TODO: move more from leftovers.md
        # "whole_experiment.md",
    ],
)

deploydocs(
    repo = "github.com/HolyLab/BlockRegistration.jl.git",
)
