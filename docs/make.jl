using Documenter, BlockRegistration

makedocs(
    modules = [BlockRegistration],
    checkdocs = :exports,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "BlockRegistration.jl",
    authors = "Timothy E. Holy",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "cookbook.md",
        "details.md",
        "improving.md",
    ],
)

deploydocs(
    repo = "github.com/HolyLab/BlockRegistration.jl.git",
    devbranch = "master",
)
