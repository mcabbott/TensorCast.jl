
using Documenter
using TensorCast

makedocs(
    sitename = "TensorCast",
    modules = [TensorCast],
    pages = [
        "Home" => "index.md",
        "Basics" => "basics.md",
        "Reduction" => "reduce.md",
        "Options" => "options.md",
        "Docstrings" => "docstrings.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
)

deploydocs(
    repo = "github.com/mcabbott/TensorCast.jl"
)

# julia --color=yes make.jl
