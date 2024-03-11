using GridapPertubedSaddlePointPreconditioners
using Documenter

DocMeta.setdocmeta!(GridapPertubedSaddlePointPreconditioners, :DocTestSetup, :(using GridapPertubedSaddlePointPreconditioners); recursive=true)

makedocs(;
    modules=[GridapPertubedSaddlePointPreconditioners],
    authors="Rishi Das <drishi0077@gmail.com>, Ricardo Ruiz Baier <ricardo.ruizbaier@monash.edu>",
    repo="https://github.com/gridap/GridapPertubedSaddlePointPreconditioners.jl/blob/{commit}{path}#{line}",
    sitename="GridapPertubedSaddlePointPreconditioners.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gridap.github.io/GridapPertubedSaddlePointPreconditioners.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gridap/GridapPertubedSaddlePointPreconditioners.jl",
    devbranch="main",
)
