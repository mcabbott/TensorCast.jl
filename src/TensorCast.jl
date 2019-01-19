
module TensorCast

using MacroTools

V = false # verbose debugging
const OLD = false # a few things in parse

include("parse.jl")

include("macro.jl")

include("icheck.jl")

include("pretty.jl")

if VERSION < v"1.1.0"
    include("eachslice.jl") 
end

include("slice.jl")

include("recursive.jl")

using Requires

function __init__()

    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        include("static.jl")
    end

    @require Strided = "5e0ebb24-38b0-5f93-81fe-25c709ecae67" begin
        include("strided.jl")
    end

    @require JuliennedArrays = "5cadff95-7770-533d-a838-a1bf817ee6e0" begin
        include("julienne.jl")
    end

    # @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
    # end

end

end # module
