
module TensorCast

using MacroTools

const V = false # verbose debugging

include("parse.jl")

include("macro.jl") # @cast and @reduce

include("matmul.jl") # @mul

include("icheck.jl")

include("pretty.jl")

if VERSION < v"1.1.0"
    include("eachslice.jl")
end

include("slice.jl") # slice, glue, orient, etc

include("recursive.jl") # RecursiveArrayTools

include("order.jl") # Reverse{D} etc

using Requires

function __init__()

    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        include("static.jl")
    end

    # @require Strided = "5e0ebb24-38b0-5f93-81fe-25c709ecae67" begin
    #     include("strided.jl")
    # end

    @require JuliennedArrays = "5cadff95-7770-533d-a838-a1bf817ee6e0" begin
        include("julienne.jl")
    end

    # @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
    # end

    # @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    #   include("flux.jl")
    # end

    # @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    #     include("zygote.jl")
    # end

    @require NamedArrays = "86f7a689-2022-50b4-a561-43c23ac3c673" begin
        include("named.jl")
    end

    # @require AxisArrays = "39de3d68-74b9-583c-8d2d-e117c070f3a9" begin
    #     include("axis.jl")
    # end

end

end # module
