
module TensorCast

export @cast, @reduce, @matmul, @pretty

using MacroTools, Requires
using LinearAlgebra, Random

include("macro.jl")
include("pretty.jl")
include("string.jl")

include("slice.jl")     # slice, glue, etc
include("view.jl")      # orient, Reverse{D} etc
include("recursive.jl") # RecursiveArrayTools
include("lazy.jl")      # LazyCast

function __init__()
    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        include("static.jl")
    end
    @require JuliennedArrays = "5cadff95-7770-533d-a838-a1bf817ee6e0" begin
        include("julienne.jl")
    end
end

if VERSION < v"1.1.0"
    include("eachslice.jl")
end

end # module
