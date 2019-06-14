
module TensorCast

export @cast, @reduce, @matmul, @pretty

using MacroTools, Requires #, Parameters
using LinearAlgebra, Random

include("macro.jl")
include("pretty.jl")
include("string.jl")

include("slice.jl")     # slice, glue, etc
include("view.jl")      # orient, Reverse{d} etc
include("recursive.jl") # RecursiveArrayTools
include("lazy.jl")      # LazyCast

@init @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" include("static.jl")
@init @require JuliennedArrays = "5cadff95-7770-533d-a838-a1bf817ee6e0" include("julienne.jl")

VERSION < v"1.1.0" && include("eachslice.jl")

end # module
