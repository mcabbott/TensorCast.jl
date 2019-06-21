
module TensorCast

export @cast, @reduce, @matmul, @pretty

using MacroTools, StaticArrays
using LinearAlgebra, Random

include("macro.jl")
include("pretty.jl")
include("string.jl")

include("slice.jl")     # slice, glue, etc
include("view.jl")      # orient, Reverse{d} etc
include("recursive.jl") # RecursiveArrayTools
include("lazy.jl")      # LazyCast
include("static.jl")    # StaticArrays

@static VERSION < v"1.1.0" && include("eachslice.jl")

include("warm.jl")

end # module
