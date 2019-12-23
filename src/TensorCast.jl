
module TensorCast

export @cast, @reduce, @matmul, @pretty

using MacroTools, StaticArrays, Compat
using LinearAlgebra, Random

include("macro.jl")
include("pretty.jl")
include("string.jl")

include("slice.jl")     # slice, glue, etc
include("view.jl")      # orient, Reverse{d} etc
include("recursive.jl") # RecursiveArrayTools
include("lazy.jl")      # LazyCast
include("static.jl")    # StaticArrays

# include("warm.jl")
include("snoop.jl")

end # module
