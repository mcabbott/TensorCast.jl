
module TensorCast

# This speeds up loading a bit... but might slow down functions which act on data:
# https://github.com/JuliaPlots/Plots.jl/pull/2544/files
if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    @eval Base.Experimental.@optlevel 1
end

export @cast, @reduce, @matmul, @pretty

using MacroTools, StaticArrays, Compat
using LinearAlgebra, Random

include("capture.jl")
include("macro.jl")
include("pretty.jl")
include("string.jl")

include("slice.jl")     # slice, glue, etc
include("view.jl")      # orient, Reverse{d} etc
include("lazy.jl")      # LazyCast
include("static.jl")    # StaticArrays

include("warm.jl")

end # module
