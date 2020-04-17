
module TensorCast

# This speeds up loading a bit, on Julia 1.5, about 1s in my test.
# https://github.com/JuliaPlots/Plots.jl/pull/2544/files
if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    @eval Base.Experimental.@optlevel 1
end

export @cast, @reduce, @matmul, @pretty

using MacroTools, StaticArrays, Compat
using LinearAlgebra, Random

include("macro.jl")
include("pretty.jl")
include("string.jl")

module Fast # shield non-macro code from @optlevel 1
    using LinearAlgebra, StaticArrays

    include("slice.jl")     # slice, glue, etc
    export sliceview, slicecopy, glue, glue!, red_glue, cat_glue, copy_glue, lazy_glue, iscodesorted, countcolons

    include("view.jl")      # orient, Reverse{d} etc
    export diagview, orient, rview, mul!, star, PermuteDims, Reverse, Shuffle

    # include("lazy.jl")      # LazyCast # this costs about 3s in my test, 3.8s -> 7.7s

    include("static.jl")    # StaticArrays
    export static_slice, static_glue

end
using .Fast
const mul! = Fast.mul!

include("warm.jl") # worth 2s in my test

end # module
