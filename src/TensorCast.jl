
module TensorCast

# This speeds up loading a bit, on Julia 1.5, about 1s in my test.
# https://github.com/JuliaPlots/Plots.jl/pull/2544/files
if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    @eval Base.Experimental.@optlevel 1
end

export @cast, @reduce, @matmul, @pretty

using MacroTools, StaticArrays, LazyStack, Compat
using LazyStack: stack_iter
using LinearAlgebra, Random

if VERSION < v"1.5" # not sure!
    using Compat # takes about 0.5s
end

include("macro.jl")
include("pretty.jl")
include("string.jl")

module Fast # shield non-macro code from @optlevel 1
    using ..TensorCast: pretty
    using LinearAlgebra, StaticArrays, Compat

    include("slice.jl")     # slice, glue, etc
    export sliceview, slicecopy, glue, glue!, red_glue, cat_glue, copy_glue, lazy_glue, iscodesorted, countcolons

    include("view.jl")      # orient, Reverse{d} etc
    export diagview, orient, rview, mul!, star, PermuteDims, Reverse, Shuffle

    include("static.jl")    # StaticArrays
    export static_slice, static_glue

end
using .Fast
const mul! = Fast.mul!

using Requires

@init @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
    include("lazy.jl")      # LazyCast # this costs about 3s in my test, 3.8s -> 7.7s
end

include("warm.jl") # was worth 2s in my test

end # module
