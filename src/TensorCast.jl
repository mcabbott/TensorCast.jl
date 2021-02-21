
module TensorCast

# This speeds up loading a bit, on Julia 1.5, about 1s in my test.
# https://github.com/JuliaPlots/Plots.jl/pull/2544/files
if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    @eval Base.Experimental.@optlevel 1
end

export @cast, @reduce, @matmul, @pretty

using LinearAlgebra, Random

using MacroTools, StaticArrays

using TransmuteDims, LazyStack
using LazyStack: stack_iter

if VERSION < v"1.5" # not sure!
    using Compat # takes about 0.5s
end

using TransmuteDims, LazyStack
using LazyStack: stack_iter

include("macro.jl")
include("pretty.jl")
include("string.jl")

module Fast # shield non-macro code from @optlevel 1
    using ..TensorCast: pretty
    using LinearAlgebra, StaticArrays, Compat

    include("slice.jl")
    export sliceview, slicecopy, copy_glue, glue!, iscodesorted, countcolons

    include("view.jl")
    export diagview, mul!, rview, star, Reverse, Shuffle

    include("static.jl")
    export static_slice, static_glue

end
using .Fast
const mul! = Fast.mul!

using Requires

@init @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
    include("lazy.jl")      # LazyCast # this costs about 3s in my test, 3.8s -> 7.7s
end

include("warm.jl") # saves 3s in my test

end # module
