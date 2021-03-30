
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

include("tools.jl")
include("macro.jl")
include("pretty.jl")

module Fast # shield non-macro code from @optlevel 1
    using ..TensorCast: pretty
    using LinearAlgebra, StaticArrays

    include("slice.jl")
    export sliceview, slicecopy, copy_glue, glue!, iscodesorted, countcolons

    include("view.jl")
    export diagview, mul!, rview, star

    include("static.jl")
    export static_slice, static_glue

end
using .Fast
const mul! = Fast.mul!

include("warm.jl") # saves 3s in my test

end # module
