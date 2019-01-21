module TensorSlice

@warn """Thanks for trying out my package! 

I have now changed its name TensorSlice.jl -> TensorCast.jl

You should probably remove this, and download again under the new name:

] rm TensorSlice
  add https://github.com/mcabbott/TensorCast.jl

You will also need to replace @shape with @cast everywhere 
(and possibly replace some @cast with @reduce, sorry.) 
It now handles arbitrary broadcasting at the same time as 
all previous slicing, squeezing & dicing functions. """

export @shape

macro shape(exs...)
    where = (mod=__module__, src=__source__)
    _macro(exs...; reduce=false, where=where)
end

# the rest is identical to TensorCast.jl

using MacroTools

V = false # verbose debugging

include("parse.jl")

include("macro.jl")

include("icheck.jl")

include("pretty.jl")

if VERSION < v"1.1.0"
    include("eachslice.jl") 
end

include("slice.jl")

include("recursive.jl")

using Requires

function __init__()

    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        include("static.jl")
    end

    @require Strided = "5e0ebb24-38b0-5f93-81fe-25c709ecae67" begin
        include("strided.jl")
    end

    @require JuliennedArrays = "5cadff95-7770-533d-a838-a1bf817ee6e0" begin
        include("julienne.jl")
    end

    # @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
    # end

end

end # module
