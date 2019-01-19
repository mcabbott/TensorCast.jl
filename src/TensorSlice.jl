module TensorSlice

using MacroTools

V = false # capital V for extremely verbose
W = false # printouts where I'm working on things

include("parse.jl")
include("shape.jl")
include("icheck.jl")
include("cast.jl")
include("pretty.jl")

@warn """Thanks for trying out my package! 

I have now changed its name TensorSlice.jl -> TensorCast.jl

You should probably remove this, and download again under the new name:

] rm TensorSlice
  add https://github.com/mcabbott/TensorCast.jl

You will also need to replace @shape with @cast everywhere 
(and possibly replace some @cast with @reduce, sorry.) 
It now handles arbitrary broadcasting at the same time as 
all previous slicing, squeezing & dicing functions. """

if VERSION < v"1.1.0"
    include("eachslice.jl") # functions from the future, TODO figure out Compat
end

include("cat-and-slice.jl")

using Requires

function __init__()

    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        include("glue-a-view.jl")
        V && @info "loaded code for StaticArrays"
    end

    @require JuliennedArrays = "5cadff95-7770-533d-a838-a1bf817ee6e0" begin
        include("julienne.jl")
        V && @info "loaded code for JuliennedArrays"
    end

    @require Strided = "5e0ebb24-38b0-5f93-81fe-25c709ecae67" begin
        include("lazy-stride.jl")
        V && @info "loaded code for Strided"
    end

    @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
        @info "loaded code for LazyArrays"
    end

end

end # module
