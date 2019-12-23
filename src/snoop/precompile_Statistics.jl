function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Core.kwftype(typeof(Statistics.mean)),NamedTuple{(:dims,),Tuple{Tuple{Int64,Int64}}},typeof(mean),Array{Float64,4}})
    precompile(Tuple{Core.kwftype(typeof(Statistics.std)),NamedTuple{(:dims,),Tuple{Tuple{Int64,Int64}}},typeof(std),PermutedDimsArray{Float64,5,(5, 1, 4, 2, 3),(2, 4, 5, 3, 1),Array{Float64,5}}})
end
