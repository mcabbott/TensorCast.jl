function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Core.kwftype(typeof(Base.sum)),NamedTuple{(:dims,),Tuple{Int64}},typeof(sum),LazyArrays.BroadcastArray{Float64,6,typeof(*),Tuple{Array{Float64,6},Array{Float64,6}}}})
    precompile(Tuple{typeof(sum!),Array{Float64,5},LazyArrays.BroadcastArray{Float64,6,typeof(*),Tuple{Array{Float64,6},Array{Float64,6}}}})
end
