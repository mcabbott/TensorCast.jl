function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(rand),StepRange{Int64,Int64},Int64,Int64,Int64})
    precompile(Tuple{typeof(rand),Type{Int64},Int64,Int64})
    precompile(Tuple{typeof(rand),UnitRange{Int64},Int64,Int64,Int64,Vararg{Int64,N} where N})
    precompile(Tuple{typeof(rand),UnitRange{Int64},NTuple{4,Int64}})
    precompile(Tuple{typeof(randn),Int64,Int64})
end
