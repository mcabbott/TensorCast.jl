
using StaticArrays

# import TensorSlice: static_slice, static_glue # because Revise doesn't track these

@inline function static_slice(A::Array{T,N}, sizes::Size{tup}, finalshape::Bool=true) where {T,N,tup}
    IN = length(tup) # number of dimensions of the slice
    IT = SArray{Tuple{tup...}, T, IN, prod(tup)}
    if finalshape
        out_sizes = ntuple(d -> size(A,d+N-IN), Val(N-IN)) # reinterpret makes a vector!
        reshape(reinterpret(IT, vec(A)), out_sizes)
    else
        println("static_slice without reshape")
        reinterpret(IT, vec(A)) # if this is followed by a reshape anyway...
    end
end

@inline function static_glue(A::AbstractArray{IT,ON}, finalshape::Bool=true) where {IT,ON}
    if IT <: StaticArray
        if finalshape
            out_sizes = (size(IT)..., size(A)...)
            reshape(reinterpret(eltype(eltype(A)), A), out_sizes)
        else
            reinterpret(eltype(eltype(A)), A)
        end
    else
        throw(ArgumentError("@shape can't static_glue slices which aren't StaticArrays"))
    end
end
