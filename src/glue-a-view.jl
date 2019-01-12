
using .StaticArrays

# import TensorSlice: static_slice, static_glue # because Revise doesn't track these

@inline function static_slice(A::Array{T,N}, sizes::Size{tup}, finalshape::Bool=true) where {T,N,tup}
    IN = length(tup)
    IT = SArray{Tuple{tup...}, T, IN, prod(tup)}
    if N-IN>1 && finalshape
        finalsize = ntuple(d -> size(A,d+IN), Val(N-IN))
        reshape(reinterpret(IT, vec(A)), finalsize)
    else
        reinterpret(IT, vec(A)) # always a vector
    end
end

@inline function static_glue(A::AbstractArray{IT,ON}, finalshape::Bool=true) where {IT,ON}
    IT <: StaticArray || throw(TypeError("Can't static_glue slices which aren't StaticArrays"))
    if finalshape
        finalsize = (size(IT)..., size(A)...)
        reshape(reinterpret(eltype(IT), A), finalsize)
    else
        reinterpret(eltype(eltype(A)), A)
    end
end
