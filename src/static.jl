
using .StaticArrays

# import TensorCast: static_slice, static_glue # because Revise doesn't track these

"""
    static_slice(A, sizes)

Slice array `A` into an `AbstractArray{StaticArray}`,
with the slices always belonging to the first few indices, i.e. `code = (:,:,...,*,*)`.
Expects `sizes = Size(size(A)[1:ncolons])` for stability.

Typically produces `reshape(reinterpret(SArray...`; copy this to get `Array{SArray}`,
or call `static_slice(A, sizes, false)` to omit the reshape.
"""
@inline function static_slice(A::AbstractArray{T,N}, sizes::Size{tup}, finalshape::Bool=true) where {T,N,tup}
    IN = length(tup)
    IT = SArray{Tuple{tup...}, T, IN, prod(tup)}
    if N-IN>1 && finalshape
        finalsize = size(A)[1+IN:end]
        reshape(reinterpret(IT, vec(A)), finalsize)
    else
        reinterpret(IT, vec(A)) # always a vector
    end
end

"""
    static_glue(A)

Glues the output of `static_slice` back into one array, again with `code = (:,:,...,*,*)`.
"""
@inline function static_glue(A::AbstractArray{IT,ON}, finalshape::Bool=true) where {IT,ON}
    IT <: StaticArray || error("static_glue needs an array of StaticArrays")
    if finalshape
        finalsize = (size(IT)..., size(A)...)
        reshape(reinterpret(eltype(IT), A), finalsize)
    else
        reinterpret(eltype(eltype(A)), A)
    end
end

function auto_glue(A::AbstractArray{IT,ON}, code::Tuple) where {IT<:StaticArray,ON}
    if iscodesorted(code)
        static_glue(A)
    else
        copy_glue(A, code)
    end
end
