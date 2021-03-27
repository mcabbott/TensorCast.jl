
"""
    static_slice(A, sizes)
    static_slice(A, code)

Slice array `A` into an `AbstractArray{StaticArray}`,
with the slices always belonging to the first few indices, i.e. `code = (:,:,...,*,*)`.
Prefers to get `sizes = Size(size(A)[1:ncolons])` for stability,
but if given code instead it will do what it can.

Typically produces `reshape(reinterpret(SArray...`; copy this to get `Array{SArray}`,
or call `static_slice(A, sizes, false)` to omit the reshape.
"""
@inline function static_slice(A::AbstractArray{T,N}, sizes::Size{tup}, finalshape::Bool=true) where {T,N,tup}
    IN = length(tup)
    for d in 1:IN
        size(A,d) == tup[d] || throw(DimensionMismatch("cannot slice array of size $(size(A)) using Size$tup"))
    end
    IT = SArray{Tuple{tup...}, T, IN, prod(tup)}
    if N-IN>1 && finalshape
        finalaxes = axes(A)[1+IN:end]
        reshape(reinterpret(IT, vec(A)), finalaxes)
    else
        reinterpret(IT, vec(A)) # always a vector
    end
end

function static_slice(A::AbstractArray, code::Tuple, finalshape::Bool=true)
    iscodesorted(code) || error("expected to make slices of left-most dimensions")
    tup = ntuple(d -> size(A,d), countcolons(code))
    static_slice(A, Size(tup), finalshape)
end

"""
    static_glue(A)

Glues the output of `static_slice` back into one array, again with `code = (:,:,...,*,*)`.
For `MArray` slices, which can't be reinterpreted, this reverts to `red_glue`.
"""
@inline function static_glue(A::AbstractArray{IT}, finalshape=true) where {IT<:SArray}
    if finalshape
        finalsize = (size(IT)..., size(A)...)
        reshape(reinterpret(eltype(IT), A), finalsize)
    else
        reinterpret(eltype(eltype(A)), A)
    end
end

function static_glue(A::AbstractArray{IT}, finalshape=true) where {IT<:MArray}
    stack(A)
end

# This is now used only by maybestaticsizes...
iscodesorted(code::Tuple{}) = true
iscodesorted(code::Tuple{Any}) = true
iscodesorted(code::Tuple) = !(code[1]===(*) && code[2]===(:)) && iscodesorted(code[2:end])
