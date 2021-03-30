
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
@inline function static_slice(A::AbstractArray{T,N}, sizes::Size{tup}) where {T,N,tup}
    IN = length(tup)
    for d in 1:IN
        size(A,d) == tup[d] || throw(DimensionMismatch("cannot slice array of size $(size(A)) using Size$tup"))
        first(axes(A,d)) == 1 || throw(ArgumentError("cannot creat static slices with offset indices"))
    end
    IT = SArray{Tuple{tup...}, T, IN, prod(tup)}
    finalaxes = axes(A)[1+IN:end]
    reshape(reinterpret(IT, vec(A)), finalaxes)
end

function static_slice(A::AbstractArray, code::Tuple)
    iscodesorted(code) || error("expected to make slices of left-most dimensions")
    tup = ntuple(d -> size(A,d), countcolons(code))
    static_slice(A, Size(tup))
end

"""
    static_glue(A)

Glues the output of `static_slice` back into one array, again with `code = (:,:,...,*,*)`.
For `MArray` slices, which can't be reinterpreted, this reverts to `red_glue`.
"""
@inline function static_glue(A::AbstractArray{IT}) where {IT<:SArray}
    finalaxes = (axes(IT)..., axes(A)...)
    reshape(reinterpret(eltype(IT), A), finalaxes)
end

function static_glue(A::AbstractArray{IT}) where {IT<:MArray}
    LazyStack.stack(A)
end

# This is now used only by maybestaticsizes...
iscodesorted(code::Tuple{}) = true
iscodesorted(code::Tuple{Any}) = true
iscodesorted(code::Tuple) = !(code[1]===(*) && code[2]===(:)) && iscodesorted(code[2:end])
