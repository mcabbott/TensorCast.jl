# eachslice, eachcol, eachrow exactly as in Julia 1.1
# https://github.com/JuliaLang/julia/pull/29749/files
# https://github.com/JuliaLang/julia/blob/master/HISTORY.md#new-library-functions

export eachrow, eachcol, eachslice

"""
    eachrow(A::AbstractVecOrMat)
Create a generator that iterates over the first dimension of vector or matrix `A`,
returning the rows as views.
See also [`eachcol`](@ref) and [`eachslice`](@ref).
"""
eachrow(A::AbstractVecOrMat) = (view(A, i, :) for i in axes(A, 1))


"""
    eachcol(A::AbstractVecOrMat)
Create a generator that iterates over the second dimension of matrix `A`, returning the
columns as views.
See also [`eachrow`](@ref) and [`eachslice`](@ref).
"""
eachcol(A::AbstractVecOrMat) = (view(A, :, i) for i in axes(A, 2))

"""
    eachslice(A::AbstractArray; dims)
Create a generator that iterates over dimensions `dims` of `A`, returning views that select all
the data from the other dimensions in `A`.
Only a single dimension in `dims` is currently supported. Equivalent to `(view(A,:,:,...,i,:,:
...)) for i in axes(A, dims))`, where `i` is in position `dims`.
See also [`eachrow`](@ref), [`eachcol`](@ref), and [`selectdim`](@ref).
"""
@inline function eachslice(A::AbstractArray; dims)
    length(dims) == 1 || throw(ArgumentError("only single dimensions are supported"))
    dim = first(dims)
    dim <= ndims(A) || throw(DimensionMismatch("A doesn't have $dim dimensions"))
    idx1, idx2 = ntuple(d->(:), dim-1), ntuple(d->(:), ndims(A)-dim)
    return (view(A, idx1..., i, idx2...) for i in axes(A, dim))
end

# also from 1.1, like === not ==
isnothing(::Any) = false
isnothing(::Nothing) = true
