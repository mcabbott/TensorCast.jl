
struct Reverse{D} end

"""
    Reverse{d}(A)
Lazy version of `reverse(A, dims=d)`.
"""
function Reverse{D}(A::AbstractArray{T,N}) where {D,T,N}
    1 ≤ D ≤ N || throw(ArgumentError("dimension $D is not 1 ≤ $D ≤ $N"))
    tup = ntuple(i -> i==D ? reverse(axes(A,D)) : Colon(), N)
    view(A, tup...)
end

Reverse(A::AbstractArray; dims=ndims(A)) = Reverse{dims}(A)

# The reverse() function in Base is twice as quick as copying this view,
# somewhere I had a real type in order to exploit this?

using Random

struct Shuffle{D} end

"""
    Shuffle{d}(A)
For a vector this is a lazy version of `shuffle(A)`,
for a tensor this shuffles along one axis only.
"""
function Shuffle{D}(A::AbstractArray{T,N}) where {D,T,N}
    1 ≤ D ≤ N || throw(ArgumentError("dimension $D is not 1 ≤ $D ≤ $N"))
    tup = ntuple(i -> i==D ? shuffle(axes(A,D)) : Colon(), N)
    view(A, tup...)
end

Shuffle(A::AbstractArray; dims=ndims(A)) = Shuffle{dims}(A)

"""
    multidiag(A, dims)
Like `permutedims` except that repeated dimensions mean a diagonal.
Thus `multidiag(M, (1,1)) == diag(M)` ...
"""
function multidiag(A::AbstractArray{T,N}, dims::NTuple{N,Int}) where {T,N}
    out = Array{T}(undef, [size(A,i) for i in unique(dims)]...)
    for I in CartesianIndices(out)
        out[I] = A[multiget(I,dims)]
    end
    out
end

multiget(ind::Tuple, tup::NTuple{N,Int}) where {N} = ntuple(d -> ind[tup[d]], N)

multiget(ind::CartesianIndex, tup::Tuple) = CartesianIndex(multiget(ind.I, tup))

# [i for i in Iterators.product(1:3,1:3)]
