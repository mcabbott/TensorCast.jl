
"""
    diagview(M) = view(M, diagind(M))

Like `diag(M)` but makes a view.
"""
diagview(A::AbstractMatrix) = view(A, diagind(A))

diagview(A::LinearAlgebra.Diagonal) = A.diag

"""
    rvec(x) = [x]
    rvec(A) = vec(A)

Really-vec... extends `LinearAlgebra.vec` to work on scalars too.
"""
rvec(x::Number) = [x]
rvec(A) = LinearAlgebra.vec(A)

"""
    mul!(Z,A,B)

Exactly `LinearAlgebra.mul!` except that it can write into a zero-array.
"""
mul!(Z,A,B) = LinearAlgebra.mul!(Z,A,B)
mul!(Z::AbstractArray{T,0}, A,B) where {T} = copyto!(Z, A * B)

"""
    star(x,y,...)

Like `*` but intended for multiplying sizes, and understands that `:` is a wildcard.
"""
star(x,y) = *(x,y)
star(::Colon,y) = Colon()
star(x,::Colon) = Colon()
star(x,y,zs...) = star(star(x,y), zs...)

"""
    onetolength(1:10) == 10

Used to digest options `i in 1:10`, size calculation for reshaping only allows ranges starting at 1 for now.
"""
onetolength(ax::Base.OneTo) = length(ax)
onetolength(ax::AbstractUnitRange) = first(ax) == 1 ? length(ax) : error("ranges have to start at 1, for now")
onetolength(ax) = error("ranges must be AbstractUnitRange")

struct Reverse{D} end

"""
    Reverse{d}(A)

Lazy version of `reverse(A; dims=d)`. Also constructed by `Reverse(A; dims=d)`.
"""
function Reverse{D}(A::AbstractArray{T,N}) where {D,T,N}
    if D isa Tuple
        length(D)==0 && return A
        return Reverse{Base.tail(D)}(Reverse{first(D)}(A))
    end
    1 ≤ D ≤ N || throw(ArgumentError("dimension $D is not 1 ≤ $D ≤ $N"))
    tup = ntuple(i -> i==D ? reverse(axes(A,D)) : Colon(), N)
    view(A, tup...)
end

Reverse(A::AbstractArray; dims=ndims(A)) = _Reverse(A, dims...)

_Reverse(A::AbstractArray) = A
_Reverse(A::AbstractArray, d::Int, dims::Int...) = _Reverse(Reverse{d}(A), dims...)

Reverse!(A::AbstractArray; dims=ndims(A)) = A .= _Reverse(A, dims...)

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
    D isa Tuple && return _Shuffle(A, D)
    1 ≤ D ≤ N || throw(ArgumentError("dimension $D is not 1 ≤ $D ≤ $N"))
    tup = ntuple(i -> i==D ? shuffle(axes(A,D)) : Colon(), N)
    view(A, tup...)
end

Shuffle(A::AbstractArray; dims=ndims(A)) = _Shuffle(A, Tuple(dims))

_Shuffle(A::AbstractArray, dims::Tuple{}) = A
_Shuffle(A::AbstractArray, dims::Tuple) = _Shuffle(Shuffle{first(dims)}(A), Base.tail(dims))

Shuffle!(A::AbstractArray; dims=ndims(A)) = A .= _Shuffle(A, dims)
