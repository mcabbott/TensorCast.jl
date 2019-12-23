
"""
    diagview(M) = view(M, diagind(M))

Like `diag(M)` but makes a view.
"""
diagview(A::AbstractMatrix) = view(A, diagind(A))

diagview(A::LinearAlgebra.Diagonal) = A.diag

"""
    B = orient(A, code)

Reshapes `A` such that its nontrivial axes lie in the directions where `code` contains a `:`,
by inserting axes on which `size(B, d) == 1` as needed.
Throws an error if `ndims(A) != length(code)`.

When acting on `A::Transpose`, `A::PermutedDimsArray` etc, it will `collect(A)` first,
because reshaping these is very slow.
"""
@generated function orient(A::AbstractArray, code::Tuple)
    list = []
    pretty = [] # just for error
    d = 1
    for s in code.parameters
        if s == Colon
            push!(list, :( size(A,$d) ))
            push!(pretty, ":")
            d += 1
        else
            push!(list, 1)
            push!(pretty, "*")
        end
    end
    str = join(pretty, ", ")
    d-1 == ndims(A) || throw(ArgumentError(
        "orient(A, ($str)) expeceted $(d-1) dimensions, got ndims(A) = $(ndims(A))"))
    :(reshape(A, ($(list...),)))
end

# performance: transpose faster than reshape? https://github.com/JuliaArrays/LazyArrays.jl/issues/16
orient(A::AbstractVector{T}, ::Tuple{typeof(*),Colon}) where {T <: Number} = PermuteDims(A)
#=
V = rand(500);
@time @reduce W[i] := sum(j,k) V[i]*V[j]*V[k] lazy;
# was 0.140318 seconds seconds, now 0.030957 seconds, factor 4
=#

# performance: avoid reshaping lazy transposes, as this is very slow in broadcasting
const LazyPerm = Union{PermutedDimsArray, LinearAlgebra.Transpose, LinearAlgebra.Adjoint}
orient(A::Union{LazyPerm, SubArray{<:Any,<:Any,<:LazyPerm}}, code::Tuple) = orient(collect(A), code)
#=
A = rand(10,100); B = rand(100,100); C = rand(100,10);
@time @reduce D[a,d] := sum(b,c) A[a,b] * B[b,c] * C[c,d] lazy;
# was 0.142307 seconds 76.296 MiB, now 0.001855 seconds 10.516 KiB, factor 75
=#

# performance: avoid that if you can just extract A.parent. TODO make @generated perhaps?
const LazyRowVec = Union{
    LinearAlgebra.Transpose{<:Any,<:AbstractVector},
    LinearAlgebra.Adjoint{<:Real,<:AbstractVector} } # NB only Adjoint{Real}
orient(A::LazyRowVec, ::Tuple{typeof(*),Colon,Colon}) = orient(A.parent, (*,*,:))
orient(A::LazyRowVec, ::Tuple{Colon,typeof(*),Colon}) = orient(A.parent, (*,*,:))
orient(A::LazyRowVec, ::Tuple{typeof(*),typeof(*),Colon,Colon}) = orient(A.parent, (*,*,*,:))
orient(A::LazyRowVec, ::Tuple{typeof(*),Colon,typeof(*),Colon}) = orient(A.parent, (*,*,*,:))
orient(A::LazyRowVec, ::Tuple{Colon,typeof(*),typeof(*),Colon}) = orient(A.parent, (*,*,*,:))

"""
    rview(A, :,1,:) ≈ (@assert size(A,2)==1; view(A, :,1,:))

This simply reshapes `A` so as to remove a trivial dimension where indicated.
Throws an error if size of `A` is not 1 in the indicated dimension.

Will fail silently if given a `(:,3,:)` or `(:,\$c,:)`,
for which `needview!(code) = true`, so the macro should catch such cases.
"""
rview(A::AbstractArray, code...) = rview(A, code)

@generated function rview(A::AbstractArray, code::Tuple)
    list = []
    pretty = [] # just for error
    ex = quote end
    d = 1
    for s in code.parameters
        if s == Colon
            push!(list, :( size(A,$d) ))
            push!(pretty, ":")
            d += 1
        elseif s == Int
            push!(pretty, "1")
            str = "rview(A) expected size(A,$d) == 1"
            push!(ex.args, :(size(A,$d)==1|| throw(DimensionMismatch($str)) ) )
            d += 1
        end
    end
    str = join(pretty, ", ")
    d-1 == ndims(A) || throw(ArgumentError(
        "rview(A, ($str)) expeceted $(d-1) dimensions, got ndims(A) = $(ndims(A))"))
    push!(ex.args, :(reshape(A, ($(list...),))) )
    ex
end

rview(A::LazyRowVec, ::Tuple{Int,Colon}) = A.parent

rview(A::LazyPerm, code::Tuple) = view(A, code)
# cannot use rview(collect(A), code) as this may be on LHS, e.g. sum!(rview(),...)

"""
    rvec(x) = [x]
    rvec(A) = vec(A)

Really-vec... extends `LinearAlgebra.vec` to work on scalars too.
"""
rvec(x::Number) = [x]
rvec(A) = LinearAlgebra.vec(A)

"""
    zeroarray(x)

Like `[x]` but with one less dimension.
"""
function zeroarray(x::T) where T
    out = Array{T,0}(undef)
    out[] = x
    out
end

"""
    mul!(Z,A,B)

Exactly `LinearAlgebra.mul!` except that it can write into a zero-array.
"""
mul!(Z,A,B) = LinearAlgebra.mul!(Z,A,B)
mul!(Z::AbstractArray{T,0}, A,B) where {T} = copyto!(Z, A * B)

"""
    apply(f,x...) = f(x...)

For broadcasting a list of functions.
"""
apply(f,x...) = f(x...)

"""
    star(x,y,...)

Like `*` but intended for multiplying sizes, and understands that `:` is a wildcard.
"""
star(x,y) = *(x,y)
star(::Colon,y) = Colon()
star(x,::Colon) = Colon()
star(x,y,zs...) = star(star(x,y), zs...)

"""
    PermuteDims(A::Matrix)
    PermuteDims(A::Vector)

Lazy like `transpose`, but not recursive:
calls `PermutedDimsArray` unless `eltype(A) <: Number`.
"""
PermuteDims(A::AbstractMatrix) = PermutedDimsArray(A, (2,1))
PermuteDims(A::AbstractMatrix{T}) where {T<:Number} = transpose(A)

PermuteDims(A::AbstractVector) = reshape(A,1,:)
PermuteDims(A::AbstractVector{T}) where {T<:Number} = transpose(A)


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

# _Shuffle & _Reverse are varying attempts, none seem make constant propagation work

struct Circshift{D,S} end

"""
    Circshift{d,s}(A)

Lazy version of `circshift(A, (0,0,0,s,0,0))` in the `d`-th dimension.
"""
function Circshift{D,S}(A::AbstractArray{T,N}) where {D,S, T,N}
    1 ≤ D ≤ N || throw(ArgumentError("dimension $D is not 1 ≤ $D ≤ $N"))
    tup = ntuple(i -> i==D ? circshift(axes(A,D), S) : Colon(), N)
    view(A, tup...)
end

Circshift(A::AbstractArray, s::Int; dims::Int=ndims(A)) = Circshift{dims,s}(A)

# A real type with shift baked into getindex might be better?

struct Clip{D,A,Z} end

"""
    Clip{d,α,ω}(A)

Shortens the range of the `d`-th index at start & end; `Clip{d,0,0}(A)` does nothing.
"""
function Clip{D,a,z}(A::AbstractArray{T,N}) where {D,a,z, T,N}
    1 ≤ D ≤ N || throw(ArgumentError("dimension $D is not 1 ≤ $D ≤ $N"))
    tup = ntuple(i -> i==D ? axes(A,D)[1+a,end-z] : Colon(), N)
    view(A, tup...)
end

Clip(A::AbstractArray, a::Int=0, z::Int=0; dims::Int=ndims(A)) = Circshift{dims,a,z}(A)

# @cast A[i] := A[i+1] - A[i]/B[i-1]  clip   # global, should I re-write somehow?
# @cast A[i] := A[i+1] - A[i]  cyclic        # easy, per-term
