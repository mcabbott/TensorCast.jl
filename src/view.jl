
"""
    diagview(M) = view(M, diagind(M))

Like `diag(M)` but makes a view.
"""
diagview(A::AbstractMatrix) = view(A, diagind(A))

diagview(A::LinearAlgebra.Diagonal) = A.diag

"""
    storage_type(A)

Return the type of the underlying matrix for PermutedDimsArray, Transpose, etc...,
(e.g., `CuArray{Float64,2}` or `Array{Float64,1}`).
"""
function storage_type(A::AbstractArray)
    P = parent(A)
    typeof(A) === typeof(P) ? typeof(A) : storage_type(P)
end

"""
    B = orient(A, code)

Usually this calls `_orient(A, code)`, which reshapes `A` such that its
nontrivial axes lie in the directions where `code` contains a `:`,
by inserting axes on which `size(B, d) == 1` as needed.

When acting on `A::Transpose`, `A::PermutedDimsArray` etc, it will `collect(A)` first,
because reshaping these is very slow. However, this is only done when the underlying array
is a "normal" CPU `Array`, since e.g., for GPU arrays, `collect(A)` copies the array to
the CPU. Fortunately, the speed penalty for reshaping transposed GPU arrays is lower than
on the CPU.
"""
orient(A::AbstractArray, code::Tuple) = _orient(A::AbstractArray, code::Tuple)

@generated function _orient(A::AbstractArray, code::Tuple)
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

# performance: avoid reshaping lazy transposes (of CPU arrays), as this is very slow in broadcasting
const LazyPerm = Union{PermutedDimsArray, LinearAlgebra.Transpose, LinearAlgebra.Adjoint}
orient(A::Union{LazyPerm, SubArray{<:Any,<:Any,<:LazyPerm}}, code::Tuple) = begin
    if storage_type(A) <: Array
        # for "normal" CPU arrays, collect before reshaping
        _orient(collect(A), code)
    else
        # for other storage (e.g., GPU arrays), call the original algorithm with reshape,
        # since collect copies to CPU. Fortunately, reshape is not slow on GPU
        _orient(A,code)
    end
end
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

using OffsetArrays
# reshaping causes offsets to be forgotten, https://github.com/mcabbott/TensorCast.jl/issues/11
# so explicitly copy them around:
@generated function _orient(A::OffsetArray, code::Tuple)
    off = []
    d = 1
    for s in code.parameters
        if s == Colon
            push!(off, :( A.offsets[$d] ))
            d += 1
        else
            push!(off, 0)
        end
    end
    :( OffsetArray(_orient(parent(A), code), ($(off...),)) )
end

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
