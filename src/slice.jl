
"""
    sliceview(A, code)
    slicecopy(A, code)

Slice array `A` according to `code`, a tuple of length `ndims(A)`,
in which `:` indicates a dimension of the slices, and `*` a dimension separating them.
For example if `code = (:,*,:)` then slices are either `view(A, :,i,:)`
or `A[:,i,:]` with `i=1:size(A,2)`.
"""
function sliceview(A::AbstractArray{T,N}, code::Tuple) where {T,N}
    N == length(code) || throw(ArgumentError("wrong code length"))
    if code == (:,*)
        collect(eachcol(A))
    elseif code == (*,:)
        collect(eachrow(A))
    elseif count(isequal(*), code) == 1
        collect(eachslice(A, dims = findfirst(isequal(*), code)))
    else
        iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(A,d) : Ref(:), Val(N))...)
        [ view(A, i...) for i in iter ]
    end
end

@doc @doc(sliceview)
function slicecopy(A::AbstractArray{T,N}, code::Tuple) where {T,N}
    N == length(code) || throw(ArgumentError("wrong code length"))
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(A,d) : Ref(:), Val(N))...)
    [ A[i...] for i in iter ]
end

sliceview(A::AbstractArray{T,N}) where {T,N} = sliceview(A, ntuple(i-> i==N ? (*) : (:), N))
slicecopy(A::AbstractArray{T,N}) where {T,N} = slicecopy(A, ntuple(i-> i==N ? (*) : (:), N))

"""
    glue!(B, A, code)
    glue(A, code) = glue!(Array{T}(...), A, code)

Copy the contents of an array of arrays into one larger array,
un-doing `sliceview` / `slicecopy` with the same `code`.
Also called `stack` or `align` elsewhere.

    cat_glue(A, code)
    red_glue(A, code)

The same result, but calling either things like `hcat(A...)`
or things like `reduce(hcat, A)`.
The code must be sorted like `(:,:,:,*,*)`, except that `(*,:)` is allowed.
"""
glue(A::AbstractArray, code::Tuple) = copy_glue(A, code)

@inline function red_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    if code == (:,*)
        reduce(hcat, A) # this is fast, specially optimised
    elseif code == (*,:)
        # mapreduce(transpose, vcat, A) # this is slow
        reduce(vcat, PermuteDims.(A)) # now avoiding transpose
    # elseif count(isequal(*), code) == 1 && code[end] == (*)
    #     reduce((x,y) -> cat(x,y; dims = length(code)), A) # this is slow
    elseif iscodesorted(code)
        flat = reduce(hcat, vec(vec.(A)))
        reshape(flat, gluedsize(A, code))
    else
        throw(ArgumentError("can't glue code = $code with reduce(cat...)"))
    end
end

@inline function cat_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    if code == (:,*)
        hcat(A...)
    elseif code == (*,:)
        vcat(PermuteDims.(A)...)
    elseif count(isequal(*), code) == 1 && code[end] == (*)
        cat(A...; dims = length(code))
    elseif iscodesorted(code)
        flat = cat(vec(A)...; dims = length(code)-ndims(A)+1)
        reshape(flat, gluedsize(A, code))
    else
        throw(ArgumentError("can't glue code = $code with cat(A...)"))
    end
end

@inline function copy_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    B = Array{eltype(first(A))}(undef, gluedsize(A, code))
    glue!(B, A, code)
end

function glue!(B::AbstractArray{T,N}, A::AbstractArray{IT,ON}, code::Tuple) where {T,N,IT,ON}
    gluecodecheck(A, code)
    N == ndims(A) + ndims(first(A))  || throw(DimensionMismatch("wrong size target"))
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(B,d) : Ref(:), Val(N))...)
    for i in iter
        copyto!(view(B, i...), A[decolonise(i)...] )
    end
    B
end

function gluecodecheck(A::AbstractArray, code::Tuple)
    countcolons(code) == ndims(first(A)) || throw(ArgumentError("wrong number of : in code"))
    length(code) == ndims(A) + ndims(first(A)) || throw(ArgumentError("wrong code length"))
end

@generated function decolonise(i::Tuple)
    ind = Int[]
    for k in 1:length(i.parameters)
        if i.parameters[k] != Colon
            push!(ind, k)
        end
    end
    Expr(:tuple, [Expr(:ref, :i, k) for k in ind]...)
end

@generated function countcolons(code::Tuple)
    n = 0
    for s in code.parameters
        if s == Colon
            n += 1
        end
    end
    n
end

@generated function iscodesorted(code::Tuple) # true if all colons before all stars
    colons = true
    sorted = true
    for s in code.parameters
        if s != Colon && colons # then we're at transition, change flag
            colons = false
        elseif s == Colon && !colons # then a : is following a *
            sorted = false
        end
    end
    sorted
end

@generated function gluedsize(A::AbstractArray{IT, N}, code::Tuple) where {IT, N}
    list = Any[]
    dout = 1
    din = 1
    for s in code.parameters
        if s == Colon
            push!(list, :( size(first(A),$din) ))
            din += 1
        else
            push!(list, :( size(A,$dout) ))
            dout += 1
        end
    end
    :( ($(list...),) )
end


"""
    B = orient(A, code)

Reshapes `A` such that its nontrivial axes lie in the directions where `code` contains a `:`,
by inserting axes on which `size(B, d) == 1` as needed.
Throws an error if `ndims(A) != length(code)`.
"""
@generated function orient(A::AbstractArray, code::Tuple)
    list = Any[]
    pretty = Any[] # just for error
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
orient(A::Union{PermutedDimsArray, LinearAlgebra.Transpose, LinearAlgebra.Adjoint}, code::Tuple) =
    orient(collect(A), code)
#=
A = rand(10,100); B = rand(100,100); C = rand(100,10);
@time @reduce D[a,d] := sum(b,c) A[a,b] * B[b,c] * C[c,d] lazy;
# was 0.142307 seconds 76.296 MiB, now 0.001855 seconds 10.516 KiB, factor 75
=#

# performance: avoid that if you can just extract A.parent. TODO make @generated perhaps?
const LazyRowVec = Union{
    LinearAlgebra.Transpose{<:Any,<:AbstractVector},
    LinearAlgebra.Adjoint{<:Real,<:AbstractVector} }
orient(A::LazyRowVec, ::Tuple{typeof(*),Colon,Colon}) = orient(A.parent, (*,*,:))
orient(A::LazyRowVec, ::Tuple{Colon,typeof(*),Colon}) = orient(A.parent, (*,*,:))
orient(A::LazyRowVec, ::Tuple{typeof(*),typeof(*),Colon,Colon}) = orient(A.parent, (*,*,*,:))
orient(A::LazyRowVec, ::Tuple{typeof(*),Colon,typeof(*),Colon}) = orient(A.parent, (*,*,*,:))
orient(A::LazyRowVec, ::Tuple{Colon,typeof(*),typeof(*),Colon}) = orient(A.parent, (*,*,*,:))

# tidier but not strictly necc: for @cast Z[i] = A[3] + B[i]
orient(A::AbstractArray{<:Any, 0}, ::Tuple{typeof(*)}) = first(A)

"""
	rview(A, :,1,:) ≈ (@assert size(A,2)==1; view(A, :,1,:))

This simply reshapes `A` so as to remove a trivial dimension where indicated.
Throws an error if size of `A` is not 1 in the indicated dimension.

Will fail silently if given a `(:,3,:)` or `(:,\$c,:)`,
for which `needview!(code) = true`, so the macro should catch such cases.
"""
function rview(A::AbstractArray, code...)
	# This nice error has to happen at runtime, and thus is slow
	# all(decolonise(code) .== 1) ||
	#     throw(ArgumentError("can't use rview(A, code...) with code = $(pretty(code)), it accepts only 1 and :"))
	rview(A, code)
end

@generated function rview(A::AbstractArray, code::Tuple)
    list = Any[]
    pretty = Any[] # just for error
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

rview(A::Union{PermutedDimsArray, LinearAlgebra.Transpose, LinearAlgebra.Adjoint}, code::Tuple) =
    view(A, code)
# cannot use rview(collect(A), code) as this may be on LHS, e.g. sum!(rview(),...)

"""
    rvec(x) = [x]
    rvec(A) = vec(A)

Really-vec... extends `LinearAlgebra.vec` to work on scalars too.
"""
rvec(x::Number) = [x]
rvec(A) = LinearAlgebra.vec(A)

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

"""
    diagview(M)

Like `diag(M)` but makes a view.
"""
function diagview(A::AbstractMatrix)
    r,c = size(A)
    view(A, 1:(r+1):r*c)
end

diagview(A::LinearAlgebra.Diagonal) = A.diag
