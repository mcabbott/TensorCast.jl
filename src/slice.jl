
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
    copy_glue(A, code) = glue!(Array{T}(...), A, code)

Copy the contents of an array of arrays into one larger array,
un-doing `sliceview` / `slicecopy` with the same `code`.
This function can handle arbitrary codes.
(Also called `stack` or `align` elsewhere.)

    cat_glue(A, code)
    red_glue(A, code)

The same result, but calling either things like `hcat(A...)`
or things like `reduce(hcat, A)`.
The code must be sorted like `(:,:,:,*,*)`, except that `(*,:)` is allowed.

    glue(A)
    glue(A, code)

If `code` is omitted, the default is like `(:,:,*,*)`
with `ndims(first(A))` colons first, then `ndims(A)` stars.
If the inner arrays are `StaticArray`s (and the code is sorted) then it calls `static_glue`.
Otherwise it will call `red_glue`, unless code is unsuitable for that, in which case `copy_glue`.
"""
function glue(A::AbstractArray{<:AbstractArray{T,IN},ON}, code::Tuple=defaultcode(IN,ON)) where {T,IN,ON}
    if iscodesorted(code) || code == (*,:)
        red_glue(A, code)
    else
        copy_glue(A, code)
    end
end

glue(A::AbstractArray{<:Number,ON}, code::Tuple=defaultcode(0,ON)) where {ON} = A

defaultcode(IN::Int, ON::Int) = ntuple(d-> d<=IN ? (:) : (*), IN+ON)

@doc @doc(glue)
@inline function red_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    if code == (:,*)
        reduce(hcat, A) # this is fast, specially optimised
    elseif code == (*,:)
        reduce(vcat, PermuteDims.(A))
    elseif code == (:,:,*)
        mat = reduce(hcat, A)
        reshape(mat, gluedsize(A, code))
    elseif iscodesorted(code)
        mat = reduce(hcat, vec(vec.(A)))
        reshape(mat, gluedsize(A, code))
    else
        throw(ArgumentError("can't glue code = $code with reduce(cat...)"))
    end
end

@doc @doc(glue)
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

@doc @doc(glue)
@inline function copy_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    B = Array{eltype(first(A))}(undef, gluedsize(A, code))
    glue!(B, A, code)
end

@doc @doc(glue)
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
    colons = countcolons(code)
    inner = ndims(first(A))
    outer = ndims(A)
    colons == inner || throw(DimensionMismatch("wrong number of dimensions: " *
        "ndims(first(A)) == $inner not cannot be glued with code = $(pretty(code))"))
    length(code) - colons == outer || throw(DimensionMismatch("wrong number of dimensions: " *
        "ndims(A) == $outer cannot be glued with code = $(pretty(code))"))
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

using LazyStack

@inline function lazy_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    if code == (*,:)
        PermuteDims(stack(A))
    elseif iscodesorted(code)
        stack(A)
    else
        error("can't glue code = $code with LazyStack")
    end
end

using ZygoteRules # TODO add tests?

# Rules moved from SliceView.jl

@adjoint sliceview(A::AbstractArray, code::Tuple) =
    sliceview(A, code), Δ -> (glue(Δ, code), nothing)

@adjoint red_glue(A::AbstractArray, code::Tuple) =
    red_glue(A, code), Δ -> (sliceview(Δ, code), nothing)

@adjoint copy_glue(A::AbstractArray, code::Tuple) =
    copy_glue(A, code), Δ -> (sliceview(Δ, code), nothing)

