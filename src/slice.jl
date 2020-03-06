
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
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(A,d) : Ref(:), Val(N))...)
    [ view(A, i...) for i in iter ]
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
    copy_glue(A, code)

This is the inverse of `sliceview`, kept around only for ZygoteRules...
"""
@inline function copy_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    B = Array{eltype(first(A))}(undef, gluedsize(A, code))
    glue!(B, A, code)
end

@doc @doc(glue)
function glue!(B::AbstractArray{T,N}, A::AbstractArray{IT,ON}, code::Tuple) where {T,N,IT,ON}
    gluecodecheck(A, code)
    N == _ndims(A) + _ndims(first(A))  || throw(DimensionMismatch("wrong size target"))
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(B,d) : Ref(:), Val(N))...)
    for i in iter
        copyto!(view(B, i...), A[decolonise(i)...] )
    end
    B
end

function gluecodecheck(A::AbstractArray, code::Tuple)
    colons = countcolons(code)
    inner = _ndims(first(A))
    outer = _ndims(A)
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

_ndims(A) = ndims(A)
_ndims(A::Tuple) = 1

using ZygoteRules # TODO add tests?

# Rules moved from SliceView.jl

@adjoint sliceview(A::AbstractArray, code::Tuple) =
    sliceview(A, code), Δ -> (copy_glue(Δ, code), nothing)

@adjoint copy_glue(A::AbstractArray, code::Tuple) =
    copy_glue(A, code), Δ -> (sliceview(Δ, code), nothing)
