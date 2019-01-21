
"""
    sliceview(A, code)
    slicecopy(A, code)

Slice array `A` according to `code`, a tuple of length `ndims(A)`, 
in which `:` indicates a dimension of the slices, and `*` a dimension separating them. 
For example if `code = (:,*,:)` then slices are either `view(A, :,i,:)` 
or `A[:,i,:]` with `i=1:size(A,2)`. 
"""
function sliceview(A::AbstractArray{T,N}, code::Tuple) where {T,N}
    N == length(code) || error("wrong code length")
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
    N == length(code) || error("wrong code length")
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(A,d) : Ref(:), Val(N))...)
    [ A[i...] for i in iter ]
end

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
        reduce(hcat, A)
    elseif code == (*,:)
        mapreduce(transpose, vcat, A)
    elseif count(isequal(*), code) == 1 && code[end] == (*)
        reduce((x,y) -> cat(x,y; dims = length(code)), A)
    elseif iscodesorted(code)
        flat = reduce((x,y) -> cat(x,y; dims = length(code)-ndims(A)+1), vec(A))
        finalsize = (size(first(A))..., size(A)...)
        reshape(flat, finalsize)
    else
        error("can't glue code = $code with reduce(cat...)")
    end
end

@inline function cat_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    if code == (:,*)
        hcat(A...)
    elseif code == (*,:)
        vcat(transpose.(A)...)
    elseif count(isequal(*), code) == 1 && code[end] == (*)
        cat(A...; dims = length(code))
    elseif iscodesorted(code)
        flat = cat(vec(A)...; dims = length(code)-ndims(A)+1)
        finalsize = (size(first(A))..., size(A)...)
        reshape(flat, finalsize)
    else
        error("can't glue code = $code with cat(A...)")
    end
end

@inline function copy_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    B = Array{eltype(first(A))}(undef, gluedsize(A, code))
    glue!(B, A, code)
end

@doc @doc(glue)
function glue!(B::AbstractArray{T,N}, A::AbstractArray{IT,ON}, code::Tuple) where {T,N,IT,ON}
    gluecodecheck(A, code)
    N == ndims(A) + ndims(first(A))  || error("wrong size target")
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(B,d) : Ref(:), Val(N))...)
    for i in iter
        copyto!(view(B, i...), A[decolonise(i)...] )
    end
    B
end

function gluecodecheck(A::AbstractArray, code::Tuple)
    countcolons(code) == ndims(first(A)) || error("wrong number of : in code")
    length(code) == ndims(A) + ndims(first(A)) || error("wrong code length")
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
        "orient(A, ($str)) got ndims(A) = $(ndims(A)), expeceted n = $(d-1)"))
    :(reshape(A, ($(list...),))) 
end

# because of https://github.com/JuliaArrays/LazyArrays.jl/issues/16
orient(A::AbstractVector, ::Tuple{typeof(*),Colon}) = transpose(A)
orient(A::AbstractVector, ::Tuple{typeof(*),Colon,typeof(*)}) = transpose(A)

