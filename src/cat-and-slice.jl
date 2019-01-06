
### Functions called by the output of @shape and @reduce, visible in @pretty output

function sliceview(A::AbstractArray{T,N}, code::Tuple, sizes=nothing) where {T,N}
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
# sliceview(A::AbstractArray{T,2}, code::Tuple{Colon,typeof(*)}, sizes=nothing) where {T} = collect(eachcol(A))
# sliceview(A::AbstractArray{T,2}, code::Tuple{typeof(*),Colon}, sizes=nothing) where {T} = collect(eachrow(A))

function slicecopy(A::AbstractArray{T,N}, code::Tuple, sizes=nothing) where {T,N}
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(A,d) : Ref(:), Val(N))...)
    [ maybecopy(A[i...]) for i in iter ]
end

maybecopy(A) = A # overloaded for strided views

glue(A::AbstractArray{IT,N}, code::Tuple, sizes=nothing) where {IT,N} = cat_glue(A, code, sizes)
# this separation is only so that I can test mine against JuliennedArrays easily
@inline function cat_glue(A::AbstractArray{IT,N}, code::Tuple, sizes=nothing) where {IT,N}
    if code == (:,*)
        reduce(hcat, A)
    elseif code == (*,:)
        reduce((x,y) -> vcat(maybetranspose(x),maybetranspose(y)), A)
    elseif count(isequal(*), code) == 1 && code[end] == (*)
        reduce((x,y) -> cat(x,y; dims = length(code)), A)
    else
        throw(ArgumentError("Don't know how to glue code = $(pretty(code)) with cat, try using JuliennedArrays"))
        # now that glue! works, you could just use that?
        glue!(Array{eltype(first(A))}(undef, final_size), A, code)
    end
end

maybetranspose(x::AbstractVector) = transpose(x)
maybetranspose(x) = x

function glue!(B::AbstractArray{T,N}, A::AbstractArray{IT,ON}, code::Tuple, sizes=nothing) where {T,N,IT,ON}
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(B,d) : Ref(:), Val(N))...)
    for i in iter
        # B[i...] .= A[decolonise(i)...]
        copyto!(view(B, i...), A[decolonise(i)...] )
    end
    B
end

@generated function decolonise(i::Tuple)  # thanks to @Mateusz Baran
    ind = Int[]
    for k in 1:length(i.parameters)
        if i.parameters[k] != Colon
            push!(ind, k)
        end
    end
    Expr(:tuple, [Expr(:ref, :i, k) for k in ind]...)
end

@inline sum_drop(A;  dims) = dropdims(sum(A; dims=dims); dims=dims)
@inline prod_drop(A; dims) = dropdims(prod(A; dims=dims); dims=dims)
@inline max_drop(A;  dims) = dropdims(maximum(A; dims=dims); dims=dims)


