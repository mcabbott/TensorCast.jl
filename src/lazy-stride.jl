
using Strided

# import TensorSlice: strided_permutedims, strided_permutedims!, maybecopy # because Revise doesn't track these

@inline function strided_permutedims(A::DenseArray, perm::Tuple)
    # @pretty @strided permutedims(A, perm)
    permutedims(Strided.maybestrided(A), Strided.maybestrided(perm))
end

@inline function strided_permutedims!(A::DenseArray, B::DenseArray, perm::Tuple)
    # @pretty @strided permutedims!(A, B, perm)
    permutedims!(Strided.maybestrided(A), Strided.maybestrided(B), Strided.maybestrided(perm))
end

## fall-back
@inline strided_permutedims(A::AbstractArray, perm::Tuple) = permutedims(A, perm) # but this copies!
@inline strided_permutedims!(A::AbstractArray, B::AbstractArray, perm::Tuple) = permutedims(A, perm)

## it turns out that A[1,:] is made a view not a copy
# @inline function slicecopy(A::StridedView{T,N}, code::Tuple, sizes=nothing) where {T,N}
#     # slicecopy(copy(A), code, sizes) # nope, copy(A) is still a StridedView
#     iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(A,d) : Ref(:), Val(N))...)
#     B = copy(A)
#     [ B[i...] for i in iter ]
# end

maybecopy(A::StridedView) = copy(A)

# If you ask for :=, this may be able to guarantee a copy -- tidier thing for slicecopy?
# it won't help with permutedims fallback story... maybe that should be stricter? spermutedims?
