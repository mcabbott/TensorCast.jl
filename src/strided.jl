
using .Strided

# import TensorCast: strided_permutedims, strided_permutedims!, maybecopy # because Revise doesn't track these

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

maybecopy(A::StridedView) = copy(A)

# If you ask for |=, this may be able to guarantee a copy -- tidier thing for slicecopy?
# it won't help with permutedims fallback story... maybe that should be stricter? spermutedims?
