
using .JuliennedArrays

# import TensorSlice: sliceview, glue # because Revise doesn't track these

if isdefined(JuliennedArrays, :julienne) # old notation

    @inline sliceview(A::AbstractArray{T,N}, code::Tuple, sizes=nothing) where {T,N} =
        collect(julienne(A, code)) # TODO test whether this is slow? it's tidier

    @inline glue(A::AbstractArray{IT,N}, code::Tuple, sizes=nothing) where {IT,N} =
        align(A, code)

elseif isdefined(JuliennedArrays, :Slices) # new notation

    @warn "TensorSlice only partially works with the new version of JuliennedArrays, sorry."

    @inline sliceview(A::AbstractArray{T,N}, code::Tuple, sizes=nothing) where {T,N} =
        Slices(A, newcode(code)...)

    @inline glue(A::AbstractArray{IT,N}, code::Tuple, sizes=nothing) where {IT,N} =
        collect(Align(A, newcode(code)...))

    @generated newcode(code::Tuple) = Expr(:tuple, [ c == Colon ? True() : False() for c in code.parameters ]...)

else
    @warn "@shape can't seem to find functions from JuliennedArrays. It should fall back to base methods in most cases."
end
