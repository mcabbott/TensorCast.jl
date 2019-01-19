
using .JuliennedArrays

# import TensorCast: julienne_slice, julienne_glue # because Revise doesn't track these

if isdefined(JuliennedArrays, :julienne) # old notation

    @inline julienne_slice(A::AbstractArray, code::Tuple) where {T,N} =
        collect(julienne(A, code)) 

    @inline julienne_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N} =
        align(A, code)

elseif isdefined(JuliennedArrays, :Slices) # new notation

    @warn "TensorCast only partially works with the new version of JuliennedArrays, sorry."

    @inline julienne_slice(A::AbstractArray{T,N}, code::Tuple) where {T,N} =
        Slices(A, newcode(code)...)

    @inline julienne_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N} =
        collect(Align(A, newcode(code)...))

    @generated newcode(code::Tuple) = Expr(:tuple, [ c == Colon ? True() : False() for c in code.parameters ]...)

else
    @warn "@shape can't seem to find functions from JuliennedArrays. It should fall back to base methods in most cases."
end
