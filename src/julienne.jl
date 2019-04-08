
using .JuliennedArrays

# import TensorCast: julienne_slice, julienne_glue, newcode, jcollect # because Revise doesn't track these

if isdefined(JuliennedArrays, :julienne) # old notation

    @inline julienne_slice(A::AbstractArray, code::Tuple) where {T,N} =
        collect(julienne(A, code))

    @inline julienne_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N} =
        align(A, code)

elseif isdefined(JuliennedArrays, :Slices) # new notation

    @inline julienne_slice(A::AbstractArray{T,N}, code::Tuple) where {T,N} =
        Slices(A, newcode(code)...)

    @inline julienne_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N} =
        jcollect(Align(A, newcode(code)...))

    @generated newcode(code::Tuple) = Expr(:tuple, [ c == Colon ? True() : False() for c in code.parameters ]...)

    # collect(Align(... and copy(Align(... both give errors
    function jcollect(A::JuliennedArrays.Align{T,N}) where {T,N}
        out = Array{T}(undef, size(A))
        @inbounds for I in CartesianIndices(A)
            out[I] = A[I.I...]
        end
        out
    end

else
    @warn "TensorCast failed to load functions from JuliennedArrays"
end
