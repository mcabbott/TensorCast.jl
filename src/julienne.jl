
using JuliennedArrays

# import TensorSlice: sliceview, glue # because Revise doesn't track these

if isdefined(JuliennedArrays, :julienne) # old notation

    @inline sliceview(A::AbstractArray{T,N}, code::Tuple, sizes=nothing) where {T,N} =
        collect(julienne(A, code)) # TODO test whether this is slow? it's tidier

    @inline glue(A::AbstractArray{IT,N}, code::Tuple, sizes=nothing) where {IT,N} =
        align(A, code)

elseif isdefined(JuliennedArrays, :Slices) # new notation, just saw it

    @warn "haven't yet tried out JuliennedArrays's new notation!" # TODO

    @inline sliceview(A::AbstractArray{T,N}, code::Tuple, sizes=nothing) where {T,N} =
        Slices(A, code .== (*) )

    @inline glue(A::AbstractArray{IT,N}, code::Tuple, sizes=nothing) where {IT,N} =
        Align(A, code .== (*) )

else
    throw(ErrorException("@shape can't seem to find functions from JuliennedArrays"))
end
