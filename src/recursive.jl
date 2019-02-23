
using RecursiveArrayTools

@inline function lazy_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
    gluecodecheck(A, code)
    # if code == (:,*)
    #     VectorOfArray(A)
    # elseif
    if code == (*,:)
        transpose(VectorOfArray(A))
    # elseif count(isequal(*), code) == 1 && code[end] == (*)
    #     VectorOfArray(A)
    elseif iscodesorted(code)
        flat = VectorOfArray(vec(A))
        finalsize = (size(first(A))..., size(A)...)
        reshape(flat, finalsize)
        # always reshape so that it prints like an array,
        # also makes copy() produce an array
    else
        error("can't glue code = $code with VectorOfArray")
    end
end
