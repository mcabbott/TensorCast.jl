
# using .RecursiveArrayTools

# import TensorCast: recursive_glue, gluecodecheck, iscodesorted

using RecursiveArrayTools

@inline function recursive_glue(A::AbstractArray{IT,N}, code::Tuple) where {IT,N}
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

# https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/47
function Base._reshape(parent::VectorOfArray, dims::Base.Dims)
    n = prod(size(parent))
    prod(dims) == n || Base._throw_dmrs(n, "size", dims)
    Base.__reshape((parent, IndexStyle(parent)), dims)
end
