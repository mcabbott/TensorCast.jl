
using .NamedArrays

# import TensorCast: namedarray # because Revise doesn't track these

namedarray(A::AbstractArray, syms...) = namedarray(A, syms)

namedarray(A::AbstractArray, syms::Tuple) = namedarray(NamedArray(A), syms)

function namedarray(A::NamedArrays.NamedArray, syms::Tuple)
    for d=1:min(ndims(A), length(syms))
        if syms[d] isa Symbol
            NamedArrays.setdimnames!(A, syms[d], d)
        elseif syms[d] === 1
            NamedArrays.setdimnames!(A, :_, d)
        end
    end
    A
end
