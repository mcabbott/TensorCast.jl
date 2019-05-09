
using LazyArrays

#=
The macro option "lazy" always produces things like sum(@__dot__(lazy(x+y)))

TensorCast.lazy duplicates the behaviour of LazyArrays.lazy before this PR:
https://github.com/JuliaArrays/LazyArrays.jl/pull/31

Eventually it will be OK not to materialize the BroadcastArray before summing:
https://github.com/JuliaLang/julia/pull/31020
=#

if VERSION < v"2.0.0"

    lazy(::Any) = throw(ArgumentError("function `lazy` exists only for its effect on broadcasting"))

    struct LazyCast{T}
        value::T
    end

    Broadcast.broadcasted(::typeof(lazy), x) = LazyCast(x)

    Broadcast.materialize(x::LazyCast) = LazyArrays.BroadcastArray(x.value)

else

    lazy(x) = LazyArrays.lazy(x)

end
