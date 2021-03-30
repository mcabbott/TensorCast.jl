
struct DotDict
    store::Dict{Symbol,Any}
end
DotDict(; @nospecialize kw...) = DotDict(Dict(pairs(kw)...))

Base.parent(x::DotDict) = getfield(x, :store)

Base.propertynames(x::DotDict) = Tuple(sort(collect(keys(parent(x)))))
Base.getproperty(x::DotDict, s::Symbol) = getindex(parent(x), s)
function Base.setproperty!(x::DotDict, s::Symbol, @nospecialize v)
    s in propertynames(x) || throw("DotDict has no field $s")
    T = typeof(getproperty(x, s))
    if T == Nothing
        setindex!(parent(x), v, s)
    else
        setindex!(parent(x), convert(T, v), s)
    end
end

Base.iterate(x::DotDict, r...) = iterate(parent(x), r...)

function Base.show(io::IO, x::DotDict)
    print(io, "DotDict(")
    strs = map(k -> string(k, " = ", getproperty(x, k)), propertynames(x))
    print(io, join(strs, ", "), ")")
end
