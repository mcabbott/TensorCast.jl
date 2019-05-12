export @cast_str, @reduce_str, @mul_str

"""
    cast" Z_ij := A_i + B_j "

String macro version of `@cast`, which translates things like `"A_ijk" == A[i,j,k]`.
Indices should be single letters, except for primes and constants, for example:
```
julia> @pretty cast" X_αβ' = A_α33β' * log(B_β'\$k)"
# @cast  X[α,β′] = A[α,33,β′] * log(B[β′,\$k])
begin
    local kangaroo = view(A, :, 33, :)
    local turtle = transpose(view(B, :, k))
    X .= @. kangaroo * log(turtle)
end
```
Operators `:=` and `=` work as usual, as do options.
There are similar macros `reduce"Z_i = Σ_j A_ij"` and `mul"Z_ik = Σ_j A_ij * B_jk"`.
"""
macro cast_str(str::String)
    Meta.parse("@cast " * cast_string(str)) |> esc
end
macro tensor_str(str::String)
    Meta.parse("@tensor " * cast_string(str)) |> esc
end
macro einsum_str(str::String)
    Meta.parse("@einsum " * cast_string(str)) |> esc
end

"""
    reduce" Z_i := sum_j A_ij + B_j "

String macro version of `@reduce`.
Indices should be single letters, except for primes, and constants.
You may write \\Sigma `Σ` for `sum` (and \\Pi `Π` for `prod`):
```
julia> @pretty reduce" W_i = Σ_i' A_ii' / B_i'3^2  lazy"
# @reduce  W[i] = sum(i′) A[i,i′] / B[i′,3]^2  lazy
begin
    local mallard = orient(view(B, :, 3), (*, :))
    sum!(W, @__dot__(lazy(A / mallard ^ 2)))
end
```
"""
macro reduce_str(str::String)
    Meta.parse(reduce_string(str)) |> esc
end

macro mul_str(str::String)
    Meta.parse(mul_string(str)) |> esc
end

function cast_string(str)
    replace(str, r"_([\w\d\$\'\′]+)" => indexsquare)
end

function reduce_string(str::String)
    str2 = replace(str, r"=\s+\w+_[\w\'\′]+" => indexfun)
    str3 = replace(str2, r"_([\w\d\$\'\′]+)" => indexsquare)
    "@reduce " * str3
end

function mul_string(str::String)
    str2 = replace(str, r"=\s+sum_[\w\'\′]+" => indexfun)
    str3 = replace(str2, r"=\s+Σ_[\w\'\′]+" => indexfun)
    str4 = replace(str3, r"_([\w\d\$\'\′]+)" => indexsquare)
    "@reduce " * str4
end

function indexcomma(str)
    list = []
    for c in collect(str)
        if (c=='\'') || (c=='′') && length(list)>0
            list[end] *= "′"
        elseif isnumber(c) && length(list)>0 && isnumber(list[end])
            list[end] *= c
        elseif length(list)>0 && list[end] =="\$"
            list[end] = '\$' * c
        else
            push!(list, string(c))
        end
    end
    join(list, ',')
end

isnumber(c::Char) = '0' <= c <= '9'
isnumber(s::String) = all(isnumber, collect(s))

function indexsquare(str)
    @assert str[1] == '_'
    "[" * indexcomma(str[2:end]) * "]"
end

#=
indexsquare("_âb′c")
indexsquare("_a\$b1")
indexsquare("_ijk''") # failed locally
indexsquare("_ij'k''")
=#

function indexfun(str)
    @assert str[1] == '='

    start, ind = split(str, '_')
    fun = split(start, ('=', ' '))[end]

    fun == "Σ" ? fun = "sum" :
    fun == "Π" ? fun = "prod" : nothing

    "= " * fun * "(" * indexcomma(ind) * ")"
end

#=
indexfun("= sum_i")
indexfun("= Π_ii'") # failed locally
=#

