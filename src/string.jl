
# Define string macro versions, which take A_ij notation?

export @cast_str, @reduce_str

macro cast_str(str::String)
    Meta.parse("@cast " * cast_string(str))
end
macro tensor_str(str::String)
    Meta.parse("@tensor " * cast_string(str))
end
macro einsum_str(str::String)
    Meta.parse("@einsum " * cast_string(str))
end

macro reduce_str(str::String)
    Meta.parse(reduce_string(str))
end

function cast_string(str)
    replace(str, r"_([\w\d\$\'\′]+)" => indexsquare)
end

function reduce_string(str::String)
    str2 = replace(str, r"=\s+\w+_[\w\'\′]+" => indexfun)
    str3 = replace(str2, r"_([\w\d\$\'\′]+)" => indexsquare)
    "@reduce " * str3
end

function indexcomma(str)
    list = []
    for c in collect(str)
        # @show c, Int(c)
        if (c=='\'') | (c=='′')
            # @show list
            list[end] *= "′"
            # @show list
        elseif length(list)>0 && list[end] =="\$"
            list[end] = '\$' * c
        else
            push!(list, string(c))
        end
    end
    join(list, ',')
end

function indexsquare(str)
    @assert str[1] == '_'
    "[" * indexcomma(str[2:end]) * "]"
end

#=
indexsquare("_âb′c")
indexsquare("_a\$b1")
indexsquare("_ijk''") # wtf?
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
indexfun("= Π_ii'") # wtf?
=#

