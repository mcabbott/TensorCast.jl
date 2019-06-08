
"""
    @pretty @cast A[...] := B[...]

Prints an approximately equivalent expression with the macro expanded.
Compared to `@macroexpand1`, generated symbols are replaced with animal names (from MacroTools),
comments are deleted, module names are removed from functions,
and the final expression is fed to `println()`.

To copy and run the printed expression, you may need various functions which aren't exported.
Try something like `using TensorCast: orient, star, rview, @assert_, red_glue, sliceview`
"""
macro pretty(ex)
    if @capture(ex, @cast_str str_)
        full = "@cast " * cast_string(str)
        println("# " * full)
        return :(@pretty $(Meta.parse(full)) )

    elseif @capture(ex, @reduce_str str_)
        full = reduce_string(str)
        println("# " * full)
        return :(@pretty $(Meta.parse(full)) )

    else
        :( macroexpand($(__module__), $(ex,)[1], recursive=false) |> pretty |> println )
    end
end

function pretty(ex::Union{Expr,Symbol})
    ex = MacroTools.alias_gensyms(ex) # animal names
    ex = MacroTools.striplines(ex)    # remove line references
    pretty(string(ex))
end

function pretty(str::String)
    str = replace(str, r"\n(\s+\n)" => "\n")      # remove empty lines

    str = replace(str, r"\w+\.(\w+)\(" => s"\1(") # remove module names on functions
    str = replace(str, r"\w+\.(\w+)!\(" => s"\1!(")
    str = replace(str, r"\w+\.(@\w+)" => s"\1")   # and on macros
    str = replace(str, r"\w+\.(\w+){" => s"\1{")  # and on structs...

    str = replace(str, "Colon()" => ":")
    str = replace(str, r"\$\(QuoteNode\((\d+)\)\)" => s":\1")  # not right yet!
end

pretty(tup::Tuple) = pretty(string(tup))

pretty(vec::Vector) = pretty(join(something.(vec, "nothing"), ", "))

pretty(x) = string(x)

function unparse(str::String, exs...)
    @capture(exs[1], left_ = right_ ) && return string(str, " ", left, " = ", right, "  ", join(exs[2:end],"  "))
    @capture(exs[1], left_ := right_ ) && return string(str, " ", left, " := ", right, "  ", join(exs[2:end],"  "))
    @capture(exs[1], left_ |= right_ ) && return string(str, " ", left, " |= ", right, "  ", join(exs[2:end],"  "))

    @capture(exs[1], left_ += right_ ) && return string(str, " ", left, " += ", right, "  ", join(exs[2:end],"  "))
    @capture(exs[1], left_ -= right_ ) && return string(str, " ", left, " -= ", right, "  ", join(exs[2:end],"  "))
    @capture(exs[1], left_ *= right_ ) && return string(str, " ", left, " *= ", right, "  ", join(exs[2:end],"  "))

    return string(exs)
end
