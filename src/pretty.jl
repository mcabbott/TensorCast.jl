
export @pretty

"""
    @pretty @shape A[...] := B[...]

Prints an approximately equivalent expression with the macro expanded.
Compared to `@macroexpand1`, generated symbols are replaced with animal names,
comments are deleted, module names are removed from functions,
and the final expression is fed to `println()`.
"""
macro pretty(ex)
    :( macroexpand($(__module__), $(ex,)[1], recursive=false) |> MacroTools.alias_gensyms |> pretty |> println )
end

function pretty(ex::Union{Expr,Symbol})
    str = string(ex)

    str = replace(str, r"(#=.+=#)\s" => "")  # remove line references
    str = replace(str, r"\n(\s+\n)" => "\n") # and empty lines
    
    str = replace(str, r"\w+\.(\w+)\(" => s"\1(") # remove module names on functions
    str = replace(str, r"\w+\.(@\w+)" => s"\1")   # and on macros
    str = replace(str, r"\w+\.(\w+){" => s"\1{")  # and on structs...

    str = replace(str, "Colon()" => ":")
end

pretty(tup::Tuple) = replace(string(tup), "Colon()" => ":")
