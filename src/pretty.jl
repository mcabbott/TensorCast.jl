
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

# I made this recursive=false like @macroexpand1 so that @assert won't be expanded. This works:
# @assert $(chex) "@shape failed explicit size checks"
# but is ugly because @assert's argument is printed as quoted,  @assert 2 == size($(Expr(:escape, :B)), 1).

function pretty(ex::Union{Expr,Symbol})
    # ex = prettify(ex) # gets most of the line number comments ... but is messing up Colon?
    str = string(ex)

    str = replace(str, r"\(\w+\.(\w+)\)" => s"\1") # remove module names on functions
    str = replace(str, r"\(\w+\.(\w+!)\)" => s"\1")

    str = replace(str, r"(,)\s(\d)" => s"\1\2") # un-space (1,2,3,4) things 

    str = replace(str, "Colon()" => ":")

    # str = replace(str, r"(#=.+=#\s)" => "") # @assert statements were missed? 

    str = replace(str, r"\n(\s+.+=#)" => "") # remove line references
end

pretty(tup::Tuple) = replace(string(tup), "Colon()" => ":")

# TODO teach @pretty to un-escape things?
