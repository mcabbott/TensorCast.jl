
export @check!, @einsum!, @vielsum!, @tensor!

"""
    @check!(A[i, j, μ, ν])

Adds `A` to the store of known tensors, and records that it expects indices `i,j,μ,ν`.
If it is already in the store, then instead this checks whether the present indices differ
from the saved ones. Only the first letter is examined: `α` and `α2` are similar, as are nearby
letters `β`, `γ3`. More complicated indices like `Z[(i,j), -k, _, 3]` will be ignored.
This happens while parsing your source code, there is zero run-time penalty. Returns `A`.

In addition, with `size=true` option, it can insert size checks to be performed at run-time,
by returning `check!(A, stuff)`.
At the first occurrance this saves `i: => size(A,1)` etc., and on subsequent uses of
the same index (even on different tensors) it gives an error if the sizes do not match.
Here the whole index is used: `α`, `β` and `β2` may have different ranges.
This will need to look up indices in a dictionary, which takes ... 50ns, really?


    @check! B[i,j] C[μ,ν] D[i] E[j]

Checks several tensors, returns nothing.


    @check!  alpha=true  tol=3  size=false  throw=false  info  empty

Controls options for `@check!` and related macros (`@shape!`, `@reduce!`, `@einsum!` etc).
These are the default settings:
* `alpha=true` turns on the parse-time checking, based on index letters.
* `tol=3` sets how close together letters must be: `B[j,k]` is not an error but `B[a,b]` will be.
* `size=false` turns off run-time size checking.
* `throw=false` means that errors are given using `@error`, without interrupting your program.
* `empty` deletes all saved letters and sizes -- there is one global store for each, for now.
* `info` prints what's currently saved.


    @cast! B[i,j] := D[i] * E[j]
    @reduce! B[i,j] := sum(μ,ν) A[i,j,μ,ν] / C[μ,ν]

    @einsum!  B[i,j] := D[i] * E[j]
    @vielsum! B[i,j] := D[i] * E[j]
    @tensor!  B[i,j] := D[i] * E[j]

Versions of the macros from this package, and from Einsum.jl and TensorOperations.jl,
which call `@check!` on each of their tensors, before proceeding as normal.
"""
macro check!(exs...)
    where = (mod=__module__, src=__source__, str=unparse("@check!", exs...))
    _check!(exs...; where=where)
end

const index_store = Dict{Symbol, Tuple}()

const size_store  = Dict{Symbol, Int}() # TODO: alter check!() to avoid dictionary lookup, the macro
                                        # can push nothing to size_store::Vector, and index...

mutable struct CheckOptions
    alpha::Bool
    tol::Int
    size::Bool
    throw::Bool
end

const check_options = CheckOptions(true, 3, false, false)

function _check!(exs...; where=nothing)
    for ex in exs
        if @capture(ex, A_[vec__])
            if length(exs)==1
                return esc(check_one(ex, where))
            else
                check_one(ex, where)
            end

        elseif @capture(ex, alpha=val_Bool)
            check_options.alpha = val
        elseif @capture(ex, tol=val_Int)
            check_options.tol = val

        elseif @capture(ex, size=val_Bool)
            check_options.size = val

        elseif @capture(ex, throw=val_Bool)
            check_options.throw = val

        elseif ex == :empty
            empty!(index_store)
            empty!(size_store)
            @info "@check! stores emptied"

        elseif ex == :info
            @info "@check! info" check_options index_store size_store

        else
            @error "@check! doesn't know what to do with $ex"

        end
    end
    return nothing
end

"""
    check_one(A[i,j,k], (mod=Module, src=...))

Does the work of `@check!`, on one index expression,
returning `A` or `check!(A,...)` according to global flags.
"""
function check_one(ex, where=nothing)
    @capture(ex, A_[vec__]) || error("check_one can't understand $ex, expected something like A[i,j]")
    ind = Tuple(vec)

    if !isa(A, Symbol) # ignore things like f(x)[i,j]
        return A
    end

    if check_options.alpha
        got = get!(index_store, A, ind)

        if length(ind) > length(got)
            check_err("@check! $ex now has more indices than previous $got", where)
        elseif length(ind) < length(got)
            check_err("@check! $ex now has fewer indices than previous $got", where)
        else
            for (i, j) in zip(ind, got)
                if !isa(i, Symbol) || i==(:_) # ignore Int, (i,j), etc.
                    continue
                end
                si = String(i)
                sj = String(j)
                length(si)>1 || length(sj)>1 && continue
                if abs(Int(si[1])-Int(sj[1])) > check_options.tol
                    check_err("@check! $ex now has index $i where previously it had $j", where)
                end
            end
        end
    end

    if check_options.size
        Astring = string(A,"[",join(ind," ,"),"]")
        return :( TensorCast.check!($A, $ind, $Astring, $where) )
    else
        return A
    end
end

function check_err(str::String, where=nothing)
    if check_options.throw
        error(str)
    elseif where==nothing
        @error str
    else
        @error str  _module=where.mod  _line=where.src.line  _file=string(where.src.file)
    end
end

"""
    check!(A, (:i,:j), "A[i,j]", (mod=..., src=...))

Performs run-time size checking, on behalf of the `@check!` macro, returns `A`.
The string and tuple are just for the error message.
"""
function check!(A::AbstractArray{T,N}, ind::Tuple, str::String, where=nothing) where {T,N}
    if N != length(ind)
        check_err("check! expected $str, but got ndims = $N", where)
    else
        for (d,i) in enumerate(ind)
            if !isa(i, Symbol) || i==(:_) # ignore Int, (i,j), etc.
                continue
            end
            sizeAd = size(A,d)
            got = get!(size_store, i, sizeAd)
            got == sizeAd || check_err("check! $str, index $i now has range $sizeAd instead of $got", where)
        end
    end
    A
end

"""
    @einsum! A[i,j] := B[i,k] * C[k,j]

Variant of `@einsum` from package Einsum.jl,
equivalent to wrapping every tensor with `@check!()`.
"""
macro einsum!(ex)
    where = (mod=__module__, src=__source__, str=unparse("@einsum!", exs...))
    _einsum!(ex, where)
end

"""
    @vielsum! A[i,j] := B[i,k] * C[k,j]

Variant of `@vielsum` from package Einsum.jl,
equivalent to wrapping every tensor with `@check!()`.
"""
macro vielsum!(ex)
    where = (mod=__module__, src=__source__, str=unparse("@vielsum!", exs...))
    _einsum!(ex, where; threads=true)
end

"""
    @tensor! A[i,j] := B[i,k] * C[k,j]

Variant of `@tensor` from package TensorOperations.jl,
equivalent to wrapping every tensor with `@check!()`.
"""
macro tensor!(ex)
    where = (mod=__module__, src=__source__, str=unparse("@tensor!", exs...))
    _tensor!(ex, where)
end

function _tensor!(ex, where=nothing)
    if @capture(ex, lhs_ := rhs_ ) || @capture(ex, lhs_ = rhs_ )

        outex = quote end
        function f(x)
            if @capture(x, A_[ijk__] )
                push!(outex.args, check_one(x, where))
            end
            x
        end
        MacroTools.prewalk(f, rhs)

        if check_options.size == false
            check_one(lhs) # then these are only parse checks, outex is trash
        else
            push!(outex.args, :(out = TensorOperations.@tensor $ex) )
            push!(outex.args, check_one(lhs, where)) # lhs size may not be known until after @tensor
            push!(outex.args, :out )                 # make sure we still return what @tensor did
            return esc(outex)
        end
    else
        @warn "@tensor! not smart enough to process $ex yet, so ignoring checks"
    end
    return esc(:( TensorOperations.@tensor $ex ))
end

function _einsum!(ex, where=nothing; threads = false)
    if @capture(ex, lhs_ := rhs_ ) || @capture(ex, lhs_ = rhs_ )

        outex = quote end
        function f(x)
            if @capture(x, A_[ijk__] )
                push!(outex.args, check_one(x, where))
            end
            x
        end
        MacroTools.prewalk(f, rhs)

        if check_options.size == false
            check_one(lhs) # then these are only parse checks, outex is trash
        else
            if threads
                push!(outex.args, :(out = Einsum.@vielsum $ex) )
            else
                push!(outex.args, :(out = Einsum.@einsum $ex) )
            end
            push!(outex.args, check_one(lhs, where)) # lhs size may not be known until after @einsum
            push!(outex.args, :out )                 # make sure we still return what @einsum did
            return esc(outex)
        end
    else
        @warn "@einsum! not smart enough to process $ex yet, so ignoring checks"
    end
    if threads
        return esc(:( Einsum.@vielsum $ex ))
    else
        return esc(:( Einsum.@einsum $ex ))
    end
end

#==

using MacroTools, Einsum, TensorOperations
B = rand(2,3); C = rand(3,2);
A = B * C
@einsum A[i,j] := B[i,k] * C[k,j]
@tensor A[i,j] := B[i,k] * C[k,j]


using TensorCast


@check! size=true throw=false info

@einsum! A[i,j] := B[i,k] * C[k,j]

@tensor! A[i,j] := B[i,k] * C[k,j]

@check! info   # has ABC and ijk

@check! A[z,j] # compains about z
@check! B[i]   # complains about number

B5 = rand(2,5); C5 = rand(5,2);
@einsum! A[i,j] := B5[i,k] * C5[k,j] # complains about sizes
@tensor! A[i,j] := B5[i,k] * C5[k,j]

@einsum! A[i,j] := B5[i,k] * C5[k,zz] # complains about zz
@tensor! A[i,j] := B5[i,k] * C5[k,zz] # and errors


==#

