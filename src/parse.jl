
const flag_list = Any[ :!, :assert, :cat, :glue, :strided, :lazy, :julienne]

#= TODO

* make sz a constant gensym()
* multiply with ̂* or something, which understands :

=#

struct SizeDict
    dict::Dict{Any, Any}
    checks::Vector{Any}
    seen::Vector{Any}
end

SizeDict() = SizeDict(Dict{Any, Any}(), Any[], Any[])

"""
    flat, getafix, putsz, negated = parse!(sdict, A, outer, inner)

Use this for `A[outer...][inner...]`. 
* `flat` is a list of symbols, naming the indices, in order. 
* `negated` is the subset which had - in front. 
* `getafix` is the thing for making views of given input or target array, : in non-fixed directions. 
* `putsz` is a tuple of `sz[n]` and products, which you want for reshaping to the very final array; 
  the numbers `n` refer to position in `flat`. 
* `sdict::SizeDict` ends up with `sdict.dict[:i] = ` an expr for length of this index.
  To avoid saving these (if you don't have A yet) set `A = nothing`.

    parse!(sdict, A, outer, reduce_inner, true)

This is for `A[outer...] = sum(inner...) ...` LHS, which are allowed to have sum(a:2, ...) ranges. 
The allowranges flag *also* disables the use of `size(first(A), 2)` etc,
to disable `size(A,2)` set `A = nothing` again. 

    parse!(sdict, nothing, [], rex, true, flag_vector)

For dimensions & annotations. 
Will look for flags because it was given something to push them into. 
You should now do this first, so that `sdict` is most likely to have these (neater) entries. 
They are added using savesize!(sdict,...) which now puts later sizes into list of checks. 
"""
function parse!(sdict::SizeDict, A, outer, inner, allowranges=false, flags=nothing)

    flat = Any[]
    getafix = Any[]
    putsz = Any[]
    negated = Any[]

    ### INNER dimensions come first in flat
    for (din, ind) in enumerate(inner)

        # look for flags only in rex case
        if flags != nothing && ind in flag_list # global const
            push!(flags, ind)
            continue
        end

        # @info "parse! inner" A repr(inner) allowranges din ind 

        # in rex or sum(a, b:3, c), some indices may have explicit ranges
        if allowranges 

            if @capture(ind , i_:ilength_ )
                savesize!(sdict, i, ilength) 
            else
                i = ind    # not nothing from @capture
            end
            push!(flat, i) # all need to go into flat!

        # for simple slicing, always want iflat, but size(first(A),d) only if we know A. 
        else
            i = stripminus!(negated, ind) # now i isa Symbol
            push!(flat, i)

            if !isnothing(A) # only save these if we can access A. 
                savesize!(sdict, i, :( size(first($A), $din) ) ) 
            end
        end
    end

    ### OUTER dimensions never allow flags or ranges,
    # but do allow tuples etc.
    for (dout, ind) in enumerate(outer)
        if isconstant(ind)              # then it's a constant slice,
            if ind == :_                # treat _ almost like 1, but possibly with a check
                if !isnothing(A)
                    str = "direction marked _ must have size 1"
                    push!(sdict.checks, :(TensorCast.@assertsize size($A, $dout) == 1 $str) )
                end
                ind = 1
            end
            if ind isa Expr             # constant slice with $c
                @assert ind.head == :($)
                ind = ind.args[1]
            end
            push!(getafix, ind)         # so when we view(A, getafix...) we need it
            push!(putsz, 1)             # or if driving the other direction, make size=1.
        else
            push!(getafix, Colon() )           # and we want this on the list:
            ind = stripminus!(negated, ind) # this looks inside for -, and returns Symbol or Vector{Symbol}
            if !isnothing(A)
                savesize!(sdict, ind, :( size($A, $dout) ) )
            end

            d = length(flat) + 1 # index in flat of ind, or ind[1]. 
            push!(putsz, szwrap(ind, d)) 

            push_or_append!(flat, ind) 
        end
    end

    return flat, getafix, putsz, negated
end

isconstant(i::Int) = true
isconstant(i::Symbol) = i == :_ 
isconstant(ex::Expr) = ex.head == :($)

function findcheck(i::Symbol, ind::Vector)
    d = findfirst(isequal(i), ind)
    d isa Nothing && throw(ArgumentError("can't find index $i where it should be!"))
    return d
end
findcheck(i::Expr, ind::Vector) = throw(ArgumentError("don't know what to do with index $i, expected a single symbol"))

function savesize!(store::SizeDict, ind, long)
    if !haskey(store.dict, ind) # then we don't yet have a size for this index (or tuple)
        store.dict[ind] = long
    else                        # we have seen it before,
        if isa(ind, Symbol)     # but list of checks is only for actual indices
            str = "range of index $ind must agree"
            push!(store.checks, :(TensorCast.@assertsize $(store.dict[ind]) == $long $str) )
        end
    end
end

push_or_append!(list, ind::Symbol) = push!(list, ind)
push_or_append!(list, inds::Vector) = append!(list, inds)

"""
    stripminus(negated, i)

For any index, or vector of indices, or vector of indices containing tuples / backslash combos, 
this pushes every letter with a minus in front into negated list.

It returns an index, or a tidied-up vector of indices, in which tuples etc have become vectors.   
"""
stripminus!(negated::Vector, ind::Symbol) = ind  # get one Symol & you're done

function stripminus!(negated::Vector, ind::Expr)
    # minus one index
    if @capture(ind, -i_Symbol)
        push!(negated, i)       # add that to the negated list
        return i                # and go home. 

    # tuple notation
    elseif @capture(ind, -(ijk__,) )
        ijkplus = stripminus!(negated, ijk)
        append!(negated, ijkplus)

    elseif @capture(ind, (ijk__,) )

    # backslash notation -- if you want more than four \ then I'm impressed! TODO make recursive
    elseif @capture(ind, i_\j_\k_\l_\m )
        ijk = Any[i,j,k,l,m]
    elseif @capture(ind, i_\j_\k_\l_ )
        ijk = Any[i,j,k,l]
    elseif @capture(ind, i_\j_\k_ )
        ijk = Any[i,j,k]
    elseif @capture(ind, i_\j_ )
        ijk = Any[i,j]

    elseif @capture(ind, i_⊗j_⊗k_⊗l_⊗m )
        ijk = Any[i,j,k,l,m]
    elseif @capture(ind, i_⊗j_⊗k_⊗l_ )
        ijk = Any[i,j,k,l]
    elseif @capture(ind, i_⊗j_⊗k_ )
        ijk = Any[i,j,k]
    elseif @capture(ind, i_⊗j_ )
        ijk = Any[i,j]

    else 
        error("stripminus! is stuck on $ind")
    end

    # now we have a vector ijk of symbols, or expressions :(-i), no more
    return Any[ stripminus!(negated, i) for i in ijk ]
end

szwrap(i::Symbol, d::Int) = :( sz[$d] )
szwrap(ijk::Vector, d::Int) = :( *($([ :(sz[$n]) for n=d:(d+length(ijk)-1) ]...)) )

"""
    oddunique(negated)
Returns a list in which anything repeated evenly many times has been removed, then `unique`. 
"""
function oddunique(negated)  # -1 * -1 = +1
    set = Set{Symbol}()
    for i in negated
        if i in set
            pop!(set, i)
        else
            push!(set, i)
        end
    end
    Any[ i for i in set ]
end

"""
    checkrepeats(flat)
Throws an error if there are repeated indices.
"""
function checkrepeats(flat, msg="")
    once = Set{Symbol}()
    twice = Set{Symbol}()
    for i in flat
        if i != nothing
            if i in once
                push!(twice, i)
            else
                push!(once, i)
            end
        end
    end
    if length(twice) > 0
        str = join(twice, ", ")
        error("repeated index/indices $str" * msg)
    end
    nothing
end


"""
    sizeinfer(store, icannon, leaveone=true)

This is the point of SizeDict. 
The goal is to produce a canonical vector of sizes, corresponding to vector of symbols icannon. 
If these sizes are known, easy!

But for unknown ones, we do a second pass, looking for entries in sizedict like [:i, :j]
which come from tuples of indices, for which we know the product of their dimensions. 
"""
function sizeinfer(store::SizeDict, icanon::Vector, leaveone = true)
    sizes = Any[ (:) for i in icanon ]

    # first pass looks for single indices whose lengths are known directly
    for pair in store.dict
        if isa(pair.first, Symbol)  
            d = findcheck(pair.first, icanon)
            sizes[d] = pair.second  # so write it into the answer
        end
    end

    if leaveone && count(isequal(:), sizes) < 2 
        V && count(isequal(:), sizes)==1 &&  @info "sizeinfer -- leaving one : in sizes" repr(sizes)
        return sizes 
    end
    V && @info "sizeinfer -- first pass" sizes

    # second pass looks for tuples (whose product-length is known) where exactly one entry has unknown length
    for pair in store.dict
        if isa(pair.first, Vector)
            known = haskey.(Ref(store.dict), pair.first)

            if sum(.!known) == 1    # bingo! now work out its size:
                num = pair.second
                denfacts = [ store.dict[i] for i in pair.first[known] ]
                if length(denfacts) > 1
                    den = :( *($(denfacts...)) )
                else
                    den = :( $(denfacts[1]) )
                end
                rat = :( $num ÷ $den )

                d = findcheck(first(pair.first[.!known]), icanon)
                sizes[d] = rat      # save that size

                str = "inferring range of $(icanon[d]) from range of $(join(pair.first, " ⊗ "))" 
                push!(store.checks, :( TensorCast.@assertsize rem($num, $den)==0 $str) )
            end
        end
    end
    V && @info "sizeinfer -- second pass" sizes

    unknown = Any[ i for (d,i) in enumerate(icanon) if sizes[d] == (:) ]
    str = join(unknown, ", ")
    length(unknown) <= 1 || throw(ArgumentError("unable to infer ranges for indices $str"))

    return sizes
end

sorted_starcode(tot, inner) = (ntuple(i -> :, inner)..., ntuple(i -> *, tot-inner)...)

"""
    colonise!(isz, slist)

This aims to simplify "isz" which is going to be used `reshape(A, isz)`. 
Partly for cosmetic reasons... some of which are now caught elsewhere. 

But partly because `slist` may contain one `:`, and `isz` may try to multiply that by `sz[d]`, 
correct answer is `:` again. 
"""
function colonise!(isz, slist::Vector) 
    szall = Any[:(sz[$d]) for d=1:length(slist)]

    if isz == szall
        # example from tests:  @pretty @shape g[(b,c),x,y,e] := bcde[b,c,(x,y),e] x:2;
        V && println("colonise! replaced ",isz, 
            " with just sz... because that's what it is")
        resize!(isz, 1)
        isz[1] = :(sz...)
        return true
    end

    d = findfirst(isequal(:), slist) # slist may have one colon, or none

    function f!(nsz_ref, x)
        if x == :(sz[$d])
            nsz_ref[1] = 9999
        elseif @capture(x, sz[other_])
            nsz_ref[1] += 1
        end
        x
    end

    N = length(slist)
    needsizes = true

    for (n,sprod) in enumerate(isz)
        nsz = [0]
        MacroTools.postwalk(x -> f!(nsz, x), sprod);
        if nsz[1] > N
            isz[n] = Colon()
            V && println("colonise! replaced isz/osz/fsz[$n]= ",sprod, 
                " with just : because really sz[$d]=:")
        elseif nsz[1] == N
            isz[n] = Colon()
            needsizes = false
            V && println("colonise! replaced isz/osz/fsz[$n]= ",sprod, 
                " with just : because it containts all N=$N of the sz[d]")
        end
    end

    return needsizes
end

"""
    p, (bef, aft) = simplicode(p, (bef, aft))

The idea here is to absorb permutedims into alterations of the slicing / gluing codes. 
Done as an optimisation pass, after figuring all of those out, before building expression. 

* If we also have reversal or shift, those dimensions will need to be updated.
  Maybe then don't bother. 

* If there is a reduction, then it's totally worth bothering! You have more freedom than with slicing.

* It will also tend to break agreement between canonical sizes & size of actual array at any stage. 
  This may matter for in-place case. 

For now, only apply this when those are not in use. 
"""
function simplicode(perm, before, after)
    ## just copied here from main file, this function isn't called yet 
    ## and certainly won't work!! 

    # TODO Think more about this -- it could also tidy up @pretty @shape A[(i,j)] := B[i][j]
    # but not for static glue
    if willpermute && willslice # hence not willstaticslice
        if perm==(2,1)
            ocode = (ocode[2],ocode[1])
            willpermute = false
            W && println("removed permutation by changing slicing: now ocode = ",ocode)
            # W && println("... for expr = ",expr)
        else
            V && println("am going to slice a permuted array. Can I remove permutation by re-ordering slice?")
        end
    end
    # if willpermute && willglue # hence not willstaticglue
    #     if perm==(2,1)
    #         icode = (icode[2],icode[1])
    #         willpermute = false
    #         W && println("removed permutation by changing gluing: now icode = ",icode)
    #         W && println("... for expr = ",expr)
    #     else
    #         V && println("am going to glue a permuted array. Can I remove permutation by re-ordering ?")
    #     end
    # end

end

"""
    @assertsize cond str

Like `@assert`, but prints both the given string and the condition. 
"""
macro assertsize(ex, str)
    strex = Main.Base.string(ex)
    msg = str * ": " * strex
    # return :($(esc(ex)) ? $(nothing) : throw(DimensionMismatch($msg)))
    # return :( $(esc(ex)) ? $(nothing) : error($msg) )
    return esc(:($ex || error($msg)))
end


