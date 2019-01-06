module TensorSlice

export @shape, @reduce, @pretty

using MacroTools

include("parse.jl")

V = false # capital V for extremely verbose
W = false # printouts where I'm working on things

"""
    @shape A[i...] := B[j...] opt

Tensor reshaping and slicing macro. Undestands the following things:
* `A[i,j,k]` is a three-tensor with these indices
* `B[(i,j),k]` is the same thing, reshaped to a matrix. This may also be written `B[i\\j,k]`
* `C[k][i,j]` is a vector of matrices
* `D[j,k]{i}` is an ordinary matrix of `SVector`s, reinterpreted from `A`
* `E[i,1,k]` or `E[i,_,k]` has two nontrivial dimensions, and `size(E,2)==1`.

They can be related in a few ways:
* `=` writes into an existing object
* `:=` creates a new object... which may or may not be a view:
* `==` insists on a view of the old object (error if impossible)
* `|=` insists on a copy. 

Options can be specified at the end (if several, separated by `,` i.e. `opt::Tuple`)
* `i:3` fixes the range of index `i`
* `assert` or `!` force explicit dimension checks
* `base` or `_` restricts to methods from plain Julia.

The left and right sides must have all the same indices. 
See `@reduce` for a related macro which can sum over things. 

Static slices `D[j,k]{i}` need `using StaticArrays`, and to create them you must give all 
slice dimensions explicitly. 
If `using Strided` then we have a lazy `permutedims`, allowing more cases to be `==` views.
These are disabled by the Base option `_`.
If `using JuliennedArrays` then slicing and gluing will be done by this package.
"""
macro shape(expr, rex=nothing)

    if @capture(expr, left_ = right_ )
        sign = :(=)
    elseif @capture(expr, left_ := right_ )
        sign = :(:=)
    elseif @capture(expr, left_ == right_ )
        sign = :(==)
    elseif @capture(expr, left_ |= right_ )
        sign = :(|=)
    else
        throw(ArgumentError("@shape can't begin to understand $expr"))
    end

    V && @info "@shape" sign left right rex

    #==================== LEFT = OUTPUT = STEPS 6,7 ====================#
    # Do this first to get a list of indices in canonical order, oflat
    # And parse expr here not in tensor_slice_main as LHS differs for @reduce

    willslice = false
    willstaticslice = false

    if @capture(left, ( outn_[oind__][oi__] | [oind__][oi__] ) )
        willslice = true
    elseif @capture(left, ( outn_[oind__]{oi__} | [oind__]{oi__} ) )
        willstaticslice = true
    elseif @capture(left, ( outn_[oind__] | [oind__] ) )
        oi = Any[]
    else
        throw(ArgumentError("@shape can't understand left hand side $left"))
    end

    V && @info "@capture LHS" outn repr(oind) repr(oi) willslice willstaticslice

    tensor_slice_main(outn, oind, oi, # parsing of LHS in progress
        willslice, willstaticslice,   # flags from LHS of @shape
        false, nothing,               # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        "@shape")
end

"""
    @reduce A[j] := sum(i,k) B[i,j,k]

Tensor reduction macro:
* The reduction funcition can be anything which works like `sum(B, dims=(1,3))`, 
  for instance `prod` and `maximum` and `Statistics.mean`. 
* The right side can be anything that `@shape` understands, including gluing of slices `B[i,k][j]` 
  and reshaping `B[i\\j,k]`.
* The left side again works as in `@shape`, although slicing is not allowed.
* In-place operations `A[j] = sum(...` will construct the banged version of the given function's name, 
  which must work like `sum!(B, A)`.
* Index ranges may be given afterwards (as for `@shape`) or inside the reduction `sum(i:3, k:4)`. 
* All indices appearing on the right must appear either within `sum(...)` etc, or on the left. 
"""
macro reduce(expr, right, rex=nothing)

    if @capture(expr, left_ = red_ )
        sign = :(=)
    elseif @capture(expr, left_ := red_ )
        sign = :(:=)
    elseif @capture(expr, left_ |= right_ )
        sign = :(|=)
    else
        throw(ArgumentError("@reduce can't begin to understand $expr"))
    end

    V && @info "@reduce" sign left right rex

    #==================== LEFT = OUTPUT = STEPS 6,7 ====================#
    # Do this first to get a list of indices in canonical order, oflat
    # And parse expr here not in tensor_slice_main as LHS differs for @shape

    if @capture(left, ( outn_[oind__] | [oind__] ) )
    else
        throw(ArgumentError("@reduce can't understand left hand side $left"))
    end

    if @capture(red, redfun_(oi__) )
        if sign == :(=) # inplace=true, later
            if !endswith(string(redfun), '!')
                redfun = Symbol(redfun, '!')
            end
        end
    else
        throw(ArgumentError("@reduce can't understand reduction formula $red"))
    end

    V && @info "@capture LHS" outn redfun repr(oind) repr(oi)

    tensor_slice_main(outn, oind, oi, # parsing of LHS in progress
        false, false,                 # flags from LHS of @shape
        true, redfun,                 # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        "@reduce")
end

############################### NOTATION ################################
#=

After getting hopelessly confused, I decided to aphabet-ise my notation: 

nameA, indA, indAsub   -- given RHS            -- inn, iind, ii
sizeA      -- known & used for fixing ranges
nameAesc   -- unglified

getB, numB -- for view(A, get), and number of fixed indices
sizeC      -- after reshaping outer indices    -- "isz"
codeD      -- for gluing                       -- icode
indEflat   -- after gluing                     -- iflat
revFdims   -- done before permutedims
shiftG     -- ditto, but not yet written

permM      -- perm(indEflat -> infVflat)
storeAZ    -- information from parse! both     -- sdict

indVflat   -- canonical list                   -- oflat
sizeV      -- filled in... at most one (:)     -- slist

codeW      -- for slicing                      -- ocode
sizeX      -- after                            -- "osz"
getY, numY -- for view(Z, get) in-place, and number fixed

nameZesc
sizeZ      -- for final reshape                -- "fsz"
nameZ, indZ, indZsub   -- given LHS            -- outn, oind, oi

=#
############################### GIANT MONO-FUNCTION ################################

function tensor_slice_main(outn, oind, oi,
        willslice, willstaticslice,   # flags from LHS of @shape
        willreduce, redfun,           # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        macroname)

    ## test what packages are loaded
    usingstrided = isdefined(TensorSlice, :Strided)
    usingstatic =  isdefined(TensorSlice, :StaticArrays)

    willstaticslice && !usingstatic &&
        throw(ArgumentError("@shape can't statically glue slice without using StaticArrays"))

    ## make sign == :(:=) mean simple whatever is easiest:
    mustcopy =    sign == :(|=)
    mustnotcopy = sign == :(==)
    inplace =     sign == :(=)

    V && @info "todo & packages" mustcopy mustnotcopy inplace  usingstatic usingstrided

    #==================== LEFT, CONTINUED ====================#

    if outn == nothing
        hasoutputname = false
        outn = gensym()
        inplace && throw(ArgumentError("$macroname must have an output name for in-place operations. Try with := instead"))
    else
        hasoutputname = true
        outn = esc(outn)
    end

    ## new parsing code! 
    # "o" refers to stage 5, right after permutedims, all indices "flat"
    # "f" refers to step 7, final reshaping to size fsz... or in-place, view fget.
    #     maybe oind is really find, but I can't have that!
    # szdict contains things like size(H,3) which will only be useful if in-place... 
    # willreduce tells parse! whether to allow i:3 annotations on inner indices. 

    sdict = SizeDict()

    oflat, fget, fsz, oneg = parse!(sdict, ifelse(inplace, outn, nothing), oind, oi, willreduce)

    V && @info "parse! LHS -- including canonical oflat" repr(oflat) repr(fget) repr(fsz) repr(oneg) length(sdict.dict) length(sdict.checks)

    # for step 6, slicing:
    ocode = sorted_starcode(length(oflat), length(oi)) # star for each outer dim, at the right 
    # or else step 6', reduce: 
    if length(oi) ==1
        odims = 1  # reduction, like inner indices, is leftmost
    else
        odims = Tuple( 1 : length(oi) )
    end

    # for step 8, only used if working in-place:
    fnum_fixed = count(!isequal(:), fget) # fnum_fixed = count(i -> i != (:), fget)
    willfinalview = fnum_fixed > 0

    if willfinalview && !inplace
        for i in fget
            if i != (:) && i != 1
                throw(ArgumentError("$macroname can't create an array with constant index $i != 1"))
            end
        end
    end

    # for step 7 reshape when working in-place: very simple? 
    # the relevant lenghts are the canonical ones, skipping any inner indices
    osz = Any[:(sz[$d]) for d=length(oi)+1:length(oflat)]

    # for step 7, reshape -> fsz. Only needed if ndims differs:
    willshapeout = (length(fsz) + fnum_fixed) != length(oflat) - length(oi)

    V && @info "left..." willslice ocode willreduce odims repr(osz) fnum_fixed willshapeout repr(fsz)

    #==================== RIGHT = INPUT = STEPS 1,2,3 ====================#

    willglue = false
    willstaticglue = false
    mustcheck = false

    if @capture(right, inn_[iind__][ii__] )
        willglue = true
        mustnotcopy && throw(ArgumentError("$macroname can't create a view of glued-together slices, unless they are StaticArrays"))
    elseif @capture(right, inn_[iind__]{ii__} )
        willstaticglue = true
        usingstatic || throw(ArgumentError("$macroname can't statically glue slice without using StaticArrays"))
    elseif @capture(right, inn_[iind__] ) # this must be tested last, else inn_ matches A[] outer indices too!
        ii = Any[]
    else
        throw(ArgumentError("$macroname can't understand right hand side $right"))
    end
    inn = esc(inn)

    V && @info "@capture RHS" inn repr(iind) repr(ii) willglue willstaticglue

    ## new parsing code! 
    # "i" means input as before, meaning stage 3-ish, before/after gluing
    # "a" means step 1, where the very first thing to do is view(A, aget...)
    # Writing into the same dictionary etc sdict.

    iflat, aget, _, ineg = parse!(sdict, inn, iind, ii)

    V && @info "parse! RHS" repr(iflat) repr(aget) repr(ineg) length(sdict.dict) length(sdict.checks)

    if sort(oflat) != sort(iflat)
        throw(ArgumentError("mismatched indices on left & right: $(Tuple(oflat)) <- $(Tuple(iflat))"))
    end

    ## finally we can parse "rex" too, done last as its dimensions are prettier (and stabler).
    rflags = Any[]
    @capture(rex, (rexi__,)) || (rexi = Any[rex])
    parse!(sdict, nothing, [], rexi, true, rflags)

    V && @info "parse! OPT, mostly into dict" repr(rexi) repr(rflags) length(sdict.dict) length(sdict.checks)

    if :base in rflags || :_ in rflags
        usingstatic = false
        usingstrided = false
    end
    if :assert in rflags || :! in rflags
        mustcheck = true
    end

    # for step 1, view:
    anum_fixed = count(!isequal(:), aget) # count(i -> i != (:), aget)
    willinitview = anum_fixed > 0

    # for step 2, shape
    isz = Any[]
    for i in iflat[length(ii)+1 : end]  # outer directions just before glue
        d = findcheck(i, oflat)         # look up in canonical list
        push!(isz, :( sz[$d] ) )
    end
    willshapeinput = (length(isz) + anum_fixed) != length(iind) # if equal then no tuples

    # for step 3, gluing:
    icode = sorted_starcode(length(iflat), length(ii)) # star for each outer dim, at the right 

    # for step 4 reversing: along these indices: 
    neg_ind = oddunique(vcat(ineg, oneg))
    willreverse = length(neg_ind) > 0 
    if willreverse
        # this is before permutedims, hence need iflat's numbers
        revdims = Tuple(sort([ findcheck(i, iflat) for i in neg_ind ]))
    else
        revdims = ()
    end

    V && @info "right..." willshapeinput repr(isz) icode willreverse revdims willinitview anum_fixed

    #==================== MIDDLE = STEP 4,5 ====================#
    # Or perhaps 3,4,5,6: glue, reverse, permute, slice/reduce

    willpermute = false
    if iflat != oflat
        perm = sortperm(oflat)[invperm(sortperm(iflat))] |> invperm |> Tuple # surely can be more concise!
        willpermute = true

        V && println("will permute, perm = ", perm)
    end

    if willstaticslice && usingstatic
        for i in oi # check that all sizes were given
            if @capture(sdict.dict[i], size(__)) # then not from rex
                throw(ArgumentError("$macroname needs explicit size for $i, to perform StaticArray slicing"))
            end
        end
        osdimex = :( Size($([sdict.dict[i] for i in oi]...)) ) # construct Size() object for SArray
        mustcheck = true

        V && println("static slicing sizes: osdimex = ",osdimex)
    end

    # TODO think more -- see "simplicode" function now
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

    ## Sizes: most of the hard work is now done by parse! and friends. 
    # but we still need a few things:

    slist = sizeinfer(sdict, oflat) # canonical size list -- expressions like size(A,d)
    slistex = :(($(slist...) ,))    # if needsizes=true,  sz = $slistex  will be inserted
                                    # and all my reshapes isz, fsz (or in-place, osz) refer to sz[d].
    szall = Any[:(sz[$d]) for d=1:length(oflat)]
    szex = :(($(szall...) ,))       # this complete list is helpful to colonise! function,
                                    # whose job in life is to simplify reshape size arguments.

    #==================== ASSEMBLE ====================#

    needsizes = false
    havecopied = false
    
    ex = inn

    ## 1. VIEW I.E. DISCARD FIXED DIMENSIONS 
    if willinitview
        agetex = :(($(aget...) ,))
        ex = :( view($ex, $(agetex)...) )
    end

    ## 2. RESHAPE CONTAINER
    if willshapeinput
        needsizes |= colonise!(isz, slist)
        iszex = :(($(isz...) ,))
        ex = :( reshape($ex, $iszex) )
    end

    ## 3. GLUE SLICES
    if willglue
        ex = :(glue($ex, $icode))
        havecopied = true
    elseif willstaticglue
        ex = :( static_glue($ex) )
    end

    ## 4. REVERSE
    if willreverse
        # revdims = filter(d -> isigns[d]==-1, 1:length(iflat)) |> Tuple
        if length(revdims) == 1
            revdims = revdims[1]
        end
        ex = :( reverse($ex, dims=$revdims) )
        havecopied = true
    end

    if !inplace
        V && @info "steps 1-4 done, working forwards..." ex

        ## 5. PERMUTEDIMS
        if willpermute
            if willstaticslice # then we cannot use lazy permutedims, including transpose
                ex = :( permutedims($ex, $perm) )
                havecopied = true
            elseif perm==(2,1)
                ex = :( transpose($ex) )
            elseif usingstrided
                ex = :( strided_permutedims($ex, $perm) ) # this is lazy # TODO maybe not always!!  @pretty @shape N[k,i] == S[k]{i} from readme with Strided ??
            else
                ex = :( permutedims($ex, $perm) )
                havecopied = true
            end
            V && println("step 5:  ex = ",ex)
        end

        if havecopied && mustnotcopy
            throw(ArgumentError(string("@shape can't figure out how to do what you ask without copying... ",
                "the problem might be gluing, or might be permutedims... if the latter, try using Strided")))
        end
        needcopy = !havecopied && mustcopy

        ## 6. SLICE, OR REDUCE
        if willstaticslice
            if needcopy
                ex = :( copy(ex) )
                havecopied = true
            end
            if willshapeout
                ex = :( static_slice($ex, $osdimex, false) ) # TODO invent a case where this happens?
            else
                ex = :( static_slice($ex, $osdimex) )
            end
            V && println("step 6 static:  ex = ",ex)

        elseif willslice
            if needcopy
                ex = :( slicecopy($ex, $ocode) )
                havecopied = true
            else
                ex = :( sliceview($ex, $ocode) )
            end
            V && println("step 6 slice:  ex = ",ex)

        elseif willreduce
            if redfun == :sum
                ex = :( sum_drop($ex, dims = $odims) ) # just to look tidy!
            elseif redfun == :prod
                ex = :( prod_drop($ex, dims = $odims) )
            elseif redfun == :maximum
                ex = :( max_drop($ex, dims = $odims) )
            else
                redfun = esc(redfun)
                ex = :( dropdims( $redfun($ex, dims = $odims), dims = $odims) )
            end
            havecopied = true
            V && println("step 6' reduction:  ex = ",ex)
        end

        ## 7. RESHAPE
        if willshapeout
            needsizes |= colonise!(fsz, slist) # simplify fsz
            if fsz == [Colon()]
                ex = :( vec($ex) )
            else
                fszex = :(($(fsz...) ,))
                ex = :( reshape($ex, $fszex) )
            end
            V && println("step 7:  ex = ",ex)
        end

        if !havecopied && mustcopy
            ex = :( copy($ex) )
            havecopied = true
        end

        if hasoutputname # only uncertain in forward case, remember
            ex = :( $outn = $ex )
        end

    elseif inplace && willreduce
        V && @info "steps 1-4 done, one more step, then reduce!()" ex

        ## 5. PERMUTEDIMS -- same as fowards, except willstaticslice = false
        if willpermute
            if usingstrided
                ex = :( strided_permutedims($ex, $perm) ) # this is lazy
            elseif perm==(2,1)
                ex = :( transpose($ex) )
            else
                ex = :( permutedims($ex, $perm) )
                havecopied = true
            end
            V && println("step 5:  ex = ",ex)
        end

        # now backwards
        rout = outn

        ## -8. VIEW
        if willfinalview
            ex = :( view($ex, $fget) )
        end

        ## -7. RESHAPE
        if willshapeout
            oszex = :(($(osz...) ,))
            rout =  :( reshape($rout, $oszex) ) # osz now means exactly this, not used if going forwards 
            needsizes = true 

            V && println("step 7 backward:  rout = ",rout)
        end

        redfun = esc(redfun) # already has ! added
        ex = :( $redfun($rout, $ex) )
        V && println("step 6 reduction:  ex = ",ex)


    elseif inplace
        V && @info "steps 1-4 done, now working BACKWARDS" ex willfinalview willshapeout willslice willpermute

        rout = outn

        ## -8. VIEW -- same as above
        if willfinalview
            ex = :( view($ex, $fget) )
        end

        ## -7. RESHAPE -- same as above
        if willshapeout
            oszex = :(($(osz...) ,))
            rout =  :( reshape($rout, $oszex) ) # osz now means exactly this, not used if going forwards 
            needsizes = true 

            V && println("step 7 backward:  rout = ",rout)
        end

        ## -6. UN-SLICE
        if willstaticslice
            rout = :( static_glue( $rout ) ) # TODO make an example that does this
            # note that this mutates the SArray slices, is that weird?
        elseif willslice
            throw(ArgumentError("@shape can't write in-place to slices right now, unless they are StaticArrays"))
            # Or better, you should NOT be working backwards at this point, you should write slices into mutable array
            # but this will be a bit of a mess to organise.
        end

        ## 5. PERMUTEDIMS!
        if willpermute
            if usingstrided
                ex = quote
                    strided_permutedims!($rout, $ex, $perm)
                    $outn
                end
            else
                ex = quote
                    permutedims!($rout, $ex, $perm) # Couldn't find transpose!
                    $outn
                end
            end
            V && println("step 5 backward:  ex = ",ex)

        else # throw out some steps...
            # best case is that we had nothing but reshaping -- go back to beginning!
            if !willslice && !willstaticslice && !willglue && !willstaticglue
                W && println("step 3' throwing out ex = ",ex, "  and rout = ", rout, "  to go back to inn = ", inn)
                W && println("... had needsizes = ",needsizes," -> false, and mustcheck = ",mustcheck, " -> true")
                ex = :( copyto!($outn, $inn) )
                needsizes = false
                mustcheck = true
                W && println("... finally ex = ",ex)
                # W && println("... for expr = ",expr)

            # second-best case is that we did some gluing but no slicing -- ignore rout
            elseif !willslice && !willstaticslice
                W && println("step 3'' throwing away rout = ", rout, "  to go back to ex = ", ex) ## WTF?
                W && println("... had needsizes = ",needsizes," -> false, and mustcheck = ",mustcheck, " left alone")
                ex = :( copyto!($outn, $ex) )
                needsizes = false # not very confident about this
                W && println("... finally ex = ",ex)
                # W && println("... for expr = ",expr)

            # third-best... does this happen? what was I thinking? TODO think more about what stage you roll back to
            else
                W && println("step 3''' ... got 3rd case! does this happen? rout = ", rout, " and ex = ",ex)
                ex = :( copyto!($rout, $ex) )
                W && println("... finally ex = ",ex)
            end
        end
    end

    #==================== FINALISE ====================#
    # TODO push don't nest, like https://github.com/JuliaLang/julia/blob/c8450d862f0e2653011f68118daecfe12b398c90/base/show.jl#L551

    if needsizes
        ex = quote
            local sz = $slistex
            $ex
        end
    end

    if mustcheck && length(sdict.checks) > 0
        ex = quote
            $(sdict.checks...)
            $ex
        end
    end

    return ex
end


############################### DATA FUNCTIONS ################################

if VERSION < v"1.1.0"
    include("eachslice.jl") # functions from the future
end

include("cat-and-slice.jl") # my simple functions

using Requires

function __init__()

    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        include("glue-a-view.jl")
        V && @info "loaded code for StaticArrays"
    end

    @require JuliennedArrays = "5cadff95-7770-533d-a838-a1bf817ee6e0" begin
        include("julienne.jl")
        V && @info "loaded code for JuliennedArrays"
    end

    @require Strided = "5e0ebb24-38b0-5f93-81fe-25c709ecae67" begin
        include("lazy-stride.jl")
        V && @info "loaded code for Strided"
    end

end

############################### PRETTY ################################

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

end # module
