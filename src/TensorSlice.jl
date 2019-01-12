module TensorSlice

export @shape, @reduce, @pretty

using MacroTools

include("parse.jl")
include("icheck.jl")
include("cast.jl")

V = false # capital V for extremely verbose
W = false # printouts where I'm working on things

"""
    @shape Z[i...] := A[j...] opt

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
    _shape(expr, rex; icheck=false)
end

"""
    @shape! Z[i...] := A[j...] opt

Variant of `@shape` which effectively runs `@check!()` on each tensor.
"""
macro shape!(expr, rex=nothing)
    where = (mod=__module__, src=__source__)
    @warn "@shape! doesn't work well yet"
    _shape(expr, rex; icheck=true, where=where)
end

function _shape(expr, rex=nothing; icheck=false, where=nothing)
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
    # Do this first to get a list of indices in canonical order, indVflat
    # And parse expr here not in tensor_slice_main as LHS differs for @reduce

    willslice = false
    willstaticslice = false

    if @capture(left, ( nameZ_[indZ__][indZsub__] | [indZ__][indZsub__] ) )
        willslice = true
    elseif @capture(left, ( nameZ_[indZ__]{indZsub__} | [indZ__]{indZsub__} ) )
        willstaticslice = true
    elseif @capture(left, ( nameZ_[indZ__] | [indZ__] ) )
        indZsub = Any[]
    else
        throw(ArgumentError("@shape can't understand left hand side $left"))
    end

    V && @info "@capture LHS" nameZ repr(indZ) repr(indZsub) willslice willstaticslice

    tensor_slice_main(nameZ, indZ, indZsub, # parsing of LHS in progress
        willslice, willstaticslice,   # flags from LHS of @shape
        false, nothing,               # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        "@shape", icheck, where)
end

"""
    @reduce Z[j] := sum(i,k) A[i,j,k]

Tensor reduction macro:
* The reduction funcition can be anything which works like `sum(B, dims=(1,3))`, 
  for instance `prod` and `maximum` and `Statistics.mean`. 
* The right side can be anything that `@shape` understands, including gluing of slices `B[i,k][j]` 
  and reshaping `B[i\\j,k]`.
* The left side again works as in `@shape`, although slicing is not allowed.
* In-place operations `Z[j] = sum(...` will construct the banged version of the given function's name, 
  which must work like `sum!(Z, A)`.
* Index ranges may be given afterwards (as for `@shape`) or inside the reduction `sum(i:3, k:4)`. 
* All indices appearing on the right must appear either within `sum(...)` etc, or on the left. 
"""
macro reduce(expr, right, rex=nothing)
    _reduce(expr, right, rex; icheck=false)
end

"""
    @reduce! Z[j] := sum(i,k) A[i,j,k]

Variant of `@reduce` which effectively runs `@check!()` on each tensor.
"""
macro reduce!(expr, right, rex=nothing)
    where = (mod=__module__, src=__source__)
    _reduce(expr, right, rex; icheck=true, where=where)
end

function _reduce(expr, right, rex=nothing; icheck=false, where=nothing)
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
    # Do this first to get a list of indices in canonical order, indVflat
    # And parse expr here not in tensor_slice_main as LHS differs for @shape

    if @capture(left, ( nameZ_[indZ__] | [indZ__] ) )
    else
        throw(ArgumentError("@reduce can't understand left hand side $left"))
    end

    if @capture(red, redfun_(indZsub__) )
        if sign == :(=) # inplace=true, later
            if !endswith(string(redfun), '!')
                redfun = Symbol(redfun, '!')
            end
        end
    else
        throw(ArgumentError("@reduce can't understand reduction formula $red"))
    end

    V && @info "@capture LHS" nameZ redfun repr(indZ) repr(indZsub)

    tensor_slice_main(nameZ, indZ, indZsub, # parsing of LHS in progress
        false, false,                 # flags from LHS of @shape
        true, redfun,                 # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        "@reduce", icheck, where)
end

############################### NOTATION ################################
#=

After getting hopelessly confused, I decided to aphabet-ise my notation: 

nameA, indA, indAsub   -- given RHS
sizeA      -- known & used for fixing ranges
nameAesc   -- unglified

getB, numB -- for view(A, get), and number of fixed indices
sizeC      -- after reshaping outer indices, in terms of sz[]
codeD      -- for gluing
indEflat   -- after gluing 
revFdims   -- done before permutedims
shiftG     -- ditto, but not yet written

permHU     -- permHU(indEflat -> infVflat)
storeAZ    -- information from parse! both

indVflat   -- canonical list
sizeVlist  -- filled in... at most one (:)

codeW      -- for slicing. sizeWstatic is Size() of the slice.
redWdims   -- if reducing instead, over these dims

sizeX      -- after slicing, container size, in sz[]
getY, numY -- for view(Z, get) in-place, and number fixed

nameZesc
sizeZ      -- for final reshape, in sz[]
nameZ, indZ, indZsub   -- given LHS

=#
############################### GIANT MONO-FUNCTION ################################

function tensor_slice_main(nameZ, indZ, indZsub,
        willslice, willstaticslice,   # flags from LHS of @shape
        willreduce, redfun,           # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        macroname, icheck, where)

    ## test what packages are loaded
    usingstrided = isdefined(TensorSlice, :Strided)
    usingstatic =  isdefined(TensorSlice, :StaticArrays)

    willstaticslice && !usingstatic &&
        throw(ArgumentError("@shape can't statically glue slice without using StaticArrays"))

    ## make sign == :(:=) mean simple whatever is easiest:
    mustcopy =    sign == :(|=)
    mustnotcopy = sign == :(==)
    inplace =     sign == :(=)

    V && @info "todo & packages" mustcopy mustnotcopy inplace usingstatic usingstrided

    #==================== LEFT, CONTINUED ====================#

    if nameZ == nothing
        hasoutputname = false
        nameZesc = gensym()
        inplace && throw(ArgumentError("$macroname must have an output name for in-place operations. Try with := instead"))
    else
        hasoutputname = true
        nameZesc = esc(nameZ)
    end

    ## new parsing code! 

    storeAZ = SizeDict()

    indVflat, getY, sizeZ, negZ = parse!(storeAZ, ifelse(inplace, nameZesc, nothing), indZ, indZsub, willreduce)

    V && @info "parse! LHS -- including canonical indVflat" repr(indVflat) repr(getY) repr(sizeZ) repr(negZ) length(storeAZ.dict) length(storeAZ.checks)

    # for step 6, slicing:
    codeW = sorted_starcode(length(indVflat), length(indZsub)) # star for each outer dim, at the right 
    # or else step 6', reduce: 
    if length(indZsub) ==1
        redWdims = 1  # reduction, like inner indices, is leftmost
    else
        redWdims = Tuple( 1 : length(indZsub) )
    end

    # for step 8, only used if working in-place:
    numY = count(!isequal(:), getY) # numY = count(i -> i != (:), getY)
    willfinalview = numY > 0

    if willfinalview && !inplace
        for i in getY
            if i != (:) && i != 1
                throw(ArgumentError("$macroname can't create an array with constant index $i != 1"))
            end
        end
    end

    # for step 7 reshape when working in-place: very simple? 
    # the relevant lenghts are the canonical ones, skipping any inner indices
    sizeX = Any[:(sz[$d]) for d=length(indZsub)+1:length(indVflat)]

    # for step 7, reshape -> sizeZ. Only needed if ndims differs:
    willshapeout = (length(sizeZ) + numY) != length(indVflat) - length(indZsub)

    V && @info "left..." willslice codeW willreduce redWdims repr(sizeX) numY willshapeout repr(sizeZ)

    #==================== RIGHT = INPUT = STEPS 1,2,3 ====================#

    willglue = false
    willstaticglue = false
    mustcheck = false

    if @capture(right, nameA_[indA__][indAsub__] )
        willglue = true
        mustnotcopy && throw(ArgumentError("$macroname can't create a view of glued-together slices, unless they are StaticArrays"))
    elseif @capture(right, nameA_[indA__]{indAsub__} )
        willstaticglue = true
        usingstatic || throw(ArgumentError("$macroname can't statically glue slice without using StaticArrays"))
    elseif @capture(right, nameA_[indA__] ) # this must be tested last, else nameA_ matches A[] outer indices too!
        indAsub = Any[]
    else
        throw(ArgumentError("$macroname can't understand right hand side $right"))
    end
    nameAesc = esc(nameA)

    V && @info "@capture RHS" nameAesc repr(indA) repr(indAsub) willglue willstaticglue

    ## new parsing code! 
    # Writing into the same dictionary etc storeAZ.

    indEflat, getB, _, negA = parse!(storeAZ, nameAesc, indA, indAsub)

    V && @info "parse! RHS" repr(indEflat) repr(getB) repr(negA) length(storeAZ.dict) length(storeAZ.checks)

    if sort(indVflat) != sort(indEflat)
        throw(ArgumentError("mismatched indices on left & right: $(Tuple(indVflat)) <- $(Tuple(indEflat))"))
    end

    ## finally we can parse "rex" too, done last as its dimensions are prettier (and stabler).
    rflags = Any[]
    @capture(rex, (rexi__,)) || (rexi = Any[rex])
    parse!(storeAZ, nothing, [], rexi, true, rflags)

    V && @info "parse! OPT, mostly into dict" repr(rexi) repr(rflags) length(storeAZ.dict) length(storeAZ.checks)

    if :base in rflags || :_ in rflags
        usingstatic = false
        usingstrided = false
    end
    if :assert in rflags || :! in rflags
        mustcheck = true
    end

    # for step 1, view:
    numB = count(!isequal(:), getB) # count(i -> i != (:), getB)
    willinitview = numB > 0

    # for step 2, shape
    sizeC = Any[]
    for i in indEflat[length(indAsub)+1 : end]  # outer directions just before glue
        d = findcheck(i, indVflat)         # look up in canonical list
        push!(sizeC, :( sz[$d] ) )
    end
    willshapeinput = (length(sizeC) + numB) != length(indA) # if equal then no tuples

    # for step 3, gluing:
    codeD = sorted_starcode(length(indEflat), length(indAsub)) # star for each outer dim, at the right 

    # for step 4 reversing: along these indices: 
    neg_ind = oddunique(vcat(negA, negZ))
    willreverse = length(neg_ind) > 0 
    if willreverse
        # this is before permutedims, hence need indEflat's numbers
        revFdims = Tuple(sort([ findcheck(i, indEflat) for i in neg_ind ]))
    else
        revFdims = ()
    end

    V && @info "right..." willshapeinput repr(sizeC) codeD willreverse revFdims willinitview numB

    #==================== MIDDLE = STEP 4,5 ====================#
    # Or perhaps 3,4,5,6: glue, reverse, permute, slice/reduce

    willpermute = false
    if indEflat != indVflat
        permHU = sortperm(indVflat)[invperm(sortperm(indEflat))] |> invperm |> Tuple # surely can be more concise!
        willpermute = true

        V && println("will permute, permHU = ", permHU)
    end

    if willstaticslice && usingstatic
        for i in indZsub # check that all sizes were given
            if @capture(storeAZ.dict[i], size(__)) # then not from rex
                throw(ArgumentError("$macroname needs explicit size for $i, to perform StaticArray slicing"))
            end
        end
        sizeWstaticex = :( Size($([storeAZ.dict[i] for i in indZsub]...)) ) # construct Size() object for SArray
        mustcheck = true

        V && println("static slicing sizes: sizeWstaticex = ",sizeWstaticex)
    end

    # TODO think more -- see "simplicode" function now
    if willpermute && willslice # hence not willstaticslice
        if permHU==(2,1)
            codeW = (codeW[2],codeW[1])
            willpermute = false
            W && println("removed permutation by changing slicing: now codeW = ",codeW)
            # W && println("... for expr = ",expr)
        else
            V && println("am going to slice a permuted array. Can I remove permutation by re-ordering slice?")
        end
    end

    ## Sizes: most of the hard work is now done by parse! and friends. 
    # but we still need a few things:

    sizeVlist = sizeinfer(storeAZ, indVflat) # canonical size list -- expressions like size(A,d)
    sizeVlistex = :(($(sizeVlist...) ,))    # if needsizes=true,  sz = $sizeVlistex  will be inserted
                                    # and all my reshapes sizeC, sizeZ (or in-place, sizeX) refer to sz[d].
    # szall = Any[:(sz[$d]) for d=1:length(indVflat)]
    # szex = :(($(szall...) ,))       # this complete list is helpful to colonise! function,
    #                                 # whose job in life is to simplify reshape size arguments.

    #==================== ASSEMBLE ====================#

    needsizes = false
    havecopied = false
    
    ex = nameAesc

    ## 1. VIEW I.E. DISCARD FIXED DIMENSIONS 
    if willinitview
        getBex = :(($(getB...) ,))
        ex = :( view($ex, $(getBex)...) )
    end

    ## 2. RESHAPE CONTAINER
    if willshapeinput
        needsizes |= colonise!(sizeC, sizeVlist)
        sizeCex = :(($(sizeC...) ,))
        ex = :( reshape($ex, $sizeCex) )
    end

    ## 3. GLUE SLICES
    if willglue
        ex = :(glue($ex, $codeD))
        havecopied = true
    elseif willstaticglue
        ex = :( static_glue($ex) )
    end

    ## 4. REVERSE
    if willreverse
        for d in revFdims # reverse cannot take dims=(2,3), surprisingly
            ex = :( reverse($ex, dims=$d) )
        end
        havecopied = true
    end

    if !inplace
        V && @info "steps 1-4 done, working forwards..." ex

        ## 5. PERMUTEDIMS
        if willpermute
            if willstaticslice # then we cannot use lazy permutedims, including transpose
                ex = :( permutedims($ex, $permHU) )
                havecopied = true
            elseif permHU==(2,1)
                ex = :( transpose($ex) )
            elseif usingstrided
                ex = :( strided_permutedims($ex, $permHU) ) # this is lazy # TODO maybe not always!!  @pretty @shape N[k,i] == S[k]{i} from readme with Strided ??
            else
                ex = :( permutedims($ex, $permHU) )
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
                ex = :( static_slice($ex, $sizeWstaticex, false) ) # TODO invent a case where this happens?
            else
                ex = :( static_slice($ex, $sizeWstaticex) )
            end
            V && println("step 6 static:  ex = ",ex)

        elseif willslice
            if needcopy
                ex = :( slicecopy($ex, $codeW) )
                havecopied = true
            else
                ex = :( sliceview($ex, $codeW) )
            end
            V && println("step 6 slice:  ex = ",ex)

        elseif willreduce
            if willshapeout # then no need for dropdims
                redfun = esc(redfun)
                ex = :( $redfun($ex, dims = $redWdims) )
            else
                if redfun == :sum
                    ex = :( sum_drop($ex, dims = $redWdims) ) # just to look tidy!
                elseif redfun == :prod
                    ex = :( prod_drop($ex, dims = $redWdims) )
                elseif redfun == :maximum
                    ex = :( max_drop($ex, dims = $redWdims) )
                else
                    redfun = esc(redfun)
                    ex = :( dropdims( $redfun($ex, dims = $redWdims), dims = $redWdims) )
                end
            end
            havecopied = true
            V && println("step 6' reduction:  ex = ",ex)
        end

        ## 7. RESHAPE
        if willshapeout
            needsizes |= colonise!(sizeZ, sizeVlist) # simplify sizeZ
            if sizeZ == [Colon()]
                ex = :( vec($ex) )
            else
                sizeZex = :(($(sizeZ...) ,))
                ex = :( reshape($ex, $sizeZex) )
            end
            V && println("step 7:  ex = ",ex)
        end

        if !havecopied && mustcopy
            ex = :( copy($ex) )
            havecopied = true
        end

        if hasoutputname # only uncertain in forward case, remember
            ex = :( $nameZesc = $ex )
        end

    elseif inplace && willreduce
        V && @info "steps 1-4 done, one more step, then reduce!()" ex

        ## 5. PERMUTEDIMS -- same as fowards, except willstaticslice = false
        if willpermute
            if usingstrided
                ex = :( strided_permutedims($ex, $permHU) ) # this is lazy
            elseif permHU==(2,1)
                ex = :( transpose($ex) )
            else
                ex = :( permutedims($ex, $permHU) )
                havecopied = true
            end
            V && println("step 5:  ex = ",ex)
        end

        # now backwards
        rout = nameZesc

        ## -8. VIEW
        if willfinalview
            ex = :( view($ex, $getY) )
        end

        ## -7. RESHAPE
        if willshapeout
            sizeXex = :(($(sizeX...) ,))
            rout =  :( reshape($rout, $sizeXex) ) # sizeX now means exactly this, not used if going forwards 
            needsizes = true 

            V && println("step 7 backward:  rout = ",rout)
        end

        redfun = esc(redfun) # already has ! added
        ex = :( $redfun($rout, $ex) )
        V && println("step 6 reduction:  ex = ",ex)


    elseif inplace
        V && @info "steps 1-4 done, now working BACKWARDS" ex willfinalview willshapeout willslice willpermute

        rout = nameZesc

        ## -8. VIEW -- same as above
        if willfinalview
            ex = :( view($ex, $getY) )
        end

        ## -7. RESHAPE -- same as above
        if willshapeout
            sizeXex = :(($(sizeX...) ,))
            rout =  :( reshape($rout, $sizeXex) ) # sizeX now means exactly this, not used if going forwards 
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
                    strided_permutedims!($rout, $ex, $permHU)
                    $nameZesc
                end
            else
                ex = quote
                    permutedims!($rout, $ex, $permHU) # Couldn't find transpose!
                    $nameZesc
                end
            end
            V && println("step 5 backward:  ex = ",ex)

        else # throw out some steps...
            # best case is that we had nothing but reshaping -- go back to beginning!
            if !willslice && !willstaticslice && !willglue && !willstaticglue
                W && println("step 3' throwing out ex = ",ex, "  and rout = ", rout, "  to go back to nameAesc = ", nameAesc)
                W && println("... had needsizes = ",needsizes," -> false, and mustcheck = ",mustcheck, " -> true")
                ex = :( copyto!($nameZesc, $nameAesc) )
                needsizes = false
                mustcheck = true
                W && println("... finally ex = ",ex)
                # W && println("... for expr = ",expr)

            # second-best case is that we did some gluing but no slicing -- ignore rout
            elseif !willslice && !willstaticslice
                W && println("step 3'' throwing away rout = ", rout, "  to go back to ex = ", ex) ## WTF?
                W && println("... had needsizes = ",needsizes," -> false, and mustcheck = ",mustcheck, " left alone")
                ex = :( copyto!($nameZesc, $ex) )
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
            local sz = $sizeVlistex
            $ex
        end
    end

    if mustcheck && length(storeAZ.checks) > 0
        ex = quote
            $(storeAZ.checks...)
            $ex
        end
    end

    return ex
end


############################### DATA FUNCTIONS ################################

if VERSION < v"1.1.0"
    include("eachslice.jl") # functions from the future, TODO figure out Compat
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
