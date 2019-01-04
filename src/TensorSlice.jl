module TensorSlice

export @shape, @reduce, @pretty

using MacroTools

V = false # capital V for extremely verbose
W = false # printouts where I'm working on things

"""
    @shape A[i...] := B[j...]  opt
Tensor reshaping and slicing macro. Understands the following things:
* `A[i,j,k]` is a 3-tensor with these indices
* `B[(i,j),k]` is the same thing, reshaped to a matrix
* `C[k][i,j]` is a vector of matrices
* `D[j,k]{i}` is an ordinary matrix of StaticVectors, reinterpreted from `A`.

They can be related in three ways:
* `:=` creates a new object
* `=` writes into an existing object
* `==` creates a view of the old object.

Options can be specified at the end (if several, separated by `,`)
* `i:3` fixes the range of index `i`
* `+` or `!` force explicit dimension checks (these may change)
* `_` restricts to methods from basic Julia (also may change).

Static slices need `using StaticArrays`, and to create them you must give all slice dimensions explicitly.
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

    V && println("sign =  ", sign)#, "     of type ", typeof(sign))
    V && println("left =  ", left)
    V && println("right = ", right)
    V && println("rex =   ", something(rex,"nothing"))

    ## test what packages are loaded
    usingstrided = isdefined(TensorSlice, :Strided)
    usingstatic =  isdefined(TensorSlice, :StaticArrays)

    if rex != nothing
        @capture(rex, (rvec__,)) || (rvec = Any[rex])
        for r in rvec
            if r == :(_) # this flag means use base only
                usingstrided = false
                usingstatic = false
            end
        end
    end

    Vvec = [usingstatic ? "static" : missing, usingstrided ? "strided" : missing]
    V && println("using = ", join(skipmissing(Vvec), " & "))

    #==================== LEFT = OUTPUT = STEPS 4,5 ====================#
    # Do this first to get a list of indices in canonical order, oflat

    willslice = false
    willstaticslice = false

    if @capture(left, ( outn_[oind__][oi__] | [oind__][oi__] ) )
        willslice = true
        V && println("will slice, inner indices = ", oi)

    elseif @capture(left, ( outn_[oind__]{oi__} | [oind__]{oi__} ) )
        willstaticslice = true
        V && println("will static slice, inner indices = ", oi)
        usingstatic || throw(ArgumentError("@shape can't statically glue slice without using StaticArrays"))

    elseif @capture(left, ( outn_[oind__] | [oind__] ) )
        oi = Any[]
        V && println("no slicing")

    else
        throw(ArgumentError("@shape can't understand left hand side $left"))
    end

    tensor_slice_main(outn, oind, oi, # parsing of LHS in progress
        willslice, willstaticslice,   # flags from LHS of @shape
        false, nothing,               # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        usingstatic, usingstrided,    # what's loaded
        "@shape")
end

"""
    @reduce A[j] := sum(i,k) B[i,j,k]
Tensor reduction macro.
* The right-hand-side can be anything that `@shape` understands, including gluing of slices, and reshaping.
* The reduction funcition can be anything which works like `sum(B, dims=(1,3))`.
* The left-hand side again works as in `@shape`, although slicing is not allowed.
* Only creation `:=` is currently working (although in-place `=` will eventually).
"""
macro reduce(expr, right, rex=nothing)

    if @capture(expr, left_ = red_ )
        sign = :(=)
    elseif @capture(expr, left_ := red_ )
        sign = :(:=)
    else
        throw(ArgumentError("@reduce can't begin to understand $expr"))
    end

    V && println("sign =  ", sign)#, "     of type ", typeof(sign))
    V && println("left =  ", left)
    V && println("right = ", red)
    V && println("right = ", right)
    V && println("rex =   ", something(rex,"nothing"))

    ## test what packages are loaded
    usingstrided = isdefined(TensorSlice, :Strided)
    usingstatic =  isdefined(TensorSlice, :StaticArrays)

    if rex != nothing
        @capture(rex, (rvec__,)) || (rvec = Any[rex])
        for r in rvec
            if r == :(_) # this flag means use base only
                usingstrided = false
                usingstatic = false
            end
        end
    end

    Vvec = [usingstatic ? "static" : missing, usingstrided ? "strided" : missing]
    V && println("using = ", join(skipmissing(Vvec), " & "))

    #==================== LEFT = OUTPUT = STEPS 4,5 ====================#
    # Do this first to get a list of indices in canonical order, oflat

    if @capture(left, ( outn_[oind__] | [oind__] ) )
    else
        throw(ArgumentError("@reduce can't understand left hand side $left"))
    end

    if @capture(red, redfun_(oi__) )
        if sign == :(=)
            if !endswith(string(redfun), '!')
                redfun = Symbol(redfun, '!')
            end
        end
        V && println("will reduce using redfun = ", redfun, ", inner indices = ", oi)
    else
        throw(ArgumentError("@reduce can't understand reduction formula $red"))
    end

    tensor_slice_main(outn, oind, oi, # parsing of LHS in progress
        false, false,                 # flags from LHS of @shape
        true, redfun,                 # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        usingstatic, usingstrided,    # what's loaded
        "@reduce")
end

############################### GIANT MONO-FUNCTION ################################

function tensor_slice_main(outn, oind, oi,
        willslice, willstaticslice,   # flags from LHS of @shape
        willreduce, redfun,           # info from LHS of @reduce
        sign, right, rex,             # RHS still to be done
        usingstatic, usingstrided,    # what's loaded
        macroname)

    ## make sign == :(:=) mean simple whatever is easiest:
    mustcopy =    sign == :(|=)
    mustnotcopy = sign == :(==)
    inplace =     sign == :(=)

    #==================== LEFT = OUTPUT, CONTINUED ====================#

    if outn == nothing
        hasoutputname = false
        outn = gensym()
        sign == :(=) && throw(ArgumentError("$macroname must have an output name for in-place operations. Try with := instead"))
    else
        hasoutputname = true
        outn = esc(outn)
    end

    if all(isa.(oind, Symbol)) # outer indices oind already flat i,j,k
        willshapeout = false

        oflat = vcat(oi, oind) # inner indices first, if any
        osz = Any[:(sz[$d]) for d = length(oi)+1:length(oflat)] # what you would reshape output to, if you changed your mind
        ocode = (ntuple(i -> :, length(oi))..., ntuple(i -> *, length(oind))...) # needed for slicing, perhaps

        V && println("no output reshape")

    else # outer indices oind something like i,(j,k),l
        willshapeout = true

        oflat = Any[]
        osz = Any[]
        for o in oind
            if o isa Symbol
                push!(oflat, o)
                push!(osz, :(sz[$(length(oi) + length(oflat))]))
            elseif @capture(o, (tind__,)) || @capture(o, t1_\ t2_\ t3_ ) || @capture(o, t1_\ t2_ )
                if tind == nothing
                    tind = t3==nothing ? (t1,t2) : (t1,t2,t3)
                end
                append!(oflat, tind)
                d = length(oi) + length(oflat) - length(tind) + 1 # d is index of _final_ oflat, the canonical list
                tsz = :(sz[$d])
                for i=2:length(tind)
                    d += 1
                    tsz = :($tsz * sz[$d]) # sz isn't defined yet... it'll will have values of slist below
                end
                push!(osz, tsz)
            else
                throw(ArgumentError("$macroname can't understand $o on left hand side"))
            end
        end
        # ocode for slicing needs length of oflat without the sliced-off indices oi
        ocode = (ntuple(i -> :, length(oi))..., ntuple(i -> *, length(oflat))...)
        oflat = vcat(oi, oflat)
        V && println("will reshape output, osz = ",osz)
    end

    if length(oi)==1
        odims = 1 # prettier output, but may make permutedims cleverness harder
    else
        odims = Tuple(1:length(oi))
    end

    V && (willslice || willstaticslice) && println("ocode = ", ocode, " for slicing")
    V && willreduce && println("odims = ", odims, " for reduction over oi = ",oi)
    V && println("oind =  ", oind) # oind lists indices whose lengths are some size(A,d)
    V && println("oflat = ", oflat, ", this is the canonical order")

    #==================== RIGHT = INPUT = STEPS 1,2 ====================#

    willglue = false
    willstaticglue = false

    if @capture(right, inn_[iind__][ii__] )
        willglue = true
        V && println("will glue, inner indicies = ", ii)
        sign == :(==) && throw(ArgumentError("$macroname can't create a view of glued-together slices, unless they are StaticArrays"))

    elseif @capture(right, inn_[iind__]{ii__} )
        willstaticglue = true
        V && println("will static glue, inner indicies = ", ii)
        usingstatic || throw(ArgumentError("$macroname can't statically glue slice without using StaticArrays"))

    elseif @capture(right, inn_[iind__] ) # this must be tested last, else inn_ matches A[] outer indices too!
        ii = Any[]
        V && println("no glue")

    else
        throw(ArgumentError("$macroname can't understand right hand side $right"))
    end
    inn = esc(inn)

    if all(isa.(iind, Symbol)) # easy input, just a list of indices
        willshapeinput = false
        iflat = vcat(ii, iind) # inner indices first, if any
        icode = (ntuple(i -> :, length(ii))..., ntuple(i -> *, length(iind))...) # needed for gluing
        V && println("no input shaping")

    else # we need reshaping, BUT now only of OUTER array
        willshapeinput = true
        # iflat, isz =  flat_and_sz(iind, oflat, "right")
        iflat = Any[]
        isz = Any[]
        for i in iind
            if i isa Symbol
                push!(iflat, i)
                d = findcheck(i, oflat)
                push!(isz, :(sz[$d])) # osh is for reshaping stage -- has only the sizes of outer indices, and products
            elseif @capture(i, (tind__,)) || @capture(i, t1_\ t2_\ t3_ ) || @capture(i, t1_\ t2_ )
                if tind == nothing
                    tind = t3==nothing ? (t1,t2) : (t1,t2,t3)
                end
                for ti in tind
                    push!(iflat, ti)
                    d = findcheck(ti, oflat) # d is the index of oflat, the canonical list
                    push!(isz, :(sz[$d])) # sz isn't defined yet, will be indexed like oflat
                end
            else
                throw(ArgumentError("$macroname can't understand $i on right hand side"))
            end
        end
        # if we are gluing, we do this after reshaping, thus need length of flattened iind
        icode = (ntuple(i -> :, length(ii))..., ntuple(i -> *, length(iflat))...)
        # still have to glue on inner indices, if any -- iflat is for permutedims stage
        iflat = vcat(ii, iflat)
        V && println("will reshape input, isz = ", isz, ")")
    end

    V && (willglue || willstaticglue) && println("icode = ", icode, " for gluing")
    V && println("iind =  ", iind) # iind lists indices whose lengths are some size(B,d)
    V && println("iflat = ", iflat) # iflat is some permutation of oflat

    #==================== SIZES ====================#

    if sort(oflat) != sort(iflat)
        throw(ArgumentError("mismatched indices on left & right: $(Tuple(oflat)) <- $(Tuple(iflat))"))
    end

    ## parse size annotations "rex"
    mustcheck = false
    rind = Any[]
    rsz = Any[]
    if rex != nothing
        @capture(rex, (rvec__,)) || (rvec = Any[rex])
        for r in rvec
            if @capture(r, ir_:lenr_ ) # | (ir_ <= lenr_)  ( ir_ ≦ lenr_ ) I can't make these work

                findcheck(ir, oflat) # error if it cannot find it on left
                push!(rind, ir)
                if lenr isa Int
                    push!(rsz, lenr) # don't escape, statically known
                else
                    push!(rsz, esc(lenr)) # escaping these allows ranges i:β with β=2 as in test
                end
                mustcheck = true

            elseif r == :(!) || r == :(+) # this forces size checks below, even if no dimensions given
                mustcheck = true
            elseif r == :(_) # also allowed but already dealt with above
                nothing
            else
                throw(ArgumentError("$macroname doesn't know what to do with $r. Index lengths should be  i:3  or else  (i:3, j:4)"))
            end
        end
        V && println("rind =  ",rind, "   rsz = ",rsz)
        V && mustcheck && println("mustcheck = true")
    end

    ## read off sizes whereever we can: input, output, rex
    V && println("where to get sizes:")
    # the goal is to make slist the most tidy list, and scheck the most comprehensive
    slist = Vector{Any}(undef, length(oflat)) # will be same order as oflat, canonical list
    schecks = Any[]
    for (n,i) in enumerate(oflat)
        id = findfirst(isequal(i), iind) # index of what dimension's size
        od = findfirst(isequal(i), oind)
        idi = findfirst(isequal(i), ii) # inner, meaning of constituent arrays
        odi = findfirst(isequal(i), something(oi,Any[]))
        rd = findfirst(isequal(i), rind) # index of rsz

        V && println("-- $i : id=$(something(id,"?")), idi=$(something(idi,"?")), ",
                "od=$(something(od,"?")), odi=$(something(odi,"?")), rd=$(something(rd,"?"))")

        if rd isa Int # best case
            slist[n] = rsz[rd]
            push_checks!(schecks, rsz[rd], getsize(inn, id, idi), getsize(outn, od, odi, sign))

        elseif id isa Int # then you can get size from input array
            slist[n] = getsize(inn, id)
            push_checks!(schecks, getsize(inn, id, idi), getsize(outn, od, odi, sign))

        elseif od isa Int && sign == :(=) # then you can get size from output array
            slist[n] = getsize(outn, od)
            push_checks!(schecks, getsize(inn, id, idi), getsize(outn, od, odi, sign)) # checks include sub-arrays

        elseif idi isa Int # only if we can't get a nice size(A,d) do we turn to sub-arrays
            slist[n] = getsize(inn, id, idi) # this is size(first(A),idi) since id==nothing
            push_checks!(schecks, getsize(inn, id, idi), getsize(outn, od, odi, sign))

        elseif odi isa Int && sign == :(=)
            slist[n] = getsize(outn, od, odi)

        else
            slist[n] = (:) # it's OK to leave one size a colon, checked a few lines down!
        end
    end

    if count(isequal(:), slist) >1
        ns = findall(isequal(:), slist)
        throw(ArgumentError(string("$macroname can't infer ranges of indices ",join(oflat[ns],", "))))
    end

    szall = Any[:(sz[$d]) for d=1:length(oflat)]

    V && println("slist=   ",slist)
    V && println("schecks= ",schecks)

    #==================== MIDDLE = STEP 3 ====================#

    willpermute = false
    if iflat != oflat
        perm = sortperm(oflat)[invperm(sortperm(iflat))] |> invperm |> Tuple # surely can be more concise!
        willpermute = true

        V && println("will permute, perm = ", perm)
    end

    if willstaticslice # already checked usingstatic
        for i in oi # check that all sizes were given
            if findfirst(isequal(i), rind) == nothing
                throw(ArgumentError("$macroname needs explicit size for $i, to perform StaticArray slicing"))
            end
        end
        osdimex = :( Size($(slist[1:length(oi)]...)) )
        mustcheck = true

        V && println("static slicing sizes: osdimex = ",osdimex)
    end
    # Here is a check that Size is working well:
    # f(M, valn::Val{N}) where N = @shape A[j]{i} == M[i,j] i:N
    # @code_warntype f(rand(2,3), Val(2))

    # TODO Think more about this -- it could also tidy up @pretty @shape A[(i,j)] := B[i][j]
    # but not for static glue
    if willpermute && willslice # hence not willstaticslice
        if perm==(2,1)
            ocode = (ocode[2],ocode[1])
            willpermute = false
            W && println("removed permutation by changing slicing: now ocode = ",ocode)
            W && println("... for expr = ",expr)
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

    #==================== ASSEMBLE ====================#

    needsizes = false
    slistex = :(($(slist...) ,)) # canonical size list -- expressions like size(A,d)
    szex = :(($(szall...) ,))    # simply sz[d], needsizes=true ensures sz = slistex at end

    ## 1. RESHAPE CONTAINER
    ex = inn

    if willshapeinput
        needsizes |= colonise!(isz, slist, szall)
        iszex = :(($(isz...) ,))
        ex = :( reshape($ex, $iszex) )
    end

    ## 2. GLUE SLICES
    havecopied = false
    if willglue
        ex = :(glue($ex, $icode))
        havecopied = true
    elseif willstaticglue
        ex = :( static_glue($ex) )
    end

    V && println("steps 1 & 2:  ex = ",ex)

    if sign == :(:=) || sign == :(==)
        V && println("... working forwards ...")

        ## 3. PERMUTEDIMS
        if willpermute
            if willstaticslice # then we cannot use lazy permutedims, including transpose
                ex = :( permutedims($ex, $perm) )
                havecopied = true
            elseif usingstrided
                ex = :( strided_permutedims($ex, $perm) ) # this is lazy
                # when A::DenseArray all is OK. If not it falls back, but may be wrong about views?
            elseif perm==(2,1)
                ex = :( transpose($ex) )
            else
                ex = :( permutedims($ex, $perm) )
                havecopied = true
            end
            V && println("step 3:  ex = ",ex)
        end

        if havecopied && sign == :(==)
            throw(ArgumentError(string("@shape can't figure out how to do what you ask without copying... ",
                "the problem might be gluing, or might be permutedims. using Strided may help")))
        end

        needcopy = !havecopied && sign == :(:=)

        ## 4. SLICE, OR REDUCE
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
            V && println("step 4 static:  ex = ",ex)

        elseif willslice
            if needcopy
                ex = :( slicecopy($ex, $ocode) )
                havecopied = true
            else
                ex = :( sliceview($ex, $ocode) )
            end
            V && println("step 4 slice:  ex = ",ex)

        elseif willreduce
            redfun = esc(redfun)
            ex = :( dropdims( $redfun($ex, dims = $odims), dims = $odims) )
            havecopied = true
            V && println("step 4 reduction:  ex = ",ex)
        end

        ## 5. RESHAPE
        if willshapeout
            needsizes |= colonise!(osz, slist, szall) # simplify osz
            if osz == [Colon()]
                ex = :( vec($ex) )
            else
                oszex = :(($(osz...) ,))
                ex = :( reshape($ex, $oszex) )
            end
            V && println("step 5:  ex = ",ex)
        end

        if !havecopied && sign == :(:=)
            ex = :( copy($ex) )
            havecopied = true
        end

        if hasoutputname # only uncertain in forward case, remember
            ex = :( $outn = $ex )
        end

    elseif sign == :(=) && willreduce
        V && println("... one more step forwards, then reduce!() ...")

        ## 3. PERMUTEDIMS
        if willpermute
            # only difference is that willstaticslice = false
            if usingstrided
                ex = :( strided_permutedims($ex, $perm) ) # this is lazy
            elseif perm==(2,1)
                ex = :( transpose($ex) )
            else
                ex = :( permutedims($ex, $perm) )
                havecopied = true
            end
            V && println("step 3:  ex = ",ex)
        end

        rout = outn

        ## 5. UN-RESHAPE
        if willshapeout
            rout =  :( reshape($rout, sz) ) # resize FROM osz to sz, the canonical sizes alla oflat TODO is this right? outer sz only?
            needsizes = true  # this surely needs sz to be defined

            V && println("step 5 backward:  rout = ",rout)
        end

        redfun = esc(redfun) # already has ! added
        ex = :( $redfun($rout, $ex) )
        V && println("step 4 reduction:  ex = ",ex)


    elseif sign == :(=)
        V && println("... now working backwards for in-place output ...")

        rout = outn

        ## 5. UN-RESHAPE
        if willshapeout
            rout =  :( reshape($rout, sz) ) # resize FROM osz to sz, the canonical sizes alla oflat TODO is this right? outer sz only?
            needsizes = true  # this surely needs sz to be defined

            V && println("step 5 backward:  rout = ",rout)
        end

        ## 4. UN-SLICE
        if willstaticslice
            rout = :( static_glue( $rout ) ) # TODO make an example that does this
            # note that this mutates the SArray slices, is that weird?
        elseif willslice
            throw(ArgumentError("@shape can't write in-place to slices right now, unless they are StaticArrays"))
            # Or better, you should NOT be working backwards at this point, you should write slices into mutable array
            # but this will be a bit of a mess to organise.
        end

        ## 3. PERMUTEDIMS!
        if willpermute
            if usingstrided
                ex = quote
                    strided_permutedims!($rout, $ex, $perm)
                    $outn
                end
            else
                ex = quote
                    permutedims!($rout, $ex, $perm)
                    $outn
                end
            end
            V && println("step 3 backward:  ex = ",ex)

        else # throw out some steps...
            # best case is that we had nothing but reshaping -- go back to beginning!
            if !willslice && !willstaticslice && !willglue && !willstaticglue
                W && println("step 3' throwing out ex = ",ex, "  and rout = ", rout, "  to go back to inn = ", inn)
                W && println("... had needsizes = ",needsizes," -> false, and mustcheck = ",mustcheck, " -> true")
                ex = :( copyto!($outn, $inn) )
                needsizes = false
                mustcheck = true
                W && println("... finally ex = ",ex)
                W && println("... for expr = ",expr)

            # second-best case is that we did some gluing but no slicing -- ignore rout
            elseif !willslice && !willstaticslice
                W && println("step 3'' throwing away rout = ", rout, "  to go back to ex = ", ex)
                W && println("... had needsizes = ",needsizes," -> false, and mustcheck = ",mustcheck, " left alone")
                ex = :( copyto!($outn, $ex) )
                needsizes = false # not very confident about this
                W && println("... finally ex = ",ex)
                W && println("... for expr = ",expr)

            # third-best... does this happen? what was I thinking? TODO maybe think more about what stage you roll back to
            else
                W && println("step 3''' ... got 3rd case! does this happen? rout = ", rout, " and ex = ",ex)
                ex = :( copyto!($rout, $ex) )
                W && println("... finally ex = ",ex)
            end
        end
    end

    #==================== FINALISE ====================#

    if needsizes
        ex = quote
            local sz = $slistex
            $ex
        end
    end

    if length(schecks) > 0
        chex = schecks[1]
        for ch in Iterators.drop(schecks,1)
            chex = :( $chex && $ch )
        end
    end

    if mustcheck && length(schecks) > 0
        ex = quote
            if !$chex
                throw(DimensionMismatch("$macroname failed explicit size checks"))
            end
            $ex
        end
    end

    return ex
end


############################### HELPER FUNCTIONS ################################

function findcheck(i::Symbol, ind::Vector, side="left", macroname="@shape")
    d = findfirst(isequal(i), ind)
    d isa Nothing && throw(ArgumentError("$macroname can't find index $i on $side hand side"))
    return d
end

function findcheck(i::Expr, ind::Vector, side="left", macroname="@shape")
    throw(ArgumentError("$macroname doesn't know what to do with $i, possibly nested brackets?"))
end

function push_checks!(checks::Vector, s1v, s2, s3=nothing)
    if @capture(s1v, Val(s1_))
    else s1 = s1v
    end
    nn = filter(!isequal(nothing), [s1, s2, s3] )
    if length(nn) == 3
        push!(checks, :( $(nn[1]) == $(nn[2]) == $(nn[3]) ) )
    elseif length(nn) == 2
        push!(checks, :( $(nn[1]) == $(nn[2]) ) )
    end
end

function getsize(name::Union{Expr,Symbol}, d::Union{Int,Nothing}, di::Union{Int,Nothing}=nothing, sign=:(=))
    # println("     ",name," d=",something(d,"nothing")," di=",something(d,"nothing")," sign ",sign)
    if sign == :(=) # getsize() gets given sign only when called on output array
        if d isa Int
            :( size($name, $d) )
        elseif di isa Int
            :( size(first($name), $di) )
        end
    else
        nothing
    end
end

function colonise!(isz, slist::Vector, szall::Vector)
    if isz == szall
        # example from tests:  @pretty @shape g[(b,c),x,y,e] := bcde[b,c,(x,y),e] x:2;
        V && println("colonise! replaced isz/osz= ",isz, " with just sz... because that's what it is")
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
            V && println("colonise! replaced isz/osz[$n]= ",sprod, " with just : because really sz[$d]=:")
        elseif nsz[1] == N
            isz[n] = Colon()
            needsizes = false
            V && println("colonise! replaced isz/osz[$n]= ",sprod, " with just : because it containts all N=$N of the sz[d]")
        end
    end
    return needsizes
end


############################### DATA FUNCTIONS ################################

if VERSION < v"1.1.0"
    include("eachslice.jl")
end

function sliceview(A::AbstractArray{T,N}, code::Tuple, sizes=nothing) where {T,N}
    if code == (:,*)
        collect(eachcol(A))
    elseif code == (*,:)
        collect(eachrow(A))
    elseif count(isequal(*), code) == 1
        collect(eachslice(A, dims = findfirst(isequal(*), code)))
    else
        iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(A,d) : Ref(:), Val(N))...)
        [ view(A, i...) for i in iter ]
    end
end
# sliceview(A::AbstractArray{T,2}, code::Tuple{Colon,typeof(*)}, sizes=nothing) where {T} = collect(eachcol(A))
# sliceview(A::AbstractArray{T,2}, code::Tuple{typeof(*),Colon}, sizes=nothing) where {T} = collect(eachrow(A))

function slicecopy(A::AbstractArray{T,N}, code::Tuple, sizes=nothing) where {T,N}
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(A,d) : Ref(:), Val(N))...)
    [ maybecopy(A[i...]) for i in iter ]
end

maybecopy(A) = A # overloaded for strided views

glue(A::AbstractArray{IT,N}, code::Tuple, sizes=nothing) where {IT,N} = cat_glue(A, code, sizes)
# this separation is only so that I can test mine against JuliennedArrays easily
@inline function cat_glue(A::AbstractArray{IT,N}, code::Tuple, sizes=nothing) where {IT,N}
    if code == (:,*)
        reduce(hcat, A)
    elseif code == (*,:)
        reduce((x,y) -> vcat(maybetranspose(x),maybetranspose(y)), A)
    elseif count(isequal(*), code) == 1 && code[end] == (*)
        reduce((x,y) -> cat(x,y; dims = length(code)), A)
    else
        throw(ArgumentError("Don't know how to glue code = $(pretty(code)) with cat, try using JuliennedArrays"))
        # now that glue! works, you could just use that?
        glue!(Array{eltype(first(A))}(undef, final_size), A, code)
    end
end

maybetranspose(x::AbstractVector) = transpose(x)
maybetranspose(x) = x

function glue!(B::AbstractArray{T,N}, A::AbstractArray{IT,ON}, code::Tuple, sizes=nothing) where {T,N,IT,ON}
    iter = Iterators.product(ntuple(d -> (code[d]==*) ? axes(B,d) : Ref(:), Val(N))...)
    for i in iter
        # B[i...] .= A[decolonise(i)...]
        copyto!(view(B, i...), A[decolonise(i)...] )
    end
    B
end

# decolonise(i::Tuple) = filter(!isequal(:), i) # sadly not

# @inline function decolonise(i::Tuple, valn::Val{N}) where {N}
#     ind = findall(i->i!==(:), i)
#     ntuple(j -> i[ind[j]]::Int, valn)
# end
# @code_warntype decolonise((1,2,:,3), Val(3))
# using BenchmarkTools
# @btime decolonise((1,2,:,3), Val(3)) # 334.288 ns (7 allocations: 336 bytes)

@generated function decolonise(i::Tuple)  # thanks to @Mateusz Baran
    ind = Int[]
    for k in 1:length(i.parameters)
        if i.parameters[k] != Colon
            push!(ind, k)
        end
    end
    Expr(:tuple, [Expr(:ref, :i, k) for k in ind]...)
end

# @code_warntype decolonise((1,2,:,3), Val(3))
# using BenchmarkTools
# @btime decolonise((1,2,:,3), Val(3)) # 0.035 ns (0 allocations: 0 bytes)

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
    str = string(ex)

    str = replace(str, r"\(\w+\.(\w+)\)" => s"\1") # remove module names
    str = replace(str, r"\(\w+\.(\w+!)\)" => s"\1")

    str = replace(str, "Colon()" => ":")

    str = replace(str, r"\n(\s+.+=#)" => "") # remove line references
end

pretty(tup::Tuple) = replace(string(tup), "Colon()" => ":")

end # module
