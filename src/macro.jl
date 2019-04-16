export @cast, @cast!, @reduce, @reduce!

"""
    @cast Z[i,j,...] := f(A[j,k,...])  options

Macro for broadcasting, reshaping, and slicing of arrays in index notation.
Understands the following things:
* `A[i,j,k]` is a three-tensor with these indices.
* `B[(i,j),k]` is the same thing, reshaped to a matrix. Its first axis (the bracket) is indexed
  by `n = i + (j-1) * N` where `i ∈ 1:N`. This may also be written `B[i\\j,k]` or `B[i⊗j,k]`.
* `C[k][i,j]` is a vector of matrices.
* `D[j,k]{i}` is an ordinary matrix of `SVector`s, which may be reinterpreted from `A[i,j,k]`.
* `E[i,_,k]` has two nontrivial dimensions, and `size(E,2)==1`. On the right hand side
  (or when writing to an existing array) you may also write `E[i,3,k]` meaning `view(E, :,3,:)`,
  or `E[i,\$c,j]` to use a variable `c`. Fixing inner indices, like `C[k][i,_]`, is not allowed.
* `F[i,-j,k]` means roughly `reverse(F, dims=2)`.
* `g(x)[i,j,k]` is allowed, and `g(x)` should be evaluated only once.
* `H[i,j]'` conjugates each element, equivalent to `H'[j,i]` which is the
  conjugate-transpose of the matrix.
* `J[i,i]` means `diag(J)[i]`, but only for matrices: `K[i,i,k]` is an error.

The left and right hand sides must have all the same indices,
with the sole exception of `J[i,i]`.
See `@reduce` and `@mul` for related macros which can sum over things.

If several tensors appear on the right hand side, then this represents a broadcasting operation,
and the necessary re-orientations of axes are automatically inserted.
Some attempt is made to shield scalar functions from broadcasting,
e.g. `@cast A[i] := log(B[i]) / log(2)` will avoid `log.(2)` and evaluate `log(2)` once.
But this is imperfect, confirm with `@pretty @cast ...` if concerned.

The following actions are possible:
* `=` writes into an existing array, `copyto!(Z, ...)`.
* `:=` creates a new object... which may or may not be a view of the input:
* `==` insists on a view of the old object (error if impossible), and `|=` insists on a copy.
* `=>` creates an anonymous function, for which the output must be on the right.
   For example `@cast A[i] + B[j]' => Z[i,j]` gives `(A,B) -> A .+ B'`.

Re-ordering of indices `Z[k,j,i]` is done lazily with `PermutedDimsArray(A, ...)`.
Reversing of an axis `F[i,-j,k]` is also done lazily, by `Reverse{2}(F)` which makes a `view`.
Using `|=` (or broadcasting) will produce a simple `Array`.

Options can be specified at the end (if several, separated by `,` i.e. `options::Tuple`)
* `i:3` supplies the range of index `i`. Variables and functions like `j:Nj, k:length(K)`
  are allowed.
* `assert` or `!` will turn on explicit dimension checks of the input.
  (Providing ranges may also turn these on, but imperfectly, confirm with `@pretty @cast ...`.)
* `cat` will glue slices by things like `hcat(A...)` instead of the default `reduce(hcat, A)`,
  and `lazy` will instead make a `VectorOfArrays` container.
* `nolazy` disables `PermutedDimsArray` and `Reverse` in favour of `permutedims` and `reverse`,
  and `Diagonal` in favour of `diagm` for `Z[i,i]` output.
* `strided` will place `@strided` in front of broadcasting operations,
  and use `@strided permutedims(A, ...)` instead of `PermutedDimsArray(A, ...)`.
  For this you need to install and load the package: `using Strided`.

Static slices `D[j,k]{i}` need `using StaticArrays`, and to create them you should give all
slice dimensions explicitly. You may write `D[k]{i:2,j:2}` to specify `Size(2,2)` slices.
They are made most cleanly from the first indices of the input, i.e. this `D` from `A[i,j,k]`.
"""
macro cast(exs...)
    where = (mod=__module__, src=__source__, str=unparse("@cast", exs...))
    _macro(exs...; reduce=false, where=where)
end

"""
    @cast! Z[i...] := A[j...] opt

Variant of `@cast` which effectively runs `@check!()` on each tensor.
"""
macro cast!(exs...)
    where = (mod=__module__, src=__source__, str=unparse("@cast!", exs...))
    _macro(exs...; reduce=false, where=where, icheck=true)
end

"""
    @reduce A[i] := sum(j,k) B[i,j,k]             # A = vec(sum(B, dims=(2,3)))
    @reduce A[i] := prod(j) B[i] + ε * C[i,j]     # A = vec(prod(B .+ ε .* C, dims=2))
    @reduce A[i] = sum(j) exp( C[i,j] / D[j] )    # sum!(A, exp.(C ./ D') )

Tensor reduction macro:
* The reduction function can be anything which works like `sum(B, dims=(1,3))`,
  for instance `prod` and `maximum` and `Statistics.mean`.
* In-place operations `Z[j] = sum(...` will construct the banged version of the given function's name,
  which must work like `sum!(Z, A)`.
* The tensors can be anything that `@cast` understands, including gluing of slices `B[i,k][j]`
  and reshaping `B[i\\j,k]`. See `? @cast` for the complete list.
* If there are several tensors (or scalars) on the right, then this is a broadcasting operation.
* Index ranges may be given afterwards (as for `@cast`) or inside the reduction `sum(i:3, k:4)`.
* All indices appearing on the right must appear either within `sum(...)` etc, or on the left.


    F = @reduce sum(i,j)  B[i] + γ * D[j]         # sum(B .+ γ .* D')
    @reduce G[] := sum(i,j)  B[i] + γ * D[j]      # F == G[]

Complete reduction to a scalar output `F`, or a zero-dim array `G`.
`G[]` involves `sum(A, dims=(1,2))` rather than `sum(A)`.

    @reduce Z[k] := sum(i,j) A[i] * B[j] * C[k]  lazy, i:N, j:N, k:N

The option `lazy` replaces the broadcast expression with a `BroadcastArray`,
to avoid `materialize`ing the entire array before summing. In the example this is of size `N^3`.

The option `strided` will place `@strided` in front of the broadcasting operation.
You need `using Strided` for this to work.
"""
macro reduce(exs...)
    where = (mod=__module__, src=__source__, str=unparse("@reduce", exs...)) # wtf?
    _macro(exs...; reduce=true, where=where)
end

"""
    @reduce! Z[j] := sum(i,k) A[i,j,k]

Variant of `@reduce` which effectively runs `@check!()` on each tensor.
"""
macro reduce!(exs...)
    where = (mod=__module__, src=__source__, str=unparse("@reduce!", exs...))
    _macro(exs...; reduce=true, where=where, icheck=true)
end

#=

This much is now per RHS term. None need exist outside the function:

nameA, indA, indAsub
sizeA      -- known & used for fixing ranges
getB, numB -- for view(A, get), and number of fixed indices
sizeC      -- after reshaping outer indices, in terms of sz_i
codeD      -- for gluing
indEflat   -- after gluing
negF       -- done before permutedims
shiftG     -- ditto, but not yet written
permH      --
codeI      -- to orient, if lacking some indices.

Of these things from LHS, only really store + canon need to be passed to RHS function inputex()
via walker(). But many need to be passed from readleft() to output, packaged into:
outUZ = (redUind, negV, codeW, sizeX, getY, numY, indZ, sizeZ, nameZ)

flags      -- todo, to-done, and options
store      -- information from parse! with sizes etc.
canon      -- canonical list of indices
canonsize  -- filled with size(A,2) etc, from store, at most one (:)

redUind    -- if reducing, this is done before slicing
negV       -- directions to reverse, done here.
codeW      -- for slicing. sizeWstatic is Size() of the slice.
sizeX      -- after slicing, container size, in sz_i
getY, numY -- for view(Z, get) in-place, and number fixed
sizeZ      -- for final reshape, in sz_i
nameZ, indZ, indZsub, indZred -- given LHS

=#

function _macro(exone, extwo=nothing, exthree=nothing;
    reduce=false, icheck=false, where=nothing, recurse=false)

    flags = Set{Symbol}()

    if reduce
        #===== parse @reduce input =====#

        if @capture(exone, left_ := redfun_(redind__) )     # Z[i] := sum(j) A[i] * B[j]
            right = extwo
            options = exthree
            push!(flags, :reduce)
            V && @info "partial reduce" left right redfun Tuple(redind) options
        elseif @capture(exone, left_ = redfun_(redind__) )
            right = extwo
            options = exthree
            push!(flags, :reduce)
            push!(flags, :inplace)
        elseif @capture(exone, left_ |= redfun_(redind__) )
            right = extwo
            options = exthree
            push!(flags, :reduce)
            push!(flags, :mustcopy)

        elseif @capture(exone, redfun_(redind__) )          # sum(i,j) A[i] * B[j]
            right = extwo
            options = exthree
            push!(flags, :reduce)
            push!(flags, :scalar)
            V && @info "full reduce" right redfun Tuple(redind)

        else
            throw(MacroError("don't know what to do with $exone", where))
        end

    else
        #===== parse @cast input =====#

        if @capture(exone, left_ := right_ )                # Z[i,j] := A[i] * B[j]
        elseif @capture(exone, left_ = right_ )
            push!(flags, :inplace)
        elseif @capture(exone, left_ |= right_ )
            push!(flags, :mustcopy)
        elseif @capture(exone, left_ == right_ )
            push!(flags, :mustview)
                                                            # A[i,j] -> Z[j][i]
        elseif @capture(exone, right_ -> left_ ) ||  @capture(exone, right_ => left_ )
            push!(flags, :anonfunc)
        else
            throw(MacroError("@cast doesn't know what to do with $exone", where))
        end
        V && @info "no reduction" left right
        options = extwo
        redind = []
        redfun = identity
        exthree == nothing || throw(MacroError("@cast doesn't know what to do with $exthree", where))

    end

    #===== parse LHS to get canonical list =====#

    store = SizeDict()
    canon, outUZ, nameZ, checkZ = readleft(left, redind, flags, store, icheck, where)

    # parse options both to look for keywords and sizes
    @capture(options, (optvec__,)) || (optvec = Any[options])
    optind, _,_,_ = parse!(store, nothing, [], optvec; allowranges=true, flags=flags)

    if count(i -> i != nothing, setdiff(optind, canon)) > 0
        str = join(something.(setdiff(optind, canon), "nothing"), ", ")
        m_error("attempting to ignore unrecognised options: $str", where)
    end

    #===== parse and process RHS =====#

    outex = MacroTools.@q begin end

    if @capture(right, AA_[ii__] ) || @capture(right, AA_{ii__} )
        newright = walker(outex, right, canon, flags, store, icheck, where)
    else
        newright = MacroTools.prewalk(
            x -> walker(outex, x, canon, flags, store, icheck, where), right)
        push!(flags, :broadcast)
    end

    notseen = setdiff(canon, unique(store.seen))
    isempty(notseen) || throw(MacroError("did not see index $(join(notseen, ", ")) on the right", where))

    #===== almost done =====#

    packagecheck(flags, where) # disabled for issue #2

    canonsize = sizeinfer(store, canon, where, true)

    V && @info "before in/out choice" store flags Tuple(canonsize)

    if :inplace in flags

        #===== in-place output  =====#

        if checkZ != nothing
            push!(outex.args, checkZ)
        end

        inout = outputinplace(newright, outUZ, redfun, canonsize, canon, flags, store, nameZ, where)
        push!(outex.args, inout)
        if :finalres in flags
            push!(outex.args, nameZ)
        end

    else
        #===== out-of-place output  =====#

        if :broadcast in flags
            newright = Broadcast.__dot__(newright)
            if (:reduce in flags) && (:lazy in flags)
                newright = makelazy(newright)
            end
            if :strided in flags
                newright = :( Strided.@strided $newright )
            end
        end
        finalright = outputnew(newright, outUZ, redfun, canonsize, canon, flags, store, where)
        push!(outex.args, :( $nameZ =  $finalright ) )

        if checkZ != nothing
            push!(outex.args, checkZ)
            push!(outex.args, nameZ)
        end
    end

    #===== finalise =====#

    if :needsize in flags
        szcanon = Any[ Symbol(:sz_,i) for i in canon ]
        pushfirst!(outex.args, :( local ($(szcanon...),) = ($(canonsize...),) ) )
    end

    if :assert in flags || :(!) in flags || check_options.size
        for ch in store.checks
            pushfirst!(outex.args, ch)
        end
    end

    for tex in store.topex
        pushfirst!(outex.args, tex)
    end

    if length(outex.args) == 1
        outex = outex.args[1]
    end

    if recurse==true # only for @reduce inside something
        indZ = outUZ[end-1]
        # @info "recursive" outex indZ
        return outex, indZ
    end

    if :anonfunc in flags
        leftex = anoninput(store.rightnames, where)
        outex = :( $leftex -> $outex ) # TODO make this not introduce begin end
        return esc(outex)
    else
        return esc(outex)
    end
end


"""
    canon, outUZ, nameZ, checkZ = readleft(left, redind, flags, store, icheck, where)

outUZ = (redUind, negV, codeW, sizeX, getY, numY, indZ, sizeZ)
are things passed to output construction.
"""
function readleft(left, redind, flags, store, icheck, where)

    if @capture(left, Z_[outer__][inner__]) ||  @capture(left, [outer__][inner__])
        push!(flags, :slice)
    elseif @capture(left, Z_[outer__]{inner__}) || @capture(left, [outer__]{inner__})
        push!(flags, :staticslice)
    elseif @capture(left, Z_[outer__]) || @capture(left, [outer__])
        inner = []
    elseif left==nothing
        @assert :scalar in flags
        inner = []
        outer = []
        Z = nothing
    else
        throw(MacroError("readleft doesn't know what to do with $left", where))
    end

    if Z == nothing
        nameZ = gensym(:Z) # no output name
        if :inplace in flags
            throw(MacroError("can't write in-place into nowhere!", where))
        end
    else
        nameZ = Z
    end

    if !(:inplace in flags)
        Z = nothing # tells parse! not to think about size(Z,...)
    end
    flat12, getY, sizeZ, negV = parse!(store, Z, outer, inner; allowranges=true) # true allows A[i]{j:3}
    redUind, _,_,_ = parse!(store, nothing, [], redind; allowranges=true) # true means sum(i:3) allowed

    if length(flat12)-length(inner) == 2 && flat12[end] == flat12[end-1] # Z[i,i] special case
        length(getY) == 2 || throw(MacroError("can't fix output index with Diagonal output", where)) # because I'm lazy, this would not be impossible
        push!(flags, :diagleft)
        pop!(flat12) # remove index from flat list
        pop!(getY)
        pop!(sizeZ)
    end

    canon = vcat(flat12, redUind) # order here is [inner, outer, reduced]

    checkrepeats(flat12, " on left hand side", where)
    checkrepeats(canon, " in reduction function", where)

    codeW = repeat(Any[*], length(canon) - length(redUind))
    codeW[1:length(inner)] .= (:)

    indW = canon[1:length(inner)] # inner without minuses etc

    indX = flat12[length(inner)+1:end] # flattened outer without minuses, also without fixed
    sizeX = Any[ Symbol(:sz_,i) for i in indX ] # only for in-place

    numY = count(!isequal(:), getY) # the number of fixed indices in output

    if length(getY) - numY != length(sizeX)
        push!(flags, :backshape) # whether reshaping of view(Z, getY) is needed, only in-place
    end

    if (length(sizeZ) + numY) != (length(canon) - length(inner) - length(redUind))
        push!(flags, :outshape) # whether final reshaping is needed, for := case
    end

    indZ = [] # like indX, but with fixed indices re-inserted, for :named / :axis
    iz = 1
    for g in getY
        if g == (:)
            push!(indZ, indX[iz])
            iz += 1
        else
            push!(indZ, g)
        end
    end

    checkZ = nothing
    if icheck
        checknow = check_one(:($nameZ[$(outer...)]), where)
        if check_options.size
            checkZ = checknow # check!(...) to be inserted
        end
    end

    outUZ = (redUind, negV, codeW, indW, sizeX, getY, numY, indZ, sizeZ)
    V && @info "readleft" Tuple(canon) outUZ nameZ
    return canon, outUZ, nameZ, checkZ
end

"""
    walker(outex, x, canon, flags, store, icheck, where)

Called by `MacroTools.prewalk` on RHS, finds tensors & pushes `:( sym = inputex(this) )` into
`outex.args`, and then replaces them with `sym`.
"""
function walker(outex, ex, canon, flags, store, icheck, where)

    # allow @reduce inside RHS, by simply calling the entire macro first
    if @capture(ex, @reduce(redex__))
        Rsym = gensym(:R)
        Rval, Rind = _macro(redex...; reduce=true, where=where, recurse=true)

        # sew the result of that into outex
        push!(outex.args, :(local $Rsym = $Rval ) )

        # construct name for resulting tensor, for outer @cast to further process
        ex =  :( $Rsym[$(Rind...)] )

    elseif @capture(ex, @mul(mulex__))
        Msym = gensym(:M)
        Mval, Mind = _mul(mulex...; where=where, recurse=true)

        push!(outex.args, :(local $Msym = $Mval ) )

        ex =  :( $Msym[$(Mind...)] )
    end

    # find any indexed tensor, process it, and replace it
    if @capture(ex, A_[ij__][kl__] ) || @capture(ex, A_[ij__]{kl__} ) || @capture(ex, A_[ij__] )

        push!(store.rightnames, A) # used for @cast A[i,j] -> B[i\j]

        # if we have f(x)[i,j] then we should evaluate f just once, e.g.
        # @pretty @cast A[i]{j:5} |= rand(15)[i\j]
        if !isa(A, Symbol)
        # if isa(A, Symbol) || @capture(A, Astruct_.field_)
        #     @show A
        # else
            Atop = gensym(:A)
            push!(store.topex, :(local $Atop = $A ) )
            A = Atop
        end

        Aval = inputex(A, ex, canon, flags, store, icheck, where)

        if isa(Aval, Symbol)
            ex = Aval
        else
            Asym = gensym(:A)
            push!(outex.args, :(local $Asym = $Aval) )
            ex =  Asym
        end
        V && @info "walker" ex Aval

    # catch A[i,j]' as element-wise conj
    elseif @capture(ex, arg_')
        ex = :( Base.conj($arg) )

    # try to protect log(2) etc from @. , but not log(A[i])
    elseif @capture(ex, f_(x_Symbol) ) || @capture(ex, f_(x_Int) ) || @capture(ex, f_(x_Float64) )
        fxsym = gensym(:fx)
        push!(outex.args, :( local $fxsym = $ex ))
        ex = fxsym
    end

    return ex
end


"""
    inputex(:A, :( A[i,j][k] ), target, flags, store, icheck, where)

Figures out all the steps needed to transform the given tensor to a boring one,
aligned with the given target, and returns the necessary expression.
Writes sizes which can be read from `A` into `store`, and necessary reshapes in terms of `sz_i`.

Now `target` need not be `== canon`, since `sz_i` is independent of that.

Now needs explicitly the name of tensor `A`,
so that when `inex = rand(2,3)[i,j]` this is not evaluated twice.
"""
function inputex(A, inex, target, flags, store, icheck, where)

    if @capture(inex, Aa_[outer__][inner__])
        glue = :yes
    elseif @capture(inex, Aa_[outer__]{inner__})
        glue = :static
    elseif @capture(inex, Aa_[outer__])
        inner = []
        glue = :no
    else error("inputex should not have been called")
    end

    flatE, getB, _, negF = parse!(store, A, outer, inner)

    append!(store.seen, flatE)

    ex = A
    if icheck
        ex = check_one(:($A[$(outer...)]), where)           # @check!
    end

    numB = count(!isequal(:), getB)
    if numB > 0                                             # A[_,i]
        if numB == length(getB) || :nolazy in flags
            needview!(getB) # replace _ with 1 if any
            ex = :( $ex[$(getB...)] )
        elseif needview!(getB) # then at least one index fixed and not _
            ex = :(view($ex, $(getB...) ))
        else
            ex = :(TensorCast.rview($ex, $(getB...) )) # really a reshape
        end
    end

    sizeC = Any[ Symbol(:sz_, i) for i in flatE[length(inner)+1 : end] ]

    if (length(sizeC) + numB) != length(outer)              # A[i\j]
        sizeCex = :(($(sizeC...) ,))
        ex = :( reshape($ex, $sizeCex) )
        push!(flags, :needsize)
    end

    if length(flatE)-length(inner) == 2 && flatE[end] == flatE[end-1]  # A[i,i] special case
        if :nolazy in flags
            ex = :( TensorCast.diag($ex) ) # LinearAlgebra might not be loaded by caller
            push!(flags, :havecopied)
        else
            ex = :( TensorCast.diagview($ex) )
        end
        pop!(flatE) # remove repeated index from the end
    end

    checkrepeats(flatE, " in term $inex", where) # after dealing with diag story

    dirs = [ findcheck(i, target, where) for i in flatE ]

    if glue == :yes                                         # A[i][k]
        codeD = repeat(Any[*],length(flatE))
        codeD[1:length(inner)] .= (:)

        if codeD == [:,*] && dirs[1] > dirs[2] # then we can avoid a transpose
            codeD = [*,:]
            dirs = [dirs[2], dirs[1]]
        end
        # you could perform more elaborate versions of that, e.g. for this:
        # @pretty @reduce A[i\j,_] = sum(k) B[i,j][k]
        # however only copy_glue and julienne_glue understand arbitrary codes

        if :lazy in flags || :mustview in flags
            ex = :( TensorCast.lazy_glue($ex, $(codeD...,))  )

        elseif :cat in flags
            ex = :( TensorCast.cat_glue($ex, $(codeD...,))  )
            push!(flags, :havecopied)
        elseif :glue in flags
            ex = :( TensorCast.copy_glue($ex, $(codeD...,))  )
            push!(flags, :havecopied)
        elseif :julienne in flags
            ex = :( TensorCast.julienne_glue($ex, $(codeD...,))  )
            push!(flags, :havecopied)
        else
            ex = :( TensorCast.red_glue($ex, $(codeD...,))  )
            push!(flags, :havecopied)
        end

    elseif glue == :static                                  # A[i]{k}
        ex = :( TensorCast.static_glue($ex)  )
        push!(flags, :staticglue) # for packagecheck
    end

    perm = ntuple(identity, length(dirs))
    if dirs != sort(dirs)                                   # A[j,i]
        perm = Tuple(sortperm(dirs))
        if :nolazy in flags
            ex = :( permutedims($ex, $perm) )
        elseif :strided in flags
            ex = :( TensorCast.strided_permutedims($ex, $perm) )
        # elseif perm == (2,1)
        #     ex = :( transpose($ex) ) # now avoiding transpose because it's recursive
        else
            ex = :( PermutedDimsArray($ex, $perm) )
        end
    end

    for i in negF                                           # A[-i,j]
        d = invperm(perm)[findcheck(i, flatE, where)]
        if :nolazy in flags
            ex = :( reverse($ex, dims=$d) )
            push!(flags, :havecopied)
        else
            ex = :( TensorCast.Reverse{$d}($ex) )
        end
    end

    if length(flatE) != length(target) && dirs != 1:length(dirs) # A[i] + B[j]
        codeH = repeat(Any[*],length(target))
        codeH[dirs] .= (:)
        ex = :( TensorCast.orient($ex, $(codeH...,)) )
    end

    # if :strided in flags
    #     ex = :( Strided.@strided $ex )
    # end

    return ex
end

"""
     outputnew(newright, outUZ, redfun, canonsize, canon, flags, store, where)

For the case of `:=`, this constructs the expression to do reduction if needed,
and slicing/reshaping/reversing for LHS.

outUZ = (redUind, negV, codeW, indW, sizeX, getY, numY, indZ, sizeZ)
"""
function outputnew(newright, (redUind, negV, codeW, indW, sizeX, getY, numY, indZ, sizeZ),
        redfun, canonsize, canon, flags, store, where)

    ex = newright

    for ri in negV                                          # Z[-i,j]
        d = findcheck(ri, canon, where)
        ex = :( reverse($ex, dims=$d) )
    end

    if :reduce in flags && :scalar in flags                 # Z = @reduce sum(i) ...
        ex = :( $redfun($ex) )
    elseif :reduce in flags                                 # Z[i] := sum(j) ...
        rdims = Tuple([findcheck(i, canon, where) for i in redUind])
        if length(rdims)==1
            rdims = first(rdims)
        end
        ex = :( $redfun($ex, dims=$rdims) )
        if :outshape in flags && !(:slice in flags || :staticslice in flags)
            # then we need not dropdims, as reshape will handle it
        else
            ex = :( dropdims($ex, dims=$rdims) )
        end
    end

    if :slice in flags                                      # Z[i][k] :=
        if :mustcopy in flags # && !(:havecopied in flags) # make |= slightly stronger
            ex = :( TensorCast.slicecopy($ex, $(codeW...,)) )
            push!(flags, :havecopied)
        elseif :julienne in flags
            ex = :( TensorCast.julienne_slice($ex, $(codeW...,)) )
        else
            ex = :( TensorCast.sliceview($ex, $(codeW...,)) )
        end
    elseif :staticslice in flags                            # Z[i]{k} :=
        # codeW worked out already, but sizeWstatic must be done here
        # TODO sizeinfer!(store, canon, where, false)
        sizeWstatic = :( StaticArrays.Size($([store.dict[i] for i in indW]...)) )
        if :outshape in flags
            ex = :( TensorCast.static_slice($ex, $sizeWstatic, false) )
        else
            ex = :( TensorCast.static_slice($ex, $sizeWstatic) )
        end
    end

    if :outshape in flags                                   # Z[i\j, _, k]
        sizeZex = :(($(sizeZ...) ,))
        ex = :( reshape($ex, $sizeZex) )
        push!(flags, :needsize)
        for n in filter(!isequal(:), getY)
            n == 1 || n == :_ || throw(MacroError("can't fix output index to $n != 1, when creating a new array", where))
        end
    end

    if :mustcopy in flags && !(:havecopied in flags) && !(:broadcast in flags)
        ex = :( copy($ex) )                                 # Z[i] |= ...
    elseif :mustcopy in flags && :staticslice in flags
        ex = :( copy($ex) )

    elseif :mustview in flags && :havecopied in flags       # Z[i] == ...
        m_error("can't do what you ask without copying, sorry", where)
    elseif :mustview in flags && :broadcast in flags
        m_error("can't broadcast without copying, sorry", where)
    end

    if :diagleft in flags                                   # Z[i,i] := ...
        if :nolazy in flags
            ex = :( TensorCast.diagm(0 => $ex) )
        else
            ex = :( TensorCast.Diagonal($ex) )
        end
    end

    # if :strided in flags
    #     ex = :( Strided.@strided $ex )
    # end

    if :named in flags                                      # Z[i] := ... named
        ex = :( TensorCast.namedarray($ex, $(indZ...,) ) )
    # elseif :axis in flags
    #     ex = :( TensorCast.axisarray($ex, $(indZ...,) ) )
    end

    return ex
end

"""
    outputinplace(newright, outUZ, redfun, canonsize, canon, flags, store, nameZ, where)

For the case of `=` this figures out how to write RHS into LHS, in one of three ways:
* reduction `sum!(Z, newright)`
* broadcasting `@. Z[...] = newright`
* neither, `copyto!(Z, newright)`

No longer attempts to write `permutedims!(Z, A, ...)`, now just `copyto!(Z, PermutedDimsArray(A, ...))`.
Doesn't really need so many arguments...
"""
function outputinplace(newright, (redUind, negV, codeW, indW, sizeX, getY, numY, indZ, sizeZ),
        redfun, canonsize, canon, flags, store, nameZ, where)

    if :slice in flags
        throw(MacroError("can't write to sliced arrays in-place, for now", where))
    end
    if length(negV) > 0
        m_error("can't reverse axes of in-place output, try moving -$(negV[1]) to right hand side", where)
    end

    if :reduce in flags                                     # sum!(revleft, newright)

        if :broadcast in flags
            newright = Broadcast.__dot__(newright)
            if :lazy in flags
                newright = makelazy(newright)
            end
        end

        # working backwards
        revleft = nameZ

        if numY > 0
            if needview!(getY)
                revleft = :(view($revleft, $(getY...) ))
            else
                revleft = :(TensorCast.rview($revleft, $(getY...) )) # really a reshape
            end
            push!(flags, :finalres)
        end

        if :diagleft in flags
            revleft = :( TensorCast.diagview($revleft) )
        end

        if :backshape in flags
            sizeXex = :(($(sizeX...) ,))
            revleft = :( reshape($revleft, $sizeXex) )
            push!(flags, :needsize)
            push!(flags, :finalres)
        end

        if !endswith(string(redfun), '!')
            redfun = Symbol(redfun, '!')
        end

        ex = :( $redfun($revleft, $newright) )

    elseif :broadcast in flags                              # @. revleft[...] = newright

        # working backwards
        revleft = nameZ

        if numY > 0 # when getY has only : and 1, and backshape, then you could skip this
            if needview!(getY)
                revleft = :(view($revleft, $(getY...) ))
            else
                revleft = :(TensorCast.rview($revleft, $(getY...) )) # really a reshape
            end
            push!(flags, :finalres)
        end

        if :diagleft in flags
            revleft = :( TensorCast.diagview($revleft) )
        end

        if :backshape in flags
            sizeXex = :(($(sizeX...) ,))
            revleft = :( reshape($revleft, $sizeXex) )
            push!(flags, :needsize)
            push!(flags, :finalres)
        end

        bc = Broadcast.__dot__(newright)
        ex = :( $revleft .= $bc )

    else                                                    # copyto!(revleft, newright)

        # working backwards
        revleft = nameZ

        if numY > 0
            if needview!(getY)
                revleft = :(view($revleft, $(getY...) ))
            else
                revleft = :(TensorCast.rview($revleft, $(getY...) )) # really a reshape
            end
            push!(flags, :finalres)
        end

        if :diagleft in flags
            revleft = :( TensorCast.diagview($revleft) )
        end

        ex = :( $copyto!($revleft, $newright) )

    end

    if :strided in flags
        ex = :( Strided.@strided $ex )
    end

    return ex
end

"""
    anoninput(store.rightnames)

Given a vector of names captured from `A_[i...]`, returns ex needed for `(A,B) -> ...`
"""
function anoninput(rightnames, where)
    for A in rightnames
        isa(A,Symbol) || throw(MacroError("can't use $A as anonymous function input", where))
        # TODO make this understand that $B means interpolate not arg
    end
    if length(rightnames) == 1
        return rightnames[1]
    else
        return :( ($(rightnames...),) )
    end
end

function packagecheck(flags, where)
    where === nothing && return
    # now check in caller's scope?
    if :staticslice in flags || :staticglue in flags
        isdefined(where.mod, :StaticArrays) || m_error("can't use static arrays without using StaticArrays", where)
    end
    if :strided in flags
        isdefined(where.mod, :Strided) || m_error("can't use option strided without using Strided", where)
    end
    if :julienne in flags
        isdefined(where.mod, :JuliennedArrays) || m_error("can't use option julienne without using JuliennedArrays", where)
    end
    if :named in flags
        isdefined(where.mod, :NamedArrays) || m_error("can't use option named without using NamedArrays", where)
    end
    # if :axis in flags
    #     isdefined(where.mod, :AxisArrays) || m_error("can't use option axis without using AxisArrays", where)
    # end
end

using LazyArrays # for BroadcastArray

using LinearAlgebra  # for diag()

"""
    makelazy(bc)

Takes the result of `Broadcast.__dot__()` and converts it to have a `LazyArrays.BroadcastArray`.
"""
makelazy(sym::Symbol) = sym

# TODO replace this with lazy.() trick
function makelazy(bc::Expr)
    V && @info "before LazyArrays" bc

    @assert bc.head == :(.)      # always a dot
    oprator = bc.args[1]         # is the first operator
    bc.args[2]                   # is its tuple of arguments
    @assert length(bc.args) == 2 # and there's nothing more
    @assert bc.args[2].head == :tuple
    arguments = bc.args[2].args  # is the args of first op

    # lazybc = Expr(:call, :(LazyArrays.BroadcastArray), oprator, arguments...)
    lazybc = Expr(:call, :(TensorCast.BroadcastArray), oprator, arguments...)


    V && @info "after LazyArrays" lazybc
    return lazybc
end
