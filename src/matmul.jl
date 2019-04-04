
export @mul

#= TODO:

# vector times matrix:
@mul Z[i] := rand(2)[j] * rand(2,2)[j,i]

# from a typo in readme, but should work:
X = rand(2,3,4); Y = rand(3,2,4)
@mul W[β,i,j] := X[i,k,β] * Y[k,i,β]

# add @check! version

# shouldn't this be an error?
@pretty @mul A[n][i,k] = B[i,j,n] * C[k,j,n]

# Mason Potter's example:
W = rand(2,2,2,2); M = rand(2,2,2);
@reduce N[σ, b\a, b′\a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′]
@mul N2[σ, b\a, b′\a′] := W[σ,σ′,b,b′] * M[σ′,a,a′]
sort(vec(N)) ≈ sort(vec(N2)) # true

=#

"""
    @mul A[i,j] := B[i,k,k′] * C[k,k′,j]

Matrix multiplication macro. This expects two tensors on the right,
whose shared index (or indices) will be sumed over.

    @mul A[_,i][j] := B[i\\k] * C[j,2,k]

Each tensor factor will be processed in the same way as for `@cast` / `@reduce`,
allowing for slicing and reshaping, permuting indices, and fixing their values.
(The same tuple of options understood by `@cast` may be given after the expression, too.)
But the right hand side must be simply a product of two tensors, nothing else.
Thus it is both more limited, and more general, than `@tensor` / `@einsum`.

The same operations can be written `@reduce A[i,j] := sum(k) B[i,k] * C[k,j]`,
but this broadcasts out to a 3-tensor before reducing,
rather than just calling `A = B * C`.

The in-place form `@mul A[i,j] = B[i,k] * C[k,j]` calls instead `mul!(A,B,C)`.

    @mul A[i,j,n] := B[i,k,n] * C[k,j,n]

Batched matrix multiplication. Right now done by slicing each tensor and mapping `*` or `mul!`
over the slices. But it should be easy to hook up to BatchedRoutines.jl or something.
"""
macro mul(exs...)
    where = (mod=__module__, src=__source__, str=unparse("@mul", exs...))
    _mul(exs...; icheck=false, where=where)
end

#=
"""
    @mul! A[i,j] := B[i,k] * C[k,j]

Variant of `@mul` which effectively runs `@check!()` on each tensor.
"""
macro mul!(exs...)
    where = (mod=__module__, src=__source__, str=unparse("@mul", exs...))
    _mul(exs...; icheck=true, where=where)
end
=#

function _mul(ex, options=nothing; icheck=false, where=where)
    flags = Set{Symbol}()

    # @warn "The macro @mul has bugs!" maxlog=1

    #===== parse basic expression =====#

    if @capture(ex, left_ = mid_ * right_ )
        push!(flags, :inplace)
    elseif @capture(ex, left_ := mid_ * right_ )
    elseif @capture(ex, left_ |= mid_ * right_ )
        push!(flags, :mustcopy) # this will only affect output slicing I think
    else
        throw(MacroError("don't understand input, should be @mul A[...] := B[...] * C[...]", where))
    end

    #===== find indices to sum, batch, etc =====#

    store = SizeDict()

    # parse options both to look for keywords and sizes
    @capture(options, (optvec__,)) || (optvec = Any[options])
    optind, _,_,_ = parse!(store, nothing, [], optvec; allowranges=true, flags=flags)

    # get list of indices from each factor
    indZ, nameZ = firstpass!(store, left, where, :inplace in flags)
    indA, nameA = firstpass!(store, mid, where)
    indB, nameB = firstpass!(store, right, where)

    if nameZ == nothing
        :inplace in flags && throw(MacroError("can't write in-place to an un-named tensor",where))
        nameZ == gensym(:Z)
    end
    nameA == nothing || nameB == nothing && throw(MacroError("can't work without tensor names",where))
    checkrepeats(indZ, " on left hand side", where)
    checkrepeats(indA, " in first factor", where)
    checkrepeats(indB, " in second factor", where)

    sumind = setdiff(intersect(indA, indB), indZ)
    batchind = intersect(indA, indB, indZ)

    if length(batchind) > 0
        push!(flags, :bmm)
    end
    if length(sumind)>1 || length(batchind)>1 # then result of * (or batchmul) is wrong shape
        push!(flags, :outshape)
    end

    indAonly = setdiff(indA, sumind, batchind)
    indBonly = setdiff(indB, sumind, batchind)

    if length(indAonly) > 1 || length(indBonly) > 1 || length(batchind) > 1
        push!(flags, :midshape) # result of * will not be have canonical dimensions
    end

    canon = vcat(indZ, sumind)

    V && @info "@mul indices" Tuple(indAonly) Tuple(indBonly) Tuple(canon) Tuple(sumind) Tuple(batchind)

    # having got canonical list, we can check the options
    if count(i -> i != nothing, setdiff(optind, canon)) > 0
        str = join(something.(setdiff(optind, canon), "nothing"), ", ")
        m_error("attempting to ignore unrecognised options: $str", where)
    end

    #===== process each factor in detail =====#

    outex = MacroTools.@q begin end

    if :nolazy in flags
        m_error("@mul always ignores option nolazy, try |= to copy output", where)
    end
    push!(flags, :nolazy) # exA & exB should use permutedims

    exA = matexin!(outex, nameA, mid, (indAonly, sumind, batchind), flags, store, icheck, where)
    exB = matexin!(outex, nameB, right, (sumind, indBonly, batchind), flags, store, icheck, where)

    if :inplace in flags
        pop!(flags, :nolazy) # exZ must not use permutedims

        exZ = matexin!(outex, nameZ, left, (indAonly, indBonly, batchind), flags, store, icheck, where)
    else
        # exZ will not be used for in-place mul!, but for out-of-place... can borrow readleft from @cast:
        _, outUZ, nameZ, checkZ = readleft(left, sumind, flags, store, icheck, where)

        # but unlike @cast/@reduce, the RHS does not arrive in canonical order...
        V && @info "more" Tuple(indZ) Tuple(vcat(indAonly,indBonly,batchind))

        p1 = sortperm(indZ)
        p2 = sortperm(vcat(indAonly,indBonly,batchind))
        permU = invperm(p1)[p2]
        if permU != 1:length(permU)
            push!(flags, :outperm)
        end
    end

    packagecheck(flags, where)

    canonsize = sizeinfer(store, canon, where, true)

    if :inplace in flags

        # if :slice in flags # doesn't work
        #     throw(MacroError("can't write to sliced arrays in-place, for now", where))
        # end

        #===== in-place output =====#

        if :batched in flags && :bmm in flags
            error("can't use batched yet!")
        elseif :bmm in flags
            opex =  :( TensorCast.batchmul!($exZ, $exA, $exB))
        else
            opex = :( TensorCast.mul!($exZ, $exA, $exB))  # LinearAlgebra.mul! is available here
        end

        if :strided in flags # this functions mostly to prevent a slowdown if "inputex" has used strided
            opex = :( Strided.@strided $opex )
        end

        push!(outex.args, opex )
        push!(outex.args, nameZ )

    else

        #===== out-of-place output =====#

        if :batched in flags && :bmm in flags
            error("can't use batched yet!")
        elseif :bmm in flags
            newright = :( TensorCast.batchmul($exA, $exB) )
        else
            newright = :( $exA * $exB )
        end

        if :midshape in flags
            sizeM = Any[ Symbol(:sz_, i) for i in vcat(indAonly,indBonly,batchind) ]
            sizeMex = :(($(sizeM...) ,))
            newright = :( reshape( $newright, $sizeMex ) )
        end

        if :outperm in flags
            if :strided in flags
                ex = :( strided_permutedims($ex, $perm) )
            elseif permU == [2,1]
                newright = :( transpose($newright) )
            else
                newright = :( PermutedDimsArray($newright, ($(permU...),) ))
            end
        end

        if :strided in flags # this functions mostly to prevent a slowdown if "inputex" has used strided
            newright = :( Strided.@strided $newright )
        end

        finalright = outputnew(newright, outUZ, nothing, canonsize, canon, flags, store, where)

        push!(outex.args, :( $nameZ =  $finalright ) )
    end

    #===== finalise (identical to @cast) =====#

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
        return esc(outex.args[1])
    else
        return esc(outex)
    end
end

"""
    indA, A = firstpass(store, ex)

Aim is just to get the list of indices for each piece, to figure out which ones are contracted,
and to have canonical order (now can't be got from one side alone).
"""
function firstpass!(sdict, ex, where, canseeA=true)

    if @capture(ex, A_[outer__][inner__]) ||  @capture(ex, [outer__][inner__])
    elseif @capture(ex, A_[outer__]{inner__}) || @capture(ex, [outer__]{inner__})
    elseif @capture(ex, A_[outer__]) || @capture(ex, [outer__])
        inner = []
    else
        throw(MacroError("don't know what to do with $left", where))
    end

    if canseeA

        # if we have f(x)[i,j] then we should evaluate f just once
        if !isa(A, Symbol)
            Atop = gensym(:A)
            push!(sdict.topex, :(local $Atop = $A ) )
            A = Atop
        end

        flat, getafix, putsz, negated = parse!(sdict, A, outer, inner)
    else
        flat, getafix, putsz, negated = parse!(sdict, nothing, outer, inner)
    end
    flat, A
end

"""
    matexin!(outex, A, ex, (ind1,ind2,ind3), flags, store, icheck, where)

Produces the expression which will glue/permutedims etc A to form `ind1,ind2,ind3` (as `@cast`),
and then reshape A to be either a matrix (indices `⨷ind1, ⨷ind2`)
or for the batched case, a 3-tensor (indices `⨷ind1, ⨷ind2, ⨷ind3`).
Pushes this into outex & returns the symbol it is assigned to.
"""
function matexin!(outex, A, ex, (ind1,ind2,ind3), flags, store, icheck, where)

    # same function called by @cast walker -- produces reshapes to 2nd input, was canon
    Aval = inputex(A, ex, vcat(ind1,ind2,ind3), flags, SizeDict(), icheck, where)
    # @pretty @mul A[n][i] |= B[k,i,n] * C[k\n] # TODO fix this, inputex has assumptions...

    # one more processing step: reshape to matrix or batch-tensor
    if length(ind1)<=1 && length(ind2)<=1 && length(ind3) <= 1
    else
        prodsA = prodsizeex(ind1,ind2,ind3)
        Aval = :( reshape($Aval, ($(prodsA...),)) )
        push!(flags, :needsize) # resize needs sz_i
    end

    # push pre-processing step into outex if nontrivial, as walker did
    if isa(Aval, Symbol)
        ex = Aval
    else
        Asym = gensym(:A)
        push!(outex.args, :(local $Asym = $Aval) )
        ex =  Asym
    end

    # return what should appear in mul! etc.
    return ex
end

function prodsizeex(ind123...)
    out = Any[]
    for ind in ind123
        if length(ind) == 1
            push!(out, szwrap(ind[1]) )
        elseif length(ind) >= 2
            push!(out, szwrap(ind) )
        end
    end
    out
end

using LinearAlgebra

"""
    batchmul(A, B)      # @mul Z[i,j,n] := A[i,k,n] * B[k,j,n]
    batchmul!(Z, A, B)  # @mul Z[i,j,n] = A[i,k,n] * B[k,j,n]

Batched matrix multiplication; batch index is the last dimension.
"""
function batchmul!(Z::AbstractArray{TZ,N}, A::AbstractArray{TA,3}, B::AbstractArray{TB,N}) where {TA,TZ,TB,N}
    2 <= N <= 3 || throw(DimensionMismatch("expected ndims = 3,3,3 or else 2,3,2, got ndims = $N,3,$N"))
    size(Z,N) == size(A,3) == size(B,N) || throw(DimensionMismatch("arrays must agree on batch dimension"))
    codeB = ntuple(i-> i==N ? (*) : (:), N)
    sZ = sliceview(Z, codeB)
    sA = sliceview(A, (:,:,*))
    sB = sliceview(B, codeB)
    foreach(mul!, sZ, sA, sB)
    Z
end

@doc @doc(batchmul!)
function batchmul(A::AbstractArray{TA,3}, B::AbstractArray{TB,N}) where {TA,TB,N}
    2 <= N <= 3 || throw(DimensionMismatch("expected ndims = 3,3 or else 3,2, got ndims = 3,$N"))
    size(A,3) == size(B,N) || throw(DimensionMismatch("arrays must agree on batch dimension"))
    codeB = ntuple(i-> i==N ? (*) : (:), N)
    sA = sliceview(A, (:,:,*))
    sB = sliceview(B, codeB)
    sZ = map(*, sA, sB)
    red_glue(sZ, codeB)
end

# https://github.com/FluxML/Flux.jl/issues/544
# https://github.com/Roger-luo/BatchedRoutines.jl

