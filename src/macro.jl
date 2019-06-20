
#==================== The Macros ====================#

export @cast, @reduce, @matmul

struct CallInfo
    mod::Module
    src::LineNumberNode
    string::String
    flags::Set{Symbol}
    CallInfo(mod::Module, src, str, flags=Set{Symbol}()) = new(mod, src, str, flags)
    CallInfo(syms::Symbol...) = new(Main, LineNumberNode(0), "", Set([syms...]))
end

"""
    @cast Z[i,j,...] := f(A[i,j,...], B[j,k,...])  options

Macro for broadcasting, reshaping, and slicing of arrays in index notation.
Understands the following things:
* `A[i,j,k]` is a three-tensor with these indices.
* `B[(i,j),k]` is the same thing, reshaped to a matrix. Its first axis (the bracket) is indexed
  by `n = i + (j-1) * N` where `i ∈ 1:N`. This may also be written `B[i\\j,k]` or `B[i⊗j,k]`.
* `C[k][i,j]` is a vector of matrices, either created by slicing (if on the left)
  or implying glueing into (if on the right) a 3-tensor `A[i,j,k]`.
* `D[j,k]{i}` is an ordinary matrix of `SVector`s, which may be reinterpreted from `A[i,j,k]`.
* `E[i,_,k]` has two nontrivial dimensions, and `size(E,2)==1`. On the right hand side
  (or when writing to an existing array) you may also write `E[i,3,k]` meaning `view(E, :,3,:)`,
  or `E[i,\$c,j]` to use a variable `c`.
* `f(x)[i,j,k]` is allowed; `f(x)` must return a 3-tensor (and will be evaluated only once).
* `g(H[:,k])[i,j]` is a generalised `mapslices`, with `g` mapping columns of `H`
    to matrices, which are glued into a 3-tensor `A[i,j,k]`.
* `h(I[i], J[j])[k]` expects an `h` which maps two scalars to a vector,
  which gets broadcasted `h.(I,J')` then glued into to a 3-tensor.
* `K[i,j]'` conjugates each element, equivalent to `K'[j,i]` which is the
  conjugate-transpose of the matrix.
* `M[i,i]` means `diag(M)[i]`, but only for matrices: `N[i,i,k]` is an error.
* `P[i,i']` is normalised to `P[i,i′]` with unicode \\prime.
* `R[i,-j,k]` means roughly `reverse(R, dims=2)`, and `Q[i,-j,k]` similar with `shuffle`.
* `S[i,T[j,k]]` is the 3-tensor `S[:,T]` created by indexing a matrix `S` with `T`,
  where these integers are `all(1 .<= T .<= size(S,2))`.

The left and right hand sides must have all the same indices,
and the only repeated index allowed is `M[i,i]`, which is a diagonal not a trace.
See `@reduce` and `@mul` for related macros which can sum over things.

If a function of one or more tensors appears on the right hand side,
then this represents a broadcasting operation,
and the necessary re-orientations of axes are automatically inserted.

The following actions are possible:
* `=` writes into an existing array, overwriting its contents,
  while `+=` adds (precisely `Z .= Z .+ ...`) and `*=` multiplies.
* `:=` creates a new array. This need not be named: `Z = @cast [i,j,k] := ...` is allowed.
* `|=` insists that this is a copy, not a view.

Re-ordering of indices `Z[k,j,i]` is done lazily with `PermutedDimsArray(A, ...)`.
Reversing of an axis `F[i,-j,k]` is also done lazily, by `Reverse{2}(F)` which makes a `view`.
Using `|=` (or broadcasting) will produce a simple `Array`.

Options can be specified at the end (if several, separated by `,` i.e. `options::Tuple`)
* `i:3` supplies the range of index `i`. Variables and functions like `j:Nj, k:length(K)`
  are allowed.
* `assert` will turn on explicit dimension checks of the input.
  (Providing any ranges will also turn these on.)
* `cat` will glue slices by things like `hcat(A...)` instead of the default `reduce(hcat, A)`,
  and `lazy` will instead make a `VectorOfArrays` container.
* `nolazy` disables `PermutedDimsArray` and `Reverse` in favour of `permutedims` and `reverse`,
  and `Diagonal` in favour of `diagm` for `Z[i,i]` output.
* `strided` will place `@strided` in front of broadcasting operations,
  and use `@strided permutedims(A, ...)` instead of `PermutedDimsArray(A, ...)`.
  For this you need `using Strided` to load that package.

To create static slices `D[k]{i,j}` you should give all slice dimensions explicitly.
You may write `D[k]{i:2,j:2}` to specify `Size(2,2)` slices.
They are made most cleanly from the first indices of the input,
i.e. this `D` from `A[i,j,k]`. The notation `A{:,:,k}` will only work in this order,
and writing `A{:2,:2,k}` provides the sizes.
"""
macro cast(exs...)
    call = CallInfo(__module__, __source__, TensorCast.unparse("@cast", exs...))
    _macro(exs...; call=call)
end

"""
    @reduce A[i] := sum(j,k) B[i,j,k]             # A = vec(sum(B, dims=(2,3)))
    @reduce A[i] := prod(j) B[i] + ε * C[i,j]     # A = vec(prod(B .+ ε .* C, dims=2))
    @reduce A[i] = sum(j) exp( C[i,j] / D[j] )    # sum!(A, exp.(C ./ D') )

Tensor reduction macro:
* The reduction function can be anything which works like `sum(B, dims=(1,3))`,
  for instance `prod` and `maximum` and `Statistics.mean`.
* In-place operations `Z[j] = sum(...` will construct the banged version of the given
  function's name, which must work like `sum!(Z, A)`.
* The tensors can be anything that `@cast` understands, including gluing of slices `B[i,k][j]`
  and reshaping `B[i\\j,k]`. See `? @cast` for the complete list.
* If there is a function of one or more tensors on the right,
  then this is a broadcasting operation.
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

    @reduce sum(i) A[i] * log(@reduce [i] := sum(j) A[j] * exp(B[i,j]))
    @cast W[i] := A[i] * exp(- @reduce S[i] = sum(j) exp(B[i,j]) lazy)

Recursion like this is allowed, inside either `@cast` or `@reduce`.
The intermediate array need not have a name, like `[i]`,
unless writing into an existing array, like `S[i]` here.
"""
macro reduce(exs...)
    call = CallInfo(__module__, __source__, TensorCast.unparse("@reduce", exs...), Set([:reduce]))
    _macro(exs...; call=call)
end

"""
    @matmul M[a,c] := sum(b)  A[a,b] * B[b,c]

Matrix multiplication macro. Uses the same syntax as `@reduce`, but instead of broadcasting
the expression on the right out to a 3-tensor before summing it along one dimension,
this calls `*` or `mul!`, which is usually much faster.
But it only works on expressions of suitable form.

With more than two tensors on the right, it proceeds left to right, and each summed index
must appear on two tensors, probably neighbours.

Note that unlike `@einsum` and `@tensor`, you must explicitly specify
what indices to sum over. Both this macro and `@reduce` could in principal infer this,
and perhaps I will add that later... but then I find myself commenting `# sum_μ` anyway.

    @matmul Z[a⊗b,z] = sum(i,j,k)  D[i,a][j] * E[i⊗j,_,k,b] * F[z,3,k]

Each tensor will be pre-processed exactly as for `@cast` / `@reduce`,
here glueing slices of `D` together, reshaping `E`, and taking a view of `F`.
Once this is done, the right hand side must be of the form `(tensor) * (tensor) * ...`,
which becomes `mul!(ZZ, (DD * EE), FF)`.

    @reduce V[i] := sum(k) W[k] * exp(@matmul [i,k] := sum(j) A[i,j] * B[j,k])
    @reduce V[i] := sum(k) W[k] * exp(@matmul sum(j) A[i,j] * B[j,k]) # soon!

You should be able to use this within the other macros,
eventually without specifying intermediate arrays by hand.
"""
macro matmul(exs...)
    call = CallInfo(__module__, __source__, TensorCast.unparse("@matmul", exs...), Set([:matmul]))
    _macro(exs...; call=call)
end

macro mul(exs...)
    call = CallInfo(__module__, __source__, TensorCast.unparse("@matmul", exs...), Set([:matmul]))
    @warn "please replace @mul with @matmul, and ensure that it has explicit sum()" call.string maxlog=1 _id=hash(exs)
    _macro(exs...; call=call)
end

#==================== The Main Functions ====================#

function _macro(exone, extwo=nothing, exthree=nothing; call::CallInfo=CallInfo(), dict=Dict())
    store = (dict=dict, assert=[], mustassert=[], seen=[], need=[], top=[], main=[])
    # TODO use OrderedDict() for main? To allow duplicate removal

    if (:reduce in call.flags) || (:matmul in call.flags)
        # First the options:
        optionparse(exthree, store, call)
        # Then the LHS, to get canonical list of indices:
        canon, parsed = reduceparse(exone, extwo, store, call)
    else
        # Much the same for @cast:
        isnothing(exthree) || throw(MacroError("@cast doesn't take three expressions: $exthree", call))
        optionparse(extwo, store, call)
        canon, parsed = castparse(exone, store, call)
    end

    # First pass over RHS just to read sizes, prewalk sees A[i][j] before A[i]
    MacroTools.prewalk(x -> rightsizes(x, store, call), parsed.right)

    # To look for recursion, we need another prewalk:
    right2 = MacroTools.prewalk(x -> recursemacro(x, store, call), parsed.right)

    # Third pass to standardise & then glue, postwalk sees A[i] before A[i][j]
    right3 = MacroTools.postwalk(x -> standardglue(x, canon, store, call), right2)

    if !(:matmul in call.flags)
        # Then finally broadcasting if necc (or just permutedims etc. if not):
        right4 = targetcast(right3, canon, store, call)
    else
        # or, for matmul, can I change just this? outputinplace is also going to need changing.
        right4 = matmultarget(right3, canon, parsed, store, call)
    end

    checkallseen(canon, store, call) # this must be run before inplaceoutput()

    # Return to LHS, build up what it requested:
    if :inplace in call.flags
        rightlist = inplaceoutput(right4, canon, parsed, store, call)
    else
        right5 = newoutput(right4, canon, parsed, store, call)
        rightlist = [:( $(parsed.name) = $right5 )]
    end

    # Sew all these pieces into output:
    outex = quote end
    append!(outex.args, store.top)
    append!(outex.args, findsizes(store, call)) # this must be run after newoutput() etc.
    append!(outex.args, store.main)
    append!(outex.args, rightlist)

    if :recurse in call.flags
        return (name=parsed.name, ind=parsed.outer, scalar=(:scalar in call.flags), steps=outex.args)
    else
        return esc(outex) # could use MacroTools.unblock()?
    end
end

"""
    standardise(f(x)[i,3]) -> view(A,:,3)[i]
    standardise(A[(i,j)]) -> reshape(A, sz_i, sz_j)[i,j]

This mostly aims to re-work the given expression into `some(steps(A))[i,j]`,
but also pushes `A = f(x)` into `store.top`, and sizes into `store.dict`,
"""
function standardise(ex, store::NamedTuple, call::CallInfo; LHS=false)
    # This acts only on single indexing expressions:
    if @capture(ex, A_{ijk__})
        static=true
        push!(call.flags, :staticslice)
    elseif @capture(ex, A_[ijk__])
        static=false
    else
        return ex
    end

    # Ensure that f(x)[i,j] will evaluate once, including in size(A)
    if A isa Symbol || @capture(A, AA_.ff_) # caller has ensured !containsindexing(A)
    else
        Asym = Symbol(A,"_val") # exact same symbol is used by rightsizes()
        push!(store.top,  :( local $Asym = $A ) )
        A = Asym
    end

    # Constant indices A[i,3], A[i,_], A[i,$c], and also A[i,3:5]
    if all(isCorR, ijk)
        needview!(ijk)                                         # replace _ with 1, if any
        Asym = maybepush(:( $A[$(ijk...)] ), store, :allconst) # protect from later processing
        return Asym                                            # and nothing more to do here
    elseif any(isCorR, ijk)
        ijcolon = map(i -> isCorR(i) ? i : (:), ijk)
        if needview!(ijcolon)
            A = :( view($A, $(ijcolon...)) ) # TODO nolazy?
        else
            A = :( TensorCast.rview($A, $(ijcolon...)) )
        end
        ijk = filter(!isconstant, ijk)              # remove actual constants from list,
        ijk = map(i -> isrange(i) ? :(:) : i, ijk)  # but for A[i,3:5] replace with : symbol, for slicing
    end

    # Slicing operations A[i,:,j], and A[i,3:5] after the above treatment
    if any(iscolon, ijk)
        code = Tuple(map(i -> iscolon(i) ? (:) : (*), ijk))
        if !static && (:collect in call.flags)
            A = :( TensorCast.slicecopy($A, $code) )
        elseif !static
            A = :( TensorCast.sliceview($A, $code) )
        else
            sizeorcode = maybestaticsizes(ijk, code)
            A = :( TensorCast.static_slice($A, $sizeorcode) )
        end
        ijk = filter(!iscolon, ijk)
    elseif static
        error("shouldn't use curly brackets here")
    end

    # Nested indices A[i,j,B[k,l,m],n] -- experimental!
    # TODO: do with the constants,  @pretty @cast C[i,j] := M[1,N[i,j]] # makes view(view(M, 1, :), N)
    if any(i -> @capture(i, B_[klm__]), ijk)
        newijk = []
        beecolon = []
        for i in ijk
            if @capture(i, B_[klm__])
                append!(newijk, klm)
                push!(beecolon, B)
            else
                push!(newijk, i)
                push!(beecolon, (:))
            end
        end
        ijk = newijk
        A = :( view($A, $(beecolon...)) )
    end

    # Diagonal extraction A[i,i]
    if length(ijk)==2 && ijk[1]==ijk[2]
        if (:nolazy in call.flags) && !LHS # don't do this for in-place output
            A = :( TensorCast.diag($A) ) # LinearAlgebra really
        else
            A = :( TensorCast.diagview($A) )
        end
        pop!(ijk)
    end

    # Parse the remaining indices; saving of their sizes was done elsewhere:
    flat, _, reversed, shuffled = indexparse(A, ijk)
    append!(store.seen, flat)

    # Combined indices A[i,(j,k)]
    if any(istensor, ijk)
        flatsize = map(szwrap, flat)
        A = :( reshape($A, ($(flatsize...),)) )
        append!(store.need, flat)
    end

    # Reversed A[-i,j] and shuffled A[i,~j]
    for i in reversed
        di = findfirst(isequal(i), flat)
        if (:nolazy in call.flags) && !LHS
            A = :( reverse($A, dims=$di) )
        else
            A = :( TensorCast.Reverse{$di}($A) )
        end
    end

    if length(shuffled) > 0
        if (:nolazy in call.flags) && !LHS
            if length(flat) == 1
                A = :( TensorCast.shuffle(A) )
            else # sadly shuffle(A, dims=...) doesn't exist!
                code = Tuple(Any[ i in shuffled ? (*) : (:) for i in flat ])
                A = :( TensorCast.red_glue(TensorCast.shuffle(
                    TensorCast.slicecopy($A, $code)),$code) )
            end
        else
            for i in shuffled
                di = findfirst(isequal(i), flat)
                A = :( TensorCast.Shuffle{$di}($A) )
            end
        end
    end

    # Construct final expression
    return :( $A[$(flat...)] )
end

"""
    standardglue(expr) -> func(old)[i,j,k]

This gets walked over the RHS and:
* standardises any simple expression `f(x)[i⊗j,3,k]` by calling `standardise()`,
* finds anything which needs glueing either directly `A[j,k][i]`,
  or after broadcasting `(A[j]+B[k])[i]` using `targetcast()` first.

target dims not correctly handled yet -- what do I want? TODO
Simple glue / stand. does not permutedims, but broadcasting may have to... avoid twice?
"""
function standardglue(ex, target, store::NamedTuple, call::CallInfo)

    # The sole target here is indexing expressions:
    if @capture(ex, A_[inner__])
        static=false
    elseif @capture(ex, A_{inner__})
        static=true
    else
        return ex
    end

    # Do we have something simeple like M[i,j], no worse than f(x)[i,:,j⊗k,3]?
    if !containsindexing(A)
        return standardise(ex, store, call)
    end

    # Otherwise there are two options, (brodcasting...)[k] or simple B[i,j][k]
    needcast = !@capture(A, B_[outer__])

    if needcast
        outer = unique(reduce(vcat, listindices(A)))
        outer = sort(outer, by = i -> findcheck(i, target, call)) # , " in target list"
        Bex = targetcast(A, outer, store, call)
        B = maybepush(Bex, store, :innercast)
    end

    # Before gluing, deal with any constant inner indices
    Bsym = gensym(:innerfix)
    if all(isconstant, inner)
        B = maybepush(B, store, :preget) # protect any previous operations from @.
        needview!(inner) # replace _ with 1, if any
        push!(store.main, :( $Bsym =  @__dot__ getindex($B, $(inner...)) ) )
        return  :( $Bsym[$(outer...)] ) # actually, no glueing at all to do!

    elseif any(isconstant, inner)
        B = maybepush(B, store, :preconst)
        ijcolon = map(i -> isconstant(i) ? i : (:), inner)
        if needview!(ijcolon)
            ex = :( $Bsym =  @__dot__ view($B, $(ijcolon...)) ) # TODO nolazy here
        else
            ex = :( $Bsym = @__dot__ TensorCast.rview($B, $(ijcolon...)) )
        end
        push!(store.main, ex)
        B = Bsym
        inner = filter(!isconstant, inner)
    end

    # And some quick error checks; standsrdise() already put inner into store.seen
    append!(store.seen, inner)
    checknorepeats(vcat(inner, outer), call,
        " in gluing " * string(:([$(outer...)])) * string(:([$(inner...)])))

    # Now we glue, always in A[k,l][i,j] -> B[i,j,k,l] order to start with:
    ijk = vcat(inner, outer)
    code = Tuple(Any[ i in inner ? (:) : (*) for i in ijk ])

    if static
        AB = :( TensorCast.static_glue($B) )
        pop!(call.flags, :collected, :ok)
    elseif :glue in call.flags
        # allow maximum freedom? TODO
        code, ijk = gluereorder(code, ijk, target)

        AB = :( TensorCast.copy_glue($B, $code) )
    elseif :cat in call.flags
        AB = :( TensorCast.cat_glue($B, $code) )
        push!(call.flags, :collected)
    elseif :lazy in call.flags
        AB = :( TensorCast.lazy_glue($B, $code) )
        pop!(call.flags, :collected, :ok)
    else
        # allow a little freedom?
        if code == (*,:) && ijk == reverse(target)
            code = (:,*)
            ijk = target
        end

        AB = :( TensorCast.red_glue($B, $code) )
        push!(call.flags, :collected)
    end

    return :( $AB[$(ijk...)] )
end

"""
    targetcast(A[i] + B[j], [i,j]) -> @__dot__(A + B')
    targetcast(A[j,i], [i,j]) -> transpose(A)

This beings the expression to have target indices,
by permutedims and if necessary broadcasting, always using `readycast()`.
"""
function targetcast(ex, target, store::NamedTuple, call::CallInfo)

    # If just one naked expression, then we won't broadcast:
    if @capture(ex, A_[ijk__])
        containsindexing(A) && error("that should have been dealt with")
        return readycast(ex, target, store, call)
    end

    # But for anything more complicated, we do:
    ex = MacroTools.prewalk(x -> readycast(x, target, store, call), ex)
    if :lazy in call.flags
        ex = :( TensorCast.lazy($ex) )
        pop!(call.flags, :collected, :ok)
    else
        push!(call.flags, :collected)
    end
    if :strided in call.flags
        ex = Broadcast.__dot__(ex) # @strided does not work on @.
        ex = :( Strided.@strided $ex )
    else
        ex = :( @__dot__($ex) )
    end
    return ex

end

"""
    readycast(A[j,i], [i,j,k]) -> transpose(A)
    readycast(A[k], [i,j,k]) -> orient(A, (*,*,:))

This is walked over the expression to prepare for `@__dot__` etc, by `targetcast()`.
"""
function readycast(ex, target, store::NamedTuple, call::CallInfo)

    # Scalar functions can be protected entirely from broadcasting:
    # TODO this means A[i,j] + rand()/10 doesn't work, /(...,10) is a function!
    if @capture(ex, fun_(arg__) | fun_(arg__).field_ ) && !containsindexing(fun) &&
            length(arg)>0 && !any(containsindexing, arg) # pull out log(2) but not randn()
        return maybepush(ex, store, :scalarfun)
    end

    # Some things ought to apply elementwise: conjugation,
    @capture(ex, A_[ijk__]') && return :( Base.adjoint($A[$(ijk...)]) )
    # .fields... only one deep for now,
    @capture(ex, A_[ijk__].field_ ) &&
        return :( getproperty($A[$(ijk...)], $(QuoteNode(field))) )
    @capture(ex, fun_(arg__).field_ ) && any(containsindexing, arg) &&
        return :( getproperty($fun($(arg...)), $(QuoteNode(field))) )
    # tuple creation... fails for namedtuple, TODO? NamedTupleTools...
    @capture(ex, (args__,) ) && any(containsindexing, args) &&
        return :( tuple($(args...)) )
    # and arrays of functions, using apply:
    @capture(ex, funs_[ijk__](args__) ) &&
        return :( TensorCast.apply($funs[$(ijk...)], $(args...) ) )


    # Apart from those, readycast acts only on lone tensors:
    @capture(ex, A_[ijk__]) || return ex

    dims = Int[ findcheck(i, target, call, " on the left") for i in ijk ]

    # Do we need permutedims, or equvalent?
    perm = sortperm(dims)
    if perm != 1:length(dims)
        tupleperm = Tuple(perm)
        if :strided in call.flags
            A = :( Strided.@strided permutedims($A, $tupleperm) )
        elseif :nolazy in call.flags
            A = :( permutedims($A, $tupleperm) )
            push!(call.flags, :collected)
        elseif tupleperm == (2,1)
            A = :( TensorCast.PermuteDims($A) )
        else
            A = :( PermutedDimsArray($A, $tupleperm) )
            pop!(call.flags, :collected, :ok)
        end
    end

    # Do we need orient()?
    if length(dims)>0 && maximum(dims) != length(dims)
        code = Tuple(Any[ d in dims ? (:) : (*) for d=1:maximum(dims) ])
        A = :( TensorCast.orient($A, $code) )
    end
    # Those two steps are what might be replaced with TransmuteDims

    # Does A need protecting from @__dot__?
    A = maybepush(A, store, :nobroadcast)

    return A
end

"""
    matmultarget(A[i,j] * B[j,k], [i,k]) -> A * B

For inplace, instead it returns tuple `(A,B)`, which `inplaceoutput()` can use.
If there are more than two factors, it recurses, and you get `(A*B) * C`,
or perhaps tuple `(A*B, C)`.
"""
function matmultarget(ex, target, parsed, store::NamedTuple, call::CallInfo)

    @capture(ex, A_ * B_ * C__ | *(A_, B_, C__) ) || error("can't @matmul that!")

    # Figure out what to sum over, and make A,B into matrices ready for *
    iA = guesstarget(A)
    iB = guesstarget(B)

    isum = sort(intersect(iA, iB, parsed.reduced),
        by = i -> findfirst(isequal(i), parsed.reduced))
    iAnosum = setdiff(iA, isum)
    iBnosum = setdiff(iB, isum)

    Aex = targetcast(A, vcat(iAnosum, isum), store, call)
    Bex = targetcast(B, vcat(isum, iBnosum), store, call)

    Aex = matrixshape(Aex, iAnosum, isum, store, call)
    Bex = matrixshape(Bex, isum, iBnosum, store, call)

    # But don't actually * if you are going to mul!(Z,A,B) instead later:
    if (:inplace in call.flags) && length(C)==0
        return (Aex, Bex)
    end

    ABex = unmatrixshape(:( $Aex * $Bex ), iAnosum, iBnosum, store, call)
    ABsym = gensym(:matmul)
    push!(store.main, :( local $ABsym = $ABex ))
    ABind = :( $ABsym[$(iAnosum...), $(iBnosum...)] )

    if length(C) == 0
        # Last step for Z := A * B case is reshape & permutedims etc:
        subtarget = setdiff(target, isum)
        return targetcast(ABind, subtarget, store, call)

    else
        # But if we have (A*B) * C then continue, whether in- or out-of-place:
        ABCex = :( *($ABind, $(C...)) )
        return matmultarget(ABCex, target, parsed, store, call)
    end
end

"""
    recursemacro(@reduce sum(i) A[i,j]) -> G[j]

Walked over RHS to look for `@reduce ...`, and replace with result,
pushing calculation steps into store.

Also a convenient place to tidy all indices, including e.g. `fun(M[:,j],N[j]).same[i']`.
"""
function recursemacro(ex, store::NamedTuple, call::CallInfo)

    # Actually look for recursion
    if @capture(ex, @reduce(subex__) )
        subcall = CallInfo(call.mod, call.src,
            TensorCast.unparse("innder @reduce", subex...), Set([:reduce, :recurse]))
        name, ind, scalar, steps = _macro(subex...; call=subcall, dict=store.dict)
        append!(store.main, steps)
        ex = scalar ? :($name) :  :($name[$(ind...)])

    elseif @capture(ex, @matmul(subex__) )
        subcall = CallInfo(call.mod, call.src,
            TensorCast.unparse("innder @matmul", subex...), Set([:matmul, :recurse]))
        name, ind, scalar, steps = _macro(subex...; call=subcall, dict=store.dict)
        append!(store.main, steps)
        ex = scalar ? :($name) :  :($name[$(ind...)])

    elseif @capture(ex, @cast(subex__) )
        subcall = CallInfo(call.mod, call.src,
            TensorCast.unparse("inner @cast", subex...), Set([:recurse]))
        name, ind, scalar, steps = _macro(subex...; call=subcall, dict=store.dict)
        append!(store.main, steps)
        ex = scalar ? :($name) :  :($name[$(ind...)])
    end

    # Tidy up indices, A[i,j][k] will be hit on different rounds...
    if @capture(ex, A_[ijk__])
        return :( $A[$(tensorprimetidy(ijk)...)] )
    elseif @capture(ex, A_{ijk__})
        return :( $A{$(tensorprimetidy(ijk)...)} )
    else
        return ex
    end
end

"""
    rightsizes(A[i,j][k])

This saves to `store` the sizes of all input tensors, and their sub-slices if any.
* Deals with `A[i,j][k]` at once, before seeing `A[i,j]`, hence `prewalk` & destruction.
* For `fun(x)[i,j]` it saves `sz_i` under name `Symbol(fun(x),"_val")` used later by standardise
* But for `fun(M[:,j],N[j]).same[i]` it can't save `sz_i` as this isn't calculated yet,
  however it should not destroy this so that `sz_j` can be got later.
"""
function rightsizes(ex, store::NamedTuple, call::CallInfo)
    :recurse in call.flags && return nothing # outer version took care of this

    if @capture(ex, A_[outer__][inner__] | A_[outer__]{inner__} )
        field = nothing
    elseif @capture(ex, A_[outer__].field_[inner__] | A_[outer__].field_{inner__} )
    elseif  @capture(ex, A_[outer__] | A_{outer__} )
        field = nothing
    else
        return ex
    end

    # Special treatment for  fun(x)[i,j], goldilocks A not just symbol, but no indexing
    if A isa Symbol || @capture(A, AA_.ff_)
    elseif !containsindexing(A)
        A = Symbol(A,"_val") # the exact same symbol is used by standardiser
    end

    # When we can save the sizes, then we destroy so as not to save again:
    if A isa Symbol || @capture(A, AA_.ff_) && !containsindexing(A)
        indexparse(A, outer, store, call; save=true)
        if field==nothing
            innerparse(:(first($A)), inner, store, call; save=true)
        else
            innerparse(:(first($A).$field), inner, store, call; save=true)
        end
        return nothing
    end

    return ex
end

#==================== Various Parsing Functions ====================#

"""
    canon, parsed = castparse(Z[out][inner] := right)

For `@cast`, this digests the LHS.
`parsed.outer` is the indices exactly as given, `parsed.flat` cleans up,
and `canon` prepends inner indices too.
"""
function castparse(ex, store::NamedTuple, call::CallInfo; reduce=false)
    Z = gensym(:left)

    # Do we make a new array? With or without collecting:
    if @capture(ex, left_ := right_ )
    elseif @capture(ex, left_ == right_ )
        @warn "using == no longer does anything" call.string maxlog=1 _id=hash(call.string)
    elseif @capture(ex, left_ |= right_ )
        push!(call.flags, :collect)

    # Do we write into an exising array? Possibly updating it:
    elseif @capture(ex, left_ = right_ )
        push!(call.flags, :inplace)
    elseif @capture(ex, left_ += right_ )
        push!(call.flags, :inplace)
        right = :( $left + $right )
        reduce && throw(MacroError("can't use += with @reduce", call))
    elseif @capture(ex, left_ -= right_ )
        push!(call.flags, :inplace)
        right = :( $left - ($right) )
        reduce && throw(MacroError("can't use -= with @reduce", call))
    elseif @capture(ex, left_ *= right_ )
        push!(call.flags, :inplace)
        right = :( $left * ($right) )
        reduce && throw(MacroError("can't use *= with @reduce", call))

    # @reduce might not have a LHS, otherwise some errors:
    elseif @capture(ex, left_ => right_ )
        throw(MacroError("anonymous functions using => currently not supported", call))
    elseif reduce
        return (Any[], nothing)
    else
        error("wtf is $ex")
    end

    static = @capture(left, ZZ_{ii__})

    if @capture(left, Z_[outer__][inner__] | [outer__][inner__] | Z_[outer__]{inner__} | [outer__]{inner__} )
        isnothing(Z) && (:inplace in call.flags) && throw(MacroError("can't write into a nameless tensor", call))
        Z = isnothing(Z) ? gensym(:output) : Z
        parsed = indexparse(Z, outer, store, call, save=(:inplace in call.flags))
        innerflat = innerparse(:(first($Z)), inner, store, call, save=(:inplace in call.flags))
        canon = vcat(innerflat, parsed.flat)
        checknorepeats(canon, call, " on the left")

    elseif @capture(left, Z_[outer__] | [outer__] )
        isnothing(Z) && (:inplace in call.flags) && throw(MacroError("can't write into a nameless tensor", call))
        Z = isnothing(Z) ? gensym(:output) : Z
        parsed = indexparse(Z, outer, store, call, save=(:inplace in call.flags))
        canon = parsed.flat
        inner, innerflat = [], []

    elseif left isa Symbol # only for @reduce A := sum(i) ...
        (:reduce in call.flags) || throw(MacroError("@cast really needs an indexing expression on left!", call))
        Z = left
        push!(call.flags, :scalar)
        parsed = indexparse(Z, [], store, call) # just to get the right namedtuple
        canon, outer, inner, innerflat = [], [], [], []

    else
        error("don't know what to do with left = $left")
    end

    return canon, (parsed..., name=Z, outer=outer, inner=inner, innerflat=innerflat,
        static=static, left=left, right=right)
end

"""
    canon, parsed = reduceparse(Z[i,j] := sum(j), right)

Similar to `castparse()`, which it uses before adding on what `@reduce` needs.
Tries to be smart about the order of `canon` in the hope that reduction over `parsed.rdims`
can be done without `permutedims`.
"""
function reduceparse(ex1, ex2, store::NamedTuple, call::CallInfo)

    # Parse the LHS if there is one, else only @reduce sum(i,j) A[i,j]
    leftcanon, parsed = castparse(ex1, store, call; reduce=true)
    if parsed==nothing
        @capture(ex1, redfun_(redlist__))
        # TODO allow @reduce sum(i) A[i,j]
        push!(call.flags, :scalar)
        parsed = (indexparse(nothing, [])..., name=gensym(:noleft), outer=[], inner=[], static=false)
    else
        @capture(parsed.right, redfun_(redlist__) ) || error("how do I reduce over $(parsed.right) ?")
    end

    # Then parse redlist, decoding ranges like sum(i:10,j) which specify sizes
    reduced = []
    for item in tensorprimetidy(redlist) # normalise θ'
        i = @capture(item, j_:s_) ? saveonesize(j, s, store) : item
        push!(reduced, i)
    end
    checknorepeats(reduced, call, " in the reduction") # catches sum(i,i) B[i,j,k]

    # Now work out canonical list. For sum!(), reduced indices must come last.
    if :inplace in call.flags
        canon = vcat(leftcanon, reduced)
    else
        # But for Z = sum(A, dims=...) can try to avoid permutedims,
        guess = guesstarget(ex2) # TODO make guess smarter, use leftcanon as a target
        # @show guess reduced
        [ deleteat!(guess, findcheck(i, guess, call, " on the right")) for i in reduced ]
        if leftcanon == guess
            canon = guesstarget(ex2)
        else
            guess = guesstarget(ex2)
            @info "guess failed" repr(guess) repr(leftcanon) repr(reduced)
            canon = vcat(leftcanon, reduced)
        end
    end
    checknorepeats(canon, call, " in the reduction") # catches A[i] := sum(i)

    # Finally, record positions in canon of reductions
    rdims = sort([ findfirst(isequal(i), canon) for i in reduced ])

    return canon, (parsed..., redfun=redfun, reduced=reduced, rdims=rdims, right=ex2) # this replaces parsed.right
end

"""
    p = indexparse(A, [i,j,k])

`p.flat` strips constants, colons, tensor/brackets, minus, tilde, and doubled indices.
`p.outsize` is a list like `[sz_i, 1, star(...)]` for use on LHS.
`p.reversed` has all those with a minus.
Sizes are saved to `store` only with keyword `save=true`, i.e. only when called by `rightsizes()`
"""
function indexparse(A, ijk::Vector, store=nothing, call=nothing; save=false)
    flat, outsize, reversed, shuffled = [], [], [], []

    ijk = tensorprimetidy(ijk) # un-wrap i⊗j to tuples, and normalise i' to i′

    stripminustilde!(ijk, reversed, shuffled)

    for (d,i) in enumerate(ijk)

        if iscolon(i) || isrange(i)
            if i isa QuoteNode
                str = "fixed size in $A[" * join(ijk, ", ") * "]" # DimensionMismatch("fixed size in M[i, \$(QuoteNode(5))]: size(M, 2) == 5") TODO print more nicely
                push!(store.mustassert, :( TensorCast.@assert_ size($A,$d)==$(i.value) $str) )
            end
            continue
        end

        if isconstant(i)
            push!(outsize, 1)
            if i == :_ && save
                str = "underscore in $A[" * join(ijk, ", ") * "]"
                push!(store.mustassert, :( TensorCast.@assert_ size($A,$d)==1 $str) )
            end
            continue
        end

        if @capture(i, (ii__,))
            stripminustilde!(ii, reversed, shuffled)
            append!(flat, ii)
            push!(outsize, szwrap(ii))
            save && saveonesize(ii, :(size($A, $d)), store)

        elseif @capture(i, B_[klm__])
            innerparse(B, klm, store, call) # called just for error on tensor/colon/constant
            sub = indexparse(B, klm, store, call; save=save) # I do want to save size(B,1) etc.
            append!(flat, sub.flat)

        elseif i isa Symbol
            push!(flat, i)
            push!(outsize, szwrap(i))
            save && saveonesize(i, :(size($A, $d)), store)

        else
            throw(MacroError("don't understand index $i", call))
        end
    end

    if save && length(ijk)>0
        N = length(ijk)
        str = "expected a $N-tensor $A[" * join(ijk, ", ") * "]"
        push!(store.assert, :( TensorCast.@assert_ ndims($A)==$N $str) )
    end

    if length(flat)==2 && flat[1]==flat[2] # allow for diag, A[i,i]
        pop!(flat)
    end

    checknorepeats(flat, call)

    reversed = filter(i -> isodd(count(isequal(i), reversed)), reversed)
    shuffled = unique(shuffled)

    return (flat=flat, outsize=outsize, reversed=reversed, shuffled=shuffled)
end

function stripminustilde!(ijk::Vector, reversed, shuffled)
    # @show ijk
    for (d,i) in enumerate(ijk)
        # if @capture(i, -(j, jj__))
        #     append!(reversed, vcat(j,jj))
        #     ijk[d] = :( ($j,$(jj...),) )

        if @capture(i, -j_)
            push!(reversed, j)
            ijk[d] = j
        elseif @capture(i, ~j_)
            push!(shuffled, j)
            ijk[d] = j
        end
    end
end

"""
    innerparse(firstA, [i,j])
Checks that inner indices are kosher, and returns the list.
If `save=true` then it now expects expr. `first(A)` to allow `first(A).field` too,
otherwise it ignores first arg.
"""
function innerparse(firstA, ijk, store::NamedTuple, call::CallInfo; save=false)
    isnothing(ijk) && return []

    ijk = tensorprimetidy(ijk) # only for primes really

    innerflat = []
    for (d,i) in enumerate(ijk)
        iscolon(i) && throw(MacroError("can't have a colon in inner index!", call))
        isrange(i) && @capture(i, alpha_Int:omega_)  && throw(MacroError("can't have a range in inner index!", call)) # this is an imperfect check, must not be triggered by @cast R2[j]{i:3} := M[i,j]  which is another kind of range!
        istensor(i) && throw(MacroError("can't tensor product inner indices", call))
        @capture(i, -j_) || @capture(i, ~j_) && throw(MacroError("can't reverse or shuffle along inner indices", call))

        if @capture(i, j_:s_)
            push!(innerflat, j)
            saveonesize(j, s, store) # save=true on LHS only for in-place, save this anyway
        elseif isconstant(i)
            i == :_ && save && push!(store.mustassert, :(TensorCast.@assert_ size($firstA, $d)==1 "inner underscore") )
        else
            push!(innerflat, i)
        end
        save && saveonesize(i, :(size($firstA, $d)), store)
    end

    checknorepeats(innerflat, call)

    return innerflat
end

function optionparse(opt, store::NamedTuple, call::CallInfo)
    if isnothing(opt)
        return
    elseif @capture(opt, (opts__,) )
        [ optionparse(o, store, call) for o in opts ]
        return
    end

    if @capture(opt, i_:s_)
        saveonesize(tensorprimetidy(i), s, store)
        push!(call.flags, :assert)
    elseif opt in (:assert, :lazy, :nolazy, :cat, :strided)
        push!(call.flags, opt)
    elseif opt == :(!)
        @warn "please replace option ! with assert" call.string maxlog=1 _id=hash(call.string)
        push!(call.flags, :assert)
    else
        throw(MacroError("don't understand option $opt", call))
    end
end

#==================== Smaller Helper Functions ====================#

"""
    saveonesize(:i, size(A,3), store) -> :i

Saves sizes for indices `:i` or products like  `[:i,:j]` to `store`.
It it already has a size for this single index,
then save an assertion that new size is equal to old.
"""
function saveonesize(ind, long, store::NamedTuple)
    if !haskey(store.dict, ind)
        store.dict[ind] = long
    else
        if isa(ind, Symbol)
            str = "range of index $ind must agree"
            push!(store.assert, :(TensorCast.@assert_ $(store.dict[ind]) == $long $str) )
        end
    end
    ind
end

function findsizes(store::NamedTuple, call::CallInfo)
    out = []
    append!(out, store.mustassert)
    if :assert in call.flags
        append!(out, store.assert)
        empty!(store.assert)
    end
    if length(store.need) > 0
        sizes = sizeinfer(store, call)
        sz_list = map(szwrap, store.need)
        push!(out, :( local ($(sz_list...),) = ($(sizes...),) ) )
    end
    out
end

function sizeinfer(store::NamedTuple, call::CallInfo)
    sort!(unique!(store.need))
    sizes = Any[ (:) for i in store.need ]

    # First pass looks for single indices whose lengths are known directly
    for pair in store.dict
        if isa(pair.first, Symbol)
            d = findfirst(isequal(pair.first), store.need)
            d != nothing && (sizes[d] = pair.second)
        end
    end

    count(isequal(:), sizes) < 2 && return sizes

    # Second pass looks for tuples where exactly one entry has unknown length
    for pair in store.dict
        if isa(pair.first, Vector)
            known = [ haskey(store.dict, j) for j in pair.first ]

            if sum(.!known) == 1 # bingo! now work out its size:
                num = pair.second
                denfacts = [ store.dict[i] for i in pair.first[known] ]
                if length(denfacts) > 1
                    den = :( *($(denfacts...)) )
                else
                    den = :( $(denfacts[1]) )
                end
                rat = :( $num ÷ $den )

                i = pair.first[.!known][1]

                d = findfirst(isequal(i), store.need)
                d != nothing && (sizes[d] = rat)

                str = "inferring range of $i from range of $(join(pair.first, " ⊗ "))"
                push!(store.mustassert, :( TensorCast.@assert_ rem($num, $den)==0 $str) )
            end
        end
    end

    unknown = store.need[sizes .== (:)]
    str = join(unknown, ", ")
    length(unknown) <= 1 || throw(MacroError("unable to infer ranges for indices $str", call))

    return sizes
end

"""
    maybestaticsizes([:, :, i], (:,:,*)) -> (:,:,*)
    maybestaticsizes([:3, :4, i], (:,:,*)) -> Size(3,4)
Produces the 2nd argument of `static_slice()`, for slicing `A{:3, :4, i}`.
"""
function maybestaticsizes(ijk::Vector, code::Tuple)
    iscodesorted(code) || error("not sorted!")
    length(ijk) == length(code) || error("wrong length of code!")
    staticsize = Any[ i.value for i in ijk if i isa QuoteNode ]
    if length(staticsize) == count(iscolon, ijk)
        return :( StaticArrays.Size($(staticsize...)) )
    else
        return code
    end
end

"""
    maybestaticsizes([:i,:j,:k], (:,:,*), store) -> Size(3,4)
Produces the 2nd argument of `static_slice()`, using sizes from `store.dict` if available.
"""
function maybestaticsizes(ijk::Vector, code::Tuple, store::NamedTuple)
    iscodesorted(code) || error("not sorted!")
    length(ijk) == length(code) || error("wrong length of code!")
    staticsize = []
    for d=1:countcolons(code)
        if haskey(store.dict, ijk[d])
            push!(staticsize, store.dict[ijk[d]])
        else
            return code
        end
    end
    return :( StaticArrays.Size($(staticsize...)) )
end

"""
    A = maybepush(ex, store, :name)
If `ex` is not just a symbol, then it pushes `:(Asym = ex)` into `store.main`
and returns `Asym`.
"""
maybepush(s::Symbol, any...) = s
function maybepush(ex::Expr, store::NamedTuple, name::Symbol=:A) # TODO make this look for same?
    Asym = gensym(name)
    push!(store.main, :( local $Asym = $ex ) )
    return Asym
end

tensorprimetidy(v::Vector) = Any[ tensorprimetidy(x) for x in v ]
function tensorprimetidy(ex)
    MacroTools.postwalk(ex) do x

        @capture(x, ((ij__,) \ k_) ) && return :( ($(ij...),$k) )
        @capture(x, i_ \ j_ ) && return :( ($i,$j) )

        @capture(x, ((ij__,) ⊗ k_) ) && return :( ($(ij...),$k) )
        @capture(x, i_ ⊗ j_ ) && return :( ($i,$j) )

        @capture(x, ((ij__,), k__) ) && return :( ($(ij...),$(k...)) )

        @capture(x, i_') && return Symbol(i,"′")
        x
    end
end

szwrap(i::Symbol) = Symbol(:sz_,i)
function szwrap(ijk::Vector)
    length(ijk) == 0 && return nothing
    length(ijk) == 1 && return szwrap(first(ijk))
    return :( TensorCast.star($([ Symbol(:sz_,i) for i in ijk ]...)) )
end

isconstant(n::Int) = true
isconstant(s::Symbol) = s == :_
isconstant(ex::Expr) = ex.head == :($)
isconstant(q::QuoteNode) = false

isindexing(s) = false
isindexing(ex::Expr) = @capture(x, A_[ijk__])

isCorI(i) = isconstant(i) || isindexing(ii)

isrange(ex::Expr) = @capture(ex, alpha_:omega_)
isrange(i) = false

isCorR(i) = isconstant(i) || isrange(i) # ranges are treated a bit like constants!

istensor(n::Int) = false
istensor(s::Symbol) = false
function istensor(ex::Expr)
    @capture(ex, i_' )     && return istensor(i)
    @capture(ex, -(ij__) ) && return length(ij)>1 # TODO maybe reject -(i,j) ... istensor(:( -i⊗j )) is OK without this line
    @capture(ex, i_⊗j_ )   && return true
    @capture(ex, i_\j_ )   && return true
    @capture(ex, (ij__,) ) && return length(ij)>1
end
istensor(::QuoteNode) = false

iscolon(s::Int) = false
iscolon(s::Symbol) = s == :(:)
iscolon(ex::Expr) = false
iscolon(q::QuoteNode) = true

containsindexing(s) = false
function containsindexing(ex::Expr)
    flag = false
    # MacroTools.postwalk(x -> @capture(x, A_[ijk__]) && (flag=true), ex)
    MacroTools.postwalk(ex) do x
        # @capture(x, A_[ijk__]) && !(all(isconstant, ijk)) && (flag=true)
        if @capture(x, A_[ijk__])
            # @show x ijk # TODO this is a bit broken?  @pretty @cast Z[i,j] := W[i] * exp(X[1][i] - X[2][j])
            flag=true
        end
    end
    flag
end

listindices(s::Symbol) = []
function listindices(ex::Expr)
    list = []
    MacroTools.postwalk(ex) do x
        if @capture(x, A_[ijk__])
            flat, _ = indexparse(nothing, ijk)
            push!(list, flat)
        end
        x
    end
    list
end

function guesstarget(ex::Expr)
    list = sort(listindices(ex), by=length, rev=true)
    shortlist = unique(reduce(vcat, list))
end

# function overlapsorted(x,y) # works fine but not in use yet
#     z = intersect(x,y)
#     length(z) ==0 && return true
#     xi = map(i -> findfirst(isequal(i),x), z)
#     yi = map(i -> findfirst(isequal(i),y), z)
#     return sortperm(xi) == sortperm(yi)
# end

"""
    needview!([:, 3, A])   # true, need view(A, :,3,:)
    needview!([:, :_, :])  # false, can use rview(A, :,1,:)

Mutates the given vector, replacing symbol `:_` with `1`.
If the vector contains only colons & underscores, then the result is suitable for use
with `rview`, but if not, we need a real view, so it returns `true`.
"""
function needview!(ij::Vector)
    out = false
    for k = 1:length(ij)
        if ij[k] == :_
            ij[k] = 1
        elseif ij[k] isa Int || ij[k] isa Symbol
            out = true
        elseif ij[k] isa Colon
            nothing
        elseif ij[k] isa Expr && ij[k].head == :($)
            ij[k] = ij[k].args[1]
            out = true
        elseif ij[k] isa Expr && @capture(ij[k], alpha_:omega_) # i.e. isrange(ij[k])
            out = true
        else
            error("this should never happen! needview! got ", ij[k])
        end
    end
    out
end

"""
    matrixshape(A, [i,j], [k,l]) -> reshape(A, sz_i * sz_j, sz_k * sz_l)
    matrixshape(A, [], [k,l]) -> reshape(A, 1, sz_k * sz_l) # rowvector * something
    matrixshape(A, [i,j], []) -> reshape(A, sz_i * sz_j)    # something * vector
"""
function matrixshape(ex, left::Vector, right::Vector, store::NamedTuple, call::CallInfo)
    length(left) == 1 && length(right) <= 1 && return ex
    length(left) == 0 && length(right) == 1 && return :( TensorCast.PermuteDims($ex) )

    isempty(right) && return :( reshape($ex, :) )
    right_sz = szwrap(right)

    left_sz = length(left)==0 ? 1 : szwrap(left)

    append!(store.need, left)
    append!(store.need, right)

    return :( reshape($ex, ($left_sz,$right_sz)) )
end

function unmatrixshape(ex, left::Vector, right::Vector, store::NamedTuple, call::CallInfo)
    length(left) == 1 && length(right) == 1 && return ex
    # length(left) == 1 && length(right) == 0 && return :( $ex.parent ) # maybe!

    sizes = :( ($(vcat(szwrap.(left), szwrap.(right))...),) )

    append!(store.need, left)
    append!(store.need, right)

    return :( reshape($ex, $sizes) )
end

#==================== Nice Errors ====================#

"""
    @assert_ cond str

Like `@assert`, but prints both the given string and the condition.
Throws a `DimensionMismatch` error if `cond` is false.
"""
macro assert_(ex, str)
    msg = str * ": " * string(ex)
    return esc(:($ex || throw(DimensionMismatch($msg))))
end

# m_error(str::String) = @error str
# m_error(str::String, call::CallInfo) =
#     @error str  input=call.str _module=call.mod  _line=call.src.line  _file=string(call.src.file)

struct MacroError <: Exception
    msg::String
    call::Union{Nothing, CallInfo}
    MacroError(msg, call=nothing) = new(msg, call)
end

wherecalled(call::CallInfo) = "@ " * string(call.mod) * " " * string(call.src.file) * ":" * string(call.src.line)

function Base.showerror(io::IO, err::MacroError)
    print(io, err.msg)
    if err.call isa CallInfo
        printstyled(io, "\n    ", err.call.string; color = :blue)
        printstyled(io, "\n    ", wherecalled(err.call); color = :normal)
    end
end

function checknorepeats(flat, call=nothing, msg=nothing)
    msg == nothing && (msg = " in " * string(:( [$(flat...)] )))
    seen = Set{Symbol}()
    for i in flat
        (i in seen) ? throw(MacroError("index $i repeated" * msg, call)) : push!(seen, i)
    end
end

function checkallseen(canon, store, call)
    left = setdiff(canon, unique!(store.seen))
    length(left) > 0 && throw(MacroError("index $(left[1]) appears only on the left", call))
    right = setdiff(store.seen, canon)
    length(right) > 0 && throw(MacroError("index $(right[1]) appears only on the right", call))
end

# this may never be necessary with checkallseen?
function findcheck(i::Symbol, flat::Vector, call=nothing, msg=nothing)
    msg == nothing && (msg = " in " * string(:( [$(flat...)] )))
    res = findfirst(isequal(i), flat)
    res == nothing ? throw(MacroError("can't find index $i" * msg, call)) : return res
end
findcheck(i, flat, call=nothing, msg=nothing) =
    throw(MacroError("expected a single symol not $i", call))


#==================== The Output Functions ====================#

function newoutput(ex, canon, parsed, store::NamedTuple, call::CallInfo)
    isempty(parsed.reversed) || throw(MacroError("can't reverse along indices on LHS right now", call))
    isempty(parsed.shuffled) || throw(MacroError("can't shuffle along on LHS right now", call))
    any(iscolon, parsed.outer) && throw(MacroError("can't have colons on the LHS right now", call))
    any(isrange, parsed.outer) && throw(MacroError("can't have ranges on the LHS right now", call))

    # Is there a reduction?
    if :reduce in call.flags
        if :scalar in call.flags
            ex = :( $(parsed.redfun)($ex) )
        else
            dims = length(parsed.rdims)>1 ? Tuple(parsed.rdims) : parsed.rdims[1]
            ex = :( dropdims($(parsed.redfun)($ex, dims=$dims), dims=$dims) )
        end
        canon = deleteat!(copy(canon), sort(parsed.rdims))
    end

    # Were we asked to slice the output?
    if length(parsed.inner) != 0
        code = Tuple(Any[ i in parsed.innerflat ? (:) : (*) for i in canon ])
        if parsed.static
            sizeorcode = maybestaticsizes(canon, code, store)
            ex = :( TensorCast.static_slice($ex, $sizeorcode) )
        elseif :collect in call.flags
            ex = :( TensorCast.slicecopy($ex, $code) )
            push!(call.flags, :collected)
        else
            ex = :( TensorCast.sliceview($ex, $code) )
        end
        # Now I allow fixing output indices
        if any(isconstant, parsed.inner)
            any(i -> isconstant(i) && !(i == :_ || i == 1), parsed.inner) && throw(MacroError("can't fix output index to something other than 1", call))
            code = Tuple(map(i -> isconstant(i) ? (*) : (:), parsed.inner))
            Asafe = maybepush(ex, store, :outfix)
            ex = :(TensorCast.orient.($Asafe, Ref($code)) ) # @. would need a dollar
        end
    end

    # Must we collect? Do this now, as reshape(PermutedDimsArray(...)) is awful.
    if :collect in call.flags && !(:collected in call.flags)
        ex = :( collect($ex) )
    end

    # Do we need to reshape the container? Using orient() avoids needing sz_i
    if any(i -> istensor(i) || isconstant(i), parsed.outer)
        any(i -> isconstant(i) && !(i == :_ || i == 1), parsed.outer) && throw(MacroError("can't fix output index to $i, only to 1", call))
        if any(istensor, parsed.outer)
            ex = :( reshape($ex, ($(parsed.outsize...),)) )
            append!(store.need, parsed.flat)
        else
            code = Tuple(map(i -> isconstant(i) ? (*) : (:), parsed.outer))
            ex = :( TensorCast.orient($ex, $code) )
        end
    end

    # Is the result Diagonal or friends? Doesn't allow Z[i,i,1] or Z[i,-i] but that's OK
    if length(parsed.outer)==2 && parsed.outer[1]==parsed.outer[2]
        if :nolazy in call.flags
            ex = :( TensorCast.diagm(0 => $ex) )
        else
            ex = :( TensorCast.Diagonal($ex) )
        end
    end

    return ex
end

function inplaceoutput(ex, canon, parsed, store::NamedTuple, call::CallInfo)
    if !(parsed.static)
        isempty(parsed.inner) || throw(MacroError("can't write in-place into slices right now",call))
        any(iscolon, parsed.outer) && throw(MacroError("can't have colons on the LHS right now",call))
    end

    out = []
    Zsym = gensym(:reverse)

    # We can re-use exactly the same "standardise" function as for RHS terms:
    pop!(call.flags, :nolazy, :ok) # ensure we use diagview(), Reverse{}, etc, not a copy

    if @capture(parsed.left, zed_[]) # special case Z[] = ... else allconst pulls it out
        zed isa Symbol || @capture(zed, ZZ_.field_) || error("wtf")
        str = "expected a 0-tensor $zed[]"
        push!(store.mustassert, :( TensorCast.@assert_ ndims($zed)==0 $str) )
    else
        newleft = standardise(parsed.left, store, call)
        @capture(newleft, zed_[ijk__]) || throw(MacroError("failed to parse LHS correctly, $(parsed.left) -> $newleft"))

        if !(zed isa Symbol) # then standardise did something!
            push!(call.flags, :showfinal)
            Zsym = gensym(:reverse)
            push!(out, :( local $Zsym = $zed ) )
            zed = Zsym
        end
    end

    # Now write into that, either sum!(Z,...) or mul!(Z,A,B) or Z .= ..., no copyto!(Z,...)
    if :reduce in call.flags
        redfun! = endswith(string(parsed.redfun),'!') ? parsed.redfun : Symbol(parsed.redfun, '!')
        push!(out, :( $redfun!($zed, $ex) ) )
    elseif :matmul in call.flags
        ex isa Tuple || error("wtf?")
        push!(out, :( $mul!($zed, $(ex[1]), $(ex[2])) ) )
    else
        push!(out, :( $zed .= $ex ) )
    end

    if :showfinal in call.flags
        push!(out, parsed.name)
    end

    return out
end

#==================== The End ====================#
