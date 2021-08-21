
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
  by `n = i + (j-1) * N` where `i ∈ 1:N`. This may also be written `B[i⊗j,k]`.
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
  which gets broadcasted `h.(I,J')`, then glued to make a 3-tensor.
* `K[i,j]'` conjugates each element, equivalent to `K'[j,i]` which is the
  conjugate-transpose of the matrix.
* `M[i,i]` means `diag(M)[i]`, but only for matrices: `N[i,i,k]` is an error.
* `P[i,i']` is normalised to `P[i,i′]` with unicode \\prime.
* `R[i,-j,k]` means roughly `reverse(R, dims=2)`, and `Q[i,~j,k]` similar with `shuffle`.

The left and right hand sides must have all the same indices,
and the only repeated index allowed is `M[i,i]`, which is a diagonal not a trace.
See `@reduce` and `@matmul` for related macros which can sum over things.

If a function of one or more tensors appears on the right hand side,
then this represents a broadcasting operation,
and the necessary re-orientations of axes are automatically inserted.

The following actions are possible:
* `=` writes into an existing array, overwriting its contents,
  while `+=` adds (precisely `Z .= Z .+ ...`) and `*=` multiplies.
* `:=` creates a new array. To omit the name, write `Z = @cast _[i,j,k] := ...`.
* `|=` insists that the result is an `Array` not a view, or some other lazy wapper.
  (This may still be a `reshape` of the input, it does not guarantee a copy.)

Options specified at the end (if several, separated by `,`) are:
* `i in 1:3` or `i ∈ 1:3` supplies the range of index `i`. Variables and functions like `j in 1:Nj, k in 1:length(K)`
  are allowed, but `i = 1:3` is not.
* `lazy=false` disables `PermutedDimsArray` in favour of `permutedims`, 
  and `Diagonal` in favour of `diagm` for `Z[i,i]` output.

Some modifications to broadcasting are possible, after loading the corresponding package:
* `@cast @strided Z[i,j] := ...` uses Strided.jl's macro, for multi-threaded broadcasting.
* `@cast @turbo Z[i,j] := ...` uses LoopVectorization.jl's macro, for SIMD acceleration.
* `@cast @lazy Z[i,j] := ...` uses LazyArrays.jl's BroadcastArray type, although there is no such macro.

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
  and reshaping `B[i⊗j,k]`. See `? @cast` for the complete list.
* If there is a function of one or more tensors on the right,
  then this is a broadcasting operation.
* Index ranges may be given afterwards (as for `@cast`) or inside the reduction `sum(i:3, k:4)`.
* All indices appearing on the right must appear either within `sum(...)` etc, or on the left.


    F = @reduce sum(i,j)  B[i] + γ * D[j]         # sum(B .+ γ .* D')
    @reduce G[] := sum(i,j)  B[i] + γ * D[j]      # F == G[]

Complete reduction to a scalar output `F`, or a zero-dim array `G`.
`G[]` involves `sum(A, dims=(1,2))` rather than `sum(A)`.

    @reduce @lazy Z[k] := sum(i,j) A[i] * B[j] * C[k]  (i in 1:N, j in 1:N, k in 1:N)

The option `@lazy` replaces the broadcast expression with a `BroadcastArray`,
to avoid `materialize`ing the entire array before summing. In the example this is of size `N^3`.
This needs `using LazyArrays` to work.

The options `@strided` and `@turbo` will alter broadcasting operations, 
and need `using Strided` or `using LoopVectorization` to work.

    @reduce sum(i) A[i] * log(@reduce _[i] := sum(j) A[j] * exp(B[i,j]))
    @cast W[i] := A[i] * exp(- @reduce S[i] = sum(j) exp(B[i,j]) lazy)

Recursion like this is allowed, inside either `@cast` or `@reduce`.
The intermediate array need not have a name, like `_[i]`,
unless writing into an existing array, like `S[i]` here.
"""
macro reduce(exs...)
    call = CallInfo(__module__, __source__, TensorCast.unparse("@reduce", exs...),
        Set([:reduce]))
    _macro(exs...; call=call)
end

"""
    @matmul M[a,c] := sum(b)  A[a,b] * B[b,c]

Matrix multiplication macro. Uses the same syntax as `@reduce`, but instead of broadcasting
the expression on the right out to a 3-tensor before summing it along one dimension,
this calls `*` or `mul!`, which is usually much faster.
But it only works on expressions of suitable form.

Note that unlike `@einsum` and `@tensor`, you must explicitly specify
what indices to sum over.

    @matmul Z[a⊗b,k] = sum(i,j)  D[i,a][j] * E[i⊗j,_,k,b]

Each tensor will be pre-processed exactly as for `@cast` / `@reduce`,
here glueing slices of `D` together, reshaping `E` and the output `Z`.
Once this is done, the right hand side must be of the form `(tensor) * (tensor)`,
which becomes `mul!(ZZ, DD, EE)`.

    @reduce V[i] := sum(k) W[k] * exp(@matmul _[i,k] := sum(j) A[i,j] * B[j,k])

You should be able to use this within the other macros, as shown.
"""
macro matmul(exs...)
    call = CallInfo(__module__, __source__, TensorCast.unparse("@matmul", exs...),
        Set([:matmul, :lazy_0]))
    _macro(exs...; call=call)
end


#==================== The Main Functions ====================#

function _macro(exone, extwo=nothing, exthree=nothing; call::CallInfo=CallInfo(), dict=Dict())
    store = (dict=dict, assert=[], mustassert=[], seen=[], need=[], top=[], main=[])
    # TODO use OrderedDict() for main? To allow duplicate removal

    if Meta.isexpr(exone, :macrocall)
        # New style @cast @avx A[i] := B[i]
        string(exone.args[1]) in ("@lazy", "@strided", "@avx", "@avxt", "@turbo", "@tturbo") || throw(MacroError(
            "the macro $(exone.args[1]) isn't one of the ones this understands", call))
        push!(call.flags, Symbol(string(exone.args[1])[2:end]), :premacro)
        return _macro(exone.args[3:end]...; call=call, dict=dict)
    end

    if (:reduce in call.flags) || (:matmul in call.flags)
        # First the options:
        optionparse(exthree, store, call)
        # Then the LHS, to get canonical list of indices:
        canon, parsed = reduceparse(exone, extwo, store, call)
    elseif containsindexing(extwo) # @cast A[i,j] := softmax(j) B[i,j]
        push!(call.flags, :dimcast)
        optionparse(exthree, store, call)
        canon, parsed = reduceparse(exone, extwo, store, call)
    else
        # Simple @cast case:
        isnothing(exthree) || throw(MacroError("too many expressions for @cast: $exthree", call))
        optionparse(extwo, store, call)
        canon, parsed = castparse(exone, store, call)
    end

    # First pass over RHS just to read sizes, prewalk sees A[i][j] before A[i]
    MacroTools.prewalk(x -> rightsizes(x, store, call), parsed.right)

    # To look for recursion, we need another prewalk. To find naked indices, this one stops early:
    right2 = recursemacro(parsed.right, canon, store, call)

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
but also pushes `A = f(x)` into `store.top`.
"""
function standardise(ex, store::NamedTuple, call::CallInfo; LHS=false)
    @nospecialize ex

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

    # Constant indices A[i,3], A[i,_], A[i,$c]
    if isempty(ijk)
        # empty indices do not count as constant - this fixes some issues with 0d CuArrays
    elseif all(isconstant, ijk)
        needview!(ijk)                                         # replace _ with 1, if any
        Asym = maybepush(:( $A[$(ijk...)] ), store, :allconst) # protect from later processing
        return Asym                                            # and nothing more to do here
    elseif any(isconstant, ijk)
        ijcolon = map(i -> isconstant(i) ? i : (:), ijk)
        if needview!(ijcolon)
            A = :( view($A, $(ijcolon...)) ) # TODO lazy_0?
        else
            # A = :( TensorCast.rview($A, $(ijcolon...)) )
            perm = filter(!isnothing, ntuple(d -> ijcolon[d]==(:) ? d : nothing, length(ijcolon)))
            A = :( TensorCast.transmute($A, Base.Val($perm)) )
        end
        ijk = filter(!isconstant, ijk)              # remove actual constants from list,
    end

    # Slicing operations A[i,:,j], and A[i,3:5] after the above treatment
    if any(iscolon, ijk)
        code = Tuple(map(i -> iscolon(i) ? (:) : (*), ijk))
        if static 
            sizeorcode = maybestaticsizes(ijk, code, call)
            A = :( TensorCast.static_slice($A, $sizeorcode) )
        elseif (:lazy_0 in call.flags) || (:collect in call.flags)
            A = :( TensorCast.slicecopy($A, $code) )
        else
            A = :( TensorCast.sliceview($A, $code) )
        end
        ijk = filter(!iscolon, ijk)
    elseif static
        throw(MacroError("shouldn't use curly brackets here", call))
    end

    # Nested indices A[i,j,B[k,l,m],n] or worse A[i,B[j,k],C[i,j]]
    if any(i -> @capture(i, B_[klm__]), ijk)
        throw(MacroError("indexing one array by another is not supported, sorry", call))
    end

    # Diagonal extraction A[i,i]
    if length(ijk)==2 && ijk[1]==ijk[2]
        if (:lazy_0 in call.flags) && !LHS # don't do this for in-place output
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
        flatsize = map(axwrap, flat)
        A = :( Base.reshape($A, ($(flatsize...),)) )
        append!(store.need, flat)
    end

    # Reversed A[-i,j] and shuffled A[i,~j]
    if !isempty(reversed)
        A = maybepush(A, store, :prereverse)
        rind = map(1:length(flat)) do d
            flat[d] in reversed ? :($reverse(Base.axes($A,$d))) : (:)
        end
        rdims = Tuple(indexin(reversed, flat))
        if (:lazy_0 in call.flags) && !LHS
            if length(rdims) == 1
                A = :( reverse($A, dims=$(rdims[1])) )
            elseif VERSION >= v"1.6-"
                A = :( reverse($A, dims=$rdims) )
            else
                A = :( A[$(rind...)] )
            end
            push!(call.flags, :collected)
        else
            A = :( @view($A[$(rind...)]) )
            pop!(call.flags, :collected, :ok)
        end
    end
  
    if !isempty(shuffled)
        A = maybepush(A, store, :preshuffle)
        sind = map(1:length(flat)) do d
            flat[d] in shuffled ? :($shuffle(Base.axes($A,$d))) : (:)
        end
        if (:lazy_0 in call.flags) && !LHS
            if length(flat) == 1
                A = :( $shuffle($A) )
            else
                A = :( A[$(sind...)] )
            end
            push!(call.flags, :collected)
        else
            A = :( @view($A[$(sind...)]) )
            pop!(call.flags, :collected, :ok)
        end
    end

    A = maybepush(A, store, :standardise)  # else A may be seen to contain indexing, causing confusion

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
    @nospecialize ex

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
            ex = :( $Bsym =  @__dot__ view($B, $(ijcolon...)) ) # TODO lazy_0 here
        else
            # ex = :( $Bsym = @__dot__ TensorCast.rview($B, $(ijcolon...)) )
            perm = filter(!isnothing, ntuple(d -> ijcolon[d]==(:) ? d : nothing, length(ijcolon)))
            ex = :( $Bsym = TensorCast.transmute.($B, Base.Val($perm)) )
        end
        push!(store.main, ex)
        B = Bsym
        inner = filter(!isconstant, inner)
    end

    # And some quick error checks; standsrdise() already put inner into store.seen
    append!(store.seen, inner)
    checknorepeats(vcat(inner, outer), call,
        " in gluing " * string(:([$(outer...)])) * string(:([$(inner...)])))

    # Now we glue, always in A[k,l][i,j] -> B[i,j,k,l] order:
    ijk = vcat(inner, outer)

    if static
        AB = :( TensorCast.static_glue($B) )
        pop!(call.flags, :collected, :ok)
    elseif :lazy_0 in call.flags
        AB = :( TensorCast.stack_iter($B) ) # really from LazyStack
        push!(call.flags, :collected)
    else # if :lazy in call.flags
        AB = :( TensorCast.stack($B) )
        pop!(call.flags, :collected, :ok)
    end

    return :( $AB[$(ijk...)] )
end

"""
    targetcast(A[i] + B[j], [i,j]) -> @__dot__(A + B')
    targetcast(A[j,i], [i,j]) -> transpose(A)

This brings the expression to have target indices,
by permutedims and if necessary broadcasting, always using `readycast()`.
"""
function targetcast(ex, target, store::NamedTuple, call::CallInfo)
    @nospecialize ex

    # If just one naked expression, then we won't broadcast:
    if @capture(ex, A_[ijk__])
        containsindexing(A) && error("that should have been dealt with")
        return readycast(ex, target, store, call)
    end

    # But for anything more complicated, we do:
    ex = MacroTools.prewalk(x -> readycast(x, target, store, call), ex)

    if :lazy in call.flags
        ex = :( LazyArrays.BroadcastArray(@__dot__ LazyArrays.lazy($ex)) )
        pop!(call.flags, :collected, :ok)
        #=
        Eventually it will be OK not to materialize the BroadcastArray before summing,
        this PR does it for complete sum(): https://github.com/JuliaLang/julia/pull/31020
        =#
    elseif :strided in call.flags
        ex = :( Strided.@strided @__dot__($ex) )
        push!(call.flags, :collected)
    elseif :avx in call.flags
        ex = :( LoopVectorization.@avx @__dot__($ex) )
        push!(call.flags, :collected)
    else
        ex = :( @__dot__($ex) )
        push!(call.flags, :collected)
    end

    return ex
end

"""
    readycast(A[j,i], [i,j,k]) -> transpose(A)
    readycast(A[k], [i,j,k]) -> orient(A, (*,*,:))

This is walked over the expression to prepare for `@__dot__` etc, by `targetcast()`.
"""
function readycast(ex, target, store::NamedTuple, call::CallInfo)
    @nospecialize ex

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
    # tuple creation... now including namedtuples
    @capture(ex, (args__,) ) && any(containsindexing, args) &&
        if any(a -> @capture(a, sym_ = val_), args)
            syms, vals = [], []
            map(args) do a
                @capture(a, sym_ = val_ ) || throw(MacroError("invalid named tuple element $a", call))
                push!(syms, QuoteNode(sym))
                push!(vals, val)
            end
            return :( NamedTuple{($(syms...),)}(tuple($(vals...))) )
        else
            return :( tuple($(args...)) )
        end
    # ternary operator
    @capture(ex, cond_ ? yes_ : no_ ) &&
        return ternarycast(cond, yes, no, target, store, call)
    # and arrays of functions, using apply:
    @capture(ex, funs_[ijk__](args__) ) &&
        return :( Core._apply($funs[$(ijk...)], $(args...) ) )
    # splats
    @capture(ex, fun_(pre__, arg_...)) && containsindexing(arg) && begin
        @gensym splat ys
        xs = [gensym(Symbol(:x, i)) for i in 1:length(pre)]
        push!(store.main, :( local $splat($(xs...), $ys) = $fun($(xs...), $ys...) ))
        return :( $splat($(pre...), $arg) )
    end

    # Apart from those, readycast acts only on lone tensors:
    @capture(ex, A_[ijk__]) || return ex

    dims = Int[ findcheck(i, target, call, " on the left") for i in ijk ]

    # Both orient and permutedims are now rolled into transmute:
    if !isempty(dims)
        perm = ntuple(d -> findfirst(isequal(d), dims), maximum(dims))
        if perm != ntuple(identity, maximum(dims))
            if :lazy_0 in call.flags
                A = :( TensorCast.transmutedims($A, $perm) )
                push!(call.flags, :collected)
            else
                A = :( TensorCast.transmute($A, Base.Val($perm)) )
                if ! increasing_or_zero(perm) # thus not just a reshape
                    pop!(call.flags, :collected, :ok)
                end
            end
        end
    end

    # Does A need protecting from @__dot__?
    A = maybepush(A, store, :nobroadcast)

    return A
end

function ternarycast(cond, yes, no, target, store::NamedTuple, call::CallInfo)
    args = []
    cond, yes, no = map((cond, yes, no)) do expr
        MacroTools.prewalk(expr) do ex
            rex = readycast(ex, target, store, call)
            @capture(ex, A_[ijk__]) && push!(args, rex)
            rex
        end
    end
    unique!(args)
    fun = gensym(:ternary)
    push!(store.main, :( local $fun($(args...)) = $cond ? $yes : $no ))
    :($fun($(args...)))
end

"""
    matmultarget(A[i,j] * B[j,k], [i,k]) -> A * B

For inplace, instead it returns tuple `(A,B)`, which `inplaceoutput()` can use.

If there are more than two factors, it recurses, and you get `(A*B) * C`,
or perhaps tuple `(A*B, C)`. But I'm going to remove this as I never got it working perfectly?
"""
function matmultarget(ex, target, parsed, store::NamedTuple, call::CallInfo)
    @nospecialize ex

    @capture(ex, A_ * B_ * C__ | *(A_, B_, C__) ) || throw(MacroError("can't @matmul that!", call))

    # Figure out what to sum over, and make A,B into matrices ready for *
    iA = guesstarget(A, [], [])
    iB = guesstarget(B, [], [])

    isum = sort(intersect(iA, iB, parsed.reduced),
        by = i -> findfirst(isequal(i), target)) # or target? parsed.reduced
    iAnosum = setdiff(iA, isum)
    iBnosum = setdiff(iB, isum)

    Aex = targetcast(A, vcat(iAnosum, isum), store, call)
    Bex = targetcast(B, vcat(isum, iBnosum), store, call)

    Aex = matrixshape(Aex, iAnosum, isum, store, call)
    Bex = matrixshape(Bex, isum, iBnosum, store, call)

    Aex = maybepush(Aex, store, :mulA)
    Bex = maybepush(Bex, store, :mulB)

    # But don't actually * if you are going to mul!(Z,A,B) instead later:
    if (:inplace in call.flags) && length(C)==0
        return (Aex, Bex, iAnosum, iBnosum)
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
        throw(MacroError("@matmul no longer supports more than two factors", call))
        # But if we have (A*B) * C then continue, whether in- or out-of-place:
+       # ABCex = :( *($ABind, $(C...)) )
+       # return matmultarget(ABCex, target, parsed, store, call)
    end
end

"""
    recursemacro(@reduce sum(i) A[i,j]) -> G[j]

Walks itself over RHS to look for `@reduce ...`, and replace with result,
pushing calculation steps into store.

Also a convenient place to tidy all indices, including e.g. `fun(M[:,j],N[j]).same[i']`.
And to handle naked indices, `i` => `axes(M,1)[i]` but not exactly like that.
"""
function recursemacro(ex::Expr, canon, store::NamedTuple, call::CallInfo)

    # The original purpose was to look for recursion, meaning @reduce within @cast etc:
    if @capture(ex, @reduce(subex__) )
        subcall = CallInfo(call.mod, call.src,
            TensorCast.unparse("inner @reduce", subex...), Set([:reduce, :recurse]))
        name, ind, scalar, steps = _macro(subex...; call=subcall, dict=store.dict)
        append!(store.main, steps)
        ex = scalar ? :($name) :  :($name[$(ind...)])

    elseif @capture(ex, @matmul(subex__) )
        subcall = CallInfo(call.mod, call.src,
            TensorCast.unparse("inner @matmul", subex...), Set([:matmul, :recurse]))
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
        if !(A isa Symbol)  # this check allows some tests which have c[c] etc.
            A = recursemacro(A, canon, store, call)
        end
        return :( $A[$(tensorprimetidy(ijk)...)] )
    elseif @capture(ex, A_{ijk__})
        if !(A isa Symbol)
            A = recursemacro(A, canon, store, call)
        end
        return :( $A{$(tensorprimetidy(ijk)...)} )

    # Walk inwards: not handled by prewalk, as we do NOT want to walk inside indexing
    elseif isprimedindex(ex, canon)
        return recursemacro(tensorprimetidy(ex), canon, store, call)
    elseif Meta.isexpr(ex, :$)
        return only(ex.args)
    elseif ex isa Expr
    #     && ex.head in [:call, Symbol("'")]
    #     return Expr(ex.head, map(x -> recursemacro(x, canon, store, call), ex.args)...)
    # elseif ex isa Expr
    #     @warn "not a call, what is it?:" ex
        return Expr(ex.head, map(x -> recursemacro(x, canon, store, call), ex.args)...)
    else
        return ex
    end
end

function recursemacro(i, canon, store::NamedTuple, call::CallInfo)
    @nospecialize i

    i in canon || return i
    # For naked indices, replace i with roughly axes(A,1)[i] etc:
    push!(store.need, i)
    ax = axwrap(i)
    return :( $ax[$i] )
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
    @nospecialize ex

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
    elseif @capture(ex, right_ => left_ )
        throw(MacroError("anonymous functions using => currently not supported", call))
        # @capture(right, AA_[ii__] | AA_[ii__].ff_ | AA_{ii__} ) ||
        #     throw(MacroError("anonymous functions => only accept simpler input A[...] => ...", call))
        # push!(call.flags, :anonfunc) # TODO make the output understand this
    elseif reduce
        return (Any[], nothing)
    else
        error("wtf is $ex")
    end

    static = @capture(left, ZZ_{ii__})

    if @capture(left, Z_[outer__][inner__] | [outer__][inner__] | Z_[outer__]{inner__} | [outer__]{inner__} )
        if isnothing(Z)
            (:inplace in call.flags) && throw(MacroError("can't write into a nameless tensor", call))
            @warn "please write `@cast _[i][k] := ...` to omit a name, instead of `@cast [i][k] := ...`" call.string maxlog=3 
            Z = :_ # gensym(:output)
        end
        Z = (Z == :_) ? gensym(:output) : Z
        parsed = indexparse(Z, outer, store, call, save=(:inplace in call.flags))
        innerflat = innerparse(:(first($Z)), inner, store, call, save=(:inplace in call.flags))
        canon = vcat(innerflat, parsed.flat)
        checknorepeats(canon, call, " on the left")

    elseif @capture(left, Z_[outer__] | [outer__] )
        if isnothing(Z)
            (:inplace in call.flags) && throw(MacroError("can't write into a nameless tensor", call))
            @warn "please write `@cast _[i] := ...` to omit a name, instead of `@cast [i] := ...`" call.string maxlog=3 
            Z = :_ 
        end
        Z = (Z == :_) ? gensym(:output) : Z
        parsed = indexparse(Z, outer, store, call, save=(:inplace in call.flags))
        canon = parsed.flat
        inner, innerflat = [], []

    elseif left isa Symbol # only for @reduce A := sum(i) ...
        (:reduce in call.flags) || throw(MacroError("@cast really needs an indexing expression on left!", call))
        (:inplace in call.flags) && throw(MacroError("scalar output needs @reduce Z := ...", call))
        Z = left
        push!(call.flags, :scalar)
        parsed = indexparse(Z, [], store, call) # just to get the right namedtuple
        canon, outer, inner, innerflat = [], [], [], []

    else
        throw(MacroError("don't know what to do with left = $left", call))
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
        push!(call.flags, :scalar)
        parsed = (indexparse(nothing, [])..., name=gensym(:noleft), outer=[], inner=[], static=false)
    else
        @capture(parsed.right, redfun_(redlist__) ) || throw(MacroError("how do I reduce over $(parsed.right) ?", call))
    end

    # Then parse redlist, decoding ranges like sum(i:10,j) which specify sizes
    reduced = []
    for item in tensorprimetidy(redlist) # normalise θ'
        i = @capture(item, j_:s_) ? saveonesize(j, :(Base.OneTo($s)), store) : item
        push!(reduced, i)
    end
    checknorepeats(reduced, call, " in the reduction") # catches sum(i,i) B[i,j,k]

    # Now work out canonical list. For sum!(), reduced indices must come last.
    if :inplace in call.flags
        canon = vcat(leftcanon, reduced)
    else
        # But for Z = sum(A, dims=...) can try to avoid permutedims, not sure it matters.
        guess = guesstarget(ex2, leftcanon, reduced)
        guessminus = copy(guess)
        for i in reduced
            deleteat!(guessminus, findcheck(i, guessminus, call, " on the right"))
        end
        if leftcanon == guessminus  # i.e. leftcanon is an ordered subset of guess
            canon = guess
            # canon == vcat(leftcanon, reduced) || @info "guesstarget did something!" repr(leftcanon) repr(reduced) repr(guess) ex2
        else
            canon = vcat(leftcanon, reduced)
        end
    end
    if !(:dimcast in call.flags)
        checknorepeats(canon, call, " in the reduction") # catches A[i] := sum(i)
    end

    # Finally, record positions in canon of reductions
    rdims = sort([ findfirst(isequal(i), canon) for i in reduced ])

    return canon, (parsed..., redfun=redfun, reduced=reduced, rdims=rdims, right=ex2) # this replaces parsed.right
end

"""
    p = indexparse(A, [i,j,k])

`p.flat` strips constants, colons, tensor/brackets, minus, tilde, and doubled indices.
`p.outaces` is a list like `[ax_i, 1, star(...)]` for use on LHS.
`p.reversed` has all those with a minus.
Sizes are saved to `store` only with keyword `save=true`, i.e. only when called by `rightsizes()`
"""
function indexparse(A, ijk::Vector, store=nothing, call=nothing; save=false)
    flat, outaxes, reversed, shuffled = [], [], [], []

    ijk = tensorprimetidy(ijk) # un-wrap i⊗j to tuples, and normalise i' to i′

    stripminustilde!(ijk, reversed, shuffled)

    for (d,i) in enumerate(ijk)

        if iscolon(i)
            if i isa QuoteNode && A != :_
                str = "fixed size in $A[" * join(ijk, ", ") * "]" # DimensionMismatch("fixed size in M[i, \$(QuoteNode(5))]: size(M, 2) == 5") TODO print more nicely
                pushboundscheck!(store.mustassert, :( Base.size($A,$d)==$(i.value) || throw(DimensionMismatch($str))) )
            end
            continue
        end

        if isconstant(i)
            push!(outaxes, Base.OneTo(1))
            if i == :_ && A != :_ && save
                str = "underscore in $A[" * join(ijk, ", ") * "]"
                pushboundscheck!(store.mustassert, :( Base.size($A,$d)==1 || throw(DimensionMismatch($str))) )
            end
            continue
        end

        if @capture(i, (ii__,))
            stripminustilde!(ii, reversed, shuffled)
            append!(flat, ii)
            push!(outaxes, axwrap(ii))
            save && A != :_ && saveonesize(ii, :(Base.axes($A, $d)), store)

        elseif @capture(i, B_[klm__])
            innerparse(B, klm, store, call) # called just for error on tensor/colon/constant
            sub = indexparse(B, klm, store, call; save=save) # I do want to save size(B,1) etc.
            append!(flat, sub.flat)

        elseif i isa Symbol
            push!(flat, i)
            push!(outaxes, axwrap(i))
            save && A != :_ && saveonesize(i, :(Base.axes($A, $d)), store)
        else
            throw(MacroError("don't understand index $i", call))
        end
    end

    if save && length(ijk)>0 && A != :_
        N = length(ijk)
        if N == 1
            str = "expected a vector or tuple $A[" * join(ijk, ", ") * "]"
            pushboundscheck!(store.assert, :( $A isa Tuple || Base.ndims($A)==$N || Base.throw(ArgumentError($str))) )
        else
            str = "expected a $N-tensor $A[" * join(ijk, ", ") * "]"
            pushboundscheck!(store.assert, :( Base.ndims($A)==$N || Base.throw(ArgumentError($str))) )
        end
    end

    if length(flat)==2 && flat[1]==flat[2] # allow for diag, A[i,i]
        pop!(flat)
    end

    checknorepeats(flat, call)

    reversed = filter(i -> isodd(count(isequal(i), reversed)), reversed)
    shuffled = unique(shuffled)

    return (flat=flat, outaxes=outaxes, reversed=reversed, shuffled=shuffled)
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
        istensor(i) && throw(MacroError("can't tensor product inner indices", call))
        (@capture(i, -j_) || @capture(i, ~j_)) && throw(MacroError("can't reverse or shuffle along inner indices", call))

        if @capture(i, j_:s_)
            push!(innerflat, j)
            saveonesize(j, :(Base.OneTo($s)), store) # save=true on LHS only for in-place, save this anyway
        elseif isconstant(i)
            i == :_ && save && pushboundscheck!(store.mustassert, :( size($firstA, $d)==1 || throw(DimensionMismatch("inner underscore"))) )
        else
            push!(innerflat, i)
        end
        save && saveonesize(i, :(Base.axes($firstA, $d)), store)
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

    if @capture(opt, i_ in ax_) || @capture(opt, i_ ∈ ax_)
        if @capture(ax, 1:s_)
            saveonesize(tensorprimetidy(i), :(Base.OneTo($s)), store)
        elseif ax isa Number
            @warn "did you mean `$i in 1:$ax`, not `$i in $ax`?"
            saveonesize(tensorprimetidy(i), :(Base.OneTo($ax)), store)
        else
            ax1 = maybepushtop(ax, store, :axis)
            push!(store.top, :($ax1 isa AbstractUnitRange || Base.throw(DimensionMismatch("index ranges must have step 1"))))
            if isdefined(call.mod, :OffsetArrays)
                ax2, off = gensym(:axis), gensym(:offset)
                push!(store.top, :(local $off = first($ax1)-1))
                push!(store.top, :(local $ax2 = $ax1 isa Base.OneTo ? $ax1 : OffsetArrays.IdOffsetRange($ax1 .- $off, $off)))
                saveonesize(tensorprimetidy(i), ax2, store)
            else
                pushboundscheck!(store.top, :(first($ax1)==1 || Base.throw(ArgumentError("you must load OffsetArrays to allow index ranges not starting at 1"))))
                saveonesize(tensorprimetidy(i), :(Base.OneTo($ax1)), store)
            end
        end
        push!(call.flags, :assert)
    elseif @capture(opt, lazy = val_Number) && 0 <= val <= 2
        push!(call.flags, Symbol(:lazy_, Int(val)))
    elseif @capture(opt, i_:s_)
        @warn "please replace index ranges like `i:3` with `i in 1:3` or `i ∈ 1:3`" call.string maxlog=3 
        saveonesize(tensorprimetidy(i), :(Base.OneTo($s)), store)
        push!(call.flags, :assert)
    elseif opt in (:strided, :avx)
        @warn "postfix option $opt is deprecated, please write @cast @$opt A[i] := ..." call.string maxlog=3 
        push!(call.flags, opt)
    elseif opt == :lazy
        @warn "postfix option `lazy` is deprecated, please write " * 
            "`@lazy A[i] := ...` for LazyArrays broadcasting, or " * 
            "`lazy=true` to use PermutedDimsArray-like arrays (the default)" call.string maxlog=3 
        push!(call.flags, :lazy, :lazy_1)
    elseif opt == :nolazy
        @warn "option `nolazy` is deprecated, please write keyword style `lazy=false` to disable PermutedDimsArray etc." call.string maxlog=3 
        push!(call.flags, :lazy_0)
    elseif opt in (:assert, :(!))
        @warn "option `assert` is no longer needed, this is the default" call.string maxlog=3 
    else
        throw(MacroError("don't understand option $opt", call))
    end
end

#==================== Smaller Helper Functions ====================#

"""
    saveonesize(:i, axes(A,3), store) -> :i

Saves sizes for indices `:i` or products like  `[:i,:j]` to `store`.
It it already has a size for this single index,
then save an assertion that new size is equal to old.
"""
function saveonesize(ind, ax, store::NamedTuple)
    if !haskey(store.dict, ind)
        store.dict[ind] = ax
    elseif store.dict[ind] != ax  # no need to save identical expressions
        if isa(ind, Symbol)
            str = "range of index $ind must agree"
            pushboundscheck!(store.assert, :( $(store.dict[ind]) == $ax || Base.throw(DimensionMismatch($str))) )
        end
    end
    ind
end

function findsizes(store::NamedTuple, call::CallInfo)
    out = []
    append!(out, store.assert)
    empty!(store.assert)
    if length(store.need) > 0
        inferred = sizeinfer(store, call)
        ax_list = map(axwrap, store.need)
        push!(out, :( local ($(ax_list...),) = ($(inferred...),) ) )
    end
    append!(out, store.mustassert) # NB do this after calling sizeinfer()
    unique!(out)
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

    # Second pass looks for tuples where exactly one entry has unknown length
    for pair in store.dict
        if isa(pair.first, Vector)
            known = [ haskey(store.dict, j) for j in pair.first ]

            if sum(.!known) == 1 # bingo! now work out its size:
                num = takelength(pair.second)

                denfacts = [ store.dict[i] for i in pair.first[known] ]
                if length(denfacts) > 1
                    # den = :( prod(length, ($(denfacts...),)) )
                    longs = map(takelength, denfacts)
                    den = :( Base.:*($(longs...)) )
                else
                    den = takelength(denfacts[1])
                end
                rat = :( Base.OneTo($num ÷ $den) )

                i = pair.first[.!known][1]
                d = findfirst(isequal(i), store.need)
                d != nothing && (sizes[d] = rat)

                str = "expected integer multiples, when calculating range of $i from range of $(join(pair.first, " ⊗ "))"
                pushboundscheck!(store.mustassert, :( ($num % $den)==0 || Base.throw(ArgumentError($str))) )
            end
        end
    end

    unknown = store.need[sizes .== (:)]
    str = join(unknown, ", ")
    isempty(unknown) || throw(MacroError("unable to infer ranges for indices $str", call))

    return sizes
end

"""
    takelength(OntTo(n)) -> n
    takelength(axes(A,2)) -> size(A,2)
"""
function takelength(ex)
    if Meta.isexpr(ex, :call) && ex.args[1] in (Base.OneTo, :(Base.OneTo))
        ex.args[2]
    elseif Meta.isexpr(ex, :call) && ex.args[1] in (:axes, axes, :(Base.axes))
        @assert length(ex.args) == 3
        :(Base.size($(ex.args[2:end]...)))
    else
        :(Base.length($ex))
    end
end

"""
    maybestaticsizes([:, :, i], (:,:,*)) -> (:,:,*)
    maybestaticsizes([:3, :4, i], (:,:,*)) -> Size(3,4)
Produces the 2nd argument of `static_slice()`, for slicing `A{:3, :4, i}`.
"""
function maybestaticsizes(ijk::Vector, code::Tuple, call::CallInfo)
    iscodesorted(code) || throw(MacroError("static slices need all colons to the left, " *
            "got {$(pretty(ijk))} hence code = $(pretty(code))", call))
    length(ijk) == length(code) || error("wrong length of code!")
    staticsize = Any[ i.value for i in ijk if i isa QuoteNode ]
    if length(staticsize) == count(iscolon, ijk)
        return :( TensorCast.Size($(staticsize...)) ) # really StaticArrays.
    else
        return code
    end
end

"""
    maybestaticsizes([:i,:j,:k], (:,:,*), store) -> Size(3,4)
Produces the 2nd argument of `static_slice()`, using sizes from `store.dict` if available.
"""
function maybestaticsizes(ijk::Vector, code::Tuple, store::NamedTuple, call::CallInfo)
    iscodesorted(code) || throw(MacroError("static slices need all colons to the left, " *
            "got {$(pretty(ijk))} hence code = $(pretty(code))", call))
    length(ijk) == length(code) || error("wrong length of code!")
    staticsize = []
    for d=1:countcolons(code)
        if haskey(store.dict, ijk[d])
            ax = store.dict[ijk[d]]
            push!(staticsize, :(length($ax)))
        else
            return code
        end
    end
    return :( TensorCast.Size($(staticsize...)) )
end

"""
    A = maybepush(ex, store, :name)
If `ex` is not just a symbol, then it pushes `:(Asym = ex)` into `store.main`
and returns `Asym`. Unless this already has a name, in which case it returns that.
"""
maybepush(s::Symbol, any...) = s
function maybepush(ex::Expr, store::NamedTuple, name::Symbol=:A)
    for prev in store.main
        @capture(prev, local sym_ = $ex) && return sym
    end
    Asym = gensym(name)
    push!(store.main, :( local $Asym = $ex ) )
    return Asym
end

maybepushtop(s::Symbol, any...) = s
function maybepushtop(ex::Expr, store::NamedTuple, name::Symbol=:top)
    Asym = gensym(name)
    push!(store.top, :( local $Asym = $ex ) )
    return Asym
end

"""
    pushboundscheck!(list, ex)
Approximately `push!`, but skips duplicate checks, and adds `@boundscheck`.
"""
function pushboundscheck!(list, ex::Expr)
    z = _getcheck(ex)
    for prev in list
        _getcheck(prev) == z && return list
    end
    push!(list, :(@boundscheck $ex))
end
function _getcheck(ex::Expr)
    ex.args[1] == Symbol("@boundscheck") && return _getcheck(ex.args[3])
    Meta.isexpr(ex.args[1], :call) && ex.args[1].args[1] in (:isa, :(==)) && return ex.args[1]
    return NaN  # because NaN == NaN is false
end

tensorprimetidy(v::Vector) = Any[ tensorprimetidy(x) for x in v ]
function tensorprimetidy(ex)
    MacroTools.postwalk(ex) do @nospecialize x
        @capture(x, ((ij__,) \ k_) ) && throw("i\\j is no longer accepted, please write i⊗j or (i,j)")
        @capture(x, i_ \ j_ ) && throw("i\\j is no longer accepted, please write i⊗j or (i,j)")

        @capture(x, ((ij__,) ⊗ k_) ) && return :( ($(ij...),$k) )
        @capture(x, i_ ⊗ j_ ) && return :( ($i,$j) )

        @capture(x, ((ij__,), k__) ) && return :( ($(ij...),$(k...)) )

        @capture(x, i_') && return Symbol(i,"′")
        x
    end
end

function isprimedindex(ex::Expr, canon)
    ex.head == Symbol("'") || return false
    t = tensorprimetidy(ex)  # this may not be a symbol
    return t in canon
end
isprimedindex(s::Symbol, canon) = s in canon
isprimedindex(any, canon) = false

axwrap(i::Symbol) = Symbol(:ax_,i)
function axwrap(ijk::Vector)
    length(ijk) == 0 && return nothing
    length(ijk) == 1 && return axwrap(first(ijk))
    return :( TensorCast.star($(map(axwrap, ijk)...)) )
end

isconstant(n::Int) = true
isconstant(s::Symbol) = s == :_
isconstant(ex::Expr) = ex.head == :($)
isconstant(q::QuoteNode) = false

isindexing(s) = false
isindexing(ex::Expr) = @capture(x, A_[ijk__])

isCorI(i) = isconstant(i) || isindexing(ii)

istensor(n::Int) = false
istensor(s::Symbol) = false
function istensor(ex::Expr)
    @capture(ex, i_' )     && return istensor(i)
    @capture(ex, -(ij__) ) && return length(ij)>1 # TODO maybe reject -(i,j) ... istensor(:( -i⊗j )) is OK without this line
    @capture(ex, i_⊗j_ )   && return true
    @capture(ex, i_\j_ )   && return throw("i\\j is no longer accepted, please write i⊗j or (i,j)")
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
    MacroTools.postwalk(ex) do @nospecialize x
        # @capture(x, A_[ijk__]) && !(all(isconstant, ijk)) && (flag=true)
        if @capture(x, A_[ijk__]) || @capture(x, A_{ijk__})
            # @show x ijk # TODO this is a bit broken?  @pretty @cast Z[i,j] := W[i] * exp(X[1][i] - X[2][j])
            flag=true
        end
    end
    flag
end

listindices(s::Symbol) = []
function listindices(ex::Expr)
    list = []
    MacroTools.postwalk(ex) do @nospecialize x
        if @capture(x, A_[ijk__])
            flat, _ = indexparse(nothing, ijk)
            push!(list, flat)
        end
        x
    end
    list
end

listsymbols(s::Symbol, target) = s in target ? [s] : Symbol[]
listsymbols(any, target) = Symbol[]
function listsymbols(ex::Expr, target)
    ex.head == :vec && return Symbol[]
    return union((listsymbols(a, target) for a in ex.args)...)
end

function guesstarget(ex::Expr, left, red)
    list = sort(listindices(ex), by=length, rev=true)
    naked = listsymbols(ex, vcat(left, red))
    unique(vcat(list..., naked))  # TODO make a smarter version which tries to fit to left + red?  
end

# function overlapsorted(x,y) # works fine but not in use yet
#     z = intersect(x,y)
#     length(z) ==0 && return true
#     xi = map(i -> findfirst(isequal(i),x), z)
#     @assert xi == indexin(z, x)
#     yi = map(i -> findfirst(isequal(i),y), z)
#     @assert xi == indexin(z, y)
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
        # elseif ij[k] isa Expr && @capture(ij[k], alpha_:omega_) # i.e. isrange(ij[k])
        #     out = true
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
    # Deal with simple matrix, and with empty right of V in M*V
    length(left) == 1 && length(right) <= 1 && return ex
    # and empty right because it's M * vec(T)
    isempty(right) && return :( Base.reshape($ex, :) )

    # Deal with empty left of V in V'*M, or perhaps V=vec(T) first
    if isempty(left)
        if length(right) == 1
            # return :( TensorCast.PermuteDims($ex) )
            return :( Base.transpose($ex) )
        else
            # return :( TensorCast.PermuteDims(reshape($ex, :)) )
            return :( Base.transpose(Base.reshape($ex, :)) )
        end
    end

    # Otherwise we are going to need to reshape
    left_sz = axwrap(left)
    right_sz = axwrap(right) # this is the product!
    append!(store.need, left)
    append!(store.need, right)
    # push!(call.flags, :reshaped)
    return :( Base.reshape($ex, ($left_sz,$right_sz)) )
end

function unmatrixshape(ex, left::Vector, right::Vector, store::NamedTuple, call::CallInfo)
    # Easy cases M*M and M*V
    length(left) == 1 && length(right) <= 1 && return ex

    # For V' * V, did we want a scalar or not? What we will get is unknown to the macro:
    if length(left) == 0 && length(right) == 0
        if :scalar in call.flags
            return :( Base.first($ex) )
            # If you had arrays of arrays, then PermuteDims would have permutedims-ed, and * would make an array, so this is still OK.
        else
            return :( Base.fill($ex) ) # zero-dim array
        end
    end

    # For V'*M, you may get a Transpose row-vector, for which this is .parent:
    if length(left) == 0
        ex = :( TensorCast.transmute($ex, Base.Val((2,))) )
        length(right) == 1 && return ex # literally V'*M done,
        # but for V'*T we should reshape this .parent
    end

    # For anything more complicated, we will need to reshape:
    sizes = vcat(map(axwrap, left), map(axwrap, right))
    append!(store.need, left)
    append!(store.need, right)
    # push!(call.flags, :reshaped)
    return :( Base.reshape($ex, ($(sizes...),)) )
end

function increasing_or_zero(tup::Tuple, prev=0)  # strictly increasing, allows nothing or 0
    d = first(tup)
    d === nothing && return increasing_or_zero(Base.tail(tup), prev)
    (d != 0) && (d <= prev) && return false
    return increasing_or_zero(Base.tail(tup), max(d, prev))
end
increasing_or_zero(::Tuple{}, prev=0) = true

#==================== Nice Errors ====================#

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

    # Is there a reduction?
    if :reduce in call.flags
        if :scalar in call.flags
            ex = :( $(parsed.redfun)($ex) )
        else
            dims = length(parsed.rdims)>1 ? Tuple(parsed.rdims) : parsed.rdims[1]
            # perm = Tuple(filter(d -> !(d in parsed.rdims), 1:length(canon)))
            # ex = :( TensorCast.transmute($(parsed.redfun)($ex, dims=$dims), $perm) )
            ex = :( Base.dropdims($(parsed.redfun)($ex, dims=$dims), dims=$dims) )
            if :strided in call.flags
                pop!(call.flags, :collected, :ok) # makes stridedview(...
            end
        end
        canon = deleteat!(copy(canon), sort(parsed.rdims))
    elseif :dimcast in call.flags
        dims = length(parsed.rdims)>1 ? Tuple(parsed.rdims) : parsed.rdims[1]
        ex = :( $(parsed.redfun)($ex, dims=$dims) )
    end

    # Were we asked to slice the output?
    if length(parsed.inner) != 0
        code = Tuple(Any[ i in parsed.innerflat ? (:) : (*) for i in canon ])
        if parsed.static
            sizeorcode = maybestaticsizes(canon, code, store, call)
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
            # code = Tuple(map(i -> isconstant(i) ? (*) : (:), parsed.inner))
            Asafe = maybepush(ex, store, :outfix)
            _d = 0
            perm = Tuple(map(i -> isconstant(i) ? nothing : (_d+=1), parsed.inner))
            # ex = :(TensorCast.orient.($Asafe, Ref($code)) ) # @. would need a dollar
            # refperm = maybepush(:( Ref() ), store, :zzz)
            ex = :(TensorCast.transmute.($Asafe, Base.Val($perm)) )
        end
    end

    # Must we collect? Do this now, as reshape(TransmutedDimsArray(...)) is awful.
    if :collect in call.flags
        if :strided in call.flags
            ex = :( Base.collect($ex) )
        elseif !(:collected in call.flags)
            ex = :( Base.identity.($ex) )
        end
    end

    # Do we need to reshape the container? Simple cases done with transmute(), avoiding sz_i
    if any(i -> istensor(i) || isconstant(i), parsed.outer)
        any(i -> isconstant(i) && !(i == :_ || i == 1), parsed.outer) && throw(MacroError("can't fix output index to $i, only to 1", call))
        if any(istensor, parsed.outer)
            ex = :( Base.reshape($ex, ($(parsed.outaxes...),)) )
            append!(store.need, parsed.flat)
        else
            _d = 0
            perm = Tuple(map(i -> isconstant(i) ? nothing : (_d+=1), parsed.outer))
            ex = :( TensorCast.transmute($ex, Base.Val($perm)) )
        end
    end

    # Is the result Diagonal or friends? Doesn't allow Z[i,i,1] or Z[i,-i] but that's OK
    if length(parsed.outer)==2 && parsed.outer[1]==parsed.outer[2]
        if :lazy_0 in call.flags
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
    pop!(call.flags, :lazy_0, :ok) # ensure we use diagview(), Reverse{}, etc, not a copy

    if @capture(parsed.left, zed_[]) # special case Z[] = ... else allconst pulls it out
        zed isa Symbol || @capture(zed, ZZ_.field_) || error("wtf")
        newleft = parsed.left
        str = "expected a 0-tensor $zed[]"
        pushboundscheck!(store.mustassert, :( Base.ndims($zed)==0 || Base.throw(ArgumentError($str))) )
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
    elseif :dimcast in call.flags
        redfun! = endswith(string(parsed.redfun),'!') ? parsed.redfun : Symbol(parsed.redfun, '!')
        dims = length(parsed.rdims)>1 ? Tuple(parsed.rdims) : parsed.rdims[1]
        push!(out, :( $redfun!($zed, $ex; dims=$dims) ) )
    elseif :matmul in call.flags
        ex isa Tuple || error("wtf?")

        zmul = targetcast(newleft, vcat(ex[3], ex[4]), store, call)
        zmul = matrixshape(zmul, ex[3], ex[4], store, call)
        zmul isa Symbol || push!(call.flags, :showfinal)
        # zed = maybepush(zed, out, :mul!)
        push!(out, :( TensorCast.mul!($zmul, $(ex[1]), $(ex[2])) ) )
    else
        push!(out, :( $zed .= $ex ) )
    end

    if :showfinal in call.flags
        push!(out, parsed.name)
    end

    return out
end

#==================== The End ====================#
