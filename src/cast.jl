
export @cast, @cast!

# TODO allow scalar output
# TODO allow A[_,i,_,j] output, and B[i,3] input
# TODO think about @strided here
# TODO to treat rowvector-like things, perhaps always call orient, and it can know?

"""
    @cast A[i,j] := B[i] + γ * D[j]             # A = B .+ γ .* D'
    @cast A[i,j] =  B[i] * log( C[i,j] / D[j] ) # A .= B .* log.(C ./ D')

Macro for writing broadcasting operations in index notation.
The necessary re-orientations of axes (like `D'` here) are automatically inserted. 

    @cast E[i] := prod(j) B[i] + ε * C[i,j]     # E = vec(prod(B .+ ε .* C, dims=2))
    @cast E[i] = sum(j) exp( C[i,j] / D[j] )    # sum!(E, exp.(C ./ D') )

Reductions are specified in a syntax much like `@reduce`.
If you are `using LazyArrays`, then `@cast` will avoid materializing a new array to store the result
before summing. 

    F = @cast sum(i,j)  B[i] + γ * D[j]         # sum(B .+ γ .* D')
    @cast G[] := sum(i,j)  B[i] + γ * D[j]      # F == G[]

Complete reduction to a scalar output `F`, or a zero-dim array `G`. 
"""
macro cast(ex...)
    _cast(ex...; icheck=false)
end

"""
    @cast! A[i,j] := B[i] + γ * D[j]

Variant of `@cast` which effectively runs `@check!()` on each tensor.
"""
macro cast!(ex...)
    where = (mod=__module__, src=__source__)
    _cast(ex...; icheck=true, where=where)
end

function _cast(ex; icheck=false, where=nothing) # one expression = un-reduced
    if @capture(ex, left_ := right_ )
        sign = :( := )
    elseif @capture(ex, left_ = right_ )
        sign = :( = )
    else 
        error("@cast can't understand $ex")
    end

    outex = quote end

    @capture(left, nameZ_[indZ__] ) || error(
        "@cast can't understand $left, expected something like A[i,j]")
    if icheck && check_options.size # then check_one returns check!(Z)
        nameZcheck = check_one(left, where)
        push!(outex.args, nameZcheck)
    elseif icheck # only parse-time check
        check_one(left, where)
    end

    new_right = MacroTools.prewalk(x -> walker!(outex, x, indZ, icheck, where), right)

    if sign == :(=)
        bc = Broadcast.__dot__(:($nameZ = $new_right))
        push!(outex.args, bc )
    else
        bc = Broadcast.__dot__(new_right)
        push!(outex.args, :( $nameZ = $bc ))
    end

    esc(outex)
end

function _cast(redleft, right; icheck=false, where=nothing) # two expressions = reduce
    scalarout = false
    if @capture(redleft, nameZ_[indZ__] := redfun_(indZred__))
        sign = :( := )
    elseif @capture(redleft, nameZ_[indZ__] = redfun_(indZred__))
        sign = :( = )
    elseif @capture(redleft, redfun_(indZred__))
        sign = :( := )
        nameZ = gensym(:Z)
        indZ = Any[]
        scalarout = true
    else 
        error("@cast can't understand $redleft, expected something like A[i] := sum(j)")
    end

    indVall = vcat(indZ, indZred) # reduced indices now go at the right
    redWdims = Tuple(length(indZ)+1:length(indVall))

    outex = quote end

    left = :( $nameZ[$(indZ...)] ) # for icheck
    if icheck && check_options.size # then check_one returns check!(Z)
        nameZcheck = check_one(left, where)
        push!(outex.args, nameZcheck)
    elseif icheck # only parse-time check
        check_one(left, where)
    end

    new_right = MacroTools.prewalk(x -> walker!(outex, x, indVall, icheck, where), right)

    bc = Broadcast.__dot__(new_right)
    global BC = bc
    if isdefined(TensorSlice, :LazyArrays)
        V && @info "before LazyArrays" bc
        @assert bc.head == :(.)      # always a dot
        oprator = bc.args[1]         # is the first operator 
        bc.args[2]                   # is its tuple of arguments
        @assert length(bc.args) == 2 # and there's nothing more
        @assert bc.args[2].head == :tuple
        arguments = bc.args[2].args  # is the args of first op
        bc = Expr(:call, :(LazyArrays.BroadcastArray), oprator, arguments...)
        global BC2 = bc
        V && @info "after LazyArrays" bc
    end

    if sign == :(=)
        if !endswith(string(redfun), '!')
            redfun = Symbol(redfun, '!')
        end
        push!(outex.args, :( $redfun($nameZ, $bc) ))
        push!(outex.args, nameZ)
    else
        push!(outex.args, :( $nameZ = dropdims($redfun($bc; dims=$redWdims), dims=$redWdims) ))
    end

    if scalarout
        push!(outex.args, :( $nameZ[] ) )
    end

    esc(outex)
end

#==================== Helper functions ====================#

function walker!(outex, ex, indZ, icheck=false, where=nothing)

    if @capture(ex, A_[ijk__] )
        if icheck
            A = check_one(ex, where)
        end
        Asym = gensym(:A) 

        dirs = [ findcheck(i, indZ) for i in ijk ]

        if dirs == collect(1:length(dirs)) && isa(A, Symbol)
            ex = :( $A ) # avoid this if A = rand(3), or A wrapped in check!()
        elseif !icheck
            push!(outex.args, orientex(Asym, A, dirs)) # makes Asym = orient(A, code)
            ex =  :( $Asym )
        else
            push!(outex.args, orientex(Asym, check_one(ex), dirs))
            ex =  :( $Asym )
        end
    
    elseif @capture(ex, f_(x_Symbol) ) || @capture(ex, f_(x_Int) ) # want to protect these from @. 
        fxsym = gensym(:fx)    # but not when x contains A[i,j]... crude but allows log(2) etc.
        push!(outex.args, :( local $fxsym = $ex ))
        ex = fxsym
    end

    ex
end   


function orientex(Asym, A, dirs)
    code = Vector{Any}(undef, maximum(dirs))
    code .= (*)
    for d in dirs
        code[d] = (:)
    end

    if dirs == [2] || dirs == [2,1]
        return :( local $Asym = transpose($A) )

    elseif dirs==sort(dirs)
        return :( local $Asym = TensorSlice.orient($A, $(code...,)) )

    else
        perm = Tuple(sortperm(dirs)) 
        # r = rand(1,2,3,4); p = sortperm([8,2,6,4]); permutedims(r,p) |> summary # 2×4×3×1
        if perm==(2,1)
            return :( local $Asym = TensorSlice.orient(transpose($A), $(code...,)) )
        else
            return :( local $Asym = TensorSlice.orient(permutedims($A, $perm), $(code...,)) )
        end
    end
end

#==================== Data function ====================#

# function orient(A::AbstractArray{T,N}, code::Tuple) where {T,N}
#     Ncode = countcolons(code)
#     N == Ncode || error("expected an array with ndims = $Ncode, but got ndims = $N")
#     d = 1
#     sz = ntuple(i -> code[i]==(:) ? (d+=1; size(A,d-1)) : 1, length(code))
#     reshape(A, sz)::AbstractArray{T,length(code)}
# end

@generated function countcolons(code::Tuple) # this works well
    n = 0
    for s in code.parameters
        if s == Colon
            n += 1
        end
    end
    n
end

"""
    B = TensorSlice.orient(A, code)
Reshapes `A` such that its nontrivial axes lie in the directions where `code` contains a `:`,
by inserting axes on which `size(B, d) == 1` as needed. 
"""
@generated function orient(A::AbstractArray, code::Tuple)
    list = Any[]
    pretty = Any[] # just for error
    d = 1
    for s in code.parameters
        if s == Colon
            push!(list, :( size(A,$d) ))
            push!(pretty, ":")
            d += 1
        else
            push!(list, 1)
            push!(pretty, "*")
        end
    end
    str = join(pretty, ", ")
    d-1 == ndims(A) || throw(ArgumentError(
        "orient(A, ($str)) got ndims(A) = $(ndims(A)), expeceted n = $(d-1)"))
    :(reshape(A, ($(list...),))) 
end

