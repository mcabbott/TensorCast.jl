
export @cast, @cast!

"""
    @cast A[i,j] := B[i] + γ * D[j]             # A = B .+ γ .* D'
    @cast A[i,j] =  B[i] * log( C[i,j] / D[j] ) # A .= B .* log.(C ./ D')

This macro lets you write broadcasting operations in index notation.
The necessary re-orientations of axes (like `D'` here) are automatically inserted. 

    @cast E[i] := prod(j) B[i] + ε * C[i,j]     # E = vec(prod(B .+ ε .* C, dims=2))
    @cast E[i] = sum(j) exp( C[i,j] / D[j] )    # sum!(E, exp.(C ./ D') )

You can specify reductions in a syntax much like `@reduce`.
However, while the broadcast operations shown always materialises a new array to store the result
before summing, `@cast` will avoid this, the broadcast is done lazily.
"""
macro cast(ex...)
    _cast(ex...; icheck=false)
end

"""
    @cast! A[i,j] := B[i] + γ * D[j]

Variant of `@cast` which effectively runs `@check()` on each tensor.
"""
macro cast!(ex...)
    where = (mod=__module__, src=__source__)
    _cast(ex...; icheck=true, where=where)
end

function _cast(ex; icheck=false, where=nothing)
    if @capture(ex, left_ := right_ )
        sign = :( := )
    elseif @capture(ex, left_ = right_ )
        sign = :( = )
    else error("@cast can't understand $ex")
    end

    outex = quote end

    @capture(left, nameZ_[indZ__] ) || error("@cast can't understand $left")
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

    # length(outex.args) == 1 && return esc(outex.args[1]) # it has #= ... =# in it too
    esc(outex)
end


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
            push!(outex.args, orientex(Asym, check_one(x), dirs))
            ex =  :( $Asym )
        end
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



# function orient(A::AbstractArray{T,N}, code::Tuple) where {T,N}
#     Ncode = countcolons(code)
#     N == Ncode || error("expected an array with ndims = $Ncode, but got ndims = $N")
#     d = 1
#     sz = ntuple(i -> code[i]==(:) ? (d+=1; size(A,d-1)) : 1, length(code))
#     reshape(A, sz)::AbstractArray{T,length(code)}
# end
# TODO: think about rowvector like objects? 

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
    d-1 == ndims(A) || throw(ArgumentError("orient(A, ($str)) got ndims(A) = $(ndims(A)), expeceted n = $(d-1)"))
    :(reshape(A, ($(list...),))) 
end



function _cast(ex1, rhs; icheck=false)
    @capture(ex1, nameZ_[indZ__] := red_(indZred__)) || @capture(ex1, nameZ_[indZ__] = red_(indZred__)) || error("can't read $ex1")
    @info "@cast reducing" nameZ indZ red indZred rhs
end


function padex(A, n::Int)
    ex = :( (size($A)...,) )
    @assert ex.head == :tuple
    for i=1:n
        push!(ex.args, 1)
    end
    ex
end





