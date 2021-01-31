"""
    @capture_(ex, A_[ijk__])

Faster drop-in replacement for `MacroTools.@capture`, for a few patterns only:
* `A_[ijk__]` and `A_{ijk__}`
* `[ijk__]`
* `A_.field_`
* `A_ := B_` and  `A_ = B_` and `A_ += B_` etc.
* `f_(x_)`
"""
macro capture_(ex, pat::Expr)

    H = QuoteNode(pat.head)

    A,B = if pat.head in [:ref, :curly] && length(pat.args)==2 &&
        _endswithone(pat.args[1]) && _endswithtwo(pat.args[2]) # :( A_[ijk__] )
        _symbolone(pat.args[1]), _symboltwo(pat.args[2])

    elseif pat.head == :. &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2].value) # :( A_.field_ )
        _symbolone(pat.args[1]), _symbolone(pat.args[2].value)

    elseif pat.head == :call  && length(pat.args)==2 &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2]) # :( f_(x_) )
        _symbolone(pat.args[1]), _symbolone(pat.args[2])

    elseif pat.head in [:call, :(=), :(:=), :+=, :-=, :*=, :/=] &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2]) # :( A_ += B_ )
        _symbolone(pat.args[1]), _symbolone(pat.args[2])

    elseif pat.head == :vect && _endswithtwo(pat.args[1]) # :( [ijk__] )
        _symboltwo(pat.args[1]), gensym(:ignore)

    else
        error("@capture_ doesn't work on pattern $pat")
    end

    @gensym res
    quote
        $A, $B = nothing, nothing
        $res = TensorCast._trymatch($ex, Val($H))
        # $res = _trymatch($ex, Val($H))
        if $res === nothing
            false
        else
            $A, $B = $res
            true
        end
    end |> esc
end

_endswithone(ex) = endswith(string(ex), '_') && !_endswithtwo(ex)
_endswithtwo(ex) = endswith(string(ex), "__")

_symbolone(ex) = Symbol(string(ex)[1:end-1])
_symboltwo(ex) = Symbol(string(ex)[1:end-2])

_getvalue(::Val{val}) where {val} = val

_trymatch(s, v) = nothing # Symbol, or other Expr
_trymatch(ex::Expr, pat::Union{Val{:ref}, Val{:curly}}) = # A_[ijk__] or A_{ijk__}
    if ex.head === _getvalue(pat)
        ex.args[1], ex.args[2:end]
    else
        nothing
    end
_trymatch(ex::Expr, ::Val{:.}) = # A_.field_
    if ex.head === :.
        ex.args[1], ex.args[2].value
    else
        nothing
    end
_trymatch(ex::Expr, pat::Val{:call}) =
    if ex.head === _getvalue(pat) && length(ex.args) == 2
        ex.args[1], ex.args[2]
    else
        nothing
    end
_trymatch(ex::Expr, pat::Union{Val{:(=)}, Val{:(:=)}, Val{:(+=)}, Val{:(-=)}, Val{:(*=)}, Val{:(/=)}}) =
    if ex.head === _getvalue(pat)
        ex.args[1], ex.args[2]
    else
        nothing
    end
_trymatch(ex::Expr, ::Val{:vect}) = # [ijk__]
    if ex.head === :vect
        ex.args, nothing
    else
        nothing
    end

# Cases for Tullio:
# @capture(ex, B_[inds__].field_) --> @capture_(ex, Binds_.field_) && @capture_(Binds, B_[inds__])
# @capture(ex, [inds__])

#=

julia> ex = :(Z[1,2,3])

julia> @pretty @capture(ex, A_[ijk__])
begin
    A = MacroTools.nothing
    ijk = MacroTools.nothing
    tarsier = trymatch($(Expr(:copyast, :($(QuoteNode(:(A_[ijk__])))))), ex)
    if tarsier == MacroTools.nothing
        false
    else
        A = get(tarsier, :A, MacroTools.nothing)
        ijk = get(tarsier, :ijk, MacroTools.nothing)
        true
    end
end

julia> @pretty @capture_(ex, A_[ijk__])
begin
    A = nothing
    ijk = nothing
    louse = _trymatch(ex)
    if louse == nothing
        false
    else
        (A, ijk) = louse
        true
    end
end



ex = :( A[i,j][k] + B[I[i],J[j],k]^2 / 2 )
f1(x) = MacroTools.postwalk(ex) do x
    @capture(x, A_[ijk__]) || return x
    :($A[$(ijk...),9])
    end
f2(x) = MacroTools.postwalk(ex) do x
    @capture_(x, A_[ijk__]) || return x
    :($A[$(ijk...),9])
    end
f1(ex)
f2(ex)

@btime f1(x) setup=(x=ex) # 3.181 ms
@btime f2(x) setup=(x=ex) #    31.440 μs -- 100x faster.


$ time julia -e 'using TensorCast; TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  ))'

real    0m8.132s  # was 0m8.900s on master, noise or signal?
user    0m7.747s
sys 0m0.358s
real    0m8.132s
user    0m8.295s
sys 0m0.329s

$ time julia -e 'using TensorCast; @time TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  ))'

4.899634 seconds, best run # was 5.845 on master, that's a second?

=#
