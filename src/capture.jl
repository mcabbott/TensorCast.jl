"""
    @capture_(ex, A_[ijk__])

A faster drop-in replacement for `MacroTools.@capture`, for this particular pattern only.
"""
macro capture_(ex, pat::Expr)
    pat.head == :ref &&
        length(pat.args)==2 &&
        endswith(string(pat.args[1]), '_') &&
        endswith(string(pat.args[2]), "__") || error("@capture_ only works on pattern A_[ijk__]")

    A = Symbol(string(pat.args[1])[1:end-1])
    ijk = Symbol(string(pat.args[2])[1:end-2])
    @gensym res
    quote
        $A, $ijk = nothing, nothing
        $res = TensorCast._trymatch($ex)
        if $res == nothing
            false
        else
            $A, $ijk = $res
            true
        end
    end |> esc
end

_trymatch(s) = nothing
function _trymatch(ex::Expr)
    ex.head == :ref || return nothing
    ex.args[1], ex.args[2:end]
end

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

real    0m8.567s  # was 0m9.485s
user    0m8.295s
sys 0m0.329s

=#
