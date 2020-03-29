"""
    @capture_(ex, A_[ijk__])

A faster drop-in replacement for `MacroTools.@capture`, for this particular pattern only.
"""
macro capture_(ex, pat::Expr)

    pat.head in [:ref, :curly] &&
        length(pat.args)==2 &&
        endswith(string(pat.args[1]), '_') &&
        endswith(string(pat.args[2]), "__") ||
        error("@capture_ doesn't work on pattern $pat")

    A = Symbol(string(pat.args[1])[1:end-1])
    ijk = Symbol(string(pat.args[2])[1:end-2])

    qn = QuoteNode(pat.head)

    @gensym res
    quote
        $A, $ijk = nothing, nothing
        # $res = TensorCast._trymatch($ex, Val($qn))
        $res = _trymatch($ex, Val($qn))
        if $res == nothing
            false
        else
            $A, $ijk = $res
            true
        end
    end |> esc
end

_trymatch(s, v) = nothing # s::Symbol
_trymatch(ex::Expr, ::Val{:ref}) = # A_[ijk__]
    if ex.head === :ref
        ex.args[1], ex.args[2:end]
    else
        nothing
    end
_trymatch(ex::Expr, ::Val{:curly}) = # A_{ijk__}
    if ex.head === :curly
        ex.args[1], ex.args[2:end]
    else
        nothing
    end


    # elseif pat.head == :call && pat.args[1] === :|
    #     length(pat.args)==3 || error("@capture_ doesn't work on pattern $pat") # Or syntax
    #     for i in 2:3
    #         pat.args[i].head in [:ref, :curly] &&
    #             length(pat.args[i].args)==2 &&
    #             endswith(string(pat.args[i].args[1]), '_') &&
    #             endswith(string(pat.args[i].args[2]), "__") ||
    #             error("@capture_ doesn't work on pattern $pat")

    #         A = Symbol(string(pat.args[i].args[1])[1:end-1])
    #         ijk = Symbol(string(pat.args[i].args[1])[1:end-2])
    #     end
    # end
# _trymatch(ex::Expr, ::Val{:call}) = # A | B
#     if ex.head === :call && ex.args[1] === :|
#         ex.args[1], ex.args[2:end]
#     else
#         nothing
#     end


#     elseif pat.head === :call && pat.args[1] === :| # Or syntax
#         left = _trymatch(pat.args[2], ex)
#         if left !== nothing
#             return left
#         else
#             return right = _trymatch(pat.args[3], ex)
#         end
#     end
#     nothing
# end

# || pat.head === :curly # Patten is A_[ijk__] or A_{ijk__}
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

real    0m8.828s  # was 0m9.485s
user    0m8.295s
sys 0m0.329s

$ time julia -e 'using TensorCast; @time TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  ))'

5.194806 seconds

=#
