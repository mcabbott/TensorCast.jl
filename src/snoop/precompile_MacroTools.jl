function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    precompile(Tuple{typeof(MacroTools.blockunify),Expr,Tuple{typeof(*),Colon}})
    precompile(Tuple{typeof(MacroTools.match),Symbol,Nothing,Dict{Any,Any}})
    precompile(Tuple{typeof(MacroTools.trymatch),Expr,Tuple{Colon,typeof(*),typeof(*)}})
    precompile(Tuple{typeof(MacroTools.trymatch),Expr,Tuple{typeof(*),Colon}})
end
