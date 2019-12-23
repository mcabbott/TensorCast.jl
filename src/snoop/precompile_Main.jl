function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Main, Symbol("#exp_all#101")) && precompile(Tuple{getfield(Main, Symbol("#exp_all#101")),Array{Float64,2}})
    isdefined(Main, Symbol("#g#105")) && precompile(Tuple{getfield(Main, Symbol("#g#105")),Array{Int64,1}})
end
