
module _Base
    include("snoop/precompile_Base.jl")
end
_Base._precompile_()

module _LinearAlgebra
    using LinearAlgebra
    include("snoop/precompile_LinearAlgebra.jl")
end
_LinearAlgebra._precompile_()

module _StaticArrays
    using LinearAlgebra, StaticArrays
    include("snoop/precompile_StaticArrays.jl")
end
_StaticArrays._precompile_()

module _MacroTools
    using MacroTools
    include("snoop/precompile_MacroTools.jl")
end
_MacroTools._precompile_()

module _TensorCast
    using MacroTools, LinearAlgebra, RecursiveArrayTools, StaticArrays, TensorCast
    using TensorCast: CallInfo, _macro, cat_glue, copy_glue, lazy_glue, red_glue, static_glue, slicecopy, sliceview, static_slice
    include("snoop/precompile_TensorCast.jl")
end
_TensorCast._precompile_()


#=

# https://timholy.github.io/SnoopCompile.jl/stable/snoopi/

using SnoopCompile
using Pkg
Pkg.activate("TensorCast")

pa = joinpath(Base.pathof(TensorCast), "../../")
te = normpath(joinpath(pa, "test/runtests.jl"))

si = @snoopi tmin=0.01 include(te)

pc = SnoopCompile.parcel(si)

fi = normpath(joinpath(pa, "src/snoop/"))
SnoopCompile.write(fi, pc)

# This makes precompilation take 29 sec instead of 8 sec,
# while loading remains 3-4 sec.

# This still takes 5s... vs 7s with nothing:
# @time TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  ))

# I tried adding exactly its commands, but no help:
si = @snoopi tmin=0.01 TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  ))

=#
