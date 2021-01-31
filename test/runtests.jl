
using TensorCast
using Test
using LinearAlgebra
using StaticArrays
using OffsetArrays
using Einsum
using Strided
using LazyArrays
using Compat
if VERSION >= v"1.1"
    using LoopVectorization
end
using TensorCast: @capture_

@testset "ex-@shape" begin include("shape.jl") end
@testset "@reduce" begin include("reduce.jl")  end
@testset "@cast"   begin include("casting.jl") end
@testset "@matmul" begin include("mul.jl")     end

@testset "slice/view"  begin include("cat.jl") end
@testset "old readmes" begin include("old.jl") end
@testset "new in 0.2"  begin include("two.jl") end
