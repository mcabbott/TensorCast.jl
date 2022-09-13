
using TensorCast

using Test
using LinearAlgebra
using Logging

using StaticArrays
using OffsetArrays
using Einsum
using Strided
using LazyArrays
using LoopVectorization

@testset "ex-@shape" begin include("shape.jl") end
@testset "@reduce" begin include("reduce.jl")  end
@testset "@cast"   begin include("casting.jl") end
@testset "@matmul" begin include("mul.jl")     end

@testset "slice/view"  begin include("cat.jl") end
@testset "old readmes" begin include("old.jl") end
@testset "new in 0.2"  begin include("two.jl") end
@testset "new in 0.4" begin include("four.jl") end
@testset "einops comp" begin include("einops_comparison.jl") end
