
using TensorCast
using Test
using LinearAlgebra
using StaticArrays
using Einsum
using Compat

@testset "ex-@shape" begin include("shape.jl") end
@testset "@reduce" begin include("reduce.jl")  end
@testset "@cast"   begin include("casting.jl") end
@testset "@matmul" begin include("mul.jl")     end

@testset "slice/view"  begin include("cat.jl") end
@testset "old readmes" begin include("old.jl") end
@testset "new in 0.2"  begin include("two.jl") end
