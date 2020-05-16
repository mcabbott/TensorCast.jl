
@testset "colons, mapslices etc" begin

    M = randn(3,4)

    @cast C1[i][j] := M[i,j]
    @cast C2[i][j] |= M[i,j]
    @cast C3[i] := M[i,:]
    @cast C4[i] |= M[i,:4] assert

    @test first(C1) isa SubArray
    @test first(C2) isa Array
    @test first(C3) isa SubArray
    @test first(C4) isa Array

    @cast R1[j]{i} := M[i,j]     # size from M
    @cast R2[j]{i:3} := M[i,j]   # size inside
    @cast R3[j]{i} := M[i,j] i:3 # size outside
    @cast R4[j] := M{:3,j}       # size on the colon
    @cast R5[j] := M{:,j}        # size from M

    @test first(R1) isa SArray
    @test first(R2) isa SArray
    @test first(R3) isa SArray
    @test first(R4) isa SArray
    @test first(R5) isa SArray

    γ = 3
    @cast R6[j]{i:γ} |= M[i,j] # test both |= and γ
    @cast R7[j] |= M{:γ,j} assert

    @test R5 isa Base.ReinterpretArray
    @test R6 isa Array
    @test R7 isa Array

    S1 = vec(sum(M, dims=1))
    @cast S2[j] := sum(M[:,j])
    @cast S3[j] := sum(M{:3,j})
    @reduce S4[j] := sum(i) M[i,j]
    @test S1 == S2 == S3 == S4

    @cast M1[i,j] := copy(M[:,j])[i]
    @cast M2[i,j] := copy(M{:3,j})[i]
    @cast M3[i,j] := copy(M{:,j})[i]  # size from M
    @cast M4[i,j] := copy(M{:,j}){i}  # static glue
    @test M == M1 == M2 == M3 == M4

    fun(x) = (sum=sum(x), same=x, one=1)

    @cast M5[i,j] := fun(M[:,j]).same[i]
    @cast M6[i,j] := fun(M{:3,j}).same{i}
    @cast M7[i,j] := fun(M[:,j] * 1).same[i]
    @test M == M5 == M6 == M7

    @cast S5[j] := fun(M[:,j]).sum
    @test S1 == S5

    @cast D1[i,j] := 2(C1[i][j])
    @cast D2[i,j] := copy(C1[i] + 1 * C1[i])[j]
    @test 2M == D1 == D2

end
@testset "fields & sizes" begin

    B = [ (scal=i, vect=[1,2,3,4]) for i=1:3 ]
    @cast A[j,i] := B[i].vect[j]  i:3, j:4
    @test A[4,1] == 4

    @cast C[i] := B[i].scal
    @test C == 1:3

end
@testset "int/bool indexing" begin

    ind = [1,2,1,2]
    M = rand(1:99, 3,4)

    N1 = M[1,ind]
    @cast N2[i] := M[1,ind[i]]
    @test N1 == N2

    N3 = M[ind,ind]
    @cast N4[i,j] := M[ind[i],ind[j]]
    @test N3 == N4

    bool = ind .== 1
    B1 = M[:,bool]
    @cast B2[i,j] := M[i,bool[j]]
    @test B1 == B2

    V1 = M[CartesianIndex.(ind, ind)]
    # @cast V2[i] := M[ind[i],ind[i]]

end
@testset "indexing with a range" begin

    M = rand(1:99, 3,4)

    # ranges are treated first like constants, then like colons:
    @cast C1[i,j] := identity(M[i, 1:2])[j]
    R1 = 1:2
    @cast C2[i,j] := M[i, R1[j]] # explicit inner indexing
    @test C1 == C2 == M[:, 1:2]

    β = 2
    @cast C3[i,j] := identity(M[i, 1:β])[j]
    @test C3 == C1

end
@testset "in-place += and *=" begin

    A = ones(2,3)
    B = rand(3,2)

    @cast A[i,j] += B[j,i]
    @test A == 1 .+ B'

    C = 2 .* ones(3)
    D = rand(3)

    @cast C[i] *= D[i] + 1
    @test C == 2 .* (D .+ 1)

    M = ones(3,3)

    @cast M[i,i] -= D[i] + 5
    @test diag(M) == 1 .- (D .+ 5)

end
@testset "fixing inner indices" begin

    imgs = [ rand(28,28,1,4) for i=1:2 ];
    @cast Z[i,j,k,l] := imgs[l][i,j,_,k];

    @test size(Z) == (28,28,4,2)
    @test Z[2,3,4,1] == imgs[1][2,3,1,4]

    @cast out[l][i,j,_,k] := Z[i,j,k,l];
    @test out == imgs

    uno = 1 # try again with $constant
    @cast Z[i,j,k,l] := imgs[l][i,j,$uno,k];

    @test size(Z) == (28,28,4,2)
    @test Z[2,3,4,1] == imgs[1][2,3,1,4]

    @cast out[l][i,j,1,k] := Z[i,j,k,l];
    @test out == imgs

end
@testset "scalars & 0-arrays" begin

    B = rand(2,3)
    @reduce S := sum(i,j) sqrt(B[j,i])
    S′ = @reduce sum(i,j) sqrt(B[j,i])
    @test S isa Number
    @test S ≈ S′ ≈ sum(sqrt, B)

    @reduce T[] := sum(i,j) sqrt(B[j,i])
    @test T isa Array{Float64,0}
    @test T[] ≈ sum(sqrt, B)

    @reduce T1[_] := sum(i,j) sqrt(B[j,i])
    @test T1 isa Array{Float64,1}
    @test T1[1] ≈ sum(sqrt, B)

    V = rand(4)
    @cast TV[i] := V[i] + T[]
    @test TV ≈ V .+ S

    U =  similar(T)
    @reduce U[] = sum(i,j) B[j,i]/2 # inplace
    @test U[] ≈ sum(B)/2

    @cast V[] := 2 * T[]
    @test_broken V isa Array{Float64,0}

end
@testset "tupple broadcasting" begin

    V = 1:4
    @cast T[i,j] := (V[i], V[j]^2 + 10^3, 99)
    @test T[3,4] == (3,1016,99)

    @cast N[i,j] := (a = V[i], b = V[j]^2 + 10^3, c = 99)
    @test N[3,4] == (a=3, b=1016, c=99)

end
@testset "arrays of functions" begin

    funs = [sin, cos, tan]
    vals = 1:3
    @cast M[i,j] := (funs[i])(vals[j])
    @cast V[i] := (funs[i])(vals[i])
    @test M[1,2] == sin(2)
    @test V[2] == cos(2)

    op = [+, -, *, /]
    x = 1:5
    y = 1:6
    @cast T[i,j,k] := op[i](x[j], y[k]) + 10
    @test T[2,3,4] == 3-4 + 10
    @test T[4,5,6] == 5/6 + 10

    @cast A[i] := op[i](π,ℯ)
    @test A[end] == π / ℯ

end
#=
# disabled for v0.3, since I haven't got around to fixing it.
@testset "A * B * C" begin

    A = rand(2,2)
    B = rand(2,2)
    C = rand(2,2)

    @matmul Z[i,l] := sum(j,k) A[i,j] * B[j,k] * C[k,l]
    @test_broken Z == A * B * C
    @test vec(Z) == vec(A * B * C)

end
=#
@testset "string macros" begin

    A = rand(2,10); B = rand(10,10);

    reduce" C_ii := sum_k A_1k * log(A_2i * B_ik) "
    @test C isa Diagonal

end
@testset "from todo list" begin

    list = [ i .* ones(2,2,1) for i=1:8 ];
    @cast mat[x\i, y\j] := Int(list[i\j][x,y,1])  i:2
    @cast mat2[x\i, y\j] := Int(list[i\j][x,y,1])  i:2, lazy # crazy type
    @test mat[3,5] == 6
    @test mat == mat2

    @cast C1[i,i',k] := (1:4)[i⊗i′⊗k] + im  (i:2, i′:2)  # two tensor signs
    @cast C2[i,i',k] := (1:4)[i⊗i'⊗k] + im  (i:2, i′:2)  # more primes
    @cast C3[i,i',k] := (1:4)[i⊗i'⊗k] + im  (i:2, i':2)
    @test C1[1,2,1] == C2[1,2,1] == C3[1,2,1] == 3+1im

    @cast C4[i,i'⊗k] := (1:4)[i⊗i′⊗k] + im  (i:2, i':2)
    @test size(C4) == (2,2)
    @test C4[2,1] == 2+im

    ∇λ = ones(3); topd = rand(3); logind=3; λ=rand(3); d=2
    @cast ∇λ[c] = ∇λ[c] - λ[$d] * exp(topd[c]) / logind # all const
    @test ∇λ[1] == 1 - λ[2] * exp(topd[1]) / 3

    W = rand(3); X = [rand(3) for _=1:3, _=1:2];
    @cast Z[i,j] := W[i] * exp(X[1,1][i] - X[2,2][j]) # all const
    @test Z == @. W * exp(X[1,1] - X[2,2]')

    mat = rand(3,4)
    outer(v::AbstractVector) = v * v'
    @cast out[i⊗i',j] := outer(mat[:,j])[i,i′] i:3, i':3
    @test size(out) == (9,4)

    yy = rand(3,4)
    t = 4
    xx = zeros(9)
    @cast xx[μ⊗ν] = yy[μ,$t] * yy[ν,$t]
    @test xx ≈ vec(yy[:,4] * yy[:,4]')

    A = [collect((1:4) .+ 10i) for i=1:5]
    @cast B[i,j] := getindex(A[i], (2:3)[j])
    @test B == getindex.(A, (2:3)')

    @cast C[i,j] |= mat[j,i] # now identity.() not collect()
    C[2,1] = 99
    @test mat[1,2] != 99

end
@static if VERSION >= v"1.1"
@testset "@avx" begin

    using LoopVectorization

    A = rand(4,5)
    @test exp.(A) ≈ @cast B[i,j] := exp(A[i,j]) avx

    @test exp.(A') ≈ @cast B[i,j] := exp(A[j,i]) avx
    @test exp.(A.+1) ≈ @cast B[i,j] := exp(A[i,j]+1) avx

end
end
@testset "OffsetArrays" begin

    # https://github.com/mcabbott/TensorCast.jl/issues/11
    A = OffsetArray(rand(3, 3, 5, 5), 1:3, 1:3, -2:2, -2:2);
    # A = OffsetArray(rand(3, 3, 5, 5), 1:3, 11:13, -2:2, -2:2);
    B = OffsetArray(rand(3,    5)   , 1:3,        -2:2)
    δ1 = Matrix(I,3,3)
    # δ1 = OffsetArray(Matrix(I,3,3), 11:13, 11:13)
    δ2 = OffsetArray(Matrix(I,5,5), -2:2, -2:2)
    Ap, Bp, δ1p, δ2p = parent(A), parent(B), δ1, parent(δ2);
    # Ap, Bp, δ1p, δ2p = parent(A), parent(B), parent(δ1), parent(δ2);

    @cast Cp[a, b, c, d] := Ap[a, b, c, d] + Bp[a, c] * δ1p[a, b] * δ2p[c, d]
    @cast C[ a, b, c, d] := A[ a, b, c, d] + B[ a, c] * δ1[ a, b] * δ2[ c, d]
    @test C.parent == Cp

end
@testset "Strided" begin

    bcde = rand(2,3,4,5);
    bcde2 = similar(bcde);

    @cast oo[d,e,b,c] |= bcde[b,c,d,e] strided;
    @test size(oo) == (4,5,2,3)
    @test oo[1,3,1,3] == bcde[1,3,1,3]

    oo[1] = 99
    @test bcde[1] != 99 # copy not a view

    @cast g[(y,e),(b,c),x] := bcde[b,c,(x,y),e]  x:1, strided;
    @test size(g) == (20, 6, 1)
    @cast bcde2[b,c,(x,y),e] = g[(y,e),(b,c),x]  strided;
    @test all(bcde2 .== bcde) # used to say "fails when using Strided"

    # https://github.com/mcabbott/TensorCast.jl/issues/2
    A = rand(3,3); B = rand(3,5);
    @reduce C[i, j] := sum(l) A[i, l] * B[l, j] strided
    @test C ≈ A * B
    @reduce D[i, j] |= sum(l) A[i, l] * B[l, j] strided
    @test D isa Array

end
@testset "parse-time errors" begin

    using TensorCast: MacroError, _macro, CallInfo

    @test_throws MacroError _macro(:(  A[i,j,i] := B[i,j]  )) # repeated
    @test_throws MacroError _macro(:(  A[i,j] := B[i,j,-i]  ))
    @test_throws MacroError _macro(:(  A[k] := sum(k)  ),:(  B[k]  ), call=CallInfo(:reduce))

    @test_throws MacroError _macro(:(  A[i,j] := B[k]  )) # "can't find index k on the left"
    @test_throws MacroError _macro(:(  A[i] := sum(k)  ),:(  (B[j]+ C[j])[i]  ), call=CallInfo(:reduce))

    @test_throws MacroError _macro(:(  A[i] := (B[i], c=3)  ))

    @test_throws MacroError _macro(:( A[h, w\n] := B[n][2,-h,w] )) # can't reverse inner
    @test_throws MacroError _macro(:( A[i,j] := B[i][~j] )) # can't shuffle inner
    @test_throws MacroError _macro(:( A[i,-j] := B[i,j] )) # can't reverse left

end
@testset "run-time errors" begin

    B = [ (scal=i, vect=[1,2,3,4]) for i=1:3 ]
    @test_throws DimensionMismatch @cast A[j,i] := B[i].vect[j]  i:99, j:4 # wrong size
    @test_throws DimensionMismatch @cast A[j,i] := B[i].vect[j]  i:3, j:99

    M = randn(3,4)
    fun(x) = (sum=sum(x), same=x, one=1)
    @test_throws DimensionMismatch  @cast M5[i,j] := fun(M[:,j]).same[i]  i:3, j:99
    @test_throws DimensionMismatch  @cast M5[i,j] := fun(M[:99,j]).same[i]  j:4
    # @test_throws DimensionMismatch  @cast M5[i,j] := fun(M[:,j]).same[i]  i:99, j:4 # TODO make this check canonical length?

end
