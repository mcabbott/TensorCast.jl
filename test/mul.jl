using Einsum
using LinearAlgebra

@testset "simple" begin

    ## the very basics
    bc = rand(2,3)
    cd = rand(3,4)

    @mul A[b,d] := bc[b,c] * cd[c,d]
    @test A == bc * cd

    B = similar(A)
    @mul B[b,d] = bc[b,c] * cd[c,d] # in-place
    @test B == A

    ## and with vectors
    c = randn(3)
    @mul b[b] := bc[b,c] * c[c]
    @test b == bc * c

    @mul c[c] := b[b] * bc[b,c]
    @test c == vec(b' * bc)

    @mul z[] := b[b] * b[b]
    @test z[] == dot(b,b)

    b′ = similar(b)
    @mul b′[b] = bc[b,c] * c[c] # in-place
    @test b′ == bc * c

    c′ = similar(c)
    @mul c′[c] = b[b] * bc[b,c]

end
@testset "permutedims" begin

    bbb = rand(2,2,2);
    bbb2 = randn(2,2,2);

    @mul    A[i,j] := bbb[i,k,k2] * bbb2[j,k,k2]
    @reduce B[i,j] := sum(k,k2) bbb[i,k,k2] * bbb2[j,k,k2]
    @einsum C[i,j] := bbb[i,k,k2] * bbb2[j,k,k2]
    @test A ≈ B ≈ C

    D = similar(A);
    @mul    D[i,j] = bbb[i,k,k2] * bbb2[j,k,k2] # in-place
    @test A == D


    @mul    A[i,j] := bbb[k2,i,k] * bbb2[j,k,k2]
    @reduce B[i,j] := sum(k,k2) bbb[k2,i,k] * bbb2[j,k,k2]
    @einsum C[i,j] := bbb[k2,i,k] * bbb2[j,k,k2]
    @test A ≈ B ≈ C

    D = similar(A);
    @mul D[i,j] = bbb[k2,i,k] * bbb2[j,k,k2] # in-place
    @test A == D


    cccc = rand(3,3,3,3);
    cccc2 = randn(3,3,3,3);

    @mul    A[i,j] := cccc[x,y,j,z] * cccc2[z,i,y,x]
    @reduce B[i,j] := sum(x,y,z) cccc[x,y,j,z] * cccc2[z,i,y,x]
    @einsum C[i,j] := cccc[x,y,j,z] * cccc2[z,i,y,x]
    @test A ≈ B ≈ C

    D = similar(A);
    @mul    D[i,j] = cccc[x,y,j,z] * cccc2[z,i,y,x]  # in-place
    @test A ≈ D


    @mul    A[i,j,k,l] := cccc[i,x,y,k] * cccc2[l,j,y,x]
    @reduce B[i,j,k,l] := sum(x,y) cccc[i,x,y,k] * cccc2[l,j,y,x]
    @einsum C[i,j,k,l] := cccc[i,x,y,k] * cccc2[l,j,y,x]
    @test A ≈ B ≈ C

    D = similar(A);
    @mul    D[i,j,k,l] = cccc[i,x,y,k] * cccc2[l,j,y,x]   # in-place
    @test A ≈ D


    @mul    A[i,j,k,l,m,n] := cccc[x,k,i,m] * cccc2[l,x,j,n]
    @reduce B[i,j,k,l,m,n] := sum(x) cccc[x,k,i,m] * cccc2[l,x,j,n]
    @einsum C[i,j,k,l,m,n] := cccc[x,k,i,m] * cccc2[l,x,j,n]
    @test A ≈ B  ≈ C

    D = similar(A);
    @mul    D[i,j,k,l,m,n] = cccc[x,k,i,m] * cccc2[l,x,j,n]  # in-place
    @test A ≈ D


    ## repeat with non-alphabeitcal LHS!
    @mul    A[j,i,l,k,n,m] := cccc[x,k,i,m] * cccc2[l,x,j,n]
    @reduce B[j,i,l,k,n,m] := sum(x) cccc[x,k,i,m] * cccc2[l,x,j,n]
    @einsum C[j,i,l,k,n,m] := cccc[x,k,i,m] * cccc2[l,x,j,n]
    @test A ≈ B  ≈ C

    D = similar(A);
    @mul    D[j,i,l,k,n,m] = cccc[x,k,i,m] * cccc2[l,x,j,n]  # in-place
    @test A ≈ D


    ## Mason Potter's example:
    # W = rand(2,2,2,2); M = rand(2,2,2);
    W = rand(6,7,4,5); M = rand(7,2,3);

    @reduce N[σ, b\a, b′\a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′]
    @mul N2[σ, b\a, b′\a′] := W[σ,σ′,b,b′] * M[σ′,a,a′]
    @test N ≈ N2

    @reduce R[σ, b,a, b′\a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′]  lazy
    @mul R2[σ, b,a, b′\a′] := W[σ,σ′,b,b′] * M[σ′,a,a′]
    @test R ≈ R2
    # invperm((4,2,1,3)) == (3, 2, 4, 1)

    @reduce S[σ, b,a, b′,a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′]
    @mul S2[σ, b,a, b′,a′] := W[σ,σ′,b,b′] * M[σ′,a,a′]
    @test S ≈ S2
    # invperm((4,2,1,3,5)) == (3, 2, 4, 1, 5)

    ## in-place version
    N3 = similar(N); N4 = similar(N);
    @reduce N3[σ, b\a, b′\a′] = sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′]  lazy
    @mul N4[σ, b\a, b′\a′] = W[σ,σ′,b,b′] * M[σ′,a,a′]
    @test N ≈ N3 ≈ N4


    ## these were OK:
    M = rand(3,3,3);

    @reduce Q[σ, b\a, a′] := sum(σ′) M[σ,σ′,b] * M[σ′,a,a′];
    @mul Q2[σ, b\a, a′] := M[σ,σ′,b] * M[σ′,a,a′];
    @test Q ≈ Q2

    @reduce Q3[σ, b,a, a′] := sum(σ′) M[σ,σ′,b] * M[σ′,a,a′];
    @mul    Q4[σ, b,a, a′] := sum(σ′) M[σ,σ′,b] * M[σ′,a,a′];
    @test Q3 ≈ Q4

end
@testset "in & out shapes" begin

    bc = rand(2,3)
    cd = rand(3,4)

    @mul A[b,_,d] := bc[b,c] * cd[c,d]
    @mul A1[b,1,d] := bc[b,c] * cd[c,d]
    @test size(A) == size(A1) == (2,1,4)

    B = similar(A)
    @mul B[b,_,d] = bc[b,c] * cd[c,d] # in-place
    @test B == A

    @mul C[b] := bc[b,c] * cd[c,3]
    @test C == bc * cd[:,3]

    D = similar(C)
    @mul D[b] = bc[b,c] * cd[c,3] # in-place
    @test D == C

end
@testset "batch" begin

    bbb = rand(2,2,2)
    bbb2 = randn(2,2,2)

    @mul A[i,k,n] := bbb[i,j,n] * bbb2[j,k,n]
    @reduce B[i,k,n] := sum(j) bbb[i,j,n] * bbb2[j,k,n]
    @einsum C[i,k,n] := bbb[i,j,n] * bbb2[j,k,n]
    @test A == B == C

    D = similar(A);
    E = similar(B);
    F = similar(C);
    @mul D[i,k,n] = bbb[i,j,n] * bbb2[j,k,n] # in-place
    @reduce E[i,k,n] = sum(j) bbb[i,j,n] * bbb2[j,k,n]
    @einsum F[i,k,n] = bbb[i,j,n] * bbb2[j,k,n]
    @test A == D == E == F


    ## permute
    @mul A[n,k,i] := bbb[i,j,n] * bbb2[n,k,j]
    @reduce B[n,k,i] := sum(j) bbb[i,j,n] * bbb2[n,k,j]
    @einsum C[n,k,i] := bbb[i,j,n] * bbb2[n,k,j]
    @test A == B == C

    D = similar(A);
    E = similar(B);
    F = similar(C);
    @mul D[n,k,i] := bbb[i,j,n] * bbb2[n,k,j] # in-place
    @reduce E[n,k,i] := sum(j) bbb[i,j,n] * bbb2[n,k,j]
    @einsum F[n,k,i] := bbb[i,j,n] * bbb2[n,k,j]
    @test A == D == E == F

    ## readme & help
    X = rand(2,3,4)
    Y = rand(3,2,4)
    @mul W[β][i,j] := X[i,k,β] * Y[k,j,β]
    @test W[2] ≈ X[:,:,2] * Y[:,:,2]

    B = rand(8)
    C = rand(3,2,4)
    @mul A[_,i][j] := B[i\k] * C[j,2,k]
    rB = reshape(B,2,4)
    @einsum D[i,j] := rB[i,k] * C[j,2,k]
    @test A[1,2] ≈ D[2,:]

end
@testset "recursion" begin

    A = rand(2,3); B = rand(3,4); C = rand(4,5);

    @einsum V[i,l] := A[i,j] * B[j,k] * C[k,l]
    @reduce V2[i,l] := sum(j,k) A[i,j] * B[j,k] * C[k,l]

    @reduce W[i,l] := sum(j) A[i,j] * @mul [j,l] := B[j,k] * C[k,l]

    # @mul W2[i,l] := sum(j) A[i,j] * @mul [j,l] := B[j,k] * C[k,l]
    # maybe I decided not to allow this for now.

    @test V ≈ V2 ≈ W #≈ W2

end
