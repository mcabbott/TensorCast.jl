
@testset "simple" begin

    bc = rand(2,3)
    cd = rand(3,4)

    @matmul A[b,d] := sum(c) bc[b,c] * cd[c,d]
    @test A ≈ bc * cd

    B = similar(A)
    @matmul B[b,d] = sum(c) bc[b,c] * cd[c,d] # in-place
    @test B ≈ A

    ## matrix-vector
    c = randn(3)
    @matmul b[b] := sum(c) bc[b,c] * c[c]
    @test b ≈ bc * c

    @matmul c[c] := sum(b) b[b] * bc[b,c]
    @test c ≈ vec(b' * bc)

    b′ = similar(b)
    @matmul b′[b] = sum(c) bc[b,c] * c[c] # in-place
    @test b′ ≈ bc * c

    c′ = similar(c)
    @matmul c′[c] = sum(b) b[b] * bc[b,c]
    @test c′ ≈ c

    ## vector-vector
    @matmul z[] := sum(b) b[b] * b[b]
    @test z[] ≈ dot(b,b)
    @test z isa Array{Float64,0}

    z′ = similar(z)
    @matmul z′[] = sum(b) b[b] * b[b]
    @test z′ ≈ z

end
@testset "permutedims" begin

    bbb = rand(2,2,2);
    bbb2 = randn(2,2,2);

    @matmul A[i,j] := sum(k,k2) bbb[i,k,k2] * bbb2[j,k,k2]
    @reduce B[i,j] := sum(k,k2) bbb[i,k,k2] * bbb2[j,k,k2]
    @einsum C[i,j] := bbb[i,k,k2] * bbb2[j,k,k2]
    @test A ≈ B ≈ C

    D = similar(A);
    @matmul D[i,j] = sum(k,k2) bbb[i,k,k2] * bbb2[j,k,k2] # in-place
    @test A ≈ D

    @matmul A[i,j] := sum(k,k2) bbb[k2,i,k] * bbb2[j,k,k2]
    @reduce B[i,j] := sum(k,k2) bbb[k2,i,k] * bbb2[j,k,k2]
    @einsum C[i,j] := bbb[k2,i,k] * bbb2[j,k,k2]
    @test A ≈ B ≈ C

    D = similar(A);
    @matmul D[i,j] = sum(k,k2) bbb[k2,i,k] * bbb2[j,k,k2] # in-place
    @test A ≈ D


    cccc = rand(3,3,3,3);
    cccc2 = randn(3,3,3,3);

    @matmul A[i,j] := sum(x,y,z) cccc[x,y,j,z] * cccc2[z,i,y,x]
    @reduce B[i,j] := sum(x,y,z) cccc[x,y,j,z] * cccc2[z,i,y,x]
    @einsum C[i,j] := cccc[x,y,j,z] * cccc2[z,i,y,x]
    @test A ≈ B ≈ C

    D = similar(A);
    @matmul D[i,j] = sum(x,y,z) cccc[x,y,j,z] * cccc2[z,i,y,x]  # in-place
    @test A ≈ D


    @matmul A[i,j,k,l] := sum(x,y) cccc[i,x,y,k] * cccc2[l,j,y,x]
    @reduce B[i,j,k,l] := sum(x,y) cccc[i,x,y,k] * cccc2[l,j,y,x]
    @einsum C[i,j,k,l] := cccc[i,x,y,k] * cccc2[l,j,y,x]
    @test A ≈ B ≈ C

    D = similar(A);
    @matmul D[i,j,k,l] = sum(x,y) cccc[i,x,y,k] * cccc2[l,j,y,x]   # in-place
    @test A ≈ D


    @matmul A[i,j,k,l,m,n] := sum(x) cccc[x,k,i,m] * cccc2[l,x,j,n]
    @reduce B[i,j,k,l,m,n] := sum(x) cccc[x,k,i,m] * cccc2[l,x,j,n]
    @einsum C[i,j,k,l,m,n] := cccc[x,k,i,m] * cccc2[l,x,j,n]
    @test A ≈ B  ≈ C

    D = similar(A);
    @matmul D[i,j,k,l,m,n] = sum(x) cccc[x,k,i,m] * cccc2[l,x,j,n]  # in-place
    @test A ≈ D


    ## repeat with non-alphabeitcal LHS!
    @matmul A[j,i,l,k,n,m] := sum(x) cccc[x,k,i,m] * cccc2[l,x,j,n]
    @reduce B[j,i,l,k,n,m] := sum(x) cccc[x,k,i,m] * cccc2[l,x,j,n]
    @einsum C[j,i,l,k,n,m] := cccc[x,k,i,m] * cccc2[l,x,j,n]
    @test A ≈ B  ≈ C

    D = similar(A);
    @matmul D[j,i,l,k,n,m] = sum(x) cccc[x,k,i,m] * cccc2[l,x,j,n]  # in-place
    @test A ≈ D


    ## Mason Potter's example:
    # W = rand(2,2,2,2); M = rand(2,2,2);
    W = rand(6,7,4,5); M = rand(7,2,3);

    @reduce N[σ, b\a, b′\a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′];
    @matmul N2[σ, b\a, b′\a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′];
    @test N ≈ N2

    @reduce R[σ, b,a, b′\a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′]  # lazy;
    @matmul R2[σ, b,a, b′\a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′];
    @test R ≈ R2
    # invperm((4,2,1,3)) == (3, 2, 4, 1)

    @reduce S[σ, b,a, b′,a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′];
    @matmul S2[σ, b,a, b′,a′] := sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′];
    @test S ≈ S2
    # invperm((4,2,1,3,5)) == (3, 2, 4, 1, 5)

    # ## in-place version
    N3 = similar(N); N4 = similar(N);
    @reduce N3[σ, b\a, b′\a′] = sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′]  # lazy;
    @matmul N4[σ, b\a, b′\a′] = sum(σ′) W[σ,σ′,b,b′] * M[σ′,a,a′];
    @test N ≈ N3 ≈ N4


    ## these were OK:
    M = rand(3,3,3);

    @reduce Q[σ, b\a, a′] := sum(σ′) M[σ,σ′,b] * M[σ′,a,a′];
    @matmul Q2[σ, b\a, a′] := sum(σ′) M[σ,σ′,b] * M[σ′,a,a′];
    @test Q ≈ Q2

    @reduce Q3[σ, b,a, a′] := sum(σ′) M[σ,σ′,b] * M[σ′,a,a′];
    @matmul    Q4[σ, b,a, a′] := sum(σ′) M[σ,σ′,b] * M[σ′,a,a′];
    @test Q3 ≈ Q4

end
@testset "in & out shapes" begin

    bc = rand(2,3)
    cd = rand(3,4)

    @matmul A[b,_,d] := sum(c) bc[b,c] * cd[c,d]
    @matmul A1[b,1,d] := sum(c) bc[b,c] * cd[c,d]
    @test size(A) == size(A1) == (2,1,4)

    B = similar(A)
    @matmul B[b,_,d] = sum(c) bc[b,c] * cd[c,d] # in-place
    @test B == A

    @matmul C[b] := sum(c) bc[b,c] * cd[c,3]
    @test C == bc * cd[:,3]

    D = similar(C)
    @matmul D[b] = sum(c) bc[b,c] * cd[c,3] # in-place
    @test D == C

end
@testset "recursion" begin

    A = rand(2,3); B = rand(3,4); C = rand(4,5);

    @einsum V[i,l] := A[i,j] * B[j,k] * C[k,l]
    @reduce V2[i,l] := sum(j,k) A[i,j] * B[j,k] * C[k,l]

    @reduce W[i,l] := sum(j) A[i,j] * @matmul [j,l] := sum(k) B[j,k] * C[k,l]

    # @matmul W2[i,l] := sum(j) A[i,j] * @matmul [j,l] := B[j,k] * C[k,l]
    # maybe I decided not to allow this for now.

    @test V ≈ V2 ≈ W #≈ W2

end
