using Einsum

@testset "matmul" begin 

    ## the very basics
    bc = rand(2,3)
    cd = rand(3,4)

    @mul A[b,d] := bc[b,c] * cd[c,d]
    @test A == bc * cd

    B = similar(A)
    @mul B[b,d] = bc[b,c] * cd[c,d] # in-place
    @test B == A


    ## check permutedims
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