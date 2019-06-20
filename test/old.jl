
@testset "new readme" begin

    mat = (1:4)' .+ rand(2,4)
    @cast rows[r][c] := mat[r,c]
    @cast cols[♜][🚣] := rows[🚣][♜]  # unicode 👍
    @reduce sum_r[c] := sum(r) mat[r,c]
    @test sum_r == sum(rows) # true


    B = rand(2*5, 3)
    @cast A[i,j,k] := B[(i,k),j]  i:2
    @test size(A) == (2,3,5)
    @cast A[i,j,k] = B[(i,k),j];


    imgs = [ rand(8,8) for i=1:16 ];

    @cast G[(i,I), (j,J)] := imgs[(I,J)][i,j] J:4
    @cast G[ i\I,   j\J ] = imgs[ I\J ][i,j] # in-place


    G = rand(16,32);

    @reduce H[a, b] := maximum(α,β)  G[α\a, β\b]  α:2,β:2
    @test size(G) == 2 .* size(H)

    @reduce H4[a, b] := maximum(α:4,β:4)  G[α\a, β\b]
    @test size(G) == 4 .* size(H4)

    W = randn(2,3,5,7);
    @cast Z[_,i,_,k] := W[2,k,4,i]  # equivalent to Z[1,i,1,k] on left
    @test size(Z) == (1,7,1,3)

    B = [ i .* ones(2,2) for i=1:4 ]
    @reduce A[μ,ν,J] := prod(i:2) B[(i,J)][μ,ν]
    @test size(A) == (2, 2, 2)


end
@testset "old readme" begin

    B = rand(3,4,5);
    @cast A[(i,j),k] := B[i,j,k]  # new matrix from tensor B

    B = rand(3*5,4);
    A = zeros(3,4,5);
    @cast A[i,j,k] = B[(i,k),j]   # write into an existing tensor A

    B = rand(3,4,5);
    @cast A[(i,j,k)] := B[i,j,k]  # reshaped view A = vec(B)


    B = [rand(3) for i=1:4];
    @cast A[i,j] := B[i][j]       # hcat a vector of vectors
    @test size(A) == (4,3)

    B = [rand(7) for i=1:3, k=1:4];
    A = zeros(3,7,4);
    @cast A[i,j,k] = B[i,k][j]    # write into A

    B = rand(2,3);
    @cast A[i][j] := B[j,i]       # create views A = collect(eachcol(B))


    B = [rand(3) for i=1:4];
    A = @cast [(i,j)] := B[j][i]  # vcat a vector of vectors
    @test size(A) == (12,)

    B = [rand(3,4,5,6) for i=1:7];
    A = @cast [(i,j),l][k,m] := B[i][j,k,l,m]; # glue then slice then reshape
    @test size(A) == (21,5)


    B = rand(2*5, 3);
    @cast A[i,j,k] := B[(i,k),j]  i:2  # could give (i:2, j:3, k:5)
    @test size(A) == (2,3,5)

    @cast A[i,j,k] := B[(i,k),j]  (i:2, j:3, k:5)
    @test size(A) == (2,3,5)


    @pretty @cast A[(i,j)] = B[i,j]

    @pretty @cast A[k][i,j] := B[i,(j,k)]  k:length(C)


    M = rand(3,4)
    @cast S[i][j] := M[i,j]       # S = julienne(M, (*,:)) creates views, S[i] == M[i,:]
    @cast Z[i,j] := S[i][j]       # Z = align(S, (*,:)) makes a copy
    @test size(Z) == (3,4)

    B = [rand(2,3) for k=1:4, l=1:5];
    @cast A[i,j,k,l] := B[k,l][i,j]
    @test size(A) == (2,3,4,5)


    using StaticArrays
    M = rand(1:99, 2,3)

    @cast S[k]{i} == M[i,k]  i:2  # S = reinterpret(SVector{2,Int}, vec(M)) needs the 2
    @cast N[k,i] := S[k]{i}       # such slices can be reinterpreted back again

    M[1,2]=42; N[2,1]==42          # all views of the original matrix
    @test N[2,1]==42

end
@testset "other random" begin

    A = rand(1:99, 3,3);
    X = randn(4)
    @cast B[i,j,k] := A[i,j] * X[k]
    @test B == A .* reshape(X, 1,1,:)

    G = rand(4, 4);
    two = 2
    @cast G2[k, l] := G[k,2] * G[l,2]
    @cast G3[k, l] := G[k,$two] * G[l,$two]
    @test G2 == G3 == G[:,2] .* G[:,2]'

end
