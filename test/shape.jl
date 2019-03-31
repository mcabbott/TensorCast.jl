usingstatic =   isdefined(TensorCast, :StaticArray)

@testset "scope" begin

    bc = rand(2,3)

    β = 99 # should be ignored
    cb = nothing # should be overwritten
    @cast cb[κ,β] := bc[β,κ]
    @test !isdefined(@__MODULE__,:κ)
    @test β == 99
    @test all(cb .== transpose(bc))

    @cast cb[c,b] = bc[b,c] c:3 # in-place

    β = 2
    @cast f[(c,b)] := bc[b,c] b:β, c:3 # @assert uses β
    @test size(f) == (6,)
    @cast cb2[c,b] == f[(c,b)] b:β # can't not use this
    @test size(cb2) == (3,2)
    @test all(cb2 .== transpose(bc))

    oo = [1,1]
    @cast cb3[c,b] == f[(c,b)] b:sum(oo) # with an expression
    @test size(cb3) == (3,2)

    β = 2
    @cast b2[b] := bc[b,$β] # β should be inserted
    @test b2 == bc[:,2]

end
@testset "permute" begin

    bcde = rand(2,3,4,5);
    bcdef = rand(2,3,4,5,6);

    @cast oo[d,e,b,c] |= bcde[b,c,d,e];
    @test size(oo) == (4,5,2,3)
    @test oo[1,3,1,3] == bcde[1,3,1,3]

    oo[1] = 99
    @test bcde[1] != 99 # copy not a view

    @cast oo[d,e,b,c] = bcde[b,c,d,e] ! ;
    oo[1] = 99
    @test bcde[1] != 99 # still not a view

    @cast oo[e,b,c,f,d] := bcdef[b,c,d,e,f];
    @test size(oo) == (5,2,3,6,4)
    @test oo[1,2,1,2,2] == bcdef[2,1,2,1,2]

    @cast oo[e,b,zc,f,d] := bcdef[b,zc,d,e,f]; # because I alphabetise symbols
    @test size(oo) == (5,2,3,6,4)

    @cast oo[e,zb,c,f,d] := bcdef[zb,c,d,e,f];
    @test size(oo) == (5,2,3,6,4)

end
@testset "slice" begin

    bcd = rand(2,3,4);
    bcde = rand(2,3,4,5);

    @cast Bcd[b][c,d] |= bcd[b,c,d] !
    @test size(Bcd) == (2,)
    @test size(first(Bcd)) == (3,4)
    @test Bcd[2][3,1] == bcd[2,3,1]

    Bcd[2][3,1] = 99
    @test bcd[2,3,1] != 99 # copy not a view

    @cast Cbd[c][b,d] := bcd[b,c,d] !
    @test size(Cbd) == (3,)
    @test size(first(Cbd)) == (2,4)
    @test Cbd[3][2,1] == bcd[2,3,1]

    @cast BCde[b,c][d,e] := bcde[b,c,d,e] !
    @test size(BCde) == (2,3)
    # @test size(first(BCde)) == (4,5)
    @test size(BCde[1,1]) == (4,5) # new julienne

    @cast DBec[d,b][e,c] := bcde[b,c,d,e] !
    @test size(DBec) == (4,2)
    # @test size(first(DBec)) == (5,3)
    @test size(DBec[1,1]) == (5,3) # new julienne

    @cast DEbc[d,e][b,c] == bcde[b,c,d,e] # this order can be a view
    @test size(DEbc) == (4,5)
    # @test size(first(DEbc)) == (2,3)
    @test size(DEbc[1,1]) == (2,3)

    DEbc[3,4][1,2] = 99
    @test bcde[1,2,3,4] == 99 # view not a copy

    # @test_throws LoadError @cast BCde[b,c][d,e] == bcde[b,c,d,e] # doesn't work

end
@testset "static-slice" begin

        M = rand(Int, 2,3)
        S1 = reinterpret(SVector{2,Int}, vec(M)) # as in readme
        @cast S2[k]{i} == M[i,k]  i:2
        M[end]=42
        @test S1[3][2]==42 # is a view
        @test S2[3][2]==42
        @test first(S1) isa StaticArray
        @test first(S2) isa StaticArray
        @test size(first(S1)) == (2,)
        @test size(first(S2)) == (2,)

        @cast S3[k]{i} := M[k,i]  i:3 # this has a permutedims
        @test first(S3) isa StaticArray
        @test size(first(S3)) == (3,)

        bcd = rand(2,3,4);
        bcde = rand(2,3,4,5);

        @cast DBec[d,b]{e,c} := bcde[b,c,d,e] e:5, c:3
        @test size(DBec) == (4,2)
        @test size(first(DBec)) == (5,3)
        @test first(DBec) isa StaticArray

        @cast DEbc[d,e]{b,c} == bcde[b,c,d,e] b:2, c:3
        @test size(DEbc) == (4,5)
        @test size(first(DEbc)) == (2,3)
        @test first(DEbc) isa StaticArray

        A = rand(3,3,4)
        @cast B[k]{i,j} == A[i,j,k]  i:3, j:3
        @cast C[(j,k),i] := B[k][i,j]
        @test C[1,2] == B[1][2,1] ==  A[2,1,1]

end
@testset "glue" begin

    bcd = rand(2,3,4)
    Bcd = [ rand(3,4) for b=1:2 ]

    @cast bcd[b,c,d] = Bcd[b][c,d] !
    @test size(bcd) == (2,3,4)
    @test Bcd[2][3,1] == bcd[2,3,1]

    Cbd = [ rand(2,4) for c=1:3 ]

    @cast oo[b,c,d] := Cbd[c][b,d]
    @test size(oo) == (2,3,4)
    @test Cbd[3][2,1] == oo[2,3,1]

    DEbc = [ rand(2,3) for d=1:4, e=1:5 ]

    @cast bcde[b,c,d,e] := DEbc[d,e][b,c];
    @test size(bcde) == (2,3,4,5)

    BCde = [ rand(4,5) for d=1:2, e=1:3 ]

    @cast bcde[b,c,d,e] = BCde[b,c][d,e] ! ;
    @test size(bcde) == (2,3,4,5)

end
@testset "static-glue" begin

        B = [ SVector{3}(rand(3)) for i=1:2]

        @cast A[j,i] == B[i]{j} i:2, j:3 # view is impossible without Static slices
        @test size(A) == (3,2)
        @test !isa(A, Array)

        A[1,2] = 33
        @test B[2][1] == 33 # mutating B illegally?

        @cast A[i,j] := B[i]{j} j:3
        @test size(A) == (2,3)
        @test !isa(A, StaticArray)

        @test A[2,1] == 33

        C = [ SMatrix{2,3}(rand(2,3)) for i=1:4, j=1:5 ];

        @cast A[l,k,j,i] := C[i,j]{k,l} k:2, l:3
        @test A[1,2,3,4] == C[4,3][2,1]
        @test size(A) == (3,2,5,4)

        @cast A[l,k,j,i] = C[i,j]{k,l} k:2, l:3 # now in-place
        @test A[1,2,3,4] == C[4,3][2,1]

        @cast A[l,k,j,i] = C[i,j]{k,l} # now without static glue
        @test A[1,2,3,4] == C[4,3][2,1]

end
@testset "reshape" begin

    bcd = rand(2,3,4)
    bcde = rand(2,3,4,5)
    bcde2 = similar(bcde)

    @cast v[(b,zc,d)] == bcd[b,zc,d]
    @cast w[d,(zc,b)] := v[(b,zc,d)] b:2, zc:3 # add z sometimes to mess with alphabetisation
    @cast dbc[d,b,c] := w[d,(c,b)] c:3

    @test size(w) == (4,6)
    @test size(dbc) == (4,2,3)
    @test dbc[1,2,3] == bcd[2,3,1]

    @cast g[(b,c),x,y,e] := bcde[b,c,(x,y),e] x:2;
    @test size(g) == (6,2,2,5)

    @cast g[(y,e),(b,c),x] := bcde[b,c,(x,y),e] x:1;
    @test size(g) == (20, 6, 1)
    @cast bcde2[b,c,(x,y),e] = g[(y,e),(b,c),x];
    @test all(bcde2 .== bcde) # fails when using Strided, why? WTF?

end
@testset "combined" begin

    Bc = [ rand(3) for b=1:2 ] # Bc = [(1:3) .+ 10i for i=1:2]
    @cast f[(b,c)] := Bc[b][c]
    @test size(f) == (6,)
    @cast bc[b,c] |= f[(b,c)] b:2 # for in-place to work below, this bc is NOT a view of f
    @test size(bc) == (2,3)
    @test bc[1,2] == Bc[1][2]

    @cast f[(c,b)] = Bc[b][c] ! # in-place
    @cast bc[b,c] = f[(c,b)] !  # in-place
    @test bc[1,2] == Bc[1][2]

    Cdaeb = [rand(4,1,5,2) for i=1:3]; # sizes match abcde now
    Z = @cast [(c,d),e][a,b] := Cdaeb[c][d,a,e,b] # alphabetical = canonical
    @test size(Z) == (3*4, 5)
    @test size(first(Z)) == (1,2)
    @test Z[2*1,4][1,2] == Cdaeb[2][1,1,4,2]

    @cast Z[e,(c,d)][b,a] := Cdaeb[c][d,a,e,b] # same but with permutedims
    @test size(Z) == (5, 3*4)
    @test size(first(Z)) == (2,1)
    @test Z[4,2*1][2,1] == Cdaeb[2][1,1,4,2]

end
@testset "reverse" begin

    bc = rand(2,3)

    @cast A[i,-j] := bc[i,j]
    @cast B[i,j] := bc[i,-j]
    @cast C[i,-j] := bc[i,-j]
    D = reverse(bc, dims=2)
    @test A == B == D
    @test C == bc

    ccbb = rand(0:99, 3,3,2,2)
    @cast A[-b,c,d,e] := ccbb[b,c,-d,e]
    B = similar(A);
    @cast B[b,c,d,e] = ccbb[-b,c,-d,e]
    C = reverse(reverse(ccbb, dims=1), dims=3)
    @test A == B == C

    ccbb = rand(0:99, 3,3,2,2)
    @cast A[c,e,b,d] := ccbb[-b,c,d,e] + 100
    @cast B[c,e,-b,d] := ccbb[b,c,d,e] + 100
    @test A == B
    C = similar(A); D = similar(A);
    @cast C[c,e,b,d] = ccbb[-b,c,d,e] + 100
    # @cast D[c,e,-b,d] = ccbb[b,c,d,e] + 100 # can't reverse axes of in-place output
    @test A == B == C # == D

    @cast A[c,e,b,_] := ccbb[-b,c,2,e] + 100
    @cast B[c,e,-b,_] := ccbb[b,c,2,e] + 100
    @test A == B

end
@testset "fixed indices" begin

    W = rand(2,3,5,7);

    @cast Z[_,i,_,k] := W[2,k,4,i]

    @test size(Z) == (1,7,1,3)

    @cast A[k\i] := W[2,k,4,i] / Z[1,i,_,k]
    @test all(A .== 1)

end
@testset "random" begin

    A = zeros(1,2,3);
    B = randn(1,3*2)

    @cast A[i,j,k] = B[i,(k,j)] # errored on push_checks for a while

    @cast A[i,j,k] = B[i,(j,k)] i:1

    @cast V[(i,j,k)] := B[i,(k,j)]  k:3, ! # was an error for recursive colon reasons


    A = @cast [jk,i] := B[i,jk] # without left-hand name
    @test size(A) == (6,1)

    A = @cast [k,j,i] := B[i,(j,k)] k:3 # without left-hand name
    @test size(A) == (3,2,1)

    C = @cast [k][i,j] == B[i,(j,k)]  k:3 # without left-hand name
    C = @cast [k][i,j] := B[i,(j,k)]  j:2, k:3

    @test size(C) == (3,)
    @test size(first(C)) == (1,2)


    @cast A[k][i,j] == B[i,(j,k)]  k:length(C) # with an expression

    @test size(A) == (3,)

    D = randn(1*2*3, 4)
    # these two both result in a product of sizes like ((sz[2] * :) * s[4]) with : on 2nd level.
    @cast C[i,(j,k,l)] := D[(i,j,k),l] i:1, j:2 # gets sz... from colonise! too
    @cast E[i,(k,j,l)] := D[(i,j,k),l] i:1, j:2 # does not

    @test size(C) == (1, 2*3*4)
    @test size(E) == (1, 2*3*4)

    B = [rand(2) for i=1:3 ]
    A2 = @cast A[(i,j)] := B[i][j]
    A3 = @cast A[(i,j)] = B[i][j] # for a while this in-place thing returned the wrong shape
    @test size(A) == size(A2) == size(A3) == (6,)

    # This was broken above for a while:
    Bc = [(1:3) .+ 10i for i=1:2]
    @cast f[(b,c)] := Bc[b][c]
    @cast bc[b,c] := f[(b,c)] b:2 # this bc is a view of f, NB
    @cast f[(c,b)] = Bc[b][c]
    @cast bc[b,c] = f[(c,b)]  # thus this copyto! is confusing
    @test bc[1,2] != Bc[1][2] # and in fact leads to duplicates here

end
@testset "errors" begin

    bc = rand(2,3)
    cb = rand(3,2)

    ## macro

    # @test_throws ArgumentError @cast a[i] := bc[i,j]
    # @test_throws ArgumentError @cast a[i,k] := bc[(i,j),k]
    # @test_throws ArgumentError @cast a[(i,j),k] := bc[(i,k)]

    ## runtime

    @test_throws DimensionMismatch @cast oo[i,j] := bc[i,j] i:3,!
    @test_throws DimensionMismatch @cast cb[i,j] = bc[i,j] !

end
