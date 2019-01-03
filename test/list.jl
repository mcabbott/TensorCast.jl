# this whole list is intended to be run with and without various packages

usingstrided =  isdefined(TensorSlice, :Strided)
usingstatic =   isdefined(TensorSlice, :StaticArray)
usingjulienne = isdefined(TensorSlice, :JuliennedArrays)

vv = []
usingstrided  && push!(vv, "strided")
usingstatic   && push!(vv, "static")
usingjulienne && push!(vv, "julienne")
length(vv)>0 && println("using ", join(vv, " & "), " in tests")

@testset "scope" begin

    bc = rand(2,3)

    β = 99 # should be ignored
    cb = nothing # should be overwritten
    @shape cb[c,β] := bc[β,c]
    @test !isdefined(@__MODULE__,:c)
    @test β == 99
    @test all(cb .== transpose(bc))

    @shape cb[c,b] = bc[b,c] c:3 # in-place

    β = 2
    @shape f[(c,b)] := bc[b,c] b:β, c:3 # @assert uses β
    @test size(f) == (6,)
    @shape cb2[c,b] == f[(c,b)] b:β # can't not use this
    @test size(cb2) == (3,2)
    @test all(cb2 .== transpose(bc))

    oo = [1,1]
    @shape cb3[c,b] == f[(c,b)] b:sum(oo) # with an expression
    @test size(cb3) == (3,2)

end
@testset "permute" begin

    bcde = rand(2,3,4,5);
    bcdef = rand(2,3,4,5,6);

    @shape oo[d,e,b,c] := bcde[b,c,d,e];
    @test size(oo) == (4,5,2,3)
    @test oo[1,3,1,3] == bcde[1,3,1,3]

    oo[1] = 99
    @test bcde[1] != 99 # copy not a view

    @shape oo[d,e,b,c] = bcde[b,c,d,e] ! ;
    oo[1] = 99
    @test bcde[1] != 99 # still not a view

    @shape oo[e,b,c,f,d] := bcdef[b,c,d,e,f];
    @test size(oo) == (5,2,3,6,4)
    @test oo[1,2,1,2,2] == bcdef[2,1,2,1,2]

    @shape oo[e,b,zc,f,d] := bcdef[b,zc,d,e,f]; # because I alphabetise symbols
    @test size(oo) == (5,2,3,6,4)

    @shape oo[e,zb,c,f,d] := bcdef[zb,c,d,e,f];
    @test size(oo) == (5,2,3,6,4)

end
@testset "slice" begin

    bcd = rand(2,3,4);
    bcde = rand(2,3,4,5);

    @shape Bcd[b][c,d] := bcd[b,c,d] !
    @test size(Bcd) == (2,)
    @test size(first(Bcd)) == (3,4)
    @test Bcd[2][3,1] == bcd[2,3,1]

    Bcd[2][3,1] = 99
    @test bcd[2,3,1] != 99 # copy not a view

    @shape Cbd[c][b,d] := bcd[b,c,d] !
    @test size(Cbd) == (3,)
    @test size(first(Cbd)) == (2,4)
    @test Cbd[3][2,1] == bcd[2,3,1]

    @shape BCde[b,c][d,e] := bcde[b,c,d,e] !
    @test size(BCde) == (2,3)
    @test size(first(BCde)) == (4,5)

    @shape DBec[d,b][e,c] := bcde[b,c,d,e] !
    @test size(DBec) == (4,2)
    @test size(first(DBec)) == (5,3)

    @shape DEbc[d,e][b,c] == bcde[b,c,d,e] # this order can be a view
    @test size(DEbc) == (4,5)
    @test size(first(DEbc)) == (2,3)

    DEbc[3,4][1,2] = 99
    @test bcde[1,2,3,4] == 99 # view not a copy

    # @test_throws LoadError @shape BCde[b,c][d,e] == bcde[b,c,d,e] # doesn't work

    if usingstatic
        println("running static slice tests") # this doesn't work, as macros get run anyway?

        M = rand(Int, 2,3)
        S1 = reinterpret(SVector{2,Int}, vec(M)) # as in readme
        @shape S2[k]{i} == M[i,k]  i:2
        M[end]=42
        @test S1[3][2]==42 # is a view
        @test S2[3][2]==42
        @test first(S1) isa StaticArray
        @test first(S2) isa StaticArray
        @test size(first(S1)) == (2,)
        @test size(first(S2)) == (2,)

        @shape S3[k]{i} := M[k,i]  i:3 # this has a permutedims
        @test first(S3) isa StaticArray
        @test size(first(S3)) == (3,)

        bcd = rand(2,3,4);
        bcde = rand(2,3,4,5);

        @shape DBec[d,b]{e,c} := bcde[b,c,d,e] e:5, c:3 # this order cannot be a view
        @test size(DBec) == (4,2)
        @test size(first(DBec)) == (5,3)
        @test first(DBec) isa StaticArray

        @shape DEbc[d,e]{b,c} == bcde[b,c,d,e] b:2, c:3 # this order can be a view
        @test size(DEbc) == (4,5)
        @test size(first(DEbc)) == (2,3)
        @test first(DEbc) isa StaticArray

        A = rand(3,3,4)
        @shape B[k]{i,j} == A[i,j,k]  i:3, j:3
        @shape C[(j,k),i] := B[k][i,j]
        @test C[1,2] == B[1][2,1] ==  A[2,1,1]

    end

end
@testset "glue" begin

    bcd = rand(2,3,4)
    Bcd = [ rand(3,4) for b=1:2 ]

    @shape bcd[b,c,d] = Bcd[b][c,d] !
    @test size(bcd) == (2,3,4)
    @test Bcd[2][3,1] == bcd[2,3,1]

    Cbd = [ rand(2,4) for c=1:3 ]

    @shape oo[b,c,d] := Cbd[c][b,d]
    @test size(oo) == (2,3,4)
    @test Cbd[3][2,1] == oo[2,3,1]

    if usingjulienne
        println("running julienne glue tests")

        DEbc = [ rand(2,3) for d=1:4, e=1:5 ]

        @shape bcde[b,c,d,e] := DEbc[d,e][b,c]; # don't know how to cat this
        @test size(bcde) == (2,3,4,5)

        BCde = [ rand(4,5) for d=1:2, e=1:3 ]

        @shape bcde[b,c,d,e] = BCde[b,c][d,e] ! ; # don't know how to cat this
        @test size(bcde) == (2,3,4,5)

    end

    if usingstatic
        println("running static glue tests")

        B = [ SVector{3}(rand(3)) for i=1:2]

        @shape A[j,i] == B[i]{j} i:2, j:3 # view is impossible without Static slices
        @test size(A) == (3,2)
        # @test !isa(A, Array)

        A[1,2] = 33
        @test B[2][1] == 33 # mutating B illegally?

        @shape A[i,j] := B[i]{j} j:3
        @test size(A) == (2,3)
        # @test !isa(A, StaticArray)

        @test A[2,1] == 33

        C = [ SMatrix{2,3}(rand(2,3)) for i=1:4, j=1:5 ];

        @shape A[l,k,j,i] := C[i,j]{k,l} k:2, l:3
        @test A[1,2,3,4] == C[4,3][2,1]
        @test size(A) == (3,2,5,4)

        @shape A[l,k,j,i] = C[i,j]{k,l} k:2, l:3 # now in-place
        @test A[1,2,3,4] == C[4,3][2,1]

        @shape A[l,k,j,i] = C[i,j]{k,l} # now without static glue
        @test A[1,2,3,4] == C[4,3][2,1]

    end
end
@testset "reshape" begin

    bcd = rand(2,3,4)
    bcde = rand(2,3,4,5)
    bcde2 = similar(bcde)

    @shape v[(b,zc,d)] == bcd[b,zc,d]
    @shape w[d,(zc,b)] := v[(b,zc,d)] b:2, zc:3 # add z sometimes to mess with alphabetisation
    @shape dbc[d,b,c] := w[d,(c,b)] c:3

    @test size(w) == (4,6)
    @test size(dbc) == (4,2,3)
    @test dbc[1,2,3] == bcd[2,3,1]

    @shape g[(b,c),x,y,e] := bcde[b,c,(x,y),e] x:2;
    @test size(g) == (6,2,2,5)

    @shape g[(y,e),(b,c),x] := bcde[b,c,(x,y),e] x:1;
    @test size(g) == (20, 6, 1)
    @shape bcde2[b,c,(x,y),e] = g[(y,e),(b,c),x];
    @test all(bcde2 .== bcde) # fails when using Strided, why? WTF?

end
@testset "combined" begin

    Bc = [ rand(3) for b=1:2 ]
    @shape f[(b,c)] := Bc[b][c]
    @test size(f) == (6,)
    @shape bc[b,c] := f[(b,c)] b:2
    @test size(bc) == (2,3)
    @test bc[1,2] == Bc[1][2]

    @shape f[(c,b)] = Bc[b][c] ! # in-place
    @shape bc[b,c] = f[(c,b)] !  # in-place
    @test bc[1,2] == Bc[1][2]

    Cdaeb = [rand(4,1,5,2) for i=1:3]; # sizes match abcde now
    Z = @shape [(c,d),e][a,b] := Cdaeb[c][d,a,e,b] # alphabetical = canonical
    @test size(Z) == (3*4, 5)
    @test size(first(Z)) == (1,2)
    @test Z[2*1,4][1,2] == Cdaeb[2][1,1,4,2]

    @shape Z[e,(c,d)][b,a] := Cdaeb[c][d,a,e,b] # same but with permutedims
    @test size(Z) == (5, 3*4)
    @test size(first(Z)) == (2,1)
    @test Z[4,2*1][2,1] == Cdaeb[2][1,1,4,2]

end
@testset "random" begin

    A = zeros(1,2,3);
    B = randn(1,3*2)

    @shape A[i,j,k] = B[i,(k,j)] # errored on push_checks for a while

    @shape A[i,j,k] = B[i,(j,k)] i:1 # @pretty looks a bit weird
    # @pretty @shape A[l,k,j,i] = C[i,j][k,l] k:2, j:3 # this one looks weird too TODO figure out why

    @shape V[(i,j,k)] := B[i,(k,j)]  k:3, ! # was an error for recursive colon reasons


    A = @shape [jk,i] := B[i,jk] # without left-hand name
    @test size(A) == (6,1)

    A = @shape [k,j,i] := B[i,(j,k)] k:3 # without left-hand name
    @test size(A) == (3,2,1)

    C = @shape [k][i,j] == B[i,(j,k)]  k:3 # without left-hand name
    C = @shape [k][i,j] := B[i,(j,k)]  j:2, k:3

    @test size(C) == (3,)
    @test size(first(C)) == (1,2)


    @shape A[k][i,j] == B[i,(j,k)]  k:length(C) # with an expression

    @test size(A) == (3,)

    D = randn(1*2*3, 4)
    # these two both result in a product of sizes like ((sz[2] * :) * s[4]) with : on 2nd level.
    @shape C[i,(j,k,l)] := D[(i,j,k),l] i:1, j:2 # gets sz... from colonise! too
    @shape E[i,(k,j,l)] := D[(i,j,k),l] i:1, j:2 # does not

    @test size(C) == (1, 2*3*4)
    @test size(E) == (1, 2*3*4)

    # @shape F[(i,j),(k,l)] := D[(i,j,k),l] j≦2, k≦3 # can't make this work

    B = [rand(2) for i=1:3 ]
    A2 = @shape A[(i,j)] := B[i][j]
    A3 = @shape A[(i,j)] = B[i][j] # for a while this in-place thing returned the wrong shape
    @test size(A) == size(A2) == size(A3) == (6,)

end
@testset "errors" begin

    bc = rand(2,3)
    cb = rand(3,2)

    ## macro

    # @test_throws ArgumentError @shape a[i] := bc[i,j]
    # @test_throws ArgumentError @shape a[i,k] := bc[(i,j),k]
    # @test_throws ArgumentError @shape a[(i,j),k] := bc[(i,k)]

    ## runtime

    @test_throws DimensionMismatch @shape oo[i,j] := bc[i,j] i:3
    @test_throws DimensionMismatch @shape cb[i,j] = bc[i,j]  # this should check without !

    if !usingjulienne

        DEbc = [ rand(2,3) for d=1:4, e=1:5 ]

        @test_throws ArgumentError @shape bcde[b,c,d,e] := DEbc[d,e][b,c]; # also tested above

    end

end
