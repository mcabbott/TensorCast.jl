
using TensorSlice
using Test

using StaticArrays # you can't actually test without this, as macros using it get run anyway

using JuliennedArrays

# using Strided # TODO using this creates errors...

include("list.jl")


@testset "functions" begin
    @testset "colonise!" begin

        using TensorSlice: colonise!

        osz = Any[:(sz[1]), :((sz[2] * sz[3]) * sz[4])]

        colonise!(osz, [4,:,4,4], []) # the : in slist[2] means osz[2] should all be unkonwn
        @test osz[1] == :(sz[1])
        @test osz[2] == (:)

        osz = Any[ :((sz[2] * sz[3]) * sz[1])]

        colonise!(osz, [4,4,4], []) # since osz[1] contains all 3 sizes, it must be length(A)
        @test osz[1] == (:)

    end
    @testset "slice" begin

        using TensorSlice: sliceview, static_slice

        M = rand(2,3);

        S1 = sliceview(M, (:,*))
        S2 = static_slice(M, Size(2))

        @test all( first(S1) .== first(S2) )

        A = rand(1,2,3,4);

        S3 = sliceview(A, (:,:,*,*))
        S4 = static_slice(A, Size(1,2))

        @test all( first(S3) .== first(S4) )

    end
    @testset "glue" begin

        import TensorSlice: glue, glue!, static_glue, cat_glue

        B = [ SVector{2}(i .+ rand(2)) for i=1:3 ];

        G0 = cat_glue(B, (:,*))
        G1 = glue(B, (:,*))  # this calls JuliennedArrays
        G2 = static_glue(B)
        G3 = glue!(similar(G1), B, (:,*))
        @test all(G0 .== G1 .== G2 .== G3)

        G0T = cat_glue(B, (*,:))
        G1T = glue(B, (*,:))  # this calls JuliennedArrays
        G3T = glue!(similar(G1T), B, (*,:))
        @test all(G0T .== G1T)

        C = [ SMatrix{2,3}(rand(2,3)) for i=1:4, j=1:5 ];

        H1 = glue(C, (:,:,*,*)) # needs JuliennedArrays
        H2 = static_glue(C)
        H3 = glue!(similar(H1), C, (:,:,*,*))
        @test all(H1 .== H2 .== H3)

        D = [ SMatrix{2,3}(k .+ rand(2,3)) for k=1:4 ];

        G5 = glue(D, (:,:,*))
        G6 = cat_glue(D, (:,:,*))
        @test all(G5 .== G6)

        G7 = glue(D, (:,*,:))
        # G8 = cat_glue(D, (:,*,:)) # was completely wrong, now an error
        G9 = glue!(similar(G7), D, (:,*,:))
        @test all(G7 .== G9)

    end
end



@testset "readme" begin

    B = rand(3,4,5);
    @shape A[(i,j),k] := B[i,j,k]  # new matrix from tensor B

    B = rand(3*5,4);
    A = zeros(3,4,5);
    @shape A[i,j,k] = B[(i,k),j]   # write into an existing tensor A

    B = rand(3,4,5);
    @shape A[(i,j,k)] == B[i,j,k]  # reshaped view A = vec(B)



    B = [rand(3) for i=1:4];
    @shape A[i,j] := B[i][j]       # hcat a vector of vectors

    B = [rand(7) for i=1:3, k=1:4];
    A = zeros(3,7,4);
    @shape A[i,j,k] = B[i,k][j]    # write into A

    B = rand(2,3);
    @shape A[i][j] == B[j,i]       # create views A = collect(eachcol(B))



    B = [rand(3) for i=1:4];
    A = @shape [(i,j)] := B[j][i]  # vcat a vector of vectors

    B = [rand(3,4,5,6) for i=1:7]
    A = @shape [(i,j),l][k,m] := B[i][j,k,l,m] # glue then slice then reshape



    B = rand(2*5, 3);
    @shape A[i,j,k] := B[(i,k),j]  i:2  # could give (i:2, j:3, k:5)
    @test size(A) == (2,3,5)

    @shape A[i,j,k] := B[(i,k),j]  (i:2, j:3, k:5)
    @test size(A) == (2,3,5)



    @pretty @shape A[(i,j)] = B[i,j]
    # copyto!(A, B)

    @pretty @shape A[k][i,j] == B[i,(j,k)]  k:length(C)
    # begin
    #     local caterpillar = (size(B, 1), :, length(C))  # your animal may vary
    #     A = sliceview(reshape(B, (caterpillar...,)), (:, :, *))
    # end



    # using TestImages, ImageView, FileIO
    # V = testimage.(["mandril_gray", "cameraman", "lena_gray_512"])
    #
    # @shape M[i,(j,J)] := V[J][i,j]
    #
    # imshow(M)



    # using Flux, ImageView, FileIO, JuliennedArrays
    # imgs = Flux.Data.MNIST.images()[1:32] # vector of matrices
    #
    # @shape A[(i,I),(j,J)] := imgs[(I,J)][i,j] J:8 # eight columns
    #
    # imshow(A)



    using JuliennedArrays

    M = rand(3,4)
    @shape S[i][j] == M[i,j]       # S = julienne(M, (*,:)) creates views, S[i] == M[i,:]
    @shape Z[i,j] := S[i][j]       # Z = align(S, (*,:)) makes a copy

    B = [rand(2,3) for k=1:4, l=1:5];
    @shape A[i,j,k,l] := B[k,l][i,j]  # error without JuliennedArrays



    using StaticArrays
    M = rand(Int, 2,3)

    @shape S[k]{i} == M[i,k]  i:2  # S = reinterpret(SVector{2,Int}, vec(M)) needs the 2
    @shape N[k,i] == S[k]{i}       # such slices can be reinterpreted back again

    M[1,2]=42; N[2,1]==42          # all views of the original matrix
    @test N[2,1]==42


    # using Strided
    # A = rand(50,50,50,50);
    # B = permutedims(A, (4,3,2,1)); @strided permutedims(A, (4,3,2,1)); @strided permutedims!(B, A, (4,3,2,1)); # compile
    #
    # @time C = permutedims(A, (4,3,2,1));       # 130 ms,  47 MB
    # @time @strided permutedims(A, (4,3,2,1));  # 0.02 ms, 400 bytes, lazy
    #
    # @time @shape D[i,j,k,l] := A[l,k,j,i];     # 140 ms,  47 MB,     copy
    # @time @shape E[i,j,k,l] == A[l,k,j,i];     # 0.02 ms, 256 bytes, view
    # @time @shape C[i,j,k,l] = A[l,k,j,i];      # 15 ms,   4 KB,  in-place


end
