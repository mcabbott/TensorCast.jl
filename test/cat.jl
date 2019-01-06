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