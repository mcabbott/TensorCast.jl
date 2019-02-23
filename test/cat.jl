@testset "slice" begin

    using StaticArrays
    using TensorCast: sliceview, slicecopy, static_slice

    M = rand(2,3);

    S1 = sliceview(M, (:,*))
    S2 = slicecopy(M, (:,*))
    S3 = static_slice(M, Size(2))

    @test all( first(S1) .== first(S2) .== first(S3) )

    A = rand(1,2,3,4);

    S4 = sliceview(A, (:,:,*,*))
    S5 = slicecopy(A, (:,:,*,*))
    S6 = static_slice(A, Size(1,2))

    @test all( first(S4) .== first(S5) .== first(S6) )

    S7 = sliceview(A, (*,:,*,:))
    S8 = slicecopy(A, (*,:,*,:))

    @test first(S7) == first(S8)

end
@testset "glue" begin

    import TensorCast: copy_glue, glue!, static_glue, cat_glue, red_glue, lazy_glue

    B = [ SVector{2}(i .+ rand(2)) for i=1:3 ];

    G0 = cat_glue(B, (:,*))
    G1 = copy_glue(B, (:,*)) 
    G2 = static_glue(B)
    G3 = glue!(similar(G1), B, (:,*))
    G4 = red_glue(B, (:,*))
    @test all(G0 .== G1 .== G2 .== G3 .== G4)

    G0T = cat_glue(B, (*,:))
    G1T = copy_glue(B, (*,:)) 
    G3T = glue!(similar(G1T), B, (*,:))
    G4T = red_glue(B, (*,:))
    @test all(G0T .== G1T .== G3T .== G4T)

    C = [ SMatrix{2,3}(rand(2,3)) for i=1:4, j=1:5 ];

    H1 = copy_glue(C, (:,:,*,*)) 
    H2 = static_glue(C)
    H3 = glue!(similar(H1), C, (:,:,*,*))
    H4 = red_glue(C, (:,:,*,*)) 
    H5 = cat_glue(C, (:,:,*,*)) 
    H6 = lazy_glue(C, (:,:,*,*)) 
    @test all(H1 .== H2 .== H3 .== H4 .== H5)

    D = [ SMatrix{2,3}(k .+ rand(2,3)) for k=1:4 ];

    G5 = copy_glue(D, (:,:,*))
    G6 = cat_glue(D, (:,:,*))
    G7 = red_glue(D, (:,:,*))
    G8 = lazy_glue(D, (:,:,*))
    @test all(G5 .== G6 .== G7 .== G8)

    G8 = copy_glue(D, (:,*,:))
    G9 = glue!(similar(G8), D, (:,*,:))
    @test all(G8 .== G9)

end