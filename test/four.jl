
@testset "new macro notation" begin
    A = [i^2 for i in 1:12]
    @test A .+ 1 == @cast @strided B[i] := A[i] + 1

    @test reshape(A,3,4) == @cast C[i,j] := A[(i,j)]  i in 1:3

    @test_logs min_level=Logging.Warn @reduce D[i] := sum(j:4) A[i⊗j]
end