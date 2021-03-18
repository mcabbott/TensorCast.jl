
@testset "new macro notation" begin
    A = [i^2 for i in 1:10]
    @test A .+ 1 == @cast @strided B[i] := A[i] + 1
end