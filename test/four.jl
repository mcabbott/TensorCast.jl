
@testset "new macro notation" begin
    A = [i^2 for i in 1:12]

    # @strided broadcasting
    @test A .+ 1 == @cast @strided B[i] := A[i] + 1

    # index ranges
    @test reshape(A,3,4) == @cast C[i,j] := A[(i,j)]  i in 1:3
    @test_logs min_level=Logging.Warn @reduce D[i] := sum(j:4) A[i⊗j]

    # underscores 
    @test A' == @cast _[_,i] := A[i]

    # ... use in recursion, reads from the _ just written to:
    A = rand(4);
    B = randn(4,4);
    R1 = @reduce sum(i) A[i] * log( @reduce _[i] := sum(j) A[j] * exp(B[i,j]) ) # new
    R2 = @reduce sum(i) A[i] * log( @reduce [i] := sum(j) A[j] * exp(B[i,j]) ) # old, with warning
    @reduce inner[i] := sum(j) A[j] * exp(B[i,j])
    S = @reduce sum(i) A[i] * log(inner[i])
    @test S == R1 == R2
end