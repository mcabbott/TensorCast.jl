
@testset "new macro notation" begin
    A = [i^2 for i in 1:12]

    # index ranges
    @test reshape(A,3,4) == @cast C[i,j] := A[(i,j)]  i in 1:3
    @test_logs min_level=Logging.Warn @reduce D[i] := sum(j:4) A[i⊗j]  # no warning for this

    # underscores 
    @test A' == @cast _[_,i] := A[i]
    @test (@cast _[_,i] := A[i]) isa Matrix  # transmute is reshape here

    # ... use in recursion, reads from the _ just written to:
    A = rand(4);
    B = randn(4,4);
    R1 = @reduce sum(i) A[i] * log( @reduce _[i] := sum(j) A[j] * exp(B[i,j]) ) # new
    R2 = @reduce sum(i) A[i] * log( @reduce [i] := sum(j) A[j] * exp(B[i,j]) ) # old, with warning
    @reduce inner[i] := sum(j) A[j] * exp(B[i,j])
    S = @reduce sum(i) A[i] * log(inner[i])
    @test S == R1 == R2

    # lazier
    Cs = [fill(i^2,7) for i in 1:12]
    using LazyStack
    @test (@cast _[i,o] := Cs[o][i]) isa LazyStack.Stacked
    @test (@cast _[i,o] |= Cs[o][i]) isa Matrix  # collects afterwards
    @test (@cast _[i,o] := Cs[o][i] lazy=false) isa Matrix  # uses stack_iter

    using LinearAlgebra, TransmuteDims
    @test (@cast _[o,i] := Cs[o][i]) isa Transpose{Int, <:LazyStack.Stacked}
    @test (@cast _[o,_,i] := Cs[o][i]) isa TransmutedDimsArray

    # broadcasting
    using Strided, LazyArrays
    @test A .+ 1 == @cast @strided B[i] := A[i] + 1
    # @test (@cast @lazy B[i] := A[i] + 1) isa BroadcastVector  # fails on 1.3?
end

@testset "ternary operator" begin
    CNT = Ref(0)
    add(x) = begin CNT[] += 1; x+1 end
    xs = collect(1:12)

    @cast ys[i] := iseven(xs[i]) ? add(xs[i]) : 0
    @test CNT[] == 6
    @test ys[1:6] == [0, 3, 0, 5, 0, 7]

    @cast zs[i,j] := iseven(xs[i⊗j]+1) ? add(ys[j⊗i] / 2) : ((1:3)[i] + 100)
    @test CNT[] == 12
    @test size(zs) == (3, 4)
end
