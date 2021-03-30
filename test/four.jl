
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

    rs = randn(10,10)
    @reduce ps[i,_] := sum(j) if rs[i,j] > 0
        sqrt(rs[i,j])
    else
        0 * cbrt(rs[i,j])
    end
    @test ps ≈ sum(x -> x>0 ? sqrt(x) : 0.0, rs, dims=2)
end

@testset "naked indices" begin
    @cast A[i,j] := i+10j  (i in 1:3, j in 1:4)  # explicit size
    @test A == (1:3) .+ (10:10:40)'

    j = 99
    @cast A[i,j] = div(i, j)  # inferred from LHS
    @test A[3,1] == 3

    @reduce B[i] := sum(j) A[i,j] + i + $j  # interpolate j
    @test B[3] > 400

    @test [1, 4, 9] .- im == @cast _[i'] := i'^2 + im'  (i' in 1:3)  # primes

    @cast C[i,j,k] := 0 * A[i,(j,k)] + j  (k in 1:2)  # used to infer sz_j = (:)
    @test all(==(2), C[:,2,:])

    @reduce D[k] := sum(i) B[i]/k (k in 1:4)  # no indexing by k on RHS 
    @test D ≈ vec(sum(B ./ (1:4)', dims=1))

    @reduce E := sum(i,k) i/k (i in 1:2, k in 1:4)  # no indexing on RHS of reduction
    @test E ≈ sum((1:2) ./ (1:4)')
end

@testset "tuples" begin
    x = rand(3, 5)
    @cast vi[j,k] := findmax(x[:, k])[j]
    @test vi[1,1] == maximum(x[:,1])
    @test vi[2,2] == argmax(x[:,2])
end