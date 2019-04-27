@testset "basics" begin

	bc = rand(2,3)
	bcde = rand(2,3,4,5);

	@reduce C[c] := sum(b) bc[b,c]
	@test C == vec(sum(bc, dims=1))

	@reduce B[b] := prod(c:3) bc[b,c] !
	@test B == vec(prod(bc, dims=2))

	using Statistics

	@reduce W[b\d] := Statistics.mean(c,e) bcde[b,c,d,e]
	V = vec(mean(bcde, dims=(2,4)))
	@test V == W

	@reduce Z[e,b,β] := Statistics.std(c,α:2) bcde[b,c,α\β,e] assert
	@test size(Z) == (5,2,2)

	@reduce A[_,e,_,b] := sum(c) bcde[b,c,3,e]
	@test size(A) == (1,5,1,2)
	B = sum(bcde[:,:,3,:], dims=2)
	@test A[1,3,1,2] == B[2,1,3]
	C = similar(A);
	@reduce C[_,e,_,b] = sum(c) bcde[b,c,3,e]
	@test A ≈ C

	@reduce D[b][c] := sum(d,e) bcde[b,c,d,e] + 1
	@reduce E[b][c] := sum(e,d) bcde[b,c,d,e] + 1
	@reduce F[c][b] := sum(d:4,e:5) bcde[b,c,d,e] + 1

	@test D[1][2] ≈ sum(bcde, dims=(3,4))[1,2, 1,1] + 4*5
	@test D[1][2] ≈ E[1][2] ≈ F[2][1]

end
@testset "scalar" begin

	bc = rand(2,3)

	@reduce S[] := sum(i,j) bc[j,i]
	T = @reduce sum(i,j) bc[j,i]
	U = @reduce sum(j,i) bc[j,i]
	@test S[] ≈ sum(bc) ≈ T ≈ U

end
@testset "inference" begin

    # inference for a\b\c had an (Any[]...) dots problem at first
    B = randn(8,24);
    @reduce A[b,c, y,z] := sum(a:2, x:2) B[a\b\c, x⊗y⊗z]  b:2, y:3, assert
    @test size(A) == (2,2, 3,4)

    C = similar(A)
    @reduce C[b,c, y,z] = sum(a, x) B[a\b\c, x⊗y⊗z]  assert
    @test C ≈ A

    # with a:2 given, it doesn't try product inference, just leaves a :, which is OK.
    @reduce C[b,c, y,z] = sum(a:2, x) B[a\b\c, x⊗y⊗z]  assert
    @test C ≈ A

end
@testset "recursion" begin

    # from readme
    A = rand(4);
    B = randn(4,4);
    R = @reduce sum(i) A[i] * log( @reduce [i] := sum(j) A[j] * exp(B[i,j]) )
    @reduce inner[i] := sum(j) A[j] * exp(B[i,j])
    S = @reduce sum(i) A[i] * log(inner[i])
    @test S == R

    R2 = @reduce sum(i) A[i] * log( @reduce inner[i] = sum(j) A[j] * exp(B[i,j]) )
    @test S == R2

    # scalar result inside
    tot = @reduce sum(i) A[i]
    @cast N[i] := A[i] / tot
    @cast M[i] := A[i] / @reduce sum(i) A[i]
    @test N == M

end
