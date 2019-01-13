@testset "basics" begin

	bc = rand(2,3)
	bcde = rand(2,3,4,5)

	@reduce C[c] := sum(b) bc[b,c] 
	@test C == vec(sum(bc, dims=1))

	@reduce B[b] := prod(c:3) bc[b,c] !
	@test B == vec(prod(bc, dims=2))

	using Statistics

	@reduce W[b\d] := Statistics.mean(c,e) bcde[b,c,d,e]
	V = vec(mean(bcde, dims=(2,4)))
	@test V == W

end
@testset "scalar" begin

	bc = rand(2,3)

	@reduce S[] := sum(i,j) bc[j,i]
	@test S[] ≈ sum(bc)

end