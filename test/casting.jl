@testset "simple broadcasting" begin

	b = rand(2)
	c = rand(3)
	bc = rand(2,3)

	## compare to ordinary broadcasting
	Z = zeros(3,2)
	A = bc' .+ b' .* c .^ 3
	@cast B[j,i] := bc[i,j] .+ b[i] .* c[j]^3
	@cast Z[j,i] = bc[i,j] .+ b[i] .* c[j]^3
	@test A ==B == Z

	Z1 = zeros(3,2,1)
	@cast Z1[j,i] = bc[i,j] .+ b[i] .* c[j]^3
	@test A == Z1[:,:,1]

	A = @. exp(-bc + 2*c') / sqrt(b)	
	@cast B[i,j] := exp(-bc[i,j] + 2*c[j]) / sqrt(b[i])
	@test A == B

	## test permutedims in orient function
	bcdef = rand(2,3,4,5,6);
	@cast A[d,e,c,f,b] := 10 + bcdef[b,c,d,e,f];
	@test A[3,4,2,5,1] == 10 + bcdef[1,2,3,4,5]

	## check that f(A) is allowed as array name
	exp_all(A) = similar(A) .= exp.(A)
	A = exp_all(bc) .* sqrt.(b .+ 3)
	@cast B[i,j] := exp_all(bc)[i,j] * sqrt(b[i] + 3)
	@test A == B

	## check that log(2) doesn't get broadcast
	global log_n = 0
	two=2.0
	log_count(x) = (global log_n += 1; log(x))
	A = log.(bc) ./ log_count(2)
	@cast B[i,j] := log(bc[i,j]) / log_count(2)
	@cast C[i,j] := log(bc[i,j]) / log_count(two)
	@test A == B == C
	@test log_n == 3

end
@testset "broadcast reduction" begin

	b = rand(2)
	c = rand(3)
	bc = rand(2,3)

	Z = zeros(3)
	A = sum( bc' .+ b' .* c ; dims=2) |> vec
	@reduce B[j] := sum(i) bc[i,j] .+ b[i] .* c[j]
	@reduce Z[j] = sum(i)  bc[i,j] .+ b[i] .* c[j]
	@test A ==B == Z

	## complete reduction
	@reduce S[] := sum(i,j) bc[j,i] + 1
	T = @reduce sum(i,j) bc[j,i] + 1
	@test S[] ≈ sum(bc .+ 1) ≈ T

	## with output shaping / fixing
	bcde = rand(2,3,4,5);
	@reduce A[d\c,_,b] := sum(e) bcde[-b,c,d,e]
	B = similar(A);
	@reduce B[d\c,_,b] = sum(e) bcde[-b,c,d,e] assert
	@test A ≈ B

	@reduce A[_,d\b,_] := sum(e,c) bcde[b,c,d,e]
	B = similar(A);
	@reduce B[_,d\b,_] = sum(c,e) bcde[b,c,d,e] assert
	@test A ≈ B

end
@testset "anonymous functions" begin

	f1 = @cast A[i] + B[j]^2 => C[i,j]
	f2 = @cast (A[i] + B[j]^2) -> C[i,j]

	A = rand(3)
	B = rand(2)
	@cast C[i,j] := A[i] + B[j]^2

	@test f1(A,B) == f2(A,B) == C

	f3 = @cast D[i\j,k] -> E[i,-k,j]  i:2
	f4 = @cast D[i\j,k] => E[i,-k,j]  i:2, nolazy

	D = rand(4,3)
	@cast E[i,k,j] := D[i\j,-k]  i:2

	@test f3(D) == f4(D) == E

end