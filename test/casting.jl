@testset "simple broadcasting" begin

	b = rand(2)
	c = rand(3)
	bc = rand(2,3)

	Z = zeros(3,2)
	A = bc' .+ b' .* c
	@cast B[j,i] := bc[i,j] .+ b[i] .* c[j]
	@cast Z[j,i] = bc[i,j] .+ b[i] .* c[j]
	@test A ==B == Z

	Z1 = zeros(3,2,1)
	@cast Z1[j,i] = bc[i,j] .+ b[i] .* c[j]
	@test A == Z1[:,:,1]

end
@testset "broadcast reduction" begin

	b = rand(2)
	c = rand(3)
	bc = rand(2,3)

	Z = zeros(3)
	A = sum( bc' .+ b' .* c ; dims=2) |> vec
	@cast B[j] := sum(i) bc[i,j] .+ b[i] .* c[j]
	@cast Z[j] = sum(i)  bc[i,j] .+ b[i] .* c[j]
	@test A ==B == Z

end