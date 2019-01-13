@testset "check" begin
	
	using TensorSlice: _check!, index_store, size_store

	@check!  throw=true # empty
	
	A = rand(3)
	@check! A[i]

	@test_throws ErrorException _check!(:(A[i,j])) # parse-time error
	@test_throws ErrorException _check!(:(A[z]))

	@check! size=true
	@check! A[j]

	B = rand(4)
	@test_throws ErrorException @check! B[j] # run-time error

	sq(x) = x .^ 2
	@check! sq(B)[j] # these should be ignored
	@check! B[(i,j)]
	@check! B[-z]
	@check! B[2]
	@test length(index_store) == 2

    @test_throws ErrorException _check!(:(B[i,2]))
    @test_throws ErrorException _check!(:(B[i,j\k]))

    E = rand(5)
    @check! E[abc]

end
@testset "shape! reduce! cast!" begin

	using TensorSlice: _shape, _reduce, _cast

	@check!  throw=true size=false # empty

	A = rand(3)
	C = rand(3)
	@shape! A[j] = C[j]

	@test_throws ErrorException _shape( :( A[z] = C[z] ); icheck=true) 
	@test_throws ErrorException _reduce( :( A[z] := sum(i) ), :( D[z,i] ); icheck=true) 


end