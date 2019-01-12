@testset "parse-time" begin
	
	using TensorSlice: _check!

	@check! throw=true
	
	A = rand(3)
	@check! A[i]
	
	@test_throws ErrorException _check!(:(A[i,j])) # parse-time error
	@test_throws ErrorException _check!(:(A[z]))

	@check! size=true
	@check! A[j]

	B = rand(4)
	@test_throws ErrorException @check! B[j] # run-time error

end