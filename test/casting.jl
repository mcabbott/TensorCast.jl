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
@testset "complex" begin

    C = rand(3,3) .+ im .* rand(3,3)
    R = rand(1:33, 3,3)

    @cast A[i,j] := R[i,j] + C[j,i]'
    @test A == R .+ C'

    @cast A[i,j] = R[j,i] + C[i,j]' + 10im
    @test A == R' .+ conj.(C) .+ 10im

    @cast B[i,j] := R[i,j] + C'[i,j]
    @test B == R .+ C'

end
@testset "diag" begin

    B = rand(4,5)
    @test TensorCast.diagview(B) == TensorCast.diag(B) # really LinearAlgebra!

    R = rand(2,2)
    @cast V[i] := R[i,i]
    @cast V2[i] := R[i,i] nolazy

    @test V[1] == R[1,1]
    @test V[2] == R[2,2]
    @test V == V2

    # R4 = rand(1:10, 4)
    # @cast W[i] := R4[i\i]^2 # diag(reshape(R4, (sz_i, sz_i))) with sz_i = (:)
    # @test W[1] == R4[1]^2
    # @test W[2] == R4[4]^2

    M = [repeat([10i+j],2) for i=1:3, j=1:3]
    @cast A[i,k] := M[i,i][k]

    @test A[1,1] == A[1,2] == 11
    @test A[3,1] == A[3,2] == 33

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
@testset "mapslices etc" begin

    M = rand(-10:10, 3,4)
    g(x) = x ./ sum(abs,x)
    @cast N[i,j] := g(M[:,j])[i]
    N2 = mapslices(g, M, dims=1)
    @test N == N2

    V = [ rand(1:99, 2) for _=1:5, _=1:1 ]
    @cast W[i,j] := (v->v .// 2)(V[j,_])[i]
    W2 = hcat(V...) .// 2
    @test W == W2

    fun(m::AbstractMatrix, a=0) = vec(sum(m,dims=1)) .+ a
    X = rand(2,3,4)
    Y = rand(4)
    @cast Z[i,j] := fun(X[:,i,:],42)[j] + Y[j]^2
    Z2 = dropdims(sum(X, dims=1), dims=1) .+ 42 .+ transpose(Y).^2
    @test Z ≈ Z2

    A = rand(2:2:10, 3,2,4)
    ff(v::AbstractVector) = v .^2
    B = rand(-1000:1000:1000, 2,1,4)
    CM = rand(2,3)
    C = eachcol(CM) |> collect
    cast" Y_ijk := ff(A_j:k)_i + B_i_k + (C_j)_i"
    Y2 = permutedims(A, (2,1,3)) .^2 .+ B .+ CM
    @test Y ≈ Y2

end
