@testset "colonise!" begin

    using TensorSlice: colonise!

    osz = Any[:(sz[1]), :((sz[2] * sz[3]) * sz[4])] # nested *
     # the : in slist[2] means osz[2] should all be unkonwn
    @test colonise!(osz, [4,:,4,4]) # should return true as sz[1] is still needed
    @test osz[1] == :(sz[1])
    @test osz[2] == (:)

    osz = Any[ :((sz[2] * sz[3]) * sz[1])]
    # since osz[1] contains all 3 sizes, it must be length(A)
    @test !colonise!(osz, [4,4,4])  # return false as sz not required anymore
    @test osz[1] == (:)

    osz = Any[:(sz[1]), :(sz[2] * sz[3] * sz[4])] 
    # now I produce un-nested *
    @test_broken colonise!(osz, [4,4,4]) # should return true as sz[1] is still needed
    @test osz[1] == :(sz[1])
    @test osz[2] == (:)


end
@testset "parse!" begin 

    using TensorSlice: SizeDict, parse!

    ## case "rex" 
    dd = SizeDict()
    ff = Any[]
    flat, getafix, negated = parse!(dd, nothing, [], [:(i:3), :(j:4), :assert], true, ff)
    @test ff == [:assert]
    @test length(dd.dict) == 2
    @test dd.dict[:i] == 3
    @test flat == [:i, :j]

    ## simplest array
    dd = SizeDict()
    flat, getafix, negated = parse!(dd, :Z, [:i, :_, :j, 1, :(-k)], [])
    @test flat == [:i, :j, :k]
    @test getafix == [:, 1, :, 1, :]
    @test_broken negated == [:k]
    @test dd.dict[:k] == :(size(Z, 5))

    ## outside and inside
    dd = SizeDict()
    flat, getafix, negated = parse!(dd, :X, [:a, :(b\-c), 9], [:k, :(-l)])
    @test flat == [:k, :l, :a, :b, :c]
    @test getafix == [:,:,9]
    @test_broken negated == [:l, :c]
    @test_broken dd.dict[:l] == :(size(first(X, 2)))

    ## outside and inside, reduction allowing ranges
    dd = SizeDict()
    flat, getafix, negated = parse!(dd, :Y, [:a, :(-b\c\e)], [:(k:10), :l], true)
    @test dd.dict[:k] == 10
    @test dd.dict[ [:b, :c, :e] ] == :(size(Y, 2))


end
@testset "SizeDict" begin

    using TensorSlice: SizeDict, sizeinfer, savesize!

    dd = SizeDict()
    dd.dict[:a] = 3
    dd.dict[:b] = 4
    dd.dict[[:c,:d]] = 100

    @test_throws ArgumentError sizeinfer(dd, [:a, :b, :c, :d])

    dd.dict[:d] = 10

    ss = sizeinfer(dd, [:a, :b, :c, :d])
    @test_broken eval.(ss) == [3,4,10,10]

    savesize!(dd, :a, 99)
    @test dd.dict[:a] == 99
    @test_throws AssertionError eval.(dd.checks) 

end
@testset "minus" begin 

    using TensorSlice: oddunique, stripminus!

    @test oddunique([:a, :b, :c]) == [:a, :b, :c]
    @test oddunique([:a, :b, :c, :a]) == [:b, :c]
    @test oddunique([:a, :b, :c, :a, :b]) == [:c]
    @test oddunique([:a, :b, :c, :a, :b, :a]) == [:a, :c]

    nn = []
    @test stripminus!(nn, :z) == :z
    @test stripminus!(nn, :(-z)) == :z
    @test nn == [:z]

    @test stripminus!(nn, :((a, b, -c))) == [:a, :b, :c]
    @test nn == [:z, :c]

    @test stripminus!(nn, :(a\b)) == [:a, :b]
    @test stripminus!(nn, :(a\-b\c)) == [:a, :b, :c]
    @test stripminus!(nn, :(-a\b\c\-d)) == [:a, :b, :c, :d]
    @test nn == [:z, :c, :b, :a, :d]

end