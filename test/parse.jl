@testset "parse!" begin 

    using TensorCast: SizeDict, parse!, needview!

    ## case "rex" 
    dd = SizeDict()
    ff = Any[]
    flat, getafix, negated = parse!(dd, nothing, [], [:(i:3), :(j:4) ]; allowranges=true, flags=ff)
    @test unique(ff) == [:assert]
    @test length(dd.dict) == 2
    @test dd.dict[:i] == 3
    @test flat == [:i, :j]

    ## simplest array
    dd = SizeDict()
    flat, getafix, negated = parse!(dd, :Z, [:i, :_, :j, 1, :(-k)], [])
    @test flat == [:i, :j, :k]
    @test getafix == [:, :_, :, 1, :]
    @test needview!(getafix)          # true because it contains 1
    @test getafix == [:, 1, :, 1, :]  # mutated by needview!
    @test_broken negated == [:k]
    @test dd.dict[:k] == :(size(Z, 5))

    ## outside and inside
    dd = SizeDict()
    flat, getafix, negated = parse!(dd, :X, [:a, :(b\-c), 9], [:k, :(-l)])
    @test flat == [:k, :l, :a, :b, :c]
    @test getafix == [:,:,9]
    @test needview!(getafix)
    @test_broken negated == [:l, :c]
    @test_broken dd.dict[:l] == :(size(first(X, 2)))

    ## outside and inside, reduction allowing ranges
    dd = SizeDict()
    flat, getafix, negated = parse!(dd, :Y, [:a, :(-b\c\e)], [:(k:10), :l], allowranges=true)
    @test dd.dict[:k] == 10
    @test dd.dict[ [:b, :c, :e] ] == :(size(Y, 2))


end
@testset "SizeDict" begin

    using TensorCast: SizeDict, sizeinfer, savesize!, MacroError

    dd = SizeDict()
    dd.dict[:a] = 3
    dd.dict[:b] = 4
    dd.dict[[:c,:d]] = 100

    @test_throws MacroError sizeinfer(dd, [:a, :b, :c, :d])

    dd.dict[:d] = 10

    ss = sizeinfer(dd, [:a, :b, :c, :d], nothing, true) # leave one :
    @test eval.(ss) == [3,4,:,10]

    ss = sizeinfer(dd, [:a, :b, :c, :d], nothing, false)
    @test eval.(ss) == [3,4,10,10]

    savesize!(dd, :a, 99)
    @test dd.dict[:a] == 3 # logic changed here, first value kept
    @test_throws DimensionMismatch eval.(dd.checks) 

end
@testset "minus" begin 

    using TensorCast: oddunique, stripminus!

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