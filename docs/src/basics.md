# Basics

Install and set up:

```julia-repl
(v1.5) pkg> add TensorCast
```

```jldoctest mylabel
julia> using TensorCast

julia> V = [10,20,30];

julia> M = collect(reshape(1:12, 3,4))
3×4 Array{Int64,2}:
 1  4  7  10
 2  5  8  11
 3  6  9  12
```

## Broadcasting

Here's a simple use of `@cast`, broadcasting addition over a matrix, a vector, and a scalar. 
Using `:=` makes a new array, after which the left and right will be equal for all possible 
values of the indices:

```jldoctest mylabel
julia> @cast A[j,i] := M[i,j] + 10 * V[i]
4×3 Array{Int64,2}:
 101  202  303
 104  205  306
 107  208  309
 110  211  312

julia> all(A[j,i] == M[i,j] + 10 * V[i] for i in 1:3, j in 1:4)
true
```

To make this happen, first `M` is transposed, and then the axis of `V` is re-oriented 
to lie in the second dimension. The macro `@pretty` prints out what `@cast` produces:

```julia
julia> @pretty @cast A[j,i] := M[i,j] + 10 * V[i]
begin
    local panda = transmute(M, (2, 1))
    local bat = transmute(V, (nothing, 1))
    A = @__dot__(panda + 10bat)
end
```

The function `transmute` is a generalised version of `permutedims`. It transposes `M`,
and then places `V`'s first dimension to lie along the second dimension -- also a transpose,
here, it allows for things like `transmute(M, (2,nothing,1))`. This is from [TransmuteDims.jl](https://github.com/mcabbott/TransmuteDims.jl).

Of course `@__dot__` is the full name of the broadcasting macro `@.`, 
which simply produces `A = pelican .+ 10 .* termite`.
And the animal names (in place of generated symbols) are from the indispensible [MacroTools.jl](https://github.com/MikeInnes/MacroTools.jl).

## Slicing

Here are two ways to slice `M` into its columns,
both of these produce `S = sliceview(M, (:, *))` which is simply `collect(eachcol(M))`:

```jldoctest mylabel
julia> @cast S[j][i] := M[i,j]
4-element Array{SubArray{Int64,1,Array{Int64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true},1}:
 [1, 2, 3]
 [4, 5, 6]   
 [7, 8, 9]   
 [10, 11, 12]

julia> all(S[j][i] == M[i,j] for i in 1:3, j in 1:4)
true

julia> @cast S2[j] := M[:,j]; 

julia> S == S2
true
```

We can glue slices back together with the same notation, `M2[i,j] := S[j][i]`. 
Combining this with slicing from `M[:,j]` is a convenient way to perform `mapslices` operations:

```jldoctest mylabel
julia> @cast C[i,j] := cumsum(M[:,j])[i]
3×4 Array{Int64,2}:
 1   4   7  10
 3   9  15  21
 6  15  24  33

julia> C == mapslices(cumsum, M, dims=1)
true
```

This is both faster and more general than `mapslices`, 
as it does not have to infer what shape the function produces:

```julia-repl
julia> using BenchmarkTools

julia> f(M) = @cast _[i,j] := cumsum(M[:,j])[i];

julia> M10 = rand(10,1000);

julia> @btime mapslices(cumsum, $M10, dims=1);
  630.725 μs (7508 allocations: 400.28 KiB)

julia> @btime f($M10);
  64.056 μs (2006 allocations: 297.25 KiB)
```

It's also more readable, in less trivial examples:

```jldoctest mylabel
julia> using LinearAlgebra, Random; Random.seed!(42);

julia> T = rand(3,3,5);

julia> @cast E[i,n] := eigen(T[:,:,n]).values[i];

julia> E == dropdims(mapslices(x -> eigen(x).values, T, dims=(1,2)), dims=2)
true
```

## Fixed indices

Sometimes it's useful to insert a trivial index into the output -- for instance 
to match the output of `mapslices(x -> eigen(...` just above:

```jldoctest mylabel
julia> @cast E3[i,_,n] := eigen(T[:,:,n]).values[i];

julia> size(E3)
(3, 1, 5)
```

Using `_` on the right will demand that `size(E3,2) == 1`;
but you can also fix indices to other values. 
If these are variables not integers, they must be interpolated `M[$row,j]` 
to distinguish them from index names:

```jldoctest mylabel
julia> col = 3;

julia> @cast O[i,j] := (M[i,1], M[j,$col])
3×3 Array{Tuple{Int64,Int64},2}:
 (1, 7)  (1, 8)  (1, 9)
 (2, 7)  (2, 8)  (2, 9)
 (3, 7)  (3, 8)  (3, 9)
```

## Reshaping

Sometimes it's useful to combine two (or more) indices into one, 
which may be written  either `(i,j)` or `i⊗j`:

```jldoctest mylabel
julia> vec(M) == @cast _[(i,j)] := M[i,j]
true
```

The next-simplest version of this is precisely what the built-in function `kron` does:

```jldoctest mylabel
julia> W = 1 ./ [10,20,30,40];

julia> @cast K[_, i⊗j] := V[i] * W[j]
1×12 Array{Float64,2}:
 1.0  2.0  3.0  0.5  1.0  1.5  0.333333  0.666667  1.0  0.25  0.5  0.75

julia> @cast K[1, (i,j)] := V[i] * W[j]  # identical!
1×12 Array{Float64,2}:
 1.0  2.0  3.0  0.5  1.0  1.5  0.333333  0.666667  1.0  0.25  0.5  0.75

julia> K == kron(W, V)'
true

julia> all(K[i + 3(j-1)] == V[i] * W[j] for i in 1:3, j in 1:4)
true

julia> Base.kron(A::Array{T,3}, X::Array{T′,3}) where {T,T′} =    # extend kron to 3-tensors
           @cast _[x⊗a, y⊗b, z⊗c] := A[a,b,c] * X[x,y,z]
```

If an array on the right has a combined index, then it may be ambiguous how to divide up its range. You can resolve this by providing explicit ranges, after the main expression: 

```jldoctest mylabel
julia> @cast A[i,j] := collect(1:12)[i⊗j]  i in 1:2
2×6 Array{Int64,2}:
 1  3  5  7   9  11
 2  4  6  8  10  12

julia> @cast A[i,j] := collect(1:12)[i⊗j]  (i ∈ 1:4, j ∈ 1:3)
4×3 Array{Int64,2}:
 1  5   9
 2  6  10
 3  7  11
 4  8  12
```

Writing into a given array with `=` instead of `:=` will also remove ambiguities, 
as `size(A)` is known:

```jldoctest mylabel
julia> @cast A[i,j] = 10 * collect(1:12)[i⊗j];
```

## Glue & reshape

As a less trivial example of these combined indices, here is one way to combine a list of matrices
into one large grid. Notice that `x` varies faster than `i`, and so on -- the linear order of index `x⊗i`
agrees with that of two indices `x,i`. 

```jldoctest mylabel
julia> list = [fill(k,2,2) for k in 1:8];

julia> @cast mat[x⊗i, y⊗j] |= list[i⊗j][x,y]  i in 1:2
4×8 Array{Int64,2}:
 1  1  3  3  5  5  7  7
 1  1  3  3  5  5  7  7
 2  2  4  4  6  6  8  8
 2  2  4  4  6  6  8  8

julia> vec(mat) == @cast _[xi⊗yj] := mat[xi, yj]
true

julia> mat == hvcat((4,4), transpose(reshape(list,2,4))...)
true
```

Alternatively, this reshapes each matrix to a vector, and makes them columns of the output:

```jldoctest mylabel
julia> @cast colwise[x⊗y,i] := (list[i][x,y])^2
4×8 Array{Int64,2}:
 1  4  9  16  25  36  49  64
 1  4  9  16  25  36  49  64
 1  4  9  16  25  36  49  64
 1  4  9  16  25  36  49  64

julia> colwise == reduce(hcat, vec.(list)) .^ 2
true
```

## Index values

Mostly the indices appearing in `@cast` expressions are just notation, to indicate what permutation / reshape is required. 
But if an index appears outside of square brackets, this is understood as a value, implemented by broadcasting over a range (appropriately permuted):

```jldoctest
julia> @cast rat[i,j] := 0 * M[i,j] + i // j
3×4 Array{Rational{Int64},2}:
 1//1  1//2  1//3  1//4
 2//1  1//1  2//3  1//2
 3//1  3//2  1//1  3//4

julia> rat == @cast _[i,j] := axes(M,1)[i] // axes(M,2)[j]  # more verbose @cast
true

julia> rat == axes(M,1) .// transpose(axes(M,2))  # what it aims to generate
true

julia> @cast _[r,c] := r + 10c  (r in 1:2, c in 1:7)  # no array for range inference
2×7 Array{Int64,2}:
 11  21  31  41  51  61  71
 12  22  32  42  52  62  72
```

Writing `$i` will interpolate the variable `i`, distinct from the index `i`:

```jldoctest
julia> i, k = 10, 100;

julia> @cast ones(3)[i] = i + $i + k
3-element Vector{Float64}:
 111.0
 112.0
 113.0
```

## Reverse & shuffle

A minus in front of an index will reverse that direction, and a tilde will shuffle it. 
Both create views, which you may explicitly `collect` using `|=`:

```jldoctest mylabel
julia> @cast M2[i,j] := M[i,-j]
3×4 view(::Array{Int64,2}, :, 4:-1:1) with eltype Int64:
 10  7  4  1
 11  8  5  2
 12  9  6  3

julia> all(M2[i,j] == M[i, end+begin-j] for i in 1:3, j in 1:4)
true

julia> using Random; Random.seed!(42); 

julia> @cast M3[i,j] |= M[i,~j]
3×4 Array{Int64,2}:
 7  4  1  10
 8  5  2  11
 9  6  3  12
```

Note that the minus is a slight deviation from the rule that left equals right for all indices,
it should really be `M[i, end+1-j]`.

## Primes `'`

Acting on indices, `A[i']` is normalised to `A[i′]` unicode \prime (which looks identical in some fonts).
Acting on elements, `C[i,j]'` means `adjoint.(C)`, elementwise complex conjugate, equivalent to `(C')[j,i]`.
If the elements are matrices, as in `C[:,:,k]'`, then  `adjoint` is conjugate transpose.

```jldoctest mylabel
julia> @cast C[i,i'] := (1:4)[i⊗i′] + im  (i ∈ 1:2, i′ ∈ 1:2)
2×2 Array{Complex{Int64},2}:
 1+1im  3+1im
 2+1im  4+1im

julia> @cast _[i,j] := C[i,j]'
2×2 Array{Complex{Int64},2}:
 1-1im  3-1im
 2-1im  4-1im

julia> C' == @cast _[j,i] := C[i,j]'
true
```

## Splats

You can use a slice of an array as the arguments of a function, or a `struct`:

```jldoctest mylabel
julia> struct Tri{T} x::T; y::T; z::T end;

julia> @cast triples[k] := Tri(M[:,k]...)
4-element Array{Tri{Int64},1}:
 Tri{Int64}(1, 2, 3)
 Tri{Int64}(4, 5, 6)
 Tri{Int64}(7, 8, 9)
 Tri{Int64}(10, 11, 12)

julia> ans == Base.splat(Tri).(eachcol(M))
true
```

This one can also be done as `reinterpret(reshape, Tri{Int64}, M)`. 
But what would be smarter in the general case is to do one splat, not many:

```julia-repl
julia> Tri.(eachrow(M)...)
4-element Vector{Tri{Int64}}:
 Tri{Int64}(1, 2, 3)
 Tri{Int64}(4, 5, 6)
 Tri{Int64}(7, 8, 9)
 Tri{Int64}(10, 11, 12)

julia> @btime Base.splat(tuple).(eachcol(m))  setup=(m=rand(4,100));
  38.041 μs (1411 allocations: 48.33 KiB)

julia> @btime tuple.(eachrow(m)...)  setup=(m=rand(4,100));
  824.256 ns (12 allocations: 4.06 KiB)
```

## Arrays of functions

Besides arrays of numbers (and arrays of arrays) you can also broadcast an array of functions,
which is done by calling `Core._apply(f, xs...) = f(xs...)`: 

```jldoctest mylabel; filter = r"begin\n.*\n.*\nend"
julia> funs = [identity, sqrt];

julia> @cast applied[i,j] := funs[i](V[j])
2×3 Array{Real,2}:
 10        20        30      
  3.16228   4.47214   5.47723

julia> @pretty @cast applied[i,j] := funs[i](V[j])
begin
    local pelican = transmute(V, (nothing, 1))
    applied = @__dot__(_apply(funs, pelican))
end
```

## Repeated indices

The only situation in which repeated indices are allowed is when they either 
extract the diagonal of a matrix, or create a diagonal matrix:

```jldoctest mylabel
julia> @cast D[i] |= C[i,i]
2-element Array{Complex{Int64},1}:
 1 + 1im
 4 + 1im

julia> D2 = @cast _[i,i] := V[i]
3×3 Diagonal{Int64,Array{Int64,1}}:
 10   ⋅   ⋅
  ⋅  20   ⋅
  ⋅   ⋅  30
```

All indices appearing on the left must also appear on the right. 
There is no implicit sum over repeated indices on different tensors.
To sum over things, you need `@reduce` or `@matmul`, described on the next page.

