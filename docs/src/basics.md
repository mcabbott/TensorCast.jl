# Basics

Install and set up:

```julia-repl
(v1.1) pkg> add TensorCast
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

julia> all(A[j,i] == M[i,j] + 10 * V[i] for i=1:3, j=1:4)
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

julia> all(S[j][i] == M[i,j] for i=1:3, j=1:4)
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
which may be written  either `i⊗j` or `(i,j)`. 
The simplest version of this is precisely what the built-in function `kron` does:

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

julia> all(K[i + 3(j-1)] == V[i] * W[j] for i=1:3, j=1:4)
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

julia> @cast A[i,j] := collect(1:12)[i⊗j]  i ∈ 1:4, j ∈ 1:3
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

Aside, note that providing explicit ranges will also turn on checks of the input, for example:

```julia
julia> @pretty @cast W[i] := V[i]^2  i ∈ 1:3 
begin
    @assert_ 3 == size(V, 1) "range of index i must agree"
    @assert_ ndims(V) == 1 "expected a 1-tensor V[i]"
    W = @__dot__(V ^ 2)
end
```

## Glue & reshape

As a less trivial example of these combined indices, here is one way to combine a list of matrices
into one large grid. Notice that `x` varies faster than `i`, and so on -- the linear order of index `x⊗i`
agrees with that of two indices `x,i`. 

```jldoctest mylabel
julia> list = [ i .* ones(2,2) for i=1:8 ];

julia> @cast mat[x⊗i, y⊗j] := Int(list[i⊗j][x,y])  i in 1:2
4×8 Array{Int64,2}:
 1  1  3  3  5  5  7  7
 1  1  3  3  5  5  7  7
 2  2  4  4  6  6  8  8
 2  2  4  4  6  6  8  8

julia> vec(mat) == @cast _[xi⊗yj] := mat[xi, yj]
true

julia> mat == Int.(hvcat((4,4), transpose(reshape(list,2,4))...))
true
```

Alternatively, this reshapes each matrix to a vector, and makes them columns of the output:

```jldoctest mylabel
julia> @cast colwise[x⊗y,i] := Int(list[i][x,y])
4×8 Array{Int64,2}:
 1  2  3  4  5  6  7  8
 1  2  3  4  5  6  7  8
 1  2  3  4  5  6  7  8
 1  2  3  4  5  6  7  8

julia> colwise == Int.(reduce(hcat, vec.(list)))
true
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

julia> using Random; Random.seed!(42); 

julia> @cast M3[i,j] |= M[i,~j]
3×4 Array{Int64,2}:
 7  4  1  10
 8  5  2  11
 9  6  3  12
```

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

## Arrays of indices or functions

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

You can also index one array using another, this example is just `view(M, :, ind)`:

```jldoctest mylabel
julia> ind = [1,1,2,2,4];

julia> @cast _[i,j] := M[i,ind[j]]
3×5 view(::Array{Int64,2}, :, [1, 1, 2, 2, 4]) with eltype Int64:
 1  1  4  4  10
 2  2  5  5  11
 3  3  6  6  12
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

