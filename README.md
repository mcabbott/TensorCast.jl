
# TensorCast.jl

[![Build Status](https://travis-ci.org/mcabbott/TensorCast.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorCast.jl)
[![Documentation](https://camo.githubusercontent.com/f7b92a177c912c1cc007fc9b40f17ff3ee3bb414/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f63732d737461626c652d626c75652e737667)](https://pkg.julialang.org/docs/TensorCast/lkx9a/0.1.3/)

This package lets you write expressions involving many-dimensional arrays in index notation,
by defining a few macros. The first is `@cast`, which deals both with "casting" into new shapes 
(including going to and from an array-of-arrays) and with broadcasting:

```julia
@cast A[row][col] := B[row, col]            # slice a matrix B into its rows

@cast C[(i,j), (k,ℓ)] := D[i,j,k,ℓ]         # reshape a 4-tensor D to give a matrix

@cast E[x,y] = F[x]^2 * exp(G[y])           # broadcast E .= F.^2 .* exp.(G') into existing E
```

Next, `@reduce` takes sums (or other reductions) over some directions, 
but otherwise understands all the same things: 

```julia
@reduce H[a] := sum(b,c) L[a,b,c]                # sum over dims=(2,3), and dropdims

@reduce S[i] = sum(n) -P[i,n] * log(P[i,n]/Q[n]) # sum!(S, @. -P*log(P/Q')) into exising S

@reduce W[μ,ν,_,J] := prod(i:2) V[(i,J)][μ,ν]    # products of pairs of matrices, stacked
```

<!-- # master only for now
  
Finally `@mul` handles matrix multiplication of exactly two tensors:

```julia
@mul T[i,_,j] := U[i,k,k′] * V[(k,k′),j]    # matrix multiplication, summing over (k,k′)

@mul W[β][i,j] := X[i,k,β] * Y[k,j,β]       # batched W[β] = X[:,:,β] * Y[:,:,β] ∀ β
```
-->

These are intended to complement the macros from some existing packages.
[TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) 
performs Einstein-convention contractions and traces, in a very similar notation. 
Here it is implicit that repeated indices are summed over: 

```julia
@tensor A[i] := B[i,j] * C[j,k] * D[k]      # matrix multiplication, A = B * C * D
@tensor D[i] := E[i] + F[i,k,k]             # partial trace of F only, Dᵢ = Eᵢ + Σⱼ Fᵢⱼⱼ
```

Similar notation is also used by the macro from [Einsum.jl](https://github.com/ahwillia/Einsum.jl),
which sums the entire right hand side over any indices not appearing on the left. 
This allows arbitrary (element-wise) functions:

```julia
@einsum S[i] := -P[i,n] * log(P[i,n]/Q[n])  # sum over n, for each i (also with @reduce above)
@einsum G[i] := E[i] + F[i,k,k]             # the sum includes everyting:  Gᵢ = Σⱼ (Eᵢ + Fᵢⱼⱼ)
```

There is some overlap of operations which can be done with two (or all three) of these packages. 
However they produce very different code for actually doing what you request. 
The original `@einsum` simply writes the necessary set of nested loops. 
Instead `@tensor` works out a sequence of contraction and trace operations, 
calling optimised BLAS routines where possible. 
(And [ArrayMeta.jl](https://github.com/shashi/ArrayMeta.jl) aimed to do a wide variety of operations efficiently, 
but seems to be abandonned.)

The  macros from this package aim instead to produce simple Julia commands: 
often just a string of `reshape` and `permutedims` and `eachslice` and so on,
plus a native [broadcasting expression](https://julialang.org/blog/2017/01/moredots) if needed, 
and `sum` /  `sum!`, or `*` / `mul!`. 
This means that they are very generic, and will (mostly) work well 
with small [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl), 
with [Flux](https://github.com/FluxML/Flux.jl)'s TrackedArrays, 
on the GPU via [CuArrays](https://github.com/JuliaGPU/CuArrays.jl),
and on almost anything else.

For those who speak Python, `@cast` and `@reduce` allow similar operations to 
[`einops`](https://github.com/arogozhnikov/einops) (minus the cool video, but plus broadcasting)
while Einsum / TensorOperations map very roughly to [`einsum`](http://numpy-discussion.10968.n7.nabble.com/einsum-td11810.html) 
/ [`opt_einsum`](https://github.com/dgasmith/opt_einsum).
The function of `@check!` (see [below](#checking)) is similar to [`tsalib`](https://github.com/ofnote/tsalib)'s shape annotations.

## Installation

You need [Julia](https://julialang.org/downloads/) 1.0 or later:

```julia
] add TensorCast
```

There is help available as `? @cast` etc, or at [pkg.julialang.org](https://pkg.julialang.org/docs/TensorCast/lkx9a/0.1.3/).
And also some notebooks in folder [/docs/](https://github.com/mcabbott/TensorCast.jl/tree/master/docs). 

## Inside

Use the macro `@pretty` to print out the generated expression: 

```julia
@pretty @cast A[(i,j)] = B[i,j]
# copyto!(A, B)

@pretty @cast A[k][i,j] := B[i,(j,k),3]  k:length(C)
# begin
#     @assert_ ndims(B) == 3 "expected a 3-tensor B[i, (j, k), 3]"
#     local (sz_i, sz_j, sz_k) = (size(B, 1), :, length(C))
#     local emu = reshape(view(B, :, :, 3), (sz_i, sz_j, sz_k))
#     A = sliceview(emu, (:, :, *))
# end

@pretty @reduce V[r] = sum(c) exp( fun(M)[r,c]^2 / R[c]' ) * D[c,c]
# begin
#     local bat = fun(M)  # your animals may vary
#     local hare = orient(R, (*, :))
#     local zebra = orient(diag(D), (*, :))
#     sum!(V, (*).(exp.((/).((^).(bat, 2), Base.conj.(hare))), zebra))
# end
```

Here `TensorCast.sliceview(D, (:,:,*)) = collect(eachslice(D, dims=3))`, 
and  `TensorCast.orient(R, (*,:))` will reshape or tranpose `R` to lie long the second direction. 
Notice that `R[c]'` means element-wise complex conjugation,
and `D[c,c]` means `diag(D)` -- this is the only repeated index allowed. 

(`@pretty` is just a variant of the built-in `@macroexpand1`, with animal names from
[MacroTools.jl](https://github.com/MikeInnes/MacroTools.jl) in place of generated symbols.)

## Checking

When writing complicated index expressions by hand, it is conventional to use different groups of letters 
for indices which mean different things. If `a,b,c...` label some objects, while `μ,ν,...` are components 
of their positions (in units of meters) then any expression which mixes these up is probably a mistake. 
This package also can automate checks for such mistakes: 

```julia
@reduce!  A[α] := sum(k) B[α,k]     # memorises that A takes α, etc.
@cast!  C[α,β] := A[α] * A[β]       # no problem: β is close to α
@cast! D[n][β] := C[n,β]            # warning! C does not accept n
```

There are also macros `@tensor!` and `@einsum!` which perform the same checks, 
before calling the usual `@tensor` / `@einsum`. 

If you need to leave index notation and return, you can insert `@check!` to confirm. 
(The `!` is because it alters a dictionary, off-stage somewhere.)

```julia
@cast! E[α,_,β,_] := C[α,β]         # reshape to size(E,2) == size(D,4) == 1
F = calculate(A,E)
@check! F[n,α]                      # just the check, with no calculation
```

These macros are (by definition) run when your code is loaded, not during the calculation, 
and thus such checks have zero speed penalty. But you can turn on explicit run-time size checks too 
(and, if you wish, an error not a warning) by passing these options:

```julia
@check!  size=true  throw=true
```

After this, `@check!(A[α])` will insert the function `check!(A, ...)` which (when run) saves the range 
of every distinct index name, and gives an error if it is subsequently used to indicate a dimension of different size. This is based on the complete name, thus `α` and `α2` may have distinct ranges, 
while the above slot-checking is based on the first letter.  

(For now there is one global list of settings, index names, and run-time sizes.)

## Options

Expressions with `=` write into an existing array, 
while those with `:=` do not. This is the same notation as 
[TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [Einsum.jl](https://github.com/ahwillia/Einsum.jl). 
But unlike those packages, sometimes the result of `@cast` is a view of the original, for instance 
`@cast A[i,j] := B[j,i]` gives `A = transpose(B)`. You can forbid this, and insist on a copy, 
by writing `|=` instead. And conversely, if you expect a view, writing `==` will give an error if not.

Various other options can be given after the main expression. `assert` turns on explicit size checks, 
and ranges like `i:3` specify the size in that direction (sometimes this is necessary to specify the shape).
Adding these to the example above: 
```julia
@pretty @cast A[(i,j)] = B[i,j]  i:3, assert
# begin
#     @assert_ ndims(B) == 2 "expected a 2-tensor B[i, j]"
#     @assert_ 3 == size(B, 1) "range of index i must agree"
#     @assert_ ndims(A) == 1 "expected a 1-tensor A[(i, j)]"
#     copyto!(A, B)
# end
```

### Ways of slicing

The default way of slicing creates an array of views, 
but if you use `|=` instead then you get copies: 

```julia
M = rand(1:99, 3,4)

@cast S[k][i] := M[i,k]             # collect(eachcol(M)) ≈ [ view(M,:,k) for k=1:4 ]
@cast S[k][i] |= M[i,k]             # [ M[:,k] for k=1:4 ]; using |= demands a copy
```

The default way of un-slicing is `reduce(hcat, ...)`, which creates a new array. 
But there are other options, controlled by keywords after the expression:

```julia
@cast A[i,k] := S[k][i]             # A = reduce(hcat, B)
@cast A[i,k] := S[k][i]  cat        # A = hcat(B...); often slow
@cast A[i,k] := S[k][i]  lazy       # A = VectorOfArrays(B)

size(A) == (3, 4) # true
```

The option `lazy` uses [RecursiveArrayTools.jl](https://github.com/JuliaDiffEq/RecursiveArrayTools.jl)
to create a view of the original vectors. This would also be possible with 
[JuliennedArrays.jl](https://github.com/bramtayl/JuliennedArrays.jl), I may change what gets used later. 

Another kind of slices are provided by [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl),
in which a Vector of SVectors is just a different interpretation of the same memory as a Matrix. 
By another slight abuse of notation, such slices are written here as curly brackets:

```julia
using StaticArrays

@cast S[k]{i} == M[i,k]  i:3        # S = reinterpret(SVector{3,Int}, vec(M)) 
@cast R[k,i] == S[k]{i}             # such slices can be reinterpreted back again
```

Both `S` and `R` here are views of the original matrix `M`. 
When creating such slices, their size ought to be provided, either as a literal integer or 
through the types. Note that you may also write `S[k]{i:3}`. 

### Better broadcasting

When broadcasting and then summing over some directions, it can be faster to avoid creating the 
entire array, then throwing it away. This can be done with the package 
[LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl) which has a lazy `BroadcastArray`. 
In the following example, the product `V .* V' .* V3` contains about 1GB of data, 
the writing of which is avoided by giving the option `lazy`: 

```julia
V = rand(500); V3 = reshape(V,1,1,:);

@time @reduce W[i] := sum(j,k) V[i]*V[j]*V[k];        # 0.6 seconds, 950 MB
@time @reduce W[i] := sum(j,k) V[i]*V[j]*V[k]  lazy;  # 0.025 s, 5 KB
```

Finally, the package [Strided.jl](https://github.com/Jutho/Strided.jl) can apply multi-threading to 
broadcasting, and some other magic. You can enable it with the option `strided`, like this: 

```julia
using Strided # and export JULIA_NUM_THREADS = 4 before starting
A = randn(4000,4000); B = similar(A);

@time @cast B[i,j] = (A[i,j] + A[j,i])/2;             # 0.12 seconds
@time @cast B[i,j] = (A[i,j] + A[j,i])/2 strided;     # 0.025 seconds
```

### Less lazy

To disable the default use of `PermutedDimsArray` etc, give the option `nolazy`: 

```julia
@pretty @cast Z[y,x] := M[x,-y]  nolazy
# Z = reverse(permutedims(M), dims=1)
@pretty @cast Z[y,x] := M[x,-y] 
# Z = Reverse{1}(PermutedDimsArray(M, (2, 1)))
@pretty @cast Z[y,x] |= M[x,-y]
# Z = copy(Reverse{1}(PermutedDimsArray(M, (2, 1))))
```

Here `TensorCast.Reverse{1}(B)` creates a view, with `reverse(axes(B,1))`.

## Caveat Emptor

Some new features, not well tested, and some only on master branch:

### Recursion

The macro now looks for `@reduce` inside other expressions, and processes this first. 
It isn't smart enough to infer the order of the un-summed indices, so you must tell it, 
although you need not name the intermediate array. 
For example, this is `Σᵢ Aᵢ log(Σⱼ Aⱼ exp(Bᵢⱼ))` with `caribou[i]` the result of `Σⱼ`:

```julia
@pretty @reduce sum(i) A[i] * log( @reduce [i] := sum(j) A[j] * exp(B[i,j]) )
# begin
#     local kangaroo = begin
#         local turtle = orient(A, (*, :))
#         caribou = dropdims(sum((*).(turtle, exp.(B)), dims=2), dims=2)
#     end
#     mallard = sum((*).(A, log.(kangaroo)))
# end
```

### Matrix multiplication

The macro `@mul` expects exactly two tensors, nothing else.
There is an implicit sum over indices repeated on the right: 

```julia
@mul T[i,_,j] := U[i,k,k′] * V[(k,k′),j]    # matrix multiplication, summing over (k,k′)
```

But this still has some errors (especially with `Vector * Matrix` cases). 
`TensorCast.batchmul(B,C)` is a naiive implementation of batched matrix multiplication:

```julia
@pretty @mul A[n][i,k] := B[i,j,n] * C[k,j,n]
# begin
#     local donkey = permutedims(C, (2, 1, 3))
#     A = sliceview(batchmul(B, donkey), (:, :, *))
# end
```

### Anonymous functions

Also a little experimental, you can make functions with `=>`, like this:

```julia
@pretty @cast A[i,j] + 3 + B[j,j]^2 => Z[i,j]  nolazy
# (A, B) -> begin
#     local herring = orient(diag(B), (*, :))
#     Z = (+).(A, 3, (^).(herring, 2))
# end
```

## About

First uploaded January 2019 as `TensorSlice.jl`. 

