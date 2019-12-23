
# TensorCast.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://pkg.julialang.org/docs/TensorCast/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mcabbott.github.io/TensorCast.jl/dev)
[![Build Status](https://travis-ci.org/mcabbott/TensorCast.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorCast.jl)

This package lets you write expressions involving many-dimensional arrays in index notation, 
by defining a few macros. The first is `@cast`, which deals both with "casting" into new shapes 
(including going to and from an array-of-arrays) and with broadcasting:

```julia
@cast A[row][col] := B[row, col]        # slice a matrix B into its rows, also @cast A[r] := B[r,:]

@cast C[(i,j), (k,ℓ)] := D[i,j,k,ℓ]     # reshape a 4-tensor D to give a matrix

@cast E[φ,γ] = F[φ]^2 * exp(G[γ])       # broadcast E .= F.^2 .* exp.(G') into existing E

@cast T[x,y,n] := outer(M[:,n])[x,y]    # generalised mapslices, vector -> matrix function
```

Next, `@reduce` takes sums (or other reductions) over the indicated directions, 
but otherwise understands all the same things. Among such sums is matrix multiplication,
which can be performed much more efficiently by using `@matmul` instead:

```julia
@reduce K[_,b] := prod(a,c) L[a,b,c]                 # product over dims=(1,3), and drop dims=3

@reduce S[i] = sum(n) -P[i,n] * log(P[i,n]/Q[n])     # sum!(S, @. -P*log(P/Q')) into exising S

@matmul M[i,j] := sum(k,k′) U[i,k,k′] * V[(k,k′),j]  # matrix multiplication, plus reshape

@reduce W[μ,ν,_,J] := maximum(i:4) X[(i,J)][μ,ν]     # elementwise maxima across sets of 4 matrices
```

The main goal is to allow you to write complicated expressions very close to how you would 
on paper (or in LaTeX), and avoiding having to write elaborate comments that 
"dimensions 4,5 of this tensor are μ,ν" etc. 

The notation used is very similar to that of some existing packages, 
although all of them use an implicit sum over repeated indices. 
[TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) 
performs Einstein-convention contractions and traces:

```julia
@tensor A[i] := B[i,j] * C[j,k] * D[k]      # matrix multiplication, A = B * C * D
@tensor D[i] := 2 * E[i] + F[i,k,k]         # partial trace of F only, Dᵢ = 2Eᵢ + Σⱼ Fᵢⱼⱼ
```

More general contractions are allowed by the new 
[OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl), but only one term:
```julia
@ein W[i,j,k] := X[i,ξ] * Y[j,ξ] * Z[k,ξ]   # star contraction
W = ein" iξ,jξ,kξ -> ijk "(X,Y,Z)           # numpy-style notation
```

Instead [Einsum.jl](https://github.com/ahwillia/Einsum.jl) sums the entire right hand side,
but also allows arbitrary (element-wise) functions:

```julia
@einsum S[i] := -P[i,n] * log(P[i,n]/Q[n])  # sum over n, for each i (also with @reduce above)
@einsum G[i] := 2 * E[i] + F[i,k,k]         # the sum includes everyting:  Gᵢ = Σⱼ (2Eᵢ + Fᵢⱼⱼ)
```

There is some overlap of operations which can be done with several of these packages, 
but they produce very different code for actually doing what you request. 
The original `@einsum` simply writes the necessary set of nested loops, 
while `@tensor` and `@ein` work out a sequence of basic operations (like contraction and traces).

The macros from this package aim instead to produce simple Julia array commands: 
often just a string of `reshape` and `permutedims` and `eachslice` and so on,
plus a native [broadcasting expression](https://julialang.org/blog/2017/01/moredots) if needed, 
and `sum` /  `sum!`, or `*` / `mul!`. 
This means that they are very generic, and will (mostly) work well 
with small [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl), 
with [Flux](https://github.com/FluxML/Flux.jl)'s TrackedArrays, 
on the GPU via [CuArrays](https://github.com/JuliaGPU/CuArrays.jl),
and on almost anything else. 

These commands are usually what you would write anyway, with zero runtime penalty. 
However some operations can sometime be very slow -- for instance using `@reduce` for matrix
multiplication will broadcast out a complete 3-tensor, while `@matmul` calls `*` instead.
And some operations can be much faster, particularly when replacing `mapslices` with 
explicit slicing.

For those who speak Python, `@cast` and `@reduce` allow similar operations to 
[`einops`](https://github.com/arogozhnikov/einops) (minus the cool video, but plus broadcasting)
while Einsum / TensorOperations map very roughly to [`einsum`](http://numpy-discussion.10968.n7.nabble.com/einsum-td11810.html) 
/ [`opt_einsum`](https://github.com/dgasmith/opt_einsum).

## Installation

You need [Julia](https://julialang.org/downloads/) 1.0 or later:

```julia
] add TensorCast
```

The registered version is 0.1.5, for which you want the 
[stable](https://pkg.julialang.org/docs/) docs above. 
Version 0.2.0 from  `] add TensorCast#two` is a re-write with some new features, 
described below. <!--, with [new docs](https://mcabbott.github.io/TensorCast.jl/dev). 

There are also some notebooks: [docs/einops.ipynb](docs/einops.ipynb) explaining with images,
and [docs/speed.ipynb](docs/speed.ipynb) explaining what's fast and what's slow.
-->

## What's new

Version 0.2 has substantially re-worked logic. Maybe new bugs too.

Added:

* Slicing can be written `A[i,:]`, which allows for generalised mapslices operations, 
  such as `@cast V[i,k] := real(eigen(T[:,:,k]).values[i])`. This is done by two broadcasting
  operations, the first of which includes `getproperty(...,:values)` here,
  the second applies `real(...)`.

* Arrays can be indexed by other arrays, for instance `A[i, B[j,k]]` is the 
  3-tensor `A[:,B]`, where `eltype(B)==Int`.

* An array of functions can be applied to other arrays, for instance 
  `@cast A[i,j,k] := F[i](X[j], y, Z[k])`.

* Updating an array can be written `@cast A[i] += f(B[i])` or similarly `*=` or `-=`.

* You can shuffle along a direction by writing `A[i,~j]`, 
  in addition to reversing with `A[i,-j]`.

* Inner indices can now be fixed, for instance `A[i][3,k]` takes the 3rd row 
  of each element of `A`, which is a vector of matrices.

* Prime `'` now means `adjoint`, which is complex conjugation when applied to 
  numbners `A[i,j,k]'`, but conjugate-transpose when applied to matrices `A[:,:,k]'`.
  Applied to indices, `A[i']` is normalised to the unicode \\prime `A[i′]`.

* Slicing into StaticArrays can be written `A{:,j}`, always the leftmost indices. 
  The fact that such slices have (say) `Size(3)` can be provided by writing `A{:3, j}`. 
  This allows static-mapslices operations: `@cast C[i,j] |= fun(A{:3,j}){i}`. 

* Lazy broadcasting `@cast A[i] := f(B[i]) lazy` will return the [BroadcastArray](https://github.com/JuliaArrays/LazyArrays.jl#broadcasting).

* Zygote gradient definitions for slicing/glueing from [SliceMap.jl](https://github.com/mcabbott/SliceMap.jl)
  have moved here. Thus mapslices-like operations `@cast A[i,j] := f(B[:,j])[i]` should be differentiable.

Removed:

* `@mul` replaced by `@matmul`, which for now requires you to explicitly write what 
  indices are summed. And no longer supports a batch index, sorry.

* The proofreading / named-tensor functions of `@check!`, `@cast!` etc. will move to
  another package.

* The sign `==` as in `@cast A[i] == B[i]` no longer works. 
  Using `:=` still returns a view of `B` when it can do so efficiently, 
  and `|=` still insists on `collect`ing this. 

* Anonymous functions `@cast f(A[i]) => B[i]` are currently don't work, 
  but may be revived in more limited form.  

* You cannot reverse indices on the left `A[i,-j] := ...` when making a new array `A`.

* Staic slicing `A[j]{i}` no longer requires you to load `StaticArrays`, this removes 
  all optional dependencies. 


## About

This was a holiday project to learn a bit of metaprogramming, originally `TensorSlice.jl`. 
But it suffered a little scope creep. 

