
# TensorCast.jl

[![Stable Docs](https://img.shields.io/badge/docs-juliahub-blue.svg)](https://juliahub.com/docs/TensorCast/)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg?logo=github)](https://mcabbott.github.io/TensorCast.jl/dev)
[![Build Status](https://github.com/mcabbott/TensorCast.jl/workflows/CI/badge.svg)](https://github.com/mcabbott/TensorCast.jl/actions)

This package lets you work with many-dimensional arrays in index notation, 
by defining a few macros. The first is `@cast`, which deals both with "casting" into 
new shapes (including going to and from an array-of-arrays) and with broadcasting:

```julia
@cast A[row][col] := B[row, col]        # slice a matrix B into its rows, also @cast A[r] := B[r,:]

@cast C[(i,j), (k,ℓ)] := D.x[i,j,k,ℓ]   # reshape a 4-tensor D.x to give a matrix

@cast E[φ,γ] = F[φ]^2 * exp(G[γ])       # broadcast E .= F.^2 .* exp.(G') into existing E

@cast T[x,y,n] := outer(M[:,n])[x,y]    # generalised mapslices, vector -> matrix function
```

Second, `@reduce` takes sums (or other reductions) over the indicated directions. Among such sums is 
matrix multiplication, which can be done more efficiently using `@matmul` instead:

```julia
@reduce K[_,b] := prod(a,c) L.field[a,b,c]           # product over dims=(1,3), and drop dims=3

@reduce S[i] = sum(n) -P[i,n] * log(P[i,n]/Q[n])     # sum!(S, @. -P*log(P/Q')) into exising S

@matmul M[i,j] := sum(k,k′) U[i,k,k′] * V[(k,k′),j]  # matrix multiplication, plus reshape
```

All of these are converted into simple Julia array commands like `reshape` and `permutedims` 
and `eachslice`, plus a [broadcasting expression](https://julialang.org/blog/2017/01/moredots) if needed, 
and `sum` /  `sum!`, or `*` / `mul!`. This means that they are very generic, and will (mostly) work well 
with [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl), on the GPU via 
[CuArrays](https://github.com/JuliaGPU/CuArrays.jl), and with almost anything else. 
For operations with arrays of arrays like `mapslices`, this package defines gradients for 
[Zygote.jl](https://github.com/FluxML/Zygote.jl) (similar to those of [SliceMap.jl](https://github.com/mcabbott/SliceMap.jl)).

Similar notation is used by some other packages, although all of them use an implicit sum over 
repeated indices. [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) performs 
Einstein-convention contractions and traces:

```julia
@tensor A[i] := B[i,j] * C[j,k] * D[k]      # matrix multiplication, A = B * C * D
@tensor D[i] := 2 * E[i] + F[i,k,k]         # partial trace of F only, Dᵢ = 2Eᵢ + Σⱼ Fᵢⱼⱼ
```

More general contractions are allowed by 
[OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl), but only one term:
```julia
@ein W[i,j,k] := X[i,ξ] * Y[j,ξ] * Z[k,ξ]   # star contraction
W = ein" iξ,jξ,kξ -> ijk "(X,Y,Z)           # numpy-style notation
```

Instead [Einsum.jl](https://github.com/ahwillia/Einsum.jl) and [Tullio.jl](https://github.com/mcabbott/Tullio.jl) 
sum the entire right hand side, but also allow arbitrary (element-wise) functions:

```julia
@einsum S[i] := -P[i,n] * log(P[i,n]/Q[n])  # sum over n, for each i (also with @reduce above)
@einsum G[i] := 2 * E[i] + F[i,k,k]         # the sum includes everyting:  Gᵢ = Σⱼ (2Eᵢ + Fᵢⱼⱼ)
@tullio Z[i,j] := abs(A[i+x, j+y] * K[x,y]) # convolution, summing over x and y
```

These produce very different code for actually doing what you request:
The macros `@tensor` and `@ein` work out a sequence of basic operations (like contraction and traces),
while `@einsum` and `@tullio` simply write the necessary set of nested loops.

For those who speak Python, `@cast` and `@reduce` allow similar operations to 
[`einops`](https://github.com/arogozhnikov/einops) (minus the cool video, but plus broadcasting)
while Einsum / TensorOperations map very roughly to [`einsum`](http://numpy-discussion.10968.n7.nabble.com/einsum-td11810.html) 
/ [`opt_einsum`](https://github.com/dgasmith/opt_einsum).

## Installation

You need [Julia](https://julialang.org/downloads/) 1.0 or later:

```julia
] add TensorCast
```

## About

This was a holiday project to learn a bit of metaprogramming, originally `TensorSlice.jl`. 
But it suffered a little scope creep. 

