
<a href="https://travis-ci.org/mcabbott/TensorSlice.jl"><img src="https://travis-ci.org/mcabbott/TensorSlice.jl.svg?branch=master" align="right" alt="Build Status" padding="20"></a>

# TensorSlice.jl

The package which slices and dices, squeezes and splices!

It provides two macros: `@shape` performs reshaping and slicing of tensors,
and `@reduce` takes sums (or other reductions) over some directions:

```julia
@shape A[j][i] := B[i,j]            # slice a matrix
@shape A[i,(j,k)] := B[i,j,k]       # reshape 3-tensor to a matrix

@reduce A[i] := sum(j,k) B[i,j,k]   # sum over dims=(2,3)
```

These are intended to complement the macro from [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl),
which instead performs Einstein-convention contractions and traces:

```julia
@tensor A[i,k] := B[i,j] * C[j,k]   # matrix multiplication, A = B * C
@tensor A[i] := B[i,k,k]            # partial trace, Aᵢ = Σⱼ Bᵢⱼⱼ
```

More general operations can also be performed with the macro from [Einsum.jl](https://github.com/ahwillia/Einsum.jl),
which sums over all indices appearing only on the right, and allows arbitrary functions:

```julia
@einsum S[i] := - P[i,n] * log( P[i,n] )  # sum over n, for each i
```

These three packages work in very different ways. The original `@einsum` simply writes the nested loops,
`@tensor` performs a series of optimised contraction and trace operations,
while `@shape` and `@reduce` instead write basic Julia commands.
This means that they will work equally well on [Flux](https://github.com/FluxML/Flux.jl)'s TrackedArrays, 
on the GPU via [CuArrays](https://github.com/JuliaGPU/CuArrays.jl),
and probably on almost any other kind of N-dimensional array. 

This package is roughly the equivalent of [`einops`](https://github.com/arogozhnikov/einops) in Python,
while TensorOperations is more like [`opt_einsum`](https://github.com/dgasmith/opt_einsum).

## Examples

This simply slices a matrix into its rows, and re-glues and re-slices to obtain the columns instead:

```julia
mat = (1:4)' .+ rand(2,4)

@shape rows[r][c] := mat[r,c]
@shape cols[♜][🚣] := rows[🚣][♜]

@reduce sum_r[c] := sum(r) mat[r,c]

sum_r == sum(rows) # true
```

This reshapes a matrix into a 3-tensor. The ranges of `i` and `k` would be ambiguous unless you specify 
(at least) one of them, which is written like so:

```julia
M = rand(2*5, 3)

@shape A[i,j,k] := M[(i,k),j]  i:2, k:5
size(A) == (2,3,5)

@shape A[i,j,k] = M[(i,k),j]; # in-place version unambiguous, knows size(A)
```

This glues and reshapes a list of images into one large image:

<img src="test/famous-digits.png?raw=true" width="336" height="168" align="right" alt="MNIST" padding="20">

```julia
using Flux, ImageView, FileIO, JuliennedArrays
imgs = Flux.Data.MNIST.images()[1:32] # vector of matrices

@shape G[(i,I), (j,J)] := imgs[(I,J)][i,j] J:8
@shape G[ i\I,   j\J ] := imgs[ I\J ][i,j] J:8 # identical

imshow(G) # grid with eight columns, 1 ≤ J ≤ 8

save("famous-digits.png", G)
```

Note that the order here `(i,I) = (pixel, grid)` is a choice made by this package,
such that `A[(i,j),k]` and `B[i,j,k]` have the same linear order `A[:] == B[:]`.
And entries `i` and `i+1` are neighbours because Julia `Array`s are column-major 
(the opposite of C, and hence of NumPy).
The alternative notation `(i,I) == i\I` used here is meant to help me remember which is the large-grid index.
(The vector of matrices `C[k]{i,j}` also has the same order, if the slices are StaticArrays, below.)

This defines a function which extends [`kron(A,B)`](https://docs.julialang.org/en/latest/stdlib/LinearAlgebra/#Base.kron) one step beyond vectors & matrices: 

```julia
using TensorOperations, TensorSlice

function tensorkron(A::Array{T,3}, B::Array{T,3}) where {T}
    @tensor C[i,I, j,J, k,K] := A[I,J,K] * B[i,j,k]   # no indices are summed over
    @shape  D[i\I, j\J, k\K] == C[i,I, j,J, k,K]      # @shape with == demands a view
end

A = rand(-20:20, 2,3,1); B = ones(Int, 5,7,1);  # test with 3rd index trivial

D = tensorkron(A, B) 
size(D) == (2*5, 3*7, 1*1)

kron(A[:,:,1], B[:,:,1])  # matrix, same values
```

While *tensor* is often just a fancy word for *N-dimensional array*, it has more specific meanings, 
and one of them is that the the tensor product of two vector spaces `V⊗V` is the one with the product of 
their dimensions (as opposed to `V⊕V = V×V` which has the sum). The Kronecker product 
`kron` maps to such a tensor product space (as `vcat` maps into the direct sum `V⊕V`). 
We can always think of these combined indices `(i,I) = i\I` in this way.
<!--- in fact perhaps `i⊗I` ought to be another accepted notation. --->

This does max-pooling on the above image grid `G`: 

<img src="test/famous-digits-2.png?raw=true" width="224" height="112" align="right" alt="MNIST" padding="20">

```julia
@reduce H[a, b] := maximum(α,β)  G[α\a, β\b]  α:2,β:2
size(G) == 2 .* size(H)

@reduce H4[a, b] := maximum(α:4,β:4)  G[α\a, β\b]
size(G) == 4 .* size(H4)   # ↑ also notice ranges

imshow(H); imshow(H4)
```

In words: consider a horizontal line of pixels in `G` and re-arrange them into two rows, 
so that each column contains two formerly-neighbouring pixes. The horizontal position is now `a`, 
vertical is `α ∈ 1:2`. Take the maximum along these new columns, giving us one line again (half as long). 
Do this to every line, and also to every vertical line, to obtain `H`. 

<!---
Notice that ranges `α:2, β:2` can be specified inside the reduction function, instead of at the end. 
--->

This takes a 2D slice `W[2,:,4,:]` of a 4D array, transposes it, and then forms it into a 4D array
with two trivial dimensions -- such output can be useful for interacting with broadcasting:

```julia
W = rand(2,3,5,7);

@shape Z[_,i,_,k] := W[2,k,4,i]  # equivalent to Z[1,i,1,k] on left

size(Z) == (1,7,1,3)
```

## Inside

To inspect what this package produces, there is a third macro `@pretty` which works like this:

```julia
@pretty @shape A[(i,j)] = B[i,j]
# copyto!(A, B)

@pretty @shape A[k][i,j] := B[i,(j,k)]  k:length(C)
# begin
#     local caterpillar = (size(B, 1), :, length(C))  # your animal may vary
#     A = sliceview(reshape(B, (caterpillar...,)), (:, :, *))
# end
```

Here `TensorSlice.sliceview(D, (:,:,*)) = collect(eachslice(D, dims=3))`
using the new  [eachcol & eachrow](https://github.com/JuliaLang/julia/blob/master/HISTORY.md#new-library-functions) functions,
but allowing more general `sliceview(D, (:,*,:,*) ≈ eachslice(D, dims=(2,4))`. 
(In notation borrowed from [JuliennedArrays.jl](https://github.com/bramtayl/JuliennedArrays.jl), see below.)

Adding `assert` or just `!` inserts explicit size checks:

```julia
@pretty @reduce H[a, b] := maximum(α:2,β:2) G[α\a, β\b] !
# begin
#     @assert rem(size(G, 1), 2) == 0 
#     @assert rem(size(G, 2), 2) == 0
#     local fox = (2, 2, size(G, 1) ÷ 2, size(G, 2) ÷ 2)
#     H = dropdims(maximum(
#         permutedims(reshape(G, (fox[1], fox[3], fox[2], fox[4])), (1, 3, 2, 4)), 
#             dims=(3, 4)), dims=(3, 4))
# end
```

This is really just a variant of the built-in `@macroexpand1`, with animal names from
[MacroTools.jl](https://github.com/MikeInnes/MacroTools.jl) in place of generated symbols, 
and some tidying up.

## Notation

The complete list: 

* `A[i][j,k]` indicates a vector of matrices, like `[rand(2,2) for i=1:3]`.

* `A[(i,j),k]` is a matrix whose first axis (the bracket) is indexed by `n = i + (j-1) * N` where `i ∈ 1:N`.
  You can write this more compactly as `A[i\j, k]`.

* `A[...] := B[...]` creates a new object, like `A = f(B)`,
  while `A[...] = B[...]` writes into an existing object, like `A .= f.(B)`.
  (The same notation is used by [`@einsum`](https://github.com/ahwillia/Einsum.jl#basics) and [`@tensor`](http://jutho.github.io/TensorOperations.jl/latest/indexnotation/).)

* `A[...] := sum(i,j) B[...]` applies `sum(B, dims=...)` over the indicated dimensions.
  The in-place version `A[...] = sum(...) B[...]` is instead `sum!(A, B)`.
  This works for any function `f` which takes the keyword `dims`; the in-place form assumes that `f!` exists.  

* `... = A[i, ...]  k:5` indicates that `k` runs over 5 values: `size(A, 1) == 5`.
  It is only necessary to specify this when ranges would otherwise be ambiguous.
  You can also write `prod(i, k:5)` with `@reduce`. 

* `A[i,-j]` means just `reverse(A, dims=2)`.

* All indices must appear both on the left (or inside a reduction function) and on the right.  

* Inserting a `1` in the output `A[i,1,j] := ...` ensures `size(A) = (N, 1, M)`. 
  You may also use an underscore. Constants indices in the input can be more general, 
  `... := B[i,2,j,4]` means `view(B, :,2,:,4)`.  <!-- $c ... 
  An underscore in the input instead simply asserts (or assumes) that the size of this dimension is `1`. -->
  (Note that numbers [mean something else entirely](http://jutho.github.io/TensorOperations.jl/latest/indexnotation/) in `@tensor`.)

* These operations can be combined, with some limits: you can only reshape the outer container
  of slices `A[(i,j),k][l,m]` (whether we are creating them, on the left, or gluing the given slices, on the right),
  and only the un-reduced indices `A[i,(j,k)] = prod(l,m) = ...`.

* For `@shape` operations, writing `==` demands that the object created is a view of the original data. 
  If this cannot be done, the macro will give an error.
  And using `|=` demands that it is a copy instead, if necessary literally `copy(B)`.
  Slices, `reshape`ing, and `transpose` are typically views,
  while gluing and `permutedims` usually require a copy. Using `:=` expresses indifference.

* The output array need not have a name: `A = @shape [i,j] := ...` is fine, except for in-place operations `=`.

* Appending `assert` or `!` will check all sizes, 
  and appending `base` or `_` will disable the use of other packages (see below). 
  These go with the dimensions, like `A[i,j]  i:2, _, !`.

* `A[...]{j,k}` indicates slicing into a `SMatrix`-es, discussed below. 
  Dimensions must always be given to create such slices, `j:2,k:2`.

## ¬ Base

Various clever packages will be used if you load them, but are not required. 

First, [JuliennedArrays.jl](https://github.com/bramtayl/JuliennedArrays.jl) gives fast slices,
and is also able to re-assemble more complicated slices than something like `reduce(cat,...)` can handle.
There is no downside to this, perhaps it should be the default:

```julia
using JuliennedArrays

@shape S[i][j] == M[i,j]       # S = julienne(M, (*,:)) creates views, S[i] == M[i,:]
@shape Z[i,j] := S[i][j]       # Z = align(S, (*,:)) makes a copy
@shape A[i,j,k,l] := B[k,l][i,j]  # error without JuliennedArrays
```

Second, [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) allows us to reinterpret a Matrix as a Vector of SVectors,
and so on, which is fast for small slices of fixed size.
This is only possible if the first indices of the array are the indices of the slice (such that they share a memory layout),
and if dimensions of the slice are known to the macro (by annotations like `i:2, j:3` again).
By slight abuse of notation, such slices are written as curly brackets:

```julia
using StaticArrays
M = rand(1:99, 2,3)

@shape S[k]{i} == M[i,k]  i:2  # S = reinterpret(SVector{2,Int}, vec(M)) needs the 2

@shape N[k,i] == S[k]{i}       # such slices can be reinterpreted back again

M[1,2]=42; N[2,1]==42          # all views of the original data
```

If they aren't literal integers, such sizes ought to be fixed by the types.
For example, this function is about 100x slower if not given the
[value type](https://docs.julialang.org/en/latest/manual/types/index.html#"Value-types"-1) `Val(2)`,
as the size of the SVector is then not known to the compiler:

```julia
cols_slow(M::Matrix) where N = @shape A[j]{i} == M[i,j] i:size(M,1)

cols_fast(M::Matrix, ::Val{N}) where N = @shape A[j]{i} == M[i,j] i:N

@code_warntype cols_slow(M) # with complaints ::Any in red
@code_warntype cols_fast(M, Val(2))
```

Third, [Strided.jl](https://github.com/Jutho/Strided.jl) contains (among other things) faster methods for permuting dimensions,
especially for large arrays (as used by [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)).
Loading it will cause `@shape` to call these methods.

```julia
using Strided
A = rand(50,50,50,50); p = (4,3,2,1);
B = permutedims(A, p); @strided permutedims(A, p); @strided permutedims!(B, A, p); # compile

@time C = permutedims(A, (4,3,2,1));       # 130 ms,  47 MB
@time @strided permutedims(A, (4,3,2,1));  # 0.02 ms, 400 bytes, lazy

@time @shape D[i,j,k,l] |= A[l,k,j,i];     # 140 ms,  47 MB,     copy
@time @shape E[i,j,k,l] == A[l,k,j,i];     # 0.02 ms, 256 bytes, view -- same as @tensor
@time @shape B[i,j,k,l] =  A[l,k,j,i];     # 15 ms,   4 KB,  in-place
```

<!--
There is a slight potential for bugs here, in that `@strided permutedims` usually creates a view not a copy, and `@shape` knows this.
But this is true only for `A::DenseArray`, and on more exotic arrays it will fall back, in which case `==` may give a copy without warning
(as the macro cannot see the type of the array). -->

## Wishlist

* More torture-tests. This is very new, and probably has many bugs.

* Intermediate optimisations: sometimes this produces `permutedims` before a reduction `sum(..., dims=...)`
  or slicing, which could be removed by changing `dims`. 
  (E.g. `@pretty @shape A[(i,j)] := B[i][j]` could be just `vec(glue(B, (*, :)))`.)

* Better writing into sliced arrays? Right now `@shape A[i]{j} = B[i,j] j:3` is allowed, 
  but `A[i][j]` with ordinary sub-arrays is not.

* More compact notation? This gets messy with many indices, perahps something closer to [`einops`](https://github.com/arogozhnikov/einops)'s notation could be used
  without having to parse strings (try this with `macro arrow(exs...) @show(exs); nothing end `):
```julia
Y = @arrow [X]  i j k l -n  =>  (i,k) (j,l) n   [i:2, j:3]

Z = @arrow [Y / sum, i:2, j:3]  i\k  j\l n  =>  k l n
```

* Support for mutating operators `+=` and `*=` etc. like [@einsum](https://github.com/ahwillia/Einsum.jl) does. Should be fairly easy.

* Ability to write shifts in this notation:
```julia
@shape A[i,j] = B[i+1,j+3]     # circshift!(A, B, (1,3)) or perhaps (-1,-3)
```
<!--<img src="as-seen-on-tv.png?raw=true" width="167" height="130" align="right" alt="As Seen On TV!" padding="20">-->

* A mullet as awesome as [Rich DuLaney](https://www.youtube.com/watch?v=Ohidv69WfNQ) had.


## About

You need [Julia](https://julialang.org/downloads/) 1.0. This package is not yet registered, install like this: 

```julia
pkg> add https://github.com/mcabbott/TensorSlice.jl # press ] for pkg, backspace to leave
                                                    # these image packages will take a while:
pkg> add StaticArrays JuliennedArrays Strided  TensorOperations Einsum  Flux ImageView FileIO

julia> using TensorSlice
```

First uploaded January 2019.

<!--
[![Build Status](https://travis-ci.org/mcabbott/TensorSlice.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorSlice.jl)

<img src="https://raw.githubusercontent.com/mcabbott/TensorSlice.jl/master/as-seen-on-TV.png" width="50" height="40" align="right"><img src = "https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667" align="right">
-->

<!-- pandoc -s -o README.html  README.md -->
