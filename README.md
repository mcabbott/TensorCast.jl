# TensorSlice.jl

The one macro which slices and dices, squeezes and glues!

It uses easy notation you already know from [`@tensor`](https://github.com/Jutho/TensorOperations.jl)
and [`@einsum`](https://github.com/ahwillia/Einsum.jl) to express all kinds of reshaping and slicing of tensors, no matter how complicated!
There are three basic ways to use it, rotate the dial to `:=` for a new object, `=` to use a pre-allocated one, and `==` for a view of the old:

```julia
@shape A[(i,j),k] := B[i,j,k]  # new matrix from tensor B
@shape A[i,j,k] = B[(i,k),j]   # write into an existing tensor A
@shape A[(i,j,k)] == B[i,j,k]  # reshaped view A = vec(B)
```

Slicing works the same way:

```julia
@shape A[i,j] := B[i][j]       # hcat a vector of vectors
@shape A[i,j,k] = B[i,k][j]    # write into tensor A
@shape A[i][j] == B[j,i]       # create views A = collect(eachcol(B))
```

Only for `=` is it necessary to name the output. And you can combine these operations, to the following extent:

```julia
A = @shape [(i,j)] := B[j][i]  # vcat a vector of vectors
A = @shape [(i,j),l][k,m] := B[i][j,k,l,m] # glue then slice then reshape
```

Reshaping where ranges of any indices are ambiguous will give an error.
You can provide explicit sizes by writing things like `k:5` after the expression, either to avoid such ambiguities,
or simply to assert that sizes are as you expect.
For example, changing the second example above `A[i,j,k] = B[(i,k),j]` to use `:=` instead
will leave the lengths of `i` and `k` unfixed (as they can no longer be read off from `A`),
unless we specify one of them:

```julia
B = rand(2*5, 3);
@shape A[i,j,k] := B[(i,k),j]  i:2  # could give (i:2, j:3, k:5)
size(A) == (2,3,5)
```


## But wait, there's more!
<!--## `@pretty` -->

Since this is a dinky piece of plastic (costing well under $9.99) you should not be too surprised that,
once the wrapping paper is thrown away, your computer ends up doing the actual work with exactly the same methods it would have used anyway.
To see what's going on, we include (at no extra charge!) another macro which works like this:

```julia
@pretty @shape A[(i,j)] = B[i,j]
# copyto!(A, B)

@pretty @shape A[k][i,j] == B[i,(j,k)]  k:length(C)
# begin
#     local caterpillar = (size(B, 1), :, length(C))  # your animal may vary
#     A = sliceview(reshape(B, (caterpillar...,)), (:, :, *))
# end
```

This is in fact just like the `@macroexpand` you already have, but shinier
(thanks to animal names from [MacroTools](https://github.com/MikeInnes/MacroTools.jl))
and less functional (deleting out line number comments, and most module qualifiers).
Here `TensorSlice.sliceview(D, (:,:,*)) = collect(eachslice(D, dims=3))`
using the new  [eachcol & eachrow](https://github.com/JuliaLang/julia/blob/master/HISTORY.md#new-library-functions) functions,
but allowing also things like `sliceview(D, (:,*,:,*) ≈ eachslice(D, dims=(2,4))`.

Moving on, here's a picture with a celebrity!

```julia
using TestImages, ImageView, FileIO
V = testimage.(["mandril_gray", "cameraman", "lena_gray_512"])

@shape M[i,(j,J)] := V[J][i,j]

imshow(M)

save("monkey-man-lena.jpeg", M)
```

<p align="center">
<img src="monkey-man-lena.jpeg?raw=true" width="600" height="200" alt="Lena, Человек с кино-аппаратом, Mandrillus sphinx" padding="5">
</p>

Or with one more dimension:

<img src="famous-digits.png?raw=true" width="336" height="168" align="right" alt="MNIST" padding="20">

<!--
# @shape mid[IJ,i,j] := imgs[IJ][i,j] # history!!
# @shape A[(i,I),(j,J)] := mid[(I,J),i,j] J:8
-->

```julia
using Flux, ImageView, FileIO, JuliennedArrays
imgs = Flux.Data.MNIST.images()[1:32] # vector of matrices

@shape A[(i,I),(j,J)] := imgs[(I,J)][i,j] J:8 # eight columns

imshow(A)

save("famous-digits.png", A)
```

Note that the order here `(i,I) = (pixel, grid)` is a choice made by this package,
such that `A[(i,j),k]` and `B[i,j,k]` have the same linear order `A[:] == B[:]`,
and entries `i` and `i+1` are neighbours because Julia `Array`s are column-major.
The vector of matrices `C[k][i,j]` also has the same order, if the slices are StaticArrays, below.

## Power Tools Sold Separately
<!-- ## ¬ Base -->

Some of the work will be out-sourced to various clever packages if you load them.

First, [JuliennedArrays](https://github.com/bramtayl/JuliennedArrays.jl) gives fast slices,
and is also able to re-assemble more complicated slices than something like `reduce(cat,...)` can handle.
There is no downside to this, perhaps it should be the default:

```julia
using JuliennedArrays

@shape S[i][j] == M[i,j]       # S = julienne(M, (*,:)) creates views, S[i] == M[i,:]
@shape Z[i,j] := S[i][j]       # Z = align(S, (*,:)) makes a copy
@shape A[i,j,k,l] := B[k,l][i,j]  # error without JuliennedArrays
```

Second, [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) allows us to reinterpret a Matrix as a Vector of SVectors,
and so on, which is fast for small slices of fixed size.
This is only possible if fist indices of the array are the indices of the slice (such that they share a memory layout),
and if dimensions of the slice are known to the macro (by annotations like `i:2, j:3` again).
By slight abuse of notation, such slices are written as curly brackets:

```julia
using StaticArrays
M = rand(Int, 2,3)

@shape S[k]{i} == M[i,k]  i:2  # S = reinterpret(SVector{2,Int}, vec(M)) needs the 2
@shape N[k,i] == S[k]{i}       # such slices can be reinterpreted back again

M[1,2]=42; N[2,1]==42          # all views of the original matrix
```
<!--
The re-gluing doesn't really need to be told the dimensions, but giving them tells `@shape` to try.
Perhaps this should be more explicit... some options:
```julia
@shape N[k,i] == S[k][i]  &    # ! already means check sizes, things like *,-,+,j,s,_ parse fine
@shape N[k,i] == S[k]{i}       # cute to indicate slice size is a type?
@shape S[k]{i} == M[i,k]  i:2  # similarly for slicing
```
-->

Third, [Strided](https://github.com/Jutho/Strided.jl) contains (among other things) faster methods for permuting dimensions,
especially for large arrays, as used by [`TensorOperations`](https://github.com/Jutho/TensorOperations.jl).
Loading it will cause `@shape` to call these methods.

```julia
using Strided
A = rand(50,50,50,50);
B = permutedims(A, (4,3,2,1)); @strided permutedims(A, (4,3,2,1)); @strided permutedims!(B, A, (4,3,2,1)); # compile

@time C = permutedims(A, (4,3,2,1));       # 130 ms,  47 MB
@time @strided permutedims(A, (4,3,2,1));  # 0.02 ms, 400 bytes, lazy

@time @shape D[i,j,k,l] := A[l,k,j,i];     # 140 ms,  47 MB,     copy
@time @shape E[i,j,k,l] == A[l,k,j,i];     # 0.02 ms, 256 bytes, view
@time @shape C[i,j,k,l] = A[l,k,j,i];      # 15 ms,   4 KB,  in-place
```

There is a slight potential for bugs here, in that `@strided permutedims` usually creates a view not a copy, and `@shape` knows this.
But this is true only for `A::DenseArray`, and on more exotic arrays it will fall back, in which case `==` may give a copy without warning
(as the macro cannot see the type of the array).

<!--
Perhaps there should be notation here too.
Maybe `@shape B[i,j,k,l] := A[l,k,j,i] s` or `@shape B[i,j,k,l] := A[l,k,j,i] &` to opt in?
Maybe `@shape B[i,j,k,l] := A[l,k,j,i] _` to opt out, `_ == Base` sort-of?

For comparison, here are single- and multi-threaded nested loops from [Einsum](https://github.com/ahwillia/Einsum.jl) instead:
BUT these are junk, with `@btime` like 90ms... and above 20μs -> 80ns too...

```julia
@time @einsum B[i,j,k,l] = A[l,k,j,i];     # 1.5 seconds, 290 MB
@time @vielsum B[i,j,k,l] = A[l,k,j,i];    # 0.9 seconds, 231 MB
```
-->

## Forthcoming Attractions
<!-- ## Wishlist -->

* More torture-tests. They may still be bugs.

* Perhaps there ought to be an operation in-between `:=` which guarantees a copy, and `==` which guarantees a view,
  just do whatever is easiest. Abuse `<<` or `<<=` to mean this? Or abuse `!=` for guaranteeing a copy & free `:=` to mean whatever is easy?

* Slicing and gluing usually has sorted `code = (:,:,:,*,*)` by default. Sometimes we could remove `permutedims` by altering this,
  so far this happens only for `transpose`, and only for slicing.
  (E.g. `@pretty @shape A[(i,j)] := B[i][j]` could be just `vec(glue(B, (*, :)))`.)

* Now ` @shape A[i]{j} = B[i,j] j;3` is allowed, but in-place writing to (or replacing of) ordinary sub-arrays `A[i][j]` is not.

* Would be nice if `copyto!(A, glue(B, ...))` could be just use `glue!(A, B, ...)`.

Wishlist:

* Support `+=` and `*=` etc. like [Einsum](https://github.com/ahwillia/Einsum.jl) does, should be fairly easy.

* Allow constant indices:
```julia
@shape A[i,j] := B[j,3,i]      # allow constants
@shape A[i,j,$k] = B[j,i]     # ... including k interpolated
```

* Treat reverse, and shifts, in this notation:
```julia
@shape A[i,j] := B[i,-j]       # reverse(B, dims=2)
@shape A[i,j] = B[i+1,j+3]    # circshift!(A, B, (1,3))
```
<!--<img src="as-seen-on-tv.png?raw=true" width="167" height="130" align="right" alt="As Seen On TV!" padding="20">-->

* A mullet as awesome as [Rich DuLaney](https://www.youtube.com/watch?v=Ohidv69WfNQ) had.


<!-- ## About -->
## Call 1-800-WHY-CANT-IT now!

Our minions are standing by for your call! For your convenience, they are located in another time zone
(and are heavy users of google translate) so please open an issue if you have found a way to break your gadget.
We guarantee a 100% refund... and double your money back if you open a pull request!

No need to wait for international shipping:

```julia
pkg> add https://github.com/mcabbott/TensorSlice.jl

pkg> add StaticArrays, JuliennedArrays, Strided

julia> using TensorSlice
```

[![Build Status](https://travis-ci.org/mcabbott/TensorSlice.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorSlice.jl)


<!-- pandoc -s -o README.html  README.md -->

<!--

ANN: TensorSlice.jl

Tens-Or-Slice is a little gadget which aims to make all kinds of slicing, dicing, and squeezing look easy on TV.
For example, this is how you slice a 3-tensor into 3×3 SMatrix pieces
```
@shape B[k]{i,j} == A[i,j,k]  i:3, j:3
```
And this glues them together again, using `reduce(cat,...)` as if they were ordinary matrices,
and then reshapes & transposes to get an N×3 matrix:
```
@shape C[(j,k),i] := B[k][i,j]
```
This macro doesn't really do any of the work, it just calls standard Julia things,
and can be hooked up to StaticArrays, JuliennedArrays, and Strided.

And was largely a holiday project (once the puzzles in advent of code got too long)
to teach myself a little macrology, which suffered some mild scope creep.
But perhaps it will be useful to some people.

-->
