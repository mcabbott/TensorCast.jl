
# TensorCast.jl

[![Build Status](https://travis-ci.org/mcabbott/TensorCast.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorCast.jl)
<!--
<a href="https://travis-ci.org/mcabbott/TensorCast.jl"><img src="https://travis-ci.org/mcabbott/TensorCast.jl.svg?branch=master" align="right" alt="Build Status" padding="20"></a>-->

This package lets you write many expressions involving N-dimensional arrays in index notation,
which is often the least confusing way. 
It defines a pair of macros: `@cast` deals both with "casting" into new shapes (including going 
to and from an array-of-arrays) and with broadcasting:

```julia
@cast A[row][col] := B[row, col]            # slice a matrix into its rows

@cast C[(i,j), (k,ℓ)] := D[i,j,k,ℓ]         # reshape a 4-tensor to give a matrix

@cast E[x,y] = F[x]^2 * log(G[y])           # broadcast E .= F.^2 .* log.(G') into existing E
```

And `@reduce` takes sums (or other reductions) over some directions, 
but otherwise understands all the same things: 

```julia
@reduce H[a] := sum(b,c) L[a,b,c]                # sum over dims=(2,3), and dropdims

@reduce S[i] = sum(n) -P[i,n] * log(P[i,n]/Q[n]) # sum!(S, @. -P*log(P/Q')) into exising S

@reduce W[μ,ν,J] := prod(i:2) V[(i,J)][μ,ν]      # products of pairs of matrices, stacked
```

<!--
Finally `@mul` handles certain matrix multiplications:

```julia
@mul T[i,_,j] := U[i,k,k′] * V[(k,k′),j]         # matrix multiplication, summing over (k,k′)

@mul W[i,j,β] := X[i,k,β] * Y[k,i,β]             # batched W[:,:,β] = X[:,:,β] * Y[:,:,β] ∀ β
```
-->

These are intended to complement the macro from [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl),
which instead performs Einstein-convention contractions and traces, in a very similar notation. 
Here it is implicit that repeated indices are summed over: 

```julia
@tensor A[i] := B[i,j] * C[j,k] * D[k]      # matrix multiplication, A = B * C * D
@tensor D[i] := E[i] + F[i,k,k]             # partial trace of F only, Dᵢ = Eᵢ + Σⱼ Fᵢⱼⱼ
```

Similar notation is also used by the macro from [Einsum.jl](https://github.com/ahwillia/Einsum.jl),
where again it is implicit that all indices appearing only on the right are summed over. 
This allows arbitrary (element-wise) functions:

```julia
@einsum S[i] := -P[i,n] * log(P[i,n]/Q[n])  # sum over n, for each i (also with @reduce above)
@einsum G[i] := E[i] + F[i,k,k]             # the sum includes everyting:  Gᵢ = Σⱼ (Eᵢ + Fᵢⱼⱼ)
```

There is some overlap of operations which can be done with two (or all three) of these packages. 
However they produce very different code for actually doing what you request. 
The original `@einsum` simply writes the necessary set of nested loops. 
Instead `@tensor` works out a sequence of contraction and trace operations, calling optimised BLAS routines where possible. 
<!-- (And [ArrayMeta.jl](https://github.com/shashi/ArrayMeta.jl) aimed to do a wide variety of operations efficiently, but seems to be abandonned.) -->

The  macros from this package aim to produce simple Julia commands: 
often just a string of `reshape` and `permutedims` and `eachslice` and so on,
plus a native [broadcasting expression](https://julialang.org/blog/2017/01/moredots) if needed, 
and `sum` or  `sum!`.
This means that they are very generic, and will (mostly) work well on 
[Flux](https://github.com/FluxML/Flux.jl)'s TrackedArrays, on the GPU via 
[CuArrays](https://github.com/JuliaGPU/CuArrays.jl),
and on almost any other kind of N-dimensional array.

For those who speak Python, `@cast` and `@reduce` allow similar operations to 
[`einops`](https://github.com/arogozhnikov/einops) (minus the cool video, but plus broadcasting)
while Einsum / TensorOperations map very roughly to [`einsum`](http://numpy-discussion.10968.n7.nabble.com/einsum-td11810.html) 
/ [`opt_einsum`](https://github.com/dgasmith/opt_einsum).
The function of `@check!` (see [below](#checking)) is similar to [`tsalib`](https://github.com/ofnote/tsalib)'s shape annotations.

## Installation

You need [Julia](https://julialang.org/downloads/) 1.0 or later. This package is now registered, install it like this: 

```julia
pkg> add TensorCast  # press ] for pkg, backspace to leave

pkg> add StaticArrays Strided  TensorOperations Einsum  # optional extras, see below
pkg> add Flux ImageView FileIO                          # for image examples

julia> using TensorCast

help?> @cast  # press ? for help
```

From a Jupyter notebook, write instead `using Pkg; pkg"add TensorCast"`.
If you downloaded this under its former name, you should `rm TensorSlice`.  

## Examples

This simply slices a matrix into its rows, then re-glues and re-slices to obtain the columns instead:

```julia
mat = (1:4)' .+ rand(2,4)

@cast rows[r][c] := mat[r,c]
@cast cols[♜][🚣] := rows[🚣][♜]

@reduce sum_r[c] := sum(r) mat[r,c]  

size(sum_r) == (4,) # @reduce gives a vector, not a 1×4 matrix
sum_r == sum(rows)  # true
```

Notice that the same indices must always appear both on the right and on the left
(unless they are explicitly reduced over). Indices may not be repeated (except on different tensors). 

This reshapes a matrix into a 3-tensor. The ranges of `i` and `k` would be ambiguous unless you specify 
(at least) one of them. Such ranges are written `i:2`, and appear as part of a tuple of options after the expression:

```julia
M = randn(Float16, 2*5, 3)

@cast A[i,j,k] := M[(i,k),j]  i:2, k:5

size(A) == (2,3,5) # true

@cast A[i,j,k] = M[(i,k),j]; # writing into existing A, it knows size(A)
```

This glues and reshapes a list of images into one large image:

<img src="test/famous-digits.png?raw=true" width="336" height="168" align="right" alt="MNIST" padding="20">

```julia
using Flux, ImageView, FileIO
imgs = Flux.Data.MNIST.images()[1:32] # vector of matrices

@cast G[(i,I), (j,J)] := imgs[(I,J)][i,j] J:8
@cast G[ i\I,   j\J ] := imgs[ I\J ][i,j] J:8 # identical

imshow(G) # grid with eight columns, 1 ≤ J ≤ 8

save("famous-digits.png", G)
```

Note that the order here `(i,I) = (pixel, grid)` is a choice made by this package,
such that `A[(i,j),k]` and `B[i,j,k]` have the same linear order `A[:] == B[:]`.
And entries `i` and `i+1` are neighbours because Julia `Array`s are column-major 
(the opposite of C, and hence of NumPy). The alternative notation `(i,I) == i\I` used here 
is meant to help me remember which is the large-grid index.
(The vector of matrices `C[k]{i,j}` also has the same order, if the slices are StaticArrays, below.)

This defines a function which extends [`kron(A,B)`](https://docs.julialang.org/en/latest/stdlib/LinearAlgebra/#Base.kron) one step beyond vectors & matrices: 

```julia
function Base.kron(A::Array{T,3}, B::Array{T,3}) where {T}
    @cast D[i\I, j\J, k\K] := A[I,J,K] * B[i,j,k]
end

A = rand(-20:20, 2,3,1)   # test with 3rd index trivial
B = ones(Int, 5,7,1);

D = kron(A, B)            # calls this new method
size(D) == (2*5, 3*7, 1*1)

kron(A[:,:,1], B[:,:,1])  # calls built-in method, same numbers
```

While *tensor* is often just a fancy word for *N-dimensional array*, it has more specific meanings, 
and one of them is that the the tensor product of two vector spaces `V ⊗ V` is the one with the product of 
their dimensions (as opposed to `V × V` which has the sum). The Kronecker product 
`kron` maps to such a tensor product space (as `vcat` maps into `V × V`). 
We can always think of these combined indices `(i,I) = i\I` in this way, and now you may write `i⊗I` too. 

This does max-pooling on the above image grid `G`: 

<img src="test/famous-digits-2.png?raw=true" width="224" height="112" align="right" alt="MNIST" padding="20">

```julia
@reduce H[a, b] := maximum(α:2,β:2)  G[α\a, β\b]  

size(G) == 2 .* size(H) # true
imshow(H)
```

In words: take a horizontal line of pixels in `G` and re-arrange them into two rows, 
so that each column contains two formerly-neighbouring pixes. The horizontal position is now `a`, 
vertical is `α ∈ 1:2`. Take the maximum along these new columns, giving us one line again (half as long). 
Do this to every line, and also to every vertical line, to obtain `H`. 

Notice also that ranges `α:2, β:2` can be specified inside the reduction function, instead of at the end. 

This takes a 2D slice `W[2,:,4,:]` of a 4D array, transposes it, and then forms it into a 4D array
with two trivial dimensions -- such output can be useful for interacting with broadcasting:

```julia
W = randn(2,3,5,7);

@cast Z[_,i,_,k] := W[2,k,4,i]  # equivalent to Z[1,i,1,k] on left

size(Z) == (1,7,1,3)
```

Finally, you can also create anonymous functions using `->` or `=>` 
(the only distinction is that `(A + B) -> ...` needs a bracket)
and writing the output on the right: 

```julia
f = @cast A[i,j,k] -> X[i\j,_,k]      # A -> reshape(A, :,1,size(A,3))

g = @cast B[b] + log(C[c]) => Y[b,c]  # (B,C) -> B .+ log.(C')

size(f(rand(5,5,9))) == (25,1,9)
size(g(rand(2), rand(3))) == (2,3)
```

## Inside

To inspect what this package produces, there is another macro `@pretty` which works like this:

```julia
@pretty @cast A[(i,j)] = B[i,j]
# copyto!(A, B)

@pretty @cast A[k][i,j] := B[i,(j,k)]  k:length(C)
# begin
#     local caterpillar = (size(B, 1), :, length(C))  # your animal may vary
#     A = sliceview(reshape(B, (caterpillar...,)), (:, :, *))
# end
```

Here `TensorCast.sliceview(D, (:,:,*)) = collect(eachslice(D, dims=3))` using the new
[eachcol & eachrow](https://github.com/JuliaLang/julia/blob/master/HISTORY.md#new-library-functions) functions,
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

This `@pretty` is really just a variant of the built-in `@macroexpand1`, with animal names from
[MacroTools.jl](https://github.com/MikeInnes/MacroTools.jl) in place of generated symbols, 
and some tidying up.

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
@cast! D[α,_,β,_] := C[α,β]         # reshape to size(D,2) == size(D,4) == 1
E = calculate(D)
@check! E[n,α]                      # just the check, with no calculation
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

As mentioned above, expressions with `=` write into an existing array, 
while those with `:=` do not. This is the same notation as 
[TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [Einsum.jl](https://github.com/ahwillia/Einsum.jl). 
But unlike those packages, sometimes the result of `@cast` is a view of the original, for instance 
`@cast A[i,j] := B[j,i]` gives `A = transpose(B)`. You can forbid this, and insist on a copy, 
by writing `|=` instead. And conversely, if you expect a view, writing `==` will give an error if not.

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
@cast A[i,k] := S[k][i]  cat        # A = hcat(B...)
@cast A[i,k] := S[k][i]  lazy       # A = VectorOfArrays(B)

size(A) == (3, 4) # true
```

The option `lazy` uses [RecursiveArrayTools.jl](https://github.com/JuliaDiffEq/RecursiveArrayTools.jl)
to create a view of the original vectors. This would also be possible with 
[JuliennedArrays.jl](https://github.com/bramtayl/JuliennedArrays.jl), I may change what gets used later. 

Combining with `cat` is often much slower, but more generic. For example it will work with 
[Flux](https://github.com/FluxML/Flux.jl)'s TrackedArrays.

Another kind of slices are provided by [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl),
in which a Vector of SVectors is just a different interpretation of the same memory as a Matrix. 
By another slight abuse of notation, such slices are written here as curly brackets:

```julia
using StaticArrays

@cast S[k]{i} == M[i,k]  i:3        # S = reinterpret(SVector{3,Int}, vec(M)) 

@cast R[k,i] == S[k]{i}             # such slices can be reinterpreted back again

M[1,2]=42; R[2,1]==42               # all views of the original data
```

When creating such slices, their size ought to be provided, either as a literal integer or 
through the types. Note that you may also write `S[k]{i:3}`. 
For example, this function is about 100x slower if not given the
[value type](https://docs.julialang.org/en/latest/manual/types/index.html#"Value-types"-1) `Val(3)`,
as the size of the SVector is then not known to the compiler:

```julia
cols_slow(M::Matrix) = @cast A[j]{i} == M[i,j]  i:size(M,1)

cols_fast(M::Matrix, ::Val{N}) where N = @cast A[j]{i:N} == M[i,j]

@code_warntype cols_slow(M) # with complaints ::Any in red
@code_warntype cols_fast(M, Val(3))
```

Another potential issue is that, if you create such slices after transposing (or some other lazy transformation), 
then accessing them tends to be slower. Making a copy with `|=` such as `@cast T[i]{k:4} |= M[i,k]` will 
avoid this.


### Better broadcasting

When broadcasting and then summing over some directions, it can be faster to avoid creating the 
entire array, then throwing it away. This can be done with the package 
[LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl) which has a lazy `BroadcastArray`. 
In the following example, the product `V .* V' .* V3` contains about 1GB of data, 
the writing of which is avoided by giving the option `lazy`: 

```julia
using LazyArrays
V = rand(500);
V3 = reshape(V, 1,1,:);

@time sum(V .* V' .* V3; dims=(2,3));                 # 0.6 seconds, 950 MB
@time @reduce W[i] := sum(j,k) V[i] * V[j] * V[k];    # about the same

@time sum(BroadcastArray(*, V, V', V3); dims=(2,3));  # 0.025 s, 5 KB
@time @reduce W[i] := sum(j,k) V[i]*V[j]*V[k]  lazy;  # about the same 
```

Finally, the package [Strided.jl](https://github.com/Jutho/Strided.jl) can apply multi-threading to 
broadcasting, and some other magic. You can enable it with the option `strided`, like this: 

```julia
using Strided # and export JULIA_NUM_THREADS = 4 before starting
A = randn(4000,4000); 
B = similar(A);
Threads.nthreads() == 4 # true

@time B .= (A .+ A') ./ 2;                            # 0.12 seconds
@time @cast B[i,j] = (A[i,j] + A[j,i])/2;             # the same 

@time @strided B .= (A .+ A') ./ 2;                   # 0.025 seconds
@time @cast B[i,j] = (A[i,j] + A[j,i])/2 strided;     # the same
```

<!--
## Wishlist

* More torture-tests. This is very new, and probably has many bugs.

* Better writing into sliced arrays? Right now `@cast A[i]{j} = B[i,j] j:3` is allowed, 
  but `A[i][j]` with ordinary sub-arrays is not.

* More compact notation? This gets messy with many indices, perahps something closer to [`einops`](https://github.com/arogozhnikov/einops)'s notation could be used
  without having to parse strings (try this with `macro arrow(exs...) @show(exs); nothing end `):
```julia
Y = @arrow [X]  i j k l -n  =>  (i,k) (j,l) n   [i:2, j:3]

Z = @arrow [Y / sum, i:2, j:3]  i\k  j\l n  =>  k l n
```

* Support for mutating operators `+=` and `*=` etc. like [@einsum](https://github.com/ahwillia/Einsum.jl) does. 

* Ability to write shifts in this notation:
```julia
@cast A[i,j] = B[i+1,j+3]     # circshift!(A, B, (1,3)) or perhaps (-1,-3)
```

* A mullet as awesome as [Rich DuLaney](https://www.youtube.com/watch?v=Ohidv69WfNQ) had.
-->

## About

First uploaded January 2019 as `TensorSlice.jl` with only the `@shape` macro, and later `@reduce`. 

Then I understood how to implement arbitrary broadcasting in `@cast`, 
and this replaced the earlier implementation. 

<!--
### See also

* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [Einsum.jl](https://github.com/ahwillia/Einsum.jl) 

* 

-->
<!--
[![Build Status](https://travis-ci.org/mcabbott/TensorCast.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorCast.jl)

<img src="https://raw.githubusercontent.com/mcabbott/TensorCast.jl/master/as-seen-on-TV.png" width="50" height="40" align="right"><img src = "https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667" align="right">
-->

<!-- pandoc -s -o README.html  README.md -->
