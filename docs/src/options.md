# Options

Expressions with `=` write into an existing array, 
while those with `:=` do not. This is the same notation as 
[TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [Einsum.jl](https://github.com/ahwillia/Einsum.jl). 
But unlike those packages, sometimes the result of `@cast` is a view of the original, for instance 
`@cast A[i,j] := B[j,i]` gives `A = transpose(B)`. You can forbid this, and insist on a copy, 
by writing `|=` instead (or passing the option `collect`). 

Various other options can be given after the main expression. 

## Ways of slicing

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
@cast A[i,k] := S[k][i]  lazy       # A = LazyStack.stack(B)

size(A) == (3, 4) # true
```

The option `lazy` uses [LazyStack.jl](https://github.com/mcabbott/LazyStack.jl)
to create a view of the original vectors. 

Another kind of slices are provided by [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl),
in which a Vector of SVectors is just a different interpretation of the same memory as a Matrix. 
By another slight abuse of notation, such slices are written here as curly brackets:

```julia
@cast S[k]{i} := M[i,k]  i in 1:3   # S = reinterpret(SVector{3,Int}, vec(M)) 
@cast S[k] := M{:3, k}              # equivalent

@cast R[k,i] := S[k]{i}             # such slices can be reinterpreted back again
```

Both `S` and `R` here are views of the original matrix `M`. 
When creating such slices, their size ought to be provided, either as a literal integer or 
through the types. Note that you may also write `S[k]{i:3} = ...`. 

The second notation (with `M{:,k}`) is useful for mapslices. Continuing the example: 

```julia
M10 = rand(10,1000); 
mapslices(cumsum, M10, dims=1)          # 630 μs using @btime
@cast [i,j] := cumsum(M10[:,j])[i]      #  64 μs
@cast [i,j] := cumsum(M10{:10,j})[i]    #  38 μs
```

## Better broadcasting

The package [Strided.jl](https://github.com/Jutho/Strided.jl) can apply multi-threading to 
broadcasting, and some other magic. You can enable it like this: 

```julia
using Strided # and export JULIA_NUM_THREADS = 4 before starting
A = randn(4000,4000); B = similar(A);

@time @cast B[i,j] = (A[i,j] + A[j,i])/2;             # 0.12 seconds
@time @cast @strided B[i,j] = (A[i,j] + A[j,i])/2;    # 0.025 seconds
```

The package [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) provides 
a macro `@avx` which modifies broadcasting to use vectorised instructions. 
This can be used as follows:

```julia
using LoopVectorization, BenchmarkTools
C = randn(40,40); 

D = @btime @cast [i,j] := exp($C[i,j]);         # 13 μs
D′ = @btime @cast @avx [i,j] := exp($C[i,j]);   #  3 μs
D ≈ D′
```

When broadcasting and then summing over some directions, it can be faster to avoid creating the 
entire array, then throwing it away. This can be done with the package 
[LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl) which has a lazy `BroadcastArray`. 
In the following example, the product `V .* V' .* V3` contains about 1GB of data, 
the writing of which is avoided by giving the option `lazy`: 

```julia
using LazyArrays
V = rand(500); V3 = reshape(V,1,1,:);

@time @reduce W[i] := sum(j,k) V[i]*V[j]*V[k];        # 0.6 seconds, 950 MB
@time @reduce @lazy W[i] := sum(j,k) V[i]*V[j]*V[k];  # 0.025 s, 5 KB
```

However, right now this gives `3.7 s (250 M allocations, 9 GB)`, something is broken!

## Less lazy

To disable the default use of `PermutedDimsArray` etc, give the option `nolazy`: 

```julia
@pretty @cast Z[y,x] := M[x,-y]  nolazy
# Z = transmutedims(reverse(M, dims = 2), (2, 1))

@pretty @cast Z[y,x] := M[x,-y] 
# Z = transmute(Reverse{2}(M), (2, 1))
```

This also controls how the extraction of diagonal elements
and creation of diagonal matrices are done:

```julia
@pretty @cast M[i,i] := A[i,i]  nolazy
# M = diagm(0 => diag(A))

@pretty @cast D[i,i] := A[i,i]
# D = Diagonal(diagview(A))

@pretty @cast M[i,i] = A[i,i]  nolazy  # into diagonal of existing matrix M
# diagview(M) .= diag(A); M
```

Here `TensorCast.Reverse{1}(B)` creates a view with `reverse(axes(B,1))`, 
and `TensorCast.diagview(A)` is just `view(A, diagind(A))`.
`TransmuteDims.transmute(M)` is `transpose(M)` on a matrix of numbers.

## Gradients

I moved some code here from [SliceMap.jl](https://github.com/mcabbott/SliceMap.jl),
which I should describe!
