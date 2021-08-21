# Reductions

```jldoctest mylabel
julia> using TensorCast

julia> M = collect(reshape(1:12, 3,4));
```

This is the basic syntax to sum over one index:

```jldoctest mylabel; filter = r"begin\n.*\nend"
julia> @reduce S[i] := sum(j) M[i,j] + 1000
3-element Array{Int64,1}:
 4022
 4026
 4030

julia> @pretty @reduce S[i] := sum(j) M[i,j] + 1000
begin
    ndims(M) == 2 || throw(ArgumentError("expected a 2-tensor M[i, j]"))
    S = dropdims(sum(@__dot__(M + 1000), dims = 2), dims = 2)
end
```

Note that:
* You must always specify the index to be summed over. 
* The sum applies to the whole right side (including the 1000 here). 
* And the summed dimensions are always dropped (unless you explicitly say `S[i,_] := ...`).

## Not just `sum`

You may use any reduction funciton which understands keyword `dims=...`, like `sum` does. 
For example:

```jldoctest mylabel
julia> using Statistics

julia> @reduce A[j] := mean(i) M[i,j]^2
4-element Array{Float64,1}:
   4.666666666666667
  25.666666666666668
  64.66666666666667
 121.66666666666667
```

If writing into an existing array, then the function must work like `sum!` does:

```julia-repl
julia> @pretty @reduce A[j] = mean(i) M[i,j]^2
begin
    local pelican = transmute(M, (2, 1))  # pelican = transpose(M)
    mean!(A, @__dot__(pelican ^ 2))
end
```

Otherwise all the same notation as `@cast` works. 
Here's a max-pooling trick with combined indices, in which we group the numbers in `R` into tiplets
and keep the maximum of each set -- equivalent to the maximum of each column below.  

```jldoctest mylabel
julia> R = collect(0:5:149);

julia> @reduce Rmax[_,c] := maximum(r) R[(r,c)]  r in 1:3
1×10 Array{Int64,2}:
 10  25  40  55  70  85  100  115  130  145

julia> reshape(R, 3,:)
3×10 Array{Int64,2}:
  0  15  30  45  60  75   90  105  120  135
  5  20  35  50  65  80   95  110  125  140
 10  25  40  55  70  85  100  115  130  145
```

Here `r in 1:3` indicates the range of `r`, implying `c ∈ 1:10`.
You may also write `maximum(r:3) R[(r,c)]`.

In the same way, this down-samples an image by a factor of 2, 
taking the mean of each 2×2 block of pixels, in each colour `c`:

```julia-repl
julia> @reduce smaller[I,J,c] := mean(i:2, j:2) orig[(i,I), (j,J), c]
```

## Scalar output

Here are different ways to write complete reduction:

```jldoctest mylabel
julia> @reduce Z := sum(i,j) M[i,j]      # Z = sum(M)
78

julia> Z = @reduce sum(i,j) M[i,j]
78

julia> @reduce Z0[] := sum(i,j) M[i,j]   # Z0 = dropdims(sum(M, dims=(1, 2)), dims=(1, 2))
0-dimensional Array{Int64,0}:
78

julia> @reduce Z1[_] := sum(i,j) M[i,j]
1-element Array{Int64,1}:
 78
```

## Recursion

If you want to sum only part of an expression, or have one sum inside another, 
then you can place a complete `@reduce` expression inside `@cast` or `@reduce`.
There is no need to name the intermediate array, here `termite[x]`, but you must show its indices:

```julia-repl
julia> @pretty @reduce sum(x,θ) L[x,θ] * p[θ] * log(L[x,θ] / @reduce _[x] := sum(θ′) L[x,θ′] * p[θ′])
begin
    ndims(L) == 2 || error()  # etc, some checks
    local goshawk = transmute(p, (nothing, 1))
    sandpiper = dropdims(sum(@__dot__(L * goshawk), dims = 2), dims = 2)  # inner sum
    bison = sum(@__dot__(L * goshawk * log(L / sandpiper)))
end
```

Notice that the inner sum here is a matrix-vector product, so it will be more efficient to 
write `@matmul _[x] := sum(θ') L[x,θ′] * p[θ′]` to call `L * p` instead of broadcasting. 

## Matrix multiplication

It's possible to multiply matrices using `@reduce`, 
but this is exceptionally inefficient, as it first broadcasts out a 3-tensor 
before summing over one index:

```julia-repl
julia> @pretty @reduce R[i,k] := sum(j) M[i,j] * N[j,k]
begin
    size(M, 2) == size(N, 1) || error()  # etc, some checks
    local fish = transmute(N, (nothing, 1, 2))   # fish = reshape(N, 1, size(N)...)
    R = dropdims(sum(@__dot__(M * fish), dims = 2), dims = 2)
end
```

The macro `@matmul` has the same syntax, but instead calls `A*B`. 

```jldoctest mylabel; filter = r"[0-9\.]+ .s \(.*\)"
julia> A = rand(100,100); B = rand(100,100);

julia> red(A,B) = @reduce R[i,k] := sum(j) A[i,j] * B[j,k];

julia> mul(A,B) = @matmul P[i,k] := sum(j) A[i,j] * B[j,k];

julia> using BenchmarkTools

julia> @btime red($A, $B);
  1.471 ms (25 allocations: 7.71 MiB)

julia> @btime mul($A, $B);
  33.489 μs (2 allocations: 78.20 KiB)
```

Of course you may just write `A * B` yourself, but in general `@matmul` will handle the same 
reshaping, transposing, fixed indices, etc. steps as `@reduce` does. 
Once all of that is done, however, the result must be a product like `A * B` in which the indices
being summed over appear on both tensors. 

To use the more flexible `@reduce` in cases like this, 
when creating the large intermediate tensor will be expensive,
the option `@lazy` which creates a `LazyArrays.BroadcastArray` instead can help: 

```julia-repl
julia> using LazyArrays

julia> lazyred(A,B) = @reduce @lazy R[i,k] := sum(j) A[i,j] * B[j,k];

julia> @btime lazyred($A, $B);
  223.172 μs (26 allocations: 78.95 KiB)  # when things worked well
  10.186 ms (23 allocations: 78.77 KiB)   # today

julia> red(A, B) ≈ mul(A, B) ≈ lazyred(A,B)
true
```

More details on the next page. 

## No reduction

The syntax of `@reduce` can also be used with `@cast`, to apply functions which 
take a `dims` keyword, but do not produce trivial dimensions. 

```jldoctest mylabel
julia> normalise(p; dims) = p ./ sum(p; dims=dims);

julia> @cast N[x,y] := normalise(x) M[x,y]
3×4 Array{Float64,2}:
 0.166667  0.266667  0.291667  0.30303
 0.333333  0.333333  0.333333  0.333333
 0.5       0.4       0.375     0.363636
```
