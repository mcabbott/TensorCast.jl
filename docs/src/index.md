# [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl)

This package lets you write complicated formulae in index notation,
which are turned into Julia's usual broadcasting, permuting, slicing, and reducing operations.
It does little you couldn't do yourself, but provides a notation in which it is often 
easier to confirm that you are doing what you intend.

Source, issues, etc: [github.com/mcabbott/TensorCast.jl](https://github.com/mcabbott/TensorCast.jl)

## Changes

Version 0.2 was a re-write, see the [release notes](https://github.com/mcabbott/TensorCast.jl/releases/tag/v0.2.0) to know what changed.

Version 0.4 has significant changes:
- Broadcasting options and index ranges are now written `@cast @avx A[i,j] := B[i⊗j] (i ∈ 1:3)` instead of `@cast A[i,j] := B[i⊗j] i:3, axv` (using [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) for the broadcast, and supplying the range of `i`).
- To return an array without naming it, write an underscore `@cast _[i] := ...` rather than omitting it entirely.
- Some fairly obscure features have been removed for simplicity: Indexing by an array `@cast A[i,k] := B[i,J[k]]` and by a range `@cast C[i] := f(D[1:3, i])` will no longer work.
- Some dimension checks are inserted by default; previously the option `assert` did this.
- It uses [LazyStack.jl](https://github.com/mcabbott/LazyStack.jl) to combine handles slices, simplifying earlier code. This is lazier by default, write `@cast A[i,k] := log(B[k][i]) lazy=false` (with a new keyword option) to glue into an `Array` before broadcasting.
- It uses [TransmuteDims.jl](https://github.com/mcabbott/TransmuteDims.jl) to handle all permutations & many reshapes. This is lazier by default -- the earlier code sometimes copied to avoid reshaping a `PermutedDimsArray`. This isn't always faster, though, and can be disabled by `lazy=false`.

New features in 0.4:
- Indices can appear ouside of indexing: `@cast A[i,j] = i+j` translates to `A .= axes(A,1) .+ axes(A,2)'`
- The ternary operator `? :` can appear on the right, and will be broadcast correctly.
- All operations should now support [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl).
- You can `repeat` by broadcasting over indices not appearing on the right, such as `@cast r[i,(k,j)] = m[i,j]`

## Pages

1. Use of `@cast` for broadcasting, dealing with arrays of arrays, and generalising `mapslices`
2. `@reduce` and `@matmul`, for taking the sum (or the `maximum`, etc) over some dimensions
3. Options: broadcasting with Strided.jl, LoopVectorization.jl, LazyArrays.jl, and slicing with StaticArrays.jl
4. Docstrings, which list the complete set of possibilities.
