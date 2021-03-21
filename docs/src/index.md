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
- It uses [LazyStack.jl](https://github.com/mcabbott/LazyStack.jl) to combine handles slices, simplifying earlier code. This is lazier by default, write `@cast A[i,k] := log(B[k][i]) lazy=false` (with a new keyword option) to glue into an `Array` before broadcasting.
- It uses [TransmuteDims.jl](https://github.com/mcabbott/TransmuteDims.jl) to handle all permutations & many reshapes. This is lazier by default -- the earlier code sometimes copied to avoid reshaping a `PermutedDimsArray`. This isn't always faster, and can be disabled by `lazy=false`.
- It inserts some dimension checks by default, previously the option `assert` did this. (Not yet merged.)

## Pages

1. Use of `@cast` for broadcasting, and slicing
2. `@reduce` and `@matmul`, for summing over some directions
3. Options: StaticArrays, LazyArrays, Strided, LoopVectorization
4. Docstrings, for all details.
