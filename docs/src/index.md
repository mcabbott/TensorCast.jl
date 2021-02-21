# [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl)

This package lets you write complicated formulae in index notation,
which are turned into Julia's usual broadcasting, permuting, slicing, and reducing operations.
It does little you couldn't do yourself, but provides a notation in which it is often 
easier to confirm that you are doing what you intend.

Version 0.2 was a re-write, see the [release notes](https://github.com/mcabbott/TensorCast.jl/releases/tag/v0.2.0) to know what changed from version 0.1.5.

Version 0.4 has significant changes:
- It uses [TransmuteDims.jl](https://github.com/mcabbott/TransmuteDims.jl) to handle all permutations & many reshapes. This is lazier by default, which isn't always faster: The earlier code avoids reshaping a `PermutedDimsArray` by copying it. 
- It uses [LazyStack.jl](https://github.com/mcabbott/LazyStack.jl) to combine handles slices, simplifying earlier code.
- It inserts some dimension checks by default, previously the option `assert` did this.

Source, issues, etc: [github.com/mcabbott/TensorCast.jl](https://github.com/mcabbott/TensorCast.jl)

## Documentation

The pages are:

1. Use of `@cast` for broadcasting, and slicing
2. `@reduce` and `@matmul`, for summing over some directions
3. Options: StaticArrays, LazyArrays, Strided
4. Docstrings, for all details.

