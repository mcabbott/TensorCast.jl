# TensorCast.jl

This package lets you write complicated formulae in index notation,
which are turned into Julia's usual broadcasting, permuting, slicing, and reducing operations. 

Version 0.2 is a re-write, see the [realse notes](https://github.com/mcabbott/TensorCast.jl/releases/tag/v0.2.0) to know what changed from version 0.1.5.

## Documentation

The pages are:

1. Use of `@cast` for broadcasting, and slicing
2. `@reduce` and `@matmul`, for summing over some directions
3. Options: StaticArrays, LazyArrays, Strided
4. Docstrings, for all details.

