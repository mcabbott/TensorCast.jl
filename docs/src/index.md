# TensorCast.jl

This package lets you write complicated formulae in index notation,
which are turned into Julia's usual broadcasting, permuting, slicing, and reducing operations. 

## Documentation

The pages are:

1. Use of `@cast` for broadcasting, and slicing
2. `@reduce` and `@matmul`, for summing over some directions
3. Options: StaticArrays, LazyArrays, Strided
4. Docstrings, for all details.

These refer to v0.2, which is currently `#master`, 
see the [readme](https://github.com/mcabbott/TensorCast.jl) for a list of what's changed.

What will show up on [pkg.julialang.org](https://pkg.julialang.org/docs/TensorCast/) once this works? Not sure.
