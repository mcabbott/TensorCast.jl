# Macros

```@meta
CurrentModule = TensorCast
```

```@docs
@cast(ex)
```

```@docs
@reduce(ex)
```

```@docs
@matmul(ex)
```

```@docs
@pretty(ex)
```

# Functions

These are not exported, but are called by the macros above, 
and visible in what `@pretty` prints out. 

```@docs
TensorCast.diagview
```

```@docs
TensorCast.sliceview
```

These are from helper packages:

```@docs
LazyStack.stack
```

```@docs
TransmuteDims.TransmutedDimsArray
```

```@docs
TransmuteDims.transmute
```

```@docs
TransmuteDims.transmutedims
```
