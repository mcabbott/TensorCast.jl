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

## String macros

These provide an alternative... but don't cover quite everything:

```@docs
@cast_str
```

```@docs
@reduce_str
```

# Functions

These are not exported, but are called by the macros above, 
and visible in what `@pretty` prints out. 

```@docs
TensorCast.rview
```

```@docs
TensorCast.diagview
```

```@docs
TensorCast.sliceview
```
