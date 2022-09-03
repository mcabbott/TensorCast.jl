
# time julia -e 'using TensorCast; TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  ))'
# 10.1 sec without this, 7.2 sec with (on 2nd run)

#=
# August 2022, Julia 1.9 master

me@ArmBook TensorCast % time julia -e '@time (using TensorCast; TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  )))'
  3.874656 seconds (6.59 M allocations: 474.681 MiB, 1.79% gc time, 76.88% compilation time: <1% of which was recompilation)
julia -e   5.39s user 1.77s system 134% cpu 5.330 total

# With SnoopPrecompile

me@ArmBook TensorCast % time julia -e '@time (using TensorCast; TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  )))'
  2.948221 seconds (2.71 M allocations: 194.022 MiB, 68.51% compilation time: <1% of which was recompilation)
julia -e   4.50s user 1.74s system 141% cpu 4.408 total

=#

using SnoopPrecompile
@precompile_all_calls begin

_macro(:(  Z[i,j] := A[i] + B[j]  ))
_macro(:(  Z[i,_,k'] := A[i] + B[k'] / log(2) ))
_macro(:(  Z[(i,j)] := A[j,-i] ))
_macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2 ))
_macro(:(  S[i,j] = sum(k)  ),:(  A[i,j] + B[j,k]  ), call=CallInfo(:reduce))
_macro(:(  S[i,j] := sum(k)  ),:(  (B[j]+ C[k])[i]  ), call=CallInfo(:reduce))

pretty(:(  (B[j]+ C[k])[i] ))
pretty(@macroexpand @cast A[i,j] := B[j,i] + 1 )

end
