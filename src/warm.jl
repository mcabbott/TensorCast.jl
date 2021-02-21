
# time julia -e 'using TensorCast; TensorCast._macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2  ))'
# 10.1 sec without this, 7.2 sec with (on 2nd run)

_macro(:(  Z[i,j] := A[i] + B[j]  ))
_macro(:(  Z[i,_,k'] := A[i] + B[k'] / log(2) ))
_macro(:(  Z[(i,j)] := A[j,-i] ))
_macro(:(  Z[i,k][j] := fun(A[i,:], B[j])[k] + C[k]^2 ))
_macro(:(  S[i,j] = sum(k)  ),:(  A[i,j] + B[j,k]  ), call=CallInfo(:reduce))
_macro(:(  S[i,j] := sum(k)  ),:(  (B[j]+ C[k])[i]  ), call=CallInfo(:reduce))

pretty(:(  (B[j]+ C[k])[i] ))
pretty(@macroexpand @cast A[i,j] := B[j,i] + 1 )
