
"""
    diagview(M) = view(M, diagind(M))

Like `diag(M)` but makes a view.
"""
diagview(A::AbstractMatrix) = view(A, diagind(A))

diagview(A::LinearAlgebra.Diagonal) = A.diag

"""
    rvec(x) = [x]
    rvec(A) = vec(A)

Really-vec... extends `LinearAlgebra.vec` to work on scalars too.
"""
rvec(x::Number) = [x]
rvec(A) = LinearAlgebra.vec(A)

"""
    mul!(Z,A,B)

Exactly `LinearAlgebra.mul!` except that it can write into a zero-array.
"""
mul!(Z,A,B) = LinearAlgebra.mul!(Z,A,B)
mul!(Z::AbstractArray{T,0}, A,B) where {T} = copyto!(Z, A * B)

"""
    star(x,y,...)

Used for multiplying axes now, not sizes.
"""
star(x, y) = Base.OneTo(length(x) * length(y))
star(x,y,zs...) = star(star(x,y), zs...)
