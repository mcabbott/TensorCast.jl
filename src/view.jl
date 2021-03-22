
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

Like `*` but intended for multiplying sizes, and understands that `:` is a wildcard.
"""
star(x,y) = *(x,y)
star(::Colon,y) = Colon()
star(x,::Colon) = Colon()
star(x,y,zs...) = star(star(x,y), zs...)

"""
    onetolength(1:10) == 10

Used to digest options `i in 1:10`, size calculation for reshaping only allows ranges starting at 1 for now.
"""
onetolength(ax::Base.OneTo) = length(ax)
onetolength(ax::AbstractUnitRange) = first(ax) == 1 ? length(ax) : error("ranges have to start at 1, for now")
onetolength(ax) = error("ranges must be AbstractUnitRange")
