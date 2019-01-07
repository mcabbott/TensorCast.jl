
export @check, check!, @einsum!, @tensor!

"""
	@check!(A[i, j, μ, ν])

Adds `A` to the store of known tensors, and records that it expects indices `i,j,μ,ν`.
If it is already in the store, then instead this checks whether the present indices differ 
from the saved ones. This happens while parsing your source code, there is zero run-time penalty. 

In addition, it can insert size checks to be performed at run-time. 
At the first occurrange these save `length(i:) == size(A,1)`, and on subsequent uses of 
the same index (even on different tensors) give an error if the sizes do not match. 
If turned on, this will need to look up indices in a dictionary, which takes about ??μs.

Returns either `A` or `check!(A, (:i,:j))`. 

	@check! B[i,j] C[μ,ν]

Checks several tensors, returns nothing. 

	@check!  alpha=true  tol=3  size=false  throw=false  info  empty

Controls options for `@check` and related macros, these are the defaults:
* `alpha=true` turns on the parse-time checking, based on index letters
* `tol=3` sets how close together letters must be: `B[j,k]` is not an error but `B[a,b]` will be
* `size=false` turns off run-time size checking
* `throw=false` means that errors are given using `@error`, without interrupting your program
* `empty` deletes all saved letters and sizes -- there is one global store for each, for now
* `info` prints what's currently saved.
"""
macro check!(exs...)
	check_macro(exs...)
end

const index_store = Dict{Symbol, Tuple}()
const size_store  = Dict{Symbol, Int}()

mutable struct CheckOptions
	alpha::Bool
	tol::Int
	size::Bool
	throw::Bool
end

const check_options = CheckOptions(true, 3, false, false)

function check_macro(exs...)
	for ex in exs
		if @capture(ex, A_[vec__])
			if length(exs)==1
				return check_one(ex)
			else
				check_one(ex)
			end

		elseif @capture(ex, alpha=val_Bool)
			check_options.alpha = val
		elseif @capture(ex, tol=val_Int)
			check_options.tol = val

		elseif @capture(ex, size=val_Bool)
			check_options.size = val

		elseif @capture(ex, throw=val_Bool)
			check_options.throw = val

		elseif ex == :empty
			empty!(index_store)
			empty!(size_store)
			@info "@check! stores emptied"

		elseif ex == :info
			@info "@check! info" check_options index_store size_store

		else 
			println(ex, "  ",typeof(ex))
			@error "@check! doesn't know what to do with $ex"

		end
	end
    return nothing
end

function check_one(ex)
	@capture(ex, A_[vec__]) || error("@check! can't understand $ex")
	ind = Tuple(vec)

	if check_options.alpha
		got = get!(index_store, A, ind)

		if length(ind) > length(got)
			check_err("@check! $ex now has more indices than previous $got")
		elseif length(ind) < length(got)
			check_err("@check! $ex now has fewer indices than previous $got")
		else
			for (i, j) in zip(ind, got)
				isa(i,Int) && continue
				si = String(i)
				sj = String(j)
				length(si)>1 || length(sj)>1 && continue	
				if abs(Int(si[1])-Int(sj[1])) > check_options.tol
					check_err("@check! $ex now has index $i where previously it had $j")
				end
			end
		end
	end

	if check_options.size
		Astring = string(A,"[",join(ind,","),"]")
		return :(check!($A, $Astring, $ind))
	else
		return A
	end
end

"""
	check!(A, (:i,:j))
Performs run-time size checking, on behalf of the `@check!` macro, returns `A`. 
"""
function check!(A::AbstractArray, str::String, ind::Tuple)
	sizeA = size(A)
	for (d,i) in enumerate(ind)
		got = get!(size_store, i, sizeA[d])
		got == sizeA[d] || check_err("@check! $str, index $i now has length $(sizeA[d]) instead of $got")
	end
	A
end

function check_err(str::String)
	if check_options.throw
		error(str)
	else
		@error str
	end
end

"""
	@einsum! A[i,j] := B[i,k] * C[k,j]

Variant of `@einsum` from package Einsum.jl, 
equivalent to wrapping every tensor with `@check!()`.
"""
macro einsum!(ex)
	check_einsum(ex)
end

"""
	@tensor! A[i,j] := B[i,k] * C[k,j]

Variant of `@tensor` from package TensorOperations.jl, 
equivalent to wrapping every tensor with `@check!()`.
"""
macro tensor!(ex)
	check_tensor(ex)
end

function check_tensor(ex)
	if @capture(ex, lhs_ := *(rhs__)) || @capture(ex, lhs_ = *(rhs__)) 
		outex = quote end

		push!(outex.args, check_one(lhs))
		for tensor in rhs
			push!(outex.args, check_one(tensor))
		end
		
		if check_options.size
			push!(outex.args, :(@tensor $ex) )
			return outex
		end

	else
		@warn "@tensor! not smart enough to process $ex yet, so ignoring it"
	end
	return :(@tensor $ex)
end

function check_einsum(ex)
	if @capture(ex, lhs_ := *(rhs__)) || @capture(ex, lhs_ = *(rhs__)) 
		outex = quote end

		push!(outex.args, check_one(lhs))
		for tensor in rhs
			push!(outex.args, check_one(tensor))
		end

		if check_options.size
			push!(outex.args, Einsum._einsum(ex))
			return outex
		end

	else
		@warn "@einsum! not smart enough to process $ex yet, so ignoring it"
	end
	println(ex)
	return Einsum._einsum(ex)
end

#==

using MacroTools, Einsum, TensorOperations
B = rand(2,3); C = rand(3,2);
A = B * C
@einsum A[i,j] := B[i,k] * C[k,j]
@tensor A[i,j] := B[i,k] * C[k,j]

@einsum A[i,j] := B[i,k] * C[k,zz] # not an error, WTF?




@check! size=true throw=false info

@einsum! A[i,j] := B[i,k] * C[k,j]

@tensor! A[i,j] := B[i,k] * C[k,j]

@check! A[z,j] # compains about z
@check! B[i]   # complains about number

B5 = rand(2,5); C5 = rand(5,2);
@einsum! A[i,j] := B5[i,k] * C5[k,j] # complains about sizes
@tensor! A[i,j] := B5[i,k] * C5[k,j]

@einsum! A[i,j] := B5[i,k] * C5[k,zz] # complains about zz
@tensor! A[i,j] := B5[i,k] * C5[k,zz]


==#

