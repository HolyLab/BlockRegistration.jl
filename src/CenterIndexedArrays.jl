__precompile__()

# Note: this requires a patch to julia itself. See julia PR??.

module CenterIndexedArrays

import Base: size, eachindex, getindex, setindex!, linearindexing, writemime
import Base: ==, +, -, *, /, .+, .-, .*, ./
import Base: isequal, maximum, minimum, cumsum, permutedims, ipermutedims
import Base: mapslices, flipdim
using Base.Cartesian

export CenterIndexedArray

immutable CenterIndexedArray{T,N,A<:AbstractArray} <: DenseArray{T,N}
    data::A
    halfsize::NTuple{N,Int}

    function CenterIndexedArray(data::A)
        new(data, _halfsize(data))
    end
end

CenterIndexedArray{T,N}(A::AbstractArray{T,N}) = CenterIndexedArray{T,N,typeof(A)}(A)

# This is the AbstractArray default, but do this just to be sure
linearindexing{A<:CenterIndexedArray}(::Type{A}) = Base.LinearSlow()

size(A::CenterIndexedArray) = size(A.data)

@generated function _halfsize{T,N}(A::AbstractArray{T,N})
    args = [:(size(A,$d)>>1) for d = 1:N]
    quote
        @nexprs $N d->(isodd(size(A,d)) || error("Must have all-odd sizes"))
        tuple($(args...))
    end
end

@generated function eachindex{T,N}(::Base.LinearSlow, A::CenterIndexedArray{T,N})
    startargs = [:(-A.halfsize[$i]) for i = 1:N]
    stopargs  = [:( A.halfsize[$i]) for i = 1:N]
    meta = Expr(:meta, :inline)
    :($meta; CartesianRange(CartesianIndex{$N}($(startargs...)), CartesianIndex{$N}($(stopargs...))))
end

@generated function getindex{T,N}(A::CenterIndexedArray{T,N}, i::Integer...)
    length(i) == N || error("Must use $N indexes")
    args = [:(i[$d]+A.halfsize[$d]+1) for  d = 1:N]
    meta = Expr(:meta, :inline)
    :($meta; A.data[$(args...)])
end

@generated function setindex!{T,N}(A::CenterIndexedArray{T,N}, v, i::Integer...)
    length(i) == N || error("Must use $N indexes")
    args = [:(i[$d]+A.halfsize[$d]+1) for  d = 1:N]
    meta = Expr(:meta, :inline)
    :($meta; A.data[$(args...)] = v)
end

(==)(A::CenterIndexedArray, B::CenterIndexedArray) = A.data == B.data
(==)(A::CenterIndexedArray, B::AbstractArray) = A.data == B
(==)(A::AbstractArray, B::CenterIndexedArray) = A == B.data

isequal(A::CenterIndexedArray, B::CenterIndexedArray) = isequal(A.data, B.data)
isequal(A::CenterIndexedArray, B::AbstractArray) = isequal(A.data, B)
isequal(A::AbstractArray, B::CenterIndexedArray) = isequal(A, B.data)

maximum(A::CenterIndexedArray, region) = maximum(A.data, region)
minimum(A::CenterIndexedArray, region) = minimum(A.data, region)

cumsum(A::CenterIndexedArray, region) = cumsum(A.data, region)

 permutedims(A::CenterIndexedArray, perm) = CenterIndexedArray(permutedims(A.data, perm))
ipermutedims(A::CenterIndexedArray, perm) = CenterIndexedArray(ipermutedims(A.data, perm))

mapslices(f, A::CenterIndexedArray, dims::AbstractVector) = mapslices(f, A.data, dims)
flipdim{T}(A::CenterIndexedArray{T,1}, dim::Integer) = CenterIndexedArray(flipdim(A.data, dim))  # ambiguity
flipdim(A::CenterIndexedArray, dim::Integer) = CenterIndexedArray(flipdim(A.data, dim))

(+)(A::CenterIndexedArray{Bool},x::Bool) = CenterIndexedArray(A.data + x)
(+)(x::Bool,A::CenterIndexedArray{Bool}) = CenterIndexedArray(x + A.data)
(-)(A::CenterIndexedArray{Bool},x::Bool) = CenterIndexedArray(A.data - x)
(-)(x::Bool,A::CenterIndexedArray{Bool}) = CenterIndexedArray(x - A.data)
(+)(A::CenterIndexedArray,x::Number) = CenterIndexedArray(A.data + x)
(+)(x::Number,A::CenterIndexedArray) = CenterIndexedArray(x + A.data)
(-)(A::CenterIndexedArray,x::Number) = CenterIndexedArray(A.data - x)
(-)(x::Number,A::CenterIndexedArray) = CenterIndexedArray(x - A.data)
(.+)(A::CenterIndexedArray{Bool},x::Bool) = CenterIndexedArray(A.data .+ x)
(.+)(x::Bool,A::CenterIndexedArray{Bool}) = CenterIndexedArray(x .+ A.data)
(.-)(A::CenterIndexedArray{Bool},x::Bool) = CenterIndexedArray(A.data .- x)
(.-)(x::Bool,A::CenterIndexedArray{Bool}) = CenterIndexedArray(x .- A.data)
(.+)(A::CenterIndexedArray,x::Number) = CenterIndexedArray(A.data .+ x)
(.+)(x::Number,A::CenterIndexedArray) = CenterIndexedArray(x .+ A.data)
(.-)(A::CenterIndexedArray,x::Number) = CenterIndexedArray(A.data .- x)
(.-)(x::Number,A::CenterIndexedArray) = CenterIndexedArray(x .- A.data)

(*)(A::CenterIndexedArray,x::Number) = CenterIndexedArray(A.data * x)
(*)(x::Number,A::CenterIndexedArray) = CenterIndexedArray(x * A.data)
(/)(A::CenterIndexedArray,x::Number) = CenterIndexedArray(A.data / x)
(.*){T<:Dates.Period}(A::CenterIndexedArray{T},x::Real) = CenterIndexedArray(A.data .* x)  # ambiguity
(.*)(A::CenterIndexedArray,x::Number) = CenterIndexedArray(A.data .* x)
(.*){T<:Dates.Period}(x::Real,A::CenterIndexedArray{T}) = CenterIndexedArray(x .* A.data)  # ambiguity
(.*)(x::Number,A::CenterIndexedArray) = CenterIndexedArray(x .* A.data)
(./){T<:Dates.Period}(A::CenterIndexedArray{T},x::Real) = CenterIndexedArray(A.data ./ x)  # ambiguity
(./)(A::CenterIndexedArray,x::Number) = CenterIndexedArray(A.data ./ x)
(./)(x::Number,A::CenterIndexedArray) = CenterIndexedArray(x ./ A.data)

writemime(io::IO, ::MIME"text/plain", X::CenterIndexedArray) =
    Base.with_output_limit(()->begin
        print(io, summary(X))
        !isempty(X) && println(io, ":")
        Base.showarray(io, X.data, header=false, repr=false)
    end)

end
