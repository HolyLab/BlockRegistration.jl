__precompile__()

# Note: this requires a patch to julia itself. See julia PR#13203/#13235.

module CenterIndexedArrays

using Images
using Compat

export CenterIndexedArray

## SymRange, an AbstractUnitRange that's symmetric around 0
# These are used as indices for CenterIndexedArrays
immutable SymRange <: AbstractUnitRange{Int}
    n::Int  # goes from -n:n
end

Base.first(r::SymRange) = -r.n
Base.last(r::SymRange) = r.n

Base.start(r::SymRange) = first(r)
Base.done(r::SymRange, i) = i == last(r) + 1

@inline function Base.getindex(v::CenterIndexedArrays.SymRange, i::Int)
    ret = first(v) + i - 1
    @boundscheck abs(ret) <= v.n || Base.throw_boundserror(v, i)
    ret
end

Base.intersect(r::SymRange, s::SymRange) = SymRange(min(last(r), last(s)))

@inline function Base.getindex(r::SymRange, s::SymRange)
    @boundscheck checkbounds(r, s)
    s
end

Base.promote_rule{UR<:AbstractUnitRange}(::Type{SymRange}, ::Type{UR}) =
    UR
Base.promote_rule{T2}(::Type{UnitRange{T2}}, ::Type{SymRange}) =
    UnitRange{promote_type(T2, Int)}
function Base.convert(::Type{SymRange}, r::AbstractUnitRange)
    first(r) == -last(r) || error("cannot convert $r to a SymRange")
    SymRange(last(r))
end

Base.show(io::IO, r::SymRange) = print(io, "SymRange(", repr(last(r)), ')')


"""
A `CenterIndexedArray` is one for which the array center has indexes
`0,0,...`. Along each coordinate, allowed indexes range from `-n:n`.

CenterIndexedArray(A) "converts" `A` into a CenterIndexedArray. All
the sizes of `A` must be odd.
"""
immutable CenterIndexedArray{T,N,A<:AbstractArray} <: AbstractArray{T,N}
    data::A
    halfsize::NTuple{N,Int}

    function (::Type{CenterIndexedArray{T,N,A}}){T,N,A<:AbstractArray}(data::A)
        new{T,N,A}(data, _halfsize(data))
    end
end

CenterIndexedArray{T,N}(A::AbstractArray{T,N}) = CenterIndexedArray{T,N,typeof(A)}(A)
CenterIndexedArray{T}(::Type{T}, dims) = CenterIndexedArray(Array{T}(dims))
CenterIndexedArray{T}(::Type{T}, dims...) = CenterIndexedArray(Array{T}(dims))

# This is the AbstractArray default, but do this just to be sure
@compat Base.IndexStyle{A<:CenterIndexedArray}(::Type{A}) = IndexCartesian()

Base.size(A::CenterIndexedArray) = size(A.data)
Base.indices(A::CenterIndexedArray) = map(SymRange, A.halfsize)

function Base.similar{T}(A::CenterIndexedArray, ::Type{T}, inds::Tuple{SymRange,Vararg{SymRange}})
    data = Array{T}(map(length, inds))
    CenterIndexedArray(data)
end
function Base.similar(T::Union{Type,Function}, inds::Tuple{SymRange, Vararg{SymRange}})
    data = T(map(length, inds))
    CenterIndexedArray(data)
end

function _halfsize(A::AbstractArray)
    all(isodd, size(A)) || error("Must have all-odd sizes")
    map(n->n>>UInt(1), size(A))
end

@inline function Base.getindex{T,N}(A::CenterIndexedArray{T,N}, i::Vararg{Number,N})
    @boundscheck checkbounds(A, i...)
    @inbounds val = A.data[map(offset, A.halfsize, i)...]
    val
end

offset(off, i) = off+i+1

const Index = Union{Colon,AbstractVector}

Base.getindex{T}(A::CenterIndexedArray{T,1}, I::Index) = CenterIndexedArray([A[i] for i in _cindex(A, 1, I)])
Base.getindex{T}(A::CenterIndexedArray{T,2}, I::Index, J::Index) = CenterIndexedArray([A[i,j] for i in _cindex(A,1,I), j in _cindex(A,2,J)])
Base.getindex{T}(A::CenterIndexedArray{T,3}, I::Index, J::Index, K::Index) = CenterIndexedArray([A[i,j,k] for i in _cindex(A,1,I), j in _cindex(A,2,J), k in _cindex(A,3,K)])

_cindex(A::CenterIndexedArray, d, I::Range) = convert(SymRange, I)
_cindex(A::CenterIndexedArray, d, I::AbstractVector) = error("unsupported, use a range")
_cindex(A::CenterIndexedArray, d, ::Colon) = SymRange(A.halfsize[d])


@inline function Base.setindex!{T,N}(A::CenterIndexedArray{T,N}, v, i::Vararg{Number,N})
    @boundscheck checkbounds(A, i...)
    @inbounds A.data[map(offset, A.halfsize, i)...] = v
    v
end

# TODO: make these behave sensibly in Base so that these are not needed
Base.vec(A::CenterIndexedArray) = vec(A.data)
Base.minimum(A::CenterIndexedArray, region) = CenterIndexedArray(minimum(A.data, region))
Base.maximum(A::CenterIndexedArray, region) = CenterIndexedArray(maximum(A.data, region))

end  # module
