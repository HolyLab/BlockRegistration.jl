using FixedSizeArrays, Images

typealias AbstractImage1{T} AbstractImage{T,1}
typealias AbstractImage2{T} AbstractImage{T,2}

# setindex! for conventional AbstractArrays
# Some of these are needed only for ambiguity resolution
for V in (Vector, AbstractImage1, AbstractVector)
    @eval begin
        Base.setindex!(x::$V, v::Vec, index::AbstractVector{Int}) =
            _setindex!(x, v, index)
    end
end

function _setindex!{Tx,Tv,N}(x::AbstractVector{Tx}, v::Vec{N,Tv}, index::AbstractVector{Int})
    length(index) == N || throw(DimensionMismatch("tried to assign $N elements to $(length(index)) destinations"))
    for (ix,iv) in zip(index, 1:N)
        x[ix] = v[iv]
    end
    v
end

Base.setindex!(x::AbstractVector, v::Vec, index::Colon) =
    _setindex!(x, v, 1:length(x))

for M in (AbstractImage2, AbstractMatrix)
    @eval begin
        function Base.setindex!(x::$M, m::Mat, index1::AbstractVector{Int}, index2::AbstractVector{Int})
            _setindex!(x, m, index1, index2)
        end
    end
end

function _setindex!{Tx,Tm,M,N}(x::AbstractMatrix{Tx}, m::Mat{M,N,Tm}, index1::AbstractVector{Int}, index2::AbstractVector{Int})
    length(index1) == M || throw(DimensionMismatch("tried to assign $M elements to $(length(index1)) destinations"))
    length(index2) == N || throw(DimensionMismatch("tried to assign $N elements to $(length(index2)) destinations"))
    for (jx,jm) in zip(index2, 1:N)
        for (ix,im) in zip(index1, 1:M)
            x[ix, jx] = m[im, jm]
        end
    end
    m
end

for M in (SparseMatrixCSC, AbstractMatrix)
    @eval begin
        Base.setindex!(x::$M, m::Mat, ::Colon, ::Colon) =
            _setindex!(x, m, 1:size(x,1), 1:size(x,2))

        Base.setindex!(x::$M, m::Mat, index1::AbstractVector{Int}, ::Colon) =
            setindex!(x, m, index1, 1:size(x,2))

        Base.setindex!(x::$M, m::Mat, ::Colon, index2::AbstractVector{Int}) =
            setindex!(x, m, 1:size(x,1), index2)
    end
end
