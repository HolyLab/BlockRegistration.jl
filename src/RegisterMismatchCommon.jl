# Note: not a module, included into RegisterMismatch or RegisterMismatchCuda

using RegisterCore, CenterIndexedArrays

export correctbias!, nanpad, mismatch0, aperture_grid, allocate_mmarrays, default_aperture_width, truncatenoise!

const DimsLike = Union{AbstractVector{Int}, Dims}
const WidthLike = Union{AbstractVector,Tuple}

mismatch{T<:AbstractFloat}(fixed::AbstractArray{T}, moving::AbstractArray{T}, maxshift::DimsLike; normalization = :intensity) = mismatch(T, fixed, moving, maxshift; normalization=normalization)
mismatch(fixed::AbstractArray, moving::AbstractArray, maxshift::DimsLike; normalization = :intensity) = mismatch(Float32, fixed, moving, maxshift; normalization=normalization)

mismatch_apertures{T<:AbstractFloat}(fixed::AbstractArray{T}, moving::AbstractArray{T}, args...; kwargs...) = mismatch_apertures(T, fixed, moving, args...; kwargs...)
mismatch_apertures(fixed::AbstractArray, moving::AbstractArray, args...; kwargs...) = mismatch_apertures(Float32, fixed, moving, args...; kwargs...)

function mismatch_apertures{T}(::Type{T}, fixed::AbstractArray, moving::AbstractArray, gridsize::DimsLike, maxshift::DimsLike; kwargs...)
    cs = coords_spatial(fixed)
    aperture_centers = aperture_grid(size(fixed, cs...), gridsize)
    aperture_width = default_aperture_width(fixed, gridsize)
    mismatch_apertures(T, fixed, moving, aperture_centers, aperture_width, maxshift; kwargs...)
end

"""
`correctbias!(mm::MismatchArray)` replaces "suspect" mismatch
data with imputed data.  If each pixel in your camera has a different
bias, then matching that bias becomes an incentive to avoid
shifts.  Likewise, CMOS cameras tend to have correlated row/column
noise. These two factors combine to imply that `mm[i,j,...]` is unreliable
whenever `i` or `j` is zero.

Data are imputed by averaging the adjacent non-suspect values.  This
function works in-place, overwriting the original `mm`.
"""
function correctbias!{ND,N}(mm::MismatchArray{ND,N}, w = correctbias_weight(mm))
    T = eltype(ND)
    mxshift = maxshift(mm)
    Imax = CartesianIndex(mxshift)
    Imin = CartesianIndex(map(x->-x,mxshift)::NTuple{N,Int})
    I1 = CartesianIndex(ntuple(d->d>2?0:1, N)::NTuple{N,Int})  # only first 2 dims
    for I in eachindex(mm)
        if w[I] == 0
            mms = NumDenom{T}(0,0)
            ws = zero(T)
            for J in CartesianRange(max(Imin, I-I1), min(Imax, I+I1))
                wJ = w[J]
                if wJ != 0
                    mms += wJ*mm[J]
                    ws += wJ
                end
            end
            mm[I] = mms/ws
        end
    end
    mm
end

"`correctbias!(mms)` runs `correctbias!` on each element of an array-of-MismatchArrays."
function correctbias!{M<:MismatchArray}(mms::AbstractArray{M})
    for mm in mms
        correctbias!(mm)
    end
    mms
end

function correctbias_weight{ND,N}(mm::MismatchArray{ND,N})
    T = eltype(ND)
    w = CenterIndexedArray(ones(T, size(mm)))
    for I in eachindex(mm)
        anyzero = false
        for d = 1:min(N,2)   # only first 2 dims
            anyzero |= I[d] == 0
        end
        if anyzero
            w[I] = 0
        end
    end
    w
end

"""
`fixedpad, movingpad = nanpad(fixed, moving)` will pad `fixed` and/or
`moving` with NaN as needed to ensure that `fixedpad` and `movingpad`
have the same size.
"""
function nanpad(fixed, moving)
    ndims(fixed) == ndims(moving) || error("fixed and moving must have the same dimensionality")
    if size(fixed) == size(moving)
        return fixed, moving
    end
    rng = map(d->1:max(size(fixed,d), size(moving,d)), 1:ndims(fixed))
    T = promote_type(eltype(fixed), eltype(moving))
    get(fixed, rng, nanval(T)), get(moving, rng, nanval(T))
end

nanval{T<:AbstractFloat}(::Type{T}) = convert(T, NaN)
nanval{T}(::Type{T}) = convert(Float32, NaN)

"""
`mm0 = mismatch0(fixed, moving, [normalization])` computes the
"as-is" mismatch between `fixed` and `moving`, without any shift.
`normalization` may be either `:intensity` (the default) or `:pixels`.
"""
function mismatch0{Tf,Tm,N}(fixed::AbstractArray{Tf,N}, moving::AbstractArray{Tm,N}; normalization = :intensity)
    size(fixed) == size(moving) || throw(DimensionMismatch("Size $(size(fixed)) of fixed is not equal to size $(size(moving)) of moving"))
    _mismatch0(zero(Float64), zero(Float64), fixed, moving; normalization=normalization)
end

function _mismatch0{T,Tf,Tm,N}(num::T, denom::T, fixed::AbstractArray{Tf,N}, moving::AbstractArray{Tm,N}; normalization = :intensity)
    if normalization == :intensity
        for i in eachindex(fixed, moving)
            vf = T(fixed[i])
            vm = T(moving[i])
            if isfinite(vf) && isfinite(vm)
                num += (vf-vm)^2
                denom += vf^2 + vm^2
            end
        end
    elseif normalization == :pixels
        for i in eachindex(fixed, moving)
            vf = T(fixed[i])
            vm = T(moving[i])
            if isfinite(vf) && isfinite(vm)
                num += (vf-vm)^2
                denom += 1
            end
        end
    else
        error("Normalization $normalization not recognized")
    end
    NumDenom(num, denom)
end

"""
`mm0 = mismatch0(mms)` computes the "as-is"
mismatch between `fixed` and `moving`, without any shift.  The
mismatch is represented in `mms` as an aperture-wise
Arrays-of-MismatchArrays.
"""
function mismatch0{M<:MismatchArray}(mms::AbstractArray{M})
    mm0 = eltype(M)(0, 0)
    cr = eachindex(first(mms))
    z = cr.start+cr.stop  # all-zeros CartesianIndex
    for mm in mms
        mm0 += mm[z]
    end
    mm0
end

"""
`ag = aperture_grid(ssize, gridsize)` constructs a uniformly-spaced
grid of aperture centers.  The grid has size `gridsize`, and is
constructed for an image of spatial size `ssize`.  Along each
dimension the first and last elements are at the image corners.
"""
function aperture_grid{N}(ssize::Dims{N}, gridsize)
    if length(gridsize) != N
        if length(gridsize) == N-1
            info("ssize and gridsize disagree; possible fix is to use a :time axis (AxisArrays) for the image")
        end
        error("ssize and gridsize must have the same length, got $ssize and $gridsize")
    end
    grid = Array{NTuple{N,Float64},N}((gridsize...))
    centers = map(i-> gridsize[i] > 1 ? collect(linspace(1,ssize[i],gridsize[i])) : [(ssize[i]+1)/2], 1:N)
    for I in CartesianRange(size(grid))
        grid[I] = ntuple(i->centers[i][I[i]], N)
    end
    grid
end

"""
`mms = allocate_mmarrays(T, gridsize, maxshift)` allocates storage for
aperture-wise mismatch computation. `mms` will be an
Array-of-MismatchArrays with element type `NumDenom{T}` and half-size
`maxshift`. `mms` will be an array of size `gridsize`. This syntax is
recommended when your apertures are centered at points of a grid.

`mms = allocate_mmarrays(T, aperture_centers, maxshift)` returns `mms`
with a shape that matches that of `aperture_centers`. The centers can
in general be provided as an vector-of-tuples, vector-of-vectors, or a
matrix with each point in a column.  If your centers are arranged in a
rectangular grid, you can use an `N`-dimensional array-of-tuples (or
array-of-vectors) or an `N+1`-dimensional array with the center
positions specified along the first dimension.  (But you may find the
`gridsize` syntax to be simpler.)
"""
function allocate_mmarrays{T,C<:Union{AbstractVector,Tuple}}(::Type{T}, aperture_centers::AbstractArray{C}, maxshift)
    isempty(aperture_centers) && error("aperture_centers is empty")
    N = length(first(aperture_centers))
    sz = map(x->2*x+1, maxshift)
    mm = MismatchArray(T, sz...)
    mms = Array{typeof(mm)}(size(aperture_centers))
    f = true
    for i in eachindex(mms)
        if f
            mms[i] = mm
            f = false
        else
            mms[i] = MismatchArray(T, sz...)
        end
    end
    mms
end

function allocate_mmarrays{T,R<:Real}(::Type{T}, aperture_centers::AbstractArray{R}, maxshift)
    N = ndims(aperture_centers)-1
    mms = Array{MismatchArray{T,N}}(size(aperture_centers)[2:end])
    sz = map(x->2*x+1, maxshift)
    for i in eachindex(mms)
        mms[i] = MismatchArray(T, sz...)
    end
    mms
end

function allocate_mmarrays{T<:Real,N}(::Type{T}, gridsize::NTuple{N,Int}, maxshift)
    mms = Array{MismatchArray{NumDenom{T},N}}(gridsize)
    sz = map(x->2*x+1, maxshift)
    for i in eachindex(mms)
        mms[i] = MismatchArray(T, sz...)
    end
    mms
end

immutable ContainerIterator{C}
    data::C
end

Base.start(iter::ContainerIterator) = start(iter.data)
Base.done(iter::ContainerIterator, state) = done(iter.data, state)
Base.next(iter::ContainerIterator, state) = next(iter.data, state)

immutable FirstDimIterator{A<:AbstractArray,R<:CartesianRange}
    data::A
    rng::R

    (::Type{FirstDimIterator{A,R}}){A,R}(data::A) = new{A,R}(data, CartesianRange(Base.tail(size(data))))
end
FirstDimIterator(A::AbstractArray) = FirstDimIterator{typeof(A),typeof(CartesianRange(Base.tail(size(A))))}(A)

Base.start(iter::FirstDimIterator) = start(iter.rng)
Base.done(iter::FirstDimIterator, state) = done(iter.rng, state)
function Base.next(iter::FirstDimIterator, state)
    index, state = next(iter.rng, state)
    iter.data[:, index], state
end

"""
`iter = each_point(points)` yields an iterator `iter` over all the
points in `points`. `points` may be represented as an
AbstractArray-of-tuples or -AbstractVectors, or may be an
`AbstractArray` where each point is represented along the first
dimension (e.g., columns of a matrix).
"""
each_point{C<:Union{AbstractVector,Tuple}}(aperture_centers::AbstractArray{C}) = ContainerIterator(aperture_centers)

each_point{R<:Real}(aperture_centers::AbstractArray{R}) = FirstDimIterator(aperture_centers)

"""
`rng = aperture_range(center, width)` returns a tuple of
`UnitRange{Int}`s that, for dimension `d`, is centered on `center[d]`
and has width `width[d]`.
"""
function aperture_range(center, width)
    length(center) == length(width) || error("center and width must have the same length")
    ntuple(d->leftedge(center[d], width[d]):rightedge(center[d], width[d]), length(center))
end

"""
`aperturesize = default_aperture_width(img, gridsize, [overlap])`
calculates the aperture width for a regularly-spaced grid of aperture
centers with size `gridsize`.  Apertures that are adjacent along
dimension `d` may overlap by a number pixels specified by
`overlap[d]`; the default value is 0.  For non-negative `overlap`, the
collection of apertures will yield full coverage of the image.
"""
function default_aperture_width(img, gridsize::DimsLike, overlap::DimsLike = zeros(Int, sdims(img)))
    sc = coords_spatial(img)
    length(sc) == length(gridsize) == length(overlap) || error("gridsize and overlap must have length equal to the number of spatial dimensions in img")
    for i = 1:length(sc)
        if gridsize[i] > size(img, sc[i])
            error("gridsize $gridsize is too large, given the size $(size(img)[sc]) of the image")
        end
    end
    gsz1 = max.(1, [gridsize...].-1)
    gflag = [gridsize...].>1
    tuple((([size(img, sc...)...]-gflag)./gsz1+2*[overlap...].*gflag)...)
end

"""
`truncatenoise!(mm, thresh)` zeros out any entries of the
MismatchArray `mm` whose `denom` values are less than `thresh`.
"""
function truncatenoise!{T<:Real}(mm::AbstractArray{NumDenom{T}}, thresh::Real)
    for I in eachindex(mm)
        if mm[I].denom <= thresh
            mm[I] = NumDenom{T}(0,0)
        end
    end
    mm
end

function truncatenoise!{A<:MismatchArray}(mms::AbstractArray{A}, thresh::Real)
    for i = 1:length(denoms)
        truncatenoise!(mms[i], thresh)
    end
    nothing
end

"""
`shift = register_translate(fixed, moving, maxshift, [thresh])`
computes the integer-valued translation which best aligns images
`fixed` and `moving`. All shifts up to size `maxshift` are considered.
Optionally specify `thresh`, the fraction (0<=thresh<=1) of overlap
required between `fixed` and `moving` (default 0.25).
"""
function register_translate(fixed, moving, maxshift, thresh=nothing)
    mm = mismatch(fixed, moving, maxshift)
    _, denom = separate(mm)
    if thresh==nothing
        thresh = 0.25maximum(denom)
    end
    indmin_mismatch(mm, thresh)
end


function checksize_maxshift(A::AbstractArray, maxshift)
    ndims(A) == length(maxshift) || error("Array is $(ndims(A))-dimensional, but maxshift has length $(length(maxshift))")
    for i = 1:ndims(A)
        size(A,i) == 2*maxshift[i]+1 || error("Along dimension $i, the output size $(size(A,i)) does not agree with maxshift[$i] = $(maxshift[i])")
    end
    nothing
end

function padranges(blocksize, maxshift)
    padright = [maxshift...]
    transformdims = find(padright.>0)
    paddedsz = [blocksize...] + 2*padright
    for i in transformdims
        # Pick a size for which one can efficiently calculate ffts
        padright[i] += padsize(blocksize, maxshift, i) - paddedsz[i]
    end
    rng = UnitRange{Int}[ 1-maxshift[i]:blocksize[i]+padright[i] for i = 1:length(blocksize) ]
end

function padsize!(sz::Vector, blocksize, maxshift)
    n = length(blocksize)
    for i = 1:n
        sz[i] = padsize(blocksize, maxshift, i)
    end
    sz
end
function padsize(blocksize, maxshift)
    sz = Vector{Int}(length(blocksize))
    padsize!(sz, blocksize, maxshift)
end

function padsize(blocksize, maxshift, dim)
    m = maxshift[dim]
    p = blocksize[dim] + 2m
    return m > 0 ? (dim == 1 ? nextpow2(p) : nextprod(FFTPROD, p)) : p   # we won't FFT along dimensions with maxshift[i]==0
end

function assertsamesize(A, B)
    if !issamesize(A,B)
        error("Arrays are not the same size")
    end
end

function issamesize(A::AbstractArray, B::AbstractArray)
    n = ndims(A)
    ndims(B) == n || return false
    for i = 1:n
        indices(A, i) == indices(B, i) || return false
    end
    true
end

function issamesize(A::AbstractArray, indexes)
    n = ndims(A)
    length(indexes) == n || return false
    for i = 1:n
        size(A, i) == length(indexes[i]) || return false
    end
    true
end

safe_get!(dest::AbstractArray, src, isrc, default) = get!(dest, src, isrc, default)

"""
`safe_get!(dest, src, isrc, default)` is a variant of `get!` that is
safe for `src` SubArrays whose `indexes` may not be in-bounds.
"""
function safe_get!(dest::AbstractArray, src::SubArray, isrc, default)
    # Trim the source region, ignoring bounds constraints
    src2 = extraunsafe_view(src, isrc...)
    assertsamesize(dest, src2)
    # Determine the in-bounds region. If src slices some dimensions,
    # we need to skip over them.
    newindexes = Vector{Any}(0)
    sizehint!(newindexes, ndims(src2))
    psize = Vector{Int}(0)
    sizehint!(psize, ndims(src2))
    for i = 1:length(src2.indexes)
        j = src2.indexes[i]
        if !isa(j, Real)     # not a slice dimension
            push!(newindexes, j)
            push!(psize, size(src2.parent, i))
        end
    end
    idestcopy, _ = Base.indcopy(tuple(psize...), newindexes)
    if !issamesize(dest, idestcopy)
        fill!(dest, default)
        dest[idestcopy...] = src2[idestcopy...]  # src2 is already shifted, so use dest indexes
    else
        copy!(dest, sub(src2, idestcopy...))
    end
    dest
end

# This yields the _effective_ overlap, i.e., sets to zero if gridsize==1 along a coordinate
# imgssz = image spatial size
function computeoverlap(imgssz, blocksize, gridsize)
    gsz1 = max(1,[gridsize...].-1)
    tmp = [imgssz...]./gsz1
    blocksize - [ceil(Int, x) for x in tmp]
end

leftedge(center, width) = ceil(Int, center-width/2)
rightedge(center, width) = leftedge(center+width, width) - 1

# These avoid making a copy if it's not necessary
tovec(v::AbstractVector) = v
tovec(v::Tuple) = [v...]


### Utilities for unsafe indexing of views
# TODO: redesign this whole thing to be safer?
using Base: ViewIndex, to_indexes, unsafe_length, index_shape, tail

if VERSION < v"0.6.0-dev"
    @inline function extraunsafe_view{T,N}(V::SubArray{T,N}, I::Vararg{ViewIndex,N})
        idxs = unsafe_reindex(V, V.indexes, to_indexes(I...))
        SubArray(V.parent, idxs, map(unsafe_length, (index_shape(V.parent, idxs...))))
    end
else
    @inline function extraunsafe_view{T,N}(V::SubArray{T,N}, I::Vararg{ViewIndex,N})
        idxs = unsafe_reindex(V, V.indexes, to_indices(V, I))
        SubArray(V.parent, idxs)
    end
end

unsafe_reindex(V, idxs::Tuple{UnitRange, Vararg{Any}}, subidxs::Tuple{UnitRange, Vararg{Any}}) =
    (Base.@_propagate_inbounds_meta; @inbounds new1 = idxs[1][subidxs[1]]; (new1, unsafe_reindex(V, tail(idxs), tail(subidxs))...))

unsafe_reindex(V, idxs, subidxs) = Base.reindex(V, idxs, subidxs)
