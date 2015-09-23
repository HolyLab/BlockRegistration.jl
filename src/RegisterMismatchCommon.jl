# Note: not a module, included into RegisterMismatch or RegisterMismatchCuda

using RegisterCore

export nanpad, mismatch0, aperture_grid, allocate_mmarrays, default_aperture_size, highpass, truncatenoise!

typealias DimsLike Union{AbstractVector{Int}, Dims}
typealias WidthLike Union{AbstractVector,Tuple}

mmtype(T) = typeof((one(T)+one(T))/1)

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
function mismatch0{T,N}(fixed::AbstractArray{T,N}, moving::AbstractArray{T,N}; normalization = :intensity)
    size(fixed) == size(moving) || throw(DimensionMismatch("Size $(size(fixed)) of fixed is not equal to size $(size(moving)) of moving"))
    num = denom = zero(mmtype(T))
    if normalization == :intensity
        for i in eachindex(fixed, moving)
            vf = fixed[i]
            vm = moving[i]
            if isfinite(vf) && isfinite(vm)
                num += (vf-vm)^2
                denom += vf^2 + vm^2
            end
        end
    elseif normalization == :pixels
        for i in eachindex(fixed, moving)
            vf = fixed[i]
            vm = moving[i]
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
function aperture_grid(ssize, gridsize)
    N = length(ssize)
    length(gridsize) == N || error("ssize and gridsize must have the same length")
    grid = Array(NTuple{N,Float64}, gridsize...)
    centers = map(i-> gridsize[i] > 1 ? collect(linspace(1,ssize[i],gridsize[i])) : [(ssize[i]+1)/2], 1:N)
    for I in CartesianRange(size(grid))
        grid[I] = ntuple(i->centers[i][I[i]], N)
    end
    grid
end

"""
`mms = allocate_mmarrays(T, aperture_centers, maxshift)` allocates
storage for aperture-wise mismatch computation. `mms` will be an
Array-of-MismatchArrays with element type `NumDenom{T}` and half-size
`maxshift`, with the outer array having the same "grid" shape as
`aperture_centers`.  The centers can in general be provided as an
vector-of-tuples, vector-of-vectors, or a matrix with each point in a
column.  If your centers are arranged in a rectangular grid, you can
use an `N`-dimensional array-of-tuples (or array-of-vectors) or an
`N+1`-dimensional array with the center positions specified along the
first dimension.
"""
function allocate_mmarrays{T,C<:Union{AbstractVector,Tuple}}(::Type{T}, aperture_centers::AbstractArray{C}, maxshift)
    isempty(aperture_centers) && error("aperture_centers is empty")
    N = length(first(aperture_centers))
    mms = Array(MismatchArray{NumDenom{T},N}, size(aperture_centers))
    sz = map(x->2*x+1, maxshift)
    for i in eachindex(mms)
        mms[i] = MismatchArray(T, sz...)
    end
    mms
end

function allocate_mmarrays{T,R<:Real}(::Type{T}, aperture_centers::AbstractArray{R}, maxshift)
    N = ndims(aperture_centers)-1
    mms = Array(MismatchArray{T,N}, size(aperture_centers)[2:end])
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

    FirstDimIterator(data) = new(data, CartesianRange(Base.tail(size(data))))
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
            error("gridsize is too large, given the size of the image")
        end
    end
    gsz1 = max(1,[gridsize...].-1)
    gflag = [gridsize...].>1
    tuple((([size(img)[sc]...]-gflag)./gsz1+2*[overlap...].*gflag)...)
end

"""
`datahp = highpass([T], data, sigma)` returns a highpass-filtered
version of `data`, with all negative values truncated at 0.  The
highpass is computed by subtracting a lowpass-filtered version of
data, using Gaussian filtering of width `sigma`.  As it is based on
`Image.jl`'s Gaussian filter, it gracefully handles `NaN` values.

If you do not wish to highpass-filter along a particular axis, put
`Inf` into the corresponding slot in `sigma`.

You may optionally specify the element type of the result, which for
`Integer` or `FixedPoint` inputs defaults to `Float32`.
"""
function highpass{T}(::Type{T}, data::AbstractArray, sigma)
    sigmav = [sigma...]
    if any(isinf(sigmav))
        datahp = convert(Array{T,ndims(data)}, data)
    else
        datahp = data - imfilter_gaussian(data, sigmav, astype=T)
    end
    datahp[datahp .< 0] = 0  # truncate anything below 0
    datahp
end
highpass{T<:AbstractFloat}(data::AbstractArray{T}, sigma) = highpass(T, data, sigma)
highpass(data::AbstractArray, sigma) = highpass(Float32, data, sigma)

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
    sz = Array(Int, length(blocksize))
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
        size(A, i) == size(B, i) || return false
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
    src2 = Base.sub_unsafe(src, tuple(isrc...))
    assertsamesize(dest, src2)
    # Determine the in-bounds region. If src slices some dimensions,
    # we need to skip over them.
    newindexes = Array(Any, 0)
    sizehint!(newindexes, ndims(src2))
    psize = Array(Int, 0)
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