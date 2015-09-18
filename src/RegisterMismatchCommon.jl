# Note: not a module

using RegisterCore, Compat

export defaultblocksize, highpass, truncatenoise!, mismatch0, mismatchcenter

typealias DimsLike Union(Vector{Int}, Dims)

mmtype(T) = typeof((one(T)+one(T))/1)

"""
`num, denom = mismatch0(fixed, moving, [normalization])` computes the
"as-is" mismatch between `fixed` and `moving`, without any shift.
`normalization` may be either `:intensity` (the default) or `:pixels`.
The mean-square mismatch is `num/denom`.
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
    num, denom
end

"""
`num0, denom0 = mismatchcenter(nums, denoms)` computes the "as-is"
mismatch between `fixed` and `moving`, without any shift.  The
mismatch is already computed "blockwise" and stored as the
Arrays-of-Arrays `nums, denoms`. The outputs are the same as generated
by `mismatch0`.
"""
function mismatchcenter(nums, denoms)
    gsize = gridsize(nums)
    num0 = denom0 = 0.0
    for i = 1:prod(gsize)
        s = ind2sub(gsize, i)
        B = getblock(nums, s...)
        l = length(B)
        num0 += B[l>>1+1]
        B = getblock(denoms, s...)
        denom0 += B[l>>1+1]
    end
    num0, denom0
end

function defaultblocksize(img, gridsize::DimsLike, overlap::DimsLike = zeros(Int, ndims(img)))
    sc = coords_spatial(img)
    for i = 1:length(sc)
        if gridsize[i] > size(img, sc[i])
            error("gridsize is too large, given the size of the image")
        end
    end
    gsz1 = max(1,[gridsize...].-1)
    tmp = [size(img)[sc]...]./gsz1
    tuple([ceil(Int, t) for t in tmp] + [overlap...].*([gridsize...].>1)...)
end

function highpass(data, sigma; astype=Float64)
    sigmav = [sigma...]
    if any(isinf(sigmav))
        datahp = convert(Array{astype,ndims(data)}, data)
    else
        datahp = data - imfilter_gaussian(data, sigmav, astype=astype)
    end
    datahp[datahp .< 0] = 0  # truncate anything below 0
    datahp
end

function truncatenoise!{T<:Real}(num::AbstractArray{T}, denom::AbstractArray{T}, thresh::Real)
    for j = 1:length(denom)
        if abs(denom[j]) <= thresh
            num[j] = 0
            denom[j] = 0
        end
    end
    nothing
end

function truncatenoise!{A<:AbstractArray}(nums::Array{A}, denoms::Array{A}, thresh::Real)
    for i = 1:length(denoms)
        truncatenoise!(nums[i], denoms[i], thresh)
    end
    nothing
end

function register_translate(fixed, moving, maxshift, thresh=nothing)
    num, denom = mismatch(fixed, moving, maxshift)
    if thresh==nothing
        thresh = 0.01maximum(denom)
    end
    ind2disp(size(num), indminmismatch(num, denom, thresh))
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

function blockspan(img, blocksize, gridsize)
    imgssz = size(img)[coords_spatial(img)]
    overlap = computeoverlap(imgssz, blocksize, gridsize)
    N = length(imgssz)
    centers = Vector{Int}[ gridsize[i] > 1 ? round(Int, linspace(1,imgssz[i],gridsize[i])) : [imgssz[i]>>1] for i = 1:N ]
    blk = [b>>1 for b in blocksize]
    lower = Vector{Int}[ centers[i].-blk[i].+1 for i = 1:N ]
    upper = Vector{Int}[ min(centers[i].+blk[i].+1, [lower[i][2:end].-1.+overlap[i];typemax(Int)]) for i = 1:N]
    lower, upper
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

# These avoid making a copy if it's not necessary
tovec(v::AbstractVector) = v
tovec(v::Tuple) = [v...]
