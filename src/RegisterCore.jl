#__precompile__()

module RegisterCore

using CenterIndexedArrays
using Base.Cartesian: @nloops, @nref, @ntuple
using FileIO

import Base: +, -, *, /
import Base: eltype, getindex, ndims, pointer, setindex!, show, size
import Base: checksize, unsafe_getindex
import FileIO: save
import CenterIndexedArrays: CenterIndexedArray

export
    # types
    MismatchArray,
    MismatchData,
    NumDenom,
    # functions
    save,
    indminmismatch

"""
# RegisterCore

`RegisterCore` contains low-level utilities for working with "mismatch
data."

## Mismatch

*Mismatch* refers to the mean-square pixelwise difference between two
images of the same size.  The mismatch is computed from two images,
called `fixed` and `moving`.  The mismatch is computed for a set of
translations (shifts) of the `moving` image, with a single number
returned for each possible translation. All translations by an integer
number of pixels are considered, up to a size `maxshift`.

As a simple example, suppose that `fixed` and `moving` are grayscale
two-dimensional images of size `m`-by-`n`. Computing the mismatch `D`
for a zero-pixel shift would result in a single number; computing for
shift by one pixel along the first coordinate (a shift of `(1,0)`)
would result in a different number. Using a `maxshift` of `(3,4)`, we
could store the mismatch for all possible shifts in an array `D` of
size `(7,9)`.

Shift data are stored in a type called a `CenterIndexedArray`, in
which indexing is performed relative to the center.  Consequently,
`D[0,0]` (the center point) would correspond to
`sum((fixed-moving).^2)`, meaning that the two images were directly
overlapped without translation.  (For the precise definition of `D`,
which also includes a normalization term, see the next paragraph.)
`D[1,0]`, displaced from the center by `(1,0)`, represents the
mismatch for a single-pixel shift of `moving` along the first
coordinate, corresponding to
`sum((fixed[1:end-1,:]-moving[2:end,:]).^2)`.  Likewise, `D[-1,0]`
corresponds to an identical shift in the opposite direction.

Mismatch computations actually return two numbers, conventionaly
called `num` and `denom` (packed into a type called `NumDenom`), and
the actual mean-square mismatch is `num/denom`. `num` represents the
numerator of the mismatch, for example
`sum((fixed[1:end-1,:]-moving[2:end,:]).^2)`.  `denom` is used for
normalization, and can follow one of two conventions. `:pixel`
normalization returns the number of valid pixels in the overlap
region, including the effects of any shift; for a shift of `(0,0)`
that would be `m*n`, but for a shift of `(1,0)` it would be `(m-1)*n`,
because we clip one row of each image.  `:intensity` normalization
computes the sum-of-square intensities within the overlap region,
e.g., `denom = sum(fixed[1:end-1,:].^2) + sum(moving[2:end,:].^2)` for
a `(1,0)` shift. The default is `:intensity`, because that makes the
overall mismatch `D = num/denom` a dimensionless quantity that does
not depend on the brightness of your illumination, etc.


While one might initially imagine returning the ratio `num/denom`
directly, there are several reasons to return `num` and `denom`
separately:

- Mathematically, "apertured" (block) computation involves sums of `num`
  and `denom` arrays separately (see below);
- If the shift is so large that there are no pixels of overlap between
  `fixed` and `moving`, both `num` and `denom` should be zero.
  However, because `num` and `denom` are computed by Fourier methods,
  there will be roundoff error.  Returning them separately allows you
  to control the threshold for what is considered "signal" or "noise"
  (see `truncatenoise!` and related functions below). Indeed, by
  appropriate choice of threshold you can require a minimum finite
  overlap, for example in terms of numbers of pixels (for `:pixel`
  normalization) or amount of image intensity (for `:intensity`
  normalization).

### Apertured mismatch: computing mismatch in blocks

Mismatch can be computed as a whole, or over *apertures*. The basic
concept behind apertures is simple: given that an image may deform
differently in different portions of space, restrict the
mean-square-error computation to a local group of pixels.

Currently, apertures are arranged in a grid, although this may change
in the future.  Conceptually, if you want a (2,2) grid of blocks, you
break the `fixed` and `moving` images up into quadrants and compute
the mismatch separately for each quadrant. The actual implementation
is a bit more complex, but also a bit more useful:

- Apertures are not "clipped" before computing the mismatch as a function
  of shift; instead, clipping at boundaries effectively happens
  after shifting. This allows one to use all the information available
  in both images.
- One might naively assume that, when using a `gridsize` of (3,3), a
  33x36 image would be split into nine 11x12 apertures. However, this
  strategy corresponds to having the *centers* of each block at the
  following grid of locations:

```
    (6, 6.5)    (6, 18.5)    (6, 30.5)
    (17,6.5)    (17,18.5)    (17,30.5)
    (28,6.5)    (28,18.5)    (28,30.5)
```

Instead, here the convention is that the aperture centers are on a grid
that spans the fixed image:

```
    (1, 1)      (1, 18.5)    (1, 36)
    (17,1)      (17,18.5)    (17,36)
    (33,1)      (33,18.5)    (33,36)
```

In each aperture, the data used for comparison are symmetric around the
block center. As a consequence, the `[1,1]` aperture has 3/4 of its data
(upper-left, upper-right, and lower-left quadrants) missing. By
contrast, the `[2,2]` aperture does not have any missing data, and by
default the `[2,2]` aperture includes `9/16 = (3/4)^2` of the pixels in
the image (with the boundary at the halfway point between block
centers). The motivation for this convention is that it reduces the
need to *extrapolate* shifts, because the aperture centers span the
entire fixed image.

### Representation of mismatch data for apertured computations

When the apertures are arranged in a grid pattern, the mismatch arrays
for each aperture can be stored in an array-of-arrays.  The "inner"
arrays have type `CenterIndexedArray{NumDenom{T}}` and are indexed by
shifts (of either sign).  The "outer" array is indexed in convention
Julia style.

### NaN values

Any pixels with `NaN` values are omitted from mismatch computation,
for both the numerator and denominator. This is different from filling
`NaN`s with 0; instead, it's as if those pixels simply don't
exist. This provides several nice features:

- You can register a smaller image to a larger one by padding the
  smaller image with NaN. The registration will not be affected by the
  fact that there's an "edge" at the padding location.
- You can re-register a warped moving image to the fixed image (hoping
  to further improve the registration), and not worry about the fact
  that the edges of the warped image likely have NaNs
- You can mark "bad pixels" produced by your camera.


## API

The major functions/types exported by this module are:

- `` and `gridsize`:
- `NumDenom` and `pack_nd`: packed pair representation of
  `(num,denom)` mismatch data.
- `indminmismatch`: a utility function for finding the location of the
  minimum mismatch
- `MismatchData` and `save`: utilities for disk storage of mismatch data.

"""
RegisterCore

"""
`NumDenom{T}` is a 2-vector containing a `(num,denom)` pair.  If `x`
is a `NumDenom`, `x.num` is `num` and `x.denom` is `denom`.
`NumDenom` objects act like vectors, and can be added and multiplied
by scalars.

This representation is efficient for `Interpolations.jl`, because it
allows interpolation to be performed on "both arrays" at once without
recomputing the interpolation coefficients.
"""
immutable NumDenom{T<:Number}
    num::T
    denom::T
end

(+)(p1::NumDenom, p2::NumDenom) = NumDenom(p1.num+p2.num, p1.denom+p2.denom)
(-)(p1::NumDenom, p2::NumDenom) = NumDenom(p1.num-p2.num, p1.denom-p2.denom)
(*)(n::Number, p::NumDenom) = NumDenom(n*p.num, n*p.denom)
(*)(p::NumDenom, n::Number) = n*p
(/)(p::NumDenom, n::Number) = NumDenom(p.num/n, p.denom/n)
Base.one{T}(::Type{NumDenom{T}}) = NumDenom(one(T),one(T))
Base.zero{T}(::Type{NumDenom{T}}) = NumDenom(zero(T),zero(T))
Base.promote_rule{T1,T2<:Number}(::Type{NumDenom{T1}}, ::Type{T2}) = NumDenom{promote_type(T1,T2)}

typealias MismatchArray{T<:Number,N} CenterIndexedArray{NumDenom{T},N}

"""
`numdenom = MismatchArray(nums, denoms)` packs the array-pair
`(nums,denoms)` into a single `MismatchArray`.  This is useful
preparation for interpolation.
"""
function Base.call{M<:MismatchArray}(::Type{M}, nums::AbstractArray, denoms::AbstractArray)
    size(nums) == size(denoms) || throw(DimensionMismatch("nums and denoms must have the same size"))
    T = promote_type(eltype(nums), eltype(denoms))
    numdenom = Array(NumDenom{T}, size(nums))
    @simd for I in eachindex(nums)
        @inbounds numdenom[I] = NumDenom(nums[I], denoms[I])
    end
    CenterIndexedArray(numdenom)
end

#### MismatchData ####
"""
`MismatchData` is designed as an accessor for a file format, although
it can be used in-memory as well. The most important fields are:

- `maxshift`, the maximum displacement along each axis for which
  shifted mismatch data are available;
- `gridsize`: a vector of length `ndims`, giving the size of the grid
  of "control points"
- `data`: a NumDenom array with the dimensions
  `[2*maxshift+1;gridsize;nstacks]`, holding the mismatch `num/denom`
  data.  For a 2d registration, this will be a 5-dimensional array
  even if you only have two images you're registering.
- `stacks`: a vector of stack indexes from the original images
- `stack_base`: the stack index of the reference ("fixed") image
"""
type MismatchData{T,N}
    data::Array{MismatchArray{T,N},N}
    stack_base::Int
    stacks::Vector{Int}
    maxshift::Vector{Int}
    gridsize::Vector{Int}
    normalization::Char
    key::Int32
end
MismatchData(data, stack_base, stacks, maxshift, gridsize, normalization) =
    MismatchData(data, stack_base, stacks, maxshift, gridsize, normalization, Int32(0))

"""
`mmd = MismatchData(numdenoms; [normalization=:intensity],
[stack_base=-1])` constructs a `MismatchData` object from a single
fixed/moving image pair.
"""
function MismatchData(numdenoms::AbstractArray; normalization = :intensity, stack_base=-1)
    error("FIXME")
    gsize = gridsize(nums)
    N = length(gsize)
    numB = getblock(nums, ones(Int, N)...)
    T = eltype(numB)
    blocksize = size(numB)
    data = Array(T, blocksize..., 2, gsize...)
    rng = ntuple(i->1:blocksize[i], N)
    for i = 1:prod(gsize)
        t = ind2sub(gsize, i)
        data[rng..., 1, i] = getblock(nums, t...)
        data[rng..., 2, i] = getblock(denoms, t...)
    end
    MismatchData(data, stack_base, [1], [blocksize...], [gsize...], normalization==:intensity ? 'I' : 'P', Int32(0))
end

#### MismatchData files ####
"""
`mmd = MismatchData(filename)` loads `MismatchData` from the specified file.
"""
function MismatchData(filename::AbstractString)
    error("FIXME")
    filename = endswith(filename, ".mismatch") ? filename : filename*".mismatch"
    s = open(filename, "r");
    magic = b"MISMATCH"
    magic_check = read(s, UInt8, length(magic))
    if magic_check != magic
        error("The format of ", filename, " does not seem to be correct")
    end
    version = read(s, Int32)
    if version < 1 || version > 2
        error("Version not recognized")
    end
    stack_base = Int(read(s, Int32))
    n_stacks = Int(read(s, Int32))
    stacks = convert(Vector{Int}, read(s, Int32, n_stacks))
    n_dims = Int(read(s, Int32))
    gridsize = convert(Vector{Int}, read(s, Int32, n_dims))
    blocksize = convert(Vector{Int}, read(s, Int32, n_dims))
    if any(blocksize .< 1)
        error(filename, " has incorrect block size; perhaps execution terminated prematurely?")
    end
    nbits = read(s, Int32)
    if nbits == 32
        T = Float32
    elseif nbits == 64
        T = Float64
    else
        error("Number of bits not recognized")
    end
    normint = read(s, Int32)
    normchar = 'I'
    if normint == 1
    elseif normint == 2
        normchar = 'P'
    else
        error("Normalization not recognized")
    end
    if version >= 2
        key = read(s, Int32)
    else
        key = Int32(0)
    end
    offset = position(s)
    # Check the file size
    n_blocks = prod(gridsize)
    n_pix = prod(blocksize)
    fileszexpected = 2*n_stacks*n_pix*n_blocks*sizeof(T)+offset
    filesz = filesize(filename)
    if filesz != fileszexpected
        error("File size ", filesz, " does not match the expected ", fileszexpected, " for file ", filename)
    end
    datasize = tuple(blocksize..., 2, gridsize..., n_stacks)
    data = Mmap.mmap(s, Array{T,length(datasize)}, datasize, offset)
    return MismatchData{T,ndims(data)}(data, stack_base, stacks, blocksize, gridsize, normchar, key)
end

"""
`mmd = MismatchData(filenames::Vector)` loads `MismatchData` from a list of files.
"""
function MismatchData(filenames::Vector{AbstractString})
    mmd = MismatchData(filenames[1])
    T = eltype(mmd)
    mmd_all = Array(MismatchData{T}, length(filenames))
    mmd_all[1] = mmd
    for i = 2:length(filenames)
        mmd = MismatchData(filenames[i])
        if eltype(mmd) != T
            error("All mismatch files must use the same data type")
        end
        mmd_all[i] = mmd
    end
    return mmd_all
end

function write_header{T<:AbstractFloat}(s::IO, mmd::MismatchData{T})
    error("FIXME")
    magic = b"MISMATCH"
    write(s, magic)
    write(s, Int32(1))  # version
    write(s, Int32(mmd.stack_base))
    write(s, Int32(length(mmd.stacks)))
    stacks32 = convert(Vector{Int32}, mmd.stacks)
    write(s, stacks32)
    write(s, Int32(length(mmd.gridsize)))
    gridsize32 = convert(Vector{Int32}, mmd.gridsize)
    write(s, gridsize32)
    blocksize32 = convert(Vector{Int32}, mmd.blocksize)
    write(s, blocksize32)
    nbits = sizeof(T)*8
    write(s, Int32(nbits))
    if mmd.normalization == 'I'
        write(s, Int32(1))
    elseif mmd.normalization == 'P'
        write(s, Int32(2))
    else
        error("Normalization not recognized")
    end
end

function create_mmd(filename::AbstractString)
    ext = ".mismatch"
    if !endswith(filename, ext)
        filename = filename*ext
    end
    s = open(filename, "w")
    return s, filename
end

"""
`save(filename, mmd::MismatchData)` writes `mmd` to the file specified
by `filename`.  If not included, a `".mismatch"` extension will be
added to the filename.
"""
function save(filename::AbstractString, mmd::MismatchData)
    s, fname = create_mmd(filename)
    write_header(s, mmd)
    write(s, mmd.data)
    close(s)
end

function append_mmd(s::IO, nums::AbstractArray, denoms::AbstractArray)
    gsz = gridsize(nums)
    for i = 1:prod(gsz)
        t = ind2sub(gsz, i)
        write(s, getblock(nums, t...))
        write(s, getblock(denoms, t...))
    end
end

normdict = Dict('I' => "intensity", 'P' => "pixels")

#### MismatchData utilities ####
function show(io::IO, mmd::MismatchData)
    println(io, typeof(mmd), ":")
    println(io, "  stack_base: ", mmd.stack_base)
    print(io, "  stacks: ", mmd.stacks')
    print(io, "  maxshift: ", mmd.maxshift')
    print(io, "  gridsize:  ", mmd.gridsize')
    print(io, "  normalization: ", mmd.normalization)
end

eltype{T}(mmd::MismatchData{T}) = T
ndims{T,N}(mmd::MismatchData{T,N}) = N


#### Utility functions ####

"""
`index = indmin_mismatch(numdenom, thresh)` returns the location of
the minimum value of what is effectively `num./denom`.  However, it
considers only those points for which `denom .> thresh`; moreover, it
will never choose an edge point.  `index` is a CartesianIndex into the
arrays.
"""
@generated function indmin_mismatch{T,N}(numdenom::MismatchArray{T,N}, thresh::Real)
    icenter = ntuple(d->0, N)
    quote
        imin = $icenter   # default is center of the array
        rmin = typemax(T)
        threshT = convert(T, thresh)
        halfsize = numdenom.halfsize
        @inbounds @nloops $N i d->-halfsize[d]+1:halfsize[d]-1 begin
            nd = @nref $N numdenom i
            if nd.denom > threshT
                r = nd.num/nd.denom
                if r < rmin
                    imin = @ntuple $N i
                    rmin = r
                end
            end
        end
        CartesianIndex(imin)
    end
end

end
