#__precompile__()

module RegisterCore

using Base.Cartesian: @nloops, @nref
using FileIO

import Base: +, -, *, /
import Base: eltype, getindex, ndims, pointer, setindex!, show, size
import Base: checksize, unsafe_getindex
import FileIO: save

export
    # types
    Block,
    MismatchData,
    NumDenom,
    # functions
#    append,
    blockeltype,
    blocksize,
    save,
    getblock,
    getindex!,
    gridsize,
    indminmismatch,
    ind2disp,
    ind2disp!,
    pack_nd
#    write_header

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
with a `maxshift` of (3,4) would mean that `D` would have size
(7,9). `D[4,5]` (the center point) would correspond to
`sum((fixed-moving).^2)`, meaning that the two images were directly
overlapped without translation.  (For the precise definition of `D`,
which also includes a normalization term, see the next paragraph.)
`D[5,5]`, displaced from the center by (1,0), represents the mismatch
for a single-pixel shift of `moving` along the first coordinate,
corresponding to `sum((fixed[1:end-1,:]-moving[2:end,:]).^2)`.  Note
that the top row of `moving` is not used (because of the upward shift,
it does not overlap `fixed`), and likewise neither is the bottom row
of `fixed`. Conversely, `D[3,5]` would correspond to a downward
translation of `moving`.

Mismatch computations actually return two arrays, conventionaly called
`num` and `denom`, and `D = num./denom`.  `num` represents the
numerator of the mismatch, for example
`sum((fixed[1:end-1,:]-moving[2:end,:]).^2)`.  `denom` is used for
normalization, and can follow one of two conventions. `:pixel`
normalization returns the number of valid pixels in the overlap
region, including the effects of any shift; for `denom[4,5]` that
would be `m*n`, but for `denom[5,5]` it would be `(m-1)*n`, because we
clip one row of each image.  `:intensity` normalization computes the
sum-of-square intensities within the overlap region, e.g., `denom[5,5]
= sum(fixed[1:end-1,:].^2) + sum(moving[2:end,:].^2)`. The default is
`:intensity`, because that makes the overall mismatch `D = num./denom`
a dimensionless quantity.


While one might initially imagine returning `D = num./denom` directly,
there are several reasons to return `num` and `denom` separately:

- Mathematically, block computation (see below) involves sums of `num`
  and `denom` arrays separately;
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

### Computing mismatch in blocks

Mismatch can be computed as a whole, or in *blocks*. The basic concept
behind blocks is simple: if you want a (2,2) grid of blocks, you break
the `fixed` and `moving` images up into quadrants and compute the
mismatch separately for each quadrant. The actual implementation is a
bit more complex, but also a bit more useful:

- Blocks are not "clipped" before computing the mismatch as a function
  of shift; instead, clipping at block boundaries effectively happens
  after shifting. This allows one to use all the information available
  in both images.
- One might naively assume that, when using a `gridsize` of (3,3), a
  33x36 image would be split into nine 11x12 blocks. However, this
  strategy corresponds to having the *centers* of each block at the
  following grid of locations:

```
    (6, 6.5)    (6, 18.5)    (6, 30.5)
    (17,6.5)    (17,18.5)    (17,30.5)
    (28,6.5)    (28,18.5)    (28,30.5)
```

Instead, here the convention is that the block centers are on a grid
that spans the fixed image:

```
    (1, 1)      (1, 18.5)    (1, 36)
    (17,1)      (17,18.5)    (17,36)
    (33,1)      (33,18.5)    (33,36)
```

In each block, the data used for comparison are symmetric around the
block center. As a consequence, the `[1,1]` block has 3/4 of its data
(upper-left, upper-right, and lower-left quadrants) missing. By
contrast, the `[2,2]` block does not have any missing data, and by
default the `[2,2]` block includes `9/16 = (3/4)^2` of the pixels in
the image (with the boundary at the halfway point between block
centers). The motivation for this convention is that it reduces the
need to *extrapolate* shifts, because the block centers span the
entire fixed image.

### Representation of `(nums, denoms)` for block computations

When computing the mismatch in blocks, one obtains one `(num, denom)`
array-pair for each block; the aggregate over all blocks is sometimes
denoted `nums`, `denoms`. These may be represented in one of three
ways:

- As Arrays-of-Arrays: `nums[i,j]` is the `num` array for the `[i,j]` block
- As tiled Arrays: `nums[:,:,i,j]` is the `num` array for the `[i,j]` block
- As a tiled array of `NumDenom` pairs: `numdenom[:,:,i,j]` is the
  `numdenom` array for the `[i,j]` block.  (See `NumDenom` for more
  information.)

The first form is conceptually simpler.  The second and third forms
are more suitable for `SharedArray`s or memory-mapped files.

The `Block` type has been defined to provide a fast and efficient
"view" of one tile in a tiled Array: if `nums` is 4d,
```
num = Block(nums, i, j)
```
would return a 2d object equivalent to `nums[:,:,i,j]` but without copying data.

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

- `blocksize` and `gridsize`:
- `Block`: a special-purpose SubArray where slicing occurs along the
  trailing dimensions
- `NumDenom` and `pack_nd`: packed pair representation of
  `(num,denom)` mismatch data.
- `indminmismatch`: a utility function for finding the location of the
  minimum mismatch
- `ind2disp` and `ind2disp!`: convert linear-indexing
  minimum-locations (from `indminmismatch`) into cartesian
  *displacements*
- `MismatchData` and `save`: utilities for disk storage of mismatch data.

"""
RegisterCore

"""
`NumDenom{T}` is a 2-vector containing a `(num,denom)` pair.  If `x`
is a `NumDenom`, `x.num` is `num` and `x.denom` is `denom`.

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

"""
`numdenom = pack_nd(nums, denoms)` packs the array-pair
`(nums,denoms)` into a single `Array{NumDenom}`.  This is useful
preparation for interpolation.
"""
function pack_nd{T<:Number}(nums::AbstractArray{T}, denoms::AbstractArray{T})
    size(nums) == size(denoms) || throw(DimensionMismatch("nums and denoms must have the same size"))
    numdenom = Array(NumDenom{T}, size(nums))
    @simd for I in eachindex(nums)
        @inbounds numdenom[I] = NumDenom(nums[I], denoms[I])
    end
    numdenom
end

###
### Block, a type to simplify working with tiled arrays
###
"""
`B = Block(A, i, j)` returns a slice of `A`.  If `A` is a
5-dimensional array, `Block(A, i, j)` will return a view of
`A[:,:,:,i,j]`.  These can be viewed as special-purpose SubArrays.
"""
immutable Block{T,N,NE,A<:DenseArray} <: DenseArray{T,N}
    parent::A
    extraindexes::NTuple{NE,Int}
    offset::Int

    function Block(parent::A, extraindexes::NTuple{NE,Int})
        s = size(parent, 1)
        for i = 2:N
            s *= size(parent,i)
        end
        offset = 0
        for i = 1:NE
            offset += s*(extraindexes[i]-1)
            s *= size(parent,N+i)
        end
        new(parent, extraindexes, offset)
    end
end

function Block(parent::DenseArray, extraindexes::Tuple{Vararg{Int}})
    NE = length(extraindexes)
    N = ndims(parent)-NE
    Block{eltype(parent),N,NE,typeof(parent)}(parent, extraindexes)
end

Block(parent::DenseArray, extraindexes::Int...) = Block(parent, extraindexes)

size{T,N}(B::Block{T,N}) = size(B.parent)[1:N]
size{T,N}(B::Block{T,N}, i::Integer) = i <= N ? size(B.parent, i) : 1

pointer{T}(B::Block{T}) = pointer(B.parent)+B.offset*sizeof(T)

@inline function getindex!(dest, src, indexes...)
    checkbounds(src, indexes...)
    ndims(dest) == length(indexes) || throw(DimensionMismatch("dest and indexes don't match"))
    for d = 1:ndims(dest)
        size(dest, d) == length(indexes[d]) || throw(DimensionMismatch("dest and indexes don't match"))
    end
    Base._unsafe_getindex!(dest, Base.linearindexing(src, dest), src, indexes...)
end

# linear indexing
# FIXME: would be nice to have bounds-checking here, although currently the performance penalty is too high
getindex(B::Block, i::Real) = B.parent[i+B.offset]
setindex!(B::Block, v, i::Real) = setindex!(B.parent, v, i+B.offset)

# cartesian indexing
_eindexes(NE) = ntuple(i->:(B.extraindexes[$i]), NE)
@generated function getindex{T,N,NE}(B::Block{T,N,NE}, uindexes...)
    meta = Expr(:meta, :inline)
    eindexes = _eindexes(NE)
    quote
        $meta
        getindex(B.parent, uindexes..., $(eindexes...))
    end
end
@generated function setindex!{T,N,NE}(B::Block{T,N,NE}, v, uindexes...)
    meta = Expr(:meta, :inline)
    eindexes = _eindexes(NE)
    quote
        $meta
        setindex!(B.parent, v, uindexes..., $(eindexes...))
    end
end

### Generic interface for tiled array & array-of-arrays
getblock{T<:AbstractArray}(A::AbstractArray{T}, I...) = A[I...]
getblock{T<:Real}(A::AbstractArray{T}, I...) = Block(A, I...)

### Size & eltype support for the conventions of mismatch data representation
gridsize{T<:AbstractArray}(A::AbstractArray{T}) = size(A)
gridsize{T<:Union(Real,NumDenom)}(A::AbstractArray{T}) = size(A)[ndims(A)>>1+1:end]
gridsize(ND::Tuple) = gridsize(ND[1])

blocksize{T<:AbstractArray}(A::AbstractArray{T}) = size(A[1])
blocksize{T<:Union(Real,NumDenom)}(A::AbstractArray{T}) = size(A)[1:ndims(A)>>1]
blocksize(ND::Tuple) = blocksize(ND[1])

blockeltype{T<:AbstractArray}(A::AbstractArray{T}) = eltype(A[1])
blockeltype{T<:Real}(A::AbstractArray{T}) = T
blockeltype{T<:Real}(A::AbstractArray{NumDenom{T}}) = T
blockeltype(ND::Tuple) = blockeltype(ND[1])


#### MismatchData ####
"""
`MismatchData` is designed as an accessor for a file format, although
it can be used in-memory as well. The most important fields are:

- `blocksize`: a vector of length `ndims`, equal to `2*maxshift+1`
- `gridsize`: a vector of length `ndims`, giving the size of the grid
  of "control points"
- `data`: an array with the dimensions
  `[blocksize,2,gridsize,nstacks]`, holding the mismatch `num/denom`
  data.  For a 2d registration, this will be a 5-dimensional array
  even if you only have two images you're registering.
- `stacks`: a vector of stack indexes from the original images
- `stack_base`: the stack index of the reference ("fixed") image
"""
type MismatchData{T,N}
    data::Array{T,N}
    stack_base::Int
    stacks::Vector{Int}
    blocksize::Vector{Int}
    gridsize::Vector{Int}
    normalization::Char
    key::Int32
end
MismatchData(data, stack_base, stacks, blocksize, gridsize, normalization) =
    MismatchData(data, stack_base, stacks, blocksize, gridsize, normalization, Int32(0))

"""
`mmd = MismatchData(nums, denoms; [normalization=:intensity],
[stack_base=-1])` constructs a `MismatchData` object from a single
fixed/moving image pair.
"""
function MismatchData(nums::AbstractArray, denoms::AbstractArray; normalization = :intensity, stack_base=-1)
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
    print(io, "  blocksize: ", mmd.blocksize')
    print(io, "  gridsize:  ", mmd.gridsize')
    print(io, "  normalization: ", mmd.normalization)
end

eltype{T}(mmd::MismatchData{T}) = T
ndims{T,N}(mmd::MismatchData{T,N}) = N


#### Utility functions ####

"""
`@nlinear N i` constructs a linear index from a `Base.Cartesian` "tuple".
"""
macro nlinear(N, A, i)
    sym = symbol(string(i)*"_"*string(N))
    ex = :($sym-1)
    for n = N-1:-1:1
        sym = symbol(string(i)*"_"*string(n))
        ex = :(size($(esc(A)), $n)*$ex + $sym - 1)
    end
    :($ex+1)
end

"""
`index = indminmismatch(num, denom, thresh)` returns the location of
the minimum value of what is effectively `num./denom`.  However, it
considers only those points for which `denom .> thresh`; moreover, it
will never choose an edge point.  `index` is a linear index into the
arrays.
"""
@generated function indminmismatch{T,N}(num::AbstractArray{T,N}, denom::AbstractArray{T,N}, thresh::Real)
    quote
        size(num) == size(denom) || error("size(num) = $(size(num)), but size(denom) = $(size(denom))")
        imin = length(num)>>1+1   # default is center of the array
        rmin = typemax(T)
        threshT = convert(T, thresh)
        @inbounds @nloops $N i d->2:size(num,d)-1 begin
            den = @nref $N denom i
            if den > threshT
                r = (@nref $N num i)/den
                if r < rmin
                    imin = @nlinear $N num i
                    rmin = r
                end
            end
        end
        imin
    end
end

"""
`ind2disp(out, blocksize, index)` takes `index`, interpreted as a
linear index into an array of size `blocksize`, and returns (in `out`)
the displacement of this location from the center (in Cartesian
coordinates).  `out` and `blocksize` must be of the same length.
"""
function ind2disp!(out::AbstractArray, blocksize, ind::Int)
    n = length(blocksize)
    length(out) == n || error("out has length $(length(out)) but blocksize has length $n.")
    s = blocksize[1]
    for i = 2:n-1
        s *= blocksize[i]
    end
    ind -= 1
    for i = n-1:-1:1
        c = div(ind, s)
        out[i+1] = c - ((blocksize[i+1]+1)>>1) + 1
        ind -= c*s
        s /= blocksize[i]
    end
    out[1] = ind - ((blocksize[1]+1)>>1) + 1
    out
end

"""
`disp = ind2disp(blocksize, index)` takes `index`, interpreted as a
linear index into an array of size `blocksize`, and returns
the displacement `disp` of this location from the center (in Cartesian
coordinates). See also `ind2disp!`.
"""
ind2disp(blocksize, ind::Int) = ind2disp!(Array(Int, length(blocksize)), blocksize, ind)

end
