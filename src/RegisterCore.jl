__precompile__()

module RegisterCore

using CenterIndexedArrays
using Base.Cartesian: @nloops, @nref, @ntuple
using Images, ColorTypes

using Compat

import Base: +, -, *, /
import Base: eltype, getindex, ndims, pointer, setindex!, show, size
import Base: unsafe_getindex
import CenterIndexedArrays: CenterIndexedArray
import Images: separate

export
    # types
    MismatchArray,
    NumDenom,
    ColonFun,
    # functions
    highpass,
    indmin_mismatch,
    maxshift,
    mismatcharrays,
    ratio,
    separate,
    paddedview,
    trimmedview

"""
# RegisterCore

`RegisterCore` contains low-level utilities for working with "mismatch
data," as well as a few miscellaneous utilities.

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

- `NumDenom` and `MismatchArray`: packed pair representation of
  `(num,denom)` mismatch data
- `separate`: splits a `NumDenom` array into its component `num,denom` arrays
- `indmin_mismatch`: find the location of the minimum mismatch
- `highpass`: highpass filter an image before performing registration

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
NumDenom(n::Gray, d::Gray) = NumDenom(gray(n), gray(d))
NumDenom(n::Gray, d) = NumDenom(gray(n), d)
NumDenom(n, d::Gray) = NumDenom(n, gray(d))
NumDenom(n, d) = NumDenom(promote(n, d)...)

(+)(p1::NumDenom, p2::NumDenom) = NumDenom(p1.num+p2.num, p1.denom+p2.denom)
(-)(p1::NumDenom, p2::NumDenom) = NumDenom(p1.num-p2.num, p1.denom-p2.denom)
(*)(n::Number, p::NumDenom) = NumDenom(n*p.num, n*p.denom)
(*)(p::NumDenom, n::Number) = n*p
(/)(p::NumDenom, n::Number) = NumDenom(p.num/n, p.denom/n)
Base.one{T}(::Type{NumDenom{T}}) = NumDenom(one(T),one(T))
Base.one(p::NumDenom) = one(typeof(p))
Base.zero{T}(::Type{NumDenom{T}}) = NumDenom(zero(T),zero(T))
Base.zero(p::NumDenom) = zero(typeof(p))
Base.promote_rule{T1,T2<:Number}(::Type{NumDenom{T1}}, ::Type{T2}) = NumDenom{promote_type(T1,T2)}
Base.eltype{T}(::Type{NumDenom{T}}) = T
Base.convert{T}(::Type{NumDenom{T}}, p::NumDenom{T}) = p
Base.convert{T}(::Type{NumDenom{T}}, p::NumDenom) = NumDenom{T}(p.num, p.denom)
Base.show(io::IO, p::NumDenom) = print(io, "NumDenom(", p.num, ",", p.denom, ")")
function Base.showcompact(io::IO, p::NumDenom)
    print(io, "NumDenom(")
    showcompact(io, p.num)
    print(io, ",")
    showcompact(io, p.denom)
    print(io, ")")
end

@compat const MismatchArray{ND<:NumDenom,N,A} = CenterIndexedArray{ND,N,A}

maxshift(A::MismatchArray) = A.halfsize

"""
`numdenom = MismatchArray(num, denom)` packs the array-pair
`(num,denom)` into a single `MismatchArray`.  This is useful
preparation for interpolation.
"""
function (::Type{M}){M<:MismatchArray}(num::AbstractArray, denom::AbstractArray)
    size(num) == size(denom) || throw(DimensionMismatch("num and denom must have the same size"))
    T = promote_type(eltype(num), eltype(denom))
    numdenom = CenterIndexedArray(NumDenom{T}, size(num))
    _packnd!(numdenom, num, denom)
end

function _packnd!(numdenom::AbstractArray, num::AbstractArray, denom::AbstractArray)
    Rnd, Rnum, Rdenom = eachindex(numdenom), eachindex(num), eachindex(denom)
    if Rnum == Rdenom
        for (Idest, Isrc) in zip(Rnd, Rnum)
            @inbounds numdenom[Idest] = NumDenom(num[Isrc], denom[Isrc])
        end
    elseif Rnd == Rnum
        for (Inum, Idenom) in zip(Rnum, Rdenom)
            @inbounds numdenom[Inum] = NumDenom(num[Inum], denom[Idenom])
        end
    else
        for (Ind, Inum, Idenom) in zip(Rnd, Rnum, Rdenom)
            @inbounds numdenom[Ind] = NumDenom(num[Inum], denom[Idenom])
        end
    end
    numdenom
end

function _packnd!(numdenom::CenterIndexedArray, num::CenterIndexedArray, denom::CenterIndexedArray)
    @simd for I in eachindex(num)
        @inbounds numdenom[I] = NumDenom(num[I], denom[I])
    end
    numdenom
end

# The next are mostly used just for testing
"""
`mms = mismatcharrays(nums, denoms)` packs array-of-arrays num/denom pairs as an array-of-MismatchArrays.

`mms = mismatcharrays(nums, denom)`, for `denom` a single array, uses the same `denom` array for all `nums`.
"""
function mismatcharrays{A<:AbstractArray,T<:Number}(nums::AbstractArray{A}, denom::AbstractArray{T})
    first = true
    local mms
    for i in eachindex(nums)
        num = nums[i]
        mm = MismatchArray(num, denom)
        if first
            mms = Array{typeof(mm)}(size(nums))
            first = false
        end
        mms[i] = mm
    end
    mms
end

function mismatcharrays{A1<:AbstractArray,A2<:AbstractArray}(nums::AbstractArray{A1}, denoms::AbstractArray{A2})
    size(nums) == size(denoms) || throw(DimensionMismatch("nums and denoms arrays must have the same number of apertures"))
    first = true
    local mms
    for i in eachindex(nums, denoms)
        mm = MismatchArray(nums[i], denoms[i])
        if first
            mms = Array{typeof(mm)}(size(nums))
            first = false
        end
        mms[i] = mm
    end
    mms
end

"""
`num, denom = separate(mm)` splits an `AbstractArray{NumDenom}` into separate
numerator and denominator arrays.
"""
function separate{T}(data::AbstractArray{NumDenom{T}})
    num = Array{T}(size(data))
    denom = similar(num)
    for I in eachindex(data)
        nd = data[I]
        num[I] = nd.num
        denom[I] = nd.denom
    end
    num, denom
end

function separate(mm::MismatchArray)
    num, denom = separate(mm.data)
    CenterIndexedArray(num), CenterIndexedArray(denom)
end

function separate{M<:MismatchArray}(mma::AbstractArray{M})
    T = eltype(eltype(M))
    nums = Array{CenterIndexedArray{T,ndims(M)}}(size(mma))
    denoms = similar(nums)
    for (i,mm) in enumerate(mma)
        nums[i], denoms[i] = separate(mm)
    end
    nums, denoms
end

"""
`r = ratio(mm, thresh, [fillval])` returns an array with the ratio
`num/denom` at each location. `fillval` is used everywhere where
`denom < thresh`, and `fillval`'s type determines the type of the
output. The default is NaN.
"""
function ratio{T}(mm::MismatchArray, thresh, fillval::T)
    out = CenterIndexedArray(T, size(mm))
    for I in eachindex(mm)
        nd = mm[I]
        out[I] = ratio(nd, thresh, fillval)
    end
    out
end
ratio(mm::MismatchArray, thresh) = ratio(mm, thresh, convert(eltype(eltype(mm)), NaN))
@inline ratio{T}(nd::NumDenom{T}, thresh, fillval=convert(T,NaN)) = nd.denom < thresh ? fillval : nd.num/nd.denom

ratio{T<:Real}(r::CenterIndexedArray{T}, thresh, fillval=convert(T,NaN)) = r

(::Type{M}){M<:MismatchArray,T}(::Type{T}, dims) = CenterIndexedArray(NumDenom{T}, dims)
(::Type{M}){M<:MismatchArray,T}(::Type{T}, dims...) = CenterIndexedArray(NumDenom{T}, dims)

function Base.copy!(M::MismatchArray, nd::Tuple{AbstractArray, AbstractArray})
    num, denom = nd
    size(M) == size(num) == size(denom) || error("all sizes must match")
    for (IM, Ind) in zip(eachindex(M), eachindex(num))
        M[IM] = NumDenom(num[Ind], denom[Ind])
    end
    M
end


#### Utility functions ####

"""
`index = indmin_mismatch(numdenom, thresh)` returns the location of
the minimum value of what is effectively `num./denom`.  However, it
considers only those points for which `denom .> thresh`; moreover, it
will never choose an edge point.  `index` is a CartesianIndex into the
arrays.
"""
@generated function indmin_mismatch(numdenom::MismatchArray, thresh::Real)
    N = ndims(numdenom)
    T = eltype(eltype(numdenom))
    icenter = ntuple(d->0, N)
    quote
        imin = $icenter   # default is center of the array
        rmin = typemax($T)
        threshT = convert($T, thresh)
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

function indmin_mismatch{T<:Number}(r::CenterIndexedArray{T})
    ind = ind2sub(size(r), indmin(r.data))
    indctr = map(d->ind[d]-(size(r,d)+1)>>1, (1:ndims(r)...))
    CartesianIndex(indctr)
end

### Miscellaneous

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
    if any(isinf, sigma)
        datahp = convert(Array{T,ndims(data)}, data)
    else
        datahp = data - imfilter(T, data, KernelFactors.IIRGaussian(T, (sigma...,)), NA())
    end
    datahp[datahp .< 0] = 0  # truncate anything below 0
    datahp
end
highpass{T<:AbstractFloat}(data::AbstractArray{T}, sigma) = highpass(T, data, sigma)
highpass(data::AbstractArray, sigma) = highpass(Float32, data, sigma)


"""
`Apad = paddedview(A)`, for a SubArray `A`, returns a SubArray that
extends to the full parent along any non-sliced dimensions of the
parent.

See also `trimmedview`.
"""
paddedview(A::SubArray) = _paddedview(A, (), (), A.indexes...)
if VERSION < v"0.6.0-dev"
    _paddedview{T,N,P,I}(A::SubArray{T,N,P,I}, newindexes, newsize) =
        SubArray(A.parent, newindexes, newsize)
else
    _paddedview{T,N,P,I}(A::SubArray{T,N,P,I}, newindexes, newsize) =
        SubArray(A.parent, newindexes)
end
@inline function _paddedview(A, newindexes, newsize, index, indexes...)
    d = length(newindexes)+1
    _paddedview(A, (newindexes..., pdindex(A.parent, d, index)), pdsize(A.parent, newsize, d, index), indexes...)
end
pdindex(A, d, i::Colon) = i
pdindex(A, d, i::Real) = i
pdindex(A, d, i::UnitRange) = 1:size(A,d)
pdindex(A, d, i) = error("Cannot pad with an index of type ", typeof(i))

pdsize(A, newsize, d, i::Colon) = tuple(newsize..., size(A,d))
pdsize(A, newsize, d, i::Real) = newsize
pdsize(A, newsize, d, i::UnitRange) = tuple(newsize..., size(A,d))

"""
`B = trimmedview(Bpad, A::SubArray)` returns a SubArray `B` with
`size(B) = size(A)`, taking hints from the slice indexes of `A` about
the view region. `Bpad` must have the same size as `paddedview(A)`.

See also `paddedview`.
"""
function trimmedview(Bpad, A::SubArray)
    ndims(Bpad) == ndims(A) || throw(DimensionMismatch("dimensions $(ndims(Bpad)) and $(ndims(A)) of Bpad and A must match"))
    _trimmedview(Bpad, A.parent, 1, (), A.indexes...)
end
_trimmedview(Bpad, P, d, newindexes) = view(Bpad, newindexes...)
@inline _trimmedview(Bpad, P, d, newindexes, index::Real, indexes...) =
    _trimmedview(Bpad, P, d+1, newindexes, indexes...)
@inline function _trimmedview(Bpad, P, d, newindexes, index, indexes...)
    dB = length(newindexes)+1
    Bsz = size(Bpad, dB)
    Psz = size(P, d)
    Bsz == Psz || throw(DimensionMismatch("dimension $dB of Bpad has size $Bsz, should have size $Psz"))
    _trimmedview(Bpad, P, d+1, (newindexes..., index), indexes...)
end


# For faster and type-stable slicing
immutable ColonFun end
(::Type{ColonFun})(::Int) = Colon()

end
