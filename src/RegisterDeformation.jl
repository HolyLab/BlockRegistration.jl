__precompile__()

module RegisterDeformation

using Images, AffineTransforms, Interpolations, RegisterUtilities
using Base.Cartesian
import Interpolations: AbstractInterpolation, AbstractExtrapolation

export WarpedArray, tform2u, translate, uarray2coords, ucoords2array, warp, warp!, warpgrid

typealias DimsLike Union(Vector{Int}, Dims)
typealias InterpExtrap Union(AbstractInterpolation,AbstractExtrapolation)
typealias Extrapolatable{T,N} Union(TransformedArray{T,N},AbstractExtrapolation{T,N})

"""
# RegisterDeformation

A deformation (or warp) of space is represented by a function `u(x)`,
where pixel values of an image `img` are "looked up" at a location `x
-> x + u(x)`.  `u(x)` thus expresses the displacement, in pixels, at
position `x`.  Note that a constant deformation, `u(x) = x0`,
corresponds to a shift of the *coordinates* by `x0`, and therefore a
shift of the *image* in the opposite direction.

In reality, deformations will be represented on a grid, and
interpolation is implied at locations between grid points. Given an
image of size `imsize`, the "control points" of the grid are assumed
to be equally spaced, with the first and last at the corners of the
image.

There are two main representations of `u`:

- Packed single-array representation: `u[dim, i, j]` is the
  displacement along coordinate `dim` for grid point `i,j`. `u[:,i,j]`
  is the vector of displacements.
- Multi-coordinate representation: the deformation is expressed as a
  collection of arrays, `u1, ..., ud` in `d` dimensions. `u1[i,j]` is
  the displacement along coordinate 1 for grid point `i,j`.

For image registration, deformations are closely coupled to the
mismatch.  `u[:,i,j,...]` corresponds to a particular location in the
`i,j,...` block of the `nums`, `denoms` arrays. For example, when
`u[:,i,j,...] = 0`, it corresponds to the element in the center; when
`u` has fractional values, the mismatch will be interpolated. A key
constraint is that, for the mismatch to be well-defined at a
particular `u`, we require `|u| <= maxshift-0.5`, where the 0.5 arises
from the need to perform quadratic interpolation. When `u` does not
satisfy this condition, you may trigger an error indicating that the
mismatch does not have a finite value.

The major functions/types exported by RegisterDeformation are:

    - `warp` and `warp!`: deform an image
    - `WarpedArray`: create a deformed array lazily
    - `warpgrid`: visualize a deformation
    - `tform2u`: convert an `AffineTransform` to a deformation
    - `uarray2coords` and `ucoords2array`: convert between the two represenations of `u`

See also: `compose_u` for composition of two deformations.
"""
RegisterDeformation

### WarpedArray
"""
A `WarpedArray` `W` is an AbstractArray for which `W[x,y,...] = A[g_x(x,y,...), g_y(x,y,...), ...]` for some parent array `A` and some deformation `(g_x, g_y, ...)`.  The object is created lazily, meaning that computation of the displaced values occurs only when you ask for them explicitly.

Create a `WarpedArray` like this:

```
W = WarpedArray(A, (u_x, u_y, ...))
```
where

- The first argument `A` is an `AbstractExtrapolation` that can be
  evaluated anywhere.  See the Interpolations package.
- The second argument, `(u_x,u_y,...)`, specifies the deformation. It
  must be encoded as a difference from the identity deformation (i.e.,
  no deformation), `g_x(x,y,...) = x + u_x(x,y,...)`.  `u_i` expresses
  the number of pixels of shift along dimension `i` at the evaluation
  location.  Each `u_i` is supplied either as an
  `AbstractInterpolation` or as an N-dimensional array which is
  assumed to span "corner-to-corner" the whole array `A`. In the
  latter case, the `u_i` will use quadratic interpolation.

"""
type WarpedArray{T,N,U<:InterpExtrap,A<:Extrapolatable} <: AbstractArray{T,N}
    data::A
    u::NTuple{N,U}
end

# Ambiguity resolution
function WarpedArray{T,N}(p::Extrapolatable{T,N}, u::Tuple{})
    if N == 0
        # kinda dumb
        return WarpedArray{T,N,Interpolations.BSplineInterpolation{Float64,0,Float64,BSpline{Linear},OnGrid,0},typeof(p)}(p, u)
    end
    error("Dimensionality $N does not match 0-dimensional u")
end

WarpedArray{T,N,U<:InterpExtrap}(p::Extrapolatable{T,N}, u::NTuple{N,U}) =
    WarpedArray{T,N,U,typeof(p)}(p, u)

function WarpedArray{T,N,UT<:AbstractFloat}(p::Extrapolatable{T,N}, u::NTuple{N,Array{UT,N}})
    uq = map(A->interpolate(A, BSpline{Quadratic{Flat}}, OnCell), u)
    WarpedArray{T,N,typeof(uq[1]),typeof(p)}(p, uq)
end

#WarpedArray{N,U<:InterpExtrap}(p, u::NTuple{N,U}) = WarpedArray(to_etp(p), u)

function WarpedArray(p::Extrapolatable, u::AbstractArray)
    ucoords = uarray2coords(u)
    WarpedArray(p, ucoords)
end

WarpedArray(p, u) = WarpedArray(to_etp(p), u)


Base.size(A::WarpedArray) = size(A.data)
Base.size(A::WarpedArray, i::Integer) = size(A.data, i)
Base.ndims{T,N}(A::WarpedArray{T,N}) = N
Base.eltype{T}(A::WarpedArray{T}) = T

@generated function Base.getindex{T,N}(W::WarpedArray{T,N}, x::Number...)
    length(x) == N || error("Must use $N indexes")
    getindex_impl(N)
end

function getindex_impl(N)
    y_syms      = [:($(symbol("y_",d))) for d = 1:N]
    y_conv      = [:(n = size(W.data,$d); m = length(W.u[$d]); $(y_syms[d]) = ((m-1)*x[$d] - m + n)/(n-1)) for d = 1:N]
    index_exprs = [:(x[$d] + W.u[$d][$(y_syms...)]) for d = 1:N]
    meta = Expr(:meta, :inline)
    :($meta; $(y_conv...); W.data[$(index_exprs...)])
end

getindex!(dest, W::WarpedArray, coords...) = Base._unsafe_getindex!(dest, Base.LinearSlow(), W, coords...)

"""
`Atrans = translate(A, displacement)` shifts `A` by an amount
specified by `displacement`.  Specifically, in simple cases `Atrans[i,
j, ...] = A[i+displacement[1], j+displacement[2], ...]`.  More
generally, `displacement` is applied only to the spatial coordinates
of `A`; if `A` is an `Image`, dimensions marked as time or color are
unaffected.

`NaN` is filled in for any missing pixels.
"""
function translate(A::AbstractArray, displacement::DimsLike)
    disp = zeros(Int, ndims(A))
    disp[coords_spatial(A)] = displacement
    indx = UnitRange{Int}[ (1:size(A,i))+disp[i] for i = 1:ndims(A) ]
    get(A, indx, NaN)
end

"""
`u = tform2u(tform, arraysize, gridsize)` constructs a deformation
grid `u` from the affine transform `tform` suitable for warping arrays
of size `arraysize`.  The origin-of-coordinates for `tform` is the
center of the array, meaning that if `tform` is a pure rotation the
array "spins" around its center.  The array of grid points in `u` has
size specified by `gridsize`.  The dimensionality of `tform` must
match that specified by `arraysize` and `gridsize`.
"""
function tform2u{T,N}(tform::AffineTransform{T,N}, arraysize, gridsize)
    if length(arraysize) != N || length(gridsize) != N
        error("Dimensionality mismatch")
    end
    A = tform.scalefwd - eye(N)   # this will compute the difference
    ngrid = prod(gridsize)
    u = Array(T, N, ngrid)
    asz = [arraysize...]
    s = (asz.-1)./([gridsize...].-1)
    k = 0
    center = (asz.-1)/2  # adjusted for unit-offset
    for c in Counter(gridsize)
        x = (c.-1).*s - center
        u[:,k+=1] = A*x+tform.offset
    end
    return reshape(u, N, gridsize...)
end

"""
`(u1,u2...) = uarray2coords(u)` converts the array representation of
`u` (of size `(ndims, sz...)`) into a `ndims` tuple of arrays of size
`sz`.
"""
function uarray2coords(u)
    nd = size(u,1)
    if size(u, nd+2) > 1
        error("This is not a single image pair, it's a sequence")
    end
    rng = ntuple(i->1:size(u,i+1), size(u,1))
    ucoords = ntuple(i->squeeze(u[i,rng...], 1), size(u,1))
end

"""
`u = ucoords2array((u1,u2...))` converts the tuple-representation of
of a deformation into an array `u` of size `(ndims, sz...)`.
"""
function ucoords2array(ucoords::Tuple)
    nd = length(ucoords)
    gridsize = size(ucoords[1])
    rng = ntuple(i->1:gridsize[i], nd)
    uarray = Array(eltype(ucoords[1]), nd, gridsize...)
    for i = 1:prod(gridsize), idim = 1:nd
        uarray[idim, i] = ucoords[idim][i]
    end
    uarray
end

"""
`wimg = warp(img, u)` warps the array `img` according to the
deformation `u`, represented as an array.

`wimg = warp(img, ucoords...)` parametrizes the deformation as a
tuple-of-arrays.
"""
function warp(img, ucoords...)
    if length(ucoords) == 1 && ndims(ucoords[1]) == ndims(img)+1
        ucoords = ucoords[1]
    end
    wimg = WarpedArray(img, ucoords)
    dest = similar(img)
    warp!(dest, wimg)
end

"""
`warp!(dest, w::WarpedArray)` instantiates a `WarpedArray` in the output `dest`.
"""
function warp!(dest::AbstractArray, wimg::WarpedArray)
    for I in CartesianRange(size(wimg))
        dest[I] = wimg[I]
    end
    dest
end


"""
`warp!(dest, img, ucoords...)` warps `img` using the deformation `ucoords`.  The result is stored in `dest`.
"""
function warp!(dest::AbstractArray, img::AbstractArray, ucoords...)
    wimg = WarpedArray(to_etp(img), ucoords)
    warp!(dest, wimg)
end

"""
`warp!(dest, img, tform, ucoords...)` warps `img` using a combination of the affine transformation `tform` followed by deformation with `ucoords`.  The result is stored in `dest`.
"""
function warp!(dest::AbstractArray, img::AbstractArray, A::AffineTransform, ucoords...)
    wimg = WarpedArray(to_etp(img, A), ucoords)
    warp!(dest, wimg)
end

"""
`img = warpgrid(u, imsize; [normalized=false])` returns an image `img`
that permits visualization of the deformation `u` (which must be in
array representation).  The output is a warped rectangular grid with
nodes centered on the control points as specified by the size of `u`
(specifying the `gridsize`) and `imsize` (specifying the size of the
fixed/moving images and `img`).

If `normalize==true`, the values in `u` are interpreted as a fraction
of the distance to the adjacent control point (i.e, scaled by the
block size).
"""
function warpgrid(u::AbstractArray, imsz; normalized::Bool=false)
    length(imsz) == size(u,1) || throw(DimensionMismatch("u is for $(size(u,1)) dimensions, but the image has $(length(imsz)) dimensions"))
    img = zeros(eltype(u), imsz)
    imsza = Any[imsz...]
    for idim = 1:ndims(img)
        x = round(Int, linspace(1, imsz[idim], size(u, idim+1)))
        indexes = map(s -> 1:s, imsza)
        indexes[idim] = x
        img[indexes...] = 1
    end
    if normalized
        blocksize = zeros(ndims(img))
        for idim = 1:ndims(img)
            gsz = size(u,idim+1)
            blocksize[idim] = imsz[idim]/(gsz > 1 ? gsz-1 : gsz)
        end
        u = u .* blocksize
    end
    warp(img, u)
end

to_etp(img) = extrapolate(interpolate(data(img), BSpline{Linear}, OnGrid), convert(promote_type(eltype(img), Float32), NaN))

to_etp(itp::AbstractInterpolation) = extrapolate(itp, convert(promote_type(eltype(itp), Float32), NaN))

to_etp(etp::AbstractExtrapolation) = etp

to_etp(img, A::AffineTransform) = TransformedArray(to_etp(img), A)

end
