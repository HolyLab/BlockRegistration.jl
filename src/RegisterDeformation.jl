__precompile__()

module RegisterDeformation

using Images, AffineTransforms, Interpolations, FixedSizeArrays, RegisterUtilities
using Base.Cartesian
import Interpolations: AbstractInterpolation, AbstractExtrapolation

export
    # types
    AbstractDeformation,
    GridDeformation,
    WarpedArray,
    # functions
    arraysize,
    compose,
    eachknot,
    tform2deformation,
    translate,
    vecindex,
    vecgradient!,
    warp,
    warp!,
    warpgrid

typealias DimsLike Union{Vector{Int}, Dims}
typealias InterpExtrap Union{AbstractInterpolation,AbstractExtrapolation}
typealias Extrapolatable{T,N} Union{TransformedArray{T,N},AbstractExtrapolation{T,N}}

"""
# RegisterDeformation

A deformation (or warp) of space is represented by a function `ϕ(x)`.
For an image, the warped version of the image is specified by "looking
up" the pixel value at a location `ϕ(x) = x + u(x)`.  `u(x)` thus
expresses the displacement, in pixels, at position `x`.  Note that a
constant deformation, `u(x) = x0`, corresponds to a shift of the
*coordinates* by `x0`, and therefore a shift of the *image* in the
opposite direction.

In reality, deformations will be represented on a grid, and
interpolation is implied at locations between grid points. For a
deformation defined directly from an array, make it interpolating
using `ϕi = interpolate(ϕ)`.

The major functions/types exported by RegisterDeformation are:

    - `GridDeformation`: create a deformation
    - `tform2deformation`: convert an `AffineTransform` to a deformation
    - `ϕ_old(ϕ_new)` and `compose`: composition of two deformations
    - `warp` and `warp!`: deform an image
    - `WarpedArray`: create a deformed array lazily
    - `warpgrid`: visualize a deformation

"""
RegisterDeformation

abstract AbstractDeformation{T,N}
Base.eltype{T,N}(::Type{AbstractDeformation{T,N}}) = T
Base.ndims{T,N}(::Type{AbstractDeformation{T,N}}) = N

"""
`ϕ = GridDeformation(u::Array{FixedVector}, dims)` creates a
deformation `ϕ` for an array of size `dims`.  `u` specifies the
"pixel-wise" displacement at a series of control points that are
evenly-spaced over the domain specified by `dims` (i.e., using
knot-vectors `linspace(1,dims[d],size(u,d))`).  In particular, each
corner of the array is the site of one control point.

`ϕ = GridDeformation(u::Array{FixedVector}, knots)` specifies the
knot-vectors manually. `u` must have dimensions equal to
`(length(knots[1]), length(knots[2]), ...)`.

`ϕ = GridDeformation(u::Array{T<:Real}, ...)` constructs the
deformation from a "plain" array. For a deformation in `N` dimensions,
`u` must have `N+1` dimensions, where the first dimension corresponds
to the displacement along each axis (and therefore `size(u,1) == N`).

Finally, `ϕ = GridDeformation((u1, u2, ...), ...)` allows you to
construct the deformation using an `N`-tuple of shift-arrays, each
with `N` dimensions.
"""
immutable GridDeformation{T,N,A<:AbstractArray,L} <: AbstractDeformation{T,N}
    u::A
    knots::NTuple{N,L}

    function GridDeformation{FV<:FixedVector}(u::AbstractArray{FV,N},
                                              knots::NTuple{N,L})
        length(FV) == N || throw(DimensionMismatch("Dimensionality $(length(FV)) must match $N knot vectors"))
        for d = 1:N
            size(u, d) == length(knots[d]) || error("size(u) = $(size(u)), but the knots specify a grid of size $(map(length, knots))")
        end
        new(u, knots)
    end
    function GridDeformation{FV<:FixedVector}(u::ScaledInterpolation{FV,N})
        new(u, u.ranges)
    end
end

# Ambiguity avoidance
function GridDeformation{FV<:FixedVector,N}(u::AbstractArray{FV,N},
                                            knots::Tuple{})
    error("Cannot supply an empty knot tuple")
end

# With knot ranges
function GridDeformation{FV<:FixedVector,N,L<:AbstractVector}(u::AbstractArray{FV,N},
                                                              knots::NTuple{N,L})
    T = eltype(FV)
    length(FV) == N || throw(DimensionMismatch("$N-dimensional array requires Vec{$N,T}"))
    GridDeformation{T,N,typeof(u),L}(u, knots)
end

# With image spatial size
function GridDeformation{FV<:FixedVector,N,L<:Integer}(u::AbstractArray{FV,N},
                                                       dims::NTuple{N,L})
    T = eltype(FV)
    length(FV) == N || throw(DimensionMismatch("$N-dimensional array requires Vec{$N,T}"))
    knots = ntuple(d->linspace(1,dims[d],size(u,d)), N)
    GridDeformation{T,N,typeof(u),typeof(knots[1])}(u, knots)
end

# Construct from a plain array
function GridDeformation{T<:Number,N}(u::Array{T}, knots::NTuple{N})
    ndims(u) == N+1 || error("Need $(N+1) dimensions for $N-dimensional deformations")
    size(u,1) == N || error("First dimension of u must be of length $N")
    uf = convert_to_fixed(u)
    GridDeformation(uf, knots)
end

# Construct from a (u1, u2, ...) tuple
function GridDeformation{N}(u::NTuple{N}, knots::NTuple{N})
    ndims(u[1]) == N || error("Need $N dimensions for $N-dimensional deformations")
    ua = permutedims(cat(N+1, u...), (N+1,(1:N)...))
    uf = convert_to_fixed(ua)
    GridDeformation(uf, knots)
end

function convert_to_fixed{T}(u::Array{T})
    N = size(u,1)
    if isbits(T)
        uf = reinterpret(Vec{N,T}, u, Base.tail(size(u)))
    else
        uf = Array(Vec{N,T}, Base.tail(size(u)))
        for i = 1:length(uf)
            uf[i] = Vec(u[:,i]...)
        end
    end
    uf
end

function GridDeformation{FV<:FixedVector}(u::ScaledInterpolation{FV})
    N = length(FV)
    ndims(u) == N || throw(DimensionMismatch("Dimension $(ndims(u)) incompatible with vectors of length $N"))
    GridDeformation{eltype(FV),N,typeof(u),typeof(u.ranges[1])}(u)
end

# # TODO: flesh this out
# immutable VoroiDeformation{T,N,Vu<:AbstractVector,Vc<:AbstractVector} <: AbstractDeformation{T,N}
#     u::Vu
#     centers::Vc
#     simplexes::??
# end
# (but there are several challenges, including the lack of a continuous gradient)

function Interpolations.interpolate{BC}(ϕ::GridDeformation, ::Type{BC})
    itp = scale(interpolate(ϕ.u, BSpline{Quadratic{BC}}, OnCell), ϕ.knots...)
    GridDeformation(itp)
end
Interpolations.interpolate(ϕ::GridDeformation) = interpolate(ϕ, Flat)

function Interpolations.interpolate!{BC}(ϕ::GridDeformation, ::Type{BC})
    itp = scale(interpolate!(ϕ.u, BSpline{Quadratic{BC}}, OnCell), ϕ.knots...)
    GridDeformation(itp)
end
Interpolations.interpolate!(ϕ::GridDeformation) = interpolate!(ϕ, InPlace)

Interpolations.interpolate{ T,N,A<:AbstractInterpolation}(ϕ::GridDeformation{T,N,A}) = error("ϕ is already interpolating")

Interpolations.interpolate!{T,N,A<:AbstractInterpolation}(ϕ::GridDeformation{T,N,A}) = error("ϕ is already interpolating")

function vecindex{T,N,A<:AbstractInterpolation}(ϕ::GridDeformation{T,N,A}, x::FixedVector{N})
    x + vecindex(ϕ.u, x)
end

@generated function Base.getindex{T,N,A<:AbstractInterpolation}(ϕ::GridDeformation{T,N,A}, xs::Number...)
    length(xs) == N || throw(DimensionMismatch("$(length(xs)) indexes is not consistent with ϕ dimensionality $N"))
    xindexes = [:(xs[$d]) for d = 1:N]
    ϕxindexes = [:(xs[$d]+ux[$d]) for d = 1:N]
    quote
        ux = ϕ.u[$(xindexes...)]
        Vec($(ϕxindexes...))
    end
end

# Composition ϕ_old(ϕ_new(x))
function Base.call{T1,T2,N,A<:AbstractInterpolation}(
        ϕ_old::GridDeformation{T1,N,A}, ϕ_new::GridDeformation{T2,N})
    u, knots = ϕ_old.u, ϕ_old.knots
    if !isa(ϕ_new.u, AbstractInterpolation)
        ϕ_new.knots == knots || error("If knots are incommensurate, ϕ_new must be interpolating")
    end
    sz = map(length, knots)
    x = knot(knots, 1)
    out = _compose(u, ϕ_new, x, 1)
    ucomp = similar(u, typeof(out))
    for I in CartesianRange(sz)
        ucomp[I] = _compose(u, ϕ_new, knot(knots, I), I)
    end
    GridDeformation(ucomp, knots)
end

function _compose(u, ϕ_new, x, i)
    dx = lookup(ϕ_new.u, x, i)
    dx + vecindex(u, x+dx)
end

lookup(u::AbstractInterpolation, x, i) = vecindex(u, x)
lookup(u, x, i) = u[i]

@generated function knot{N}(knots::NTuple{N}, i::Integer)
    args = [:(knots[$d][s[$d]]) for d = 1:N]
    quote
        s = ind2sub(map(length, knots), i)
        Vec($(args...))
    end
end

@generated function knot{N}(knots::NTuple{N}, I)
    args = [:(knots[$d][I[$d]]) for d = 1:N]
    :(Vec($(args...)))
end

arraysize(knots::NTuple) = map(k->(x = extrema(k); convert(Int, x[2]-x[1]+1)), knots)

immutable KnotIterator{K,N}
    knots::K
    iter::CartesianRange{N}
end

eachknot(knots) = KnotIterator(knots, CartesianRange(map(length, knots)))

Base.start(ki::KnotIterator) = start(ki.iter)
Base.done(ki::KnotIterator, state) = done(ki.iter, state)
function Base.next(ki::KnotIterator, state)
    I, state = next(ki.iter, state)
    k = knot(ki.knots, I)
    k, state
end

"""
`ϕ_c = ϕ_old(ϕ_new)` computes the composition of two deformations,
yielding a deformation for which `ϕ_c(x) ≈ ϕ_old(ϕ_new(x))`.

`ϕ_c, g = compose(ϕ_old, ϕ_new)` also yields the gradient `g` of `ϕ_c`
with respect to `u_new`.  `g[i,j,...]` is the Jacobian matrix at grid
position `(i,j,...)`.

You can use `_, g = compose(identity, ϕ_new)` if you need the gradient
for when `ϕ_old` is equal to the identity transformation.
"""
function compose{T1,T2,N,A<:AbstractInterpolation}(
        ϕ_old::GridDeformation{T1,N,A}, ϕ_new::GridDeformation{T2,N})
    u, knots = ϕ_old.u, ϕ_old.knots
    ϕ_new.knots == knots || error("Not yet implemented for incommensurate knots")
    sz = map(length, knots)
    x = knot(knots, 1)
    out = _compose(u, ϕ_new, x, 1)
    ucomp = similar(u, typeof(out))
    TG = Mat{N,N,eltype(out)}
    g = Array(TG, size(u))
    gtmp = Array(typeof(out), N)
    eyeN = eye(TG)
    for I in CartesianRange(sz)
        x = knot(knots, I)
        dx = lookup(ϕ_new.u, x, I)
        y = x + dx
        ucomp[I] = dx + vecindex(u, y)
        vecgradient!(gtmp, u, y)
        g[I] = convert(TG, gtmp) + eyeN
    end
    GridDeformation(ucomp, knots), g
end

function compose{T,N}(f::Function, ϕ_new::GridDeformation{T,N})
    f == identity || error("Only the identity function is supported")
    ϕ_new, fill(eye(Mat{N,N,T}), size(ϕ_new.u))
end

### WarpedArray
"""
A `WarpedArray` `W` is an AbstractArray for which `W[x] = A[ϕ(x)]` for
some parent array `A` and some deformation `ϕ`.  The object is created
lazily, meaning that computation of the displaced values occurs only
when you ask for them explicitly.

Create a `WarpedArray` like this:

```
W = WarpedArray(A, ϕ)
```
where

- The first argument `A` is an `AbstractExtrapolation` that can be
  evaluated anywhere.  See the Interpolations package.
- ϕ is an `AbstractDeformation`
"""
type WarpedArray{T,N,A<:Extrapolatable,D<:AbstractDeformation} <: AbstractArray{T,N}
    data::A
    ϕ::D
end

# User already supplied an interpolatable ϕ
function WarpedArray{T,N,S,A<:AbstractInterpolation}(data::Extrapolatable{T,N},
                                                     ϕ::GridDeformation{S,N,A})
    WarpedArray{T,N,typeof(data),typeof(ϕ)}(data, ϕ)
end

# Create an interpolatable ϕ
function WarpedArray{T,N}(data::Extrapolatable{T,N}, ϕ::GridDeformation)
    itp = scale(interpolate(ϕ.u, BSpline{Quadratic{Flat}}, OnCell), ϕ.knots...)
    ϕ′ = GridDeformation(itp, ϕ.knots)
    WarpedArray{T,N,typeof(data),typeof(ϕ′)}(data, ϕ′)
end

WarpedArray(data, ϕ::GridDeformation) = WarpedArray(to_etp(data), ϕ)


Base.size(A::WarpedArray) = size(A.data)
Base.size(A::WarpedArray, i::Integer) = size(A.data, i)
Base.ndims{T,N}(A::WarpedArray{T,N}) = N
Base.eltype{T}(A::WarpedArray{T}) = T

@generated function Base.getindex{T,N}(W::WarpedArray{T,N}, x::Number...)
    length(x) == N || error("Must use $N indexes")
    getindex_impl(N)
end

function getindex_impl(N)
    indxx = [:(x[$d]) for d = 1:N]
    indxϕx = [:(ϕx[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        ϕx = W.ϕ[$(indxx...)]
        W.data[$(indxϕx...)]
    end
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
`ϕ = tform2deformation(tform, arraysize, gridsize)` constructs a deformation
`ϕ` from the affine transform `tform` suitable for warping arrays
of size `arraysize`.  The origin-of-coordinates for `tform` is the
center of the array, meaning that if `tform` is a pure rotation the
array "spins" around its center.  The array of grid points defining `ϕ` has
size specified by `gridsize`.  The dimensionality of `tform` must
match that specified by `arraysize` and `gridsize`.
"""
function tform2deformation{T,N}(tform::AffineTransform{T,N}, arraysize, gridsize)
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
    urs = reshape(u, N, gridsize...)
    knots = ntuple(d->linspace(1,arraysize[d],gridsize[d]), N)
    GridDeformation(urs, knots)
end

"""
`wimg = warp(img, ϕ)` warps the array `img` according to the
deformation `ϕ`.
"""
function warp(img, ϕ)
    wimg = WarpedArray(img, ϕ)
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
`warp!(dest, img, ϕ)` warps `img` using the deformation `ϕ`.  The result is stored in `dest`.
"""
function warp!(dest::AbstractArray, img::AbstractArray, ϕ)
    wimg = WarpedArray(to_etp(img), ϕ)
    warp!(dest, wimg)
end

"""
`warp!(dest, img, tform, ϕ)` warps `img` using a combination of the affine transformation `tform` followed by deformation with `ϕ`.  The result is stored in `dest`.
"""
function warp!(dest::AbstractArray, img::AbstractArray, A::AffineTransform, ϕ)
    wimg = WarpedArray(to_etp(img, A), ϕ)
    warp!(dest, wimg)
end

"""
`img = warpgrid(ϕ)` returns an image `img` that permits visualization
of the deformation `ϕ`.  The output is a warped rectangular grid with
nodes centered on the control points as specified by the knots of `ϕ`.
"""
function warpgrid(ϕ)
    imsz = map(x->convert(Int, last(x)), ϕ.knots)
    img = zeros(eltype(u), imsz)
    imsza = Any[imsz...]
    for idim = 1:ndims(img)
        indexes = map(s -> 1:s, imsza)
        indexes[idim] = round(Int, ϕ.knots[idim])
        img[indexes...] = 1
    end
    warp(img, ϕ)
end

# TODO?: do we need to return real values beyond-the-edge for a SubArray?
to_etp(img) = extrapolate(interpolate(data(img), BSpline{Linear}, OnGrid), convert(promote_type(eltype(img), Float32), NaN))

to_etp(itp::AbstractInterpolation) = extrapolate(itp, convert(promote_type(eltype(itp), Float32), NaN))

to_etp(etp::AbstractExtrapolation) = etp

to_etp(img, A::AffineTransform) = TransformedArray(to_etp(img), A)

# Extensions to Interpolations and FixedSizedArrays
@generated function vecindex{N}(A::AbstractArray, x::FixedVector{N})
    args = [:(x[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        getindex(A, $(args...))
    end
end

@generated function vecgradient!{N}(g, itp::AbstractArray, x::FixedVector{N})
    args = [:(x[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        gradient!(g, itp, $(args...))
    end
end

# Note this is a bit unsafe as it requires the user to specify C correctly
@generated function Base.convert{R,C,T}(::Type{Mat{R,C,T}}, v::Vector{Vec{R,T}})
    args = [:(v[$d]._) for d = 1:C]
    :(Mat{R,C,T}(($(args...),)))
end

end  # module
