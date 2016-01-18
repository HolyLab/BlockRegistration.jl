__precompile__()

module RegisterDeformation

using Images, AffineTransforms, Interpolations, ColorTypes, FixedSizeArrays, HDF5, JLD, ProgressMeter
using RegisterUtilities
using Base: Cartesian, tail
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
    griddeformations,
    medfilt,
    tform2deformation,
    tinterpolate,
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
Base.eltype{D<:AbstractDeformation}(::Type{D}) = eltype(super(D))
Base.ndims{D<:AbstractDeformation}(::Type{D}) = ndims(super(D))
Base.eltype(d::AbstractDeformation) = eltype(typeof(d))
Base.ndims(d::AbstractDeformation) = ndims(typeof(d))

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
    GridDeformation{T,N,typeof(u),LinSpace{Float64}}(u, knots)
end

# Construct from a plain array
function GridDeformation{T<:Number,N}(u::Array{T}, knots::NTuple{N})
    ndims(u) == N+1 || error("Need $(N+1) dimensions for $N-dimensional deformations")
    size(u,1) == N || error("First dimension of u must be of length $N")
    uf = convert_to_fixed(Vec{N,T}, u, tail(size(u)))
    GridDeformation(uf, knots)
end

# Construct from a (u1, u2, ...) tuple
function GridDeformation{N}(u::NTuple{N}, knots::NTuple{N})
    ndims(u[1]) == N || error("Need $N dimensions for $N-dimensional deformations")
    ua = permutedims(cat(N+1, u...), (N+1,(1:N)...))
    uf = convert_to_fixed(ua)
    GridDeformation(uf, knots)
end

# When knots is a vector
GridDeformation{V<:AbstractVector}(u, knots::AbstractVector{V}) = GridDeformation(u, (knots...,))
GridDeformation{R<:Real}(u, knots::AbstractVector{R}) = GridDeformation(u, (knots,))

function GridDeformation{FV<:FixedVector}(u::ScaledInterpolation{FV})
    N = length(FV)
    ndims(u) == N || throw(DimensionMismatch("Dimension $(ndims(u)) incompatible with vectors of length $N"))
    GridDeformation{eltype(FV),N,typeof(u),typeof(u.ranges[1])}(u)
end

"""
`ϕs = griddeformations(u, knots)` constructs a vector `ϕs` of
seqeuential deformations.  The last dimension of the array `u` should
correspond to time; `ϕs[i]` is produced from `u[..., i]`.
"""
function griddeformations{N,T<:Number}(u::AbstractArray{T}, knots::NTuple{N})
    ndims(u) == N+2 || error("Need $(N+2) dimensions for a vector of $N-dimensional deformations")
    size(u,1) == N || error("First dimension of u must be of length $N")
    uf = RegisterDeformation.convert_to_fixed(Vec{N,T}, u, Base.tail(size(u)))
    griddeformations(uf, knots)
end

function griddeformations{N,FV<:FixedVector}(u::AbstractArray{FV}, knots::NTuple{N})
    ndims(u) == N+1 || error("Need $(N+1) dimensions for a vector of $N-dimensional deformations")
    length(FV) == N || throw(DimensionMismatch("Dimensionality $(length(FV)) must match $N knot vectors"))
    colons = ntuple(d->Colon(), N)::NTuple{N,Colon}
    [GridDeformation(slice(u, colons..., i), knots) for i = 1:size(u, N+1)]
end

Base.copy{T,N,A,L}(ϕ::GridDeformation{T,N,A,L}) = (u = copy(ϕ.u); GridDeformation{T,N,typeof(u),L}(u, copy(ϕ.knots)))

# # TODO: flesh this out
# immutable VoroiDeformation{T,N,Vu<:AbstractVector,Vc<:AbstractVector} <: AbstractDeformation{T,N}
#     u::Vu
#     centers::Vc
#     simplexes::??
# end
# (but there are several challenges, including the lack of a continuous gradient)

function Interpolations.interpolate(ϕ::GridDeformation, BC)
    itp = scale(interpolate(ϕ.u, BSpline(Quadratic(BC)), OnCell()), ϕ.knots...)
    GridDeformation(itp)
end
Interpolations.interpolate(ϕ::GridDeformation) = interpolate(ϕ, Flat())

function Interpolations.interpolate!(ϕ::GridDeformation, BC)
    itp = scale(interpolate!(ϕ.u, BSpline(Quadratic(BC)), OnCell()), ϕ.knots...)
    GridDeformation(itp)
end
Interpolations.interpolate!(ϕ::GridDeformation) = interpolate!(ϕ, InPlace())

Interpolations.interpolate{ T,N,A<:AbstractInterpolation}(ϕ::GridDeformation{T,N,A}) = error("ϕ is already interpolating")

Interpolations.interpolate!{T,N,A<:AbstractInterpolation}(ϕ::GridDeformation{T,N,A}) = error("ϕ is already interpolating")

function vecindex{T,N,A<:AbstractInterpolation}(ϕ::GridDeformation{T,N,A}, x::FixedVector{N})
    x + vecindex(ϕ.u, x)
end

@generated function Base.getindex{T,N,A<:AbstractInterpolation}(ϕ::GridDeformation{T,N,A}, xs::Number...)
    length(xs) == N || throw(DimensionMismatch("$(length(xs)) indexes is not consistent with ϕ dimensionality $N"))
    xindexes = [:(xs[$d]) for d = 1:N]
    ϕxindexes = [:(xs[$d]+ux[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        ux = ϕ.u[$(xindexes...)]
        Vec($(ϕxindexes...))
    end
end

# Composition ϕ_old(ϕ_new(x))
function Base.call{T1,T2,N,A<:AbstractInterpolation}(
        ϕ_old::GridDeformation{T1,N,A}, ϕ_new::GridDeformation{T2,N})
    uold, knots = ϕ_old.u, ϕ_old.knots
    if !isa(ϕ_new.u, AbstractInterpolation)
        ϕ_new.knots == knots || error("If knots are incommensurate, ϕ_new must be interpolating")
    end
    ucomp = _compose(uold, ϕ_new.u, knots)
    GridDeformation(ucomp, knots)
end

Base.call(ϕ_old::GridDeformation, ϕ_new::GridDeformation) =
    error("ϕ_old must be interpolating")

function _compose(uold, unew, knots)
    sz = map(length, knots)
    x = knot(knots, 1)
    out = _compose(uold, unew, x, 1)
    ucomp = similar(uold, typeof(out))
    for I in CartesianRange(sz)
        ucomp[I] = _compose(uold, unew, knot(knots, I), I)
    end
    ucomp
end

function _compose(uold, unew, x, i)
    dx = lookup(unew, x, i)
    dx + vecindex(uold, x+dx)
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
yielding a deformation for which `ϕ_c(x) ≈ ϕ_old(ϕ_new(x))`. `ϕ_old`
must be interpolating (see `interpolate(ϕ_old)`).

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
    unew = ϕ_new.u
    sz = map(length, knots)
    x = knot(knots, 1)
    out = _compose(u, unew, x, 1)
    ucomp = similar(u, typeof(out))
    TG = Mat{N,N,eltype(out)}
    g = Array(TG, size(u))
    gtmp = Array(typeof(out), N)
    eyeN = eye(TG)
    for I in CartesianRange(sz)
        x = knot(knots, I)
        dx = lookup(unew, x, I)
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

function medfilt{D<:AbstractDeformation}(ϕs::AbstractVector{D}, n)
    nhalf = n>>1
    2nhalf+1 == n || error("filter size must be odd")
    T = eltype(eltype(D))
    v = Array(T, ndims(D), n)
    vs = ntuple(d->slice(v, d, :), ndims(D))
    ϕ1 = copy(ϕs[1])
    ϕout = Array(typeof(ϕ1), length(ϕs))
    ϕout[1] = ϕ1
    _medfilt!(ϕout, ϕs, v, vs)  # function barrier due to instability of vs
end

@noinline function _medfilt!{N,T}(ϕout, ϕs, v, vs::NTuple{N,T})
    n = size(v,2)
    nhalf = n>>1
    tmp = Array(eltype(T), N)
    u1 = ϕout[1].u
    for i = 1+nhalf:length(ϕs)-nhalf
        u = similar(u1)
        for I in eachindex(ϕs[i].u)
            for j = -nhalf:nhalf
                utmp = ϕs[i+j].u[I]
                for d = 1:N
                    v[d, j+nhalf+1] = utmp[d]
                end
            end
            for d = 1:N
                tmp[d] = median!(vs[d])
            end
            u[I] = tmp
        end
        ϕout[i] = GridDeformation(u, ϕs[i].knots)
    end
    # Copy the beginning and end
    for i = 2:nhalf  # we did [1] in medfilt
        ϕout[i] = copy(ϕs[i])
    end
    for i = length(ϕs)-nhalf+1:length(ϕs)
        ϕout[i] = copy(ϕs[i])
    end
    ϕout
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
    itp = scale(interpolate(ϕ.u, BSpline(Quadratic(Flat())), OnCell()), ϕ.knots...)
    ϕ′ = GridDeformation(itp, ϕ.knots)
    WarpedArray{T,N,typeof(data),typeof(ϕ′)}(data, ϕ′)
end

WarpedArray(data, ϕ::GridDeformation) = WarpedArray(to_etp(data), ϕ)


Base.size(A::WarpedArray) = size(A.data)
Base.size(A::WarpedArray, i::Integer) = size(A.data, i)

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
    dest = similar(img, warp_type(img))
    warp!(dest, wimg)
end

warp_type{T<:AbstractFloat}(img::AbstractArray{T}) = T
warp_type{T<:Number}(img::AbstractArray{T}) = Float32
warp_type{C<:Colorant}(img::AbstractArray{C}) = warp_type(img, eltype(eltype(C)))
warp_type{C<:Colorant, T<:AbstractFloat}(img::AbstractArray{C}, ::Type{T}) = C
warp_type{C<:Colorant, T}(img::AbstractArray{C}, ::Type{T}) = base_colorant_type(C){Float32}

"""
`warp!(dest, src::WarpedArray)` instantiates a `WarpedArray` in the output `dest`.
"""
@generated function warp!{_,N}(dest::AbstractArray{_,N}, src::WarpedArray)
    ϕxindexes = [:(I[$d]+ux[$d]) for d = 1:N]
    quote
        size(dest) == size(src) || error("dest must have the same size as src")
        # Can use zip once MAX_TYPE_DEPTH gets bumped, see julia #13561
        destiter = CartesianRange(size(dest))
        deststate = start(destiter)
        for ux in eachvalue(src.ϕ.u)
            I, deststate = next(destiter, deststate)
            dest[I] = src.data[$(ϕxindexes...)]
        end
        dest
    end
end


"""
`warp!(dest, img, ϕ)` warps `img` using the deformation `ϕ`.  The
result is stored in `dest`.
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

`warp!(T, io, img, ϕs; [nworkers=1])` writes warped images to
disk. `io` is an `IO` object or HDF5/JLD dataset (the latter must be
pre-allocated using `d_create` to be of the proper size). `img` is an
image sequence, and `ϕs` is a vector of deformations, one per image in
`img`.  If `nworkers` is greater than one, it will spawn additional
processes to perform the deformation.

An alternative syntax is `warp!(T, io, img, uarray; [nworkers=1])`,
where `uarray` is an array of `u` values with `size(uarray)[end] ==
nimages(img)`.
"""
function warp!{T}(::Type{T}, dest::Union{IO,HDF5Dataset,JLD.JldDataset}, img, ϕs; nworkers=1)
    n = nimages(img)
    ssz = size(img)[coords_spatial(img)]
    if n == 1
        ϕ = extract1(ϕs, sdims(img), ssz)
        destarray = Array(T, ssz)
        warp!(destarray, img, ϕ)
        warp_write(dest, destarray)
        return nothing
    end
    checkϕdims(ϕs, sdims(img), n)
    if nworkers > 1
        return _warp!(T, dest, img, ϕs, nworkers)
    end
    destarray = Array(T, ssz)
    @showprogress 1 "Stacks:" for i = 1:n
        ϕ = extracti(ϕs, i, ssz)
        warp!(destarray, slice(img, "t", i), ϕ)
        warp_write(dest, destarray, i)
    end
    nothing
end

warp!{T,R<:Real}(::Type{T}, dest::Union{IO,HDF5Dataset,JLD.JldDataset}, img, u::AbstractArray{R}; kwargs...) = warp!(T, dest, img, convert_to_fixed(u); kwargs...)

warp!(dest::Union{HDF5Dataset,JLD.JldDataset}, img, u; nworkers=1) =
    warp!(eltype(dest), dest, img, u; nworkers=nworkers)

function _warp!{T}(::Type{T}, dest, img, ϕs, nworkers)
    n = nimages(img)
    ssz = size(img)[coords_spatial(img)]
    wpids = addprocs(nworkers)
    simg = Array(Any, 0)
    swarped = Array(Any, 0)
    rrs = Array(RemoteRef, 0)
    mydir = splitdir(@__FILE__)[1]
    for p in wpids
        remotecall_fetch(p, Main.eval, :(push!(LOAD_PATH, $mydir)))
        remotecall_fetch(p, Main.eval, :(using RegisterDeformation))
        push!(simg, SharedArray(eltype(img), ssz, pids=[myid(),p]))
        push!(swarped, SharedArray(T, ssz, pids=[myid(),p]))
    end
    nextidx = 0
    getnextidx() = nextidx += 1
    writing_mutex = RemoteRef()
    prog = Progress(n, 1, "Stacks:")
    @sync begin
        for i = 1:nworkers
            p = wpids[i]
            src = simg[i]
            warped = swarped[i]
            @async begin
                while (idx = getnextidx()) <= n
                    ϕ = extracti(ϕs, idx, ssz)
                    copy!(src, slice(img, "t", idx))
                    remotecall_fetch(p, warp!, warped, src, ϕ)
                    put!(writing_mutex, true)
                    warp_write(dest, warped, idx)
                    update!(prog, idx)
                    take!(writing_mutex)
                end
            end
        end
    end
    finish!(prog)
    nothing
end

warp_write(io::IO, destarray) = write(io, destarray)
function warp_write(io::IO, destarray, i)
    offset = (i-1)*length(destarray)*sizeof(eltype(destarray))
    seek(io, offset)
    write(io, destarray)
end
function warp_write(dest, destarray, i)
    colons = [Colon() for d = 1:ndims(destarray)]
    dest[colons..., i] = destarray
end

function extract1{V<:Vec}(u::AbstractArray{V}, N, ssz)
    if ndims(u) == N+1
        ϕ = GridDeformation(reshape(u, size(u)[1:end-1]), ssz)
    else
        ϕ = GridDeformation(u, ssz)
    end
    ϕ
end
extract1{D<:AbstractDeformation}(ϕs::Vector{D}, N, ssz) = ϕs[1]

function extracti{V<:Vec}(u::AbstractArray{V}, i, ssz)
    colons = [Colon() for d = 1:ndims(u)-1]
    GridDeformation(u[colons..., i], ssz)
end
extracti{D<:AbstractDeformation}(ϕs::Vector{D}, i, _) = ϕs[i]

function checkϕdims{V<:Vec}(u::AbstractArray{V}, N, n)
    ndims(u) == N+1 || error("u's dimensionality $(ndims(u)) is inconsistent with the number of spatial dimensions $N of the image")
    if size(u)[end] != n
        error("Must have one `u` slice per image")
    end
    nothing
end
checkϕdims{D<:AbstractDeformation}(ϕs::Vector{D}, N, n) = length(ϕs) == n || error("Must have one `ϕ` per image")


"""
`img = warpgrid(ϕ; [scale=1, showidentity=false])` returns an image
`img` that permits visualization of the deformation `ϕ`.  The output
is a warped rectangular grid with nodes centered on the control points
as specified by the knots of `ϕ`.

`scale` multiplies `ϕ.u`, effectively making the deformation stronger
(for `scale > 1`).  This can be useful if you are trying to visualize
subtle changes. If `showidentity` is `true`, the actual deformation is
"""
function warpgrid(ϕ; scale=1, showidentity::Bool=false)
    imsz = map(x->convert(Int, last(x)), ϕ.knots)
    img = zeros(Float32, imsz)
    imsza = Any[imsz...]
    for idim = 1:ndims(img)
        indexes = map(s -> 1:s, imsza)
        indexes[idim] = clamp(round(Int, ϕ.knots[idim]), 2, imsz[idim]-1)
        img[indexes...] = 1
    end
    if scale != 1
        ϕ = GridDeformation(scale*ϕ.u, ϕ.knots)
    end
    wimg = warp(img, ϕ)
    if showidentity
        n = ndims(img)+1
        return reinterpret(RGB{Float32}, permutedims(cat(n, wimg, img, wimg), (n,1:ndims(img)...)))
    end
    wimg
end

"""
`ϕs = tinterpolate(ϕsindex, tindex, nstack)` uses linear
interpolation/extrapolation in time to "fill out" to times `1:nstack`
a deformation defined intermediate times `tindex` . Note that
`ϕs[tindex] == ϕsindex`.
"""
function tinterpolate(ϕsindex, tindex, nstack)
    ϕs = Array(eltype(ϕsindex), nstack)
    # Before the first tindex
    k = 0
    for i in 1:tindex[1]-1
        ϕs[k+=1] = ϕsindex[1]
    end
    # Within tindex
    for i in 2:length(tindex)
        ϕ1 = ϕsindex[i-1]
        ϕ2 = ϕsindex[i]
        Δt = tindex[i] - tindex[i-1]
        for j = 0:Δt-1
            α = convert(eltype(eltype(ϕ1.u)), j/Δt)
            ϕs[k+=1] = GridDeformation((1-α)*ϕ1.u + α*ϕ2.u, ϕ2.knots)
        end
    end
    # After the last tindex
    for i in tindex[end]:nstack
        ϕs[k+=1] = ϕsindex[end]
    end
    return ϕs
end

# TODO?: do we need to return real values beyond-the-edge for a SubArray?
to_etp(img) = extrapolate(interpolate(data(img), BSpline(Linear()), OnGrid()), convert(promote_type(eltype(img), Float32), NaN))

to_etp(itp::AbstractInterpolation) = extrapolate(itp, convert(promote_type(eltype(itp), Float32), NaN))

to_etp(etp::AbstractExtrapolation) = etp

to_etp(img, A::AffineTransform) = TransformedArray(to_etp(img), A)

# JLD extensions
# Serializer for Array{FixedSizeArray{T,...}}
immutable ArrayFSASerializer{T,DF,DT}
    A::Array{T,DT}
end

function JLD.readas{T,DT}(serdata::ArrayFSASerializer{T,1,DT})
    sz = size(serdata.A)
    reinterpret(Vec{sz[1],T}, serdata.A, tail(sz))
end
function JLD.readas{T,DT}(serdata::ArrayFSASerializer{T,2,DT})
    sz = size(serdata.A)
    reinterpret(Mat{sz[1],sz[2],T}, serdata.A, tail(tail(sz)))
end

function JLD.writeas{FSA<:FixedArray}(A::Array{FSA})
    T = eltype(FSA)
    DF = ndims(FSA)
    ArrayFSASerializer{T,DF,DF+ndims(A)}(reinterpret(T, A, (size(FSA)..., size(A)...)))
end

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

function convert_to_fixed{T}(u::Array{T}, sz=size(u))
    N = sz[1]
    convert_to_fixed(Vec{N, T}, u, tail(sz))
end

# Unlike the one above, this is type-stable
function convert_to_fixed{T,N}(::Type{Vec{N,T}}, u::AbstractArray{T}, sz=tail(size(u)))
    if isbits(T)
        uf = reinterpret(Vec{N,T}, u, sz)
    else
        uf = Array(Vec{N,T}, sz)
        copy_ctf!(uf, u)
    end
    uf
end

@generated function copy_ctf!{N,T}(dest::Array{Vec{N,T}}, src::Array)
    exvec = [:(src[offset+$d]) for d=1:N]
    quote
        for i = 1:length(dest)
            offset = (i-1)*N
            dest[i] = Vec($(exvec...))
        end
        dest
    end
end

function convert_from_fixed{N,T}(uf::AbstractArray{Vec{N,T}}, sz=size(uf))
    if isbits(T) && isa(uf, Array)
        u = reinterpret(T, uf, (N, sz...))
    else
        u = Array(T, (N, sz...))
        for i = 1:length(uf)
            for d = 1:N
                u[d,i] = uf[i][d]
            end
        end
    end
    u
end

# Note this is a bit unsafe as it requires the user to specify C correctly
@generated function Base.convert{R,C,T}(::Type{Mat{R,C,T}}, v::Vector{Vec{R,T}})
    args = [:(v[$d]._) for d = 1:C]
    :(Mat{R,C,T}(($(args...),)))
end

end  # module
