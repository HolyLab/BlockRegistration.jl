__precompile__()

module CachedInterpolations

using Interpolations, Ratios, Base.Cartesian
using Interpolations: sqr

export CachedInterpolation, cachedinterpolators

"""
CachedInterpolations implements a performance enhancement for
quadratic interpolation of a large multidimensional array.  The first
`N` dimensions are interpolating, and the remainder are tiling
(`NoInterp`), so that one is computing interpolants like
```
    for i_2 = 1:size(P, 4), i_1 = 1:size(P, 3)
        y_1 = x_1[i_1, i_2]
        y_2 = x_2[i_1, i_2]
        B[i_1, i_2] = P[y_1, y_2, i_1, i_2]
    end
```
where `x_1`, `x_2` are floating-point indexes and `i_1`, `i_2` are
integer indexes.  A CachedInterpolation simulates an array-of-arrays
interface for this task, while in reality using only a single
underlying `P` array.

The performance enhancement comes from caching: when `P` is bigger
than the computer's memory, `P` may be a mmapped-file, and direct
access to values of `P` will therefore be limited by disk I/O. If one
has a task in which all elements of `B` have to be evaluated
repeatedly for different values of `y_1`, `y_2`, but that these values
often change by only a small amount from one iteration to the next
(e.g., in a descent-based optimization task), then often the
interpolation will be computed from the same underlying entries in `P`
and just use different interpolation coefficients.  Since the
interpolation is quadratic, it is therefore sufficient to cache a
3-by-3-by-... view of the interpolating dimensions of `P`, each
centered on the current `y_1, y_2`.

Create an array-of-interpolating arrays with the function
`cachedinterpolators`.
"""
CachedInterpolations

"""
A single `CachedInterpolation` represents a "movable"
3-by-3-by... view of `P[:, :, i_1, i_2]` for a specific `(i_1,
i_2)`. An array of these objects thus implements an array-of-arrays
interface. Create them with `cachedinterpolators`.
"""
type CachedInterpolation{T,N,M,O} <: AbstractInterpolation{T,N,BSpline{Quadratic{InPlace}},OnCell}
    coefs::Array{T,M}   # tiled array of 3x3x... buffers
    parent::Array{T,M}  # the overall array (`P` in the documentation above)
    center::NTuple{N,Int}  # rounded (y_1, y_2) of prev. eval for this tile
    tileindex::Int      # equivalent to sub2ind(size(P)[N+1:end], i_1, i_2, ...)
end

Base.size{T,N}(itp::CachedInterpolation{T,N})    = size(itp.parent)[1:N]
Base.size{T,N}(itp::CachedInterpolation{T,N}, d) = d <= N ? size(itp.parent, d) : 1

"""

`itps = cachedinterpolators(parent, N, [origin=(0,0,...)])` creates an
array-of-CachedInterpolation arrays from an underlying `parent` array,
where the first `N` dimensions of `parent` are interpolating.  The
equivalent of
```
    parent[y_1, y_2, i_1, i_2]
```
becomes
```
    itp = itps[i_1, i_2]
    itp[y_1, y_2]
```
and `itp` caches the needed values of `parent[:,:,i_1,i_2]` centered
around `y_1, y_2`.

Optionally specify the `origin` argument to offset the `y_i`
coordinates within a tile.  For example, one can mimic a
`CenterIndexedArray` when `size(parent, d)` is odd for the first `N`
dimensions and you supply `origin = (div(size(parent,1)+1, 2), ...)`.
"""
function cachedinterpolators{T,M}(parent::Array{T,M}, N, origin=ntuple(d->0,N))
    0 <= N <= M || error("N must be between 0 and $M")
    length(origin) == N || throw(DimensionMismatch("length(origin) = $(length(origin)) is inconsistent with $N interpolating dimensions"))
    sz3 = ntuple(d->d<=N ? 3 : size(parent,d), M)::NTuple{M,Int}
    buffer = Array(eltype(parent), sz3)
    sztiles = size(parent)[N+1:end]  # the tiling dimensions of parent
    # use an impossible initial value (post-offset by origin) to
    # ensure the first access will result in a cache miss
    center = ntuple(d->-1, N)
    _cachedinterpolators(buffer, parent, origin, center, sztiles)
end

# function-barriered to circumvent type-instability in sztiles
@noinline function _cachedinterpolators{T,N,M}(buffer::Array{T,M}, parent::Array{T,M}, origin::NTuple{N,Int}, center::NTuple{N,Int}, sztiles)
    itps = Array(CachedInterpolation{T,N,M,origin}, sztiles)
    for tileindex = 1:prod(sztiles)
        itps[tileindex] = CachedInterpolation{T,N,M,origin}(buffer, parent, center, tileindex)
    end
    itps
end

@generated function Base.getindex{T,N,M,O}(itp::CachedInterpolation{T,N,M,O}, xs::Number...)
    length(xs) == N || error("Must use $N indexes")
    ibsyms = [Symbol("ib_", d) for d = 1:N]
    ipsyms = [Symbol("ip_", d) for d = 1:N]
    cache_ex = :(itp.coefs[$(ibsyms...), itp.tileindex] = itp.parent[$(ipsyms...), itp.tileindex])
    IT = Tuple{ntuple(d->BSpline{Quadratic{InPlace}}, N)..., NoInterp}
    ixlast = Symbol("ix_", N+1)
    ixlast_ex = :($ixlast = itp.tileindex)
    quote
        $(Expr(:meta, :inline))
        @nexprs $N d->(ix_d = round(Int, xs[d]))
        @nexprs $N d->(fx_d = xs[d] - ix_d)
        center = @ntuple $N d->(ix_d + O[d])
        if center != itp.center
            # Copy the relevant portion from parent into buffer
            @nloops $N ib d->1:3 d->(ip_d = ib_d+center[d]-2) begin
                $cache_ex
            end
            itp.center = center
        end
        # Perform the quadratic interpolation
        @nexprs $N d->(ixm_d = 1)
        @nexprs $N d->(ix_d  = 2)
        @nexprs $N d->(ixp_d = 3)
        $ixlast_ex
        $(Interpolations.coefficients(IT, N+1))
        @inbounds ret = $(Interpolations.index_gen(IT, N+1))
        ret
    end
end

immutable CoefsWrapper{N,A}
    coefs::A
end

Base.size{N}(itp::CoefsWrapper{N}, d) = d <= N ? size(itp.coefs, d) : 1

# FIXME: this function cheats dangerously, because it does _not_
# update the cache. This is equivalent to the assumption you've called
# getindex for the current `(x_1, x_2, ...)` location before calling
# gradient!. If this is not true, you'll get wrong answers.
@generated function Interpolations.gradient!{T,N,M,O}(g::AbstractVector, A::CachedInterpolation{T,N,M,O}, ys::Number...)
    length(ys) == N || error("Must use $N indexes")
    IT = Tuple{ntuple(d->BSpline{Quadratic{InPlace}}, N)..., NoInterp}
    BS = Interpolations.BSplineInterpolation{T,N+1,Array{T,N},IT,OnCell,0}
    ex = Interpolations.gradient_impl(BS)
    quote
        xs = @ntuple $(N+1) d->(d <= $N ? ys[d] - round(Int, ys[d]) + 2 : A.tileindex)
        itp = CoefsWrapper{N,typeof(A.coefs)}(A.coefs)
        $ex
    end
end

end # module
