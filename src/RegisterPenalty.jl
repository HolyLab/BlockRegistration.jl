# Penalty function for registration of a single image

__precompile__()

module RegisterPenalty

using Interpolations, StaticArrays, Base.Cartesian
using RegisterDeformation, RegisterCore, CenterIndexedArrays, CachedInterpolations
using RegisterDeformation: convert_from_fixed, convert_to_fixed

using Compat

export AffinePenalty, DeformationPenalty, penalty!, interpolate_mm!


"""
# RegisterPenalty

This module computes the total registration penalty, combining both
"data" (the mismatch between `fixed` and `moving` images) and
"regularization" (a penalty applied to deformations that do not fit
some pre-conceived notion of "goodness").

The main exported types/functions are:

- `AffinePenalty`: regularization that penalizes deviations from an affine transformation
- `penalty!`: compute the penalty
- `interpolate_mm!`: prepare the mismatch arrays for interpolation

"""
RegisterPenalty


@compat abstract type DeformationPenalty{T,N} end
Base.eltype{T,N}(::Type{DeformationPenalty{T,N}}) = T
Base.eltype{DP<:DeformationPenalty}(::Type{DP}) = eltype(supertype(DP))
Base.eltype(dp::DeformationPenalty) = eltype(typeof(dp))
Base.ndims{T,N}(::Type{DeformationPenalty{T,N}}) = N
Base.ndims{DP<:DeformationPenalty}(::Type{DP}) = ndims(supertype(DP))
Base.ndims(dp::DeformationPenalty) = ndims(typeof(dp))

"""
`p = penalty!(g, ϕ, ϕ_old, dp::DeformationPenalty, mmis, [keep])`
computes the total penalty (data penalty + regularization penalty)
associated with a deformation `ϕ`, mismatch data `mmis`, and
(optionally) an "old" deformation `ϕ_old` such that the total
deformation is the composition `ϕ_c = ϕ_old(ϕ)`. `mmis` should be with
respect to `ϕ` and not `ϕ_c`; this supports a workflow such as:

- Compute initial deformation `ϕ_0` that partially aligns `fixed` and `moving`
- Warp `moving` with `ϕ_0`
- Compute the *residual* mismatch between `fixed` and the warped version
  of `moving`
- Compute a `ϕ_1` which corrects the residual mismatch
- Final deformation is `ϕ_0(ϕ_1)`

This workflow requires that `ϕ_1` be determined by just the residual
mismatch, but also that `ϕ_1` be evaluated in terms of its impact on
the total regularization penalty (i.e., the composition `ϕ_0(ϕ_1)`).

In essence, this syntax for `penalty!` resolves to a sum of two calls:
```
    val =  penalty!(g, dp, ϕ_c, [g_c])        # regularization penalty
    val += penalty!(g, ϕ, mmis, keep)         # data penalty
```
`g_c` is the gradient of `ϕ_c` with respect to `ϕ.u`.  If `ϕ_old ==
identity`, then no composition is needed, `g_c` is the identity, and
`ϕ` is used directly.

Note that `ϕ_old`, if not equal to `identity`, must be
interpolating. In contrast, `ϕ` must not be interpolating.

`g` can be a single `Vector{T}` (for some number-type `T`), or can be
the same type and size as `ϕ.u`, i.e., an array of fixed-sized vectors
`SVector{N,T}`.

Further details are described in the help for the individual
`penalty!` calls.
"""
function penalty!(g, ϕ, ϕ_old, dp::DeformationPenalty, mmis::AbstractArray, keep = trues(size(mmis)))
    T = eltype(ϕ)
    val = zero(T)
    # Volume penalty
    if ϕ_old == identity
        val = penalty!(g, dp, ϕ)
    else
        ϕ_c, g_c = compose(ϕ_old, ϕ)
        val = penalty!(g, dp, ϕ_c, g_c)
    end
    if !isfinite(val)
        return val
    end
    # Data penalty
    val += penalty!(g, ϕ, mmis, keep)
    convert(T, val)
end

# Allow it to be called without StaticArrays
function penalty!{T<:Number}(g::Array{T}, ϕ, ϕ_old, dp::DeformationPenalty, mmis::AbstractArray, keep = trues(size(mmis)))
    gf = RegisterDeformation.convert_to_fixed(g, (ndims(dp), size(ϕ.u)...))
    penalty!(gf, ϕ, ϕ_old, dp, mmis, keep)
end


"""
`p = penalty!(gs, ϕs, ϕs_old, dp, λt, mmis, [keeps=trues(size(mmis))])`
evaluates the total penalty for temporal sequence of deformations
`ϕs`, using the temporal sequence of mismatch data `mmis`.  `λt` is
the temporal roughness penalty coefficient.
"""
function penalty!{D<:AbstractDeformation}(gs, ϕs::AbstractVector{D}, ϕs_old, dp::DeformationPenalty, λt::Number, mmis::AbstractArray, keep = trues(size(mmis)))
    ntimes = length(ϕs)
    size(mmis)[end] == ntimes || throw(DimensionMismatch("Number of deformations $ntimes does not agree with mismatch data of size $(size(mmis))"))
    s = _penalty!(gs, ϕs, ϕs_old, dp, mmis, keep, 1)
    for i = 2:ntimes
        isfinite(s) || break
        s += _penalty!(gs, ϕs, ϕs_old, dp, mmis, keep, i)
    end
    s + penalty!(gs, λt, ϕs)
end


function penalty!{T<:Number,D<:AbstractDeformation}(gs::Array{T}, ϕs::AbstractVector{D}, ϕs_old, dp::DeformationPenalty, λt::Number, mmis::AbstractArray, keep = trues(size(mmis)))
    gf = RegisterDeformation.convert_to_fixed(gs, (ndims(dp), size(first(ϕs).u)..., length(ϕs)))
    penalty!(gf, ϕs, ϕs_old, dp, λt, mmis, keep)
end

function _penalty!{T,N}(gs, ϕs, ϕs_old, dp::DeformationPenalty{T,N}, mmis, keeps, i)
    colons = ntuple(ColonFun, Val{N})
    indexes = (colons..., i)
    #mmi = view(mmis, indexes...)  # making these views runs afoul of inference limits
    mmi = mmis[indexes...]
    #keep = view(keeps, indexes...)
    keep = keeps[indexes...]
    calc_gradient = gs != nothing && !isempty(gs)
    g = calc_gradient ? view(gs, indexes...) : nothing
    if isa(ϕs_old, AbstractVector)
        penalty!(g, ϕs[i], ϕs_old[i], dp, mmi, keep)
    else
        penalty!(g, ϕs[i], ϕs_old, dp, mmi, keep)
    end
end


################
# Data penalty #
################

@compat const CenteredInterpolant{T,N,A<:AbstractInterpolation} = Union{MismatchArray{T,N,A}, CachedInterpolation{T,N}}

"""
`p = penalty!(g, ϕ, mmis, [keep=trues(size(mmis))])` computes the
data penalty, i.e., the total mismatch between `fixed`
and `moving` given the deformation `ϕ`.  The mismatch is encoded in
`mmis`, an array-of-MismatchArrays as computed via
RegisterMismatch. The `mmis[i]` arrays must be interpolating; see
`interpolate_mm!`.

`g` is pre-allocated storage for the gradient, and may be `nothing` or
empty if you want to skip gradient calculation.  **Note**: this
function *adds* to `g`; you should first fill `g` with zeros or call
the regularization penalty to initialize it.

The data penalty is defined as
```
        pnum_1 + pnum_2 + ... + pnum_n
   p = --------------------------------
        pden_1 + pden_2 + ... + pden_n
```

where each index `_i` refers to a single aperture, and each `p_i`
involves just `mmis[i]` and `ϕ.u[:,i]`.  `mmis[i]` must be
interpolating, so that it can be evaluated for fractional shifts.
"""
function penalty!{M<:CenteredInterpolant}(g, ϕ::AbstractDeformation, mmis::AbstractArray{M}, keep=trues(size(mmis)))
    penalty!(g, ϕ.u, mmis, keep)
end

function penalty!{Tg<:Number,M<:CenteredInterpolant}(g::Array{Tg}, ϕ::AbstractDeformation, mmis::AbstractArray{M}, keep=trues(size(mmis)))
    gf = RegisterDeformation.convert_to_fixed(g, (ndims(ϕ), size(ϕ.u)...))
    penalty!(gf, ϕ, mmis, keep)
end

function penalty!{Tu,Dim,M<:CenteredInterpolant}(g, u::AbstractArray{SVector{Dim,Tu}}, mmis::AbstractArray{M}, keep=trues(size(mmis)))
    # This "outer" function just handles the chain rule for computing the
    # total penalty and gradient. The "real" work is done by penalty_nd!.
    nblocks = length(mmis)
    length(u) == nblocks || error("u should have length $nblocks, but length(u) = $(length(u))")
    calc_gradient = g != nothing && !isempty(g)
    if calc_gradient
        if length(g) != length(u)
            error("length(g) = $(length(g)) but length(u) = $(length(u))")
        end
    end
    if calc_gradient
        gnd = similar(u, SVector{Dim,NumDenom{Tu}})
        nd = penalty_nd!(gnd, u, mmis, keep)
        N, D = nd.num, nd.denom
        invD = 1/D
        NinvD2 = N*invD*invD
        for i = 1:length(g)
            g[i] += _wsum(gnd[i], invD, -NinvD2)
        end
        return N*invD
    else
        nd = penalty_nd!(g, u, mmis, keep)
        N, D = nd.num, nd.denom
        return N/D
    end
end

# Computes pnum_i and pden_i and their gradients
function penalty_nd!(gnd, u::AbstractArray, mmis, keep)
    N = ndims(u)
    T = eltype(eltype(u))
    calc_grad = gnd != nothing && !isempty(gnd)
    mxs = maxshift(first(mmis))
    nd = NumDenom{T}(0,0)
    nanT = convert(T, NaN)
    local gtmp
    if calc_grad
        gtmp = Vector{NumDenom{T}}(N)
    end
    for iblock = 1:length(mmis)
        mmi = mmis[iblock]
        if !keep[iblock]
            if calc_grad
                gnd[iblock] = NumDenom{T}(0,0)
            end
            continue
        end
        # Check bounds
        dx = u[iblock]
        if !checkbounds_shift(dx, mxs)
            return NumDenom{T}(nanT,nanT)
        end
        # Evaluate the value
        nd += vecindex(mmi, dx)
        # Evaluate the gradient
        if calc_grad
            vecgradient!(gtmp, mmi, dx)
            gnd[iblock] = gtmp
        end
    end
    nd
end

penalty_nd!(gnd, u::AbstractInterpolation, mmis, keep) = error("ϕ must not be interpolating")

@generated function checkbounds_shift{N}(dx::SVector{N}, mxs)
    quote
        @nexprs $N d->(if abs(dx[d]) >= mxs[d]-0.5 return false end)
        true
    end
end

@generated function _wsum{N}(x::SVector{N}, cnum, cdenom)
    args = [:(cnum*x[$d].num + cdenom*x[$d].denom) for d = 1:N]
    quote
        SVector($(args...))
    end
end

##########################
# Regularization penalty #
##########################

### Affine-residual penalty
"""
`p = AffinePenalty(knots, λ)` initializes data defining an
affine-residual penalty. The penalty is defined in terms of a
deformation's `u` displacements as
```
    p =  λ*∑_i (u_i - a_i)^2
```
where `{a_i}` comes from a least-squares fit of `{u_i}` to an
affine transformation.

When the deformation is defined on a regular grid, `knots` can be an
NTuple of knot-vectors; otherwise, it should be an
`ndims`-by-`npoints` matrix that stores the knot locations in columns.
"""
type AffinePenalty{T,N} <: DeformationPenalty{T,N}
    F::Matrix{T}   # geometry data for the affine-residual penalty
    λ::T           # regularization coefficient

    (::Type{AffinePenalty{T,N}}){T,N}(F::Matrix{T}, λ::T, _) = new{T,N}(F, λ)

    function (::Type{AffinePenalty{T,N}}){T,N}(knots::NTuple{N}, λ)
        gridsize = map(length, knots)
        C = Matrix{Float64}(prod(gridsize), N+1)
        i = 0
        for I in CartesianRange(gridsize)
            C[i+=1, N+1] = 1
            for j = 1:N
                C[i, j] = knots[j][I[j]]  # I[j]
            end
        end
        F, _ = qr(C)
        new{T,N}(F, λ)
    end

    function (::Type{AffinePenalty{T,N}}){T,N}(knots::AbstractMatrix, λ)
        C = hcat(knots', ones(eltype(knots), size(knots, 2)))
        F, _ = qr(C)
        new{T,N}(F, λ)
    end
end

AffinePenalty{V<:AbstractVector,N}(knots::NTuple{N,V}, λ) = AffinePenalty{eltype(V),N}(knots, λ)
AffinePenalty{V<:AbstractVector}(knots::AbstractVector{V}, λ) = AffinePenalty{eltype(V),length(knots)}((knots...), λ)
AffinePenalty{T}(knots::AbstractMatrix{T}, λ) = AffinePenalty{T,size(knots,1)}(knots, λ)

Base.convert{T,N}(::Type{AffinePenalty{T,N}}, ap::AffinePenalty) = AffinePenalty{T,N}(convert(Matrix{T}, ap.F), convert(T, ap.λ), 0)

"""
`p = penalty!(g, dp::DeformationPenalty, ϕ_c, [g_c])` computes the
regularization penalty associated with a deformation `ϕ_c`. The `_c`
indicates "composed", and `g_c` is the gradient of that composition.
If your `ϕ` is not derived by composition with a previous deformation,
just supply it for `ϕ_c` and omit `g_c`.

The deformation penalty `dp` determines the type of penalty applied.
You can dispatch to your own penalty function, but the built-in is
for `dp::AffinePenalty`.

If `g` is non-empty, the gradient of the penalty with respect to `u`
will be computed.  If you write a custom `penalty!` function for a new
`DeformationPenalty`, it is your responsibility to set `g` in-place.
"""
function penalty!{T,N}(g, dp::AffinePenalty, ϕ_c::AbstractDeformation{T,N})
    penalty!(g, dp, ϕ_c.u)
end

function penalty!{T,N}(g, dp::AffinePenalty, u::AbstractArray{SVector{N,T},N})
    F, λ = dp.F, dp.λ
    if λ == 0
        if g != nothing && !isempty(g)
            fill!(g, zero(eltype(g)))
        end
        return λ * one(eltype(F)) * one(T)
    end
    n = length(u)
    U = convert_from_fixed(u, (n,))
    A = (U*F)*F'   # projection onto an affine transformation
    dU = U-A
    λ /= n
    if g != nothing && !isempty(g)
        λ2 = 2λ
        du = convert_to_fixed(SVector{N,T}, dU, (n,))
        for j=1:n
            g[j] = λ2*du[j]
        end
    end
    λ * sum(abs2, dU)
end

function penalty!(g, dp::AffinePenalty, ϕ_c, g_c)
    ret = penalty!(g, dp, ϕ_c)
    if g != nothing
        for i = 1:length(g)
            g[i] = g_c[i]'*g[i]
        end
    end
    ret
end

###
### Temporal penalty
###
"""
`penalty!(g, λt, ϕs)` calculates the temporal penalty
```
   (1/2)λt ∑_i (ϕ_{i+1} - ϕ_i)^2
```
for a vector `ϕ` of deformations. `g`, if not `nothing`, should be a
single real-valued vector with number of entries corresponding to all
of the `u` arrays in all of `ϕs`.
"""
function penalty!{D<:GridDeformation}(g, λt::Real, ϕs::Vector{D})
    if g == nothing || isempty(g)
        return penalty(λt, ϕs)
    end
    ngrid = length(first(ϕs).u)
    if length(g) != length(ϕs)*ngrid
        gsize = (size(first(ϕs).u)..., length(ϕs))
        throw(DimensionMismatch("g's length $(length(g)) inconsistent with $gsize"))
    end
    s = zero(eltype(D))
    λt2 = λt/2
    for i = 1:length(ϕs)-1
        ϕ  = ϕs[i]
        ϕp = ϕs[i+1]
        goffset = ngrid*(i-1)
        for k = 1:ngrid
            du = ϕp.u[k] - ϕ.u[k]
            dv = λt*du
            g[goffset+k] -= dv
            g[goffset+ngrid+k] += dv
            s += λt2*sum(abs2, du)
        end
    end
    s
end

function penalty!{T<:Number, D<:GridDeformation}(g::Array{T}, λt::Real, ϕs::Vector{D})
    N = ndims(first(ϕs))
    sz = size(first(ϕs).u)
    gf = RegisterDeformation.convert_to_fixed(g, (N, sz..., length(ϕs)))
    penalty!(gf, λt, ϕs)
end

function penalty{D<:GridDeformation}(λt::Real, ϕs::Vector{D})
    s = zero(eltype(D))
    ngrid = length(first(ϕs).u)
    λt2 = λt/2
    for i = 1:length(ϕs)-1
        ϕ  = ϕs[i]
        ϕp = ϕs[i+1]
        for k = 1:ngrid
            du = ϕp.u[k] - ϕ.u[k]
            s += λt2*sum(abs2, du)
        end
    end
    s
end

function RegisterDeformation.convert_to_fixed{T,N,A,L}(::Type{GridDeformation{T,N,A,L}}, g::Array{T})
    reinterpret(SVector{N,T}, g, (div(length(g), N),))
end

function vec2ϕs{T,N}(x::Array{T}, gridsize::NTuple{N,Int}, n, knots)
    xr = RegisterDeformation.convert_to_fixed(SVector{N,T}, x, (gridsize..., n))
    colons = ntuple(d->Colon(), N)::NTuple{N,Colon}
    [GridDeformation(view(xr, colons..., i), knots) for i = 1:n]
end

"""
    mmi = interpolate_mm!(mm, order=Quadratic)

Prepare a MismatchArray (returned by, e.g., RegisterMismatch) for
sub-pixel interpolation.  `order` is either `Linear` or `Quadratic`
from Interpolations.jl; with `Quadratic`, the original data are
"destroyed," in the sense that the values are changed into quadratic
interpolation coefficients. It can also be `Quadratic(InPlaceQ())` if
you want to specify the boundary condition (default is `InPlace()`).

`mmis = interpolate_mm!(mms, order)` prepares the array-of-MismatchArrays
`mms` for interpolation.
"""
function interpolate_mm!{T<:MismatchArray}(mms::AbstractArray{T}, spline::Interpolations.Degree)
    f = x->CenterIndexedArray(interpolate!(x.data, BSpline(spline), gridstyle(spline)))
    map(f, mms)
end

function interpolate_mm!(mm::MismatchArray, spline::Interpolations.Degree)
    CenterIndexedArray(interpolate!(mm.data, BSpline(spline), gridstyle(spline)))
end

interpolate_mm!{order<:Interpolations.Degree}(arg, ::Type{order}) =
    interpolate_mm!(arg, itporder(order))

interpolate_mm!(arg) = interpolate_mm!(arg, Quadratic)

itporder(::Type{Quadratic}) = Quadratic(InPlace())
itporder(::Type{Linear}) = Linear()
gridstyle(::Quadratic) = OnCell()
gridstyle(::Linear) = OnGrid()

@generated function Interpolations.gradient!{T,N}(g::AbstractVector, A::CenterIndexedArray{T,N}, i::Number...)
    length(i) == N || error("Must use $N indexes")
    args = [:(i[$d]+A.halfsize[$d]+1) for  d = 1:N]
    meta = Expr(:meta, :inline)
    :($meta; gradient!(g, A.data, $(args...)))
end

end  # module
