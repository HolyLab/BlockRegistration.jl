# Penalty function for registration of a single image

__precompile__()

module RegisterPenalty

using Interpolations, FixedSizeArrays, Base.Cartesian
using RegisterDeformation, RegisterCore, CenterIndexedArrays

export AffinePenalty, DeformationPenalty, penalty!, interpolate_mm!


abstract DeformationPenalty{T}

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

`g` should be the same type and size as `ϕ.u`, i.e., an array of
fixed-sized vectors `Vec{N,T}`.

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

################
# Data penalty #
################

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
function penalty!{T,Dim,A<:AbstractInterpolation}(g, ϕ::AbstractDeformation, mmis::AbstractArray{MismatchArray{T,Dim,A}}, keep=trues(size(mmis)))
    penalty!(g, ϕ.u, mmis, keep)
end

function penalty!{T,Dim,A<:AbstractInterpolation}(g, u::AbstractArray, mmis::AbstractArray{MismatchArray{T,Dim,A}}, keep=trues(size(mmis)))
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
        Tu = eltype(eltype(u))
        gnd = similar(u, Vec{Dim,NumDenom{Tu}})
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
        gtmp = Array(NumDenom{T}, N)
    end
    for (iblock,mmi) in enumerate(mmis)
        if !keep[iblock]
            if calc_grad
                gnd[iblock] = NumDenom{T}(0,0)
            end
            continue
        end
        # Check bounds
        dx = u[iblock]
        if !checkbounds_shift(dx, mxs)
            return (nanT,nanT)
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

@generated function checkbounds_shift{N}(dx::Vec{N}, mxs)
    quote
        @nexprs $N d->(if abs(dx[d]) >= mxs[d]-0.5 return false end)
        true
    end
end

@generated function _wsum{N}(x::Vec{N}, cnum, cdenom)
    args = [:(cnum*x[$d].num + cdenom*x[$d].denom) for d = 1:N]
    quote
        Vec($(args...))
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
type AffinePenalty{T} <: DeformationPenalty{T}
    Q::Matrix{T}   # geometry data for the affine-residual penalty
    λ::T           # regularization coefficient

    function AffinePenalty{N}(knots::NTuple{N}, λ)
        gridsize = map(length, knots)
        C = Array(Float64, prod(gridsize), N+1)
        i = 0
        for I in CartesianRange(gridsize)
            C[i+=1, N+1] = 1
            for j = 1:N
                C[i, j] = knots[j][I[j]]  # I[j]
            end
        end
        Q, _ = qr(C)
        new(Q, λ)
    end

    function AffinePenalty(knots::AbstractMatrix, λ)
        C = hcat(knots', ones(eltype(knots), size(knots, 2)))
        Q, _ = qr(C)
        new(Q, λ)
    end
end

AffinePenalty{V<:AbstractVector,N}(knots::NTuple{N,V}, λ) = AffinePenalty{eltype(V)}(knots, λ)
AffinePenalty{V<:AbstractVector}(knots::AbstractVector{V}, λ) = AffinePenalty{eltype(V)}((knots...), λ)
AffinePenalty{T}(knots::AbstractMatrix{T}, λ) = AffinePenalty{T}(knots, λ)


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
    Q, λ = dp.Q, dp.λ
    if λ == 0
        if g != nothing && !isempty(g)
            fill!(g, zero(eltype(g)))
        end
        return λ * one(eltype(Q)) * one(T)
    end
    n = length(ϕ_c.u)
    U = reinterpret(T, ϕ_c.u, (N, n))
    A = (U*Q)*Q'   # projection onto an affine transformation
    dU = U-A
    λ /= n
    if g != nothing && !isempty(g)
        λ2 = 2λ
        du = reinterpret(Vec{N,T}, dU, (n,))
        for j=1:n
            g[j] = λ2*du[j]
        end
    end
    λ * sumabs2(dU)
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

"""
`mmi = interpolate_mm!(mm)` prepares a MismatchArray (returned by,
e.g., RegisterMismatch) for sub-pixel interpolation.  The original
data are "destroyed," in the sense that the values are changed into
quadratic interpolation coefficients.

`mmis = interpolate_mm!(mms)` prepares the array-of-MismatchArrays
`mms` for interpolation.
"""
function interpolate_mm!{T<:MismatchArray}(mms::AbstractArray{T}; BC=InPlace)
    f = x->CenterIndexedArray(interpolate!(x.data, BSpline{Quadratic{BC}}, OnCell))
    map(f, mms)
end

function interpolate_mm!(mm::MismatchArray; BC=InPlace)
    CenterIndexedArray(interpolate!(mm.data, BSpline{Quadratic{BC}}, OnCell))
end

@generated function Interpolations.gradient!{T,N}(g::AbstractVector, A::CenterIndexedArray{T,N}, i::Number...)
    length(i) == N || error("Must use $N indexes")
    args = [:(i[$d]+A.halfsize[$d]+1) for  d = 1:N]
    meta = Expr(:meta, :inline)
    :($meta; gradient!(g, A.data, $(args...)))
end

end  # module
