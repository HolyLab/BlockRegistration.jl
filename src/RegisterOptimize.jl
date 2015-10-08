__precompile__()

module RegisterOptimize

using MathProgBase, Ipopt, AffineTransforms, Interpolations, ForwardDiff, FixedSizeArrays, IterativeSolvers
using RegisterCore, RegisterDeformation, RegisterMismatch, RegisterPenalty
using RegisterDeformation: convert_to_fixed, convert_from_fixed
using Base.Test

import Base: *
import MathProgBase: SolverInterface

# Some conveniences for MathProgBase
abstract GradOnly <: SolverInterface.AbstractNLPEvaluator

function SolverInterface.initialize(d::GradOnly, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac])
            error("Unsupported feature $feat")
        end
    end
end
SolverInterface.features_available(d::GradOnly) = [:Grad, :Jac]


abstract GradOnlyBoundsOnly <: GradOnly

SolverInterface.eval_g(::GradOnlyBoundsOnly, g, x) = nothing
SolverInterface.jac_structure(::GradOnlyBoundsOnly) = Int[], Int[]
SolverInterface.eval_jac_g(::GradOnlyBoundsOnly, J, x) = nothing


# Some necessary ForwardDiff extensions to make Interpolations work
Base.real(v::ForwardDiff.GradientNumber) = real(v.value)
Base.ceil(::Type{Int}, v::ForwardDiff.GradientNumber)  = ceil(Int, v.value)
Base.floor(::Type{Int}, v::ForwardDiff.GradientNumber) = floor(Int, v.value)

export
    initial_deformation,
    optimize!,
    optimize_rigid

"""
This module provides convenience functions for minimizing the mismatch
between images. It supports both rigid registration and deformable
registration.

The main functions are:

- `optimize_rigid`: iteratively improve a rigid transformation, given raw images
- `initial_deformation`: provide an initial guess based on mismatch quadratic fits
- `optimize!`: iteratively improve a deformation, given mismatch data
"""
RegisterOptimize


###
### Rigid registration from raw images
###
"""
`tform = optimize_rigid(fixed, moving, tform0, maxshift, [SD = eye];
[thresh=0, tol=1e-4, print_level=0])` optimizes a rigid transformation
(rotation + shift) to minimize the mismatch between `fixed` and
`moving`.

`tform0` is an initial guess.  Use `SD` if your axes are not uniformly
sampled, for example `SD = diagm(voxelspacing)` where `voxelspacing`
is a vector encoding the spacing along all axes of the image. `thresh`
enforces a certain amount of sum-of-squared-intensity overlap between
the two images; with non-zero `thresh`, it is not permissible to
"align" the images by shifting one entirely out of the way of the
other.
"""
function optimize_rigid(fixed, moving, A::AffineTransform, maxshift, SD = eye(ndims(A)); thresh=0, tol=1e-4, print_level=0)
    objective = RigidOpt(to_float(fixed, moving)..., SD, thresh)
    # Convert initial guess into parameter vector
    R = SD*A.scalefwd/SD
    rotp = rotationparameters(R)
    dx = A.offset
    p0 = [rotp; dx]
    T = eltype(p0)

    # Set up and run the solver
    solver = IpoptSolver(hessian_approximation="limited-memory",
                         print_level=print_level,
                         tol=tol)
    m = SolverInterface.model(solver)
    ub = T[fill(pi, length(p0)-length(maxshift)); [maxshift...]]
    SolverInterface.loadnonlinearproblem!(m, length(p0), 0, -ub, ub, T[], T[], :Min, objective)
    SolverInterface.setwarmstart!(m, p0)
    SolverInterface.optimize!(m)

    stat = SolverInterface.status(m)
    stat == :Optimal || warn("Solution was not optimal")
    p = SolverInterface.getsolution(m)
    fval = SolverInterface.getobjval(m)

    p2rigid(p, SD), fval
end

function p2rigid(p, SD)
    if length(p) == 1
        return AffineTransform([1], p)  # 1 dimension
    elseif length(p) == 3
        return AffineTransform(SD\(rotation2(p[1])*SD), p[2:end])    # 2 dimensions
    elseif length(p) == 6
        return AffineTransform(SD\(rotation3(p[1:3])*SD), p[4:end])  # 3 dimensions
    else
        error("Dimensionality not supported")
    end
end

to_float(A, B) = to_float(typeof(one(eltype(A)) - one(eltype(B))), A, B)
to_float{T<:AbstractFloat}(::Type{T}, A, B) = convert(Array{T}, A), convert(Array{T}, B)
to_float{T}(::Type{T}, A, B) = convert(Array{Float32}, A), convert(Array{Float32}, B)


###
### Rigid registration from raw images, MathProg interface
###
type RigidValue{N,A<:AbstractArray,I<:AbstractExtrapolation,SDT} <: SolverInterface.AbstractNLPEvaluator
    fixed::A
    wfixed::A
    moving::I
    SD::SDT
    thresh
end

function RigidValue{T<:Real}(fixed::AbstractArray, moving::AbstractArray{T}, SD, thresh)
    f = copy(fixed)
    fnan = isnan(f)
    f[fnan] = 0
    m = copy(moving)
    mnan = isnan(m)
    m[mnan] = 0
    metp = extrapolate(interpolate!(m, BSpline{Quadratic{InPlace}}, OnCell), NaN)
    RigidValue{ndims(f),typeof(f),typeof(metp),typeof(SD)}(f, !fnan, metp, SD, thresh)
end

function Base.call(d::RigidValue, x)
    tfm = p2rigid(x, d.SD)
    mov = transform(d.moving, tfm)
    movnan = isnan(mov)
    mov[movnan] = 0
    f = d.fixed.*!movnan
    m = mov.*d.wfixed
    den = sumabs2(f)+sumabs2(m)
    real(den) < d.thresh && return convert(typeof(den), Inf)
    sumabs2(f-m)/den
end

type RigidOpt{RV<:RigidValue,G} <: GradOnlyBoundsOnly
    rv::RV
    g::G
end

function RigidOpt(fixed, moving, SD, thresh)
    rv = RigidValue(fixed, moving, SD, thresh)
    g = ForwardDiff.gradient(rv)
    RigidOpt(rv, g)
end

SolverInterface.eval_f(d::RigidOpt, x) = d.rv(x)
SolverInterface.eval_grad_f(d::RigidOpt, grad_f, x) =
    copy!(grad_f, d.g(x))

###
### Globally-optimal initial guess for deformation given
### quadratic-fit mismatch data
###
"""
`u0 = initial_deformation(ap::AffinePenalty, cs, Qs;
[ϕ_old=identity])` prepares a globally-optimal initial guess for a
deformation, given a quadratic fit to the aperture-wise mismatch
data. `cs` and `Qs` must be arrays-of-arrays in the shape of the
u0-grid, each entry as calculated by `qfit`. The initial guess
minimizes the function

```
    ap(ϕ(u0)) + ∑_i (u0[i]-cs[i])' * Qs[i] * (u0[i]-cs[i])
```
where `ϕ(u0)` is the deformation associated with `u0`.

If `ϕ_old` is not the identity, it must be interpolating.
"""
function initial_deformation{T,N}(ap::AffinePenalty{T,N}, cs, Qs)
    b = prep_b(T, cs, Qs)
    # A = to_full(ap, Qs)
    # F = svdfact(A)
    # S = F[:S]
    # smax = maximum(S)
    # fac = sqrt(eps(typeof(smax)))
    # for i = 1:length(S)
    #     if S[i] < fac*smax
    #         S[i] = Inf
    #     end
    # end
    # x, isconverged = F\b, true
    # In case the grid is really big, solve iteratively
    # (The matrix is not sparse, but matrix-vector products can be
    # computed efficiently.)
    P = AffineQHessian(ap, Qs, identity)
    x, isconverged = find_opt(P, b)
    convert_to_fixed(x, (N,size(cs)...)), isconverged
end

function to_full{T,N}(ap::AffinePenalty{T,N}, Qs)
    FF = ap.F*ap.F'
    nA = N*size(FF,1)
    FFN = zeros(nA,nA)
    for o = 1:N
        FFN[o:N:end,o:N:end] = FF
    end
    A = ap.λ*(I - FFN)
    for i = 1:length(Qs)
        A[N*(i-1)+1:N*i, N*(i-1)+1:N*i] += Qs[i]
    end
    A
end

function prep_b{T}(::Type{T}, cs, Qs)
    n = prod(size(Qs))
    N = length(first(cs))
    b = zeros(T, N*n)
    for i = 1:n
        b[(i-1)*N+1:i*N] = Qs[i]*cs[i]
    end
    b
end

function find_opt(P, b)
    x, result = cg(P, b)
    x, result.isconverged
end

# A type for computing multiplication by the linear operator
type AffineQHessian{AP<:AffinePenalty,M<:Mat,N,Φ}
    ap::AP
    Qs::Array{M,N}
    ϕ_old::Φ
end

function AffineQHessian(ap::AffinePenalty, Qs::AbstractArray, ϕ_old)
    T = eltype(first(Qs))
    N = ndims(Qs)
    AffineQHessian{typeof(ap),Mat{N,N,T},N,typeof(ϕ_old)}(ap, Qs, ϕ_old)
end

Base.eltype{AP,M,N,Φ}(::Type{AffineQHessian{AP,M,N,Φ}}) = eltype(AP)
Base.eltype(P::AffineQHessian) = eltype(typeof(P))
Base.size(P::AffineQHessian, d) = length(P.Qs)*size(first(P.Qs),1)

# These compute the gradient of (x'*P*x)/2, where P is the Hessian
# for the objective in the doc text for initial_deformation.
function (*){T,N}(P::AffineQHessian{AffinePenalty{T,N}}, x::AbstractVector{T})
    gridsize = size(P.Qs)
    n = prod(gridsize)
    u = convert_to_fixed(x, (N,gridsize...)) #reinterpret(Vec{N,T}, x, gridsize)
    g = similar(u)
    λ = P.ap.λ
    P.ap.λ = λ*n/2
    affine_part!(g, P, u)
    P.ap.λ = λ
    sumQ = zero(T)
    for i = 1:n
        g[i] += P.Qs[i] * u[i]
        sumQ += trace(P.Qs[i])
    end
    # Add a stabilizing diagonal, for cases where λ is very small
    if sumQ == 0
        sumQ = one(T)
    end
    fac = cbrt(eps(T))*sumQ/n
    for i = 1:n
        g[i] += fac*u[i]
    end
    reinterpret(T, g, size(x))
end

affine_part!(g, P, u) = penalty!(g, P.ap, u)


function initial_deformation{T,N}(ap::AffinePenalty{T,N}, cs, Qs, ϕ_old, maxshift)
    error("This is broken, don't use it")
    b = prep_b(T, cs, Qs)
    # In case the grid is really big, solve iteratively
    # (The matrix is not sparse, but matrix-vector products can be
    # computed efficiently.)
    P0 = AffineQHessian(ap, Qs, identity)
    x0 = find_opt(P0, b)
    P = AffineQHessian(ap, Qs, ϕ_old)
    x = find_opt(P, b, maxshift, x0)
    u0 = convert_to_fixed(x, (N,size(cs)...)) #reinterpret(Vec{N,eltype(x)}, x, size(cs))
end

# type for minimization with composition (which turns the problem into
# a nonlinear problem)
type InitialDefOpt{AQH,B} <: GradOnlyBoundsOnly
    P::AQH
    b::B
end

function find_opt{AP,M,N,Φ<:GridDeformation}(P::AffineQHessian{AP,M,N,Φ}, b, maxshift, x0)
    objective = InitialDefOpt(P, b)
    solver = IpoptSolver(hessian_approximation="limited-memory",
                         print_level=0)
    m = SolverInterface.model(solver)
    T = eltype(b)
    n = length(b)
    ub1 = T[maxshift...] - T(0.5001)
    ub = repeat(ub1, outer=[div(n, length(maxshift))])
    SolverInterface.loadnonlinearproblem!(m, n, 0, -ub, ub, T[], T[], :Min, objective)
    SolverInterface.setwarmstart!(m, x0)
    SolverInterface.optimize!(m)
    stat = SolverInterface.status(m)
    stat == :Optimal || warn("Solution was not optimal")
    SolverInterface.getsolution(m)
end

# We omit the constant term ∑_i cs[i]'*Qs[i]*cs[i], since it won't
# affect the solution
SolverInterface.eval_f(d::InitialDefOpt, x::AbstractVector) =
    _eval_f(d.P, d.b, x)

function _eval_f{T,N}(P::AffineQHessian{AffinePenalty{T,N}}, b, x::AbstractVector)
    gridsize = size(P.Qs)
    n = prod(gridsize)
    u  = convert_to_fixed(x, (N,gridsize...))# reinterpret(Vec{N,T}, x, gridsize)
    bf = convert_to_fixed(b, (N,gridsize...))# reinterpret(Vec{N,T}, b, gridsize)
    λ = P.ap.λ
    P.ap.λ = λ*n/2
    val = affine_part!(nothing, P, u)
    P.ap.λ = λ
    for i = 1:n
        val += ((u[i]' * P.Qs[i] * u[i])/2 - bf[i]'*u[i])[1]
    end
    val
end

function SolverInterface.eval_grad_f(d::InitialDefOpt, grad_f, x)
    P, b = d.P, d.b
    copy!(grad_f, P*x-b)
end

function affine_part!{AP,M,N,Φ<:GridDeformation}(g, P::AffineQHessian{AP,M,N,Φ}, u)
    ϕ_c, g_c = compose(P.ϕ_old, GridDeformation(u, P.ϕ_old.knots))
    penalty!(g, P.ap, ϕ_c, g_c)
end

function affine_part!{AP,M,N,Φ<:GridDeformation}(::Void, P::AffineQHessian{AP,M,N,Φ}, u)
    # Sadly, with GradientNumbers this gives an error I haven't traced
    # down (might be a Julia bug)
    # ϕ_c = P.ϕ_old(GridDeformation(u, P.ϕ_old.knots))
    # penalty!(nothing, P.ap, ϕ_c)
    u_c = RegisterDeformation._compose(P.ϕ_old.u, u, P.ϕ_old.knots)
    penalty!(nothing, P.ap, u_c)
end

###
### Optimize (via descent) a deformation to mismatch data
###
"""
`ϕ, fval = optimize!(ϕ, ϕ_old, dp, mmis; [tol=1e-4, print_level=0])`
improves an initial deformation `ϕ` to reduce the mismatch.  The
arguments are as described for `penalty!` in RegisterPenalty.  On
output, `ϕ` is set in-place to the new optimized deformation, and
`fval` is the value of the penalty.
"""
function optimize!(ϕ, ϕ_old, dp::DeformationPenalty, mmis; tol=1e-4, print_level=0)
    objective = DeformOpt(ϕ, ϕ_old, dp, mmis)
    uvec = u_as_vec(ϕ)
    T = eltype(uvec)
    mxs = maxshift(first(mmis))

    solver = IpoptSolver(hessian_approximation="limited-memory",
                         print_level=print_level,
                         tol=tol)
    m = SolverInterface.model(solver)
    ub1 = T[mxs...] - T(0.5001)
    ub = repeat(ub1, outer=[length(ϕ.u)])
    SolverInterface.loadnonlinearproblem!(m, length(uvec), 0, -ub, ub, T[], T[], :Min, objective)
    SolverInterface.setwarmstart!(m, uvec)
    val0 = SolverInterface.eval_f(objective, uvec)
    isfinite(val0) || error("Initial value must be finite")
    SolverInterface.optimize!(m)

    stat = SolverInterface.status(m)
    stat == :Optimal || warn("Solution was not optimal")
    uopt = SolverInterface.getsolution(m)
    fval = SolverInterface.getobjval(m)
    copy!(uvec, uopt)
    ϕ, fval
end

function u_as_vec(ϕ)
    T = eltype(eltype(ϕ.u))
    N = length(eltype(ϕ.u))
    uvec = reinterpret(T, ϕ.u, (N*length(ϕ.u),))
end

function vec_as_u{T,N}(g::Array{T}, ϕ::GridDeformation{T,N})
    reinterpret(Vec{N,T}, g, size(ϕ.u))
end

type DeformOpt{D,Dold,DP,M} <: GradOnlyBoundsOnly
    ϕ::D
    ϕ_old::Dold
    dp::DP
    mmis::M
end

function SolverInterface.eval_f(d::DeformOpt, x)
    uvec = u_as_vec(d.ϕ)
    copy!(uvec, x)
    penalty!(nothing, d.ϕ, d.ϕ_old, d.dp, d.mmis)
end

function SolverInterface.eval_grad_f(d::DeformOpt, grad_f, x)
    uvec = u_as_vec(d.ϕ)
    copy!(uvec, x)
    penalty!(vec_as_u(grad_f, d.ϕ), d.ϕ, d.ϕ_old, d.dp, d.mmis)
end

###
### Mismatch-based optimization of affine transformation
###
### NOTE: not updated yet, probably broken
"""
`tform = optimize(tform0, mms, knots)` performs descent-based
minimization of the total mismatch penalty as a function of the
parameters of an affine transformation, starting from an initial guess
`tform0`.  While this is unlikely to yield very accurate results for
large rotations or skews (the mismatch data are themselves suspect in
such cases), it can be helpful for polishing small deformations.

For a good initial guess, see `mismatch2affine`.
"""
function optimize(tform::AffineTransform, mmis, knots)
    gridsize = size(mmis)
    N = length(gridsize)
    ndims(tform) == N || error("Dimensionality of tform is $(ndims(tform)), which does not match $N for nums/denoms")
    mm = first(mmis)
    mxs = maxshift(mm)
    T = eltype(eltype(mm))
    # Compute the bounds
    asz = arraysize(knots)
    center = T[(asz[i]+1)/2 for i = 1:N]
    X = zeros(T, N+1, prod(gridsize))
    for (i, knot) in enumerate(eachknot(knots))
        X[1:N,i] = knot - center
        X[N+1,i] = 1
    end
    bound = convert(Vector{T}, [mxs .- register_half; Inf])
    lower = repeat(-bound, outer=[1,size(X,2)])
    upper = repeat( bound, outer=[1,size(X,2)])
    # Extract the parameters from the initial guess
    Si = tform.scalefwd
    displacement = tform.offset
    A = convert(Matrix{T}, [Si-eye(N) displacement; zeros(1,N) 1])
    # Determine the blocks that start in-bounds
    AX = A*X
    keep = trues(gridsize)
    for j = 1:length(keep)
        for idim = 1:N
            xi = AX[idim,j]
            if xi < -mxs[idim]+register_half_safe || xi > mxs[idim]-register_half_safe
                keep[j] = false
                break
            end
        end
    end
    if !any(keep)
        @show tform
        warn("No valid blocks were found")
        return tform
    end
    ignore = !keep[:]
    lower[:,ignore] = -Inf
    upper[:,ignore] =  Inf
    # Assemble the objective and constraints

    constraints = Optim.ConstraintsL(X', lower', upper')
    gtmp = Array(Vec{N,T}, gridsize)
    objective = (x,g) -> affinepenalty!(g, x, mmis, keep, X', gridsize, gtmp)
    @assert typeof(objective(A', T[])) == T
    result = interior(DifferentiableFunction(x->objective(x,T[]), Optim.dummy_g!, objective), A', constraints, method=:cg)
    @assert Optim.converged(result)
    Aopt = result.minimum'
    Siopt = Aopt[1:N,1:N] + eye(N)
    displacementopt = Aopt[1:N,end]
    AffineTransform(convert(Matrix{T}, Siopt), convert(Vector{T}, displacementopt)), result.f_minimum
end

function affinepenalty!{N}(g, At, mmis, keep, Xt, gridsize::NTuple{N}, gtmp)
    u = _calculate_u(At, Xt, gridsize)
    @assert eltype(u) == eltype(At)
    val = penalty!(gtmp, u, mmis, keep)
    @assert isa(val, eltype(At))
    if !isempty(g)
        T = eltype(eltype(gtmp))
        nblocks = size(Xt,1)
        At_mul_Bt!(g, Xt, [reinterpret(T,gtmp,(N,nblocks)); zeros(1,nblocks)])
    end
    val
end

function _calculate_u{N}(At, Xt, gridsize::NTuple{N})
    Ut = Xt*At
    u = Ut[:,1:size(Ut,2)-1]'                   # discard the dummy dimension
    reinterpret(Vec{N, eltype(u)}, u, gridsize) # put u in the shape of the grid
end


end # module
