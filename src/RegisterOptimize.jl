__precompile__()

module RegisterOptimize

using MathProgBase, Ipopt, AffineTransforms, Interpolations, ForwardDiff, FixedSizeArrays, IterativeSolvers
using RegisterCore, RegisterDeformation, RegisterMismatch, RegisterPenalty
using Base.Test

import Base: *
import MathProgBase: SolverInterface

abstract GradOnly <: SolverInterface.AbstractNLPEvaluator

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

type RigidOpt{RV<:RigidValue,G} <: GradOnly
    rv::RV
    g::G
end

function RigidOpt(fixed, moving, SD, thresh)
    rv = RigidValue(fixed, moving, SD, thresh)
    g = ForwardDiff.gradient(rv)
    RigidOpt(rv, g)
end

function SolverInterface.initialize(d::GradOnly, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac])
            error("Unsupported feature $feat")
        end
    end
end
SolverInterface.features_available(d::GradOnly) = [:Grad, :Jac]

SolverInterface.eval_f(d::RigidOpt, x) = d.rv(x)
SolverInterface.eval_g(d::RigidOpt, g, x) = nothing
SolverInterface.eval_grad_f(d::RigidOpt, grad_f, x) =
    copy!(grad_f, d.g(x))
SolverInterface.jac_structure(d::RigidOpt) = Int[], Int[]
SolverInterface.eval_jac_g(d::RigidOpt, J, x) = nothing

###
### Globally-optimal initial guess for deformation given
### quadratic-fit mismatch data
###
"""
`u0 = initial_deformation(dp::AffinePenalty, cs, Qs)` prepares a
globally-optimal initial guess for a deformation, given a quadratic
fit to the aperture-wise mismatch data. `cs` and `Qs` must be
arrays-of-arrays in the shape of the u0-grid, each entry as calculated
by `qfit`.
"""
function initial_deformation(ap::AffinePenalty, cs, Qs)
    n = prod(size(Qs))
    c = first(cs)
    N = length(c)
    b = zeros(eltype(c), N*n)
    for i = 1:n
        b[(i-1)*N+1:i*N] = Qs[i]*cs[i]
    end
    # In case the grid is really big, solve iteratively
    # (The matrix is not sparse, but matrix-vector products can be
    # computed efficiently.)
    P = AffineQHessian(ap, Qs)
    x, _ = cg(P, b)
    u0 = reinterpret(Vec{N,eltype(x)}, x, size(cs))
end

type AffineQHessian{T,QA}
    ap::AffinePenalty{T}
    Qs::QA
end

Base.eltype{T,QA}(::Type{AffineQHessian{T,QA}}) = T
Base.size(P::AffineQHessian, d) = length(P.Qs)*size(first(P.Qs),1)

function (*)(P::AffineQHessian, x::AbstractVector)
    gridsize = size(P.Qs)
    n = prod(gridsize)
    N = size(first(P.Qs), 1)
    U = reshape(x, N, n)
    F = P.ap.F
    A = (U*F)*F'
    g = P.ap.λ*(U-A)
    u = Array(eltype(U), N)
    Qu = similar(u)
    for i = 1:n
        for d = 1:N
            u[d] = U[d,i]
        end
        A_mul_B!(Qu, P.Qs[i], u)
        for d = 1:N
            g[(i-1)*N+d] += Qu[d]
        end
    end
    vec(g)
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

type DeformOpt{D,Dold,DP,M} <: GradOnly
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

SolverInterface.eval_g(d::DeformOpt, g, x) = nothing

function SolverInterface.eval_grad_f(d::DeformOpt, grad_f, x)
    uvec = u_as_vec(d.ϕ)
    copy!(uvec, x)
    penalty!(vec_as_u(grad_f, d.ϕ), d.ϕ, d.ϕ_old, d.dp, d.mmis)
end

SolverInterface.jac_structure(d::DeformOpt) = Int[], Int[]
SolverInterface.eval_jac_g(d::DeformOpt, J, x) = nothing

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
