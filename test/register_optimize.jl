using StaticArrays, AffineTransforms, Interpolations, Base.Test
import BlockRegistration, RegisterOptimize
using RegisterCore, RegisterPenalty, RegisterDeformation, RegisterMismatch, RegisterFit
using Images, CoordinateTransformations, Rotations, RegisterOptimize

using RegisterTestUtilities

###
### Global-optimum initial guess
###
function build_Ac_b(A, cs::Matrix, Qs::Matrix)
    b = zeros(size(A,1))
    Ac = copy(A)
    for i = 1:length(Qs)
        Ac[2i-1:2i,2i-1:2i] += Qs[i]
        b[2i-1:2i] = Qs[i]*cs[i]
    end
    Ac, b
end

function initial_guess_direct(A, cs::Matrix, Qs::Matrix)
    Ac, b = build_Ac_b(A, cs, Qs)
    x = Ac\b
    reinterpret(SVector{2,Float64}, x, size(Qs))
end

function build_Ac_b{Tc,TQ}(A, λt, cs::Array{Tc,3}, Qs::Array{TQ,3})
    n = size(Qs,3)
    l = size(A,1)
    b = zeros(l*n)
    Ac = zeros(eltype(A), l*n, l*n)
    for i = 1:n
        Ac[(i-1)*l+1:i*l, (i-1)*l+1:i*l] = A
        if i == 1 || i == n
            for k = 1:l
                Ac[(i-1)*l+k,(i-1)*l+k] += λt
            end
        else
            for k = 1:l
                Ac[(i-1)*l+k,(i-1)*l+k] += 2*λt
            end
        end
    end
    for i = 1:n-1
        for k = 1:l
            Ac[(i-1)*l+k, i*l+k] -= λt
            Ac[i*l+k, (i-1)*l+k] -= λt
        end
    end
    for i = 1:length(Qs)
        Ac[2i-1:2i,2i-1:2i] += Qs[i]
        b[2i-1:2i] = Qs[i]*cs[i]
    end
    Ac, b
end

function initial_guess_direct{Tc,TQ}(A, λt, cs::Array{Tc,3}, Qs::Array{TQ,3})
    Ac, b = build_Ac_b(A, λt, cs, Qs)
    x = Ac\b
    reinterpret(SVector{2,Float64}, x, size(Qs))
end

function build_A(knots, λ)
    ap = AffinePenalty(knots, λ)
    FF = ap.F*ap.F'
    nA = 2*size(FF,1)
    FF2 = zeros(nA,nA)
    FF2[1:2:end,1:2:end] = FF
    FF2[2:2:end,2:2:end] = FF
    A = ap.λ*(I - FF2)
    A, ap
end

knots = (linspace(1,20,4),linspace(1,15,4))
A, ap = build_A(knots, 1.0)
gridsize = map(length, knots)
Qs = Array{Any}(gridsize)
cs = Array{Any}(gridsize)

# Known exact answer
tfm = tformrotate(pi/12)
for (i,knot) in enumerate(eachknot(knots))
    v = [knot[1],knot[2]]
    cs[i] = tfm*v-v
    Qs[i] = eye(2,2)
end
P = RegisterOptimize.AffineQHessian(ap, Qs, identity)
Ac, b = build_Ac_b(A, cs, Qs);
v = zeros(size(P,1)); v[1] = 1
@test ≈(P*v, vec(Ac[1,:]), atol=0.0001)
v = zeros(size(P,1)); v[4] = 1
@test ≈(P*v, vec(Ac[4,:]), atol=0.0001)
ux = initial_guess_direct(A, cs, Qs)
u, isconverged = @inferred(RegisterOptimize.initial_deformation(ap, cs, Qs))
@test isconverged
@test size(u) == size(ux)
@test eltype(u) == SVector{2,Float64}
# The accuracy here is low only because of the diagonal regularization
for I in eachindex(u)
    @test ≈(u[I], cs[I], atol=1e-3)
    @test ≈(ux[I], cs[I], atol=1e-3)
end
# Ensure that λ=0 gives the right initialization
knots0 = (linspace(1,3,3),)
ap0 = AffinePenalty(knots0, 0.0)
cs0 = Any[[17.0], [-44.0], [12.0]] # Vector{Float64}[[17.0], [-44.0], [12.0]]
Qs0 = [reshape([1], 1, 1) for i = 1:3]
u0, isconverged = @inferred(RegisterOptimize.initial_deformation(ap0, cs0, Qs0))
@test isconverged
@test u0[1] == [17.0]
@test u0[2] == [-44.0]
@test u0[3] == [12.0]

# Random initialization
for I in CartesianRange(gridsize)
    QF = rand(2,2)
    Qs[I] = QF'*QF
    cs[I] = randn(2)
end
P = RegisterOptimize.AffineQHessian(ap, Qs, identity)
Ac, b = build_Ac_b(A, cs, Qs);
v = zeros(size(P,1)); v[1] = 1
@test ≈(P*v, vec(Ac[1,:]), atol=0.0001)
v = zeros(size(P,1)); v[4] = 1
@test ≈(P*v, vec(Ac[4,:]), atol=0.0001)
ux = initial_guess_direct(A, cs, Qs)
u, isconverged = RegisterOptimize.initial_deformation(ap, cs, Qs)
@test isconverged
@test size(u) == size(ux)
@test eltype(u) == SVector{2,Float64}
for I in eachindex(u)
    @test ≈(u[I], ux[I], atol=1e-3)
end

# Random initialization with a temporal penalty
Qs = Array{SMatrix{2,2,Float64,4}}(gridsize..., 5)
cs = Array{SVector{2,Float64}}(gridsize..., 5)
for I in CartesianRange(size(Qs))
    QF = rand(2,2)
    Qs[I] = QF'*QF
    cs[I] = randn(2)
end
csr = convert(Array{Vector{Float64}, 3}, cs)
Qsr = convert(Array{Matrix{Float64}, 3}, Qs)
λt = 1.0
P = RegisterOptimize.TimeHessian(RegisterOptimize.AffineQHessian(ap, Qs, identity), λt)
Ac, b = build_Ac_b(A, λt, csr, Qsr)
v = zeros(size(P,1)); v[1] = 1
@test ≈(P*v, vec(Ac[1,:]), atol=0.0001)
v = zeros(size(P,1)); v[4] = 1
@test ≈(P*v, vec(Ac[4,:]), atol=0.0001)
v = zeros(size(P,1)); v[end] = 1
@test ≈(P*v, vec(Ac[end,:]), atol=0.0001)
ux = initial_guess_direct(A, 1.0, csr, Qsr)
u, isconverged = RegisterOptimize.initial_deformation(ap, 1.0, cs, Qs)
@test isconverged
@test size(u) == size(ux)
@test eltype(u) == SVector{2,Float64}
for I in eachindex(u)
    @test ≈(u[I], ux[I], atol=1e-3)
end


# # With composition
# # We use a larger grid because the edges are suspect
# arraysize = (100,80)
# gridsize = (7,6)
# knots = (linspace(1,arraysize[1],gridsize[1]),linspace(1,arraysize[2],gridsize[2]))
# A, ap = build_A(knots, 1.0)
# Qs = Array{Any}(gridsize)
# cs = Array{Any}(gridsize)
# for I in CartesianRange(gridsize)
#     QF = rand(2,2)
#     Qs[I] = QF'*QF
#     cs[I] = randn(2)
# end
# ux = initial_guess_direct(A, cs, Qs)
# # First, a trivial deformation
# u_old = zeros(2, gridsize...)
# ϕ_old = interpolate(GridDeformation(u_old, knots))
# u = RegisterOptimize.initial_deformation(ap, cs, Qs, ϕ_old, (10,10))
# ϕ_c = ϕ_old(GridDeformation(u, knots))
# for I in eachindex(ux)
#     @test ϕ_c.u[I] ≈ ux[I]
# end


# # We build a ϕ_old that varies quadratically, so interpolation will be
# # precise
# m = maximum(mapreduce(abs, max, ux))
# Qold1 = rand(2,2); Qold1 = Qold1'*Qold1
# u1 = quadratic(gridsize..., (0,0), Qold1)
# u1 *= m/maxabs(u1)  # make sure it's of commensurate size
# Qold2 = rand(2,2); Qold2 = Qold2'*Qold2
# u2 = quadratic(gridsize..., (1,-1), Qold2)
# u2 *= m/maxabs(u2)
# u_old = permutedims(cat(3, u1, u2), (3, 1, 2))
# tfm = tformrotate(pi/12)
# ϕ_old = interpolate(tform2deformation(tfm, arraysize, gridsize))
# Transform the cs and Qs
# csi = similar(cs)
# Qsi = similar(Qs)
# arrayc = [map(x->(x+1)/2, arraysize)...]
# for (i,knot) in enumerate(eachknot(knots))
#     x = convert(Vector, knot) + cs[i] - arrayc
#     csi[i] = tfm\x - x
#     Qsi[i] = tfm.scalefwd*Qs[i]*tfm.scalefwd'
# end
# u = RegisterOptimize.initial_deformation(ap, csi, Qsi, ϕ_old, (10,10))
# # Test gradients:
# import MathProgBase: SolverInterface
# b = RegisterOptimize.prep_b(Float64, cs, Qs)
# P = RegisterOptimize.AffineQHessian(ap, Qs, ϕ_old)
# objective = RegisterOptimize.InitialDefOpt(P, b)
# fdgrad = ForwardDiff.gradient(x->SolverInterface.eval_f(objective, x))
# error("stop")
# @test size(u) == size(ux)
# @test eltype(u) == SVector{2,Float64}
# ϕ_c = ϕ_old(GridDeformation(u, knots))
# for I in eachindex(ux)
#     @test ϕ_c.u[I] ≈ ux[I]
# end


###
### Minimization to mismatch data
###

# Set up an affine transformation and put the optimal shift
# in each block at the corresponding shifted-knot position
S = eye(2,2) + 0.1*rand(2,2)
imgsz = (100,80)
gridsize = (7,5)
cntr = ([imgsz...]+1)/2
tform = AffineTransform(S, zeros(2))
knots = (linspace(1,imgsz[1],gridsize[1]), linspace(1,imgsz[2],gridsize[2]))
shifts = Array{Any}(gridsize)
mxsv = zeros(2)
for (i,knot) in enumerate(eachknot(knots))
    knotv = [knot...]-cntr
    dx = tform*knotv - knotv
    mxsv = max.(mxsv, abs.(dx))
    shifts[i] = dx
end
# Create the fake mismatch data
m, n = 2ceil(Int,mxsv[1])+3, 2ceil(Int,mxsv[2])+3
nums = Array{Matrix{Float64}}(gridsize)
for I in eachindex(nums)
    QF = rand(2,2)   # random quadratic component
    nums[I] = quadratic(m, n, shifts[I], QF*QF')
end
denom = ones(m, n)
mms = mismatcharrays(nums, denom)
mmis = interpolate_mm!(mms, Quadratic(InPlaceQ()))

u = randn(2, gridsize...)
RegisterFit.uclamp!(u, (m>>1, n>>1))
ϕ = GridDeformation(u, knots)
λ = 1000.0
dp = AffinePenalty(knots, λ)
ϕ, fval = RegisterOptimize.optimize!(ϕ, identity, dp, mmis) #; print_level=5)
@test 0 <= fval <= 1e-5
for I in eachindex(shifts)
    @test ≈(shifts[I], ϕ.u[I], atol=0.01)
end

### Optimization with a temporal penalty
Qs = cat(3, eye(2,2), zeros(2,2), eye(2,2))
cs = cat(2, [5,-3], [0,0], [3,-1])
gridsize = (2,2)
denom = ones(15,15)
mms = tighten([quadratic(cs[:,t], Qs[:,:,t], denom) for i = 1:gridsize[1], j = 1:gridsize[2], t = 1:3])
mmis = RegisterPenalty.interpolate_mm!(mms)
knots = (linspace(1,100,gridsize[1]), linspace(1,99,gridsize[2]))
ap = RegisterPenalty.AffinePenalty(knots, 1.0)
u = randn(2, gridsize..., 3)
ϕs = tighten([GridDeformation(u[:,:,:,t], knots) for t = 1:3])
g = similar(u)
ϕs, fval = RegisterOptimize.optimize!(ϕs, identity, ap, 1.0, mmis)
c = 1/prod(gridsize)  # not sure about this
A = [2c+1 -1 0; -1 2 -1; 0 -1 2c+1]
target = (A\(2c*cs'))'
for (u, val) in ((ϕs[1].u, target[:,1]),
                 (ϕs[2].u, target[:,2]),
                 (ϕs[3].u, target[:,3]))
    for uv in u
        @test ≈(uv, val, atol=1e-2)  # look into why this is so low
    end
end

# Optimization with linear interpolation of mismatch data
fixed = 1:8
moving = fixed + 1
knots = (linspace(1,8,3),)
aperture_centers = [(1.0,), (4.5,), (8.0,)]
aperture_width = (3.5,)
mxshift = (2,)
gridsize = map(length, knots)
mms = mismatch_apertures(fixed, moving, aperture_centers, aperture_width, mxshift; normalization=:pixels)
E0 = zeros(size(mms))
cs = Array{Any}(size(mms))
Qs = Array{Any}(size(mms))
thresh = length(fixed)/prod(gridsize)/4
for i = 1:length(mms)
    E0[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt=false)
end
mmis = interpolate_mm!(mms, Linear())
λ = 0.001
ap = AffinePenalty{Float64,ndims(fixed)}(knots, λ)
ϕ, mismatch = RegisterOptimize.fixed_λ(cs, Qs, knots, ap, mmis)
@test mismatch < 1e-4
for i = 1:3
    @test -1.01 <= ϕ.u[i][1] <= -0.99
end
