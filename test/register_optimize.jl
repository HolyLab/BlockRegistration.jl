using FixedSizeArrays, AffineTransforms, Interpolations, Base.Test
using RegisterCore, RegisterPenalty, RegisterDeformation
import RegisterOptimize

include("register_test_utilities.jl")

### Rigid registration
fixed = sin(linspace(0,pi,101)).*linspace(5,7,97)'
tform = tformrotate(pi/12)
moving = transform(fixed, tform)
tform0 = tformeye(2)
tfrm, fval = RegisterOptimize.optimize_rigid(fixed, moving, tform0, (20,21); tol=1e-2) # print_level=5)
tfprod = tform*tfrm
S = tfprod.scalefwd
@test abs(S[1,2]) < 0.05
offset = tfprod.offset
@test all(abs(offset) .< 0.03)

###
### Global-optimum initial guess
###
function initial_guess_direct(A, cs, Qs)
    b = zeros(size(A,1))
    Ac = copy(A)
    for i = 1:length(Qs)
        Ac[2i-1:2i,2i-1:2i] += Qs[i]
        b[2i-1:2i] = Qs[i]*cs[i]
    end
    x = Ac\b
    reinterpret(Vec{2,Float64}, x, gridsize)
end

knots = (linspace(1,20,4),linspace(1,15,4))
ap = AffinePenalty(knots, 1.0)
FF = ap.F*ap.F'
nA = 2*size(FF,1)
FF2 = zeros(nA,nA)
FF2[1:2:end,1:2:end] = FF
FF2[2:2:end,2:2:end] = FF
A = ap.λ*(I - FF2)
gridsize = map(length, knots)
Qs = Array(Any, gridsize)
cs = Array(Any, gridsize)

# Known exact answer
tfm = tformrotate(pi/12)
for (i,knot) in enumerate(eachknot(knots))
    v = [knot[1],knot[2]]
    cs[i] = tfm*v-v
    Qs[i] = eye(2,2)
end
ux = initial_guess_direct(A, cs, Qs)
#u = @inferred(RegisterOptimize.initial_guess(ap, cs, Qs))
u = RegisterOptimize.initial_deformation(ap, cs, Qs)
@test size(u) == size(ux)
@test eltype(u) == Vec{2,Float64}
for I in eachindex(u)
    @test_approx_eq u[I] cs[I]
    @test_approx_eq ux[I] cs[I]
end

# Random initialization
for I in CartesianRange(gridsize)
    QF = rand(2,2)
    Qs[I] = QF'*QF
    cs[I] = randn(2)
end
ux = initial_guess_direct(A, cs, Qs)
u = RegisterOptimize.initial_deformation(ap, cs, Qs)
@test size(u) == size(ux)
@test eltype(u) == Vec{2,Float64}
for I in eachindex(u)
    @test_approx_eq u[I] ux[I]
end


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
shifts = Array(Any, gridsize)
mxsv = zeros(2)
for (i,knot) in enumerate(eachknot(knots))
    knotv = [knot...]-cntr
    dx = tform*knotv - knotv
    mxsv = max(mxsv, abs(dx))
    shifts[i] = dx
end
# Create the fake mismatch data
m, n = 2ceil(Int,mxsv[1])+3, 2ceil(Int,mxsv[2])+3
nums = Array(Matrix{Float64}, gridsize)
for I in eachindex(nums)
    QF = rand(2,2)   # random quadratic component
    nums[I] = quadratic(m, n, shifts[I], QF*QF')
end
denom = ones(m, n)
mms = mismatcharrays(nums, denom)
mmis = interpolate_mm!(mms; BC=InPlaceQ)

u = randn(2, gridsize...)
ϕ = GridDeformation(u, knots)
λ = 1000.0
dp = AffinePenalty(knots, λ)
ϕ, fval = RegisterOptimize.optimize!(ϕ, identity, dp, mmis) #; print_level=5)
@test 0 <= fval <= 1e-5
for I in eachindex(shifts)
    @test_approx_eq_eps shifts[I] ϕ.u[I] 0.01
end
