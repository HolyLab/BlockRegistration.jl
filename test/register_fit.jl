import BlockRegistration, RegisterFit
using Base.Test, AffineTransforms, Interpolations
using RegisterCore

include("register_test_utilities.jl")

### qfit

denom = ones(11,11)
Q = rand(2,2); Q = Q'*Q
num = quadratic(11, 11, [1,-2], Q)
E0, cntr, Qf = @inferred(RegisterFit.qfit(MismatchArray(num, denom), 1e-3))
@test abs(E0) < eps()
@test_approx_eq cntr [1,-2]
@test_approx_eq Qf Q

num = num+5
E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), 1e-3)
@test_approx_eq E0 5
@test_approx_eq cntr [1,-2]
@test_approx_eq Qf Q

num = quadratic(11, 13, [2,-4], Q)
thresh = 1e-3
scale = rand(size(num)) + thresh
denom = ones(size(num)).*scale
@test all(denom .> thresh)
num = num.*scale
E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), thresh)
@test abs(E0) < eps()
@test_approx_eq cntr [2,-4]
@test_approx_eq Qf Q

# Degenerate solutions
Q = [1 0; 0 0]
denom = ones(13, 11)
num = quadratic(13, 11, [2,-4], Q)
E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), thresh)
@test abs(E0) < eps()
@test_approx_eq cntr[1] 2
@test_approx_eq Qf Q
a = rand(2)+0.1
Q = a*a'
num = quadratic(13, 11, [2,-4], Q)
E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), thresh)
@test abs(E0) < eps()
@test abs(dot(cntr-[2,-4], a)) < eps()
@test_approx_eq_eps Qf Q 1e-12

# Settings with very few above-threshold data points
# Just make sure there are no errors
denom0 = ones(5,5)
Q = rand(2,2); Q = Q'*Q
num0 = quadratic(5, 5, [0,0], Q)
denom = copy(denom0); denom[1:2,1] *= 100; denom[5,5] *= 100
num = copy(num0); num[1:2,1] *= 100; num[5,5] *= 100
thresh = 2
E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), thresh)

### qbuild
A = RegisterFit.qbuild(2, [-1,1], [0.3 0; 0 0.5], (5,5))
v1 = 0.3*((-5:5)+1).^2
v2 = 0.5*((-5:5)-1).^2
@test_approx_eq A.data v1.+v2'+2

### Principal Axes Transformation

fixed = zeros(10,11)
fixed[2,3:7] = 1
fixed[3,2:8] = 1
moving = zeros(10,11)
moving[3:7,8] = 1
moving[2:8,7] = 1
fmean, fvar = RegisterFit.principalaxes(fixed)
tfm = RegisterFit.pat_rotation((fmean, fvar), moving)
for i = 1:2
    S = tfm[i].scalefwd
    @test_approx_eq abs(S[1,2]) 1
    @test_approx_eq abs(S[2,1]) 1
    @test abs(S[1,1]) < 1e-8
    @test abs(S[2,2]) < 1e-8
end

F = Images.meanfinite(abs(fixed), (1,2))[1]
df = zeros(2)
movinge = extrapolate(interpolate(moving, BSpline(Linear()), OnGrid()), NaN)
for i = 1:2
    mov = TransformedArray(movinge, tfm[i])
    df[i] = Images.meanfinite(abs(fixed-transform(mov)), (1,2))[1]
end
@test minimum(df) < 1e-4*F

nothing
