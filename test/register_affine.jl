using StaticArrays, AffineTransforms, Interpolations, Base.Test
import BlockRegistration, RegisterOptimize
using RegisterCore, RegisterPenalty, RegisterDeformation, RegisterMismatch, RegisterFit
using Images, CoordinateTransformations, Rotations, RegisterOptimize

#### Rigid registration
fixed = sin.(linspace(0,pi,101)).*linspace(5,7,97)'
tform = tformrotate(pi/12)
moving = AffineTransforms.transform(fixed, tform)
tform0 = tformeye(2)
tfrm, fval = RegisterOptimize.optimize_rigid(fixed, moving, tform0, (20,21); tol=1e-2) # print_level=5)
tfprod = tform*tfrm
S = tfprod.scalefwd
@test abs(S[1,2]) < 0.05
offset = tfprod.offset
@test all(abs.(offset) .< 0.03)

###### Test rotation gridsearch

## 2D
#note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
a = rand(30,30)
b = AffineTransforms.transform(a, tformtranslate([2.0;0.0]) * tformrotate(pi/6))
tfm0 = tformtranslate([-2.0;0.0]) * tformrotate(-pi/6)
#note: maxshift must be GREATER than the true shift in order to find the true shift
tfm, mm = rotation_gridsearch(a, b, [11;11], [pi/6], [11])
@assert tfm.offset == tfm0.offset 
@assert tfm.scalefwd == tfm0.scalefwd

## 3D
#note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
a = rand(30,30,30)
b = AffineTransforms.transform(a, tformtranslate([2.0;0.0;0.0]) * tformrotate([1.0;0;0], pi/4))
tfm0 = tformtranslate([-2.0;0.0;0.0]) * tformrotate([1.0;0;0], -pi/4)
#note: maxshift must be GREATER than the true shift in order to find the true shift
tfm, mm = rotation_gridsearch(a, b, [3;3;3], [pi/4, pi/4, pi/4], [5;5;5])
@assert tfm.offset == tfm0.offset
@assert tfm.scalefwd == tfm0.scalefwd

###### Test QuadDIRECT-based registration

##### Translations
print("Testing translations...\n")
#2D
moving = rand(50,50)
tfm0 = Translation(-4.7, 5.1) #ground truth
newfixed = warp(moving, tfm0)
itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
etp = extrapolate(itp, NaN)
fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
mxshift = [10;10]
SD = eye(ndims(fixed))

tfm, mm = qd_translate(fixed, moving, mxshift, SD; thresh=thresh, rtol = 1e-5, atol = 1e-9 * thresh)
@test sum(abs.(tfm0.v - tfm.v)) < 1e-3

#3D
moving = rand(30,30,30)
tfm0 = Translation(-0.9, 2.1,1.2) #ground truth
newfixed = warp(moving, tfm0)
itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
etp = extrapolate(itp, NaN)
fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
mxshift = [5;5;5]
SD = eye(ndims(fixed))

tfm, mm = qd_translate(fixed, moving, mxshift, SD; thresh=thresh, rtol = 1e-5, atol = 1e-9 * thresh)

@test mm < 1e-4
@test sum(abs.(tfm0.v - tfm.v)) < 0.1

#Rotations + Translations
print("Testing rigid transforms...\n")
#2D
moving = rand(50,50)
tfm0 = Translation(-4.0, 5.0) ∘ recenter(RotMatrix(pi/360), center(moving)) #ground truth
newfixed = warp(moving, tfm0)
itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
etp = extrapolate(itp, NaN)
fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
mxshift = [10;10]
mxrot = pi/90
minwidth_rot = [0.0002]
SD = eye(ndims(fixed))

tfm, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD; thresh=thresh, rtol = 1e-5, atol = 1e-9 * thresh)

@test sum(abs.(tfm0.m - tfm.m)) < 1e-3

#3D
moving = rand(30,30,30)
tfm0 = Translation(-1.0, 2.1,1.2) ∘ recenter(RotXYZ(pi/360, pi/180, pi/220), center(moving)) #ground truth
newfixed = warp(moving, tfm0)
itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
etp = extrapolate(itp, NaN)
fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
mxshift = [5;5;5]
mxrot = [pi/90; pi/90; pi/90]
minwidth_rot = fill(0.0002, 3)
SD = eye(ndims(fixed))

tfm, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD; thresh=thresh, rtol = 1e-5, atol = 1e-9 * thresh)

@test mm < 1e-4
@test sum(abs.(vcat(tfm0.m[:], tfm0.v) - vcat(RotXYZ(tfm.m)[:], tfm.v))) < 0.1

#Test General Affine Transformations
print("Testing general affine transforms...\n")
#2D
#Random pixels are too difficult here, so we'll use an off-center square
moving = zeros(50,50)
moving[21:32, 21:32] = 1.0
moving = imfilter(moving, Kernel.gaussian(1.0))
shft = SArray{Tuple{2}}(rand(2).+2.0)
#random displacement from the identity matrix
mat = SArray{Tuple{2,2}}(eye(2) + rand(2,2)./20 + -rand(2,2)./20)
tfm0 = recenter(AffineMap(mat, shft), center(moving)) #ground truth
newfixed = warp(moving, tfm0)
itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
etp = extrapolate(itp, NaN)
fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
mxshift = [5;5]
SD = eye(ndims(fixed))

tfm, mm = qd_affine(fixed, moving, mxshift, SD; thresh=thresh, rtol = 1e-4, atol = 1e-9 * thresh)

@test sum(abs.(vcat(tfm0.m[:], tfm0.v) - vcat(tfm.m[:], tfm.v))) < 0.1

#3D
moving = zeros(10,10,10);
moving[4:8, 4:8, 4:8] = 1.0
shft = SArray{Tuple{3}}(1.6, 2.4, -0.3);
#random displacement from the identity matrix
mat = SArray{Tuple{3,3}}(eye(3) + rand(3,3)./20 + -rand(3,3)./20);
tfm00 = AffineMap(mat, shft);
tfm0 = recenter(tfm00, center(moving)); #ground truth
newfixed = warp(moving, tfm0);
itp = interpolate(newfixed, BSpline(Linear()), OnGrid());
etp = extrapolate(itp, NaN);
fixed = etp[indices(moving)...]; #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]));
mxshift = [3;3;3];
SD = eye(ndims(fixed));

@test RegisterOptimize.aff(vcat(tfm00.v[:], tfm00.m[:]), fixed, SD) == tfm0

tfm, mm = qd_affine(fixed, moving, mxshift, SD; thresh=thresh, maxevals=2e5, fvalue=6e-5, rtol = 1e-4, atol = 1e-9 * thresh);
#@show tfm0
#@show tfm

@test mm <= 6e-5
@test sum(abs.(vcat(tfm0.m[:], tfm0.v) - vcat(tfm.m[:], tfm.v))) < 0.1
