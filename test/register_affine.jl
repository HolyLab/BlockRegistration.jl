using StaticArrays, AffineTransforms, Interpolations, Base.Test
import BlockRegistration, RegisterOptimize
using RegisterCore, RegisterPenalty, RegisterDeformation, RegisterMismatch, RegisterFit
using Images, CoordinateTransformations, Rotations, RegisterOptimize, TestImages

@testset "Derivative-based rigid registration" begin
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
end

@testset "Grid search rigid registration" begin
    ## 2D
    #note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
    a = rand(30,30)
    b = AffineTransforms.transform(a, tformtranslate([2.0;0.0]) * tformrotate(pi/6))
    tfm0 = tformtranslate([-2.0;0.0]) * tformrotate(-pi/6)
    #note: maxshift must be GREATER than the true shift in order to find the true shift
    tfm, mm = rotation_gridsearch(a, b, [11;11], [pi/6], [11])
    @test tfm.offset == tfm0.offset
    @test tfm.scalefwd == tfm0.scalefwd

    ## 3D
    #note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
    a = rand(30,30,30)
    b = AffineTransforms.transform(a, tformtranslate([2.0;0.0;0.0]) * tformrotate([1.0;0;0], pi/4))
    tfm0 = tformtranslate([-2.0;0.0;0.0]) * tformrotate([1.0;0;0], -pi/4)
    #note: maxshift must be GREATER than the true shift in order to find the true shift
    tfm, mm = rotation_gridsearch(a, b, [3;3;3], [pi/4, pi/4, pi/4], [5;5;5])
    @test tfm.offset == tfm0.offset
    @test tfm.scalefwd == tfm0.scalefwd
end

@testset "QuadDIRECT tests with random images" begin
    ##### Translations
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

    tfm, mm = qd_translate(fixed, moving, mxshift, SD; thresh=thresh, rtol=0, fvalue = 1e-8)

    @test mm <= 1e-8
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

    tfm, mm = qd_translate(fixed, moving, mxshift, SD; thresh=thresh, rtol=0, fvalue = 1e-5)

    @test mm <= 1e-5
    @test sum(abs.(tfm0.v - tfm.v)) < 0.1

    ######Rotations + Translations
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

    tfm, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD; thresh=thresh, rtol=0, fvalue=1e-8)

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

    tfm, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD; thresh=thresh, rtol=0, fvalue=1e-5)

    @test mm <= 1e-4
    @test sum(abs.(vcat(tfm0.m[:], tfm0.v) - vcat(RotXYZ(tfm.m)[:], tfm.v))) < 0.1

#NOTE: the 2D test below fails rarely and the 3D test fails often, apparently because full affine is too difficult with these images
#    #####General Affine Transformations
#    #2D
#    #Random pixels are too difficult here, so we'll use an off-center square
#    moving = zeros(50,50)
#    moving[21:32, 21:32] = 1.0
#    moving = imfilter(moving, Kernel.gaussian(1.0))
#    shft = SArray{Tuple{2}}(rand(2).+2.0)
#    #random displacement from the identity matrix
#    mat = SArray{Tuple{2,2}}(eye(2) + rand(2,2)./40 + -rand(2,2)./40)
#    tfm0 = recenter(AffineMap(mat, shft), center(moving)) #ground truth
#    newfixed = warp(moving, tfm0)
#    itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
#    etp = extrapolate(itp, NaN)
#    fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
#    thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
#    mxshift = [5;5]
#    SD = eye(ndims(fixed))
#
#    tfm, mm = qd_affine(fixed, moving, mxshift, SD; thresh=thresh, rtol=0, fvalue=1e-8)
#
#    @test sum(abs.(vcat(tfm0.m[:], tfm0.v) - vcat(tfm.m[:], tfm.v))) < 0.1
#
#    #3D
#    moving = zeros(10,10,10);
#    moving[5:7, 5:7, 5:7] = 1.0
#    shft = SArray{Tuple{3}}(0.6, 0.1, -0.3);
#    #random displacement from the identity matrix
#    mat = SArray{Tuple{3,3}}(eye(3) + rand(3,3)./50 + -rand(3,3)./50);
#    tfm00 = AffineMap(mat, shft);
#    tfm0 = recenter(tfm00, center(moving)); #ground truth
#    newfixed = warp(moving, tfm0);
#    inds = intersect.(indices(moving), indices(newfixed))
#    fixed = newfixed[inds...]
#    moving = moving[inds...]
#    thresh = 0.5 * sum(abs2.(fixed[.!(isnan.(fixed))]));
#    mxshift = [5;5;5];
#    SD = eye(ndims(fixed));
#    @test RegisterOptimize.aff(vcat(tfm00.v[:], tfm00.m[:]), fixed, SD) == tfm0
#
#    tfm, mm = qd_affine(fixed, moving, mxshift, SD; thresh=thresh, rtol=0, fvalue=1e-4);
#
#    @test mm <= 1e-4
#    @test sum(abs.(vcat(tfm0.m[:], tfm0.v) - vcat(tfm.m[:], tfm.v))) < 0.1
end

#Helper to generate test image pairs
function fixedmov(img, tfm)
    img2 = warp(img,tfm)
    inds = intersect.(indices(img), indices(img2))
    fixed = img[inds...]
    moving = img2[inds...]
	return fixed, moving
end

#helpers to convert Transformations to AffineMaps
to_affine(tfm::Translation) = AffineMap(eye(length(tfm.v)), tfm.v)
to_affine(tfm::LinearMap) = AffineMap(eye(length(tfm.v)), tfm.v)
to_affine(tfm::AffineMap) = tfm

#Helper to test that a found transform is (roughly) the inverse of the original transform
function tfmtest(tfm, tfminv)
    comp = to_affine(tfm ∘ tfminv)  #should be the identity transform
	diagtol = 0.09
	offdiagtol = 0.01
	vtol = 2
    @test all(x->(1-diagtol < x < 1+diagtol), diag(comp.m))
	@test all(x->(-offdiagtol < x < offdiagtol), comp.m.-diagm(diag(comp.m)))
    @test all(abs.(comp.v) .< vtol)
end

# tests with standard images
# (unlike the tests above these tests set up the problem so that the correct
# answer to is the inverse of an input transformation.  This seems to catch
# a different set of errors than the tests above)
@testset "QuadDIRECT tests with standard images" begin
    img = testimage("cameraman");
	img = Float64.(img)

	#Translation (subpixel)
    tfm = Translation(@SVector([14.3, 17.6]))
	fixed, moving = fixedmov(img, tfm)
    mxshift = (100,100) #make sure this isn't too small
    SD = diagm([pixelspacing(fixed)...])
    tform, mm = qd_translate(fixed, moving, mxshift, SD, rtol=0, fvalue=0.01)
	tfmtest(tfm, tform)

	#Rigid transform
    tfm = Translation(@SVector([14, 17]))∘LinearMap(RotMatrix(0.3)) #no distortion for now
	fixed, moving = fixedmov(img, tfm)
    mxshift = (100,100) #make sure this isn't too small
    mxrot = (0.5,)
    minwidth_rot = fill(0.002, 3)
    SD = diagm([pixelspacing(fixed)...])
    tform, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD, rtol=0, fvalue=0.01)
	tfmtest(tfm, tform)

	#Affine transform
    tfm = Translation(@SVector([14, 17]))∘LinearMap(RotMatrix(0.01))
	#make it harder with nonuniform scaling
	scale = @SMatrix [1.005 0; 0 0.995]
	tfm = AffineMap(tfm.m*scale, tfm.v)
    mxshift = (100,100) #make sure this isn't too small
	fixed, moving = fixedmov(img, tfm)
    SD = diagm([pixelspacing(fixed)...])
    tform, mm = qd_affine(fixed, moving, mxshift, rtol=0, fvalue=0.0002)
	tfmtest(tfm, tform)
end #tests with standard images
