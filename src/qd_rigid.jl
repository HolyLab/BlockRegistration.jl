####################  Utilities ##########################
function inds_intersect(img1, img2)
    inds1, inds2 = indices(img1), indices(img2)
    return ([intersect(x, y) for (x,y) in zip(inds1, inds2)]...)
end

function warp_and_intersect(moving, fixed, tfm)
    if tfm == IdentityTransformation()
        return moving, fixed
    end
    newmov = warp(moving, tfm)
    inds = inds_intersect(newmov, fixed)
    return newmov[inds...], view(fixed, inds...)
end

#Finds the best shift aligning moving to fixed, possibly after an initial transformation `initial_tfm`
#The shift returned should be composed with initial_tfm later to create the full transform
function best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=IdentityTransformation())
    moving, fixed = warp_and_intersect(moving, fixed, initial_tfm)
    mms = mismatch(fixed, moving, mxshift; normalization=normalization)
    best_i = indmin_mismatch(mms, thresh)
    return best_i.I, ratio(mms[best_i], 0.0, Inf)
end

#returns new minbounds and maxbounds with range sizes change by fac
function scalebounds(minb, maxb, fac::Float64)
    orng = maxb.-minb
    newradius = fac.*orng./2
    ctrs = minb.+(orng./2)
    return ctrs.-newradius, ctrs.+newradius
end

#Below two functions are applicable only to qd_affine()
#Returns two matrices describing a reasonable search space of linear transformation matrices
#representing fairly small changes in sample position/shape
#(Includes rotations, scaling, shear, etc)
#The space is centered on the identity matrix
function default_linmap_bounds(img::AbstractArray{T,N}) where {T, N}
    d=0.1; nd=0.1; #not sure yet whether off-diagonals should be treated differently
    deltas = fill(nd, N,N)
    for i=1:N
        deltas[i,i] = d
    end
    return eye(N).-deltas, eye(N).+deltas
end

function default_lin_minwidths(img::AbstractArray{T,N}) where {T,N}
    mat = fill(1e-2, N,N)
    for i = 1:N
        mat[i,i] = 1e-2
    end
    return mat[:]
end


###########  Rigid Transformation (rotation + translation) Search ############

#rotation + translation
function tfmrigid(x, img::AbstractArray{T,2}, SD=eye(2)) where {T}
    dx, dy, θ = x
    rt = rot(θ, img, SD)
    return Translation(dx, dy) ∘ rt
end
function tfmrigid(x, img::AbstractArray{T,3}, SD=eye(3)) where {T}
    dx, dy, dz, θx, θy, θz =  x
    rt = rot((θx, θy, θz), img, SD)
    return Translation(dx, dy, dz) ∘ rt
end

#rotation only
function rot(theta, img::AbstractArray{T,2}, SD=eye(2)) where {T}
    rotm = SD\RotMatrix(theta...)*SD
    return recenter(SMatrix{2,2}(rotm), center(img))
end
function rot(theta, img::AbstractArray{T,3}, SD=eye(3)) where {T}
    θx, θy, θz = theta
    rotm = RotMatrix(RotXYZ(θx,θy,θz))
    rotm = SD\rotm*SD
    return recenter(SMatrix{3,3}(rotm), center(img))
end

#translation only
tfmshift(x, img::AbstractArray{T,N}, SD=eye(N)) where {T,N} = Translation(x...)

#shift only slow because it warps for every shift instead of using fourier method
function translate_mm_slow(tfm, fixed, moving, thresh, SD; initial_tfm = IdentityTransformation())
    tfm = tfmshift(tfm, moving, SD) ∘ initial_tfm
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

#rotation + shift, slow because it warps for every rotation and shift
function rigid_mm_slow(tfm, fixed, moving, thresh, SD; initial_tfm = IdentityTransformation())
    tfm = tfmrigid(tfm, moving, SD) ∘ initial_tfm
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

#rotation + shift, fast because it uses fourier method for shift
function rigid_mm_fast(theta, mxshift, fixed, moving, thresh, SD; initial_tfm = IdentityTransformation())
    tfm = rot(theta, moving, SD) ∘ initial_tfm
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    bshft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity)
    return mm
end

function qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot, SD; initial_tfm = IdentityTransformation(), thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])), kwargs...)
    f(x) = rigid_mm_fast(x, mxshift, fixed, moving, thresh, SD; initial_tfm = initial_tfm) #note: if a trial rotation results in image overlap < thresh for all possible shifts then QuadDIRECT throws an error
    upper = [mxrot...]
    lower = -upper
    root_coarse, x0coarse = _analyze(f, lower, upper; maxevals=10^4, minwidth=minwidth_rot, print_interval=100, kwargs...)
    box_coarse = minimum(root_coarse)
    tfmcoarse0 = rot(position(box_coarse, x0coarse), moving) ∘ initial_tfm
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm = tfmcoarse0)
    @show best_shft
    tfmcoarse = Translation(best_shft) ∘ tfmcoarse0
    return tfmcoarse, mm
end

function qd_rigid_fine(fixed, moving, mxrot, minwidth_rot, SD; initial_tfm = IdentityTransformation(), thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])), kwargs...)
    f2(x) = rigid_mm_slow(x, fixed, moving, thresh, SD; initial_tfm = initial_tfm)
    upper_shft = fill(1.0, ndims(fixed))
    upper_rot = mxrot
    upper = vcat(upper_shft, upper_rot)
    lower = -upper
    minwidth_shfts = fill(0.01, ndims(fixed))
    #minwidth_rots = ndims(fixed) == 2 ? [0.0001;] : fill(0.0001, 3) #assume 2 or 3 dimensional input
    minwidth = vcat(minwidth_shfts, minwidth_rot)
    root, x0 = _analyze(f2, lower, upper; maxevals=10^4, minwidth=minwidth, print_interval=100, kwargs...)
    box = minimum(root)
    tfmfine = tfmrigid(position(box, x0), moving, SD) ∘ initial_tfm
    return tfmfine, value(box)
end

"""
`tform, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD = eye;  thresh=thresh, tfm0=IdentityTransformation(), kwargs...)`
optimizes a rigid transformation (rotation + shift) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm.  The algorithm is run twice: the first step uses a fourier method to speed up
the search for the best whole-pixel shift.  The second step refines the search for sub-pixel accuracy. `kwargs...` can include any
keyword argument that can be passed to `QuadDIRECT.analyze`. It's recommended that you pass your own stopping criteria when possible
(i.e. `rtol`, `atol`, and/or `fvalue`).

Use `SD` if your axes are not uniformly sampled, for example `SD = diagm(voxelspacing)` where `voxelspacing`
is a vector encoding the spacing along all axes of the image. `thresh` enforces a certain amount of sum-of-squared-intensity overlap between
the two images; with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.

If you have a good initial guess at the solution, pass it with the `tfm0` kwarg to jump-start the search.

Note that the rotation returned is about the center of the moving image.  Therefore if you try to use the transform to warp an image of a size other than `size(moving)` then the rotation will no longer be centered.  If you want to keep the rotation centered you must call `recenter(tfm, newctr)` where `newctr` is the displacement of the new center from the old center.
"""
function qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD=eye(ndims(fixed)); thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])), tfm0 = IdentityTransformation(), kwargs...)
    mxrot = [mxrot...]
    print("Running coarse step\n")
    tfm_coarse, mm_coarse = qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot, SD; initial_tfm = tfm0, thresh = thresh, kwargs...)
    print("Running fine step\n")
    final_tfm, mm_fine = qd_rigid_fine(fixed, moving, mxrot./10, minwidth_rot, SD; initial_tfm = tfm_coarse, thresh = thresh, kwargs...)
    return final_tfm, mm_fine
end

####################  Translation Search ##########################

function qd_translate_fine(fixed, moving, SD; initial_tfm = IdentityTransformation(), minwidth = fill(0.01, ndims(fixed)), thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])), kwargs...)
    f(x) = translate_mm_slow(x, fixed, moving, thresh, SD; initial_tfm = initial_tfm)
    upper = fill(1.0, ndims(fixed))
    lower = -upper
    root, x0 = _analyze(f, lower, upper; maxevals=10^4, minwidth=minwidth, print_interval=100, kwargs...)
    box = minimum(root)
    tfmfine = tfmshift(position(box, x0), moving) ∘ initial_tfm
    return tfmfine, value(box)
end

"""
`tform, mm = qd_translate(fixed, moving, mxshift; thresh=thresh, [SD = eye], kwargs...)`
optimizes a simple shift (translation) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm.  The algorithm is run twice: the first step uses a fourier method to speed up
the search for the best whole-pixel shift.  The second step refines the search for sub-pixel accuracy.
The default precision of this step is 1% of one pixel (0.01) for each dimension of the image.
You can override the default with the `minwidth` argument.  `kwargs...` can also include 
any other keyword argument that can be passed to `QuadDIRECT.analyze`.
It's recommended that you pass your own stopping criteria when possible (i.e. `rtol`, `atol`, and/or `fvalue`).

If you have a good initial guess at the solution, pass it with the `tfm0` kwarg to jump-start the search.

Use `SD` if your axes are not uniformly sampled, for example `SD = diagm(voxelspacing)` where `voxelspacing`
is a vector encoding the spacing along all axes of the image. `thresh` enforces a certain amount of sum-of-squared-intensity 
overlap between the two images; with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.
"""
function qd_translate(fixed, moving, mxshift, SD=eye(ndims(fixed)); thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])), tfm0 = IdentityTransformation(), minwidth=fill(0.01, ndims(fixed)), kwargs...)
    print("Running coarse step\n")
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm = tfm0)
    @show best_shft
    tfm_coarse = Translation(best_shft) ∘ tfm0
    print("Running fine step\n")
    return qd_translate_fine(fixed, moving, SD; initial_tfm = tfm_coarse, thresh = thresh, minwidth=minwidth, kwargs...)
end

####################  Affine Transformation Search ##########################

#note: recentering doesn't make sense for general affine transforms
amend_tfm(tfm::AffineMap, tfm0::T) where {T<:AffineMap} = AffineMap(tfm.m*tfm0.m, tfm0.v+tfm.v)
amend_tfm(tfm::AffineMap, tfm0::T) where {T<:Translation} = AffineMap(tfm.m, tfm0.v+tfm.v) #same as composing
amend_tfm(tfm::LinearMap, tfm0::T) where {T<:AffineMap} = AffineMap(tfm.m*tfm0.m, tfm0.v)
amend_tfm(tfm::LinearMap, tfm0::T) where {T<:Translation} = AffineMap(tfm.m, tfm0.v) #not same as composing
amend_tfm(tfm::LinearMap, tfm0::IdentityTransformation) = tfm
amend_tfm(tfm::AffineMap, tfm0::IdentityTransformation) = tfm

#linear map + translation applied after tfm0, but centered on the center of img (assumes tfm0 is already centered on img)
#(so this is not exactly newtfm ∘ tfm0)
function aff(params, img::AbstractArray{T,N}, SD=eye(N), tfm0=IdentityTransformation()) where {T,N}
    params = [params...]
    offs = params[1:N]
    linm = linmap(params[(N+1):end], img, SD; rectr=false)
    newtfm = recenter(AffineMap(linm.m, SArray{Tuple{N}}(Float64.(offs))), center(img))
    return amend_tfm(newtfm, tfm0)
end

#linear map applied after tfm0, but centered on the center of img (assumes tfm0 is already centered on img)
function linmap(mat, img::AbstractArray{T,N}, SD=eye(N), tfm0=IdentityTransformation(); rectr=true) where {T,N}
    mat = [mat...]
    mat = SD\reshape(mat, N,N)*SD
    if rectr
        return recenter(amend_tfm(LinearMap(SMatrix{N,N}(mat)), tfm0), center(img))
    else
        return amend_tfm(LinearMap(SMatrix{N,N}(mat)), tfm0)
    end
end

#here tfm contains parameters of a linear map
function affine_mm_fast(tfm, mxshift, fixed, moving, thresh, SD; initial_tfm = IdentityTransformation())
    tfm = linmap(tfm, moving, SD, initial_tfm)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    bshft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity)
    return mm
end

#here tfm contains parameters of an affine transform (linear map + shift)
function affine_mm_slow(tfm, fixed, moving, thresh, SD; initial_tfm = IdentityTransformation())
    tfm = aff(tfm, moving, SD, initial_tfm) # ∘ initial_tfm #note: should the aff() computed tform be recentered according to the offset of initial_tfm?
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

#sets splits based on lower and upper bounds
function _analyze(f, lower, upper; kwargs...)
    splits = ([[lower[i]; lower[i]+(upper[i]-lower[i])/2; upper[i]] for i=1:length(lower)]...)
    analyze(f, splits, lower, upper; kwargs...)
end

#Note the fourier trick just doesn't work well here, too many dimensions. So this is the same as qd_affine_fine for now except that
#mxshift can be larger and maxevals are handled differently
function qd_affine_coarse(fixed, moving, mxshift, linmins, linmaxs, SD; initial_tfm = IdentityTransformation(), thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])), minwidth = default_lin_minwidths(moving), kwargs...)
    f(x) = affine_mm_slow(x, fixed, moving, thresh, SD; initial_tfm = initial_tfm)
    upper = vcat(mxshift, linmaxs)
    lower = vcat(-mxshift, linmins)
    minwidth_shfts = fill(0.01, ndims(fixed))
    minwidth = vcat(minwidth_shfts, minwidth)
    root, x0 = _analyze(f, lower, upper; minwidth=minwidth, print_interval=100, kwargs..., maxevals=5e4) # nquasinewton=3^length(lower), kwargs...)
    #f(x) = affine_mm_fast(x, mxshift, fixed, moving, thresh, SD; initial_tfm = initial_tfm)
    #root, x0 = _analyze(f, linmins, linmaxs; minwidth=minwidth, print_interval=100, kwargs..., maxevals=10^4) #nquasinewton = 3^length(linmins), kwargs...)
    box = minimum(root)
    params = position(box, x0)
    tfmcoarse = aff(params, moving, SD, initial_tfm)
    #tfmcoarse0 = linmap(position(box, x0), moving, SD, initial_tfm)
    #best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm = tfmcoarse0)
    #@show best_shft
    #tfmcoarse = Translation(best_shft) ∘ tfmcoarse0
    #return tfmcoarse, mm
    return tfmcoarse, value(box)
end

function qd_affine_fine(fixed, moving, linmins, linmaxs, SD; initial_tfm = IdentityTransformation(), thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])), minwidth_mat = default_lin_minwidths(fixed)./10, maxevals=2e4, kwargs...)
    f(x) = affine_mm_slow(x, fixed, moving, thresh, SD; initial_tfm = initial_tfm)
    upper_shft = fill(2.0, ndims(fixed))
    upper = vcat(upper_shft, linmaxs)
    lower = vcat(-upper_shft, linmins)
    minwidth_shfts = fill(0.01, ndims(fixed))
    minwidth = vcat(minwidth_shfts, minwidth_mat)
    root, x0 = _analyze(f, lower, upper; minwidth=minwidth, print_interval=100, maxevals=maxevals, kwargs...) # nquasinewton=3^length(lower), kwargs...)
    box = minimum(root)
    params = position(box, x0)
    tfmfine = aff(params, moving, SD, initial_tfm)
    return tfmfine, value(box)
end

"""
`tform, mm = qd_affine(fixed, moving, mxshift, linmins, linmaxs, SD = eye; thresh=thresh,  kwargs...)`
optimizes an affine transformation (linear map + translation)to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm.  The algorithm is run twice: the first step samples the search space 
at a coarser resolution than the second.

The `linmins` and `linmaxs` arguments set the minimum and maximum allowable values in the linear map matrix.
They can be supplied as NxN matrices or flattened vectors.  If omitted then a modest default search space is chosen.
`mxshift` sets the magnitude of the largest allowable translation in each dimension (It's a vector of length N).

`kwargs...` can also include any other keyword argument that can be passed to `QuadDIRECT.analyze`.
It's recommended that you pass your own stopping criteria when possible (i.e. `rtol`, `atol`, and/or `fvalue`).

If you have a good initial guess at the solution, pass it with the `tfm0` kwarg to jump-start the search.

Use `SD` if your axes are not uniformly sampled, for example `SD = diagm(voxelspacing)` where `voxelspacing`
is a vector encoding the spacing along all axes of the image. `thresh` enforces a certain amount of sum-of-squared-intensity 
overlap between the two images; with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.

Note that the transform returned is "centered" (see docs for `recenter()`) at the center of the moving image.
Therefore if you try to use the transform to warp an image of a size other than `size(moving)` then the result will not be as expected.
If you want to keep the rotation centered you must call `recenter(tfm, newctr)` where `newctr` is the displacement of the new center from the old center.
"""
function qd_affine(fixed, moving, mxshift, linmins, linmaxs, SD=eye(ndims(fixed)); thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])), tfm0 = IdentityTransformation(), kwargs...)
    linmins = [linmins...]
    linmaxs = [linmaxs...]
    print("Running coarse step\n")
    mw = default_lin_minwidths(moving) #fill(1e-3, length(linmins))
    tfm_coarse1, mm_coarse1 = qd_affine_coarse(fixed, moving, mxshift, linmins, linmaxs, SD; minwidth = mw, initial_tfm = tfm0, thresh = thresh, kwargs...)
    print("Running fine step\n")
    mw = mw./100
    #linmins, linmaxs = scalebounds(linmins, linmaxs, 0.5)
    final_tfm, final_mm = qd_affine_fine(fixed, moving, linmins, linmaxs, SD; minwidth_mat=mw, initial_tfm = tfm_coarse1, thresh = thresh, kwargs...)
    return final_tfm, final_mm
end

function qd_affine(fixed, moving, mxshift, SD=eye(ndims(fixed)); thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])), tfm0 = IdentityTransformation(), kwargs...)
    minb, maxb = default_linmap_bounds(fixed)
    return qd_affine(fixed, moving, mxshift, minb, maxb, SD; thresh=thresh, tfm0=tfm0, kwargs...)
end
