#################### Begin Utilities ##########################
function warp_and_intersect(moving, fixed, tfm)
    if tfm == IdentityTransformation()
        if indices(moving) == indices(fixed)
            return moving, fixed
        end
    else
        moving = warp(moving, tfm)
    end
    inds = intersect.(indices(moving), indices(fixed))
    #TODO: use views after BlockRegistration #83 on Github is addressed
    return moving[inds...], fixed[inds...]
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
function scalebounds(minb, maxb, fac::Real)
    orng = maxb.-minb
    newradius = fac.*orng./2
    ctrs = minb.+(orng./2)
    return ctrs.-newradius, ctrs.+newradius
end

"""
minb, maxb = default_linmap_bounds(img::AbstractArray{T,N}; dmax=0.05, ndmax=0.05) where {T,N}

Returns two matrices describing a search space of linear transformation matrices.
(Linear transformation matrices can encode rotations, scaling, shear, etc)
`minb` and  `maxb` contain the minimum and maximum acceptable values of an
NxN transformation matrix.  The center of the space is the identity matrix.
The size of the space can be specified with `dmax` and `ndmax` kwargs.
These represent the maximum (absolute) difference from the identity matrix for elements
along the diagonal and off the diagnonal, respectively.
e.g. `dmax=0.05` implies that diagonal elements can range from 0.95 to 1.05.
The space is centered on the identity matrix
"""
function default_linmap_bounds(img::AbstractArray{T,N}; dmax=0.05, ndmax=0.05) where {T, N}
    deltas = fill(abs(ndmax), N,N)
    for i=1:N
        deltas[i,i] = abs(dmax)
    end
    return eye(N).-deltas, eye(N).+deltas
end

"""
m = default_lin_minwidths(img::AbstractArray{T,N}; dmin=1e-3, ndmin=1e-3) where {T,N}

Returns a NxN matrix describing granularity of a search space of linear transformation matrices.
This can be useful for setting the `minwidth` parameter of QuadDIRECT when performing a
full affine registration. `dmin` and `ndmin` set the tolerances for diagonal and
off-diagonal elements of the linear transformation matrix, respectively.
"""
function default_lin_minwidths(img::AbstractArray{T,N}; dmin=1e-5, ndmin=1e-5) where {T, N}
    mat = fill(abs(ndmin), N,N)
    for i=1:N
        mat[i,i] = abs(dmin)
    end
    return mat[:]
end

#sets splits based on lower and upper bounds
function _analyze(f, lower, upper; kwargs...)
    splits = ([[lower[i]; lower[i]+(upper[i]-lower[i])/2; upper[i]] for i=1:length(lower)]...)
    QuadDIRECT.analyze(f, splits, lower, upper; kwargs...)
end

#################### End Utilities ##########################


####################  Translation Search ##########################

function tfmshift(params, img::AbstractArray{T,N}) where {T,N}
    length(params) == N || throw(DimensionMismatch("expected $N parameters, got $(length(params))"))
    return Translation(params...)
end

#slow because it warps for every shift instead of using fourier method
function translate_mm_slow(params, fixed, moving, thresh; initial_tfm=IdentityTransformation())
    tfm = initial_tfm ∘ tfmshift(params, moving)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

function qd_translate_fine(fixed, moving;
                           initial_tfm=IdentityTransformation(),
                           minwidth=fill(0.01, ndims(fixed)),
                           thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                           kwargs...)
    f(x) = translate_mm_slow(x, fixed, moving, thresh; initial_tfm=initial_tfm)
    upper = fill(1.0, ndims(fixed))
    lower = -upper
    root, x0 = _analyze(f, lower, upper; minwidth=minwidth, print_interval=100, maxevals=5e4, kwargs...)
    box = minimum(root)
    tfmfine = initial_tfm ∘ tfmshift(position(box, x0), moving)
    return tfmfine, value(box)
end

"""
`tform, mm = qd_translate(fixed, moving, mxshift; thresh=thresh, kwargs...)`
optimizes a simple shift (translation) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm with the constraint that no shifts larger than
``mxshift` will be considered.

Both `mxshift` and the returned translation are specified in terms of pixel units, so the
algorithm need not be aware of anisotropic sampling.

The algorithm involves two steps: the first step uses a fourier method to speed up
the search for the best whole-pixel shift.  The second step refines the search for sub-pixel accuracy.
The default precision of this step is 1% of one pixel (0.01) for each dimension of the image.
You can override the default with the `minwidth` argument.  `kwargs...` can also include 
any other keyword argument that can be passed to `QuadDIRECT.analyze`.
It's recommended that you pass your own stopping criteria when possible (i.e. `rtol`, `atol`, and/or `fvalue`).

If you have a good initial guess at the solution, pass it with the `initial_tfm` kwarg to jump-start the search.
`thresh` enforces a certain amount of sum-of-squared-intensity overlap between the two images;
with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.
"""
function qd_translate(fixed, moving, mxshift;
                      thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                      initial_tfm=IdentityTransformation(),
                      minwidth=fill(0.01, ndims(fixed)), kwargs...)
    fixed, moving = float(fixed), float(moving)
    print("Running coarse step\n")
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=initial_tfm)
    tfm_coarse = initial_tfm ∘ Translation(best_shft)
    print("Running fine step\n")
    return qd_translate_fine(fixed, moving; initial_tfm=tfm_coarse, thresh=thresh, minwidth=minwidth, kwargs...)
end

###########  Rigid Transformation (rotation + translation) Search ############

update_SD(SD, tfm::Union{LinearMap, AffineMap}) = update_SD(SD, tfm.linear)
update_SD(SD, tfm::Transformation) = SD
update_SD(SD::AbstractArray, m::StaticArray) = update_SD(SD, Array(m))
update_SD(SD::AbstractArray, m::AbstractArray) = m\SD*m

#rotation only
function rot(theta, img::AbstractArray{T,2}, SD=eye(2)) where {T}
    rotm = SD\RotMatrix(theta...)*SD
    return LinearMap(SMatrix{2,2}(rotm))
end
function rot(thetas, img::AbstractArray{T,3}, SD=eye(3)) where {T}
    length(thetas) == 3 || throw(DimensionMismatch("expected 3 parameters, got $(length(thetas))"))
    θx, θy, θz = thetas
    rotm = RotMatrix(RotXYZ(θx,θy,θz))
    rotm = SD\rotm*SD
    return LinearMap(SMatrix{3,3}(rotm))
end
#rotation + shift, fast because it uses fourier method for shift
function rigid_mm_fast(theta, mxshift, fixed, moving, thresh, SD; initial_tfm=IdentityTransformation())
    tfm = initial_tfm ∘ rot(theta, moving, update_SD(SD, initial_tfm))
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    bshft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity)
    return mm
end

#rotation + translation
function tfmrigid(params, img::AbstractArray{T,2}, SD=eye(2)) where {T}
    length(params) == 3 || throw(DimensionMismatch("expected 3 parameters, got $(length(params))"))
    dx, dy, θ = params
    rt = rot(θ, img, SD)
    return Translation(dx, dy) ∘ rt
end
function tfmrigid(params, img::AbstractArray{T,3}, SD=eye(3)) where {T}
    length(params) == 6 || throw(DimensionMismatch("expected 6 parameters, got $(length(params))"))
    dx, dy, dz, θx, θy, θz =  params
    rt = rot((θx, θy, θz), img, SD)
    return Translation(dx, dy, dz) ∘ rt
end
#rotation + shift, slow because it warps for every rotation and shift
function rigid_mm_slow(params, fixed, moving, thresh, SD; initial_tfm=IdentityTransformation())
    tfm = initial_tfm ∘ tfmrigid(params, moving, update_SD(SD, initial_tfm))
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

function qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot, SD;
                         initial_tfm=IdentityTransformation(),
                         thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                         kwargs...)
    #note: if a trial rotation results in image overlap < thresh for all possible shifts then QuadDIRECT throws an error
    f(x) = rigid_mm_fast(x, mxshift, fixed, moving, thresh, SD; initial_tfm=initial_tfm)
    upper = [mxrot...]
    lower = -upper
    root_coarse, x0coarse = _analyze(f, lower, upper;
                                     minwidth=minwidth_rot, print_interval=100, maxevals=5e4, kwargs..., atol=0, rtol=1e-3)
    box_coarse = minimum(root_coarse)
    tfmcoarse0 = initial_tfm ∘ rot(position(box_coarse, x0coarse), moving, update_SD(SD, initial_tfm))
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=tfmcoarse0)
    tfmcoarse = tfmcoarse0 ∘ Translation(best_shft)
    return tfmcoarse, mm
end

function qd_rigid_fine(fixed, moving, mxrot, minwidth_rot, SD;
                       initial_tfm=IdentityTransformation(),
                       thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                       kwargs...)
    f(x) = rigid_mm_slow(x, fixed, moving, thresh, SD; initial_tfm=initial_tfm)
    upper_shft = fill(2.0, ndims(fixed))
    upper_rot = mxrot
    upper = vcat(upper_shft, upper_rot)
    lower = -upper
    minwidth_shfts = fill(0.005, ndims(fixed))
    minwidth = vcat(minwidth_shfts, minwidth_rot)
    root, x0 = _analyze(f, lower, upper; minwidth=minwidth, print_interval=100, maxevals=5e4, kwargs...)
    box = minimum(root)
    tfmfine = initial_tfm ∘ tfmrigid(position(box, x0), moving, update_SD(SD, initial_tfm))
    return tfmfine, value(box)
end

"""
`tform, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD=eye;  thresh=thresh, initial_tfm=IdentityTransformation(), kwargs...)`
optimizes a rigid transformation (rotation + shift) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm.  The algorithm is run twice: the first step uses a fourier method to speed up
the search for the best whole-pixel shift.  The second step refines the search for sub-pixel accuracy. `kwargs...` can include any
keyword argument that can be passed to `QuadDIRECT.analyze`. It's recommended that you pass your own stopping criteria when possible
(i.e. `rtol`, `atol`, and/or `fvalue`).  If you provide `rtol` and/or `atol` they will apply only to the second (fine) step of the registration;
the user may not adjust these criteria for the coarse step.

The rotation returned will be centered on the origin-of-coordinates, i.e. (0,0) for a 2D image.  Usually it is more natural to consider rotations
around the center of the image.  If you would like `mxrot` and the returned rotation to act relative to the center of the image, then you must
move the origin to the center of the image by calling `centered(img)` from the `ImageTransformations` package.  Call `centered` on both the
fixed and moving image to generate the `fixed` and `moving` that you provide as arguments.  If you later want to apply the returned transform
to an image you must remember to call `centered` on that image as well.  Alternatively you can re-encode the transformation in terms of a
different origin by calling `recenter(tform, newctr)` where `newctr` is the displacement of the new center from the old center.

Use `SD` if your axes are not uniformly sampled, for example `SD = diagm(voxelspacing)` where `voxelspacing`
is a vector encoding the spacing along all axes of the image. `thresh` enforces a certain amount of sum-of-squared-intensity overlap between
the two images; with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.

If you have a good initial guess at the solution, pass it with the `initial_tfm` kwarg to jump-start the search.
"""
function qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD=eye(ndims(fixed));
                  thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                  initial_tfm=IdentityTransformation(),
                  kwargs...)
    fixed, moving = float(fixed), float(moving)
    mxrot = [mxrot...]
    print("Running coarse step\n")
    tfm_coarse, mm_coarse = qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot, SD; initial_tfm=initial_tfm, thresh=thresh, kwargs...)
    print("Running fine step\n")
    final_tfm, mm_fine = qd_rigid_fine(fixed, moving, mxrot./2, minwidth_rot, SD; initial_tfm=tfm_coarse, thresh=thresh, kwargs...)
    return final_tfm, mm_fine
end


####################  Affine Transformation Search ##########################

function linmap(mat, img::AbstractArray{T,N}, SD=eye(N), initial_tfm=IdentityTransformation()) where {T,N}
    mat = [mat...]
    SD = update_SD(SD, initial_tfm)
    mat = SD\reshape(mat, N,N)*SD
    lm = LinearMap(SMatrix{N,N}(mat))
    return initial_tfm ∘ lm
end

#here params contains parameters of a linear map
function affine_mm_fast(params, mxshift, fixed, moving, thresh, SD; initial_tfm=IdentityTransformation())
    tfm = linmap(params, moving, SD, initial_tfm)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    bshft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity)
    return mm
end

function aff(params, img::AbstractArray{T,N}, SD=eye(N), initial_tfm=IdentityTransformation()) where {T,N}
    params = [params...]
    length(params) == (N+N^2) || throw(DimensionMismatch("expected $(N+N^2) parameters, got $(length(params))"))
    offs = Float64.(params[1:N])
    mat = Float64.(params[(N+1):end])
    SD = update_SD(SD, initial_tfm)
    mat = SD\reshape(mat,N,N)*SD
    return initial_tfm ∘ AffineMap(SMatrix{N,N}(mat), SVector{N}(offs))
end

#here tfm contains parameters of an affine transform (linear map + shift)
function affine_mm_slow(params, fixed, moving, thresh, SD; initial_tfm=IdentityTransformation())
    tfm = aff(params, moving, SD, initial_tfm)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

function qd_affine_coarse(fixed, moving, mxshift, linmins, linmaxs, SD;
                          initial_tfm=IdentityTransformation(),
                          thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                          minwidth=default_lin_minwidths(moving),
                          kwargs...)
    f(x) = affine_mm_fast(x, mxshift, fixed, moving, thresh, SD; initial_tfm=initial_tfm)
    upper = linmaxs
    lower = linmins
    root, x0 = _analyze(f, lower, upper;
                        minwidth=minwidth, print_interval=100, maxevals=5e4, kwargs..., atol=0, rtol=1e-3)
    box = minimum(root)
    params = position(box, x0)
    tfmcoarse0 = linmap(params, moving, SD, initial_tfm)
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=tfmcoarse0)
    tfmcoarse = tfmcoarse0 ∘ Translation(best_shft)
    return tfmcoarse, mm
end

function qd_affine_fine(fixed, moving, linmins, linmaxs, SD;
                        initial_tfm=IdentityTransformation(),
                        thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                        minwidth_mat=default_lin_minwidths(fixed)./10,
                        kwargs...)
    f(x) = affine_mm_slow(x, fixed, moving, thresh, SD; initial_tfm=initial_tfm)
    upper_shft = fill(2.0, ndims(fixed))
    upper = vcat(upper_shft, linmaxs)
    lower = vcat(-upper_shft, linmins)
    minwidth_shfts = fill(0.01, ndims(fixed))
    minwidth = vcat(minwidth_shfts, minwidth_mat)
    root, x0 = _analyze(f, lower, upper;
                        minwidth=minwidth, print_interval=100, maxevals=5e4, kwargs...)
    box = minimum(root)
    params = position(box, x0)
    tfmfine = aff(params, moving, SD, initial_tfm)
    return tfmfine, value(box)
end

"""
`tform, mm = qd_affine(fixed, moving, mxshift, linmins, linmaxs, SD=eye; thresh, initial_tfm, kwargs...)`
`tform, mm = qd_affine(fixed, moving, mxshift, SD=eye; thresh, initial_tfm, kwargs...)`
optimizes an affine transformation (linear map + translation) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm.  The algorithm is run twice: the first step samples the search space 
at a coarser resolution than the second.  `kwargs...` may contain any keyword argument that can be passed to
`QuadDIRECT.analyze`. It's recommended that you pass your own stopping criteria when possible
(i.e. `rtol`, `atol`, and/or `fvalue`).  If you provide `rtol` and/or `atol` they will apply only to the
second (fine) step of the registration; the user may not adjust these criteria for the coarse step.

`tform` will be centered on the origin-of-coordinates, i.e. (0,0) for a 2D image.  Usually it is more natural to consider rotations
around the center of the image.  If you would like `mxrot` and the returned rotation to act relative to the center of the image, then you must
move the origin to the center of the image by calling `centered(img)` from the `ImageTransformations` package.  Call `centered` on both the
fixed and moving image to generate the `fixed` and `moving` that you provide as arguments.  If you later want to apply the returned transform
to an image you must remember to call `centered` on that image as well.  Alternatively you can re-encode the transformation in terms of a
different origin by calling `recenter(tform, newctr)` where `newctr` is the displacement of the new center from the old center.

The `linmins` and `linmaxs` arguments set the minimum and maximum allowable values in the linear map matrix.
They can be supplied as NxN matrices or flattened vectors.  If omitted then a modest default search space is chosen.
`mxshift` sets the magnitude of the largest allowable translation in each dimension (It's a vector of length N).

`kwargs...` can also include any other keyword argument that can be passed to `QuadDIRECT.analyze`.
It's recommended that you pass your own stopping criteria when possible (i.e. `rtol`, `atol`, and/or `fvalue`).

If you have a good initial guess at the solution, pass it with the `initial_tfm` kwarg to jump-start the search.

Use `SD` if your axes are not uniformly sampled, for example `SD = diagm(voxelspacing)` where `voxelspacing`
is a vector encoding the spacing along all axes of the image. `thresh` enforces a certain amount of sum-of-squared-intensity 
overlap between the two images; with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.
"""
function qd_affine(fixed, moving, mxshift, linmins, linmaxs, SD=eye(ndims(fixed));
                   thresh=0.5*sum(abs2.(fixed[.!(isnan.(fixed))])),
                   initial_tfm=IdentityTransformation(),
                   kwargs...)
    fixed, moving = float(fixed), float(moving)
    linmins = [linmins...]
    linmaxs = [linmaxs...]
    print("Running coarse step\n")
    mw = default_lin_minwidths(moving)
    tfm_coarse, mm_coarse = qd_affine_coarse(fixed, moving, mxshift, linmins, linmaxs, SD;
                                             minwidth=mw, initial_tfm=initial_tfm, thresh=thresh, kwargs...)
    print("Running fine step\n")
    mw = mw./100
    linmins, linmaxs = scalebounds(linmins, linmaxs, 0.5)
    final_tfm, final_mm = qd_affine_fine(fixed, moving, linmins, linmaxs, SD;
                                         minwidth_mat=mw, initial_tfm=tfm_coarse, thresh=thresh, kwargs...)
    return final_tfm, final_mm
end

function qd_affine(fixed, moving, mxshift, SD=eye(ndims(fixed));
                   thresh=0.5*sum(abs2.(fixed[.!(isnan.(fixed))])),
                   initial_tfm=IdentityTransformation(),
                   kwargs...)
    minb, maxb = default_linmap_bounds(fixed)
    return qd_affine(fixed, moving, mxshift, minb, maxb, SD; thresh=thresh, initial_tfm=initial_tfm, kwargs...)
end
