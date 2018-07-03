function tfmx(x, img, SD=eye(ndims(img)))
    if ndims(img) == 2
        dx, dy, θ = x
        rotm = RotMatrix(θ)
        rotm = SD\rotm*SD
        return Translation(dx, dy) ∘ recenter(SMatrix{2,2}(rotm), center(img))
    elseif ndims(img) == 3
        dx, dy, dz, θx, θy, θz =  x
        rotm = RotMatrix(RotXYZ(θx,θy,θz))
        rotm = SD\rotm*SD
        return Translation(dx, dy, dz) ∘ recenter(SMatrix{3,3}(rotm), center(img))
    else
        error()
    end
end

function rot(theta, img, SD=eye(ndims(img)))
    if ndims(img) == 2
        rotm = RotMatrix(theta...)
        rotm = SD\rotm*SD
        return recenter(SMatrix{2,2}(rotm), center(img))
    elseif ndims(img) == 3
        θx, θy, θz = theta
        rotm = RotMatrix(RotXYZ(θx,θy,θz))
        rotm = SD\rotm*SD
        return recenter(SMatrix{3,3}(rotm), center(img))
    else
        error()
    end
end

#Finds the best shift aligning moving to fixed, possibly after an initial transformation `intial_tfm`
function best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=-1)
    if initial_tfm != -1
        newmov = warp(moving, initial_tfm)
        inds1, inds2 = indices(fixed), indices(newmov)
        inds = ([intersect(x, y) for (x,y) in zip(inds1, inds2)]...)
        fixed = view(fixed, inds...)
        moving = newmov[inds...]
    end
    mms = mismatch(fixed, moving, mxshift; normalization=normalization)
    best_i = indmin_mismatch(mms, thresh)
    return best_i.I, ratio(mms[best_i], 0.0, Inf)
end

#rotation + shift, slow because it warps for every rotation and shift
function slow_mm(tfm, fixed, moving, thresh, SD; initial_tfm = -1)
    tfm = tfmx(tfm, moving, SD)
    if initial_tfm != -1
        tfm = tfm ∘ initial_tfm #compose with rotation already computed
    end
    newmov = warp(moving, tfm)
    inds1, inds2 = indices(fixed), indices(newmov)
    inds = ([intersect(x, y) for (x,y) in zip(inds1, inds2)]...)
    mm = mismatch0(view(fixed, inds...), view(newmov, inds...); normalization=:pixels)
    rslt = ratio(mm, thresh, Inf)
    return rslt
end

#rotation + shift, fast because it uses fourier method for shift
function fast_mm(theta, mxshift, fixed, moving, thresh, SD)
    tfm = rot(theta, moving, SD)
    newmov = warp(moving, tfm)
    inds1, inds2 = indices(fixed), indices(newmov)
    inds = ([intersect(x, y) for (x,y) in zip(inds1, inds2)]...)
    bshft, mm = best_shift(view(fixed, inds...), newmov[inds...], mxshift, thresh; normalization=:intensity)
    return mm
end

function qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot, SD; thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])), kwargs...)
    f(x) = fast_mm(x, mxshift, fixed, moving, thresh, SD)
    upper = [mxrot...]
    lower = -upper
    splits = ([[-x; 0.0; x] for x in mxrot]...)
    root_coarse, x0coarse = analyze(f, splits, lower, upper; maxevals=10^4, minwidth=minwidth_rot, print_interval=100, kwargs...)
    box_coarse = minimum(root_coarse)
    tfmcoarse0 = rot(position(box_coarse, x0coarse), moving)
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm = tfmcoarse0)
    tfmcoarse = tfmcoarse0 ∘ Translation(best_shft)
    return tfmcoarse, mm
end

function qd_rigid_fine(fixed, moving, mxrot, minwidth, SD; initial_tfm = -1, thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])), kwargs...)
    f2(x) = slow_mm(x, fixed, moving, thresh, SD; initial_tfm = initial_tfm)
    upper_shft = fill(1, ndims(fixed))
    upper_rot = mxrot
    upper = vcat(upper_shft, upper_rot)
    lower = -upper
    splits = ([[-x; 0.0; x] for x in upper]...)
    minwidth_shfts = fill(0.01, ndims(fixed))
    minwidth_rots = ndims(fixed) == 2 ? [0.0001;] : fill(0.0001, 3) #assume 2 or 3 dimensional input
    minwidth = vcat(minwidth_shfts, minwidth_rots)
    root, x0 = analyze(f2, splits, lower, upper; maxevals=10^4, minwidth=minwidth, print_interval=100, kwargs...)
    box = minimum(root)
    params = position(box, x0)
    tfmfine = tfmx(position(box, x0), moving)
    return tfmfine, value(box)
end

"""
`tform, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot; thresh=thresh, [SD = eye], kwargs...)`
optimizes a rigid transformation (rotation + shift) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm.  The algorithm is run twice: the first step uses a fourier method to speed up
the search for the best whole-pixel shift.  The second step refines the search for sub-pixel accuracy. `kwargs...` can include any
keyword argument that can be passed to `QuadDIRECT.analyze`. It's recommended that you pass your own stopping criteria when possible
(i.e. `rtol`, `atol`, and/or `fvalue`).

Use `SD` if your axes are not uniformly sampled, for example `SD = diagm(voxelspacing)` where `voxelspacing`
is a vector encoding the spacing along all axes of the image. `thresh` enforces a certain amount of sum-of-squared-intensity overlap between
the two images; with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.
"""
function qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD=eye(ndims(fixed)); thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])), kwargs...)
    mxrot = [mxrot...]
    print("Running coarse step\n")
    tfm_coarse, mm_coarse = qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot, SD; thresh = thresh, kwargs...)
    print("Running fine step\n")
    tfm_fine, mm_fine = qd_rigid_fine(fixed, moving, mxrot./10, minwidth_rot, SD; initial_tfm = tfm_coarse, thresh = thresh, kwargs...)
    final_tfm = tfm_fine ∘ tfm_coarse
    return final_tfm, mm_fine
end
