__precompile__()

module RegisterFit

using AffineTransforms, Interpolations, StaticArrays, NLsolve
using RegisterPenalty, RegisterCore, CenterIndexedArrays

using Base: @nloops, @nexprs, @nref, @nif

export
    mismatch2affine,
    mms2fit!,
    optimize_per_aperture,
    pat_rotation,
    principalaxes,
    qbuild,
    qfit,
    uisvalid,
    uclamp!

"""
# RegisterFit

This module contains a number of functions that calculate affine
transformations that minimize mismatch.  The functions are organized
into categories:

### Global optimization

- `mismatch2affine`: a transformation computed from mismatch data by least squares
- `pat_rotation`: find the optimal rigid transform via a Principal Axes Transformation
- `optimize_per_aperture`: naive registration using independent apertures

### Utilities

- `qfit`: fit a single aperture's mismatch data to a quadratic form
- `mms2fit!`: prepare an array-of-mismatcharrays for optimization
- `qbuild`: reconstruct mismatch data from a quadratic form
- `uclamp!` and `uisvalid!`: check/enforce bounds on optimization

"""

# For bounds constraints
const register_half = 0.5001
const register_half_safe = 0.51

"""
`tform = mismatch2affine(mms, thresh, knots)` returns an affine
transformation that is a "best initial guess" for the transformation
that would minimize the mismatch.  The mismatch is encoded in `mms`
(of the format returned by RegisterMismatch), and `thresh` is the
denominator-threshold that determines regions that have sufficient
pixel/intensity overlap to be considered valid.  `knots` represents
the aperture centers (see RegisterDeformation).

The algorithm is based on fitting each aperture to a quadratic, and
then performing a least-squares minimization of the
sum-over-apertures.  The strength of this procedure is that it finds a
global solution; however, it is based on a coarse approximation, the
quadratic fit.  If you want to polish `tform` without relying on the
quadratic fit, see `optimize`.
"""
function mismatch2affine(mms, thresh, knots)
    gridsize = size(mms)
    N = length(gridsize)
    mm = first(mms)
    T = eltype(eltype(mm))
    TFT = typeof(one(T)/2)  # transformtype; it needs to be a floating-point type
    n = prod(gridsize)
    # Fit the parameters of each quadratic
    u0 = Vector{Any}(n)
    Q =  Vector{Any}(n)
    i = 0
    nnz = 0
    nreps = 0
    while nnz < N+1 && nreps < 3
        for mm in mms
            i += 1
            E0, u0[i], Q[i] = qfit(mm, thresh)
            nnz += any(Q[i].!=0)
        end
        if nnz < N+1
            warn("Insufficent valid points in mismatch2affine. Halving thresh and trying again.")
            thresh /= 2
            nreps += 1
            i = 0
            nnz = 0
        end
    end
    if nreps == 3
        error("Decreased threshold by factor of 8, but it still wasn't enough to avoid degeneracy. It's likely there is a problem with thresh or the mismatch data.")
    end
    # Solve the global sum-over-quadratics problem
    x = Vector{Vector{TFT}}(n)   # knot
    center = convert(Vector{TFT}, ([arraysize(knots)...] .+ 1)/2)
    for (i,c) in eachknot(knots)
        x[i] = convert(Vector{TFT}, c) - center
    end
    QB = zeros(T, d, d)
    tb = zeros(T, d)
    L = zeros(T, d*(d+1), d*(d+1))
    for i = 1:n
        tQ = Q[i]
        tx = x[i]
        xu = tx+u0[i]
        tmp = tQ*xu
        QB += tmp*tx'
        tb += tmp
        for m=1:d, l=1:d, k=1:d, j=1:d
            L[j+(k-1)*d, l+(m-1)*d] += tQ[j,l]*tx[m]*tx[k]
        end
        for l=1:d, k=1:d, j=1:d
            L[j+(k-1)*d, d*d+l] += tQ[j,l]*tx[k]
        end
        for m=1:d, l=1:d, j=1:d
            L[d*d+j, l+(m-1)*d] += tQ[j,l]*tx[m]
        end
        for l=1:d, j=1:d
            L[d*d+j, d*d+l] += tQ[j,l]
        end
    end
    if all(L .== 0)
        error("All elements of L are zero. It's likely thresh is too high.")
    end
    local rt
    try
        rt = L\[QB[:];tb]
    catch
        warn("The data do not suffice to determine a full affine transformation with this grid size---\n  perhaps the only supra-threshold block was the center one?\n  Defaulting to a translation (advice: reconsider your threshold).")
        t = L[d^2+1:end, d^2+1:end]\tb
        return tformtranslate(convert(Vector{T}, t))
    end
    R = reshape(rt[1:d*d], d, d)
    t = rt[d*d+1:end]
    AffineTransform(convert(Matrix{T}, R), convert(Vector{T}, t))
end


"""
`u = optimize_per_aperture(mms, thresh)` computes the "naive"
displacement in each aperture to minimize the mismatch. Each aperture
is examined independently of all others. `thresh` establishes a
threshold for the mismatch data.

See also `indmin_mismatch`.
"""
function optimize_per_aperture(mms, thresh)
    gridsize = size(mms)
    nd = length(gridsize)
    u = zeros(nd, gridsize...)
    utmp = zeros(nd)
    for (iblock,mm) in enumerate(mms)
        I = indmin_mismatch(mm, thresh)
        for idim = 1:nd
            u[idim,iblock] = I[idim]
        end
    end
    u
end


"""
`r = qbuild(E0, u0, Q, maxshift)` builds an estimate of the mismatch
ratio given the quadratic form parameters `E0, u0, Q` obtained from
`qfit`.  Often useful for debugging or visualization.
"""
function qbuild(E0::Real, umin::Vector, Q::Matrix, maxshift)
    d = length(maxshift)
    (size(Q,1) == d && size(Q,2) == d && length(umin) == d) || error("Size mismatch")
    szout = ((2*[maxshift...]+1)...)
    out = zeros(eltype(Q), szout)
    j = 1
    du = similar(umin)
    Qdu = similar(umin, typeof(one(eltype(Q))*one(eltype(du))))
    for c in CartesianRange(szout)
        for idim = 1:d
            du[idim] = c[idim] - maxshift[idim] - 1 - umin[idim]
        end
        uQu = dot(du, A_mul_B!(Qdu, Q, du))
        out[j] = E0 + uQu
        j += 1
    end
    CenterIndexedArray(out)
end

"""
`tf = uisvalid(u, maxshift)` returns `true` if all entries of `u` are
within the allowed domain.
"""
function uisvalid{T<:Number}(u::AbstractArray{T}, maxshift)
    nd = size(u,1)
    sztail = Base.tail(size(u))
    for j in CartesianRange(sztail), idim = 1:nd
        if abs(u[idim,j]) >= maxshift[idim]-register_half
            return false
        end
    end
    true
end

"""
`u = uclamp!(u, maxshift)` clamps the values of `u` to the allowed domain.
"""
function uclamp!{T<:Number}(u::AbstractArray{T}, maxshift)
    nd = size(u,1)
    sztail = Base.tail(size(u))
    for j in CartesianRange(sztail), idim = 1:nd
        u[idim,j] = max(-maxshift[idim]+register_half_safe, min(u[idim,j], maxshift[idim]-register_half_safe))
    end
    u
end

function uclamp!{T<:StaticVector}(u::AbstractArray{T}, maxshift)
    uclamp!(reinterpret(eltype(T), u, (length(T), size(u)...)), maxshift)
    u
end

"""
`center, cov = principalaxes(img)` computes the principal axes of an
image `img`.  `center` is the centroid of intensity, and `cov` the
covariance matrix of the intensity.
"""
function principalaxes{T,N}(img::AbstractArray{T,N})
    Ts = typeof(zero(T)/1)
    psums = pa_init(Ts, size(img))   # partial sums along all but one axis
    # Use a two-pass algorithm
    # First the partial sums, which we use to compute the centroid
    for I in CartesianRange(indices(img))
        @inbounds v = img[I]
        if !isnan(v)
            @inbounds for d = 1:N
                psums[d][I[d]] += v
            end
        end
    end
    s, m = pa_centroid(psums)
    # Now the variance
    cov = zeros(Ts, N, N)
    for I in CartesianRange(indices(img))
        @inbounds v = img[I]
        if !isnan(v)
            for j = 1:N
                Δj = I[j] - m[j]
                for i = j+1:N
                    cov[i, j] += v * (I[i]-m[i]) * Δj
                end
            end
        end
    end
    for d = 1:N
        cov[d, d] = sum(psums[d] .* ((1:length(psums[d]))-m[d]).^2)
    end
    for j = 1:N, i = j:N
        cov[i, j] /= s
    end
    for j = 1:N, i = 1:j-1
        cov[i, j] = cov[j, i]
    end
    m, cov
end

@noinline pa_init{S}(::Type{S}, sz) = [zeros(S, s) for s in sz]
@noinline function pa_centroid{S}(psums::Vector{Vector{S}})
    s = sum(psums[1])
    s, S[sum(psums[d] .* (1:length(psums[d]))) for d = 1:length(psums)] / s
end

"""
`tfms = pat_rotation(fixed, moving, [SD=eye])` computes the Principal
Axes Transform aligning the low-order moments of two images. The
reference image is `fixed`, and `moving` is the raw moving image that
you wish to align to `fixed`.  `SD` is a "spacedimensions" matrix, in
some cases needed to ensure that rotations in *physical* space
correspond to orthogonal matrices in array-index units.  For example,
if your axes are not uniformly sampled, `SD = diagm(voxelspacing)`.

If you're aligning many images to `fixed`, you may alternatively call
this as `tfms = pat_rotation(fixedpa, moving, [SD=eye])`.  `fixedpa`
is a `(center,cov)` tuple obtained from `principalaxes(fixed)`.

`tfms` is a list of potential AffineTransform candidates.  PA data,
being based on ellipsoids, are ambiguous up to rotations by 180
degrees (i.e., sign-flips of even numbers of coordinates).
Consequently, you may need to check all of the candidates for the one
that produces the best alignment.
"""
function pat_rotation(fixedmoments::Tuple{Vector,Matrix}, moving::AbstractArray, SD = eye(ndims(moving)))
    nd = ndims(moving)
    nd > 3 && error("Dimensions higher than 3 not supported") # list-generation doesn't yet generalize
    fmean, fvar = fixedmoments
    nd = length(fmean)
    fvar = SD*fvar*SD'
    fD, fV = eig(fvar)
    mmean, mvar = principalaxes(moving)
    mvar = SD*mvar*SD'
    mD, mV = eig(mvar)
    R = mV/fV
    if det(R) < 0     # ensure it's a rotation
        R[:,1] = -R[:,1]
    end
    c = ([size(moving)...].+1)/2
    tfms = [pat_at(R, SD, c, fmean, mmean)]
    for i = 1:nd
        for j = i+1:nd
            Rc = copy(R)
            Rc[:,i] = -Rc[:,i]
            Rc[:,j] = -Rc[:,j]
            push!(tfms, pat_at(Rc, SD, c, fmean, mmean))
        end
    end
#     # Debugging check
#     @show fvar
#     for i = 1:length(tfms)
#         Sp = tfms[i].scalefwd
#         S = SD*Sp/SD
#         @show S
#         @show R
#         @show S'*mvar*S
#     end
    tfms
end

pat_rotation(fixed::AbstractArray, moving::AbstractArray, SD = eye(ndims(fixed))) =
    pat_rotation(principalaxes(fixed), moving, SD)

function pat_at(S, SD, c, fmean, mmean)
    Sp = SD\(S*SD)
    bp = (mmean-c) - Sp*(fmean-c)
    AffineTransform(Sp, bp)
end

#### Low-level utilities

@generated function qfit_core!{T,N}(dE::Array{T,2}, V4::Array{T,2}, C::Array{T,4}, mm::Array{NumDenom{T},N}, thresh, umin::NTuple{N,Int}, E0, maxsep::NTuple{N,Int})
    # The cost of generic matrix-multiplies is too high, so we write
    # these out by hand.
    quote
        @nexprs $N i->(@nexprs $N j->j<i?nothing:(dE_i_j = zero(T); V4_i_j = 0))
        @nexprs $N d->(umin_d = umin[d])
        @nexprs $N iter1->(@nexprs $N iter2->iter2<iter1?nothing:(@nexprs $N iter3->iter3<iter2?nothing:(@nexprs $N iter4->iter4<iter3?nothing:(C_iter1_iter2_iter3_iter4 = zero(T)))))
        @nloops $N i mm begin
            @nif $(N+1) d->(abs(i_d-umin[d]) > maxsep[d]) d->(continue) d->nothing
            nd = @nref $N mm i
            num, den = nd.num, nd.denom
            if den > thresh
                @nexprs $N d->(v_d = i_d - umin_d)
                v2 = 0
                @nexprs $N d->(v2 += v_d*v_d)
                r = num/den
                dE0 = r-E0
                @nexprs $N j->(@nexprs $N k->k<j?nothing:(dE_j_k += dE0*v_j*v_k; V4_j_k += v2*v_j*v_k))
                @nexprs $N iter1->(@nexprs $N iter2->iter2<iter1?nothing:(@nexprs $N iter3->iter3<iter2?nothing:(@nexprs $N iter4->iter4<iter3?nothing:(C_iter1_iter2_iter3_iter4 += v_iter1*v_iter2*v_iter3*v_iter4))))
            end
        end
        @nexprs $N i->(@nexprs $N j->j<i?(dE[i,j] = dE_j_i; V4[i,j] = V4_j_i):(dE[i,j] = dE_i_j; V4[i,j] = V4_i_j))
        @nexprs $N iter1->(@nexprs $N iter2->iter2<iter1?nothing:(@nexprs $N iter3->iter3<iter2?nothing:(@nexprs $N iter4->iter4<iter3?nothing:(C[iter1,iter2,iter3,iter4] = C_iter1_iter2_iter3_iter4))))
        sortindex = Vector{Int}(4)
        for iter1 = 1:$N, iter2 = 1:$N, iter3 = 1:$N, iter4 = 1:$N
            sortindex[1] = iter1
            sortindex[2] = iter2
            sortindex[3] = iter3
            sortindex[4] = iter4
            sort!(sortindex)
            C[iter1,iter2,iter3,iter4] = C[sortindex[1],sortindex[2],sortindex[3],sortindex[4]]
        end
        dE, V4, C
    end
end

"""
`E0, u0, Q = qfit(mm, thresh; [maxsep=size(mm), opt=true])` performs a
quadratic fit of the mismatch data in `mm`.  On output, `u0` and `E0`
hold the position and value, respectively, of the shift with smallest
mismatch, and `Q` is a matrix representing the best fit to a model

```
    mm ≈ E0 + (u-u0)'*Q*(u-u0)
```
Only those shift-locations with `mm[i].denom > thresh` are used in
performing the fit.

`maxsep` allows you to restrict the fit to a region where each
coordinate satisfies `|u[d]-u0[d]| <= maxsep[d]`. If `opt` is false,
`Q` is a heuristic estimate of the best-fit `Q`. This can boost
performance at the cost of accuracy.
"""
function qfit(mm::MismatchArray, thresh::Real; maxsep=size(mm), opt::Bool=true)
    qfit(mm, thresh, maxsep, opt)
end

function qfit(mm::MismatchArray, thresh::Real, maxsep, opt::Bool)
    T = eltype(eltype(mm))
    threshT = convert(T, thresh)
    d = ndims(mm)
    mxs = maxshift(mm)
    E0 = typemax(T)
    imin = 0
    for (i, nd) in enumerate(mm)
        if nd.denom > thresh
            r = nd.num/nd.denom
            if r < E0
                imin = i
                E0 = r
            end
        end
    end
    if imin == 0
        return zero(T), zeros(T, d), zeros(T, d, d)  # no valid values
    end
    umin = ind2sub(size(mm), imin)  # not yet relative to center
    uout = T[umin...]
    for i = 1:d
        uout[i] -= (size(mm,i)+1)>>1
    end
    dE = Matrix{T}(d, d)
    V4 = similar(dE)
    C = zeros(T, d, d, d, d)
    qfit_core!(dE, V4, C, mm.data, thresh, umin, E0, maxsep)
    if all(dE .== 0) || any(diag(V4) .== 0)
        return E0, uout, zeros(eltype(dE), d, d)
    end
    # Initial guess for Q
    M = real(sqrtm(V4))::Matrix{T}
    # Compute M\dE/M carefully:
    U, s, V = svd(M)
    sinv = sv_inv(T, s)
    Minv = V * Diagonal(sinv) * U'
    Q = Minv*dE*Minv
    opt || return E0, uout, Q
    local QL
    try
        QL = convert(Matrix{T}, ctranspose(chol(Hermitian(Q))))
    catch err
        if isa(err, LinAlg.PosDefException)
            warn("Fixing positive-definite exception:")
            @show V4 dE M Q
            QL = convert(Matrix{T}, chol(Q+T(0.001)*mean(diag(Q))*I, Val{:L}))::Matrix{T}
        else
            rethrow(err)
        end
    end

    # Optimize QL
    x = zeros(T, (d*(d+1))>>1)
    indx = 0
    for i = 1:d,j=1:d
        if i>=j
            x[indx+=1] = QL[i,j]
        end
    end
    local results
    function solveql(C, dE, QL, x)
        nlsolve((x,fx)->QLerr!(x, fx, C, dE, similar(QL)), (x,gx)->QLjac!(x, gx, C, similar(QL)), x)
    end
    try
        results = solveql(C, dE, QL, x)
    catch err
        @show C dE QL x
        rethrow(err)
    end
    unpackL!(QL, results.zero)
    E0, uout, QL'*QL
end

@noinline function sv_inv{T}(::Type{T}, s)
    s1 = s[1]
    sinv = T[v < sqrt(eps(T))*s1 ? zero(T) : 1/v for v in s]
end

"""
`cs, Qs, mmis = mms2fit!(mms, thresh)` computes the shift and
quadratic-fit values for the array-of-mismatcharrays `mms`, using a
threshold of `thresh`. It also prepares `mms` for interpolation,
modifying the data in-place (after computing `cs` and `Qs`).

The return values are suited for input the `fixed_λ` and `auto_λ`.
"""
function mms2fit!{A<:MismatchArray,N}(mms::AbstractArray{A,N}, thresh)
    T = eltype(eltype(A))
    gridsize = size(mms)
    cs = Array{SVector{N,T}}(gridsize)
    Qs = Array{similar_type(SArray, T, Size(N, N))}(gridsize)
    for i = 1:length(mms)
        _, cs[i], Qs[i] = qfit(mms[i], thresh; opt=false)
    end
    mmis = interpolate_mm!(mms)
    cs, Qs, mmis
end

function unpackL!(QL, x)
    d = size(QL, 1)
    indx = 0
    for i = 1:d,j=1:d
        if i>=j
            QL[i,j] = x[indx+=1]
        end
    end
    QL
end

function QLerr!(x, fx, C, dE, L)
    d = size(L,1)
    fill!(L, 0)
    unpackL!(L, x)
    indx = 0
    T = typeof(C[1,1,1,1]*L[1,1] + C[1,1,1,1]*L[1,1])
    for i = 1:d, j=1:d
        if i >= j
            tmp = zero(T)
            for l=1:d,m=1:d,n=1:d
                tmp += L[l,m]*L[l,n]*C[i,j,m,n]
            end
            fx[indx+=1] = tmp - dE[i,j]
        end
    end
end

function QLjac!(x, gx, C, L)
    d = size(L,1)
    fill!(L, 0)
    unpackL!(L, x)
    T = typeof(C[1,1,1,1]*L[1,1] + C[1,1,1,1]*L[1,1])
    indx1 = 0
    for i=1:d, j=1:d
        if i >= j
            indx1 += 1
            indx2 = 0
            for a=1:d, b=1:d
                if a >= b
                    tmp = zero(T)
                    for k = 1:d
                        tmp += (C[i,j,b,k]+C[i,j,k,b])*L[a,k]
                    end
                    gx[indx1, indx2+=1] = tmp
                end
            end
        end
    end
end

end
