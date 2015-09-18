__precompile__()

module RegisterFit

using Compat, Optim, NLsolve, RegisterPenalty, RegisterCore, AffineTransforms, Grid, Interpolations, Optim1d
import Optim: optimize, nelder_mead

if VERSION < v"0.4.0-dev"
    import Base: @nloops, @nexprs, @nref, @ngenerate
else
    import Base: @nloops, @nexprs, @nref
    using Compat
    import Compat: @ngenerate
end

export
    mismatch2affine,
    optim1d,
    optimize,
    optimize_per_block,
    optimize_rigid,
    pat_affine,
    pat_rotation,
    principalaxes,
    qbuild,
    qfit,
    uisvalid,
    uclamp!

# For bounds constraints
const register_half = 0.5001
const register_half_safe = 0.51

function mismatch2affine(nums, denoms, thresh, arraysize)
    gsize = gridsize(nums)
    d = length(gsize)
    B = getblock(nums, ones(Int, d)...)
    T = eltype(B)
    TFT = typeof(one(T)/2)  # transformtype; it needs to be a floating-point type
    n = prod(gsize)
    u0 = cell(n)
    Q = cell(n)
#     denom = zeros(T, size(B))
#     for c in Counter(gsize)
#         den = getblock(denoms, c...)
#         for j = 1:length(denom)
#             denom[j] += den[j]
#         end
#     end
    i = 0
    nnz = 0
    nreps = 0
    while nnz < d+1 && nreps < 3
        for c in Counter(gsize)
            i += 1
    #         E0, u0[i], Q[i] = qfit(getblock(nums, c...), denom, thresh)
            E0, u0[i], Q[i] = qfit(getblock(nums, c...), getblock(denoms, c...), thresh)
            nnz += any(Q[i].!=0)
        end
        if nnz < d+1
            warn("Halving thresh")
            thresh /= 2
            nreps += 1
            i = 0
            nnz = 0
        end
    end
    if nreps == 3
        error("Decreased threshold by factor of 8, but it still wasn't enough to avoid degeneracy")
    end
    s = ([arraysize...].-1)./max(1, [gsize...].-1)
    x = cell(n)
    i = 0
    center = convert(Array{TFT}, ([arraysize...] .+ 1)/2)
    for c in Counter(gsize)
        x[i+=1] = convert(Array{TFT}, (c.-1).*s .+ 1 .- center)
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
        warn("The data do not suffice to determine a full affine transformation with this grid size---\n  perhaps the only supra-threshold block was the center one? Defaulting to a translation.")
        t = L[d^2+1:end, d^2+1:end]\tb
        return tformtranslate(convert(Vector{T}, t))
    end
    R = reshape(rt[1:d*d], d, d)
    t = rt[d*d+1:end]
    AffineTransform(convert(Matrix{T}, R), convert(Vector{T}, t))
end

# Optimize an affine transformation, given pre-calculated nums/denoms. This is not adequate
# for large rotations/skews etc, but may be fine for polishing
function optimize(tform::AffineTransform, numsdenoms, arraysize)
    gsize = gridsize(numsdenoms)
    N = length(gsize)
    ndims(tform) == N || error("Dimensionality of tform is $(ndims(tform)), which does not match $N for nums/denoms")
    bsize = blocksize(numsdenoms)
    T = blockeltype(numsdenoms)
    # Compute the bounds
    maxshift = Int[bsize[i]>>1 for i = 1:N]
    center = T[(arraysize[i]+1)/2 for i = 1:N]
    s = T[(arraysize[i]-1)/(max(1,gsize[i]-1)) for i = 1:N]  # scale factor
    X = zeros(T, N+1, prod(gsize))
    k = 0
    for c in Counter(gsize)
        X[1:N,k+=1] = (c.-1).*s .+ 1 - center
        X[N+1,k] = 1
    end
    bound = convert(Vector{T}, [maxshift .- register_half; Inf])
    lower = repeat(-bound, outer=[1,size(X,2)])
    upper = repeat( bound, outer=[1,size(X,2)])
    # Extract the parameters from the initial guess
    Si = tform.scalefwd
    displacement = tform.offset
    A = convert(Matrix{T}, [Si-eye(N) displacement; zeros(1,N) 1])
    # Determine the blocks that start in-bounds
    AX = A*X
    keep = trues(gsize)
    for j = 1:length(keep)
        for idim = 1:N
            xi = AX[idim,j]
            if xi < -maxshift[idim]+register_half_safe || xi > maxshift[idim]-register_half_safe
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
    ws = ROWorkspace(T, maxshift, gsize, s)
    ws.lambda_volume = zero(T)
    gtmp = Array(T, N, gsize...)
    objective = (x,g) -> affinepenalty!(g, x, numsdenoms, T[], ws, X', gsize, keep, gtmp)
    @assert typeof(objective(A', T[])) == T
    result = interior(DifferentiableFunction(x->objective(x,T[]), Optim.dummy_g!, objective), A', constraints, method=:cg)
    @assert Optim.converged(result)
    Aopt = result.minimum'
    Siopt = Aopt[1:N,1:N] + eye(N)
    displacementopt = Aopt[1:N,end]
    AffineTransform(convert(Matrix{T}, Siopt), convert(Vector{T}, displacementopt)), result.f_minimum
end

# Penalty must be a function A -> value
function optimize_rigid(penalty::Function, A::AffineTransform, SD = eye(ndims(A)); ftol=1e-4)
    R = SD*A.scalefwd/SD
    rotp = rotationparameters(R)
    dx = A.offset
    p0 = [rotp; dx]
    objective = p -> penalty(p2rigid(p, SD))
    results = nelder_mead(objective, p0, initial_step=[fill(0.05, length(rotp)); ones(length(dx))], ftol=ftol)
    p2rigid(results.minimum, SD), results.f_minimum
end

function optim1d{T,N}(penalty::Function, Anew::AffineTransform{T,N}, Aold = AffineTransform(eye(T,N),zeros(T,N)); p0 = NaN)
    if !isfinite(p0)
        p0 = penalty(Aold)
    end
    # Factor Anew so we can take powers of it
    Anewm = [Anew.scalefwd Anew.offset; zeros(T, 1, N) one(T)]
    D, V = eig(Anewm)
    penalty1d = alpha->factoredaffinepenalty(penalty, alpha, Aold, D, V)
    # Pick trial step so that 1.0 will be the highest value tested among the first three
    al, am, ar = bracket(penalty1d, 0.0, 1/2.618, p0)
    @show al, am, ar
    result = Optim.brent(penalty1d, al, ar)
    @show result.minimum
    AnewOpt = factoredaffine(result.minimum, D, V)
    @show AnewOpt
    @show Aold
    AnewOpt, result.f_minimum
end

function optimize_per_block(nums, denoms, thresh)
    gsize = gridsize(nums)
    nd = length(gsize)
    u = zeros(nd, gsize...)
    utmp = zeros(nd)
    for (iblock,c) in enumerate(Counter(gsize))
        N = getblock(nums, c...)
        D = getblock(denoms, c...)
        ind2disp!(utmp, size(N), indminmismatch(N, D, thresh))
        for idim = 1:nd
            u[idim,iblock] = utmp[idim]
        end
    end
    u
end



# Build the estimate of r back from the fitting parameters. Useful for debugging.
function qbuild(E0::Real, umin::Vector, Q::Matrix, maxshift)
    d = length(maxshift)
    (size(Q,1) == d && size(Q,2) == d && length(umin) == d) || error("Size mismatch")
    szoutv = 2*[maxshift...]+1
    out = zeros(eltype(Q), tuple(szoutv...))
    j = 1
    du = similar(umin)
    Qdu = similar(umin)
    for c in Counter(szoutv)
        for idim = 1:d
            du[idim] = c[idim] - maxshift[idim] - 1 - umin[idim]
        end
        uQu = dot(du, A_mul_B!(Qdu, Q, du))
        out[j] = E0 + uQu
        j += 1
    end
    out
end

function uisvalid(u, maxshift)
    nd = size(u,1)
    nblocks = div(length(u), nd)
    for j = 1:nblocks, idim = 1:nd
        if abs(u[idim,j]) >= maxshift[idim]-register_half
            return false
        end
    end
    true
end

function uclamp!(u, maxshift)
    nd = size(u,1)
    nblocks = div(length(u), nd)
    for j = 1:nblocks, idim = 1:nd
        u[idim,j] = max(-maxshift[idim]+register_half_safe, min(u[idim,j], maxshift[idim]-register_half_safe))
    end
    u
end

@ngenerate N (Vector, Matrix) function principalaxes{T,N}(img::AbstractArray{T,N})
    Ts = typeof(zero(T)/1)
    psums = Vector{Ts}[zeros(Ts, size(img, d)) for d = 1:N]  # partial sums along all but one axis
    # Use a two-pass algorithm
    # First the partial sums, which we use to compute the centroid
    @nloops N I img begin
        @inbounds v = @nref N img I
        if !isnan(v)
            @inbounds (@nexprs N d->(psums[d][I_d] += v))
        end
    end
    s = sum(psums[1])
    pmeans = Ts[sum(psums[d] .* (1:length(psums[d]))) for d = 1:N]
    @nexprs N d->(m_d = pmeans[d]/s)
    # Now the variance
    @nexprs N j->(@nexprs N i->(i >= j ? V_i_j = zero(Ts) : nothing))  # will hold intensity-weighted variance
    @nloops N I img begin
        @inbounds v = @nref N img I
        if !isnan(v)
            @nexprs N j->(@nexprs N i->(i > j ? V_i_j += v*(I_i-m_i)*(I_j-m_j) : nothing))
        end
    end
    @nexprs N d->(V_d_d = sum(psums[d] .* ((1:length(psums[d]))-m_d).^2))
    @nexprs N j->(@nexprs N i->(i >= j ? V_i_j /= s : nothing))
    # Package for output
    mean = Array(Ts, N)
    var = Array(Ts, N, N)
    @nexprs N d->(mean[d] = m_d)
    @nexprs N j->(@nexprs N i->(var[i,j] = i >= j ? V_i_j : V_j_i))
    mean, var
end

# Principal axes transform. SD is the spacedimensions matrix, and
# used so that rotations in _physical_ space result in orthogonal matrices.
# This returns a list of AffineTransform candidates representing rotations+translations
# that align the principal axes. The rotation is ambiguous up to 180 degrees
# (i.e., flips of even numbers of coordinates), hence the list of candidates.
function pat_rotation(fixedmoments::@compat(Tuple{Vector,Matrix}), moving::AbstractArray, SD = eye(ndims(moving)))
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

# This is ambiguous up to a rotation.
function pat_affine(fixedmoments::@compat(Tuple{Vector,Matrix}), moving::AbstractArray, SD = eye(ndims(moving)))
    nd = ndims(moving)
    fmean, fvar = fixedmoments
    nd = length(fmean)
    fvar = SD*fvar*SD'
    fD, fV = eig(fvar)
    mmean, mvar = principalaxes(moving)
    mvar = SD*mvar*SD'
    mD, mV = eig(mvar)
    S = (mV*Diagonal(sqrt(mD)))/(fV*Diagonal(sqrt(fD)))
    if det(S) < 0
        S[:,1] = -S[:,1]
    end
    c = ([size(moving)...].+1)/2
    A = pat_at(S, SD, c, fmean, mmean)
#     # Debugging check
#     @show fvar
#     Sp = A.scalefwd
#     @show Sp*mvar*Sp'
    A
end

pat_affine(fixed::AbstractArray, moving::AbstractArray, SD = eye(ndims(moving))) =
    pat_affine(principalaxes(fixed), moving, SD)

function pat_at(S, SD, c, fmean, mmean)
    Sp = SD\(S*SD)
    bp = (mmean-c) - Sp*(fmean-c)
    AffineTransform(Sp, bp)
end

#### Low-level utilities

function _calculate_u(At, Xt, gridsize)
    Ut = Xt*At
    u = Ut[:,1:size(Ut,2)-1]'                      # discard the dummy dimension
    reshape(u, tuple(size(u,1), gridsize...))  # put u in the shape of the grid
end

function affinepenalty!(g, At, numsdenoms, uold, ws, Xt, gridsize, keep, gtmp)
    u = _calculate_u(At, Xt, gridsize)
    @assert eltype(u) == eltype(At)
    val = penalty!(gtmp, u, numsdenoms, uold, ws, keep)
    @assert isa(val, eltype(At))
    nd = size(gtmp,1)
    nblocks = size(Xt,1)
    if !isempty(g)
        At_mul_Bt!(g, Xt, [reshape(gtmp,nd,nblocks); zeros(1,nblocks)])
    end
    val
end

function factoredaffine(alpha, D, V)
    Anewm = real(V*Diagonal(D.^alpha)/V)
    AffineTransform(Anewm[1:end-1,1:end-1], Anewm[1:end-1,end])
end

function factoredaffinepenalty(penalty, alpha, Aold, D, V)
    Anew = factoredaffine(alpha, D, V)
    A = Anew*Aold
    ret = penalty(A)
    @show alpha, ret
    ret
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

# Using the standard algorithms, the cost of matrix-multiplies is too high. We need to write these out by hand.
for N = 1:3
#     Np1 = N+1
    @eval begin
        function qfit_core!{T}(dE::Array{T,2}, V4::Array{T,2}, C::Array{T,4}, num::AbstractArray{T,$N}, denom::AbstractArray{T,$N}, thresh, umin::NTuple{$N,Int}, E0)#, tileindx::Int = 1)
            @nexprs $N i->(@nexprs $N j->j<i?nothing:(dE_i_j = zero(T); V4_i_j = 0))
            @nexprs $N d->(umin_d = umin[d])
            @nexprs $N iter1->(@nexprs $N iter2->iter2<iter1?nothing:(@nexprs $N iter3->iter3<iter2?nothing:(@nexprs $N iter4->iter4<iter3?nothing:(C_iter1_iter2_iter3_iter4 = zero(T)))))
            @nloops $N i num begin
                den = @nref $N denom i
                if den > thresh
                    @nexprs $N d->(v_d = i_d - umin_d)
                    v2 = 0
                    @nexprs $N d->(v2 += v_d*v_d)
                    r = (@nref $N num i)/den
                    dE0 = r-E0
                    @nexprs $N j->(@nexprs $N k->k<j?nothing:(dE_j_k += dE0*v_j*v_k; V4_j_k += v2*v_j*v_k))
                    @nexprs $N iter1->(@nexprs $N iter2->iter2<iter1?nothing:(@nexprs $N iter3->iter3<iter2?nothing:(@nexprs $N iter4->iter4<iter3?nothing:(C_iter1_iter2_iter3_iter4 += v_iter1*v_iter2*v_iter3*v_iter4))))
                end
            end
            @nexprs $N i->(@nexprs $N j->j<i?(dE[i,j] = dE_j_i; V4[i,j] = V4_j_i):(dE[i,j] = dE_i_j; V4[i,j] = V4_i_j))
            @nexprs $N iter1->(@nexprs $N iter2->iter2<iter1?nothing:(@nexprs $N iter3->iter3<iter2?nothing:(@nexprs $N iter4->iter4<iter3?nothing:(C[iter1,iter2,iter3,iter4] = C_iter1_iter2_iter3_iter4))))
            sortindex = Array(Int, 4)
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
end

function qfit{T<:Real}(num::AbstractArray{T}, denom::AbstractArray{T}, thresh::Real)
    threshT = convert(T, thresh)
    d = ndims(num)
    maxshift = ceil(Int, ([size(num)...].-1)/2)
    E0 = typemax(T)
    imin = 0
    for i = 1:length(num)
        if denom[i] > thresh
            r = num[i]/denom[i]
            if r < E0
                imin = i
                E0 = r
            end
        end
    end
    if imin == 0
        return zero(T), zeros(T, d), zeros(T, d, d)  # no valid values
    end
    umin = ind2sub(size(num), imin)  # not yet relative to center
    uout = T[umin...]
    for i = 1:d
        uout[i] -= size(num,i)>>1 + 1
    end
    dE = Array(T, d, d)
    V4 = similar(dE)
    C = zeros(T, d, d, d, d)
    qfit_core!(dE, V4, C, num, denom, thresh, umin, E0)
    if all(dE .== 0) || any(diag(V4) .== 0)
        return E0, uout, zeros(eltype(dE), d, d)
    end
    # Initial guess for Q
    M = real(sqrtm(V4))
    # Compute M\dE/M carefully:
    U, s, V = svd(M)
    s1 = s[1]
    sinv = T[v < sqrt(eps(T))*s1 ? zero(T) : 1/v for v in s]
    Minv = V*scale(sinv, U')
    Q = Minv*dE*Minv
    local QL
    try
        QL = full(@compat(chol(Q, Val{:L})))
    catch err
        if isa(err, LinAlg.PosDefException)
            warn("Fixing positive-definite exception:")
            @show V4 dE M Q
            QL = full(@compat(chol(Q+0.001*mean(diag(Q))*I, Val{:L})))
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
    #This line eventually results in a call to an NLSolve constructor that only accepts Float64 types, so registration only works with Float64's
    results = nlsolve((x,fx)->QLerr!(x, fx, C, dE, similar(QL)), (x,gx)->QLjac!(x, gx, C, similar(QL)), x)
    unpackL!(QL, results.zero)
    E0, uout, QL'*QL
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

RegisterCore.getblock(A::Interpolations.BSplineInterpolation, I...) = getblock(A.coefs, I...)

end
