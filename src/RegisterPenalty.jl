# Penalty function for registration of a single image

__precompile__()

module RegisterPenalty

export ROWorkspace, compose_u, penalty!, interpolate_nd!

using Grid  # FIXME: delete this
using Interpolations, RegisterCore, Base.Cartesian

typealias BSI Interpolations.BSplineInterpolation

function interpolate_nd!{T<:Real}(nums::AbstractArray{T}, denoms::AbstractArray{T}; BC=InPlace)
    nd = ndims(nums)>>1
    interptype = Tuple{vcat([BSpline(Quadratic(BC)) for i = 1:nd], [BSpline(NoInterp) for i = 1:nd])...}
    numsi = interpolate!(sdata(nums), interptype, OnCell)
    denomsi = interpolate!(sdata(denoms), interptype, OnCell)
    numsi, denomsi
end

function interpolate_nd!{T<:AbstractArray}(nums::AbstractArray{T}, denoms::AbstractArray{T}; BC=InPlace)
    f = x->interpolate!(sdata(x), BSpline(Quadratic(BC)), OnCell)
    map(f, nums), map(f, denoms)
end

function interpolate_nd!{T}(numdenom::AbstractArray{NumDenom{T}}; BC=InPlace)
    nd = ndims(numdenom)>>1
    interptype = Tuple{vcat([BSpline(Quadratic(BC)) for i = 1:nd], [BSpline(NoInterp) for i = 1:nd])...}
    interpolate!(T, numdenom, interptype, OnCell)
end

## A type for pre-allocating all workspace variables
abstract AbstractROWorkspace{T,N}

type ROWorkspace{T,N,U,UC} <: AbstractROWorkspace{T,N}
    maxshift::NTuple{N,Int}
    gridsize::NTuple{N,Int}
    bricksize::NTuple{N,T}
#     interpmm::InterpGridCoefs{T,InterpQuadratic}
    interpu::InterpGridCoefs{T,InterpQuadratic}
    u::U            # will be Array{T,N+1}
    gtmp::U
    ucompgrad::UC   # will be Array{T,N+2}
    # Vertex data for the volume penalty
    vpindexes::Array{Int,2}
    vpvertices::Array{Int,2}
    vpcoords::Array{T,2}
    simplexcoords::Array{T,2}
    simplexindex::Array{Int,1}
    # Geometry data for the affine-residual penalty
    Qaffine::Matrix{T}
    # Coefficient for regularization
    lambda_volume::T

    function ROWorkspace(maxshift::Union(Dims, Vector{Int}), gridsize::Union(Dims, Vector{Int}), bricksize)
        length(maxshift) == N || throw(DimensionMismatch("maxshift $maxshift is not of length $N"))
        length(gridsize) == N || throw(DimensionMismatch("gridsize $gridsize is not of length $N"))
        length(bricksize) == N || throw(DimensionMismatch("bricksize $bricksize is not of length $N"))
        br = ntuple(i->convert(T, bricksize[i]), N)
        strides = Array(Int, N)
        blocksize = [2m+1 for m in maxshift]
        nblocks = prod(gridsize)
#         strides[1] = 1
#         for idim = 2:N
#             strides[idim] = strides[idim-1]*blocksize[idim-1]
#         end
#         interpmm = InterpGridCoefs(T, InterpQuadratic, blocksize, strides)
        strides[1] = N
        for idim = 2:N
            strides[idim] = strides[idim-1]*gridsize[idim-1]
        end
        interpu = InterpGridCoefs(T, InterpQuadratic, gridsize, strides)
        u = Array(T, N, gridsize...)
        gtmp = similar(u)
        ucompgrad = Array(T, N, N, gridsize...)
        # Initialize ucompgrad for "no uold" condition
        i = 1
        while i <= N*N*nblocks
            for idimg = 1:N
                for idim = 1:N
                    ucompgrad[i] = (idim == idimg)
                    i += 1
                end
            end
        end
        strides[1] = 1
        for idim = 1:N-1
            strides[idim+1] = strides[idim]*gridsize[idim]
        end
        vpindexes, vpvertices, vpcoords = grid_simplexes(strides)
        simplexcoords = Array(T, size(vpcoords, 1), size(vpvertices, 1))
        simplexindex  = Array(Int, size(vpindexes, 1))
        Caffine = Array(Float64, N+1, nblocks)
        i = 0
        for c = Counter(gridsize)
            Caffine[1:N, i+=1] = c
            Caffine[N+1, i] = 1
        end
        Qaffine, _ = qr(Caffine')
#         new(tuple(maxshift...), tuple(gridsize...), br, interpmm, interpu, u, gtmp, ucompgrad, vpindexes, vpvertices, convert(Array{T,2}, vpcoords),simplexcoords,simplexindex,Qaffine,zero(T))
        new(tuple(maxshift...), tuple(gridsize...), br, interpu, u, gtmp, ucompgrad, vpindexes, vpvertices, convert(Array{T,2}, vpcoords),simplexcoords,simplexindex,Qaffine,zero(T))
    end
end

function ROWorkspace{T}(::Type{T}, maxshift::Union(Dims, Vector{Int}), gridsize::Union(Dims, Vector{Int}), bricksize)
    N = length(maxshift)
    ROWorkspace{T,N,Array{T,N+1},Array{T,N+2}}(maxshift, gridsize, bricksize)
end

lambda_volume(ws::ROWorkspace) = ws.lambda_volume

# nums/denoms can be expressed as an array-of-arrays, or as tiled arrays of dimensionality 2N
Vmin_simplex = 0  # we need to return an alternate value for setting Vslack
function penalty!{T,N}(g, u, numsdenoms, uold, ws::ROWorkspace{T,N}, keep = trues(ws.gridsize), Vslack = nothing)
    global Vmin_simplex
    calc_grad = !isempty(g)
    have_uold = !isempty(uold)
    nblocks = prod(ws.gridsize)
    length(u) == N*nblocks || error("u should have length $(N*nblocks), but length(u) = $(length(u))")
    if calc_grad
        if length(g) != length(u)
            error("length(g) = $(length(g)) but length(u) = $(length(u))")
        end
    end
    if have_uold
        length(u) == length(uold) || error("length(u) = $(length(u)) and length(uold) = $(length(uold))")
    end
    unew = ws.u
    copy!(unew, u)  # to convert the type (and reshape, and avoid risk of messing up the optimizer's copy)
    Vmin_simplex = convert(T, Inf)
    val = zero(T)
    # Volume penalty
    λ = lambda_volume(ws)
    valreg = valdata = val
    if λ != 0
        if have_uold && !is(uold, unew)
            normalize_u!(uold, ws.bricksize)
            normalize_u!(unew, ws.bricksize)
            ucomp = compose_u!(ws.ucompgrad, uold, unew, ws)  # note: allocating
            unnormalize_u!(uold, ws.bricksize)
            unnormalize_u!(unew, ws.bricksize)
        else
            ucomp = compose_u!(ws.ucompgrad, uold, unew, ws)  # note: allocating
            normalize_u!(ucomp, ws.bricksize)
        end
#         val, Vmin = penalty_volume(g, ucomp, ws.ucompgrad, one(T), ws, Vslack)
#         Vmin_simplex = Vmin
        val = penalty_affine_residual!(g, ucomp, ws)
        valreg = val
        if !isfinite(val)
            return val
        end
        if calc_grad
            normalize_u!(g, ws.bricksize) # not unnormalize, since d/dz ~ 1/z
        end
    else
        fill!(g, 0)
    end
    # Data penalty
    valdata = value!(ws.gtmp, unew, numsdenoms, ws, keep)
    val += valdata
#     @show valreg, valdata
    for i = 1:length(g)
        g[i] += ws.gtmp[i]
    end
    return convert(eltype(u), val)
end

################
# Data penalty #
################

function value!(g, u, numsdenoms::Tuple, ws::ROWorkspace, keep=trues(ws.gridsize))
    if !isempty(g)
        gdenom = similar(g)
        N, D = numdenom((g,gdenom), u, numsdenoms, ws, keep)
        invD = 1/D
        NinvD2 = N*invD*invD
        for i = 1:length(g)
            g[i] = invD*g[i] - NinvD2*gdenom[i]
        end
        return N*invD
    else
        N, D = numdenom((g,g), u, numsdenoms, ws, keep)
        return N/D
    end
end

function value!(g, u, numdenomi::BSI, ws::ROWorkspace, keep=trues(ws.gridsize))
    if !isempty(g)
        gnd = Array(eltype(numdenomi), size(u))
        N, D = numdenom(gnd, u, numdenomi, ws, keep)
        invD = 1/D
        NinvD2 = N*invD*invD
        for i = 1:length(g)
            g[i] = invD*gnd[i].num - NinvD2*gnd[i].denom
        end
        return N*invD
    else
        N, D = numdenom(g, u, numdenomi, ws, keep)
        return N/D
    end
end

# immutable Tiled end
# immutable NotTiled end
#
# function numdenom{T,nd}(gnum, gdenom, u::AbstractArray{T}, nums::AbstractArray, denoms::AbstractArray, ws::ROWorkspace{T,nd}, keep)
#     istiled = ndims(nums) == 2nd
#     (istiled || ndims(nums) == nd) || error("nums/denoms must either be tiled or an Array-of-Arrays")
#     _numdenom(istiled ? Tiled() : NotTiled(), gnum, gdenom, u, nums, denoms, ws, keep)
# end

function numdenom{T,nd}(gnumdenom, u::AbstractArray{T}, numsdenoms, ws::ROWorkspace{T,nd}, keep)
    sz = gridsize(numsdenoms)::NTuple{nd,Int}
    R = CartesianRange(sz)
    numdenom(gnumdenom, u, numsdenoms, R, keep)
end

# Supply as arrays-of-interpolations
@generated function numdenom{T,B<:BSI,nd}(gnumdenom::Tuple, u::AbstractArray{T}, numsdenoms::Tuple{AbstractArray{B},AbstractArray{B}}, ws::ROWorkspace{T,nd}, keep)
    args = Any[symbol("x_",d) for d = 1:nd]
    meta = Expr(:meta, :noinline)
    quote
        $meta
        gnum, gdenom = gnumdenom
        nums, denoms = numsdenoms
        calc_grad = !isempty(gnum)
        B = nums[1]
        @nexprs $nd d->(m_d = size(B,d)>>1)
        N = D = zero(T)
        nanT = convert(T, NaN)
        local gtmp
        if calc_grad
            gtmp = Array(T, $nd)
        end
        for iblock = 1:length(nums)
            if !keep[iblock]
                if calc_grad
                    gnum[:,iblock] = 0
                    gdenom[:,iblock] = 0
                end
                continue
            end
            # Check bounds
            @nexprs $nd d->(if abs(u[d,iblock]) >= m_d-0.5 return (nanT,nanT); end)
            # Evaluate the value
            @nexprs $nd d->(x_d = u[d,iblock] + m_d + 1)
            thisnum = nums[iblock]
            thisdenom = denoms[iblock]
            N += thisnum[$(args...)]
            D += thisdenom[$(args...)]
            # Evaluate the gradient
            if calc_grad
                gradient!(gtmp, thisnum, $(args...))
                @nexprs $nd d->(gnum[d, iblock] = gtmp[d])
                gradient!(gtmp, thisdenom, $(args...))
                @nexprs $nd d->(gdenom[d, iblock] = gtmp[d])
            end
        end
        N, D
    end
end

# Supply as tiled interpolations
@generated function numdenom{T}(gnumdenom::Tuple, u::AbstractArray{T}, numsdenoms::Tuple{BSI,BSI}, R::CartesianRange, keep)
    nd = ndims(R)
    args = Any[d <= nd ? symbol("x_",d) : :(I[$(d-nd)]) for d = 1:2nd]
    meta = Expr(:meta, :noinline)
    quote
        $meta
        gnum, gdenom = gnumdenom
        nums, denoms = numsdenoms
        calc_grad = !isempty(gnum)
        @nexprs $nd d->(m_d = size(nums,d)>>1)
        N = D = zero(T)
        nanT = convert(T, NaN)
        local gtmp
        if calc_grad
            gtmp = Array(T, $nd)
        end
        iblock = 0
        for I in R
            iblock += 1
            if !keep[iblock]
                if calc_grad
                    gnum[:,iblock] = 0
                    gdenom[:,iblock] = 0
                end
                continue
            end
            # Check bounds
            @nexprs $nd d->(if abs(u[d,I]) >= m_d-0.5 return (nanT,nanT); end)
            # Evaluate the value
            @nexprs $nd d->(x_d = u[d,I] + m_d + 1)
            N += nums[$(args...)]
            D += denoms[$(args...)]
            # Evaluate the gradient
            if calc_grad
                gradient!(gtmp, nums, $(args...))
                @nexprs $nd d->(gnum[d, I] = gtmp[d])
                gradient!(gtmp, denoms, $(args...))
                @nexprs $nd d->(gdenom[d, I] = gtmp[d])
            end
        end
        N, D
    end
end

# Supply as packed interpolation. This is the most efficient.
@generated function numdenom{T}(gnumdenom, u::AbstractArray{T}, numdenomi::BSI, R::CartesianRange, keep)
    nd = ndims(R)
    args = Any[d <= nd ? symbol("x_",d) : :(I[$(d-nd)]) for d = 1:2nd]
    meta = Expr(:meta, :noinline)
    quote
        $meta
        calc_grad = !isempty(gnumdenom)
        @nexprs $nd d->(m_d = size(numdenomi,d)>>1)
        N = D = zero(T)
        nanT = convert(T, NaN)
        local gtmp
        if calc_grad
            gtmp = Array(eltype(numdenomi), $nd)
        end
        iblock = 0
        for I in R
            iblock += 1
            if !keep[iblock]
                if calc_grad
                    gnumdenom[:,iblock] = zero(eltype(gnumdenom))
                end
                continue
            end
            # Check bounds
            @nexprs $nd d->(if abs(u[d,I]) >= m_d-0.5 return (nanT,nanT); end)
            # Evaluate the value
            @nexprs $nd d->(x_d = u[d,I] + m_d + 1)
            ND = numdenomi[$(args...)]
            N += ND.num
            D += ND.denom
            # Evaluate the gradient
            if calc_grad
                gradient!(gtmp, numdenomi, $(args...))
                @nexprs $nd d->(gnumdenom[d, I] = gtmp[d])
            end
        end
        N, D
    end
end

#### Volume penalty ####
# For a deformation specified as a composition, phi1(phi2(x)), this
# calculates a penalty
#    sum_n(log(V_n/Vtarget)^2)
# where V_n is the volume of the nth simplex, and Vtarget is the
# "target volume" for each simplex. Consequently, it penalizes
# deformations that make the volume different from Vtarget.  The
# gradient with respect to phi2 can also be calculated.

# For rectangular grids, all possible oriented simplices constructed
# from nearest-neighbors are included in the sum.

# Based on:
#    B. Karacali and C. Davitzikos (2003). Topology preservation and
#    regularity in estimated deformation fields. Image Processing in
#    Medical Imaging 2732: 426-437.

# Copyright (2012) by Jian Wang and Timothy E. Holy

function grid_simplexes(gridstrides::Vector{Int})
    n_dims = length(gridstrides)
    if n_dims == 1
        vertices = reshape([1,2], 2, 1)  # reshape needed to keep the return type consistent
        coords = [0 1]
    elseif n_dims == 2
        vertices = [[3,1,2] [1,2,4] [2,4,3] [4,3,1]]
        coords = [[0,0] [1,0] [0,1] [1,1]]
    elseif n_dims == 3
        vertices = [[2,5,3,1] [1,4,6,2] [1,7,4,3] [2,3,8,4] [1,6,7,5] [2,8,5,6] [3,5,8,7] [4,7,6,8]]
        coords = [[0,0,0] [1,0,0] [0,1,0] [1,1,0] [0,0,1] [1,0,1] [0,1,1] [1,1,1]]
    else
        error("Dimensionality not supported")
    end
    index = zeros(Int, size(vertices))
    for i = 1:length(vertices)
        for idim = 1:n_dims
            index[i] += coords[idim,vertices[i]]*gridstrides[idim]
        end
    end
    return index, vertices, coords
end

# Loop over all simplexes
function penalty_volume{T}(g, ucomp::Array{T}, ucompgrad, Vtarget::Number, ws::ROWorkspace, Vslack=nothing)
    calc_grad = !isempty(g)
    if calc_grad
        if length(g) != length(ucomp)
            error("Gradient dimension mismatch")
        end
        fill!(g, 0)
    end
    λ = lambda_volume(ws)
    nd = size(ucomp, 1)
    nblocks = div(length(ucomp), nd)
    ugrid_offsets, vertices, coords = ws.vpindexes, ws.vpvertices, ws.vpcoords
    thiscoord, isimplex = ws.simplexcoords, ws.simplexindex
    val = zero(T)
    Vmin = convert(T, Inf)
    gridsize1 = [s-1 for s in ws.gridsize]
    scale = 1/prod(gridsize1)/2^nd
    for c = Counter(gridsize1)
        index = sub2indv(ws.gridsize, c)
        for isimp = 1:size(ugrid_offsets,2)
            for ii = 1:length(isimplex)
                isimplex[ii] = ugrid_offsets[ii, isimp] + index
            end
            for jj = 1:size(thiscoord, 2), ii = 1:size(thiscoord, 1)
                thiscoord[ii,jj] = coords[ii, vertices[jj, isimp]]
            end
            newval, Vnorm = penalty_volume(g, ucomp, ucompgrad, Vtarget, isimplex, thiscoord, scale, Vslack)
            val += newval
            Vmin = min(Vnorm, Vmin)::T
            if !isfinite(val) && Vslack == nothing
                return val, Vmin  # Abort early. This won't be a global Vmin, but for type stability we need to return something
            end
        end
    end
    if calc_grad
        for i = 1:length(g)
            g[i] *= λ
        end
    end
    return λ*val, Vmin
end
penalty_volume{T}(g, u::Array{T}, Vtarget, ws::ROWorkspace, Vslack=nothing) = penalty_volume(g, u, ws.ucompgrad, Vtarget, ws, Vslack)

# Calculate the contribution of one simplex
function penalty_volume{T}(g, ucomp::Array{T}, ucompgrad, Vtarget, isimplex::AbstractVector{Int}, coord::Array{T}, scale, Vslack)
    calc_grad = !isempty(g)
    n_dims = size(ucomp, 1)
    if n_dims != length(isimplex)-1
        error("Must have n+1 vertices in n dimensions")
    end
    if n_dims == 1
        f1 = ucomp[1, isimplex[1]] + coord[1, 1]
        f2 = ucomp[1, isimplex[2]] + coord[1, 2]
        V = f2 - f1
        Vadj = adjV(V, Vtarget, Vslack)
        if Vadj <= 0
            return inf(T), V/Vtarget
        end
        L = log(Vadj/Vtarget)
        if calc_grad
            coef = 2*L/Vadj*scale
            g[isimplex[1]] -= coef*ucompgrad[1, 1, isimplex[1]]
            g[isimplex[2]] += coef*ucompgrad[1, 1, isimplex[2]]
        end
        return L^2*scale, V/Vtarget
    elseif n_dims == 2
        i1 = isimplex[1]
        i2 = isimplex[2]
        i3 = isimplex[3]
        f1 = ucomp[1, i1] + coord[1, 1]
        g1 = ucomp[2, i1] + coord[2, 1]
        f2 = ucomp[1, i2] + coord[1, 2]
        g2 = ucomp[2, i2] + coord[2, 2]
        f3 = ucomp[1, i3] + coord[1, 3]
        g3 = ucomp[2, i3] + coord[2, 3]
#        println(coord)
#        println(ucomp)
#        println(isimplex)
        V = f1*g2 - f2*g1 - f1*g3 + f3*g1 + f2*g3 - f3*g2
#        if calc_grad
#            println("coord: ", coord)
#            println("fg: $f1 $g1 $f2 $g2 $f3 $g3")
#            println("V: ", V)
#        end
#        error("stop")
        Vadj = adjV(V, Vtarget, Vslack)
        if Vadj <= 0
            return inf(T), V/Vtarget
        end
        L = log(Vadj/Vtarget)
        if calc_grad
            df1 = g2-g3
            dg1 = f3-f2
            df2 = g3-g1
            dg2 = f1-f3
            df3 = g1-g2
            dg3 = f2-f1
            coef = 2*L/Vadj*scale
            for idim = 1:2
                g[idim+(i1-1)*2] += coef*(df1*ucompgrad[1, idim, i1] + dg1*ucompgrad[2, idim, i1])
                g[idim+(i2-1)*2] += coef*(df2*ucompgrad[1, idim, i2] + dg2*ucompgrad[2, idim, i2])
                g[idim+(i3-1)*2] += coef*(df3*ucompgrad[1, idim, i3] + dg3*ucompgrad[2, idim, i3])
            end
        end
        return L^2*scale, V/Vtarget
    elseif n_dims == 3
        i1 = isimplex[1]
        i2 = isimplex[2]
        i3 = isimplex[3]
        i4 = isimplex[4]
        f1 = ucomp[1, i1] + coord[1, 1]
        g1 = ucomp[2, i1] + coord[2, 1]
        h1 = ucomp[3, i1] + coord[3, 1]
        f2 = ucomp[1, i2] + coord[1, 2]
        g2 = ucomp[2, i2] + coord[2, 2]
        h2 = ucomp[3, i2] + coord[3, 2]
        f3 = ucomp[1, i3] + coord[1, 3]
        g3 = ucomp[2, i3] + coord[2, 3]
        h3 = ucomp[3, i3] + coord[3, 3]
        f4 = ucomp[1, i4] + coord[1, 4]
        g4 = ucomp[2, i4] + coord[2, 4]
        h4 = ucomp[3, i4] + coord[3, 4]
        V = f1*g3*h2 - f1*g2*h3 + f2*g1*h3 - f2*g3*h1 - f3*g1*h2 + f3*g2*h1 + f1*g2*h4 - f1*g4*h2 - f2*g1*h4 + f2*g4*h1 + f4*g1*h2 - f4*g2*h1 - f1*g3*h4 + f1*g4*h3 + f3*g1*h4 - f3*g4*h1 - f4*g1*h3 + f4*g3*h1 + f2*g3*h4 - f2*g4*h3 - f3*g2*h4 + f3*g4*h2 + f4*g2*h3 - f4*g3*h2
        Vadj = adjV(V, Vtarget, Vslack)
        if Vadj <= 0
            return inf(T), V/Vtarget
        end
        L = log(Vadj/Vtarget)
        if calc_grad
            df1 = g3*h2 - g2*h3 + g2*h4 - g4*h2 - g3*h4 + g4*h3
            dg1 = f2*h3 - f3*h2 - f2*h4 + f4*h2 + f3*h4 - f4*h3
            dh1 = f3*g2 - f2*g3 + f2*g4 - f4*g2 - f3*g4 + f4*g3

            df2 = g1*h3 - g3*h1 - g1*h4 + g4*h1 + g3*h4 - g4*h3
            dg2 = f3*h1 - f1*h3 + f1*h4 - f4*h1 - f3*h4 + f4*h3
            dh2 = f1*g3 - f3*g1 - f1*g4 + f4*g1 + f3*g4 - f4*g3

            df3 = g2*h1 - g1*h2 + g1*h4 - g4*h1 - g2*h4 + g4*h2
            dg3 = f1*h2 - f2*h1 - f1*h4 + f4*h1 + f2*h4 - f4*h2
            dh3 = f2*g1 - f1*g2 + f1*g4 - f4*g1 - f2*g4 + f4*g2

            df4 = g1*h2 - g2*h1 - g1*h3 + g3*h1 + g2*h3 - g3*h2
            dg4 = f2*h1 - f1*h2 + f1*h3 - f3*h1 - f2*h3 + f3*h2
            dh4 = f1*g2 - f2*g1 - f1*g3 + f3*g1 + f2*g3 - f3*g2
            coef = 2*L/Vadj*scale
            for idim = 1:3
                g[idim+(i1-1)*3] += coef*(df1*ucompgrad[1, idim, i1] + dg1*ucompgrad[2, idim, i1] + dh1*ucompgrad[3, idim, i1])
                g[idim+(i2-1)*3] += coef*(df2*ucompgrad[1, idim, i2] + dg2*ucompgrad[2, idim, i2] + dh2*ucompgrad[3, idim, i2])
                g[idim+(i3-1)*3] += coef*(df3*ucompgrad[1, idim, i3] + dg3*ucompgrad[2, idim, i3] + dh3*ucompgrad[3, idim, i3])
                g[idim+(i4-1)*3] += coef*(df4*ucompgrad[1, idim, i4] + dg4*ucompgrad[2, idim, i4] + dh4*ucompgrad[3, idim, i4])
            end
        end
        return L^2*scale, V/Vtarget
    else
        error("Dimensionality not supported")
    end
end

adjV(V, Vtarget, Vslack) = V+Vslack*Vtarget
adjV(V, Vtarget, ::Nothing) = V

# Convert u between pixel and normalized representations
function normalize_u!(u::Array, factor)
    if isempty(u)
        return
    end
    n_dims = length(factor)
    n = div(length(u), n_dims)
    for i = 1:n
        for idim = 1:n_dims
            u[idim+(i-1)*n_dims] /= factor[idim]
        end
    end
    u
end

function unnormalize_u!(u::Array, factor)
    if isempty(u)
        return
    end
    n_dims = length(factor)
    n = div(length(u), n_dims)
    for i = 1:n
        for idim = 1:n_dims
            u[idim+(i-1)*n_dims] *= factor[idim]
        end
    end
    u
end

### Affine-residual penalty
function penalty_affine_residual!(g, ucomp, ucompgrad, ws::ROWorkspace)
    nd = size(ucomp, 1)
    nblocks = prod(size(ucomp)[2:end])
    U = reshape(ucomp, nd, nblocks)
    Q = ws.Qaffine
    A = (U*Q)*Q'
    dU = U-A
    λ = ws.lambda_volume/nblocks
    if !isempty(g)
        copy!(g, (2λ)*dU)
        chainrule_ucomp!(g, ucompgrad)
    end
    λ * sumabs2(dU)
end
penalty_affine_residual!(g, ucomp, ws::ROWorkspace) = penalty_affine_residual!(g, ucomp, ws.ucompgrad, ws)

# This implements
#     for i = 1:nblocks
#         g[:,i] = ucompgrad[:,:,i]' * g[:,i]
#     end
# without temporary allocation, and without assuming g is shaped "properly"
function chainrule_ucomp!(g, ucompgrad)
    N = size(ucompgrad, 1)
    nblocks = size(ucompgrad,3)
    for i = 4:ndims(ucompgrad)
        nblocks *= size(ucompgrad,i)
    end
    length(g) == N*nblocks || throw(DimensionMismatch("g (size $(size(g))) and ucompgrad (size $(size(ucompgrad))) disagree"))
#     size(ucompgrad,1) == size(ucompgrad,2) == N || throw(DimensionMismatch("g (size $(size(g))) and ucompgrad (size $(size(ucompgrad))) disagree"))
#     size(g)[2:end] == size(ucompgrad)[3:end] || throw(DimensionMismatch("g (size $(size(g))) and ucompgrad (size $(size(ucompgrad))) disagree"))
    tmp = Array(eltype(g), N)
    if N == 1
        for iblock = 1:nblocks
            g[iblock] *= ucompgrad[1,1,iblock]
        end
    elseif N == 2
        for iblock = 1:nblocks
            i1 = 1+(iblock-1)*N
            for i = 1:2
                tmp[i] = ucompgrad[1,i,iblock]*g[i1] + ucompgrad[2,i,iblock]*g[i1+1]
            end
            g[i1]   = tmp[1]
            g[i1+1] = tmp[2]
        end
    elseif N == 3
        for iblock = 1:nblocks
            i1 = 1+(iblock-1)*N
            for i = 1:3
                tmp[i] = ucompgrad[1,i,iblock]*g[i1] + ucompgrad[2,i,iblock]*g[i1+1] + ucompgrad[3,i,iblock]*g[i1+2]
            end
            g[i1]   = tmp[1]
            g[i1+1] = tmp[2]
            g[i1+2] = tmp[3]
        end
    else
        error("Dimensionality $N not supported")
    end
end

# Composition via quadratic interpolation
# use interp_invert! on uold before calling this function
# Storage order of gradient:
#   g[i, j, k1, k2, ...] is the derivative of ucomp[i, k1, k2, ...]
#       with respect to unew[j, k1, k2, ...]
# This requires that uold and unew be normalized upon entry (see penalty())
function compose_u!{T}(g, uold, unew, ws::AbstractROWorkspace{T})
    calc_grad = !isempty(g)
    n_dims = size(unew, 1)
    n_blocks = div(length(unew), n_dims)
    if isempty(uold)
        if calc_grad
            imax = n_dims^2*n_blocks
            i = 1
            while i <= imax
                for idimg = 1:n_dims
                    for idim = 1:n_dims
                        g[i] = (idim == idimg)
                        i += 1
                    end
                end
            end
        end
        return copy(unew)
    end
    u = similar(unew)
    coef = ws.interpu
    x = Array(T, n_dims)
    uindex = 1
    gindex = 1
    for c = Counter(ws.gridsize)
        for idim = 1:n_dims
            x[idim] = unew[uindex+idim-1] + c[idim]
        end
        set_position(coef, BCnearest, calc_grad, x)
        for idim = 1:n_dims
            u[uindex] = interp(coef, uold, idim) + unew[uindex]
            uindex += 1
        end
        if calc_grad
            for idimg = 1:n_dims
                set_gradient_coordinate(coef, idimg)
                for idim = 1:n_dims
                    g[gindex] = interp(coef, uold, idim) + (idim == idimg)
                    gindex += 1
                end
            end
        end
    end
    return u
end

type SimpleWorkspace{T,N} <: AbstractROWorkspace{T,N}
    interpu::InterpGridCoefs{T,InterpQuadratic}
    gridsize::NTuple{N,Int}
end
function SimpleWorkspace(T, gridsize)
    N = length(gridsize)
    strides = Array(Int, N)
    strides[1] = N
    for idim = 2:N
        strides[idim] = strides[idim-1]*gridsize[idim-1]
    end
    SimpleWorkspace{T,N}(InterpGridCoefs(T, InterpQuadratic, gridsize, strides), tuple(gridsize...))
end

lambda_volume{T}(ws::SimpleWorkspace{T}) = zero(T)

# user-callable composition
# Do _not_ call interp_invert on uold before calling this function
function compose_u(uold, unew, arraysize)
    if isempty(uold) || uold == nothing
        return unew
    end
    size(uold) == size(unew) || error("Sizes must match")
    unew = copy(unew)
    uold = copy(uold)
    T = eltype(unew)
    gridsize = size(unew)[2:end]
    N = length(gridsize)
    bricksize = [(arraysize[i]-1)/(max(1,gridsize[i]-1)) for i = 1:N]
    uoldi = interp_invert!(normalize_u!(uold, bricksize), BCnearest, InterpQuadratic, 2:N+1)
    normalize_u!(unew, bricksize)
    ws = SimpleWorkspace(T, gridsize)
    ucomp = compose_u!(T[], uold, unew, ws)
    unnormalize_u!(ucomp, bricksize)
end

# function invertmm!(nums, denoms)
#     if isa(nums[1], AbstractArray)
#         # array-of-arrays
#         nd = ndims(nums[1])
#         for i = 1:length(nums)
#             prefilter!(nums[i], 1:nd)
#             prefilter!(denoms[i], 1:nd)
# #             interp_invert!(nums[i], BCnil, InterpQuadratic, 1:nd)
# #             interp_invert!(denoms[i], BCnil, InterpQuadratic, 1:nd)
#         end
#     else
#         # tiled
#         iseven(ndims(nums)) || error("Must have an even number of dimensions")
#         nd = ndims(nums)>>1
#         prefilter!(nums, 1:nd)
#         prefilter!(denoms, 1:nd)
# #         interp_invert!(nums, BCnil, InterpQuadratic, 1:nd)
# #         interp_invert!(denoms, BCnil, InterpQuadratic, 1:nd)
#     end
#     nums, denoms
# end
#
# # Stolen from Interpolations.prefilter!
# function prefilter!{T,N}(A::AbstractArray{T,N}, dims)
#     sz = size(A)
#     for dim in dims
#         M, b = Interpolations.prefiltering_system(T, T, sz[dim], Interpolations.Quadratic(Interpolations.Flat), Interpolations.OnCell)
#         A_ldiv_B_md!(A, M, A, dim, b)
#     end
#     A
# end

function sub2indv{I<:Integer}(dims, v::AbstractVector{I})
    length(dims) == length(v) || throw(DimensionMismatch("dims and v must have the same length"))
    ind = convert(Int, v[end])-1
    for i = length(dims)-1:-1:1
        ind = dims[i]*ind + (v[i]-1)
    end
    ind+1
end

end
