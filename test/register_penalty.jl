using Grid, Base.Test, DualNumbers
# using Base.Test, DualNumbers
using AffineTransforms, RegisterCore
import RegisterPenalty, RegisterDeformation
RP = RegisterPenalty

gridsize = (9,7)
maxshift = (3,3)
imgsize = (1000,1002)
bricksize = [imgsize[i]/(gridsize[i]-1) for i = 1:2]
ws = RP.ROWorkspace(Float64, maxshift, gridsize, bricksize)
ws.lambda_volume = 1.0
bc = BCnearest

##############
# compose_u! #
##############
# Translations
unew_shift = [0.3,0.05]
unew = repeat(unew_shift, outer=[1,gridsize...])
uold = Float64[]
ucomp = RP.compose_u!(Float64[], uold, unew, ws)
@test ucomp == unew
g = Array(Float64, 2, 2, gridsize...)
ucomp = RP.compose_u!(g, uold, unew, ws)
@test ucomp == unew
for i = 1:prod(gridsize)
    @test g[:,:,i] == eye(2,2)
end

uold_shift = [0.1,0.2]
uold = repeat(uold_shift, outer=[1,gridsize...])
interp_invert!(uold, bc, InterpQuadratic, (2,3))
ucomp = RP.compose_u!(Float64[], uold, unew, ws)
@test_approx_eq ucomp repeat(uold_shift+unew_shift, outer = [1,gridsize...])
ucomp = RP.compose_u!(g, uold, unew, ws)
@test_approx_eq ucomp repeat(uold_shift+unew_shift, outer = [1,gridsize...])
for i = 1:prod(gridsize)
    @test_approx_eq g[:,:,i] eye(2,2)
end

# Rotations
function rotation_u(angle, gridsize)
    TF = tformrotate(angle)
    RegisterDeformation.tform2u(TF, gridsize, gridsize)
end
angle = pi/180  # a 1-degree rotation
u1 = rotation_u(angle, gridsize)
u1old = interp_invert!(copy(u1), bc, InterpQuadratic, (2,3))
ucomp = RP.compose_u!(g, u1old, u1, ws)
# check that composition of rotations is a rotation
u2 =  rotation_u(2angle, gridsize)
du = ucomp-u2
du[:,[1,end],:] = 0  # edges are a little borked, but that's expected
du[:,:,[1,end]] = 0
expectedthresh = (maximum(abs(diff(squeeze(u1[1,:,:],1)))) + maximum(abs(diff(squeeze(u1[2,:,:],1)))))*maximum(abs(u1))
@test_approx_eq_eps ucomp u2 expectedthresh
# Check that the derivative with respect to unew is accurate
wsd = RP.ROWorkspace(Dual{Float64}, maxshift, gridsize, bricksize)
wsd.lambda_volume = ws.lambda_volume
for i = 1:prod(gridsize)
    u1d = convert(Array{Dual{Float64}}, u1)
    u1d[1,i] = dual(u1[1,i], 1)
    ucompd = RP.compose_u!(Dual{Float64}[], u1old, u1d, wsd)
    gd = map(epsilon, ucompd[:,i])
    @test_approx_eq gd g[:,1,i]
    u1d = convert(Array{Dual{Float64}}, u1)
    u1d[2,i] = dual(u1[2,i], 1)
    ucompd = RP.compose_u!(Dual{Float64}[], u1old, u1d, wsd)
    gd = map(epsilon, ucompd[:,i])
    @test_approx_eq gd g[:,2,i]
end

###################
# penalty_volume! #
###################
# Zero penalty for translations
unew_shift = [0.3,0.05]
unew = repeat(unew_shift, outer=[1,gridsize...])
uold_shift = [0.1,0.2]
uold = repeat(uold_shift, outer=[1,gridsize...])
interp_invert!(uold, bc, InterpQuadratic, (2,3))
gucomp = Array(Float64, 2, 2, gridsize...)
ucomp = RP.compose_u!(gucomp, uold, unew, ws)
g = Float64[]
@test RP.penalty_volume(g, ucomp, gucomp, 1, ws) == (0.0, 1.0)
g = similar(ucomp)
@test RP.penalty_volume(g, ucomp, gucomp, 1, ws) == (0.0, 1.0)
@assert all(g .== 0)
@test RP.penalty_affine_residual!(g, ucomp, gucomp, ws) < 1e-20

# Zero penalty for rotations
u1 = rotation_u(10*pi/180, gridsize)
ucomp = RP.compose_u!(gucomp, Float64[], u1, ws)
p = RP.penalty_volume(g, ucomp, gucomp, 1, ws)[1]
@test p < eps()
@test RP.penalty_affine_residual!(g, ucomp, gucomp, ws) < 1e-20

# Non-zero penalty for stretch and skew
A = 0.2*rand(2,2) + eye(2,2)
TF = AffineTransform(A, Int[g>>1 for g in gridsize])
unew = RegisterDeformation.tform2u(TF, gridsize, gridsize)
ucomp = RP.compose_u!(gucomp, Float64[], unew, ws)
p = RP.penalty_volume(g, ucomp, gucomp, 1, ws)[1]
ncells = (gridsize[1]-1)*(gridsize[2]-1)
@test_approx_eq p log(det(A))^2
gucompd = similar(gucomp, Dual{Float64})
for i = 1:prod(gridsize)
    for j = 1:2
        unewd = convert(Array{Dual{Float64}}, unew)
        unewd[j,i] = dual(unew[j,i], 1)
        ucompd = RP.compose_u!(gucompd, Float64[], unewd, wsd)
        pd = RP.penalty_volume(Dual{Float64}[], ucompd, gucompd, 1, wsd)[1]
        @test_approx_eq_eps g[j,i] epsilon(pd) 10*eps()
    end
end
@test RP.penalty_affine_residual!(g, ucomp, gucomp, ws) < 1e-20


Aold = 0.2*rand(2,2) + eye(2,2)
TF = AffineTransform(Aold, Int[g>>1 for g in gridsize])
uold = RegisterDeformation.tform2u(TF, gridsize, gridsize)
interp_invert!(uold, bc, InterpQuadratic, (2,3))
ucomp = RP.compose_u!(gucomp, uold, unew, ws)
p = RP.penalty_volume(g, ucomp, gucomp, 1, ws)
# To test whether the penalty is accurate, we need to estimate the
# composition error from the interpolation
TFb = AffineTransform(Aold, [0,0]) * AffineTransform(A, Int[g>>1 for g in gridsize])
ub = RegisterDeformation.tform2u(TFb, gridsize, gridsize)
pb = RP.penalty_volume(Float64[], ub, Float64[], 1, ws)
udiff = ub - ucomp
pdiff = RP.penalty_volume(Float64[], udiff, Float64[], 1, ws)[1]
ddiff = sqrt(pdiff)
# @test log((1-ddiff)*det(A)*det(Aold))^2  < p < log((1+ddiff)*det(A)*det(Aold))^2
for i = 1:prod(gridsize)
    for j = 1:2
        unewd = convert(Array{Dual{Float64}}, unew)
        unewd[j,i] = dual(unew[j,i], 1)
        ucompd = RP.compose_u!(gucompd, uold, unewd, wsd)
        pd = RP.penalty_volume(Dual{Float64}[], ucompd, gucompd, 1, wsd)[1]
        @test_approx_eq_eps g[j,i] epsilon(pd) 100*eps()
    end
end

# Random deformations & affine-residual penalty
unew = randn(2, gridsize...)
RP.penalty_affine_residual!(g, unew, ws)
for i = 1:prod(gridsize)
    for j = 1:2
        unewd = dual(unew)
        unewd[j,i] = dual(unew[j,i], 1)
        pd = RP.penalty_affine_residual!(Dual{Float64}[], unewd, wsd)
        @test_approx_eq_eps g[j,i] epsilon(pd) 100*eps()
    end
end
uold = randn(2, gridsize...)
interp_invert!(uold, bc, InterpQuadratic, (2,3))
ucomp = RP.compose_u!(gucomp, uold, unew, ws)
RP.penalty_affine_residual!(g, ucomp, gucomp, ws)
for i = 1:prod(gridsize)
    for j = 1:2
        unewd = convert(Array{Dual{Float64}}, unew)
        unewd[j,i] = dual(unew[j,i], 1)
        ucompd = RP.compose_u!(gucompd, uold, unewd, wsd)
        pd = RP.penalty_affine_residual!(Dual{Float64}[], ucompd, gucompd, wsd)
        @test_approx_eq_eps g[j,i] epsilon(pd) 100*eps()
    end
end

################
# Data penalty #
################
gridsize = (1,1)
maxshift = (3,3)
imgsize = (100,102)
bricksize = [imgsize[i] for i = 1:2]
ws = RP.ROWorkspace(Float64, maxshift, gridsize, bricksize)

p = [(x-1.75)^2 for x = 1:7]
nums = reshape(p*p', length(p), length(p), 1, 1)
denoms = similar(nums); fill!(denoms, 1)
numsdenomsi = RP.interpolate_nd!(pack_nd(nums, denoms); BC=Interpolations.InPlaceQ)
numsi, denomsi = RP.interpolate_nd!(nums, denoms; BC=Interpolations.InPlaceQ)
u = zeros(2, 1, 1)
g = similar(u)
val = RP.value!(g, u, (numsi, denomsi), ws)
@test_approx_eq val (4-1.75)^4
@test_approx_eq g[1] 2*(4-1.75)^3
val = RP.value!(g, u, numsdenomsi, ws)
@test_approx_eq val (4-1.75)^4
@test_approx_eq g[1] 2*(4-1.75)^3
# Test at the minimum
u[1] = u[2] = -2.25
@test RP.value!(g, u, (numsi, denomsi), ws) < eps()
@test abs(g[1]) < eps()
@test RP.value!(g, u, numsdenomsi, ws) < eps()
@test abs(g[1]) < eps()

gridsize = (2,2)
maxshift = (3,4)
bricksize = [imgsize[i]/(gridsize[i]-1) for i = 1:2]
ws = RP.ROWorkspace(Float64, maxshift, gridsize, bricksize)
blocksize = (2*maxshift[1]+1, 2*maxshift[2]+1)

minrange = 1.6
maxrange = Float64[2*m+1-0.6 for m in maxshift]
dr = maxrange.-minrange
c = dr.*rand(2, gridsize...).+minrange
nums = Array(Float64, blocksize..., gridsize...)
for j = 1:gridsize[2], i = 1:gridsize[1]
    p = [(x-c[1,i,j])^2 for x in 1:blocksize[1]]
    q = [(x-c[2,i,j])^2 for x in 1:blocksize[2]]
    nums[:,:,i,j] = p*q'
end
denoms = similar(nums); fill!(denoms, 1)
numsdenomsi = RP.interpolate_nd!(pack_nd(nums, denoms); BC=Interpolations.InPlaceQ)
u = (dr.*rand(2, gridsize...).+minrange) .- Float64[m+1 for m in maxshift]  #zeros(size(c)...)
g = similar(u)
val = RP.value!(g, u, numsdenomsi, ws)
nblocks = prod(gridsize)
valpred = sum([prod([(maxshift[k]+1+u[k,i,j]-c[k,i,j])^2 for k = 1:2]) for i=1:gridsize[1],j=1:gridsize[2]])/nblocks
@test_approx_eq val valpred
for j = 1:gridsize[2], i = 1:gridsize[1]
    @test_approx_eq_eps g[1,i,j] 2*(maxshift[1]+1+u[1,i,j]-c[1,i,j])*(maxshift[2]+1+u[2,i,j]-c[2,i,j])^2/nblocks 1000*eps()
    @test_approx_eq_eps g[2,i,j] 2*(maxshift[1]+1+u[1,i,j]-c[1,i,j])^2*(maxshift[2]+1+u[2,i,j]-c[2,i,j])/nblocks 1000*eps()
end


#################
# total penalty #
#################
# So far we've done everything in 2d, but now test all dimensionalities
for nd = 1:3
    gridsize = collect(3:nd+2)
    maxshift = collect(nd+2:-1:3)
    imgsize  = collect(101:100+nd)
    bricksize = Float64[imgsize[i]/(gridsize[i]-1) for i = 1:nd]
    ws = RP.ROWorkspace(Float64, maxshift, gridsize, bricksize)
    blocksize = [2*m+1 for m in maxshift]
    colons = ntuple(i->1:blocksize[i], nd)
    nblocks = prod(gridsize)

    # Set up the data penalty (nums and denoms)
    minrange = 1.6
    maxrange = Float64[2*m+1-0.6 for m in maxshift]
    dr = maxrange.-minrange
    c = dr.*rand(nd, gridsize...).+minrange
    nums = Array(Float64, blocksize..., gridsize...)
    for cntr in Counter(gridsize)
        n = 1
        for j = 1:nd
            s = ones(Int, nd)
            s[j] = blocksize[j]
            n = n.*reshape([(x-c[j,cntr...])^2 for x in 1:blocksize[j]], s...)
        end
        nums[colons...,cntr...] = n
    end
    denoms = similar(nums); fill!(denoms, 1)
    numsdenomsi = RP.interpolate_nd!(pack_nd(nums, denoms); BC=Interpolations.InPlaceQ)
    numsi, denomsi = RP.interpolate_nd!(nums, denoms; BC=Interpolations.InPlaceQ)

    # If we start right at the minimum, and there is no volume penalty, the value should be zero
    ws.lambda_volume = 0.0
    uc = c .- maxshift .- 1
    g = similar(uc)
    gv = reshape(g, length(g))
    val = RP.penalty!(gv, uc[:], (numsi, denomsi), Float64[], ws)
    @test abs(val) < 100*eps()
    @test maximum(abs(g)) < 100*eps()
    val = RP.penalty!(gv, uc[:], numsdenomsi, Float64[], ws)
    @test abs(val) < 100*eps()
    @test maximum(abs(g)) < 100*eps()

    # Test derivatives with no uold
    u = dr.*rand(nd, gridsize...) .+ minrange .- maxshift .- 1  # a random location
    dx = u - (c .- maxshift .- 1)
    valpred = sum(prod(dx.^2,1))/nblocks
    val0 = 0.0
    for input in ((numsi,denomsi), numsdenomsi)
        val0 = RP.penalty!(gv, u[:], (numsi, denomsi), Float64[], ws)
        @test_approx_eq val0 valpred
        for cntr in Counter(gridsize)
            for idim = 1:nd
                gpred = 2/nblocks
                for jdim = 1:nd
                    gpred *= (jdim==idim) ? dx[jdim, cntr...] : dx[jdim, cntr...]^2
                end
                @test_approx_eq_eps g[idim,cntr...] gpred 1000*eps()
            end
        end
    end
    # set lambda so the volume and data penalties contribute equally
    unorm = copy(u)
    RP.normalize_u!(unorm, ws.bricksize)
    ws.lambda_volume = 1
#     p = RP.penalty_volume(Float64[], unorm, Float64[], 1, ws)[1]
    p = RP.penalty_affine_residual!(Float64[], unorm, ws)
    ws.lambda_volume = val0/p
    val = RP.penalty!(gv, u[:], (numsi, denomsi), Float64[], ws)
    @test_approx_eq val 2val0
    wsd = RP.ROWorkspace(Dual{Float64}, maxshift, gridsize, bricksize)
    wsd.lambda_volume = dual(ws.lambda_volume, 0)
    for i = 1:prod(gridsize)
        for idim = 1:nd
            ud = convert(Array{Dual{Float64}}, u)
            ud[idim,i] = dual(u[idim,i], 1)
            vald = RP.penalty!(Dual{Float64}[], ud[:], (numsi, denomsi), Float64[], wsd)
            @test_approx_eq g[idim,i] epsilon(vald)
        end
    end

    # Include uold
    uold = dr.*rand(nd, gridsize...) .+ minrange .- maxshift .- 1
    val = RP.penalty!(gv, u[:], (numsi, denomsi), uold, ws)
    for i = 1:prod(gridsize)
        for idim = 1:nd
            ud = convert(Array{Dual{Float64}}, u)
            ud[idim,i] = dual(u[idim,i], 1)
            vald = RP.penalty!(Dual{Float64}[], ud[:], (numsi, denomsi), uold, wsd)
            @test_approx_eq g[idim,i] epsilon(vald)
        end
    end
end
