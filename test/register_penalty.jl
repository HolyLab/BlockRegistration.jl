using Base.Test, DualNumbers, AffineTransforms, FixedSizeArrays, Interpolations
using RegisterCore, RegisterDeformation
import RegisterPenalty
RP = RegisterPenalty

gridsize = (9,7)
maxshift = (3,3)
imgsize = (1000,1002)
knots = map(d->linspace(1,imgsize[d],gridsize[d]), 1:length(gridsize))
dp = RegisterPenalty.AffinePenalty(knots, 1.0)

# Zero penalty for translations
ϕ_new = tform2deformation(tformtranslate([0.3,0.05]), imgsize, gridsize)
ϕ_old = interpolate(tform2deformation(tformtranslate([0.1,0.2]),  imgsize, gridsize))
g = similar(ϕ_new.u)
@test abs(RP.penalty!(g, dp, ϕ_new)) < 1e-12
@test all(x->sum(abs(x)) < 1e-12, g)
ϕ_c, g_c = compose(ϕ_old, ϕ_new)
@test abs(RP.penalty!(g, dp, ϕ_c, g_c)) < 1e-12
@test all(x->sum(abs(x)) < 1e-12, g)

# Zero penalty for rotations
ϕ = tform2deformation(tformrotate(10pi/180), imgsize, gridsize)
@test abs(RP.penalty!(g, dp, ϕ)) < 1e-12
@test all(x->sum(abs(x)) < 1e-12, g)

# Zero penalty for stretch and skew
A = 0.2*rand(2,2) + eye(2,2)
tform = AffineTransform(A, Int[g>>1 for g in gridsize])
ϕ = tform2deformation(tform, imgsize, gridsize)
@test abs(RP.penalty!(g, dp, ϕ)) < 1e-12
@test all(x->sum(abs(x)) < 1e-12, g)

# Random deformations & affine-residual penalty
gridsize = (3,3)
knots = map(d->linspace(1,imgsize[d],gridsize[d]), 1:length(gridsize))
dp = RegisterPenalty.AffinePenalty(knots, 1.0)
u = randn(2, gridsize...)
ϕ = GridDeformation(u, imgsize)
g = similar(ϕ.u)
RP.penalty!(g, dp, ϕ)
for i = 1:prod(gridsize)
    for j = 1:2
        ud = dual(u)
        ud[j,i] = dual(u[j,i], 1)
        ϕd = GridDeformation(ud, imgsize)
        pd = RP.penalty!(nothing, dp, ϕd)
        @test_approx_eq_eps g[i][j] epsilon(pd) 100*eps()
    end
end

# Random deformations with composition
uold = randn(2, gridsize...)
ϕ_old = interpolate(GridDeformation(uold, imgsize))
ϕ_c, g_c = compose(ϕ_old, ϕ)
RP.penalty!(g, dp, ϕ_c, g_c)
for i = 1:prod(gridsize)
    for j = 1:2
        ud = dual(u)
        ud[j,i] = dual(u[j,i], 1)
        ϕd = interpolate(GridDeformation(ud, imgsize))
        pd = RP.penalty!(nothing, dp, ϕ_old(ϕd))
        @test_approx_eq_eps g[i][j] epsilon(pd) 100*eps()
    end
end

################
# Data penalty #
################
gridsize = (1,1)
maxshift = (3,3)
imgsize = (1,1)

p = [(x-1.75)^2 for x = 1:7]
nums = reshape(p*p', length(p), length(p))
denoms = similar(nums); fill!(denoms, 1)
mm = MismatchArray(nums, denoms)
mmi = RP.interpolate_mm!(mm; BC=InPlaceQ)
mmi_array = typeof(mmi)[mmi]
ϕ = GridDeformation(zeros(2, 1, 1), imgsize)
g = similar(ϕ.u)
fill!(g, 0)
val = RP.penalty!(g, ϕ, mmi_array)
@test_approx_eq val (4-1.75)^4
@test_approx_eq g[1][1] 2*(4-1.75)^3
# Test at the minimum
fill!(g, 0)
ϕ = GridDeformation(reshape([-2.25,-2.25], (2,1,1)), imgsize)
@test RP.penalty!(g, ϕ, mmi_array) < eps()
@test abs(g[1][1]) < eps()


# A biquadratic penalty---make sure we calculate the exact values
gridsize = (2,2)
maxshift = [3,4]
imgsize = (101,100)

minrange = 1.6
maxrange = Float64[2*m+1-0.6 for m in maxshift]
dr = maxrange.-minrange
c = dr.*rand(2, gridsize...).+minrange
nums = Array(Matrix{Float64}, 2, 2)
shiftsize = 2maxshift+1
for j = 1:gridsize[2], i = 1:gridsize[1]
    p = [(x-c[1,i,j])^2 for x in 1:shiftsize[1]]
    q = [(x-c[2,i,j])^2 for x in 1:shiftsize[2]]
    nums[i,j] = p*q'
end
denom = ones(nums[1,1])
mms = mismatcharrays(nums, denom)
mmis = RP.interpolate_mm!(mms; BC=Interpolations.InPlaceQ)
u_real = (dr.*rand(2, gridsize...).+minrange) .- Float64[m+1 for m in maxshift]  #zeros(size(c)...)
ϕ = GridDeformation(u_real, imgsize)
g = similar(ϕ.u)
fill!(g, 0)
val = RP.penalty!(g, ϕ, mmis)
nblocks = prod(gridsize)
valpred = sum([prod([(maxshift[k]+1+u_real[k,i,j]-c[k,i,j])^2 for k = 1:2]) for i=1:gridsize[1],j=1:gridsize[2]])/nblocks
@test_approx_eq val valpred
for j = 1:gridsize[2], i = 1:gridsize[1]
    @test_approx_eq_eps g[i,j][1] 2*(maxshift[1]+1+u_real[1,i,j]-c[1,i,j])*(maxshift[2]+1+u_real[2,i,j]-c[2,i,j])^2/nblocks 1000*eps()
    @test_approx_eq_eps g[i,j][2] 2*(maxshift[1]+1+u_real[1,i,j]-c[1,i,j])^2*(maxshift[2]+1+u_real[2,i,j]-c[2,i,j])/nblocks 1000*eps()
end


#################
# total penalty #
#################
# So far we've done everything in 2d, but now test all relevant dimensionalities
for nd = 1:3
    gridsize = tuple(collect(3:nd+2)...)
    maxshift = collect(nd+2:-1:3)
    imgsize  = tuple(collect(101:100+nd)...)
    shiftsize = 2maxshift+1
    nblocks = prod(gridsize)

    # Set up the data penalty (nums and denoms)
    minrange = 1.6
    maxrange = Float64[2*m+1-0.6 for m in maxshift]
    dr = maxrange.-minrange
    c = dr.*rand(nd, gridsize...).+minrange
    nums = Array(Array{Float64,nd}, gridsize)
    for I in CartesianRange(gridsize)
        n = 1
        for j = 1:nd
            s = ones(Int, nd)
            s[j] = shiftsize[j]
            n = n.*reshape([(x-c[j,I])^2 for x in 1:shiftsize[j]], s...)
        end
        nums[I] = n
    end
    mms = mismatcharrays(nums, fill(1.0, size(first(nums))))
    mmis = RP.interpolate_mm!(mms; BC=Interpolations.InPlaceQ)

    # If we start right at the minimum, and there is no volume
    # penalty, the value should be zero
    ϕ = GridDeformation(c .- maxshift .- 1, imgsize)
    dp = RegisterPenalty.AffinePenalty(ϕ.knots, 0.0)
    g = similar(ϕ.u)
    val = RP.penalty!(g, ϕ, identity, dp, mmis)
    @test abs(val) < 100*eps()
    gr = reinterpret(Float64, g, (nd, length(g)))
    @test maximum(abs(gr)) < 100*eps()

    # Test derivatives with no uold
    u_raw = dr.*rand(nd, gridsize...) .+ minrange .- maxshift .- 1  # a random location
    ϕ = GridDeformation(u_raw, imgsize)
    dx = u_raw - (c .- maxshift .- 1)
    valpred = sum(prod(dx.^2,1))/nblocks
    g = similar(ϕ.u)
    val0 = RP.penalty!(g, ϕ, identity, dp, mmis)
    @test_approx_eq val0 valpred
    for I in CartesianRange(gridsize)
        for idim = 1:nd
            gpred = 2/nblocks
            for jdim = 1:nd
                gpred *= (jdim==idim) ? dx[jdim, I] : dx[jdim, I]^2
            end
            @test_approx_eq_eps g[I][idim] gpred 1000*eps()
        end
    end
    # set lambda so the volume and data penalties contribute equally
    dp.λ = 1
    p = RP.penalty!(nothing, dp, ϕ)
    dp.λ = val0/p
    val = RP.penalty!(g, ϕ, identity, dp, mmis)
    @test_approx_eq val 2val0
    for i = 1:prod(gridsize)
        for idim = 1:nd
            ud = convert(Array{Dual{Float64}}, u_raw)
            ud[idim,i] = dual(u_raw[idim,i], 1)
            vald = RP.penalty!(nothing, GridDeformation(ud, imgsize), identity, dp, mmis)
            @test_approx_eq g[i][idim] epsilon(vald)
        end
    end

    # Include uold
    uold = dr.*rand(nd, gridsize...) .+ minrange .- maxshift .- 1
    ϕ_old = interpolate(GridDeformation(uold, imgsize))
    val = RP.penalty!(g, ϕ, ϕ_old, dp, mmis)
    for i = 1:prod(gridsize)
        for idim = 1:nd
            ud = convert(Array{Dual{Float64}}, u_raw)
            ud[idim,i] = dual(u_raw[idim,i], 1)
            vald = RP.penalty!(nothing, GridDeformation(ud, imgsize), ϕ_old, dp, mmis)
            @test_approx_eq g[i][idim] epsilon(vald)
        end
    end

    @test_throws ErrorException RP.penalty!(g, interpolate(ϕ), ϕ_old, dp, mmis)
end
