using Interpolations, BlockRegistration, RegisterMismatch
using Interpolations: sqr, SimpleRatio, BSplineInterpolation, DimSpec, Degree
import RegisterHindsight
using DualNumbers, StaticArrays
using TestImages
using Base.Test

#add jitter in sampling location, simulating inconsistencies in piezo position when using OCPI under certain conditions
function jitter(img::Array{T,1}, npix::Float64) where T
    etp = extrapolate(interpolate(img, BSpline(Linear()),OnGrid()), Flat())
    out = zeros(eltype(img), size(img))
    z_def = Float64[]
    r = 0.0
    for i in 1:length(img)
        # To ensure that our sampling satisfies the Nyquist criterion, smooth r
        r = (r + (rand()*2*npix)-npix)/2  # exponential filter
        push!(z_def, r)
        out[i] = etp[i+r]
    end
    return out, z_def
end

#fill each real-valued component of g with the respective gradient entry
#temporarily sets the epsilon components of g to compute the gradient
function dualgrad_data!(g, ϕ, fixed, moving)
    ur = RegisterDeformation.convert_from_fixed(ϕ.u.itp.coefs)
    gr = RegisterDeformation.convert_from_fixed(g)
    nd = size(ur, 1)
    for i in CartesianRange(indices(ϕ.u.itp.coefs))
        for j = 1:nd
            temp = ur[j, i]
            ur[j, i] = dual(DualNumbers.value(temp), 1.0)
            gr[j, i] = epsilon(RegisterHindsight.penalty_hindsight_data(ϕ, fixed, moving))
            ur[j, i] = temp
        end
    end
end

function dualgrad_reg!(g, ap, ϕ)
    ur = RegisterDeformation.convert_from_fixed(ϕ.u.itp.coefs)
    gr = RegisterDeformation.convert_from_fixed(g)
    nd = size(ur, 1)
    for i in CartesianRange(indices(ϕ.u.itp.coefs))
        for j = 1:nd
            temp = ur[j, i]
            ur[j, i] = dual(DualNumbers.value(temp), 1.0)
            gr[j, i] = epsilon(RegisterHindsight.penalty_hindsight_reg(ap, ϕ))
            ur[j, i] = temp
        end
    end
end


function test_hindsight(fixed, moving, ϕ0, ap)
    print("Beginning new test run\n")
    u0 = RegisterDeformation.convert_from_fixed(ϕ0.u)
    ϕ = interpolate!(deepcopy(ϕ0))  # don't "destroy" u0
    g_data = similar(ϕ.u.itp.coefs)

    emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat())), OnCell()), NaN)

    #compare penalty functions with various levels of optimization. TODO: add another simpler method
    pdata1 = RegisterHindsight.penalty_hindsight_data(ϕ, fixed, emoving)
    preg1 = RegisterHindsight.penalty_hindsight_reg(ap, ϕ)
    ptotal = RegisterHindsight.penalty_hindsight(ϕ, ap, fixed, emoving)
    @test ptotal == pdata1 + preg1
    pdata2 = RegisterHindsight.penalty_hindsight_data!(g_data, ϕ, fixed, emoving) #fully optimized version
    @test pdata1 == pdata2

    ϕ0_dual = GridDeformation(map(dual, u0), ϕ.knots)
    ϕ_dual = interpolate!(ϕ0_dual)
    for i in eachindex(ϕ.u.itp.coefs)
        @test ϕ.u.itp.coefs[i] == real(ϕ_dual.u.itp.coefs[i])
    end

    g_reg = similar(g_data)
    g_total = similar(g_data)
    g_data_dual = similar(g_data)
    g_reg_dual = similar(g_data)

    #test data penalty gradient
    dualgrad_data!(g_data_dual, ϕ_dual, fixed, emoving)
    for i in eachindex(g_data)
        @test g_data[i] ≈ g_data_dual[i]
    end

    #test that gradients sum properly
    RegisterHindsight.penalty_hindsight!(g_total, ϕ, ap, fixed, emoving)
    RegisterHindsight.penalty_hindsight_reg!(g_reg, ap, ϕ)
    @test g_total == g_data .+ g_reg

    #test affine penalty gradient
    dualgrad_reg!(g_reg_dual, ap, ϕ_dual)
    for i in eachindex(g_reg)
        @test g_reg[i] ≈ g_reg_dual[i]
    end
    print("Test run successful\n")
end

#1-dimensional images
fixed = sin.(linspace(0,4π,40))
moving, z_def = jitter(fixed, 0.45);

λ = 1e-3
gridsize = (length(fixed),)
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1:ndims(fixed)...))
ap = AffinePenalty{Float64,ndims(fixed)}(knots, λ)
u0 = zeros(1, gridsize...)
ϕ0 = GridDeformation(u0, knots)

test_hindsight(fixed, moving, ϕ0, ap)

u0 = rand(1, gridsize...)./10
ϕ0 = GridDeformation(u0, knots)
test_hindsight(fixed, moving, ϕ0, ap)


emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat())), OnCell()), NaN)
ϕ, p, p0 = RegisterHindsight.optimize!(ϕ0, ap, fixed, emoving; stepsize=0.1)
@test ratio(mismatch0(fixed, moving),1) > ratio(mismatch0(fixed, warp(moving, ϕ)), 1)

#2-dimensional images
inds = (200:300, 200:300)
img = map(Float64, testimage("cameraman"))
fixed = img[inds...]
moving = img[inds[1]-3,inds[2]-2]
gridsize = (3,3)
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1:ndims(fixed)...))
ap = AffinePenalty{Float64,ndims(fixed)}(knots, λ)
u0 = zeros(2, gridsize...)
ϕ0 = GridDeformation(u0, knots)
test_hindsight(fixed, moving, ϕ0, ap)


emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat())), OnCell()), NaN)
ϕ, p, p0 = RegisterHindsight.optimize!(ϕ0, ap, fixed, emoving; stepsize=0.1)
@test ratio(mismatch0(fixed, moving),1) > ratio(mismatch0(fixed, warp(moving, ϕ)), 1)
for i in eachindex(ϕ.u)
    u = ϕ.u[i]
    @test round(Int, u[1]) == 3
    @test round(Int, u[2]) == 2
end
