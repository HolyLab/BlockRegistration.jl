using Interpolations, BlockRegistration, RegisterMismatch
using Interpolations: sqr, SimpleRatio, BSplineInterpolation, DimSpec, Degree
import RegisterPixelwise
using DualNumbers, FixedSizeArrays
using Base.Test

#add jitter in sampling location, simulating inconsistencies in piezo position when using OCPI under certain conditions
function jitter{T}(img::Array{T,1}, npix::Float64)
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

#the same as penalty_pixelwise_data, but not generated (not working right now)
#function testpenalty_data{T,N,T1,T2,A}(ϕ::RegisterPixelwise.InterpolatingDeformation{T,N,A},
#                                                            fixed::AbstractArray{T1,N},
#                                                            moving::AbstractInterpolation{T2,N})
#    IT = Interpolations.itptype(A)
#    knots = ϕ.knots
#    steps = map(step, knots)
#    offsets = map(first, knots)
#    valid = 0
#    mm = 0.0
#    for I in CartesianRange(indices(fixed))
#        fval = fixed[I]
#        if isfinite(fval)
#            uindexes = RegisterPixelwise.scaledindexes(IT, N)
##            uindexes = [eval(x) for x in uindexes]
#            u = ϕ.u.itp[eval(uindexes[1])]
#            ϕxindexes = [I[d] + u[d] for d = 1:N]
#            mval = moving[ϕxindexes]
#            if isfinite(mval)
#                valid += 1
#                diff = float64(fval)-float64(mval)
#                mm += diff^2
#            end
#        end
#    end
#    mm/valid
#end

#fill each real-valued component of g with the respective gradient entry
#temporarily sets the epsilon components of g to compute the gradient
function dualgrad_data!(g, ϕ, fixed, moving)
    ur = RegisterDeformation.convert_from_fixed(ϕ.u.itp.coefs)
    gr = RegisterDeformation.convert_from_fixed(g)
    nd = size(ur, 1)
    for i in eachindex(ϕ.u.itp.coefs)
        for j = 1:nd
            temp = ur[j, i]
            ur[j, i] = dual(value(temp), 1.0)
            gr[j, i] = epsilon(RegisterPixelwise.penalty_pixelwise_data(ϕ, fixed, moving))
            ur[j, i] = temp
        end
    end
end

function dualgrad_reg!(g, ap, ϕ)
    ur = RegisterDeformation.convert_from_fixed(ϕ.u.itp.coefs)
    gr = RegisterDeformation.convert_from_fixed(g)
    nd = size(ur, 1)
    for i in eachindex(ϕ.u.itp.coefs)
        for j = 1:nd
            temp = ur[j, i]
            ur[j, i] = dual(value(temp), 1.0)
            gr[j, i] = epsilon(RegisterPixelwise.penalty_pixelwise_reg(ap, ϕ))
            ur[j, i] = temp
        end
    end
end


function test_pixelwise(fixed, moving, ϕ0, ap)
    print("Beginning new test run\n")
    u0 = RegisterDeformation.convert_from_fixed(ϕ0.u)
    ϕ = interpolate!(deepcopy(ϕ0))  # don't "destroy" u0
    g_data = similar(ϕ.u.itp.coefs)

    emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat())), OnCell()), NaN)

    #compare penalty functions with various levels of optimization. TODO: add another simpler method
    pdata1 = RegisterPixelwise.penalty_pixelwise_data(ϕ, fixed, emoving)
    preg1 = RegisterPixelwise.penalty_pixelwise_reg(ap, ϕ)
    ptotal = RegisterPixelwise.penalty_pixelwise(ϕ, ap, fixed, emoving)
    @test ptotal == pdata1 + preg1
    pdata2 = RegisterPixelwise.penalty_pixelwise_data!(g_data, ϕ, fixed, emoving) #fully optimized version
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
    RegisterPixelwise.penalty_pixelwise!(g_total, ϕ, ap, fixed, emoving)
    RegisterPixelwise.penalty_pixelwise_reg!(g_reg, ap, ϕ)
    @test g_total == g_data .+ g_reg

    #test affine penalty gradient
    dualgrad_reg!(g_reg_dual, ap, ϕ_dual)
    for i in eachindex(g_reg)
        @test g_reg[i] ≈ g_reg_dual[i]
    end
    print("Test run successful\n")
end

fixed = sin.(linspace(0,4π,40))
moving, z_def = jitter(fixed, 0.45);

λ = 1e-3
gridsize = (length(fixed),)
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1:ndims(fixed)...))
ap = AffinePenalty{Float64,ndims(fixed)}(knots, λ)
u0 = zeros(1, gridsize...)
ϕ0 = GridDeformation(u0, knots)

test_pixelwise(fixed, moving, ϕ0, ap)

u0 = rand(1, gridsize...)./10
ϕ0 = GridDeformation(u0, knots)
test_pixelwise(fixed, moving, ϕ0, ap)

u0 = zeros(1, gridsize...)
ϕ0 = GridDeformation(u0, knots)
emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat())), OnCell()), NaN)
ϕ, p, p0 = RegisterPixelwise.optimize_pixelwise!(ϕ0, ap, fixed, emoving; stepsize=0.1)
@test ratio(mismatch0(fixed, moving),1) > ratio(mismatch0(fixed, warp(moving, ϕ)), 1)
