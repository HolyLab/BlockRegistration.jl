using Interpolations, BlockRegistration
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
    for i in eachindex(ϕ.u.itp.coefs)
        cur_input = Array(ϕ.u.itp.coefs[i])
        cur_output = Array(g[i])
        for j = 1:length(cur_input)
            temp = cur_input[j]
            cur_input[j] = dual(value(temp), 1.0)
            ϕ.u.itp.coefs[i] = Vec(cur_input)
            cur_output[j] = epsilon(RegisterPixelwise.penalty_pixelwise_data(ϕ, fixed, moving))
            cur_input[j] = temp
        end
        g[i] = Vec(cur_output)
        ϕ.u.itp.coefs[i] = Vec(cur_input)
    end
end

function dualgrad_reg!(g, ap, ϕ)
    for i in eachindex(ϕ.u.itp.coefs)
        cur_input = Array(ϕ.u.itp.coefs[i])
        cur_output = Array(g[i])
        for j = 1:length(cur_input)
            temp = cur_input[j]
            cur_input[j] = dual(value(temp), 1.0)
            ϕ.u.itp.coefs[i] = Vec(cur_input)
            cur_output[j] = epsilon(RegisterPixelwise.penalty_pixelwise_reg(ap, ϕ))
            cur_input[j] = temp
        end
        g[i] = Vec(cur_output)
        ϕ.u.itp.coefs[i] = Vec(cur_input)
    end
end


function test_pixelwise(fixed, moving, ϕ0, ap)
    print("Beginning new test run\n")
    ϕ = interpolate!(ϕ0)
    g_empty = similar(ϕ.u.itp.coefs)
    copy!(g_empty, ϕ.u.itp.coefs)
    g_data = deepcopy(g_empty)

    emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat())), OnCell()), NaN)

    #compare penalty functions with various levels of optimization. TODO: add another simpler method
    pdata1 = RegisterPixelwise.penalty_pixelwise_data(ϕ, fixed, emoving)
    preg1 = RegisterPixelwise.penalty_pixelwise_reg(ap, ϕ)
    ptotal = RegisterPixelwise.penalty_pixelwise(ϕ, ap, fixed, emoving)
    @test ptotal == pdata1 + preg1
    pdata2 = RegisterPixelwise.penalty_pixelwise_data!(g_data, ϕ, fixed, emoving) #fully optimized version
    @test pdata1 == pdata2

    ϕ0_dual = GridDeformation(map(dual, u0), [knots[i][:] for i=1:length(knots)])
    ϕ_dual = interpolate!(ϕ0_dual)

    g_reg = deepcopy(g_empty)
    g_total = deepcopy(g_empty)

    g_dual_empty = similar(ϕ_dual.u.itp.coefs)
    copy!(g_dual_empty, ϕ_dual.u.itp.coefs)
    g_data_dual = deepcopy(g_dual_empty)
    g_reg_dual = deepcopy(g_dual_empty)

    #test data penalty gradient
    dualgrad_data!(g_data_dual, ϕ_dual, fixed, emoving)
    for i in eachindex(g_data)
        @test_approx_eq Array(g_data[i]) Array(g_data_dual[i])
    end

    #test that gradients sum properly
    RegisterPixelwise.penalty_pixelwise!(g_total, ϕ, ap, fixed, emoving)
    RegisterPixelwise.penalty_pixelwise_reg!(g_reg, ap, ϕ)
    @test g_total == g_data .+ g_reg

    #test affine penalty gradient
    dualgrad_reg!(g_reg_dual, ap, ϕ_dual)
    for i in eachindex(g_reg)
        @test_approx_eq Array(g_reg[i]) Array(g_reg_dual[i])
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

test_pixelwise(fixed, moving, ϕ0, ap) #passes

u0 = rand(1, gridsize...)./10
ϕ0 = GridDeformation(u0, knots)
test_pixelwise(fixed, moving, ϕ0, ap) #fails



