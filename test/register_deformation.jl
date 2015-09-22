import RegisterDeformation
using AffineTransforms, Interpolations, ForwardDiff
using Base.Test

randrange(xmin, xmax) = (xmax-xmin)*rand() + xmin

## Deformations
for knots in ((linspace(1,15,5),),
              (linspace(1,15,5), linspace(1,8,4)),
              (linspace(1,15,3), linspace(1,8,4), linspace(1,7,2)),
              (linspace(1,15,5), linspace(1,8,4), linspace(1,7,6)))
    sz = map(length, knots)
    rng = map(z->convert(Int, maximum(z)), knots)
    # Zero deformation
    u = zeros(length(knots), sz...)
    ϕ = interpolate(RegisterDeformation.GridDeformation(u, knots))
    for i = 1:10
        x = map(z->rand(1:z), rng)
        @test_approx_eq ϕ[x...] [x...]
        if all(x->x>2, sz)
            y = map(z->rand(2:z-1)+rand()-0.5, rng)
            @test_approx_eq ϕ[y...] [y...]
        end
    end
    # Constant shift
    dx = randn(length(knots))
    u = zeros(length(knots), sz...).+dx
    ϕ = interpolate(RegisterDeformation.GridDeformation(u, knots))
    for i = 1:10
        x = map(z->rand(1:z), rng)
        @test_approx_eq ϕ[x...] [x...]+dx
        if all(x->x>2, sz)
            y = map(z->rand(2:z-1)+rand()-0.5, rng)
            @test_approx_eq ϕ[y...] [y...]+dx
        end
    end
    # "Biquadratic"
    if all(x->x>2, sz)
        N = length(knots)
        u_is = Array(Any, N)
        fs = Array(Any, N)
        R = CartesianRange(sz)
        for i = 1:N
            let cntr = Float64[rand(1:z-1)+rand() for z in rng], q = rand(N)
                f = x -> dot(q, (x-cntr).^2)
                ui = zeros(sz)
                for I in R
                    ui[I] = f(Float64[knots[d][I[d]] for d = 1:N])
                end
                u_is[i] = reshape(ui, 1, sz...)
                fs[i] = f
            end
        end
        u = cat(1, u_is...)
        ϕ = interpolate!(RegisterDeformation.GridDeformation(copy(u), knots), InPlaceQ)
        for i = 1:10
            y = Float64[randrange((knot[2]+knot[1])/2, (knot[end]+knot[end-1])/2) for knot in knots]
            dx = Float64[fs[d](y) for d=1:N]
            try
                @test_approx_eq ϕ[y...] y + Float64[fs[d](y) for d=1:N]
            catch err
                @show y knots
                @show [((knot[2]+knot[1])/2, (knot[end]+knot[end-1])/2) for knot in knots]
                @show Float64[fs[d](y) for d=1:N]
                rethrow(err)
            end
        end
    end
end


s = [3.3,-2.6]
gsize = (3,2)
A = tformtranslate(s)
ϕ = RegisterDeformation.tform2deformation(A, (500,480), gsize)
u = reinterpret(Float64, ϕ.u, tuple(2,gsize...))
@test all(u[1,:,:] .== s[1])
@test all(u[2,:,:] .== s[2])

## WarpedArray

p = (1:100)-5

dest = Array(Float32, 10)

# Simple translations in 1d
ϕ = RegisterDeformation.GridDeformation([0.0,0.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - p[1:10])) < 10*eps(maximum(abs(dest)))

ϕ = RegisterDeformation.GridDeformation([1.0,1.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - p[2:11])) < 10*eps(maximum(abs(dest)))

ϕ = RegisterDeformation.GridDeformation([-2.0,-2.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - [fill(NaN32,2);p[1:8]])) < 10*eps(maximum(abs(dest)))

RegisterDeformation.getindex!(dest, q, 3:12)
@assert maximum(abs(dest - p[1:10])) < 10*eps(maximum(abs(dest)))

ϕ = RegisterDeformation.GridDeformation([2.0,2.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - p[3:12])) < 10*eps(maximum(abs(dest)))

ϕ = RegisterDeformation.GridDeformation([5.0,5.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - p[6:15])) < 10*eps(maximum(abs(dest)))

# SubArray (test whether we can go beyond edges)
psub = sub(collect(p), 3:20)
ϕ = RegisterDeformation.GridDeformation([0.0,0.0]', size(p))
q = RegisterDeformation.WarpedArray(psub, ϕ);
RegisterDeformation.getindex!(dest, q, -1:8)
@assert maximum(abs(dest - p[1:10])) < 10*eps(maximum(abs(dest)))
any(isnan, dest) && warn("Some dest are NaN, not yet sure whether this is a problem")

# Stretches
u = [0.0,5.0,10.0]
ϕ = interpolate(RegisterDeformation.GridDeformation(u', size(p)), Line)
q = RegisterDeformation.WarpedArray(p, ϕ)
RegisterDeformation.getindex!(dest, q, 1:10)
@assert abs(dest[1] - p[1]) < sqrt(eps(1.0f0))
RegisterDeformation.getindex!(dest, q, 86:95)
@assert isnan(dest) == [falses(5);trues(5)]  # fixme
dest2 = RegisterDeformation.getindex!(zeros(Float32, 100), q, 1:100)
@assert all(abs(diff(dest2)[26:74] .- ((u[3]-u[1])/99+1)) .< sqrt(eps(1.0f0)))

#2d
p = reshape(1:120, 10, 12)
u1 = [2.0 2.0; 2.0 2.0]
u2 = [-1.0 -1.0; -1.0 -1.0]
ϕ = RegisterDeformation.GridDeformation((u1,u2), size(p))
q = RegisterDeformation.WarpedArray(p, ϕ)
dest = zeros(size(p))
rng = (1:size(p,1),1:size(p,2))
RegisterDeformation.getindex!(dest, q, rng...)
@assert maximum(abs(dest[1:7,2:end] - p[3:9,1:end-1])) < 10*eps(maximum(dest))

# Composition
# Test that two rotations approximately compose to another rotation
gridsize = (49,47)
imgsize = (200,200)
ϕ1 = RegisterDeformation.tform2deformation(tformrotate( pi/180), imgsize, gridsize)
ϕ2 = RegisterDeformation.tform2deformation(tformrotate(2pi/180), imgsize, gridsize)
ϕi = interpolate(ϕ1)
ϕc = ϕi(ϕi)
uc = reinterpret(Float64, ϕc.u, (2, gridsize...))
u2 = reinterpret(Float64, ϕ2.u, (2, gridsize...))
@test_approx_eq_eps uc u2 0.02

# Test gradient computation
gridsize = (11,11)
u1 = randn(2, gridsize...)
u2 = randn(2, gridsize...)
imsz = (100,101)
ϕ1 = interpolate(RegisterDeformation.GridDeformation(u1, imsz))
ϕ2 = interpolate(RegisterDeformation.GridDeformation(u2, imsz))
ϕ, g = RegisterDeformation.compose(ϕ1, ϕ2)
# A function that ForwardDiff will be happy with
function compose_u(u1, u2, shp, imsz)
    ϕ1 = interpolate(RegisterDeformation.GridDeformation(u1, imsz))
    ϕ2 = interpolate(RegisterDeformation.GridDeformation(reshape(u2, shp), imsz))
    ϕ = ϕ1(ϕ2)
    ret = Array(eltype(eltype(ϕ.u)), prod(shp))
    i = 0
    for v in ϕ.u
        for d = 1:length(v)
            ret[i+=1] = v[d]
        end
    end
    ret
end
Base.real(x::ForwardDiff.GradientNumber) = x.value
u2vec = vec(u2)
j = ForwardDiff.jacobian(u2vec -> compose_u(u1, u2vec, size(u1), imsz))
gj = j(u2vec)
function compare_g(g, gj, gridsize)
    for j = 1:gridsize[2], i = 1:gridsize[1]
        indx = sub2ind(gridsize, i, j)
        rng = 2*(indx-1)+1:2indx
        @test_approx_eq g[i,j] gj[rng, rng]
        for k = 1:2*prod(gridsize)
            if !in(k, rng)
                @assert abs(gj[k,rng[1]]) < 1e-15
                @assert abs(gj[k,rng[2]]) < 1e-15
            end
        end
    end
    nothing
end
compare_g(g, gj, gridsize)

# Test identity-composition
_, g = RegisterDeformation.compose(identity, ϕ2)
fill!(gj, 0)
for i = 1:prod(gridsize)
    rng = 2*(i-1)+1:2i
    gj[rng,rng] = eye(2,2)
end
compare_g(g, gj, gridsize)
