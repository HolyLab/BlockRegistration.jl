import RegisterDeformation
using AffineTransforms, Interpolations
using Base.Test

s = [3.3,-2.6]
gsize = (3,2)
A = tformtranslate(s)
u = RegisterDeformation.tform2u(A, (500,480), gsize)
@test size(u) == tuple(2,gsize...)
@test all(u[1,:,:] .== s[1])
@test all(u[2,:,:] .== s[2])

## WarpedArray

p = (1:100)-5

dest = Array(Float32, 10)

# Simple translations in 1d
u = [0.0,0.0]
q = RegisterDeformation.WarpedArray(p, (u,));
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - p[1:10])) < 10*eps(maximum(abs(dest)))

u = [1.0,1.0]
q = RegisterDeformation.WarpedArray(p, (u,));
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - p[2:11])) < 10*eps(maximum(abs(dest)))

u = [-2.0,-2.0]
q = RegisterDeformation.WarpedArray(p, (u,));
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - [fill(NaN32,2);p[1:8]])) < 10*eps(maximum(abs(dest)))

u = [-2.0,-2.0]
q = RegisterDeformation.WarpedArray(p, (u,));
RegisterDeformation.getindex!(dest, q, 3:12)
@assert maximum(abs(dest - p[1:10])) < 10*eps(maximum(abs(dest)))

u = [2.0,2.0]
q = RegisterDeformation.WarpedArray(p, (u,));
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - p[3:12])) < 10*eps(maximum(abs(dest)))

u = [5.0,5.0,5.0]
q = RegisterDeformation.WarpedArray(p, (u,));
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs(dest - p[6:15])) < 10*eps(maximum(abs(dest)))

# SubArray (test whether we can go beyond edges)
u = [0.0,0.0]
psub = sub(collect(p), 3:20)
q = RegisterDeformation.WarpedArray(psub, (u,));
RegisterDeformation.getindex!(dest, q, -1:8)
@assert maximum(abs(dest - p[1:10])) < 10*eps(maximum(abs(dest)))

# Stretches
u = [0.0,5.0,10.0]
q = RegisterDeformation.WarpedArray(p, (interpolate(u, BSpline{Quadratic{Line}}, OnCell),))
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
q = RegisterDeformation.WarpedArray(p, (u1,u2));
dest = zeros(size(p))
rng = (1:size(p,1),1:size(p,2))
RegisterDeformation.getindex!(dest, q, rng...)
@assert maximum(abs(dest[1:7,2:end] - p[3:9,1:end-1])) < 10*eps(maximum(dest))
