using RegisterDeformation, TestImages, ImageCore, Unitful
using AxisArrays: Axis, AxisArray
using Random

Random.seed!(123)
nframes = 11
grid = (3, 3)
amplitude = 10
trim = 50

ref = testimage("camera")
pre  = amplitude * randn(2, grid...)
post = amplitude * randn(2, grid...)

img = Array{Gray{N0f8}}(undef, (size(ref) .- 2*trim)..., nframes)
nhalf = nframes ÷ 2
img[:,:,nhalf+1] = ref[trim+1:end-trim, trim+1:end-trim]

for i = 1:nhalf
    ϕ = GridDeformation(pre * (i/nhalf), axes(ref))
    mov = warp(ref, ϕ)
    img[:,:,nhalf+1-i] = mov[trim+1:end-trim, trim+1:end-trim]
end
for i = 1:nhalf
    ϕ = GridDeformation(post * (i/nhalf), axes(ref))
    mov = warp(ref, ϕ)
    img[:,:,nhalf+1+i] = mov[trim+1:end-trim, trim+1:end-trim]
end

imgps = 0.577*u"μm"
img = AxisArray(img, (:y, :x, :time), (imgps, imgps, 0.5*u"s"))
