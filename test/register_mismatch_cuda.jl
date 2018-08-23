using Base.Test
import CUDArt
import BlockRegistration, RegisterMismatchCuda
using RegisterCore

function run_components(f, A)
    G1 = CUDArt.CudaPitchedArray(A)
    G0 = CUDArt.CudaPitchedArray(eltype(A), size(A))
    G2 = CUDArt.CudaPitchedArray(eltype(A), size(A))
    args = (pointer(G1), pointer(G2), pointer(G0), size(G1,1), size(G1,2), size(G1,3), CUDArt.pitchel(G1))
    argtypes = ((typeof(x) for x in args)...)
    CUDAdrv.cudacall(f, 1, (4,4), argtypes, args...)
    A0, A1, A2 = CUDArt.to_host(G0), CUDArt.to_host(G1), CUDArt.to_host(G2)
end

img = rand(map(UInt8,0:255), 256, 256)
rng = Any[1:240, 10:250]
fixed = map(Float32, img[rng...])
moving = map(Float32, img[rng[1]+13, rng[2]-8])

CUDArt.devices(dev->CUDArt.capability(dev)[1] >= 2, nmax=1) do devlist
#     CuModule("../register_mismatch_cuda.ptx") do md
    RegisterMismatchCuda.init(devlist)
    try
        f = RegisterMismatchCuda.ptxdict[(devlist[1], "components_func", Float64)]
        CUDArt.device(devlist[1])
        A = [1 2; NaN 4]
        B = [NaN NaN; 5 NaN]
        A0, A1, A2 = run_components(f, A)
        @test A0 == .!isnan.(A)
        @test A1 == [1 2; 0 4]
        @test A2 == [1 4; 0 16]
        B0, B1, B2 = run_components(f, B)
        @test B0 == .!isnan.(B)
        @test B1 == [0 0; 5 0]
        @test B2 == [0 0; 25 0]
        A = zeros(5,5)
        A[3,3] = 3
        B = zeros(5,5)
        B[4,5] = 3
        maxshift = (2,3)
        mm = RegisterMismatchCuda.mismatch(A, A, maxshift)
        num, denom = RegisterMismatchCuda.separate(mm)
        RegisterMismatchCuda.truncatenoise!(mm, 0.01)
        @test indmin_mismatch(mm, 0.01) == CartesianIndex((0,0))
        mm = RegisterMismatchCuda.mismatch(A, B, maxshift)
        RegisterMismatchCuda.truncatenoise!(mm, 0.01)
        @test indmin_mismatch(mm, 0.01) == CartesianIndex((1,2))

        # Testing on more complex objects
        maxshift = (20, 20)
        mm = RegisterMismatchCuda.mismatch(fixed, moving, maxshift)
        @test indmin_mismatch(mm, 0.01) == CartesianIndex((-13,8))
    finally
        RegisterMismatchCuda.close()
    end
end
