import RegisterMismatch
using Base.Test
using Compat

RMlist = (RegisterMismatch,)
mdutils = nothing
devlist = nothing
havecuda = isdefined(Main, :use_cuda) ? Main.use_cuda : !isempty(Libdl.find_library(["libcudart", "cudart"], ["/usr/local/cuda"]))
if havecuda
    using CUDArt
    import RegisterMismatchCuda
    RMlist = (RegisterMismatch,RegisterMismatchCuda)
    devlist = devices(dev->capability(dev)[1] >= 2, nmax=1)
    mdutils = [CuModule()]
    CUDArt.init!(mdutils, devlist)
    RegisterMismatchCuda.init(devlist)
end

const accuracy = 1e-6
for imsz in ((7,10), (6,5))
    for maxshift in ([4,3],[3,2])
        Apad = Images.padarray(reshape(1:prod(imsz), imsz[1], imsz[2]), maxshift, maxshift, "value", 0)
        Bpad = Images.padarray(rand(1:20, imsz[1], imsz[2]), maxshift, maxshift, "value", 0)
        for RM in RMlist
            # intensity normalization
            num, denom = RM.mismatch(Apad, Bpad, maxshift)
            mm = Array(Float64, 2maxshift.+1...)
            for j = -maxshift[2]:maxshift[2], i = -maxshift[1]:maxshift[1]
                Bshift = circshift(Bpad,-[i,j])
                df = Apad-Bshift
                mm[i+maxshift[1]+1,j+maxshift[2]+1] = sum(df.^2)
            end
#            @test_approx_eq mm num
#            @test_approx_eq fill(sum(Apad.^2)+sum(Bpad.^2),size(denom)) denom
            nrm = sum(Apad.^2)+sum(Bpad.^2)
            @test_approx_eq_eps mm num accuracy*nrm
            @test_approx_eq_eps fill(nrm,size(denom)) denom accuracy*nrm
            # pixel normalization
            num, denom = RM.mismatch(Apad, Bpad, maxshift, normalization=:pixels)
            n = Vector{Int}[size(Apad,i).-abs(-maxshift[i]:maxshift[i]) for i = 1:2]
            @test_approx_eq_eps denom n[1].*n[2]' accuracy*maximum(denom)
        end
    end
end

C = rand(7,9)
D = rand(7,9)
num, denom = RegisterMismatch.mismatch(C, D, (3,3))
n, d = RegisterMismatch.mismatch0(C, D)
@test_approx_eq num[4,4] n
@test_approx_eq denom[4,4] d
num, denom = RegisterMismatch.mismatch(C, D, (3,3), normalization=:pixels)
n, d = RegisterMismatch.mismatch0(C, D, normalization=:pixels)
@test_approx_eq num[4,4] n
@test_approx_eq denom[4,4] d
nums, denoms = RegisterMismatch.mismatch_blocks(C, D, (2,2), (3,2))
@test_approx_eq [RegisterMismatch.mismatch0(C, D)...] [RegisterMismatch.mismatchcenter(nums, denoms)...]

# Now do it for block mismatch
# A key property we're testing here is that
#     sum(nums) == num
# where num is equivalent to what would be computed globally (using mismatch)
for imsz in ((15,16), (14,17))
    for maxshift in ([4,3],[3,2])
        for gridsize in ([2,1], [2,3],[2,2],[1,3])
            Apad = Images.padarray(reshape(1:prod(imsz), imsz[1], imsz[2]), maxshift, maxshift, "value", 0)
            Bpad = Images.padarray(rand(1:20, imsz[1], imsz[2]), maxshift, maxshift, "value", 0)
            for RM in RMlist
                # intensity normalization
                nums, denoms = RM.mismatch_blocks(Apad, Bpad, gridsize, maxshift, display=false)
                num = sum(nums)
                denom = sum(denoms)
                mm = Array(Float64, 2maxshift.+1...)
                for j = -maxshift[2]:maxshift[2], i = -maxshift[1]:maxshift[1]
                    Bshift = circshift(Bpad,-[i,j])
                    df = Apad-Bshift
                    mm[i+maxshift[1]+1,j+maxshift[2]+1] = sum(df.^2)
                end
                nrm = sum(Apad.^2)+sum(Bpad.^2)
                @test_approx_eq_eps mm num accuracy*nrm
                @test_approx_eq_eps fill(nrm,size(denom)) denom accuracy*nrm
                # pixel normalization
                nums, denoms = RM.mismatch_blocks(Apad, Bpad, gridsize, maxshift, normalization=:pixels, display=false)
                denom = sum(denoms)
                n = Vector{Int}[size(Apad,i).-abs(-maxshift[i]:maxshift[i]) for i = 1:2]
                @test_approx_eq_eps denom n[1].*n[2]' accuracy*maximum(denom)
            end
        end
    end
end

for RM in RMlist
    # Test 3d similarly
    Apad = Images.padarray(reshape(1:80*6, 10, 8, 6), [4,3,2], [4,3,2], "value", 0)
    Bpad = Images.padarray(rand(1:80*6, 10, 8, 6), [4,3,2], [4,3,2], "value", 0)
    num, denom = RM.mismatch(Apad, Bpad, [4,3,2])
    mm = Array(Float64, 9, 7, 5)
    for k=-2:2, j = -3:3, i = -4:4
        Bshift = circshift(Bpad,-[i,j,k])
        df = Apad-Bshift
        mm[i+5,j+4,k+3] = sum(df.^2)
    end
    nrm = sum(Apad.^2)+sum(Bpad.^2)
    @test_approx_eq_eps mm num accuracy*nrm
    @test_approx_eq_eps fill(nrm,size(denom)) denom accuracy*nrm

    nums, denoms = RM.mismatch_blocks(Apad, Bpad, (2,3,2),[4,3,2], display=false)
    num = sum(nums)
    denom = sum(denoms)
    @test_approx_eq_eps mm num accuracy*nrm
    @test_approx_eq_eps fill(sum(Apad.^2)+sum(Bpad.^2),size(denom)) denom accuracy*nrm
end


### Test the utility functions that aren't implicitly tested by the above
# These only need to be tested once, since they come from RegisterMismatchCommon
RM = RegisterMismatch
A = fill(7, 8, 4)
Ahp = RM.highpass(A, (1,1), astype=Float32)
@test eltype(Ahp) == Float32
@test maximum(abs(Ahp)) < 100*eps(Float32)
Ahp = RM.highpass(A, (1.2,Inf))
@test A == Ahp

num = float([1 2; 3 4])
denom = [1 1e-6; 2 1]
RM.truncatenoise!(num, denom, 1e-5)
@test num == [1 0; 3 4]
@test denom == [1 0; 2 1]

outer = reshape(1:120, 12, 10)
A = outer[1:10,2:9]
B = outer[2:12,1:8]
maxshift = (3,11)
@test_throws ErrorException RM.register_translate(A, B, maxshift)
B = outer[3:12,1:8]
@test RM.register_translate(A, B, maxshift) == [-2,1]

if havecuda
    RegisterMismatchCuda.close()
    CUDArt.close!(mdutils, devlist)
end
