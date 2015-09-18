import RegisterCore
using Base.Test

# # MismatchData
# nums = reshape(map(Float32, 1:36), 3, 3, 2, 2)
# denoms = ones(3, 3, 2, 2)
# mmd = RegisterCore.MismatchData(nums, denoms)
# @test eltype(mmd) == Float32
# savedir = joinpath(tempdir(), ENV["USER"])
# if !isdir(savedir)
#     mkdir(savedir)
# end
# RegisterCore.save(joinpath(savedir, "test"), mmd)
# mmd2 = RegisterCore.MismatchData(joinpath(savedir, "test.mismatch"))
# for n in fieldnames(RegisterCore.MismatchData)
#     if n != :data
#         @test getfield(mmd2, n) == getfield(mmd, n)
#     end
# end
# @test squeeze(mmd2.data, 6) == mmd.data  # eliminate the temporal dimension

# Finding the location of the minimum
num = [5,4,3,4.5,7].*[2,1,1.5,2,3]'
num[1,5] = -1  # on the edge, so it shouldn't be selected
denom = ones(5,5)
mma = RegisterCore.MismatchArray(num,denom)
@test RegisterCore.indmin_mismatch(mma, 0) == CartesianIndex((0,-1))
denom = reshape(float(1:25), 5, 5)
mma = RegisterCore.MismatchArray(num,denom)
@test RegisterCore.indmin_mismatch(mma, 0) == CartesianIndex((0,1))
