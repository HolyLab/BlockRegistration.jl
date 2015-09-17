import RegisterCore
using Base.Test

# Block
A = reshape(1:32, 2, 2, 2, 2, 2)
B = RegisterCore.Block(A, 1, 1, 1)
@test size(B) == (2,2)
@test B == [1 3; 2 4]
B[3] = -1
@test A[3] == -1
B[2,2] = -5
@test A[4] == -5
@test_throws BoundsError B[3,3]
B = RegisterCore.Block(A, 1, 2, 1)
@test B == A[:,:,1,2,1]
@test B[3] == A[1,2,1,2,1]
B = RegisterCore.getblock(A, 2, 1, 2)
@test B == A[:,:,2,1,2]
Ac = Array(Vector{Int}, 2, 2)
Ac[1,1] = [1]; Ac[1,2] = [3]
Ac[2,1] = [2]; Ac[2,2] = [4]
# Ac = Array{Int,1}[[1] [3];
#                   [2] [4]]
@test RegisterCore.getblock(Ac, 2, 1) == [2]
A = reshape(1:36, 3, 3, 2, 2)
@test RegisterCore.gridsize(A) == (2,2)

# MismatchData
nums = reshape(map(Float32, 1:36), 3, 3, 2, 2)
denoms = ones(3, 3, 2, 2)
mmd = RegisterCore.MismatchData(nums, denoms)
@test eltype(mmd) == Float32
savedir = joinpath(tempdir(), ENV["USER"])
if !isdir(savedir)
    mkdir(savedir)
end
RegisterCore.save(joinpath(savedir, "test"), mmd)
mmd2 = RegisterCore.MismatchData(joinpath(savedir, "test.mismatch"))
for n in fieldnames(RegisterCore.MismatchData)
    if n != :data
        @test getfield(mmd2, n) == getfield(mmd, n)
    end
end
@test squeeze(mmd2.data, 6) == mmd.data  # eliminate the temporal dimension

# Finding the location of the minimum
num = [5,4,3,4.5].*[2,1,1.5,2,3]'
num[1,5] = -1  # on the edge, so it shouldn't be selected
denom = ones(4,5)
@test RegisterCore.indminmismatch(num, denom, 0) == 7
denom = reshape(float(1:20), 4, 5)
@test RegisterCore.indminmismatch(num, denom, 0) == 15
@test RegisterCore.ind2disp((3,5), 8) == [0,0]
@test RegisterCore.ind2disp((3,5), 9) == [1,0]
@test RegisterCore.ind2disp((3,5), 5) == [0,-1]
