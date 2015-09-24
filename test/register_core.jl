import RegisterCore
using Base.Test

# Finding the location of the minimum
num = [5,4,3,4.5,7].*[2,1,1.5,2,3]'
num[1,5] = -1  # on the edge, so it shouldn't be selected
denom = ones(5,5)
mma = RegisterCore.MismatchArray(num,denom)
@test RegisterCore.indmin_mismatch(mma, 0) == CartesianIndex((0,-1))
denom = reshape(float(1:25), 5, 5)
mma = RegisterCore.MismatchArray(num,denom)
@test RegisterCore.indmin_mismatch(mma, 0) == CartesianIndex((0,1))
