import BlockRegistration, RegisterCore
using Base.Test

nd = RegisterCore.NumDenom(3.5,10)
@test RegisterCore.ratio(nd, 5) == 3.5/10
@test isequal(RegisterCore.ratio(nd, 20), NaN)
@test convert(RegisterCore.NumDenom{Float32}, nd) == RegisterCore.NumDenom(3.5f0,10)
nd = RegisterCore.NumDenom(3.5f0,10)
@test isa(RegisterCore.ratio(nd, 5), Float32)
@test isa(RegisterCore.ratio(nd, 20), Float32)

num, denom = rand(3,3), rand(3,3)+0.5
mm = RegisterCore.MismatchArray(num, denom)
r = CenterIndexedArray(num./denom)
@test RegisterCore.ratio(mm, 0.25) == r
@test RegisterCore.ratio(r, 0.25) == r

# Finding the location of the minimum
num = [5,4,3,4.5,7].*[2,1,1.5,2,3]'
num[1,5] = -1  # on the edge, so it shouldn't be selected
denom = ones(5,5)
mma = RegisterCore.MismatchArray(num,denom)
@test RegisterCore.indmin_mismatch(mma, 0) == CartesianIndex((0,-1))
denom = reshape(float(1:25), 5, 5)
mma = RegisterCore.MismatchArray(num,denom)
@test RegisterCore.indmin_mismatch(mma, 0) == CartesianIndex((0,1))
