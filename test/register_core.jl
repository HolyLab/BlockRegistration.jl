import BlockRegistration, RegisterCore
using CenterIndexedArrays
using Base.Test

nd = RegisterCore.NumDenom(3.5,10)
@test RegisterCore.ratio(nd, 5) == 3.5/10
@test isequal(RegisterCore.ratio(nd, 20), NaN)
@test convert(RegisterCore.NumDenom{Float32}, nd) == RegisterCore.NumDenom(3.5f0,10)
nd = RegisterCore.NumDenom(3.5f0,10)
@test isa(RegisterCore.ratio(nd, 5), Float32)
@test isa(RegisterCore.ratio(nd, 20), Float32)

numer, denom = rand(3,3), rand(3,3)+0.5
mm = RegisterCore.MismatchArray(numer, denom)
r = CenterIndexedArray(numer./denom)
@test RegisterCore.ratio(mm, 0.25) == r
@test RegisterCore.ratio(r, 0.25) == r

# Finding the location of the minimum
numer = [5,4,3,4.5,7].*[2,1,1.5,2,3]'
numer[1,5] = -1  # on the edge, so it shouldn't be selected
denom = ones(5,5)
mma = RegisterCore.MismatchArray(numer,denom)
@test RegisterCore.indmin_mismatch(mma, 0) == CartesianIndex((0,-1))
denom = reshape(float(1:25), 5, 5)
mma = RegisterCore.MismatchArray(numer,denom)
@test RegisterCore.indmin_mismatch(mma, 0) == CartesianIndex((0,1))

# SubArray padding and trimming
A = reshape(1:125, 5, 5, 5)
S = view(A, 2:4, 1:3, 2)
Spad = RegisterCore.paddedview(S)
@test Spad == A[:,:,2]
S2 = RegisterCore.trimmedview(A[:,:,2], S)
@test S2 == S
S = view(A, 2:4, 2, 1:3)
Spad = RegisterCore.paddedview(S)
Aslice = view(A, :, 2, :)
@test Spad == Aslice
S2 = RegisterCore.trimmedview(copy(Aslice), S)
@test S2 == S
S2 = RegisterCore.trimmedview(Aslice, S)
@test S2 == S
