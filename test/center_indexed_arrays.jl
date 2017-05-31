# import BlockRegistration
using CenterIndexedArrays, Base.Test
using Compat

CenterIndexedArray(Float32, 3, 5)
@test_throws ErrorException CenterIndexedArray(Float32, 4, 5)

dat = rand(3,5)
A = CenterIndexedArray(dat)
@test size(A) == size(dat)
@test length(A) == length(dat)
@test ndims(A) == 2
@test eltype(A) == eltype(dat)

@test A[0,0] == dat[2,3]
k = 0
for j = -2:2, i = -1:1
    k += 1
    @test A[i,j] == dat[k]
end
@test_throws BoundsError A[3,5]
@test A[:,-1:1].data == dat[:,2:4]
@test A[-1:1,:].data == dat[:,:]
@test_throws ErrorException A[:,-2:0]
k = 0
for j = -2:2, i = -1:1
    A[i,j] = (k+=1)
end
@test dat == reshape(1:15, 3, 5)
@test_throws BoundsError A[3,5] = 15

rand!(dat)
iall = (-1:1).*ones(Int, 5)'
jall = ones(Int,3).*(-2:2)'
k = 0
for I in eachindex(A)
    k += 1
    @test I[1] == iall[k]
    @test I[2] == jall[k]
end

io = IOBuffer()
show(io, MIME("text/plain"), A)
str = String(take!(io))
@test isempty(search(str, "undef"))

# Iteration
for (a,d) in zip(A, dat)
    @test a == d
end

# Standard julia operations
B = copy(A)

@test B.data == dat
@test B == A
@test isequal(B, A)

@test vec(A) == vec(dat)

@test minimum(A) == minimum(dat)
@test maximum(A) == maximum(dat)
# @test minimum(A,1) == minimum(dat,1)
# @test maximum(A,2) == maximum(dat,2)
@test minimum(A,1) == CenterIndexedArray(minimum(dat,1))
@test maximum(A,2) == CenterIndexedArray(maximum(dat,2))

amin, iamin = findmin(A)
dmin, idmin = findmin(dat)
@test amin == dmin
@test A[iamin] == amin
@test amin == dat[idmin]

amax, iamax = findmax(A)
dmax, idmax = findmax(dat)
@test amax == dmax
@test A[iamax] == amax
@test amax == dat[idmax]

fill!(A, 2)
@test all(x->x==2, A)

i, j = findn(A)
@test vec(i) == vec(iall)
@test vec(j) == vec(jall)

rand!(dat)

# @test cat(1, A, dat) == cat(1, dat, dat)
# @test cat(2, A, dat) == cat(2, dat, dat)

@test permutedims(A, (2,1)) == CenterIndexedArray(permutedims(dat, (2,1)))
# @test ipermutedims(A, (2,1)) == CenterIndexedArray(ipermutedims(dat, (2,1)))

@test cumsum(A, 1) == CenterIndexedArray(cumsum(dat, 1))
@test cumsum(A, 2) == CenterIndexedArray(cumsum(dat, 2))

@test mapslices(v->sort(v), A, 1) == CenterIndexedArray(mapslices(v->sort(v), dat, 1))
@test mapslices(v->sort(v), A, 2) == CenterIndexedArray(mapslices(v->sort(v), dat, 2))

@test flipdim(A, 1) == CenterIndexedArray(flipdim(dat, 1))
@test flipdim(A, 2) == CenterIndexedArray(flipdim(dat, 2))

@test A + 1 == CenterIndexedArray(dat + 1)
@test 2*A == CenterIndexedArray(2*dat)
@test A+A == CenterIndexedArray(dat+dat)
