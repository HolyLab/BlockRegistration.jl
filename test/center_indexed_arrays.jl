using CenterIndexedArrays, Base.Test

CenterIndexedArray(Float32, 3, 5)
@test_throws ErrorException CenterIndexedArray(Float32, 4, 5)

data = rand(3,5)
A = CenterIndexedArray(data)
@test size(A) == size(data)
@test length(A) == length(data)
@test ndims(A) == 2
@test eltype(A) == eltype(data)

@test A[0,0] == data[2,3]
k = 0
for j = -2:2, i = -1:1
    k += 1
    @test A[i,j] == data[k]
end
@test_throws BoundsError A[3,5]
k = 0
for j = -2:2, i = -1:1
    A[i,j] = (k+=1)
end
@test data == reshape(1:15, 3, 5)
@test_throws BoundsError A[3,5] = 15

rand!(data)
iall = (-1:1).*ones(Int, 5)'
jall = ones(Int,3).*(-2:2)'
k = 0
for I in eachindex(A)
    k += 1
    @test I[1] == iall[k]
    @test I[2] == jall[k]
end

io = IOBuffer()
writemime(io, MIME("text/plain"), A)
str = takebuf_string(io)
@test isempty(search(str, "undef"))

# Iteration
for (a,d) in zip(A, data)
    @test a == d
end

# Standard julia operations
B = copy(A)

@test B == data
@test B == A
@test isequal(B, data)
@test isequal(B, A)

@test vec(A) == vec(data)

@test minimum(A) == minimum(data)
@test minimum(A,1) == minimum(data,1)
@test maximum(A,2) == maximum(data,2)

amin, iamin = findmin(A)
dmin, idmin = findmin(data)
@test amin == dmin
@test A[iamin] == amin
@test amin == data[idmin]

amax, iamax = findmax(A)
dmax, idmax = findmax(data)
@test amax == dmax
@test A[iamax] == amax
@test amax == data[idmax]

fill!(A, 2)
@test all(x->x==2, A)

i, j = findn(A)
@test vec(i) == vec(iall)
@test vec(j) == vec(jall)

rand!(data)

@test cat(1, A, data) == cat(1, data, data)
@test cat(2, A, data) == cat(2, data, data)

@test permutedims(A, (2,1)) == permutedims(data, (2,1))
@test ipermutedims(A, (2,1)) == ipermutedims(data, (2,1))

@test cumsum(A, 1) == cumsum(data, 1)
@test cumsum(A, 2) == cumsum(data, 2)

@test mapslices(v->sort(v), A, 1) == mapslices(v->sort(v), data, 1)
@test mapslices(v->sort(v), A, 2) == mapslices(v->sort(v), data, 2)

@test flipdim(A, 1) == flipdim(data, 1)
@test flipdim(A, 2) == flipdim(data, 2)

@test A + 1 == data + 1
@test 2*A == 2*data
@test A+A == data+data
