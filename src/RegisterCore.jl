__precompile__()

module RegisterCore

using Base.Cartesian: @nloops, @nref
using FileIO

import Base: +, -, *, /
import Base: eltype, getindex, ndims, pointer, setindex!, show, size
import Base: checksize, unsafe_getindex
import FileIO: save

export
    # types
    Block,
    MismatchData,
    NumDenom,
    # functions
#    append,
    blockeltype,
    blocksize,
    save,
    getblock,
    getindex!,
    gridsize,
    indminmismatch,
    ind2disp,
    ind2disp!,
    pack_nd
#    write_header

# For deformable registration, mismatch data is specified for blocks
# of the image.  This may have several different representations:
#   - (nums,denoms) tuple, where nums[i,j] is the numerator data for
#     block (i,j) ("Array-of-Arrays" representation)
#   - (nums,denoms) tuple, where nums[:,:,i,j] is the numerator data for
#     block (i,j) ("tiled array" representation)
#   - numsdenoms, where numsdenoms[:,:,i,j] contains an array of NumDenom
#     pairs (see below)

"""
`NumDenom{T}` is a 2-vector containing a `(num,denom)` pair. This
representation is efficient for `Interpolations.jl`, because it allows
interpolation to be performed on "both arrays" at once without
recomputing the interpolation coefficients.
"""
immutable NumDenom{T<:Number}
    num::T
    denom::T
end

(+)(p1::NumDenom, p2::NumDenom) = NumDenom(p1.num+p2.num, p1.denom+p2.denom)
(-)(p1::NumDenom, p2::NumDenom) = NumDenom(p1.num-p2.num, p1.denom-p2.denom)
(*)(n::Number, p::NumDenom) = NumDenom(n*p.num, n*p.denom)
(*)(p::NumDenom, n::Number) = n*p
(/)(p::NumDenom, n::Number) = NumDenom(p.num/n, p.denom/n)
Base.one{T}(::Type{NumDenom{T}}) = NumDenom(one(T),one(T))
Base.zero{T}(::Type{NumDenom{T}}) = NumDenom(zero(T),zero(T))
Base.promote_rule{T1,T2<:Number}(::Type{NumDenom{T1}}, ::Type{T2}) = NumDenom{promote_type(T1,T2)}

"""
`numdenom = pack_nd(nums, denoms)` packs the array-pair
`(nums,denoms)` into a single `Array{NumDenom}`.  This is useful
preparation for interpolation.
"""
function pack_nd{T<:Number}(nums::AbstractArray{T}, denoms::AbstractArray{T})
    size(nums) == size(denoms) || throw(DimensionMismatch("nums and denoms must have the same size"))
    numdenom = Array(NumDenom{T}, size(nums))
    @simd for I in eachindex(nums)
        @inbounds numdenom[I] = NumDenom(nums[I], denoms[I])
    end
    numdenom
end

###
### Block, a type to simplify working with tiled arrays
###
"""
`B = Block(A, i, j)` returns a slice of `A`.  If `A` is a 5-dimensional array,
`Block(A, i, j)` will return a view of `A[:,:,:,i,j]`.  These can be viewed as special-purpose SubArrays.
"""
immutable Block{T,N,NE,A<:DenseArray} <: DenseArray{T,N}
    parent::A
    extraindexes::NTuple{NE,Int}
    offset::Int

    function Block(parent::A, extraindexes::NTuple{NE,Int})
        s = size(parent, 1)
        for i = 2:N
            s *= size(parent,i)
        end
        offset = 0
        for i = 1:NE
            offset += s*(extraindexes[i]-1)
            s *= size(parent,N+i)
        end
        new(parent, extraindexes, offset)
    end
end

function Block(parent::DenseArray, extraindexes::Tuple{Vararg{Int}})
    NE = length(extraindexes)
    N = ndims(parent)-NE
    Block{eltype(parent),N,NE,typeof(parent)}(parent, extraindexes)
end

Block(parent::DenseArray, extraindexes::Int...) = Block(parent, extraindexes)

size{T,N}(B::Block{T,N}) = size(B.parent)[1:N]
size{T,N}(B::Block{T,N}, i::Integer) = i <= N ? size(B.parent, i) : 1

pointer{T}(B::Block{T}) = pointer(B.parent)+B.offset*sizeof(T)

@inline function getindex!(dest, src, indexes...)
    checkbounds(src, indexes...)
    ndims(dest) == length(indexes) || throw(DimensionMismatch("dest and indexes don't match"))
    for d = 1:ndims(dest)
        size(dest, d) == length(indexes[d]) || throw(DimensionMismatch("dest and indexes don't match"))
    end
    Base._unsafe_getindex!(dest, Base.linearindexing(src, dest), src, indexes...)
end

# linear indexing
# FIXME: would be nice to have bounds-checking here, although currently the performance penalty is too high
getindex(B::Block, i::Real) = B.parent[i+B.offset]
setindex!(B::Block, v, i::Real) = setindex!(B.parent, v, i+B.offset)

# cartesian indexing
_eindexes(NE) = ntuple(i->:(B.extraindexes[$i]), NE)
@generated function getindex{T,N,NE}(B::Block{T,N,NE}, uindexes...)
    meta = Expr(:meta, :inline)
    eindexes = _eindexes(NE)
    quote
        $meta
        getindex(B.parent, uindexes..., $(eindexes...))
    end
end
@generated function setindex!{T,N,NE}(B::Block{T,N,NE}, v, uindexes...)
    meta = Expr(:meta, :inline)
    eindexes = _eindexes(NE)
    quote
        $meta
        setindex!(B.parent, v, uindexes..., $(eindexes...))
    end
end

### Generic interface for tiled array & array-of-arrays
getblock{T<:AbstractArray}(A::AbstractArray{T}, I...) = A[I...]
getblock{T<:Real}(A::AbstractArray{T}, I...) = Block(A, I...)

### Size & eltype support for the conventions of mismatch data representation
gridsize{T<:AbstractArray}(A::AbstractArray{T}) = size(A)
gridsize{T<:Union(Real,NumDenom)}(A::AbstractArray{T}) = size(A)[ndims(A)>>1+1:end]
gridsize(ND::Tuple) = gridsize(ND[1])

blocksize{T<:AbstractArray}(A::AbstractArray{T}) = size(A[1])
blocksize{T<:Union(Real,NumDenom)}(A::AbstractArray{T}) = size(A)[1:ndims(A)>>1]
blocksize(ND::Tuple) = blocksize(ND[1])

blockeltype{T<:AbstractArray}(A::AbstractArray{T}) = eltype(A[1])
blockeltype{T<:Real}(A::AbstractArray{T}) = T
blockeltype{T<:Real}(A::AbstractArray{NumDenom{T}}) = T
blockeltype(ND::Tuple) = blockeltype(ND[1])


#### MismatchData ####
"""

`MismatchData` is designed as an accessor for a file format, although
it can be used in-memory as well. The most important fields are:

- `blocksize`: a vector of length `ndims`, equal to `2*maxshift+1`
- `gridsize`: a vector of length `ndims`, giving the size of the grid
  of "control points"
- `data`: an array with the dimensions
  `[blocksize,2,gridsize,nstacks]`, holding the mismatch `num/denom`
  data.  For a 2d registration, this will be a 5-dimensional array
  even if you only have two images you're registering.
- `stacks`: a vector of stack indexes from the original images
- `stack_base`: the stack index of the reference ("fixed") image
"""
type MismatchData{T,N}
    data::Array{T,N}
    stack_base::Int
    stacks::Vector{Int}
    blocksize::Vector{Int}
    gridsize::Vector{Int}
    normalization::Char
    key::Int32
end
MismatchData(data, stack_base, stacks, blocksize, gridsize, normalization) =
    MismatchData(data, stack_base, stacks, blocksize, gridsize, normalization, Int32(0))

"""
`mmd = MismatchData(nums, denoms; [normalization=:intensity],
[stack_base=-1])` constructs a `MismatchData` object from a single
fixed/moving image pair.
"""
function MismatchData(nums::AbstractArray, denoms::AbstractArray; normalization = :intensity, stack_base=-1)
    gsize = gridsize(nums)
    N = length(gsize)
    numB = getblock(nums, ones(Int, N)...)
    T = eltype(numB)
    blocksize = size(numB)
    data = Array(T, blocksize..., 2, gsize...)
    rng = ntuple(i->1:blocksize[i], N)
    for i = 1:prod(gsize)
        t = ind2sub(gsize, i)
        data[rng..., 1, i] = getblock(nums, t...)
        data[rng..., 2, i] = getblock(denoms, t...)
    end
    MismatchData(data, stack_base, [1], [blocksize...], [gsize...], normalization==:intensity ? 'I' : 'P', Int32(0))
end

#### MismatchData files ####
"""
`mmd = MismatchData(filename)` loads `MismatchData` from the specified file.
"""
function MismatchData(filename::AbstractString)
    filename = endswith(filename, ".mismatch") ? filename : filename*".mismatch"
    s = open(filename, "r");
    magic = b"MISMATCH"
    magic_check = read(s, UInt8, length(magic))
    if magic_check != magic
        error("The format of ", filename, " does not seem to be correct")
    end
    version = read(s, Int32)
    if version < 1 || version > 2
        error("Version not recognized")
    end
    stack_base = Int(read(s, Int32))
    n_stacks = Int(read(s, Int32))
    stacks = convert(Vector{Int}, read(s, Int32, n_stacks))
    n_dims = Int(read(s, Int32))
    gridsize = convert(Vector{Int}, read(s, Int32, n_dims))
    blocksize = convert(Vector{Int}, read(s, Int32, n_dims))
    if any(blocksize .< 1)
        error(filename, " has incorrect block size; perhaps execution terminated prematurely?")
    end
    nbits = read(s, Int32)
    if nbits == 32
        T = Float32
    elseif nbits == 64
        T = Float64
    else
        error("Number of bits not recognized")
    end
    normint = read(s, Int32)
    normchar = 'I'
    if normint == 1
    elseif normint == 2
        normchar = 'P'
    else
        error("Normalization not recognized")
    end
    if version >= 2
        key = read(s, Int32)
    else
        key = Int32(0)
    end
    offset = position(s)
    # Check the file size
    n_blocks = prod(gridsize)
    n_pix = prod(blocksize)
    fileszexpected = 2*n_stacks*n_pix*n_blocks*sizeof(T)+offset
    filesz = filesize(filename)
    if filesz != fileszexpected
        error("File size ", filesz, " does not match the expected ", fileszexpected, " for file ", filename)
    end
    datasize = tuple(blocksize..., 2, gridsize..., n_stacks)
    data = Mmap.mmap(s, Array{T,length(datasize)}, datasize, offset)
    return MismatchData{T,ndims(data)}(data, stack_base, stacks, blocksize, gridsize, normchar, key)
end

"""
`mmd = MismatchData(filenames::Vector)` loads `MismatchData` from a list of files.
"""
function MismatchData(filenames::Vector{AbstractString})
    mmd = MismatchData(filenames[1])
    T = eltype(mmd)
    mmd_all = Array(MismatchData{T}, length(filenames))
    mmd_all[1] = mmd
    for i = 2:length(filenames)
        mmd = MismatchData(filenames[i])
        if eltype(mmd) != T
            error("All mismatch files must use the same data type")
        end
        mmd_all[i] = mmd
    end
    return mmd_all
end

function write_header{T<:AbstractFloat}(s::IO, mmd::MismatchData{T})
    magic = b"MISMATCH"
    write(s, magic)
    write(s, Int32(1))  # version
    write(s, Int32(mmd.stack_base))
    write(s, Int32(length(mmd.stacks)))
    stacks32 = convert(Vector{Int32}, mmd.stacks)
    write(s, stacks32)
    write(s, Int32(length(mmd.gridsize)))
    gridsize32 = convert(Vector{Int32}, mmd.gridsize)
    write(s, gridsize32)
    blocksize32 = convert(Vector{Int32}, mmd.blocksize)
    write(s, blocksize32)
    nbits = sizeof(T)*8
    write(s, Int32(nbits))
    if mmd.normalization == 'I'
        write(s, Int32(1))
    elseif mmd.normalization == 'P'
        write(s, Int32(2))
    else
        error("Normalization not recognized")
    end
end

function create_mmd(filename::AbstractString)
    ext = ".mismatch"
    if !endswith(filename, ext)
        filename = filename*ext
    end
    s = open(filename, "w")
    return s, filename
end

"""
`save(filename, mmd::MismatchData)` writes `mmd` to the file specified
by `filename`.  If not included, a `".mismatch"` extension will be
added to the filename.
"""
function save(filename::AbstractString, mmd::MismatchData)
    s, fname = create_mmd(filename)
    write_header(s, mmd)
    write(s, mmd.data)
    close(s)
end

function append_mmd(s::IO, nums::AbstractArray, denoms::AbstractArray)
    gsz = gridsize(nums)
    for i = 1:prod(gsz)
        t = ind2sub(gsz, i)
        write(s, getblock(nums, t...))
        write(s, getblock(denoms, t...))
    end
end

normdict = Dict('I' => "intensity", 'P' => "pixels")

#### MismatchData utilities ####
function show(io::IO, mmd::MismatchData)
    println(io, typeof(mmd), ":")
    println(io, "  stack_base: ", mmd.stack_base)
    print(io, "  stacks: ", mmd.stacks')
    print(io, "  blocksize: ", mmd.blocksize')
    print(io, "  gridsize:  ", mmd.gridsize')
    print(io, "  normalization: ", mmd.normalization)
end

eltype{T}(mmd::MismatchData{T}) = T
ndims{T,N}(mmd::MismatchData{T,N}) = N


#### Utility functions ####

"""
`@nlinear N i` constructs a linear index from a `Base.Cartesian` "tuple".
"""
macro nlinear(N, A, i)
    sym = symbol(string(i)*"_"*string(N))
    ex = :($sym-1)
    for n = N-1:-1:1
        sym = symbol(string(i)*"_"*string(n))
        ex = :(size($(esc(A)), $n)*$ex + $sym - 1)
    end
    :($ex+1)
end

"""
`index = indminmismatch(num, denom, thresh)` returns the location of
the minimum value of what is effectively `num./denom`.  However, it
considers only those points for which `denom .> thresh`; moreover, it
will never choose an edge point.  `index` is a linear index into the
arrays.
"""
@generated function indminmismatch{T,N}(num::AbstractArray{T,N}, denom::AbstractArray{T,N}, thresh::Real)
    quote
        size(num) == size(denom) || error("size(num) = $(size(num)), but size(denom) = $(size(denom))")
        imin = length(num)>>1+1   # default is center of the array
        rmin = typemax(T)
        threshT = convert(T, thresh)
        @inbounds @nloops $N i d->2:size(num,d)-1 begin
            den = @nref $N denom i
            if den > threshT
                r = (@nref $N num i)/den
                if r < rmin
                    imin = @nlinear $N num i
                    rmin = r
                end
            end
        end
        imin
    end
end

"""
`ind2disp(out, blocksize, index)` takes `index`, interpreted as a
linear index into an array of size `blocksize`, and returns (in `out`)
the displacement of this location from the center (in Cartesian
coordinates).  `out` and `blocksize` must be of the same length.
"""
function ind2disp!(out::AbstractArray, blocksize, ind::Int)
    n = length(blocksize)
    length(out) == n || error("out has length $(length(out)) but blocksize has length $n.")
    s = blocksize[1]
    for i = 2:n-1
        s *= blocksize[i]
    end
    ind -= 1
    for i = n-1:-1:1
        c = div(ind, s)
        out[i+1] = c - ((blocksize[i+1]+1)>>1) + 1
        ind -= c*s
        s /= blocksize[i]
    end
    out[1] = ind - ((blocksize[1]+1)>>1) + 1
    out
end

"""
`disp = ind2disp(blocksize, index)` takes `index`, interpreted as a
linear index into an array of size `blocksize`, and returns
the displacement `disp` of this location from the center (in Cartesian
coordinates). See also `ind2disp!`.
"""
ind2disp(blocksize, ind::Int) = ind2disp!(Array(Int, length(blocksize)), blocksize, ind)

end
