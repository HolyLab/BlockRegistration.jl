__precompile__()

module RegisterMismatchCuda

using CUDArt, CUFFT, ArrayOps, Images, RegisterCore, Compat

import Base: close, eltype, ndims
import CUDArt: free, device, pitchedptr
import ArrayOps: assertsamesize
import Images: sdims, coords_spatial

include("RegisterMismatchCommon.jl")

export
    CMStorage,
    fillfixed!,
    mismatch,
    mismatch!,
    mismatch_blocks,
    mismatch_blocks!

const ptxdict = Dict()
const mdlist = Array(CuModule, 0)

function init(devlist)
    global ptxdict
    global mdlist
    isempty(mdlist) || error("mdlist is not empty")
    for dev in devlist
        device(dev)
        md = CuModule(joinpath(ENV["JULIAFUNCDIR"], "register_mismatch_cuda.ptx"), false)
        ptxdict[(dev, "components_func", Float32)] = CuFunction(md, "kernel_conv_components_float")
        ptxdict[(dev, "components_func", Float64)] = CuFunction(md, "kernel_conv_components_double")
        ptxdict[(dev, "conv_func", :pixels, Float32)] = CuFunction(md, "kernel_calcNumDenom_pixels_float")
        ptxdict[(dev, "conv_func", :pixels, Float64)] = CuFunction(md, "kernel_calcNumDenom_pixels_double")
        ptxdict[(dev, "conv_func", :intensity, Float32)] = CuFunction(md, "kernel_calcNumDenom_intensity_float")
        ptxdict[(dev, "conv_func", :intensity, Float64)] = CuFunction(md, "kernel_calcNumDenom_intensity_double")
        ptxdict[(dev, "fdshift", Float32)] = CuFunction(md, "kernel_fdshift_float")
        ptxdict[(dev, "fdshift", Float64)] = CuFunction(md, "kernel_fdshift_double")
        push!(mdlist, md)
    end
end

close() = (for md in mdlist; unload(md); end; empty!(mdlist); empty!(ptxdict))

const FFTPROD = [2,3]

type NanCorrFFTs{T<:AbstractFloat,N}
    I0::@compat(Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}})
    I1::@compat(Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}})
    I2::@compat(Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}})
end

function free(ncf::NanCorrFFTs)
    for obj in (ncf.I0, ncf.I1, ncf.I2)
        RCfree(obj[1],obj[2])
    end
end

type CMStorage{T<:AbstractFloat,N}
    blocksize::Vector{Int}
    maxshift::Vector{Int}
    getindexes::Vector{UnitRange{Int}}   # indexes for pulling padded data, in source-coordinates
    setindexes::Vector{UnitRange{Int}}   # indexes for pushing fixed data, in source-coordinates
    fixed::NanCorrFFTs{T,N}
    moving::NanCorrFFTs{T,N}
    num::@compat(Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}})
    denom::@compat(Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}})
    # the next two store the result of calling plan_fft! and plan_ifft!
    fftfunc::Function
    ifftfunc::Function
    fdshift::Vector{Int} # shift needed to unwrap (fftshift)
    stream

    function CMStorage(::Type{T}, blocksize::DimsLike, maxshift::DimsLike; stream=null_stream)
        length(blocksize) == length(maxshift) || error("Dimensionality mismatch")
        padsz = padsize(blocksize, maxshift)
        padszt = tuple(padsz...)
        getindexes = padranges(blocksize, maxshift)
        setindexes = UnitRange{Int}[(1:blocksize[i])+maxshift[i] for i = 1:length(blocksize)]
        fixed  = NanCorrFFTs(RCpair(T, padszt), RCpair(T, padszt), RCpair(T, padszt))
        moving = NanCorrFFTs(RCpair(T, padszt), RCpair(T, padszt), RCpair(T, padszt))
        num = RCpair(T, padszt)
        denom = RCpair(T, padszt)
        fftfunc = plan(num[2], num[1], stream=stream)
        ifftfunc = plan(num[1], num[2], stream=stream)
        maxshiftv = [maxshift...]
#        shiftindexes = [ (padszt[i]+(-maxshift[i]+1:0), 1:maxshift[i]+1) for i = 1:length(maxshift) ]
        fdshift = [-maxshift[i] for i = 1:length(maxshift)]
        new([blocksize...], maxshiftv, getindexes, setindexes, fixed, moving, num, denom, fftfunc, ifftfunc, fdshift, stream)
    end
end
# Note: display doesn't do anything
CMStorage{T<:Real}(::Type{T}, blocksize, maxshift; stream=null_stream, display=false) = CMStorage{T,length(blocksize)}(T, blocksize, maxshift; stream=stream)

function free(cms::CMStorage)
    free(cms.fixed)
    free(cms.moving)
    RCfree(cms.num[1], cms.num[2])
    RCfree(cms.denom[1], cms.denom[2])
end

device(cms::CMStorage) = device(cms.num[1])

eltype{T,N}(cms::CMStorage{T,N}) = T
 ndims{T,N}(cms::CMStorage{T,N}) = N

function pitchedptr(a::Block)
    1 <= ndims(a) <= 3 || error("Supports only dimensions 1, 2, or 3")
    CUDArt.rt.cudaPitchedPtr(pointer(a), size(a,1)*sizeof(eltype(a)), size(a,1), size(a,2))
end

# Some tools from Images
sdims(A::CudaPitchedArray) = ndims(A)
coords_spatial(A::CudaPitchedArray) = 1:ndims(A)

### Main API

# Simple wrapper for global translation. Synchronous with respect to the host.
function mismatch(fixed::AbstractArray, moving::AbstractArray, maxshift; normalization = :intensity, resulttype = Float64)
    assertsamesize(fixed, moving)
    d_fixed  = CudaPitchedArray(convert(Array{resulttype}, fixed))
    d_moving = CudaPitchedArray(convert(Array{resulttype}, moving))
    num, denom = mismatch(d_fixed, d_moving, maxshift, normalization=normalization, resulttype=resulttype)
    free(d_fixed)
    free(d_moving)
    num, denom
end

function mismatch(fixed::AbstractCudaArray, moving::AbstractCudaArray, maxshift; normalization = :intensity, resulttype = Float64)
    assertsamesize(fixed, moving)
    nd = ndims(fixed)
    maxshiftv = tovec(maxshift)
    cms = CMStorage(resulttype, size(fixed), maxshiftv)
    num = Array(resulttype, 2maxshiftv.+1...)
    denom = similar(num)
    fillfixed!(cms, fixed)
    mismatch!(num, denom, cms, moving, normalization=normalization)
    free(cms)
    num, denom
end

function mismatch_blocks(fixed::AbstractArray, moving::AbstractArray, gridsize::DimsLike, maxshift::DimsLike;
                         overlap::DimsLike = zeros(Int, ndims(fixed)),
                         blocksize::DimsLike = defaultblocksize(fixed, gridsize, overlap),
                         resulttype = Float64,
                         normalization = :intensity, kwargs...)
    assertsamesize(fixed, moving)
    d_fixed  = CudaPitchedArray(convert(Array{resulttype}, fixed))
    d_moving = CudaPitchedArray(convert(Array{resulttype}, moving))
    nums, denoms = mismatch_blocks(d_fixed, d_moving, gridsize, maxshift, overlap=overlap, blocksize=blocksize, normalization=normalization, resulttype=resulttype)
    free(d_fixed)
    free(d_moving)
    nums, denoms
end

function mismatch_blocks(fixed::AbstractCudaArray, moving::AbstractCudaArray, gridsize::DimsLike, maxshift::DimsLike;
                         overlap::DimsLike = zeros(Int, ndims(fixed)),
                         blocksize::DimsLike = defaultblocksize(fixed, gridsize, overlap),
                         resulttype = Float64,
                         normalization = :intensity)
    nd = sdims(fixed)
    assertsamesize(fixed,moving)
    (length(gridsize) <= nd && length(maxshift) <= nd) || error("Dimensionality mismatch")
    nums   = Array(Array{resulttype, nd}, gridsize...)
    denoms = Array(Array{resulttype, nd}, gridsize...)
    maxshiftv = tovec(maxshift)
    t = tuple(2maxshiftv.+1...)
    for i = 1:prod(gridsize)
        nums[i]   = Array(resulttype, t)
        denoms[i] = Array(resulttype, t)
    end
    cms = CMStorage(resulttype, blocksize, maxshiftv)
    mismatch_blocks!(nums, denoms, fixed, moving, cms, normalization=normalization)
    device_synchronize()
    nums, denoms
end

function fillfixed!{T}(cms::CMStorage{T}, fixed::CudaPitchedArray; f_indexes = ntuple(i->1:size(fixed,i), ndims(fixed)))
    dev = device(cms)
    device(fixed) == dev || error("Fixed and cms must be on the same device")
    nd = ndims(cms)
    ndims(fixed) == nd || error("Fixed and cms must have the same dimensionality")
    device(dev)
    components_func = ptxdict[(dev,"components_func", T)]
    stream = cms.stream
    # Pad
    paddedf = cms.fixed.I1[1]
    fill!(paddedf, NaN, stream=stream)
    dstindexes = Array(UnitRange{Int}, nd)
    srcindexes = Array(UnitRange{Int}, nd)
    for idim = 1:nd
        tmp = f_indexes[idim]
        i1 = first(tmp) >= 1 ? 1 : 2-first(tmp)
        i2 = last(tmp) <= size(fixed, idim) ? length(tmp) : length(tmp)-(last(tmp)-size(fixed, idim))
        srcindexes[idim] = tmp[i1]:tmp[i2]
        dstindexes[idim] = cms.setindexes[idim][i1]:cms.setindexes[idim][i2]
    end
    copy!(paddedf, tuple(dstindexes...), fixed, tuple(srcindexes...), stream=stream)
    # Prepare the components of the convolution
    cudablocksize = (16,16)
    nsm = attribute(device(), CUDArt.rt.cudaDevAttrMultiProcessorCount)
    mul = min(32, ceil(Int, length(paddedf)/(prod(cudablocksize)*nsm)))
    CUDArt.launch(components_func, mul*nsm, cudablocksize, (paddedf, cms.fixed.I2[1],  cms.fixed.I0[1],  size(paddedf,1), size(paddedf,2), size(paddedf,3), pitchel(paddedf)), stream=stream)
    # Compute FFTs
    obj = cms.fixed
    for item in (obj.I0, obj.I1, obj.I2)
        cms.fftfunc(item[2], item[1], true)
    end
    obj
end

function mismatch!{T}(num::DenseArray, denom::DenseArray, cms::CMStorage{T}, moving::CudaPitchedArray; normalization = :intensity, m_offset = ntuple(i->0, ndims(cms)))
    global ptxdict
    dev = device(cms)
    device(moving) == dev || error("Moving and cms must be on the same device")
    checksize_maxshift(num, cms.maxshift)
    checksize_maxshift(denom, cms.maxshift)
    device(dev)
    components_func = ptxdict[(dev,"components_func", T)]
    conv_func = ptxdict[(dev,"conv_func", normalization, T)]
    fdshift_func = ptxdict[(dev,"fdshift",T)]
    nd = ndims(cms)
    stream = cms.stream
    paddedm = cms.moving.I1[1]
    get!(paddedm, moving, ntuple(d->cms.getindexes[d]+m_offset[d], nd), NaN, stream=stream)
    # Prepare the components of the convolution
    cudablocksize = (16,16)
    nsm = attribute(device(), CUDArt.rt.cudaDevAttrMultiProcessorCount)
    mul = min(32, ceil(Int, length(paddedm)/(prod(cudablocksize)*nsm)))
    CUDArt.launch(components_func, mul*nsm, cudablocksize, (paddedm, cms.moving.I2[1], cms.moving.I0[1], size(paddedm,1), size(paddedm,2), size(paddedm,3), pitchel(paddedm)), stream=stream)
    # Compute FFTs
    obj = cms.moving
    for item in (obj.I0, obj.I1, obj.I2)
        cms.fftfunc(item[2], item[1], true)
    end
    # Perform the convolution in fourier space
    d_numC = cms.num[2]
    d_denomC = cms.denom[2]
    CUDArt.launch(conv_func, mul*nsm, cudablocksize, (
        cms.fixed.I1[2],  cms.fixed.I2[2],  cms.fixed.I0[2],
        cms.moving.I1[2], cms.moving.I2[2], cms.moving.I0[2],
        cms.num[2], cms.denom[2],
        size(d_numC,1), size(d_numC,2), size(d_numC,3), pitchel(d_numC)),
        stream=stream)
    # Perform the equivalent of the fftshift
    fdshift = cms.fdshift
    d_num = cms.num[1]
    d_denom = cms.denom[1]
    CUDArt.launch(fdshift_func, mul*nsm, cudablocksize, (
        d_numC, d_denomC,
        convert(T,fdshift[1]),
        convert(T,length(fdshift)>1 ? fdshift[2] : 0),
        convert(T,length(fdshift)>2 ? fdshift[3] : 0),
        size(d_num,1), size(d_numC,1), size(d_numC,2), size(d_numC,3), pitchel(d_numC),
        length(d_num)),
        stream=stream)
    # Compute the IFFTs
    cms.ifftfunc(d_num, d_numC, false)
    cms.ifftfunc(d_denom, d_denomC, false)
    # Copy result to host
    destI = ntuple(d->1:2*cms.maxshift[d]+1, nd)
    copy!(num,   d_num,   destI, stream=stream)
    copy!(denom, d_denom, destI, stream=stream)
end

function mismatch_blocks!(nums, denoms, fixed::CudaPitchedArray, moving::CudaPitchedArray, cms::CMStorage; normalization = :intensity)
    assertsamesize(fixed, moving)
    N = ndims(cms)
    gsize = gridsize(nums)
    lower, upper = blockspan(fixed, cms.blocksize, gsize)
    for k = 1:prod(gsize)
        c = ind2sub(gsize, k)
        f_indexes = ntuple(i->lower[i][c[i]]:upper[i][c[i]], N)
        offset = [lower[i][c[i]]-1 for i = 1:N]
        fillfixed!(cms, fixed, f_indexes=f_indexes)
        mismatch!(getblock(nums, c...), getblock(denoms, c...), cms, moving; normalization=normalization, m_offset=offset)
    end
    synchronize(cms.stream)
    nothing
end

### Utilities

assertsamesize(A::CudaPitchedArray, B::CudaPitchedArray) = size(A,1) == size(B,1) && size(A,2) == size(B,2) && size(A,3) == size(B,3)

end
