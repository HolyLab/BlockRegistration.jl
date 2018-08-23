__precompile__()

module RegisterMismatchCuda

using CUDArt, CUDAdrv, CUFFT, Images, RegisterCore

import Base: close, eltype, ndims
import CUDArt: free, device, pitchedptr
import Images: sdims, coords_spatial, data

export
    CMStorage,
    fillfixed!,
    mismatch,
    mismatch!,
    mismatch_apertures,
    mismatch_apertures!

include("RegisterMismatchCommon.jl")

"""
The major types and functions exported are:

- `mismatch` and `mismatch!`: compute the mismatch between two images
- `mismatch_apertures` and `mismatch_apertures!`: compute apertured mismatch between two images
- `mismatch0`: simple direct mismatch calculation with no shift
- `nanpad`: pad the smaller image with NaNs
- `highpass`: highpass filter an image
- `correctbias!`: replace corrupted mismatch data (due to camera bias inhomogeneity) with imputed data
- `truncatenoise!`: threshold mismatch computation to prevent problems from roundoff
- `aperture_grid`: create a regular grid of apertures
- `allocate_mmarrays`: create storage for output of `mismatch_apertures!`
- `CMStorage`: a type that facilitates re-use of intermediate storage during registration computations
"""
RegisterMismatchCuda

const ptxdict = Dict()
const mdlist = Array{CuModule}(0)

function init(devlist)
    global ptxdict
    global mdlist
    isempty(mdlist) || error("mdlist is not empty")
    for dev in devlist
        device(dev)
        thisdir = splitdir(@__FILE__)[1]
        md = CuModuleFile(joinpath(thisdir, "register_mismatch_cuda.ptx"))
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

#This should work because CUDAdrv is supposed to delegate memory management to Julia's GC
close() = (empty!(mdlist); empty!(ptxdict))

const FFTPROD = [2,3]

type NanCorrFFTs{T<:AbstractFloat,N}
    I0::Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}}
    I1::Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}}
    I2::Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}}
end

function free(ncf::NanCorrFFTs)
    for obj in (ncf.I0, ncf.I1, ncf.I2)
        RCfree(obj[1],obj[2])
    end
end

"""
`CMStorage(T, aperture_width, maxshift; [flags=FFTW.ESTIMATE],
[timelimit=Inf], [display=true])` prepares for FFT-based mismatch
computations over domains of size `aperture_width`, computing the
mismatch up to shifts of size `maxshift`.  The keyword arguments allow
you to control the planning process for the FFTs.
"""
type CMStorage{T<:AbstractFloat,N}
    aperture_width::Vector{Float64}
    maxshift::Vector{Int}
    getindexes::Vector{UnitRange{Int}}   # indexes for pulling padded data, in source-coordinates
    setindexes::Vector{UnitRange{Int}}   # indexes for pushing fixed data, in source-coordinates
    fixed::NanCorrFFTs{T,N}
    moving::NanCorrFFTs{T,N}
    num::Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}}
    denom::Tuple{CudaPitchedArray{T,N},CudaPitchedArray{Complex{T},N}}
    numhost::Array{T,N}
    denomhost::Array{T,N}
    # the next two store the result of calling plan_fft! and plan_ifft!
    fftfunc::Function
    ifftfunc::Function
    fdshift::Vector{Int} # shift needed to unwrap (fftshift)
    stream

    function CMStorage(::Type{T}, aperture_width::WidthLike, maxshift::DimsLike; stream=null_stream)
        blocksize = map(x->ceil(Int,x), aperture_width)
        length(blocksize) == length(maxshift) || error("Dimensionality mismatch")
        padsz = padsize(blocksize, maxshift)
        padszt = tuple(padsz...)
        getindexes = padranges(blocksize, maxshift)
        setindexes = UnitRange{Int}[(1:blocksize[i])+maxshift[i] for i = 1:length(blocksize)]
        fixed  = NanCorrFFTs(RCpair(T, padszt), RCpair(T, padszt), RCpair(T, padszt))
        moving = NanCorrFFTs(RCpair(T, padszt), RCpair(T, padszt), RCpair(T, padszt))
        num = RCpair(T, padszt)
        denom = RCpair(T, padszt)
        mmsz = map(x->2x+1, (maxshift...))
        numhost, denomhost = Array{T}(mmsz), Array{T}(mmsz)
        fftfunc = plan(num[2], num[1], stream=stream)
        ifftfunc = plan(num[1], num[2], stream=stream)
        maxshiftv = [maxshift...]
#        shiftindexes = [ (padszt[i]+(-maxshift[i]+1:0), 1:maxshift[i]+1) for i = 1:length(maxshift) ]
        fdshift = [-maxshift[i] for i = 1:length(maxshift)]
        new(Float64[aperture_width...], maxshiftv, getindexes, setindexes, fixed, moving, num, denom, numhost, denomhost, fftfunc, ifftfunc, fdshift, stream)
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

# Some tools from Images
sdims(A::CudaPitchedArray) = ndims(A)
coords_spatial(A::CudaPitchedArray) = 1:ndims(A)

### Main API

"""
`mm = mismatch([T], fixed, moving, maxshift;
[normalization=:intensity])` computes the mismatch between `fixed` and
`moving` as a function of translations (shifts) up to size `maxshift`.
Optionally specify the element-type of the mismatch arrays (default
`Float32` for Integer- or FixedPoint-valued images) and the
normalization scheme (`:intensity` or `:pixels`).

`fixed` and `moving` must have the same size; you can pad with
`NaN`s as needed. See `nanpad`.

This operation is synchronous with respect to the host.
"""
function mismatch{T<:Real}(::Type{T}, fixed::AbstractArray, moving::AbstractArray, maxshift::DimsLike; normalization = :intensity)
    assertsamesize(fixed, moving)
    d_fixed  = CudaPitchedArray(convert(Array{T}, fixed))
    d_moving = CudaPitchedArray(convert(Array{T}, moving))
    mm = mismatch(d_fixed, d_moving, maxshift, normalization=normalization)
    free(d_fixed)
    free(d_moving)
    mm
end

function mismatch{T}(fixed::AbstractCudaArray{T}, moving::AbstractCudaArray{T}, maxshift::DimsLike; normalization = :intensity)
    assertsamesize(fixed, moving)
    nd = ndims(fixed)
    maxshiftv = tovec(maxshift)
    cms = CMStorage(T, size(fixed), maxshiftv)
    mm = MismatchArray(T, 2maxshiftv.+1...)
    fillfixed!(cms, fixed)
    mismatch!(mm, cms, moving, normalization=normalization)
    free(cms)
    mm
end

"""
`mms = mismatch_apertures([T], fixed, moving, gridsize, maxshift;
[normalization=:pixels], [flags=FFTW.MEASURE], kwargs...)` computes
the mismatch between `fixed` and `moving` over a regularly-spaced grid
of aperture centers, effectively breaking the images up into
chunks. The maximum-allowed shift in any aperture is `maxshift`.

`mms = mismatch_apertures([T], fixed, moving, aperture_centers,
aperture_width, maxshift; kwargs...)` computes the mismatch between
`fixed` and `moving` over a list of apertures of size `aperture_width`
at positions defined by `aperture_centers`.

`fixed` and `moving` must have the same size; you can pad with `NaN`s
as needed to ensure this.  You can optionally specify the real-valued
element type mm; it defaults to the element type of `fixed` and
`moving` or, for Integer- or FixedPoint-valued images, `Float32`.

On output, `mms` will be an Array-of-MismatchArrays, with the outer
array having the same "grid" shape as `aperture_centers`.  The centers
can in general be provided as an vector-of-tuples, vector-of-vectors,
or a matrix with each point in a column.  If your centers are arranged
in a rectangular grid, you can use an `N`-dimensional array-of-tuples
(or array-of-vectors) or an `N+1`-dimensional array with the center
positions specified along the first dimension. See `aperture_grid`.
"""
function mismatch_apertures{T}(::Type{T},
                               fixed::AbstractArray,
                               moving::AbstractArray,
                               aperture_centers::AbstractArray,
                               aperture_width::WidthLike,
                               maxshift::DimsLike;
                               kwargs...)
    assertsamesize(fixed, moving)
    d_fixed  = CudaPitchedArray(convert(Array{T}, sdata(fixed)))
    d_moving = CudaPitchedArray(convert(Array{T}, moving))
    mms = mismatch_apertures(d_fixed, d_moving, aperture_centers, aperture_width, maxshift; kwargs...)
    free(d_fixed)
    free(d_moving)
    mms
end

# only difference here relative to RegisterMismatch is the lack of the
# FFTW keywords
function mismatch_apertures{T}(fixed::AbstractCudaArray{T},
                               moving::AbstractCudaArray,
                               aperture_centers::AbstractArray,
                               aperture_width::WidthLike,
                               maxshift::DimsLike;
                               normalization = :pixels,
                               kwargs...)
    nd = sdims(fixed)
    assertsamesize(fixed,moving)
    (length(aperture_width) == nd && length(maxshift) == nd) || error("Dimensionality mismatch")
    mms = allocate_mmarrays(T, aperture_centers, maxshift)
    cms = CMStorage(T, aperture_width, maxshift; kwargs...)
    mismatch_apertures!(mms, fixed, moving, aperture_centers, cms; normalization=normalization)
    free(cms)
    mms
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
    dstindexes = Array{UnitRange{Int}}(nd)
    srcindexes = Array{UnitRange{Int}}(nd)
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
    nsm = CUDArt.attribute(device(), CUDArt.rt.cudaDevAttrMultiProcessorCount)
    mul = min(32, ceil(Int, length(paddedf)/(prod(cudablocksize)*nsm)))
    args = (pointer(paddedf), pointer(cms.fixed.I2[1]), pointer(cms.fixed.I0[1]), size(paddedf,1), size(paddedf,2), size(paddedf,3), pitchel(paddedf))
    argtypes =  ((typeof(x) for x in args)...)
    CUDAdrv.cudacall(components_func, mul*nsm, cudablocksize, argtypes, args...; shmem=4, stream=convert(CUDAdrv.CuStream, stream))
    # Compute FFTs
    obj = cms.fixed
    for item in (obj.I0, obj.I1, obj.I2)
        cms.fftfunc(item[2], item[1], true)
    end
    obj
end

"""
`mismatch!(mm, cms, moving; [normalization=:intensity])`
computes the mismatch as a function of shift, storing the result in
`mm`. The `fixed` image has been prepared in `cms`, a `CMStorage` object.
"""
function mismatch!{T}(mm::MismatchArray, cms::CMStorage{T}, moving::CudaPitchedArray; normalization = :intensity, m_offset = ntuple(i->0, ndims(cms)))
    global ptxdict
    dev = device(cms)
    device(moving) == dev || error("Moving and cms must be on the same device")
    checksize_maxshift(mm, cms.maxshift)
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
    nsm = CUDArt.attribute(device(), CUDArt.rt.cudaDevAttrMultiProcessorCount)
    mul = min(32, ceil(Int, length(paddedm)/(prod(cudablocksize)*nsm)))
    args = (pointer(paddedm), pointer(cms.moving.I2[1]), pointer(cms.moving.I0[1]), size(paddedm,1), size(paddedm,2), size(paddedm,3), pitchel(paddedm))
    argtypes =  ((typeof(x) for x in args)...)
    CUDAdrv.cudacall(components_func, mul*nsm, cudablocksize, argtypes, args...; shmem=4, stream=convert(CUDAdrv.CuStream, stream))
    # Compute FFTs
    obj = cms.moving
    for item in (obj.I0, obj.I1, obj.I2)
        cms.fftfunc(item[2], item[1], true)
    end
    # Perform the convolution in fourier space
    d_numC = cms.num[2]
    d_denomC = cms.denom[2]
    args = (
        pointer(cms.fixed.I1[2]),  pointer(cms.fixed.I2[2]),  pointer(cms.fixed.I0[2]),
        pointer(cms.moving.I1[2]), pointer(cms.moving.I2[2]), pointer(cms.moving.I0[2]),
        pointer(cms.num[2]), pointer(cms.denom[2]),
        size(d_numC,1), size(d_numC,2), size(d_numC,3), pitchel(d_numC))
    argtypes =  ((typeof(x) for x in args)...)
    CUDAdrv.cudacall(conv_func, mul*nsm, cudablocksize, argtypes, args...; shmem=4, stream=convert(CUDAdrv.CuStream, stream))
    # Perform the equivalent of the fftshift
    fdshift = cms.fdshift
    d_num = cms.num[1]
    d_denom = cms.denom[1]
    args = (
        pointer(d_numC), pointer(d_denomC),
        convert(T,fdshift[1]),
        convert(T,length(fdshift)>1 ? fdshift[2] : 0),
        convert(T,length(fdshift)>2 ? fdshift[3] : 0),
        size(d_num,1), size(d_numC,1), size(d_numC,2), size(d_numC,3), pitchel(d_numC),
        length(d_num))
    argtypes =  ((typeof(x) for x in args)...)
    CUDAdrv.cudacall(fdshift_func, mul*nsm, cudablocksize, argtypes, args...; shmem=4, stream=convert(CUDAdrv.CuStream, stream))
    # Compute the IFFTs
    cms.ifftfunc(d_num, d_numC, false)
    cms.ifftfunc(d_denom, d_denomC, false)
    # Copy result to host
    destI = ntuple(d->1:2*cms.maxshift[d]+1, nd)
    copy!(cms.numhost,   d_num,   destI, stream=stream)
    copy!(cms.denomhost, d_denom, destI, stream=stream)
    copy!(mm, (cms.numhost, cms.denomhost))
end

"""
`mismatch_apertures!(mms, fixed, moving, aperture_centers, cms;
[normalization=:pixels])` computes the mismatch between `fixed` and
`moving` over a list of apertures at positions defined by
`aperture_centers`.  The parameters and working storage are contained
in `cms`, a `CMStorage` object. The results are stored in `mms`, an
Array-of-MismatchArrays which must have length equal to the number of
aperture centers.
"""
function mismatch_apertures!(mms, fixed, moving, aperture_centers, cms; normalization=:pixels)
    assertsamesize(fixed, moving)
    N = ndims(cms)
    for (mm,center) in zip(mms, each_point(aperture_centers))
        rng = aperture_range(center, cms.aperture_width)
        fillfixed!(cms, fixed; f_indexes=rng)
        offset = [first(rng[d])-1 for d = 1:N]
        mismatch!(mm, cms, moving; normalization=normalization, m_offset=offset)
    end
    device_synchronize()
    mms
end


### Utilities

function assertsamesize(A::CudaPitchedArray, B::CudaPitchedArray)
    size(A,1) == size(B,1) && size(A,2) == size(B,2) && size(A,3) == size(B,3) || error("Arrays are not the same size")
end

end # module
