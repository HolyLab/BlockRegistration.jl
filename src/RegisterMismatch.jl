__precompile__()

module RegisterMismatch

import Base: copy, eltype, isnan, ndims

using Images
using RFFT
using RegisterCore

include("RegisterMismatchCommon.jl")

export
    CMStorage,
    fillfixed!,
    mismatch,
    mismatch!,
    mismatch_blocks,
    mismatch_blocks!

if !isdefined(:A_mul_B!)
    const A_mul_B! = A_mul_B
    const A_mul_Bt! = A_mul_Bt
end

FFTW.set_num_threads(min(CPU_CORES, 8))
const FFTPROD = [2,3]

type NanCorrFFTs{T<:AbstractFloat,N}
    I0::RCpair{T,N}
    I1::RCpair{T,N}
    I2::RCpair{T,N}
end

copy(x::NanCorrFFTs) = NanCorrFFTs(copy(x.I0), copy(x.I1), copy(x.I2))

# A type that allows you to pre-allocate buffers for intermediate computations,
# and pre-plan the FFTs.
type CMStorage{T<:AbstractFloat,N}
    blocksize::Vector{Int}
    maxshift::Vector{Int}
    getindexes::Vector{UnitRange{Int}}   # indexes for pulling padded data, in source-coordinates
    padded::Array{T,N}
    fixed::NanCorrFFTs{T,N}
    moving::NanCorrFFTs{T,N}
    buf1::RCpair{T,N}
    buf2::RCpair{T,N}
    # the next two store the result of calling plan_fft! and plan_ifft!
    fftfunc!::Function
    ifftfunc!::Function
    shiftindexes::Vector{Vector{Int}} # indexes for performing fftshift & snipping from -maxshift:maxshift

    function CMStorage(::Type{T}, blocksize::DimsLike, maxshift::DimsLike; flags=FFTW.ESTIMATE, timelimit=Inf, display=true)
        length(blocksize) == length(maxshift) || error("Dimensionality mismatch")
        padsz = padsize(blocksize, maxshift)
        padszt = tuple(padsz...)
        padded = Array(T, padszt)
        getindexes = padranges(blocksize, maxshift)
        maxshiftv = [maxshift...]
        region = find(maxshiftv .> 0)
        fixed  = NanCorrFFTs(RCpair(T, padszt, region), RCpair(T, padszt, region), RCpair(T, padszt, region))
        moving = NanCorrFFTs(RCpair(T, padszt, region), RCpair(T, padszt, region), RCpair(T, padszt, region))
        buf1 = RCpair(T, padszt, region)
        buf2 = RCpair(T, padszt, region)
        tcalib = 0
        if display && flags != FFTW.ESTIMATE
            print("Planning FFTs (maximum $timelimit seconds)...")
            flush(STDOUT)
            tcalib = time()
        end
        fftfunc = plan_rfft!(fixed.I0, flags=flags, timelimit=timelimit/2)
        ifftfunc = plan_irfft!(fixed.I0, flags=flags, timelimit=timelimit/2)
        if display && flags != FFTW.ESTIMATE
            dt = time()-tcalib
            @printf("done (%.2f seconds)\n", dt)
        end
        shiftindexes = Vector{Int}[ [size(padded,i)+(-maxshift[i]+1:0); 1:maxshift[i]+1] for i = 1:length(maxshift) ]
        new([blocksize...], maxshiftv, getindexes, padded, fixed, moving, buf1, buf2, fftfunc, ifftfunc, shiftindexes)
    end
end
CMStorage{T<:Real}(::Type{T}, blocksize, maxshift; kwargs...) = CMStorage{T,length(blocksize)}(T, blocksize, maxshift; kwargs...)

eltype{T,N}(cms::CMStorage{T,N}) = T
 ndims{T,N}(cms::CMStorage{T,N}) = N

#### Compute the mismatch between two images as a function of the shift ####
function mismatch{T<:Real}(fixed::AbstractArray{T}, moving::AbstractArray{T}, maxshift::DimsLike; normalization = :intensity, resulttype = Float64)
    assertsamesize(fixed, moving)
    maxshiftv = tovec(maxshift)
    msz = 2maxshiftv.+1
    num   = Array(resulttype, msz...)
    denom = Array(resulttype, msz...)
    cms = CMStorage(resulttype, size(fixed), maxshiftv)
    fillfixed!(cms, fixed)
    mismatch!(num, denom, cms, moving, normalization=normalization)
    return num, denom
end

function mismatch!{T<:Real}(num::AbstractArray, denom::AbstractArray, cms::CMStorage{T}, moving::AbstractArray; normalization = :intensity)
    # Pad the moving snippet using any available data, including
    # regions that might be in the parent Array but are not present
    # within the boundaries of the SubArray. Use NaN only for pixels
    # truly lacking data.
    checksize_maxshift(num, cms.maxshift)
    checksize_maxshift(denom, cms.maxshift)
    safe_get!(cms.padded, data(moving), tuple(cms.getindexes...), NaN)
    fftnan!(cms.moving, cms.padded, cms.fftfunc!)
    # Compute the mismatch
    f0 = complex(cms.fixed.I0)
    f1 = complex(cms.fixed.I1)
    f2 = complex(cms.fixed.I2)
    m0 = complex(cms.moving.I0)
    m1 = complex(cms.moving.I1)
    m2 = complex(cms.moving.I2)
    tnum   = complex(cms.buf1)
    tdenom = complex(cms.buf2)
    if normalization == :intensity
        for i = 1:length(tnum)
            c = 2*conj(f1[i])*m1[i]
            q = conj(f2[i])*m0[i] + conj(f0[i])*m2[i]
            tdenom[i] = q
            tnum[i] = q - c
        end
    elseif normalization == :pixels
        for i = 1:length(tnum)
            tdenom[i] = conj(f0[i])*m0[i]
            tnum[i] = conj(f2[i])*m0[i] - 2*conj(f1[i])*m1[i] + conj(f0[i])*m2[i]
        end
    else
        error("normalization $normalization not recognized")
    end
    cms.ifftfunc!(cms.buf1)
    cms.ifftfunc!(cms.buf2)
    getindex!(num,   real(cms.buf1), cms.shiftindexes...)
    getindex!(denom, real(cms.buf2), cms.shiftindexes...)
    nothing
end

#### Compute the mismatch across blocks of the images ####
function mismatch_blocks(fixed::AbstractArray, moving::AbstractArray, gridsize::DimsLike, maxshift::DimsLike;
                         overlap::DimsLike = zeros(Int, ndims(fixed)),
                         blocksize::DimsLike = defaultblocksize(fixed, gridsize, overlap),
                         resulttype = Float64,
                         normalization = :intensity,
                         flags = FFTW.MEASURE,
                         kwargs...)
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
    cms = CMStorage(resulttype, blocksize, maxshiftv; flags=flags, kwargs...)
    mismatch_blocks!(nums, denoms, fixed, moving, cms, normalization=normalization)
    nums, denoms
end

function mismatch_blocks!(nums, denoms, fixed, moving, cms; normalization=:intensity)
    assertsamesize(fixed, moving)
    N = ndims(cms)
    gsize = gridsize(nums)
    lower, upper = blockspan(fixed, cms.blocksize, gsize)
    for k = 1:prod(gsize)
        c = ind2sub(gsize, k)
        # Create snippets
        rng = UnitRange{Int}[ lower[i][c[i]]:upper[i][c[i]] for i = 1:N ]
	fsnip = Base.sub_unsafe(data(fixed), tuple(rng...)) #sub throws an error in 0.4 when rng extends outside of bounds, see github #10296
        msnip = Base.sub_unsafe(data(moving), tuple(rng...))
        # Perform the calculation
        fillfixed!(cms, fsnip)
        mismatch!(getblock(nums, c...), getblock(denoms, c...), cms, msnip; normalization=normalization)
    end
end

# Calculate the components needed to "nancorrelate"
function fftnan!{T<:Real}(out::NanCorrFFTs{T}, A::AbstractArray{T}, fftfunc!::Function)
    I0 = real(out.I0)
    I1 = real(out.I1)
    I2 = real(out.I2)
    assertsamesize(A, I0)
    _fftnan!(parent(I0), parent(I1), parent(I2), A)
    fftfunc!(out.I0)
    fftfunc!(out.I1)
    fftfunc!(out.I2)
    out
end

function _fftnan!{T<:Real}(I0, I1, I2, A::AbstractArray{T})
    @inbounds for i in CartesianRange(size(A))
        a = A[i]
        f = !isnan(a)
        I0[i] = f
        af = f ? a : zero(T)
        I1[i] = af
        I2[i] = af*af
    end
end

function fillfixed!{T}(cms::CMStorage{T}, fixed::AbstractArray)
    fill!(cms.padded, NaN)
    X = sub(cms.padded, ntuple(d->(1:size(fixed,d))+cms.maxshift[d], ndims(fixed)))
    copy!(X, fixed)
    fftnan!(cms.fixed, cms.padded, cms.fftfunc!)
end

function fillfixed!{T}(cms::CMStorage{T}, fixed::SubArray)
    fill!(cms.padded, NaN)
    X = sub(cms.padded, ntuple(d->(1:size(fixed,d))+cms.maxshift[d], ndims(fixed)))
    get!(X, parent(fixed), parentindexes(fixed), NaN)
    fftnan!(cms.fixed, cms.padded, cms.fftfunc!)
end

#### Utilities

isnan{T}(A::Array{Complex{T}}) = isnan(real(A)) | isnan(imag(A))
function sumsq_finite(A)
    s = 0.0
    for a in A
        if isfinite(a)
            s += a*a
        end
    end
    if s == 0
        error("No finite values available")
    end
    s
end

end
