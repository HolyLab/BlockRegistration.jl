nt = 3  # number of time points
wpids = addprocs(min(nt, 3))

using BlockRegistration, BlockRegistrationScheduler, Images, JLD, StaticArrays
# using CUDArt
using Base.Test

for (sz, mxshift, gridsize, so) in (((150,), (6,), (4,), ["x"]),
                                    ((150,140), (6,5), (4,3), ["x", "y"]),
                                    ((150,140,15), (6,5,2), (4,3,3), ["x", "y", "z"]))
    N = length(sz)
    knots = ntuple(d->linspace(1,sz[d],gridsize[d]), N)
    # Create a fixed image with a uniform background except for one bright
    # pixel at the center of each aperture.
    # We create as a larger image to test that having "padding"
    # with valid values gets exploited by the mismatch computation.
    padsz = [mxshift...]+ceil(Int, [sz...]/2)  # excessive, but who cares?
    fullsz = (([sz...]+2padsz...)...,)
    fixed_full = SharedArray(Float64, fullsz)
    fill!(fixed_full, 1)
    fixed = sub(fixed_full, ntuple(d->padsz[d]+1:padsz[d]+sz[d], N))
    for I in CartesianRange(gridsize)
        c = [round(Int, knots[d][I[d]]) for d = 1:N]
        fixed[c...] = 2
    end

    # The moving image displaces the bright pixel.
    # Make sure the shift won't move bright pixel into a different aperture
#    @assert all([sz...]./[gridsize...] .> 4*[mxshift...])
    movingsz = (fullsz..., nt)
    moving_full = SharedArray(Float64, movingsz)
    fill!(moving_full, 1)
    moving = sub(moving_full, (ntuple(d->padsz[d]+1:padsz[d]+sz[d], N)..., :))
    displacements = Array(NTuple{N,Array{Int,N}}, nt)
    for t = 1:nt
        disp = ntuple(d->rand(-mxshift[d]+1:mxshift[d]-1, gridsize), N)
        displacements[t] = disp
        for I in CartesianRange(gridsize)
            c = [round(Int, knots[d][I[d]])+disp[d][I] for d = 1:N]
            moving[c..., t] = 2
        end
    end
    img = Image(moving, timedim=N+1, spatialorder=so)

    ### Compute the mismatch
    baseout = tempname()
    fnmm = string(baseout, ".mm")
    # devs = 0:2
    # wait_free(devs)
    # algorithm = AperturesMismatch[AperturesMismatch(fixed, knots, mxshift; dev=devs[i],pid=wpids[i],correctbias=false) for i = 1:length(wpids)]
    algorithm = AperturesMismatch[AperturesMismatch(fixed, knots, mxshift; pid=wpids[i], correctbias=false) for i = 1:length(wpids)]
    mon = monitor(algorithm, (:Es, :cs, :Qs, :mmis))

    driver(fnmm, algorithm, img, mon)

    # Append important extra information to the file
    jldopen(fnmm, "r+") do io
        write(io, "knots", knots)
    end

    ### Read the mismatch
    Es, cs, Qs, knots, mmis = jldopen(fnmm, mmaparrays=true) do file
        read(file, "Es"), read(file, "cs"), read(file, "Qs"), read(file, "knots"), read(file, "mmis")
    end;

    # Test that all pixels within an aperture were valid
    den = mmis[2,ntuple(d->Colon(), 2N+1)...]
    for i = 1:prod(size(den)[N+1:end])
        den1 = den[ntuple(d->Colon(), N)..., i]
        @test (1+1e-8)*minimum(den1) > maximum(den1)
    end
    # Test that a perfect match was found in each aperture
    @test all(Es .< 1e-12) # if this fails, make sure checkbias=false above
    # Test that the displacement is correct
    for t = 1:nt
        for d = 1:N
            css = slice(cs, d, ntuple(d->Colon(), N)..., t)
            @test css == displacements[t][d]
        end
    end

    # Test initialization
    csr = reinterpret(Vec{N,Float64}, cs, Base.tail(size(cs)))
    Qsr = reinterpret(Mat{N,N,Float64}, Qs, Base.tail(Base.tail(size(Qs))))
    ap = AffinePenalty(knots, 0.0)
    ur, _ = initial_deformation(ap, csr, Qsr)
    u = reinterpret(Float64, ur, (N, size(ur)...))
    @test maxabs(u-cs) <= 1e-3
end

rmprocs(wpids)
