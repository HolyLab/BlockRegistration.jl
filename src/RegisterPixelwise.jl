__precompile__()

module RegisterPixelwise

using RegisterDeformation, RegisterPenalty, ReverseDiff
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile

"""
`ϕ` is the total deformation (including composition)
"""
function penalty_pixelwise(ϕ::AbstractDeformation, dp::DeformationPenalty, fixed, moving)
    T = eltype(ϕ)
    val = zero(T)
    val = penalty!(nothing, dp, ϕ)
    val += penalty_pixelwise_data(ϕ, fixed, moving)
    convert(T, val)
end

function penalty_pixelwise(x::AbstractArray, knots, dp::DeformationPenalty, fixed, moving)
    ϕ = GridDeformation(x, knots)
    penalty_pixelwise(ϕ, dp, fixed, moving)
end

penalty_pixelwise_data(ϕ, fixed, moving) = penalty_pixelwise_data(fixed, WarpedArray(moving, ϕ))

@generated function penalty_pixelwise_data{_,N}(fixed::AbstractArray{_,N}, moving::WarpedArray)
    ϕxindexes = [:(I[$d]+ux[$d]) for d = 1:N]
    quote
        valid = 0
        mm = 0.0
        iter = CartesianRange(indices(moving))
        state = start(iter)
        for ux in eachvalue(src.ϕ.u)
            I, state = next(iter, state)
            val = moving.data[$(ϕxindexes...)]
            if isfinite(val)
                valid += 1
                diff = Float64(fixed[I] - val)
                mm += diff^2
            end
        end
        mm/valid
    end
end

function optimize_pixelwise!(ϕ, dp::DeformationPenalty, fixed, moving; stepsize = 1.0)
    x = RegisterDeformation.convert_from_fixed(ϕ.u)
    @assert pointer(x) == pointer(ϕ.u)
    g = similar(x)
    objective = x->penalty_pixelwise(x, ϕ.knots, dp, fixed, moving)
    f_tape = GradientTape(objective, (copy(x),))
    compiled_f_tape = compile(f_tape)
    ∇objective!(results, x) = gradient!(results, compiled_f_tape, x)
    p0 = p = objective(x)
    pold = oftype(p, Inf)
    while p < pold
        pold = p
        ∇objective!(g, x)
        gmax = mapreduce(abs, max, g)
        if gmax == 0 || !isfinite(gmax)
            break
        end
        s = eltype(g)(stepsize/gmax)
        xtrial = x .- s .* g  # step by 0.1 pixel in the longest direction
        p = objective(xtrial)
        if p < pold
            copy!(x, xtrial)
        end
    end
    ϕ, pold, p0
end

end
