__precompile__(false)  # can't precompile because of path issues

module BlockRegistration

using Reexport

thisdir = splitdir(@__FILE__)[1]
if !any(LOAD_PATH .== thisdir)
    push!(LOAD_PATH, thisdir)
end

@reexport using CenterIndexedArrays
@reexport using RegisterCore
@reexport using RegisterDeformation
@reexport using RegisterFit
@reexport using RegisterOptimize
@reexport using RegisterPenalty

# We can't pop!(LOAD_PATH) because of RegisterGUI & the
# RegisterMismatch/RegisterMismatchCuda issue.

end
