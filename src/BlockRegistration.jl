"""
    BlockRegistration

Umbrella package for block-based image registration. Re-exports all symbols from:

- `CenterIndexedArrays`: arrays with centered (zero-based) indexing
- `CachedInterpolations`: memoized interpolation objects
- `RegisterCore`: core data structures (`MismatchArray`, `NumDenom`) and preprocessing (`PreprocessSNF`, `highpass`)
- `RegisterDeformation`: deformation fields (`GridDeformation`), warping, and time-series utilities
- `RegisterFit`: mismatch fitting (`qfit`, `mms2fit!`) and affine extraction
- `RegisterOptimize`: full optimization loop (`RegisterOptimize.optimize!`) and penalty auto-tuning
- `RegisterPenalty`: regularization penalties (`AffinePenalty`, `DeformationPenalty`)
- `RegisterHindsight`: retrospective correction utilities
"""
module BlockRegistration

using Reexport

@reexport using CenterIndexedArrays
@reexport using CachedInterpolations
@reexport using RegisterCore
@reexport using RegisterDeformation
@reexport using RegisterFit
@reexport using RegisterOptimize
@reexport using RegisterPenalty
@reexport using RegisterHindsight

end
