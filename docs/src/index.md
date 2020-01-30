# BlockRegistration.jl

BlockRegistration is designed to perform non-rigid image registration--specifically, motion correction--for a time series of images.
It aligns images by attempting to minimize the mean square difference between images,
a strategy that does not require "control points" or ["features"](https://en.wikipedia.org/wiki/Image_registration#Intensity-based_vs_feature-based),
as BlockRegistration was designed to work with relatively low signal-to-noise
ratio images common in biomedical imaging.
BlockRegistration works on images/movies with an arbitrary number of spatial dimensions and a single temporal dimension.

Because it was designed to handle the very large data sets produced by light sheet microscopy,
it prioritizes speed over quality, primarily by modeling deformations as piecewise-constant over extended blocks of the image during its optimization phase.
It has several apparently-innovative features designed to increase the likelihood of
ending up near the global optimum, for example by initializing the deformation
via a quadratic approximation to the blockwise mismatch data.

Documentation is still a work in progress, and there is no publication yet,
but a few of the key concepts are described in the documentation for some of the dependent
modules (see below).

BlockRegistration is written in the [Julia programming language](https://julialang.org/).

## Installation

If you don't already have it, add [HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry#usage) as a registry.
Then from [package mode](https://julialang.github.io/Pkg.jl/v1/getting-started/) just do

```julia
pkg> add BlockRegistration RegisterMismatch
```

If you have GPUs available, you may also want to add `RegisterMismatchCuda`.
This requires that you have nvidia drivers installed on your system.

See also [BlockRegistrationScheduler](https://github.com/HolyLab/BlockRegistrationScheduler), which allows you to parallelize registration across worker processes.
It works for both CPU (`RegisterMismatch`) and GPU (`RegisterMismatchCuda`),
though for GPU you need a dedicated GPU card for each worker process.

## Orienting yourself

`BlockRegistration` is a meta-package that merely re-exports several lower-level modules,
of which the main ones likely to be of interest to users are (`*` indicates that the package
has additional documentation):

- [`*RegisterCore`](https://github.com/HolyLab/RegisterCore.jl): basic types and the overall framework
- [`*RegisterDeformation`](https://github.com/HolyLab/RegisterDeformation.jl): deformations (warps) of space
- [`RegisterMismatch`](https://github.com/HolyLab/RegisterMismatch.jl)/[`RegisterMismatchCuda`](https://github.com/HolyLab/RegisterMismatchCuda.jl): computing mismatch data from raw images
- [`RegisterFit`](https://github.com/HolyLab/RegisterFit.jl): approximating mismatch data with simple models
- [`RegisterPenalty`](https://github.com/HolyLab/RegisterPenalty.jl): regularized objective functions for use in optimization
- [`RegisterOptimize`](https://github.com/HolyLab/RegisterOptimize.jl): performing optimization to align images


When you're using the package, choose either CPU mode:

```julia
using BlockRegistration
using RegisterMismatch
```

or GPU mode:
```julia
using BlockRegistration
using RegisterMismatchCuda
```

You can invoke help with `?` followed by the name of a module; for
example, `?RegisterCore` will provide an overview of the
`RegisterCore` module (a good place to start if you're trying to get a
handle on the basic underpinnings).
You'll see a list of major functions in the module, and generally each of these has
its own documentation.
Given that "published" documentation is still a bit sparse,
also consider looking at the code in each package's
`test/` folder as an example of how to use these modules.
