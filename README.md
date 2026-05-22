# BlockRegistration.jl

[![CI](https://github.com/HolyLab/BlockRegistration.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/BlockRegistration.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/HolyLab/BlockRegistration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/BlockRegistration.jl)

BlockRegistration is designed to perform non-rigid image registration---specifically, motion correction---for a time series of images.
BlockRegistration works on images/movies with an arbitrary number of spatial dimensions and a single temporal dimension.
Because it was designed to handle the very large data sets produced by light sheet microscopy,
it prioritizes speed above all else.

BlockRegistration is written in the [Julia programming language](https://julialang.org/).
Installation is greatly facilitated by using the [HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry#usage); see the documentation link below for details and demonstrations.

[![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/BlockRegistration.jl/stable)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/BlockRegistration.jl/dev)
