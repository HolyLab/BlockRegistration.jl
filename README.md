# BlockRegistration

[![Build Status](https://magnum.travis-ci.com/HolyLab/BlockRegistration.svg?token=su2Bzyut6KvcqmScAAsj&branch=master)](https://magnum.travis-ci.com/HolyLab/BlockRegistration)

## Installation

Install the package from the Julia prompt with
```jl
Pkg.clone("url")
```
where `url` is the "clone URL" on this page.

Next, build it with the following statements:
```jl
Pkg.build("BlockRegistration")
using BlockRegistration
import RegisterMismatch
import RegisterMismatchCuda
```

The last two lines are needed to precompile the modules
`RegisterMismatch` and `RegisterMismatchCuda`. `BlockRegistration`
does not include these modules because they are not intended to be
used at the same time; either one or the other gets loaded depending
on choices you make.

You may want to execute those "building" lines every time
`BlockRegistration` updates.
