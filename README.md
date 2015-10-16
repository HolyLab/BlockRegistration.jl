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

## Usage

Begin with `using BlockRegistration`.

We need more instructions here. In the meantime, after `using
BlockRegistration` you can invoke help with `?` followed by the name
of a module; for example, `?RegisterCore` will provide an overview of
the `RegisterCore` module.

Also consider looking at the code in `test` as an example of how to
use these modules.

### Troubleshooting

This "package" is unconventional in having many stand-alone modules in
a single repository; this design is used in part because we are
limited by GitHub to a maximum of 20 private repositories. Certain
complications arise as a consequence, of which the most common is
that, depending on usage patterns, julia may complain about not being
able to find certain modules.  In (hopefully) all cases, such errors
can be avoided if you begin your usage with `using BlockRegistration`:
the `BlockRegistration` module adds the proper path to julia's
`LOAD_PATH` variable, making it possible to find the remaining
modules.

This same issue arises particularly when using multiple processes via
`BlockRegistrationScheduler`, and is particularly problematic for
`RegisterMismatch` and `RegisterMismatchCuda` since these do not get
built when you type `using BlockRegistration`. If you run into
problems with julia unable to build or execute certain functions,
please try the building steps listed above under "Installation" in a
single-process julia session. Then start a fresh julia session and try
your task again.
