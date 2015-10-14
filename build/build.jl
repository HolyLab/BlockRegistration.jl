if Base.find_in_path("RFFT") == nothing
    Pkg.clone("git@github.com:HolyLab/RFFT.jl.git")
end
Pkg.checkout("FixedSizeArrays")
Pkg.checkout("AffineTransforms")
