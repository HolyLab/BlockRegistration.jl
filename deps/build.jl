if Base.find_in_path("RFFT") == nothing
    Pkg.clone("git@github.com:HolyLab/RFFT.jl.git")
end
Pkg.checkout("FixedSizeArrays")
Pkg.checkout("Optim", "teh/constrained")

basedir = splitdir(splitdir(@__FILE__)[1])[1]
cd(joinpath(basedir, "src")) do
    run(`make`)
end
