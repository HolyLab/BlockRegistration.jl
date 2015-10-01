using BlockRegistration

function runtest(filename)
    println(filename)
    include(filename)
end

runtest("center_indexed_arrays.jl")
runtest("register_core.jl")
runtest("register_deformation.jl")
if !(isdefined(Main, :use_cuda) && Main.use_cuda==false) &&
        !isempty(Libdl.find_library(["libcudart", "cudart"], ["/usr/local/cuda"]))
    runtest("register_mismatch_cuda.jl")
end
runtest("register_mismatch.jl")
runtest("register_fit.jl")
runtest("register_penalty.jl")
runtest("register_optimize.jl")
