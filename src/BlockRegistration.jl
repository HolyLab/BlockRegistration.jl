module BlockRegistration

thisdir = splitdir(@__FILE__)[1]
if !any(LOAD_PATH .== thisdir)
    push!(LOAD_PATH, thisdir)
end

end
