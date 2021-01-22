using SequenceSpace
using Test

# ===== Test fastinterp.jl =====
using Interpolations

function gen_interp_data(T)
    nodes = sort(rand(T) * 10)
    vals  = log.(nodes)
    xs    = sort(rand(T*10) * 10)
    return nodes, vals, xs
end

# Checks that output from fastinterp! matches Interpolations external package
@testset "fastinterp.jl" begin
    nodes, vals, xs = gen_interp_data(100)
    itp  = interpolate((nodes,), vals, Gridded(Linear()))
    etpl = extrapolate(itp, Line())
    res  = similar(xs)

    SequenceSpace.fastinterp!(res, xs, nodes, vals)
    @test etpl.(xs) ≈ res

end