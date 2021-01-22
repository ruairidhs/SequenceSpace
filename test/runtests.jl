using SequenceSpace
using Test

# ===== fastinterp.jl =====
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

# ===== sparse block =====
using LinearAlgebra

include("../src/krusell-smith.jl")

@testset "krusell-smith firms" begin
    firms_block = SparseBlock([:k, :z], [:r, :w], (o, i) -> firms!(o, i, kss), T)
    j = jacobian(firms_block, [kss, zss])
    # Compare to analytic result
    @test all(Matrix(j[(:z, :r)]) .≈ diagm(repeat([α * kss^(α-1)], T)))
    @test all(Matrix(j[(:z, :w)]) .≈ diagm(repeat([(1-α)*kss^α], T)))
    @test all(Matrix(j[(:k, :r)]) .≈ diagm(-1 => repeat([α*(α-1)*zss*kss^(α-2)], T-1)))
    @test all(Matrix(j[(:k, :w)]) .≈ diagm(-1 => repeat([α*(1-α)*zss*kss^(α-1)], T-1)))
end