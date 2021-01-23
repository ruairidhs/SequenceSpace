using SequenceSpace

using Test
using Interpolations # for fastinterp

@testset "SequenceSpace" begin
    
    @testset "fastinterp!" begin
        function gen_interp_data(T)
            nodes = sort(rand(T) * 10)
            vals  = log.(nodes)
            xs    = sort(rand(T*10) * 10)
            return nodes, vals, xs
        end

        nodes, vals, xs = gen_interp_data(100)
        itp  = interpolate((nodes,), vals, Gridded(Linear()))
        etpl = extrapolate(itp, Line())
        res  = similar(xs)

        fastinterp!(res, xs, nodes, vals)
        @test etpl.(xs) ≈ res

    end

    @testset "Krusell & Smith" begin

        include("../src/krusell-smith.jl") # includes required packages

        @testset "Sparse Block" begin
            j = jacobian(firms_block, ([kss, zss],))
            # Compare to analytic result
            @test Matrix(j[(:z, :r)]) ≈ diagm(repeat([α * kss^(α-1)], T))
            @test Matrix(j[(:z, :w)]) ≈ diagm(repeat([(1-α)*kss^α], T))
            @test Matrix(j[(:k, :r)]) ≈ diagm(-1 => repeat([α*(α-1)*zss*kss^(α-2)], T-1))
            @test Matrix(j[(:k, :w)]) ≈ diagm(-1 => repeat([α*(1-α)*zss*kss^(α-1)], T-1))
        end

        @testset "Het Agents Block" begin
            j = jacobian(haBlock, (xss, vss, dss, yss, pss, Λss))
            # Regression test against previous result (which is visually similar to paper)
            @test j[(:r, :𝓀)] ≈ readdlm("../tempdata/jrk.csv", ',', Float64)
        end
        
    end

end