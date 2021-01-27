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

        @testset "Simple Block" begin
            j = jacobian(firms_block)
            # Compare to analytic result 
            @test Matrix(SequenceSpace.sparse(j[(:z, :r)], T)) ≈ diagm(repeat([α * kss^(α-1)], T))
            @test Matrix(SequenceSpace.sparse(j[(:z, :w)], T)) ≈ diagm(repeat([(1-α)*kss^α], T))
            @test Matrix(SequenceSpace.sparse(j[(:k, :r)], T)) ≈ diagm(-1 => repeat([α*(α-1)*zss*kss^(α-2)], T-1))
            @test Matrix(SequenceSpace.sparse(j[(:k, :w)], T)) ≈ diagm(-1 => repeat([α*(1-α)*zss*kss^(α-1)], T-1))
        end

        @testset "Het Agents Block" begin
            j = jacobian(ha_block)
            # Regression test against previous result (which is same as paper)
            @test j[(:r, :𝓀)] ≈ readdlm("../tempdata/ks_regression/jrk.csv", ',', Float64)
        end
        
        @testset "Graphs" begin
            # Regression test
            @test makeG(mg) ≈ readdlm("../tempdata/ks_regression/ksG.csv", ',', Float64)
        end
        
    end

end