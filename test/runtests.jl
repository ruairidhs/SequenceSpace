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

        # include("../src/krusell-smith.jl") # includes required packages
        include("krusell_smith_test.jl")

        @testset "Simple Block" begin
            # Compare to analytic result 
            @test Matrix(SequenceSpace.sparse(Jfirms[(:z, :r)], T)) ≈ diagm(repeat([α * kss^(α-1)], T))
            @test Matrix(SequenceSpace.sparse(Jfirms[(:z, :w)], T)) ≈ diagm(repeat([(1-α)*kss^α], T))
            @test Matrix(SequenceSpace.sparse(Jfirms[(:k, :r)], T)) ≈ diagm(-1 => repeat([α*(α-1)*zss*kss^(α-2)], T-1))
            @test Matrix(SequenceSpace.sparse(Jfirms[(:k, :w)], T)) ≈ diagm(-1 => repeat([α*(1-α)*zss*kss^(α-1)], T-1))
        end

        @testset "Het Agents Block" begin
            # Regression test against previous result (which is same as paper)
            @test Jhh[(:r, :𝓀)] ≈ readdlm("../tempdata/ks_regression/jrk.csv", ',', Float64)
        end
        
        @testset "Graphs" begin
            # Regression test
            nt, nu, nx = length(model.eqvars), length(model.unknowns), length(model.exog)
            Hu, Hx, G = zeros(model.T * nt, model.T * nu), zeros(model.T * nt, model.T * nx), zeros(model.T * nu, model.T * nx)
            # forward diff test
            SequenceSpace.fillG!(G, Hu, Hx, model, Val(:forward))
            oldG = readdlm("../tempdata/ks_regression/ksG.csv", ',', Float64)
            @test G ≈ oldG
            # backward diff test
            SequenceSpace.resetnodematrices!(model, [:h])
            fill!(G, 0)
            SequenceSpace.fillG!(G, Hu, Hx, model, Val(:backward))
            @test G ≈ oldG
        end

        @testset "Likelihood" begin
            # check that the likelihood works
            @test arma_likelihood([0.9, 0.1, 0.5]) ≈ 70.13047528997609
            @test arma_likelihood([0.5, 0.5, 2.0]) ≈ -139.20913155989157
        end
        
    end

end