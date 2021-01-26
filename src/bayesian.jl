
using Distributions
using FFTW
using DelimitedFiles
using BenchmarkTools
using Optim
using FiniteDiff
using AdvancedMH
using MCMCChains
using StatsPlots

include("krusell-smith.jl")

include("likelihood.jl")

# Ω = [ρ (AR coef), θ (MA coef), σ]

# ===== Set up the priors =====
priors = [
    Beta(2.626, 2.626),
    Beta(2.626, 2.626),
    InverseGamma(2.01, 0.404)
]

function priorpdf(Ω, priors)
    sum(logpdf(priors[i], Ω[i]) for i in eachindex(Ω))
end

insupport(Ω) = (0.0 < Ω[1] < 1.0) && (0.0 < Ω[2] < 1.0) && (Ω[3] > 0.0)

# ===== Set up the likelihood ===== 
function updateshockMA!(macoefs, Ω)
    # Ω = [ρ, θ, σ]
    # macoefs is T×1 matrix

    # always maccoefs[1, 1] = 1
    macoefs[1, 1] = 1
    macoefs[2, 1] = (Ω[1] - Ω[2])
    @inbounds for s in 3:T
        macoefs[s, 1] = Ω[1] * macoefs[s-1, 1]
    end
    macoefs .*= Ω[3]
    return macoefs
end

output_data = readdlm("data/output_detrended.csv", ',', Float64)

function make_likelihood_func(output_data, T, obsvars, Gs, priors)

    input_array = makeinput(1, 1, T)
    fft_cache = makefftcache(input_array)
    shockmat = ones(T, 1)
    Tobs = size(output_data, 1)

    function l(Ω)
        updateshockMA!(shockmat, Ω)
        updateMAcoefficients!(input_array, obsvars, shockmat, Gs)
        _likelihood(makeV(fastcov(input_array, fft_cache), Tobs), output_data) + priorpdf(Ω, priors)
    end
end

likelihood = make_likelihood_func(output_data, T, [:y], Gs, priors)
density(Ω) = insupport(Ω) ? likelihood(Ω) : -Inf

# find posterior mode
posteriormode = optimize(Ω -> -density(Ω), mode.(priors), LBFGS())
posteriormodeHessian = FiniteDiff.finite_difference_hessian(Ω->-density(Ω), posteriormode.minimizer)

model = DensityModel(density)

proposaldensity = MvNormal(inv(posteriormodeHessian))
spl = RWMH(proposaldensity)

chain = sample(model, spl, 100000; param_names = ["ρ", "θ", "σ"], chain_type=Chains)

end_chain = chain[50001:end]

#= Example
using Distributions
using AdvancedMH
using MCMCChains
using StatsPlots


const σ1 = 2.0
const σ2 = 3.0
const γ  = 0.8

sig = [σ1 γ; γ σ2]

Tobs = 100
data = rand(MvNormal(sig), Tobs)

# θ = [σ1, σ2, γ]
insupport(θ) = θ[1] >= 0 && θ[2] >= 0
dist(θ)    = MvNormal([θ[1] 0; 0 θ[2]])
density(θ) = insupport(θ) ? sum(logpdf(dist(θ), data[:, i]) for i in axes(data, 2)) : -Inf

model = DensityModel(density)
spl   = RWMH(MvNormal(2, 1))

chain = sample(model, spl, 100000; param_names = ["σ1", "σ2"], chain_type=Chains)
=#