
using DelimitedFiles
using LinearAlgebra

fp = "/Users/ruairidh/.julia/dev/SequenceSpace/"

const agrid = readdlm(fp * "tempdata/paper_ks/a_grid.csv", ',', Float64)[:, 1]
const egrid = readdlm(fp * "tempdata/paper_ks/e_grid.csv", ',', Float64)[:, 1]
const Qt    = readdlm(fp * "tempdata/paper_ks/Pi.csv", ',', Float64) |> permutedims

invariant_dist = Qt^1000 * ones((length(egrid))) / length(egrid)
@assert dot(egrid, invariant_dist) ≈ 1

const β = 0.981952788 # household discount factor
const δ = 0.025 # depreciation
const α = 0.11 # capital parameters

const T = 300 # time horizon for sequence space jacobian

# U.S. output data to use for estimation 
output_data = readdlm(fp * "data/output_detrended.csv", ',', Float64)

firms_block = @simpleblock [:k, :z] [:r, :w, :y] [3.0, 1.0] (k, z) -> begin
    r = α * z[0] * k[-1] ^ (α - 1) - δ
    w = (1 - α) * z[0] * k[-1] ^ α
    y = z[0] * k[-1] ^ α
    return [r, w, y]
end

market_clearing_block = @simpleblock [:𝓀, :k] [:h] [3.0, 3.0] (𝓀, k) -> [𝓀[0] - k[0]]

include(fp * "examples/ks_household.jl")

hh_block = HetBlock(
    [:r, :w], [:𝓀, :c], 300, # inputs, outputs, and time horizon for jacobian
    backwards_iterate!, # backwards iteration function for jacobian
    kssteadystate!, # steady state function
    makecache, # function creating temporary memory cache

    # initial steady-state inputs
    [0.005, 1 - α], 
    # value function guess
    [(0.1 + a) ^ (-1) for a in agrid, e in egrid],
    # stationary distribution guess
    (repeat(invariant_dist, inner = length(agrid)) ./ 
    (sum(invariant_dist) * length(agrid)))
)

rss, wss, kss, zss = 0.01, 1-α, α / (0.01 + δ), (α / (0.01 + δ))^(-α)
updatesteadystate!(firms_block, [kss, zss])
updatesteadystate!(market_clearing_block, [kss, kss])
updatesteadystate!(hh_block, [rss, wss], maxiter=3000, tol=1e-12) # high precision

Jfirms = jacobian(firms_block)
Jhh  = jacobian(hh_block)

model = ModelGraph(
    [hh_block, firms_block, market_clearing_block],
    [:k], [:z], [:h], [:k, :z, :y, :r, :w]
)
updatepartialJacobians!(model)
Gs = geneqjacobians(model, Val(:forward))

# ARMA(1, 1)
function updateshockMA!(ma_coefs, Ω)
    # ma_coefs is T×1 matrix
    ρ, θ, σ = Ω

    ma_coefs[1, 1] = σ # 1
    ma_coefs[2, 1] = (ρ - θ) * σ
    for s in 3:T
        ma_coefs[s, 1] = ρ * ma_coefs[s-1, 1]
    end
    return ma_coefs
end

arma_likelihood = make_likelihood(
    updateshockMA!, 1, output_data, [:y], Gs
)