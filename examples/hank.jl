using SequenceSpace

using DelimitedFiles # to read data
using StaticArrays # for transfer rules
using LinearAlgebra # for household
using Plots # for plotting
using ColorSchemes # for plotting
using Distributions # for estimation
using Optim # for posterior mode

#region ===== Import parameters =====

# Households
const agrid = readdlm("tempdata/paper_hank/hank_a_grid.csv", ',', Float64)[:, 1]
const egrid = readdlm("tempdata/paper_hank/hank_e_grid.csv", ',', Float64)[:, 1]
const Qt    = readdlm("tempdata/paper_hank/hank_Pi.csv", ',', Float64) |> permutedims

invariant_dist = Qt^1000 * (ones(axes(Qt, 1)) / size(Qt, 1))

const drule = SVector{length(egrid)}(egrid ./ dot(egrid, invariant_dist))
const trule = SVector{length(egrid)}(egrid ./ dot(egrid, invariant_dist))

const β = 0.982 # discount factor
const φ = 0.786 # disutility of labour
const σ = 2.0 # inverse IES
const ν = 2.0 # inverse Frisch elasticity
# b̲ assumed = 0

# Firms
const μss = 1.2 # steady state markup
const κ   = 0.1 # slope of Phillips curve
const Z   = 1.0 # productivity 

# Policy
const B = 5.6 # bond supply 
const Gss = 0   # steady state government spending 
const ϕ = 1.5 # Taylor rule inflation coefficient
const ϕy = 0.0 # Taylor rule output coefficient

# Additional steady states 
Yss, Πss, wss, rstarss, rss, Dss, τss = 1.0, 0.0, 0.83, 0.005, 0.005, 0.1666, 0.28

const T = 300 # time horizon for sequence space jacobian

# US data, 1966 - 2004
const observed_data = readdlm("data/hank_data.csv", ',', Float64)

# Dictionary:
#   - Unknowns:
#       - w: wages
#       - Y: output
#       - Π: inflation
#   - Exogenous:
#       - Z: productivity
#       - r*: Taylor rule parameter
#       - G: government spending
#       - μ: markups
#   - Intermediates:
#       - r: interest rate
#       - d: dividends
#       - τ: tax
#       - 𝒜: household asset demand
#       - 𝒩: household effective labour supply
#       - hours: actual hours worked (not scaled by productivity)


#region ===== Simple blocks =====

firms_block = @simpleblock(
    [:Y, :μ, :Π, :w], [:N, :d],
    [Yss, μss, Πss, wss], (Y, μ, Π, w) -> begin
        N = Y[0] / Z # eq 56
        ψ = (μ[0] / (μ[0] - 1)) * (1 / 2κ) * (log(1 + Π[0]))^2 * Y[0] # eq 57
        d = Y[0] - w[0] * N - ψ # eq 58
        return [N, d]
    end
)

taylor_rule = @simpleblock [:rs, :Π, :Y] [:i] [rstarss, Πss, Yss] (rs, Π, Y) -> begin
    return [rs[0] + ϕ*Π[0] + ϕy * (Y[-1] - Yss)]
end
fisher_eq = @simpleblock [:i, :Π] [:r] [rss, Πss] (i, Π) -> [(1+i[-1])/(1+Π[0]) - 1]

fiscal_block = @simpleblock [:r, :G] [:τ] [rss, Gss] (r, G) -> begin
    return [r[0] * B + G[0]]
end

phillips_curve = @simpleblock(
    [:Π, :w, :μ, :r, :Y], [:hP],
    [Πss, wss, μss, rss, Yss], (Π, w, μ, r, Y) -> begin
        lhs = log(1 + Π[0])
        rhs = κ*(w[0]/Z - 1/μ[0])+(1/(1+r[1])) * Y[1]/Y[0] * log(1 + Π[1])
        return [lhs - rhs]
    end
)

capital_clearing = @simpleblock [:𝒜] [:hA] [B] 𝒜 -> [𝒜[0] - B]
labour_clearing = @simpleblock [:𝒩, :N] [:hN] [1.0, 1.0] (𝒩, N) -> [𝒩[0] - N[0]]

#endregion

#region ===== Household block =====

# The heterogenous agent household block maps (r, w, d, τ) => (𝒜, hours, 𝒩, C)

# File containing iteration and steady-state functions for the household block
include("HANK_household.jl")

hh_block = HetBlock(
    [:r, :w, :d, :τ], [:𝒜, :n, :𝒩, :C], 300,
    backwards_iterate!,
    hanksteadystate!,
    makecache,

    # initial steady state inputs
    [rss, wss, Dss, τss],
    # value function guess
    [(1+rss) * (0.1 * ((1 + rss) * agrid[ai] + 0.1))^(-1/σ)
     for ai in eachindex(agrid), ei in eachindex(egrid)],
    # stationary distribution guess
    (repeat(invariant_dist, inner = length(agrid)) ./
    (sum(invariant_dist) * length(agrid)))
)

#endregion

#region ===== Calibration =====

updatesteadystate!(hh_block, [rss, wss, Dss, τss])

#endregion

#region ===== Partial jacobians =====

Jhh = jacobian(hh_block)

function plot_jacobian_columns(mat, cols; title="")
    p = hline([0], ls = :dash, color = :black, labels = false)
    plot!(p,
        0:size(mat, 1)-1,
        mat[:, 1 .+ cols],
        labels = [
            "\$s=$col\$" for col in cols
        ] |> permutedims,
        lw = 1.5, grid = :y, framestyle = :box,
        xwiden = false, xlabel = "\$\\textrm{Time } t\$",
        palette = ColorSchemes.tableau_orange_blue,
        title = title
        )
    return p
end

plot_jacobian_columns(Jhh[(:τ, :𝒩)], 0:25:100)


#endregion

#region ===== General equilibrium jacobians =====

model = ModelGraph(
    [hh_block, firms_block, fisher_eq, taylor_rule,
     fiscal_block, phillips_curve, capital_clearing, labour_clearing],
    [:w, :Y, :Π], # Unknowns
    [:rs, :G, :μ], # exogenous shocks
    [:hP, :hA, :hN], # targets
    [] # shocks to initialize as sparse
)

plotgraph(model)
updatepartialJacobians!(model)

# each T columns corresponds to one of the shocks
Gs = geneqjacobians(model, Val(:forward))

# plot_jacobian_columns(Gs[:Y][1:200, 601:900], 0:25:100)

#endregion

#region ===== Bayesian estimation =====


# rs, G, μ each follow AR(1) with:
# persistences and standard deviations Ω = ρrs, ρG, ρμ, σrs, σG, σμ
function updateshockMA!(ma_coefs, Ω)
    # updates the exogenous shock MA coefficients
    ρs = (Ω[1], Ω[2], Ω[3])
    σs = (Ω[4], Ω[5], Ω[6])
    for zi in 1:3 # for each shock
        ma_coefs[1, zi] = 1.0
        for s in 2:T
            ma_coefs[s, zi] = ρs[zi] * ma_coefs[s-1, zi]
        end
        ma_coefs .*= σs[zi]
    end
    return ma_coefs
end

likelihood = make_likelihood(
    updateshockMA!, 3, observed_data, [:Y, :Π, :i], Gs
)

priors = [
    Beta(2.626, 2.626), Beta(2.626, 2.626), Beta(2.626, 2.626),
    InverseGamma(2.01, 0.404), InverseGamma(2.01, 0.404), InverseGamma(2.01, 0.404)
]
function priordensity(Ω, priors)
    sum(logpdf(priors[i], Ω[i]) for i in eachindex(Ω))
end

function insupport(Ω)
    ρs = (Ω[1], Ω[2], Ω[3]) # ∈ (-1, 1)
    σs = (Ω[4], Ω[5], Ω[6]) # > 0
    return all(-1.0 .< ρs .< 1.0) && all(σs .> 0.0)
end

posteriordensity(Ω) = insupport(Ω) ? likelihood(Ω) + priordensity(Ω, priors) : -Inf

# posterior mode
posteriormode = optimize(Ω -> -posteriordensity(Ω), mode.(priors), LBFGS())
posteriormode.minimizer