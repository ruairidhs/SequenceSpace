start = time() # time the whole file

using SequenceSpace

using DelimitedFiles # to read data
using StaticArrays # for transfer rules
using LinearAlgebra # for household
using Optim # for calibration
using Plots # for plotting
using ColorSchemes # for plotting
using Distributions # for estimation
using Optim # for posterior mode
using BenchmarkTools
using Plots.PlotMeasures

#region ===== Import parameters =====

println("Load data...")

# Grids and exogenous transition matrix
const agrid = readdlm("tempdata/paper_hank/hank_a_grid.csv", ',', Float64)[:, 1]
const egrid = readdlm("tempdata/paper_hank/hank_e_grid.csv", ',', Float64)[:, 1]
const Qt    = readdlm("tempdata/paper_hank/hank_Pi.csv", ',', Float64) |> permutedims

invariant_dist = Qt^1000 * (ones(axes(Qt, 1)) / size(Qt, 1))

const drule = SVector{length(egrid)}(egrid ./ dot(egrid, invariant_dist))
const trule = SVector{length(egrid)}(egrid ./ dot(egrid, invariant_dist))

const σ = 2.0 # inverse IES
const ν = 2.0 # inverse Frisch elasticity
# b̲ assumed = 0

const B = 5.6  # bond supply
const κ = 0.1  # slope of Phillips curve
const Z = 1.0  # Productivity (fixed)
const ϕ = 1.5  # Taylor rule inflation coefficient
const ϕy = 0.0 # Taylor rule output coefficient

const μss = 1.2 # Steady state markup
const Πss = 0.0 # Steady state inflation
const Gss = 0.0 # steady state gov spending 
const rss = 0.005 # steady state real interest rate (calibration target)
const Nss = 1.0 # steady state labour supply (calibration target)
const Yss  = Z * Nss # output
const rstarss = rss # taylor rule shock

const T = 300 # time horizon for sequence space jacobian

# US data, 1966 - 2004
observed_data = readdlm("data/hank_data.csv", ',', Float64)

# Dictionary:
#   - Unknowns:
#       - w: wages
#       - Y: output
#       - Π: inflation
#   - Exogenous shocks:
#       - r*: Taylor rule parameter
#       - G: government spending
#       - μ: markups
#   - Intermediates:
#       - r: interest rate
#       - d: dividends
#       - τ: tax
#       - i: nominal interest rate
#       - 𝒜: household asset demand
#       - 𝒩: household effective labour supply (scaled by productivity)
#       - n: actual hours worked (not scaled by productivity)
#       - N: firm labour demand

#region ===== Household block and calibration =====
prinln("Calibration...")
# The heterogenous agent household block maps (r, w, d, τ) => (𝒜, n, 𝒩, C)

# File containing iteration and steady-state functions for the household block
include("HANK_household.jl")

# We calibrate β and φ to target r and 𝒩 in steady state
# the following function performs the calibration and returns a 
# HetBlock object set up at the steady state
function calibrate_hh(rtarget, Ntarget, guess)

    # This is slow, probably many improvements but not the focus

    # Analytic solultions
    @assert Πss == 0 "Only solved for Πss = 0 so far"    
    wss = Z / μss
    dss = Z * Ntarget - wss * Ntarget
    τss = rtarget * B + Gss
    xss = [rtarget, wss, dss, τss]

    initv = [ (1+rtarget) * (0.1 * ((1 + rtarget) * agrid[ai] + 0.1))^(-1/σ)
                for ai in eachindex(agrid), ei in eachindex(egrid)]

    initd = (repeat(invariant_dist, inner = length(agrid)) ./
            (sum(invariant_dist) * length(agrid)))
    
    yss  = zeros(length(agrid) * length(egrid), 4) # 4 for 4 outcomes
    tmps = makecache(Float64)
    
    function distance(ps, initv, initd, yss, tmps)
        res_value = steady_state_value(initv, xss, ps; maxiter=5000, tol=1e-14)
        @assert res_value.converged "Value function did not converge, parameters $ps"
        res_dist, = steady_state_distribution(initd, xss, res_value.value, ps; maxiter=5000, tol=1e-14)
        @assert res_dist.converged "Distribution did not converge, parameters $ps"

        transfers = xss[3] * drule - xss[4] * trule
        updateEGMvars!(tmps, res_value.value, xss[1], xss[2], transfers, ps)
        inner_update_outcomes!(yss, tmps[4], tmps[3], tmps[2])
        fixconstrained!(tmps[1], yss, tmps[4], xss[1], xss[2], transfers, ps[2])
        targets = (transpose(yss) * res_dist.value)[[1, 3]]

        return (targets[1] - B)^2 + (targets[2] - Ntarget)^2 # assets and labour error
    end

    βborder, φborder = 0.003, 0.1 # maximum allowed distance to theoretical constraint
    insupport(ps) = ps[1] < ((1 / (1 + rtarget)) - βborder) && ps[2] - φborder > 0 # not too patient and don't enjoy working

    optimization_result = Optim.optimize(
        ps -> insupport(ps) ? distance(ps, initv, initd, yss, tmps) : Inf,
        guess, NelderMead(), Optim.Options(time_limit = 100.0)
    )
    @assert Optim.minimum(optimization_result) < 1e-8 "Optimization not close enough"
    ps = Optim.minimizer(optimization_result)

    hhsteadystate!(ha, x; updateΛss=true, tol=1e-8, maxiter = 2000) = hanksteadystate!(
        ha, x, ps, updateΛss=updateΛss, tol=tol, maxiter=maxiter
    )
    # create household block with these parameter values
    hh = HetBlock(
        [:r, :w, :d, :τ], [:𝒜, :n, :𝒩, :C], 300,
        (vf, Y, dist, dist0, xt, tmps) -> backwards_iterate!(vf, Y, dist, dist0, xt, tmps, ps),
        hhsteadystate!,
        makecache,
        [rtarget, wss, dss, τss],
        initv, initd
    )
    updatesteadystate!(hh, [rtarget, wss, dss, τss], tol=1e-14) # update the steady state to high accuracy
    return (
        hetblock = hh,
        β = ps[1], φ = ps[2],
        rss = rtarget,
        wss = wss,
        dss = dss,
        τss = τss,
    )
end

res = calibrate_hh(rss, Nss, [0.95, 0.8])

hh_block, βcal, φcal, rcal, wcal, dcal, τcal = res

println("Calibration: β = $βcal; φ = $φcal")

# set wage, dividends and taxes from the calibration
const wss = wcal
const dss = dcal
const τss = τcal

#endregion

#region ===== Simple blocks =====

# Firm labour demand and dividend payments
firms_block = @simpleblock(
    [:Y, :μ, :Π, :w], [:N, :d],
    [Yss, μss, Πss, wss], (Y, μ, Π, w) -> begin
        N = Y[0] / Z # eq 56
        d = Y[0] - w[0] * (Y[0] / Z) - (μ[0] / (μ[0]-1)) / (2κ) * log(1+Π[0])^2 * Y[0]
        return [N, d]
    end
)

# Monetary policy
taylor_rule = @simpleblock [:rs, :Π, :Y] [:i] [rstarss, Πss, Yss] (rs, Π, Y) -> begin
    i = rs[0] + ϕ * Π[0] + ϕy * (Y[0] - Yss)
    return [i]
end
fisher_eq = @simpleblock [:i, :Π] [:r] [rss, Πss] (i, Π) -> begin
    r = (1 + i[-1]) / (1 + Π[0]) - 1
    return [r]
end

# Government (sets taxes)
fiscal_block = @simpleblock [:r, :G] [:τ] [rss, Gss] (r, G) -> begin
    return [r[0] * B + G[0]]
end

# Phillips curve (from firm problem)
phillips_curve = @simpleblock(
    [:Π, :w, :μ, :r, :Y], [:hP],
    [Πss, wss, μss, rss, Yss], (Π, w, μ, r, Y) -> begin
        lhs = log(1 + Π[0])
        rhs = κ*(w[0]/Z - 1/μ[0])+(1/(1+r[1])) * Y[1]/Y[0] * log(1 + Π[1])
        return [rhs - lhs]
    end
)
# Market clearing
capital_clearing = @simpleblock [:𝒜] [:hA] [B] 𝒜 -> [𝒜[0] - B]
labour_clearing = @simpleblock [:𝒩, :N] [:hN] [Nss, Nss] (𝒩, N) -> [𝒩[0] - N[0]]

#region ===== Partial jacobian =====
println("Jacobians...")
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

# Jacobian is similar to Krusell & Smith, except we can
# also examine response to transfers and effect on endogenous
# labour supply
plot_jacobian_columns(Jhh[(:τ, :𝒩)], 0:25:100)

#endregion

#region ===== General equilibrium jacobians =====

model = ModelGraph(
    [hh_block, firms_block, fisher_eq, taylor_rule,
     fiscal_block, phillips_curve,
     capital_clearing, labour_clearing],
    [:w, :Y, :Π], # Unknowns
    [:rs, :G, :μ], # exogenous shocks
    [:hP, :hA, :hN], # targets
    [:μ, :Π, :N, :i, :rs, :G,
     :r, :w, :d, :τ, :Y] # variables to initialize as sparse
)

plotgraph(model)
updatepartialJacobians!(model)

# each T columns corresponds to one of the shocks
# in the order specifed when creating the model: rstar, G, μ
Gs = geneqjacobians(model, Val(:forward))
plot(
    plot_jacobian_columns(Gs[:Π][:, 1:T], 0:25:100),
    layout = (1, 1), size=(600,400),
    bottom_margin = 4mm
)
savefig("./examples/hank_gepirs.pdf")

#endregion

#region ===== Bayesian estimation =====
println("Estimation...")
# rs, G, μ each follow AR(1) with:
# persistences and standard deviations Ω = ρrs, ρG, ρμ, σrs, σG, σμ
function updateshockMA!(ma_coefs, Ω)
    # ma_coefs is T×3 matrix
    # updates the exogenous shock MA coefficients
    ρs = (Ω[1], Ω[2], Ω[3])
    σs = (Ω[4], Ω[5], Ω[6])
    for zi in 1:3 # for each shock
        ma_coefs[1, zi] = σs[zi]
        for s in 2:T
            ma_coefs[s, zi] = ρs[zi] * ma_coefs[s-1, zi]
        end
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
    ρs = (Ω[1], Ω[2], Ω[3]) # ∈ (0, 1)
    σs = (Ω[4], Ω[5], Ω[6]) # > 0
    return all(0.0 .< ρs .< 1.0) && all(σs .> 0.0)
end

posteriordensity(Ω) = insupport(Ω) ? likelihood(Ω) + priordensity(Ω, priors) : -Inf

# posterior mode
function get_posterior_mode(posterior, initial_guess; time_limit=100, alg=LBFGS())
    optimize(Ω -> -posterior(Ω), initial_guess, alg, 
    Optim.Options(time_limit=time_limit)) 
end
res = get_posterior_mode(posteriordensity, mean.(priors))

function mcmc(posterior, draws, posteriormode, scaling_factor)
    posteriormodeHessian = FiniteDiff.finite_difference_hessian(
        Ω -> -posterior(Ω), posteriormode
    )
    model = DensityModel(posterior)
    proposaldensity = MvNormal(scaling_factor * inv(posteriormodeHessian))
    spl   = RWMH(proposaldensity)

    return sample(
        model, spl, draws,
        param_names = ["ρ", "θ", "σ"],
        chain_type=Chains
    )
end

chain = mcmc(
    posteriordensity, 100, Optim.minimizer(res), 2.50
)

println("===== ESTIMATION ======")
println("=========================")
println("Parameters: ρ, θ, σ")
println("Posterior modes:")
for p in Optim.minimizer(res)
    println(p)
end

println("MCMC Chain Results:")
display(chain)

#region ===== Benchmarking =====

prinln("Benchmarking...")

# Fake news
fakenews = @benchmark jacobian($hh_block)

# General equilibrium jacobians
nt, nu, nx = length(model.eqvars), length(model.unknowns), length(model.exog)
Gsbenchmark = Dict( # initialize
    var => zeros(T, T * length(model.exog)) for var in model.vars
)
Hu = zeros(T * nt, T * nu)
Hx = zeros(T * nt, T * nx)
G  = zeros(T * nu, T * nx)

geneq_forw = @benchmark geneqjacobians!($Gsbenchmark, $G, $Hu, $Hx, $model, Val(:forward))

SequenceSpace.resetnodematrices!(model, [:h])
geneq_back = @benchmark geneqjacobians!($Gsbenchmark, $G, $Hu, $Hx, $model, Val(:backward))

# likelihood
ll_bm = @benchmark likelihood([0.6, 0.9, 0.9, 0.5, 0.5, 2.0])

# posterior mode
pm_bm = get_posterior_mode($posteriordensity, mode.($priors))

println("===== BENCHMARKING ======")
println("=========================")

println("FAKE NEWS")
display(fakenews)

println("GEN EQ FORWARD")
display(geneq_forw)

println("GEN EQ BACKWARD")
display(geneq_back)

println("LIKELIHOOD")
display(ll_bm)

println("POSTERIOR MODE")
display(pm_bm)

println("========================")

#endregion

println("File finished successfully")
elapsed = time() - start
println("Elapsed time: $elapsed seconds")
println("END")