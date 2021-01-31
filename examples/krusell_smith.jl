
start = time() # time the whole file

using SequenceSpace # import functions I wrote

using DelimitedFiles # for reading in the parameters
using Optim # for calibration
using LinearAlgebra
using BenchmarkTools # for timing
using Plots
using ColorSchemes # for plotting
using Plots.PlotMeasures
using Distributions # for estimation
using FiniteDiff
using AdvancedMH
using MCMCChains
using StatsPlots

# Krusell & Smith model

println("Load data...")

#region ===== Import parameters =====

# For comparability, I import the same grids and parameters used in the paper

# file path to where grids are saved
fp = "/Users/ruairidh/.julia/dev/SequenceSpace/"

# Asset grid, exogenous productivity grid and transpose exogenous transition matrix
const agrid = readdlm(fp * "tempdata/paper_ks/a_grid.csv", ',', Float64)[:, 1]
const egrid = readdlm(fp * "tempdata/paper_ks/e_grid.csv", ',', Float64)[:, 1]
const Qt    = readdlm(fp * "tempdata/paper_ks/Pi.csv", ',', Float64) |> permutedims

# confirm that egrid and Qt are such that L is normalized to 1
invariant_dist = Qt^1000 * ones((length(egrid))) / length(egrid)
@assert dot(egrid, invariant_dist) ≈ 1

const β = 0.981952788 # household discount factor
const δ = 0.025 # depreciation
const α = 0.11 # capital parameters

const T = 300 # time horizon for sequence space jacobian

# U.S. output data to use for estimation 
output_data = readdlm(fp * "data/output_detrended.csv", ',', Float64)

#endregion =====

println("Simple block...")

#region ===== Simple blocks =====

#=
The model has two simple blocks:
    - Firms: k, z => r, w, y
    - Capital market clearing: 𝓀, k => h
=#

# k = 3.0 and z = 1.0 are initial guesses for the steady-state
firms_block = @simpleblock [:k, :z] [:r, :w, :y] [3.0, 1.0] (k, z) -> begin
    r = α * z[0] * k[-1] ^ (α - 1) - δ
    w = (1 - α) * z[0] * k[-1] ^ α
    y = z[0] * k[-1] ^ α
    return [r, w, y]
end

market_clearing_block = @simpleblock [:𝓀, :k] [:h] [3.0, 3.0] (𝓀, k) -> [𝓀[0] - k[0]]

#endregion

#region ===== Household block =====

println("Household block...")

# The heterogenous agent household block maps (r, w) -> (𝓀, c)

# File containing iteration functions for the household block
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

#endregion

#region ===== Check calibration =====

println("Calibration...")

# Solve for equilibrium steady-state r (should be 0.01)
function ks_r_error(r, ha)
    w = (1 - α) # implied by y normalized to 1
    # updates the household steady-state based on [r, w]
    updatesteadystate!(ha, [r, w], updateΛss=false)
    # first column of yss has savings for each point in the distribution
    agg_savings = @views dot(ha.yss[:, 1], ha.dss)
    capital_demand = α / (r + δ)

    return (agg_savings - capital_demand)^2
end

function calibrate_hh(ha)
    res = Optim.optimize(
        r -> ks_r_error(r, ha), 0.005, 0.015
    )
    @assert Optim.converged(res) "r did not converge!"

    rss = Optim.minimizer(res)
    wss = 1 - α
    kss = α / (rss + δ)
    zss = kss ^ (-α)
    return (r=rss, w=wss, k=kss, z=zss)
end

calres = calibrate_hh(hh_block)

# update the blocks with calibrated steady state
updatesteadystate!(firms_block, [calres.k, calres.z])
updatesteadystate!(market_clearing_block, [calres.k, calres.k])
updatesteadystate!(hh_block, [calres.r, calres.w], maxiter=3000, tol=1e-12) # high precision

#endregion

#region ===== Partial jacobians =====

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

plot(
    plot_jacobian_columns(Jhh[(:r, :𝓀)], 0:25:100),
    plot_jacobian_columns(Jhh[(:w, :𝓀)], 0:25:100),
    layout = (1, 2), size=(1200,400),
    bottom_margin=4mm
)
savefig("./examples/jhh.pdf")

#endregion

#region ===== General equilibrium jacobians =====

model = ModelGraph(
    [hh_block, firms_block, market_clearing_block],
    [:k], [:z], [:h], # Unknowns, exogenous, and equilibrium targets
    [:k, :z, :y, :r, :w]
)
updatepartialJacobians!(model)
Gs = geneqjacobians(model, Val(:forward))

plotgraph(model)

plot(
    plot_jacobian_columns(Gs[:y], 0:25:100),
    plot_jacobian_columns(Gs[:k], 0:25:100),
    layout = (1, 2), size=(1200,400),
    bottom_margin = 4mm
)
savefig("./examples/geky.pdf")

#endregion 

#region ===== Bayesian estimation =====

println("Estimation...")

# The general equilibrium jacobians can be used
# directly for estimation of parameters which do not 
# affect the steady-state.

# Estimate (1-ρL)zₜ = (1-θL)σϵₜ
# collect parameters as Ω = [ρ, θ, σ]

# First write a function mapping parameters into the 
# MA coefficients for the exogenous shock:
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

# can use this to replicate Figure 7 (b)
corrs = getcorrelations(
    :z, [:y, :c, :k], 
    updateshockMA!(ones(T, 1), [0.9, 0.0, 1.0]), Gs
)
plot(-50:50, corrs[250:350, :], labels = ["z" "y" "c" "k"])

# then the function make_likelihood will return a likelihood function
# over Ω given the observed data, the list of variables to which the data 
# correspond and the general equilibrium jacobians Gs
likelihood = make_likelihood(
    updateshockMA!, 1, output_data, [:y], Gs
)

# then set up priors to make a posterior density function
priors = [
    Beta(2.626, 2.626),
    Beta(2.626, 2.626),
    InverseGamma(2.01, 0.404)
]
function priordensity(Ω, priors)
    sum(logpdf(priors[i], Ω[i]) for i in eachindex(Ω))
end

insupport(Ω) = (-1.0 < Ω[1] < 1.0) && (-1.0 < Ω[2] < 1.0) && (Ω[3] > 0.0)

posteriordensity(Ω) = insupport(Ω) ? likelihood(Ω) + priordensity(Ω, priors) : -Inf

# find the posterior mode using optimization routine
function get_posterior_mode(posterior, initial_guess)
    optimize(Ω -> -posterior(Ω), initial_guess, LBFGS()) 
end
res = get_posterior_mode(posteriordensity, mode.(priors))
@assert Optim.converged(res) "Posterior mode did not converge"

# monte-carlo markov chain
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

#endregion

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
ll_bm = @benchmark likelihood([0.9, 0.1, 0.2])

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