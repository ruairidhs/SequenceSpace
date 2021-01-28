using SequenceSpace

using DelimitedFiles
using StaticArrays
using LinearAlgebra
using SparseArrays

#region Household block =====

# ===== Set up parameters =====

const β = 0.982
const φ = 0.786
const σ = 2.0
const ν = 2.0

const agrid = readdlm("tempdata/paper_hank/hank_a_grid.csv", ',', Float64)[:, 1] # as vector
egrid_raw = readdlm("tempdata/paper_hank/hank_e_grid.csv", ',', Float64)[:, 1] # as vector
const egrid = SVector{length(egrid_raw)}(egrid_raw)
Qt_raw   = readdlm("tempdata/paper_hank/hank_Pi.csv", ',', Float64) |> permutedims
const Qt = SMatrix{size(Qt_raw, 1), size(Qt_raw, 2)}(Qt_raw)

vss_python  = readdlm("tempdata/paper_hank/hank_Va.csv", ',', Float64) |> permutedims

# Steady state values
exog_invariant = Qt^1000 * (ones(axes(Qt, 1)) / size(Qt, 1))
drule_raw = egrid ./ dot(egrid, exog_invariant)
trule_raw = egrid ./ dot(egrid, exog_invariant)
const drule = SVector{length(egrid)}(drule_raw)
const trule = SVector{length(egrid)}(trule_raw)

# r = 0.005, w = 0.833, d = 0.166, t = 0.028
xss = [0.005, 0.833, 0.166, 0.028]

# ===== Household block =====

# v axes: (axes(agrid, 1), axes(egrid, 1))
function makecache(S)
    (   zeros(S, (axes(agrid, 1), axes(egrid, 1))),
        zeros(S, (axes(agrid, 1), axes(egrid, 1))),
        zeros(S, (axes(agrid, 1), axes(egrid, 1))),
        zeros(S, (axes(agrid, 1), axes(egrid, 1))),
    )
end

function updateEGMvars!(vars, vf, r, wage, transfers)

    # Given inputs xt and future values vf, fill vars matrices with
    # values needed for EGM. Each matrix has dims (a′, e)
    #  W  - continuation values β⋅∑v_(t+1)(a, e')⋅P(e, e')
    #  c  - current consumption from FOC
    #  n  - current labour from FOC
    #  a₋ - initial assets consistent with optimal choice a
    # ps wage instead of w to make it clear which is W and which is wage

    W, c, n, a₋ = vars

    # Set W
    mul!(W, vf, β .* Qt)
    # Set c
    c .= W .^ -(1.0/σ)
    # Set n
    for ei in axes(n, 2)
        n[:, ei] .= ((wage * egrid[ei] / φ) .* view(W, :, ei)) .^ (1.0 / ν)
    end
    # Set a₋
    for ei in axes(a₋, 2)
        a₋[:, ei] .= @views (
            c[:, ei] .+ agrid .- (wage * egrid[ei]) .* n[:, ei] .- transfers[ei]
        ) ./ (1.0 + r)
    end    
    return vars
end

# Functions for evaluating the policy function and 
# value function derivative for constrained households
function solvelogconstrained(lw, Y, nguess, tol)
    # maximises u(c, n) s.t c = exp(lw + ln) + Y
    # uses Newton's method to solve FOC
    ln = nguess
    f = tol + 1.0
    while abs(f) > tol
        ewn = exp(lw + ln)
        f   = -σ * log(ewn + Y) +lw - log(φ) - ν * ln
        f′  = -σ * ewn / (ewn + Y) - ν
        ln -= f / f′
    end
    # return log(n) (for next guess), n, and c
    return ln, exp(ln), exp(lw + ln) + Y
end

function getdv(c, n, r, w)
    # returns derivative of constrained value function wrt a₋ 
    # at c*=c and n*=n
    dn = (-σ * (1.0 + r) / c) / (ν / n + σ * w / c)
    dc = w * dn + 1.0 + r
    return dc * c^(-σ) - φ * dn * n^ν
end

function fixconstrained!(v, outcomes, a₋, r, w, transfers)
    # need to fix outcomes: [a, n]; values: v
    for ei in axes(a₋, 2)
        # amin is the value of a₋ which you choose to a′=lower_bound
        # any lower a is constrained
        amin = a₋[1, ei] 
        nguess, lw = 0.0, log(w * egrid[ei])
        for ai in eachindex(agrid)
            if agrid[ai] >= amin
                break # monotonicity => can stop checking
            else # they are constrained
                Y = (1.0 + r) * agrid[ai] + transfers[ei]
                nguess, n, c = solvelogconstrained(lw, Y, nguess, 1e-11)
                outcomes[ai+(ei-1)*length(agrid), 1] = agrid[1] # asset policy
                outcomes[ai+(ei-1)*length(agrid), 2] = n # labour policy
                v[ai, ei] = getdv(c, n, r, w * egrid[ei]) # value function derivative
            end
        end
    end
end

function inner_update_outcomes!(Y, a₋, n) 
    na = size(agrid, 1)
    # fill first column with a, second with n
    loc = 1
    for ei in axes(n, 2)
        @views fastinterp!(Y[loc:loc+na-1, 1], agrid, a₋[:, ei], agrid)
        @views fastinterp!(Y[loc:loc+na-1, 2], agrid, a₋[:, ei], n[:, ei])
        loc += na
    end
end

function inner_iterate_distribution!(d, d0, a₋, tmp1, tmp2)

    # d is filled with next period distribution, d0 is current distribution
    # tmp1 and tmp2 are caches that will be overwritten

    # fill each column of tmp1 with the mass for a′ and e
    fill!(tmp1, 0)
    loc = 1
    for ei in axes(egrid, 1)
        @views fastinterp!(tmp2[:, ei], agrid, a₋[:, ei], axes(agrid, 1))

        # set constrained to index 1
        for ai in axes(agrid, 1)
            if tmp2[ai, ei] >= 1
                break
            else
                tmp2[ai, ei] = 1
            end
        end

        for ai in axes(agrid, 1)
            mass = d0[loc]
            la   = floor(Int, tmp2[ai, ei]) # index of asset grid point one below exact policy
            ω    = tmp2[ai, ei] - la
            if la < size(tmp2, 1) # i.e. they are not constrained at the top
                tmp1[la, ei]   += mass * (1-ω)
                tmp1[la+1, ei] += mass * ω
            else
                tmp1[la, ei]   += mass
            end
            loc += 1
        end
    end

    # finally apply exogenous transition to get a′ and e′
    mul!(tmp2, tmp1, transpose(Qt)) # i.e. tmp2 = tmp1 * Q
    for i in eachindex(d)
        d[i] = tmp2[i] # can't multiply directly into d as it is wrong shape
    end
    return d
end

function constructΛ(a₋, tmp2)
    # not efficient but is only called when revaluating steady-state
    Λt = zeros(length(agrid) * length(egrid), length(agrid) * length(egrid))
    na = size(a₋, 1)
    loc = 1
    for ei in eachindex(egrid)
        @views fastinterp!(tmp2[:, ei], agrid, a₋[:, ei], axes(agrid, 1))

        # set constrained to index 1
        for ai in axes(agrid, 1)
            if tmp2[ai, ei] >= 1
                break
            else
                tmp2[ai, ei] = 1
            end
        end

        for ai in eachindex(agrid)
            la = floor(Int, tmp2[ai, ei])
            ω  = tmp2[ai, ei] - la
            if la < length(agrid)
                for ei′ in eachindex(egrid)
                    Λt[la+(ei′-1)*na, loc]   = (1-ω) * Qt[ei′, ei]
                    Λt[la+1+(ei′-1)*na, loc] = ω * Qt[ei′, ei]
                end
            else
                for ei′ in eachindex(egrid)
                    Λt[la+(ei′-1)*na, loc] = Qt[ei′, ei]
                end
            end
            loc += 1
        end
    end
    return sparse(transpose(Λt))
end

function inner_backwards_iterate!(v, a₋, W, r)
    # directly overwrites v with the new value function
    # must be corrected for constrained individuals!
    for ei in axes(v, 2)
        @views fastinterp!(v[:, ei], agrid, a₋[:, ei], (1.0 + r) .* W[:, ei])
    end
    return v
end

function combined_evaluation!(vf, Y, dist, dist0, xt, tmps)

    # given x_t and v_(t+1), calculates outcomes (a & c), forward
    # iterates the distribution and backwards iterates v without 
    # redundant computation

    r, w, d, t = xt[1], xt[2], xt[3], xt[4]
    transfers  = d * drule - t * trule
    updateEGMvars!(tmps, vf, r, w, transfers)
    W, c, n, a₋ = tmps

    inner_update_outcomes!(Y, a₋, n)
    inner_iterate_distribution!(dist, dist0, a₋, c, n)
    inner_backwards_iterate!(vf, a₋, W, r)
    fixconstrained!(vf, Y, a₋, r, w, transfers)

end

# ===== Steady state finders =====
function supnorm(x, y)
    maximum(abs.(x .- y))
end

function steady_state_value(initv, xss; maxiter=1000, tol=1e-10)

    v = initv
    holder = copy(v)
    err = 0

    tmps = makecache(eltype(xss))
    r, w, d, t = xss[1], xss[2], xss[3], xss[4]
    transfers = d * drule - t * trule
    # Y does not matter but it saves having to write a 
    # specialized fixconstrained function which does not
    # update Y
    Y = zeros(eltype(xss), size(v, 1) * size(v, 2), 2)

    for iter in 1:maxiter
        updateEGMvars!(tmps, v, r, w, transfers)
        inner_backwards_iterate!(v, tmps[4], tmps[1], r)
        fixconstrained!(v, Y, tmps[4], r, w, transfers)
        err = supnorm(v, holder)
        if err < tol
            return (value=v, converged=true, iter=iter, err=err)
        else
            copy!(holder, v)
        end
    end
    return (value=v, converged=false, iter=maxiter, err=err)
end

function steady_state_distribution(initd, xss, vss; maxiter=5000, tol=1e-8)

    dist = initd
    holder = copy(dist)
    err = 0

    tmps = makecache(eltype(xss))
    r, w, d, t = xss[1], xss[2], xss[3], xss[4]
    transfers = d * drule - t * trule
    updateEGMvars!(tmps, vss, r, w, transfers)

    Λss = constructΛ(tmps[4], tmps[1])

    for iter in 1:maxiter
        mul!(dist, transpose(Λss), holder)
        err = supnorm(dist, holder)
        if err < tol
            return (value=dist, converged=true, iter=iter, err=err), Λss
        else
            copy!(holder, dist)
        end
    end
    return (value=dist, converged=false, iter=maxiter, err=err), Λss
end

function _updatesteadystate!(ha, x)

    res_value = steady_state_value(ha.vss, x)
    @assert res_value.converged "Value function did not converge"

    res_dist, Λss = steady_state_distribution(ha.dss, x, res_value.value)
    @assert res_dist.converged "Invariant distribution did not converge"

    tmps = makecache(Float64)
    transfers = x[3] * drule - x[4] * trule
    updateEGMvars!(tmps, res_value.value, x[1], x[2], transfers)
    W, c, n, a₋ = tmps

    ha.xss .= x
    ha.vss .= res_value.value
    ha.dss .= res_dist.value
    inner_update_outcomes!(ha.yss, a₋, n)
    fixconstrained!(W, ha.yss, a₋, x[1], x[2], transfers)
    ha.Λss .= Λss

    return ha_block

end

ha_block = HetBlock(
    [:r, :w, :d, :τ], [:𝒩, :𝒜], 300,
    combined_evaluation!,
    makecache,
    xss,
    [   (1+xss[1]) * (0.1 * ((1 + xss[1]) * agrid[ai] + 0.1))^(-1/σ)
        for ai in eachindex(agrid), ei in eachindex(egrid)
    ],
    ones(length(agrid) * length(egrid)) / (length(agrid) * length(egrid)),
    spzeros(length(agrid) * length(egrid), length(agrid) * length(egrid)),
    zeros(length(agrid) * length(egrid), 2), # 2 for two outputs,
    _updatesteadystate!
)

updatesteadystate!(ha_block, xss)

#endregion