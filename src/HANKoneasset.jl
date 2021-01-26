using DelimitedFiles
using StaticArrays
using Optim
using LinearAlgebra
using Plots

include("fastinterp.jl")

# ===== Set up parameters =====

const β = 0.982
const φ = 0.786
const σ = 2.0
const ν = 2.0

const agrid = readdlm("tempdata/hank_a_grid.csv", ',', Float64)[:, 1] # as vector
const egrid = readdlm("tempdata/hank_e_grid.csv", ',', Float64)[:, 1] # as vector
const Qt    = readdlm("tempdata/hank_Pi.csv", ',', Float64) |> permutedims

vss_python  = readdlm("tempdata/hank_Va.csv", ',', Float64) |> permutedims

# Steady state values
# r = 0.005, w = 0.833, d = 0.166, t = 0.028
exog_invariant = Qt^1000 * (ones(axes(Qt, 1)) / size(Qt, 1))
drule = egrid ./ dot(egrid, exog_invariant)
trule = egrid ./ dot(egrid, exog_invariant)

xss = [0.005, 0.833, 0.166, 0.028]

# ===== Household block =====

# v axes: (axes(agrid, 1), axes(egrid, 1))
function makecache(agrid, egrid)
    (   zeros((axes(agrid, 1), axes(egrid, 1))),
        zeros((axes(agrid, 1), axes(egrid, 1))),
        zeros((axes(agrid, 1), axes(egrid, 1))),
        zeros((axes(agrid, 1), axes(egrid, 1))),
    )
end

# Functions for evaluating the value function derivative
# for constrained households
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

function iterate_value!(vc, vf, xt, tmps)

    # Endogenous grid method, v is derivative of value function wrt a₋

    r, w, d, t = xt[1], xt[2], xt[3], xt[4]
    transfers  = d * drule - t * trule

    updateEGMvars!(tmps, vf, r, w, transfers)
    W, c, n, a₋ = tmps
    
    # Linear interpolation of values onto original grid
    for ei in axes(vc, 2)
        @views fastinterp!(vc[:, ei], agrid, a₋[:, ei], (1.0 + r) .* W[:, ei])

        # correct extrapolation for agents who are constrained
        amin = a₋[1, ei]
        nguess, lw = 0.0, log(w * egrid[ei])
        for ai in eachindex(agrid)
            if agrid[ai] >= amin
                break # monotonicity -> can stop checking
            else # they are constrained and have to solve
                Y  = (1.0 + r) * agrid[ai] + transfers[ei]
                nguess, n, c = solvelogconstrained(lw, Y, nguess, 1e-11)
                vc[ai, ei] = getdv(c, n, r, w * egrid[ei])
            end
        end
        
    end

    return vc

end

function update_distribution!(d, dprev, xt, vf, tmps)

    r, w, d, t = xt[1], xt[2], xt[3], xt[4]
    transfers  = d * drule - t * trule

    updateEGMvars!(tmps, vf, r, w, transfers)
    # Only need a₋!
    Aindex, dtmp, n, a₋ = tmps

    fill!(dtmp, 0)
    loc = 1
    for ei in axes(W, 2)
        # Get the asset demand index for each point on the grid
        fastinterp!(Aindex[:, ei], agrid, view(a₋, :, ei), axes(kgrid,1))
        for ai in axes(W, 1)
            mass = dprev[loc]
            la   = floor(Int, Aindex[ai, ei])
            ω    = Aindex[ai, ei] - lk
            if la < size(W, 1) # i.e. they are not at the upper boundary
                dtmp[la, ei]   += mass * (1-ω)
                dtmp[la+1, ei] += mass * ω
            else
                dtmp[la, ei]   += mass
            end
            loc += 1
        end
    end
    # Now each column of dtmp is filled with the mass for a′ and e
    # Apply exogenous transition matrix to get a′ and e′
    mul!(Aindex, dtmp, transpose(Qt)) # Aindex just used as temporary cache
    for i in eachindex(d)
        d[i] = Aindex[i]
    end
    return d[i]
end

function update_outcomes!(Y, xt, vf, tmps) # OLD

    r, w, d, t = xt[1], xt[2], xt[3], xt[4]
    transfers  = d * drule - t * trule

    updateEGMvars!(tmps, vf, r, w, transfers)
    Amat, Nmat, n, a₋ = tmps

    # Need to interpolate policy from endogenous grid to fixed grid
    for ei in axes(n, 2)
        fastinterp!(Amat[:, ei], agrid, view(a₋, :, ei), agrid)
        fastinterp!(Nmat[:, ei], agrid, view(a₋, :, ei), view(n, :, ei))
    end

    # Fill first column of Y with A, second with N
    for i in axes(Y, 1)
        Y[i, 1] = Amat[i]
    end
    for i in axes(Y, 1)
        Y[i, 2] = Nmat[i]
    end

end

function inner_update_outcomes!(Y, a₋, n) # NEW

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
        for ai in axes(agrid, 1)
            mass = d0[loc]
            la   = floor(Int, tmp2[ai, ei]) # index of asset grid point one below exact policy
            ω    = tmp2[ai, ei] - lk 
            if la < size(W, 1) # i.e. they are not constrained at the top
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

function inner_backwards_iterate!(v, a₋, W, r, w, transfers)

    # directly overwrites v with the new value function
    for ei in axes(v, 2)
        @views fastinterp!(v[:, ei], agrid, a₋[:, ei], (1.0 + r) .* W[:, ei])
        # need to correct the extrapolation for constrained agents
        amin = a₋[1, ei]
        nguess, lw = 0.0, log(w * egrid[ei])
        for ai in eachindex(agrid)
            if agrid[ai] >= amin
                break # monotonicity => can stop checking
            else # they are constrained and have to solve
                Y = (1.0 + r) * agrid[ai] + transfers[ei]
                nguess, n, c = solvelogconstrained(lw, Y, nguess, 1e-11) # 1e-11 is tolerance
                v[ai, ei] = getdv(c, n, r, w * egrid[ei])
            end
        end

    end

end

function combined_evaluation!(vf, Y, d, d0, xt, tmps)

    # given x_t and v_(t+1), calculates outcomes (a & n), forward
    # iterates the distribution and backwards iterates v without 
    # redundant computation

    r, w, d, t = xt[1], xt[2], xt[3], xt[4]
    transfers  = d * drule - t * trule

    updateEGMvars!(tmps, vf, r, w, transfers)
    W, c, n, a₋ = tmps

    inner_update_outcomes!(Y, a₋, n)
    inner_iterate_distribution!(d, d0, a₋, c, n) # use c and n as temp caches as they are no longer needed
    inner_backwards_iterate!(vf, a₋, W, r, w, transfers)

end


# ===== steady-state =====

function supnorm(x, y)
    maximum(abs.(x .- y))
end

function steady_state_value(xss; maxiter = 1000, tol = 1e-8)

    vc = [  (1 + xss[1]) * (0.1 * ((1 + xss[1]) * agrid[ai] + 0.1)) ^ (-1/σ) 
            for ai in eachindex(agrid), ei in eachindex(egrid)
        ]
    holder = copy(vc)
    err = 0

    tmps = makecache(agrid, egrid)
    for iter in 1:maxiter
        iterate_value!(vc, vc, xss, tmps)
        err = supnorm(vc, holder)
        if err < tol
            return (value=vc, converged=true, iter=iter, err=err)
        else
            copy!(holder, vc)
        end
    end
    return (value=vc, converged=false, iter=maxiter, err=err)

end

res = steady_state_value(xss)

