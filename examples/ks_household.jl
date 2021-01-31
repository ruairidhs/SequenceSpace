using SequenceSpace 

using SparseArrays

# Functions for the household block for the Krusell-Smith block
# Required constant parameters:
#   - agrid (asset grid), vector
#   - egrid (exogenous state grid), vector
#   - Qt (transpose of exogenous transition matrix), matrix
#   - β (household discount rate), scalar

#region Iteration functions =====

function updateEGMvars!(vars, vf, r, wage)

    W, c, a₋, = vars # doesn't affect tmp1

    # Set W
    mul!(W, vf, β .* Qt)
    # Set c
    c .= 1.0 ./ W
    # Set a₋
    for ei in axes(egrid, 1)
        a₋[:, ei] .= (agrid .+ view(c, :, ei) .- (wage * egrid[ei])) ./ (1.0 + r)
    end
    return vars

end

function inner_update_outcomes!(Y, a₋, c, r, w) 

    na = size(agrid, 1)
    # fill first column with a, second with c
    loc = 1
    for ei in axes(egrid, 1)
      @views fastinterp!(Y[loc:loc+na-1, 1], agrid, a₋[:, ei], agrid)
      @views fastinterp!(Y[loc:loc+na-1, 2], agrid, a₋[:, ei], c[:, ei])
      loc += na
    end

    # reset any constrained agents:
    #   - for assets, set to lower bound
    #   - for consumption, set to: (1+rₜ)a₋ + wₜe
    mina, na = first(agrid), size(agrid, 1)
    for ei in axes(egrid, 1)
      for ai in axes(agrid, 1)
        loc = (ei - 1) * na + ai
        if Y[loc, 1] >= mina
            break # monotonicity, skips to next ei 
        else
            Y[loc, 1] = mina
            Y[loc, 2] = (1.0+r)*agrid[ai] + w * egrid[ei]
        end
      end
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
        if la < length(agrid) # i.e. they are not constrained at the top
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
    # not efficient, but is not called frequently
    # Λ is used to construct curlyEs in fake news, 
    # but not to iterate the distribution
    Λt = zeros(length(agrid) * length(egrid), length(agrid) * length(egrid))
    na = length(agrid)

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
        la   = floor(Int, tmp2[ai, ei])
        ω    = tmp2[ai, ei] - la
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

function constrained_value(r, w, z, k, kmin)
    (1+r) / (
      (1+r) * k + (w * z) - kmin
    )
end

function inner_value_iterate!(v, a₋, W, r, w)

    # directly overwrites v with the new value function
    for ei in axes(v, 2)
      @views fastinterp!(v[:, ei], agrid, a₋[:, ei], (1.0 + r) .* W[:, ei])
      # need to correct the extrapolation for constrained agents
      amin = a₋[1, ei]
      for ai in eachindex(agrid)
        if agrid[ai] >= amin
            break # monotonicity => can stop checking
        else # they are constrained
            v[ai, ei] = constrained_value(
              r, w, egrid[ei], agrid[ai], agrid[1]
            )
        end
      end
    end

end

function backwards_iterate!(vf, Y, d, d0, xt, tmps)
    # given x_t and v_(t+1), calculates outcomes (a & c), forward
    # iterates the distribution and backwards iterates v without 
    # redundant computation

    r, w = xt[1], xt[2]
    updateEGMvars!(tmps, vf, r, w)
    W, c, a₋, tmp1 = tmps

    inner_update_outcomes!(Y, a₋, c, r, w)
    inner_iterate_distribution!(d, d0, a₋, c, tmp1)
    inner_value_iterate!(vf, a₋, W, r, w)

end

function makecache(S)
    ( zeros(S, size(agrid, 1), size(egrid, 1)),
      zeros(S, size(agrid, 1), size(egrid, 1)),
      zeros(S, size(agrid, 1), size(egrid, 1)),
      zeros(S, size(agrid, 1), size(egrid, 1))
    )
end

#endregion =====

#region Steady-state functions =====

function supnorm(x, y)
    maximum(abs.(x .- y))
end

function steady_state_value(initv, xss; maxiter=1000, tol=1e-8)

    v = initv
    holder = copy(v)
    err = 0

    tmps = makecache(eltype(xss))

    r, w = xss[1], xss[2]

    for iter in 1:maxiter
      updateEGMvars!(tmps, v, r, w)
      inner_value_iterate!(v, tmps[3], tmps[1], r, w)
      err = supnorm(v, holder)
      if err < tol
        return (value=v, converged=true, iter=iter, err=err)
      else
        copy!(holder, v)
      end
    end
    return (value=v, converged=false, iter=maxiter, err=err)
end

function steady_state_distribution(initd, xss, vss; maxiter=2000, tol=1e-8)

    d = initd
    holder = copy(d)

    tmps = makecache(eltype(xss))

    r, w = xss[1], xss[2]
    updateEGMvars!(tmps, vss, r, w)
    tmp1, tmp2, a₋, = tmps

    for iter in 1:maxiter
      inner_iterate_distribution!(d, d, a₋, tmp1, tmp2)
      err = supnorm(d, holder)
      if err < tol
        return (value=d, converged=true, iter=iter, err=err)
      else
        copy!(holder, d)
      end
    end
    return (value=d, converged=false, iter=maxiter, err=err)
end

function kssteadystate!(ha, x;
    updateΛss=true, tol=1e-8, maxiter=2000
) 
    res_value = steady_state_value(ha.vss, x, maxiter = maxiter, tol = tol)
    @assert res_value.converged "Value function did not converge"

    res_dist = steady_state_distribution(ha.dss, x, res_value.value, maxiter = maxiter, tol = tol)
    @assert res_dist.converged "Invariant distribution did not converge"

    tmps = makecache(Float64)
    updateEGMvars!(tmps, res_value.value, x[1], x[2])
    W, c, a₋, tmp1 = tmps

    ha.xss .= x
    ha.vss .= res_value.value
    ha.dss .= res_dist.value
    inner_update_outcomes!(ha.yss, a₋, c, x[1], x[2])

    if updateΛss # this is slow so can turn off if not needed
        ha.Λss .= constructΛ(a₋, tmp1)
    end

end

#endregion