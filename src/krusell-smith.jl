using SequenceSpace

using DelimitedFiles
using LinearAlgebra
using SparseArrays
using StaticArrays

# absolute file path in case this file is evaluated from multiple locations
fp = "/Users/ruairidh/.julia/dev/SequenceSpace/"

const T = 300

#region Household block =====

# ===== Set up parameters =====

const agrid = readdlm(fp * "tempdata/paper_ks/a_grid.csv", ',', Float64)[:, 1]
egrid_raw   = readdlm(fp * "tempdata/paper_ks/e_grid.csv", ',', Float64)[:, 1]
const egrid = SVector{length(egrid_raw)}(egrid_raw)

Qt_raw = readdlm(fp * "tempdata/paper_ks/Pi.csv", ',', Float64) |> permutedims
const Qt = SMatrix{size(Qt_raw, 1), size(Qt_raw, 2)}(Qt_raw)

rwb = readdlm(fp * "tempdata/paper_ks/rwb.csv", ',', Float64)
const β = rwb[3]
rss, wss = rwb[1], rwb[2]

# ===== Iteration functions =====
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

function inner_backwards_iterate!(v, a₋, W, r, w)

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

function combined_evaluation!(vf, Y, d, d0, xt, tmps)
    # given x_t and v_(t+1), calculates outcomes (a & c), forward
    # iterates the distribution and backwards iterates v without 
    # redundant computation

    r, w = xt[1], xt[2]
    updateEGMvars!(tmps, vf, r, w)
    W, c, a₋, tmp1 = tmps

    inner_update_outcomes!(Y, a₋, c, r, w)
    inner_iterate_distribution!(d, d0, a₋, c, tmp1)
    inner_backwards_iterate!(vf, a₋, W, r, w)

end

function makecache(S)
    ( zeros(S, size(agrid, 1), size(egrid, 1)),
      zeros(S, size(agrid, 1), size(egrid, 1)),
      zeros(S, size(agrid, 1), size(egrid, 1)),
      zeros(S, size(agrid, 1), size(egrid, 1))
    )
end

# ===== Steady state finders =====

function supnorm(x, y)
    maximum(abs.(x .- y))
end

function steady_state_value(initv, xss; maxiter=1000, tol=1e-8)

    #v = [1 / (a+0.001) for a in agrid, e in egrid]
    v = initv
    holder = copy(v)
    err = 0

    tmps = makecache(Float64)

    r, w = xss[1], xss[2]

    for iter in 1:maxiter
      updateEGMvars!(tmps, v, r, w)
      inner_backwards_iterate!(v, tmps[3], tmps[1], r, w)
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

    # d = ones(length(agrid) * length(egrid)) / (length(agrid) * length(egrid))
    d = initd
    holder = copy(d)

    tmps = makecache(Float64)

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

function _updatesteadystate!(ha, x; updateΛss=true) 

    res_value = steady_state_value(ha.vss, x)
    @assert res_value.converged

    res_dist = steady_state_distribution(ha.dss, x, res_value.value)
    @assert res_dist.converged

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

ha_block = HetBlock(
    [:r, :w], [:𝓀, :c], T,
    combined_evaluation!,
    makecache,
    [rss, wss],
    [1 / (a+0.001) for a in agrid, e in egrid],
    ones(length(agrid) * length(egrid)) / (length(agrid) * length(egrid)),
    spzeros(length(agrid) * length(egrid), length(agrid) * length(egrid)),
    zeros(length(agrid) * length(egrid), 2), # 2 for two outputs,
    _updatesteadystate!
)

updatesteadystate!(ha_block, [rss, wss])

#endregion

#region Simple blocks =====

const δ = 0.025
const α = 0.11

kss, zss = 3.14, 1.0

firms_block = @simpleblock [:k, :z] [:r, :w, :y] [kss, zss] firms(k, z) = begin
  r = α * z[0] * k[-1]^(α-1) - δ
  w = (1-α) * z[0] * k[-1]^α
  y = z[0] * k[-1]^α
  return [r, w, y]
end

eq_block = @simpleblock [:𝓀, :k] [:h] [0.0, 0.0] target(𝓀, k) = begin
  h = 𝓀[0] - k[0]
  return [h]
end

#endregion

#region make Model graph =====

blocks = [ha_block, firms_block, eq_block]
mg = ModelGraph(blocks, [:k], [:z], [:h], [:k, :z, :y, :r, :w])
updatepartialJacobians!(mg)
Gs = generaleqJacobians(makeG(mg), mg)

#=
g_forward = makeG(mg)
g_backward = SequenceSpace.makeG_backwards(mg)
Gsback = generaleqJacobians(SequenceSpace.makeG_backwards(mg), mg)

@benchmark generaleqJacobians(makeG($mg), $mg)
@benchmark generaleqJacobians(SequenceSpace.makeG_backwards($mg), $mg)

T = mg.T
nt, nu, nx = length(mg.eqvars), length(mg.unknowns), length(mg.exog)
bigG = zeros(T * nu, T * nx)
Hu = zeros(T * nu, T * nu)
Hx = zeros(T * nu, T * nx)
Gs = Dict( # initialize
  var => zeros(T, T * length(mg.exog)) for var in mg.vars
)

@benchmark SequenceSpace.updatinggeq_backward!($Gs, $bigG, $Hu, $Hx, $mg)
@benchmark SequenceSpace.updatinggeq_forward!($Gs, $bigG, $Hu, $Hx, $mg)

function profileJacob(n, mg)
  for i in 1:n
    SequenceSpace.updatinggeq_forward!(Gs, bigG, Hu, Hx, mg)
  end
end
=#


#endregion