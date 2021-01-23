# Code for replicating the paper's Krusell-Smith example

const T = 300

# ===== Firms block =====
const δ = 0.025
const α = 0.11
const kss = 3.14
const zss = 1.0

getr(k, z) = α * z * k^(α-1.0) - δ
getw(k, z) = (1.0-α) * z * k^α

function firms!(output, input, k0)

    # inputs: k, z
    # outputs: r, w

    T = size(input, 1)

    output[1, 1] = getr(k0, input[1, 2])
    output[1, 2] = getw(k0, input[1, 2])

    for t in 2:T
        k, z = input[t-1, 1], input[t, 2]
        output[t, 1] = getr(k, z)
        output[t, 2] = getw(k, z)
    end

    return output
end

# ===== Het agents block =====
# Set up to match paper
raw_agrid = readdlm("../tempdata/a_grid.csv", ',', Float64)
raw_pi    = readdlm("../tempdata/Pi.csv", ',', Float64)
raw_egrid = readdlm("../tempdata/e_grid.csv", ',', Float64)
raw_Va    = readdlm("../tempdata/Va.csv", ',', Float64)
raw_rwb   = readdlm("../tempdata/rwb.csv", ',', Float64)

const γ = 1.0
# up(c) = c^(-γ)
# upinv(u) = u^(-1/γ)
up(c) = 1 / c
upinv(u) = 1 / u

const kgrid = raw_agrid[:,1]
const zgrid = raw_egrid[:,1]
const Qt    = permutedims(raw_pi)

const r = raw_rwb[1]
const w = raw_rwb[2]
const β = raw_rwb[3]

const kinds = reshape(
    repeat(1:length(kgrid), length(zgrid)), length(kgrid), length(zgrid)
)
const kitp = extrapolate(interpolate(kgrid, BSpline(Linear())), Flat())

const xss = [r, w]

params = (β, Qt, kgrid, zgrid)


function constrained_value(r, w, z, k, kmin)
    (1+r) * up(
        (1+r) * k + (w * z) - kmin
    )
end

function _iterate_value!(vc, vf, xt, params, tmps)

    r, w = xt[1], xt[2]
    β, Qt, kgrid, zgrid = params
    tmp1, tmp2, tmp3, tmp4 = tmps

    mul!(tmp1, vf, β .* Qt) # Expectation of future value
    tmp2 .= (1+r) .* tmp1  # Values
    tmp3 .= upinv.(tmp1)   # Actions

      for zi in eachindex(zgrid)
            tmp4[:, zi] .= ( kgrid .+ view(tmp3, :, zi) .- (w * zgrid[zi]) ) ./
                  (1+r) # Nodes
            fastinterp!(view(vc, :, zi), kgrid, view(tmp4, :, zi), view(tmp2, :, zi))
            kmin = tmp4[1, zi]
            for ki in eachindex(kgrid)
                  if kgrid[ki] < kmin
                        vc[ki, zi] = constrained_value(
                              r, w, zgrid[zi], kgrid[ki], kgrid[1]
                        )
                  end
            end
      end
      return vc
end

function _update_policy!(p, v, xt, params, tmps, kinds)

      r, w = xt[1], xt[2]
      β, Qt, kgrid, zgrid = params
      tmp1, tmp2, tmp3, tmp4 = tmps

      mul!(tmp1, v, β .* Qt) # Compute expectation of future value
      tmp3 .= upinv.(tmp1)   # Actions

      for zi in eachindex(zgrid)
            tmp4[:, zi] = ( kgrid .+ view(tmp3, :, zi) .- (w * zgrid[zi]) ) ./
                     (1+r) # Nodes
            fastinterp!(view(p, :, zi), kgrid, view(tmp4, :, zi), view(kinds, :, zi))
      end
      # Change to flat extrapolation
      nk = length(kgrid)
      for i in eachindex(p)
            if p[i] < 1
                  p[i] = 1
            elseif p[i] > nk
                  p[i] = nk
            end
      end
      return p
end

function _make_transition(pc, params, RCV)

      β, Qt, kgrid, zgrid = params
      Rs, Cs, Vs = RCV

      nk = length(kgrid)
      nz = length(zgrid)

      for zi in eachindex(zgrid)

          loc = 1
          for ki in eachindex(kgrid) # each original k

              kpol = pc[ki, zi] # real-valued index in kgrid
              lk   = floor(Int, kpol) # integer index of k below
              ω    = kpol % 1

              if lk < nk # if there is room above
                  Rs[loc:loc+1, 1, zi] .= ki
                  Cs[loc, 1, zi]        = lk
                  Cs[loc+1, 1, zi]      = lk + 1
                  Vs[loc, 1, zi]        = 1 - ω
                  Vs[loc+1, 1, zi]      = ω
              else
                  Rs[loc, 1, zi]   = ki
                  Cs[loc, 1, zi]   = lk
                  Vs[loc, 1, zi]   = 1 # place all mass on lower point
                  Rs[loc+1, 1, zi] = 1 # this doesn't matter, just need to enter something
                  Cs[loc+1, 1, zi] = 1 # ''
                  Vs[loc+1, 1, zi] = 0 # ''
              end
              loc += 2
          end # ki
          # Now compose for z'
          for ziprime in 2:nz
              Rs[:, ziprime, zi] .= Rs[:, 1, zi]
              # Columns need to be shifted along
              Cs[:, ziprime, zi] .= Cs[:, 1, zi] .+ ((ziprime - 1) * nk)
              # Values need to be multiplied by probabilities
              Vs[:, ziprime, zi] .= Vs[:, 1, zi] .* Qt[ziprime, zi]
          end
          # Also need to change the values for the first column
          Vs[:, 1, zi] .*= Qt[1, zi]
          # And finally shift the row indices down
          Rs[:, :, zi] .+= ((zi - 1) * nk)
      end # zi

      return sparse(vec(Rs), vec(Cs), vec(Vs), nk * nz, nk * nz)
end

function _apply_transition!(d, p, dp, params, tmps)
      β, Qt, kgrid, zgrid = params
      tmp1, tmp2, tmp3, tmp4 = tmps
      nk = length(kgrid)

      fill!(tmp1, 0)

      # Fill tmp1 so that each column contains the endogenous next capital
      # distribution from each starting exogenous capital
      loc = 1
      for zi in eachindex(zgrid)
            for ki in eachindex(kgrid)
                  mass = dp[loc]
                  kpol = p[ki, zi]
                  lk   = floor(Int, kpol)
                  ω    = kpol - lk
                  if lk < nk # there is room above
                        tmp1[lk,   zi] += mass * (1 - ω)
                        tmp1[lk+1, zi] += mass * ω
                  else # at the upper boundary
                        tmp1[nk, zi] += mass
                  end
                  loc += 1
            end
      end
      mul!(tmp2, tmp1, transpose(Qt))
      for i in eachindex(d)
            d[i] = tmp2[i]
      end
end

function _update_outcomes!(y, policy, xt, params, tmps, kinds)

      r, w = xt[1], xt[2]
      β, Qt, kgrid, zgrid = params
      tmp1, tmp2, tmp3, tmp4 = tmps

      for zi in eachindex(zgrid)
            # updates capital demand into tmp1
            fastinterp!(
                  view(tmp1, :, zi), view(policy, :, zi),
                  view(kinds, :, zi), kgrid
            )
            for ki in eachindex(kgrid)
                  # update consumption into tmp2
                  tmp2[ki, zi] = (1 + r) * kgrid[ki] + w * zgrid[zi] - tmp1[ki, zi]
            end
      end

      y[:, 1] .= vec(tmp1)
      y[:, 2] .= vec(tmp2)
end

# ===== Compute steady-state values =====
function supnorm(x, y)
    maximum(abs.(x .- y))
end

function steady_state_value(xt, params; maxiter = 1000, tol=1e-8)

      β, Qt, kgrid, zgrid = params

      vc = [1 / (k+0.001) for k in kgrid, z in zgrid]
      holder = copy(vc)
      err = 0

      tmps = (
            zeros(Float64, length(kgrid), length(zgrid)),
            zeros(Float64, length(kgrid), length(zgrid)),
            zeros(Float64, length(kgrid), length(zgrid)),
            zeros(Float64, length(kgrid), length(zgrid))
      )

      for iter in 1:maxiter
            _iterate_value!(vc, vc, xt, params, tmps)
            err = supnorm(vc, holder)
            if err < tol
                  return (value=vc, converged=true, iter=iter, err=err)
            else
                  copy!(holder, vc)
            end
      end
      return (value=vc, converged=false, iter=iter, err=err)
end

function steady_state_distribution(v, xt, params, kinds; maxiter = 2000, tol=1e-8)

      β, Qt, kgrid, zgrid = params
      tmps = (
            zeros(Float64, length(kgrid), length(zgrid)),
            zeros(Float64, length(kgrid), length(zgrid)),
            zeros(Float64, length(kgrid), length(zgrid)),
            zeros(Float64, length(kgrid), length(zgrid))
      )

      nk, nz = length(kgrid), length(zgrid)
      Rs = Array{Int}(undef, 2*nk, nz, nz)
      RCV  = (Rs, similar(Rs), similar(Rs, Float64))

      d0 = vec(ones(length(kgrid), length(zgrid)) ./
          (length(kgrid) * length(zgrid)))
      holder = copy(d0)
      err = 0

      policy = similar(v)
      _update_policy!(policy, v, xt, params, tmps, kinds)
      Λss = _make_transition(policy, params, RCV)

      for iter in 1:maxiter
          d0 .= transpose(Λss) * d0
          err = supnorm(d0, holder)
          if err < tol
              return (value=d0, converged=true, iter=iter, err=err)
          else
              copy!(holder, d0)
          end
      end
      return (value=d0, converged=false, iter=maxiter, err=err)
end

vss = steady_state_value(xss, params; maxiter = 1000, tol=1e-8).value
dss = steady_state_distribution(vss, xss, params, kinds; maxiter = 2000, tol=1e-8).value

nk, nz = length(kgrid), length(zgrid)
tmp = (
            zeros(nk, nz),
            zeros(nk, nz),
            zeros(nk, nz),
            zeros(nk, nz)
      )
Rs = Array{Int}(undef, 2*nk, nz, nz)
RCV  = (Rs, similar(Rs), similar(Rs, Float64))
pss = similar(vss)
_update_policy!(pss, vss, xss, params, tmp, kinds)
Λss = _make_transition(pss, params, RCV)

yss = zeros(nk * nz, 2)
_update_outcomes!(yss, pss, xss, params, tmp, kinds)

function ks_makecache(S, params)
    β, Qt, kgrid, zgrid = params
    nk = size(kgrid, 1)
    nz = size(zgrid, 1)
    (   zeros(S, nk, nz),
        zeros(S, nk, nz),
        zeros(S, nk, nz),
        zeros(S, nk, nz)
    )
end
#=
haBlock = HetBlock(
    [:r, :w], [:𝓀, :c], T,
    (vc, vf, x, cache) -> _iterate_value!(vc, vf, x, params, cache),
    (p, v, x, cache)   -> _update_policy!(p, v, x, params, cache, kinds),
    (d, dp, p, cache)  -> _apply_transition!(d, p, dp, params, cache),
    (y, p, x, cache)   -> _update_outcomes!(y, p, x, params, cache, kinds),
    S                  -> ks_makecache(S, params)
)
=#