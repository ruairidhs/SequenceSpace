using ForwardDiff

# ===== Set up structures =====
abstract type Block end

struct HetBlock <: Block

    v!::Function # iterate value function backwards
    p!::Function # update policy
    Λ!::Function # iterate distribution forwards
    y!::Function # compute outcome for each point on the distribution

    makecache::Function # make the cache required for computation

end

function (ha::HetBlock)(output, input)
    # computes the output given the input
end

function get_diffs!(
    diffs, dx, T,
    xss, vss, dss, yss, pss,
    ha
)   
    # ===== Set up =====
    # Must be differentiable!
    S = eltype(dx) # get type to construct cache and output
    cache = ha.makecache(S) 

    # Preallocate output, each has one column for s=0:(T-1)
    dindices = axes(dss, 1)
    yindices = (length(dss)+1):size(diffs, 1)

    # Need to convert to type S any objects that will be mutated
    x = convert(Array{S}, xss)
    v = convert(Array{S}, vss)
    d = convert(Array{S}, dss)
    y = convert(Array{S}, yss)
    p = convert(Array{S}, pss)

    # ===== Computation =====
    # For s=0, the inputs are shocked but next period value will be steady-state
    ha.p!(p, v, dx, cache)            # update the policy
    ha.y!(y, p, dx, cache)            # outcome based on new policy
    ha.Λ!(view(diffs, dindices, 0), dss, p, cache)       # how the distribution reacts
    mul!(view(diffs, yindices, 0), transpose(y), d)     # how the aggregate outcome reacts
    ha.v!(v, v, dx, cache)            # finally iterate the value function
    for s in 1:T-1                      # now iterate backwards
        ha.p!(p, v, xss, cache)              # now back to steady-state x
        ha.y!(y, p, xss, cache)
        ha.Λ!(view(diffs, dindices, s), dss, p, cache)
        mul!(view(diffs, yindices, s), transpose(y), d)
        ha.v!(v, v, xss, cache)
    end
    return diffs
end

function get_derivs(
    T, xss, vss, dss, yss, pss, ha
)   
    diffs = OffsetArray(
        zeros(size(yss, 2) + length(dss), T), 0, -1
    )
    raw_res = ForwardDiff.jacobian(
        (y, dx) -> get_diffs!(y, dx, T, xss, vss, dss, yss, pss, ha), diffs, xss
    ) 

    reshape(raw_res, size(yss, 2) + length(dss), T, length(xss))

end

function updatecurlyE!(Es, T, output_index, Λss, yss)
    Es[0] .= yss[:, output_index]
    for i in 1:T-1
          mul!(Es[i], Λss, Es[i-1])
    end
end

function updateF!(F, T, Es, Ys, Ds)
    for s in 1:T
        # First row
        F[1, s] = Ys[s]
        # Further rows
        for t in 2:T
            F[t, s] = transpose(Es[t-2]) * Ds[:, s]
        end
    end
    return F
end

function updateJ!(J, F, T)
    # first row and column is just copied from F
    # i allow for general indices as J may be a section of a larger Jacobian
    ax = axes(J)
    J[:, firstindex(ax[2])] .= F[:, 1]
    J[firstindex(ax[1]), :] .= F[1, :]

    # then build recursively by columns
    for s in Iterators.drop(eachindex(ax[2]), 1) # every index but the first
        for t in Iterators.drop(eachindex(ax[1]), 1) 
            J[t, s] = J[t-1, s-1] + F[t, s]
        end
    end
    return J
end

function jacobian(hb::HetBlock, T, xss, vss, dss, yss, pss, Λss)

    derivs = get_derivs(
        T, xss, vss, dss, yss, pss, hb
    )

    dindices = axes(dss, 1)
    yindices = (length(dss)+1):size(derivs, 1)

    Es = OffsetArray([zeros(axes(dss)) for t in 0:T-1], -1)
    F = zeros(T, T)
    J = zeros(T * size(yss, 2), T * length(xss))

    rloc = 1
    for output_index in axes(yss, 2)
        updatecurlyE!(Es, T, output_index, Λss, yss)
        cloc = 1
        for input_index in eachindex(xss)
            Ds = view(derivs, dindices, :, input_index)
            Ys = view(derivs, yindices[output_index], :, input_index)
            # section of the jacobian to update
            Jchunk = view(J, rloc:rloc+T-1, cloc:cloc+T-1)
            updateF!(F, T, Es, Ys, Ds)
            updateJ!(Jchunk, F, T)
            cloc += T
        end
        rloc += T
    end
    return J
end
    
# ===== Example and testing =====

include("fakenews.jl")

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

haBlock = HetBlock(
    # v!
    (vc, vf, x, cache) -> _iterate_value!(vc, vf, x, params, cache),
    # p!
    (p, v, x, cache) -> _update_policy!(p, v, x, params, cache, kinds),
    # Λ!
    (d, dp, p, cache) -> _apply_transition!(d, p, dp, params, cache),
    # y!
    (y, p, x, cache) -> _update_outcomes!(y, p, x, params, cache, kinds),
    # makecache
    S -> ks_makecache(S, params)
)

# get_derivs(300, xss, vss, dss, yss, pss, haBlock)
# J = jacobian(haBlock, 300, xss, vss, dss, yss, pss, Λss)

function plot_sec(o, i)
    oindices = ((o-1) * 300 + 1):((o-1) * 300 + T)
    iindices = ((i-1) * 300 + 1):((i-1) * 300 + T)
    subJ = J[oindices, iindices]
    subJ[:, 1 .+ [0, 25, 50, 75, 100]] |> plot
end