struct HetBlock <: Block

    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    T::Int

    v!::Function # iterate value function backwards
    p!::Function # update policy
    Λ!::Function # iterate distribution forwards
    y!::Function # compute outcome for each point on the distribution

    makecache::Function # make the cache required for computation

end

inputs(hb::HetBlock) = hb.inputs
outputs(hb::HetBlock) = hb.outputs
getT(hb::HetBlock) = hb.T

# ===== Functions for the fake news algorithm =====
# ===== Step 1 =====
function get_diffs!(
    diffs, dx,
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
    ha.p!(p, v, dx, cache) # update the policy
    ha.y!(y, p, dx, cache) # outcome based on new policy
    ha.Λ!(view(diffs, dindices, 0), dss, p, cache) # how the distribution reacts
    mul!(view(diffs, yindices, 0), transpose(y), d) # how the aggregate outcome reacts
    ha.v!(v, v, dx, cache) # finally iterate the value function
    for s in 1:ha.T-1  # now iterate backwards
        ha.p!(p, v, xss, cache) # now back to steady-state x
        ha.y!(y, p, xss, cache)
        ha.Λ!(view(diffs, dindices, s), dss, p, cache)
        mul!(view(diffs, yindices, s), transpose(y), d)
        ha.v!(v, v, xss, cache)
    end
    return diffs
end

function get_derivs(
    xss, vss, dss, yss, pss, ha
)   
    diffs = OffsetArray(
        zeros(size(yss, 2) + length(dss), ha.T), 0, -1
    )
    raw_res = ForwardDiff.jacobian(
        (y, dx) -> get_diffs!(y, dx, xss, vss, dss, yss, pss, ha), diffs, xss
    ) 

    reshape(raw_res, size(yss, 2) + length(dss), ha.T, length(xss))
end

# ===== Step 2 =====
function updatecurlyE!(Es, T, output_index, Λss, yss)
    Es[0] .= yss[:, output_index]
    for i in 1:T-1
          mul!(Es[i], Λss, Es[i-1])
    end
end

# ===== Step 3 =====
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

# ===== Step 4 =====
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

# ===== full fake news algorithm =====

function jacobian(ha::HetBlock, steady_state)

    xss, vss, dss, yss, pss, Λss = steady_state

    derivs = get_derivs(
        xss, vss, dss, yss, pss, ha
    )

    dindices = axes(dss, 1)
    yindices = (length(dss)+1):size(derivs, 1)

    Es = OffsetArray([zeros(axes(dss)) for t in 0:ha.T-1], -1)
    F = zeros(ha.T, ha.T)

    res = Dict(
        (ha.inputs[i], ha.outputs[o]) => zeros(ha.T, ha.T)
        for o in eachindex(ha.outputs), i in eachindex(ha.inputs)
    )

    for output_index in eachindex(ha.outputs)
        updatecurlyE!(Es, ha.T, output_index, Λss, yss)
        for input_index in eachindex(ha.inputs)
            Ds = view(derivs, dindices, :, input_index)
            Ys = view(derivs, yindices[output_index], :, input_index)
            updateF!(F, ha.T, Es, Ys, Ds)
            updateJ!(res[(ha.inputs[input_index], ha.outputs[output_index])], F, ha.T)
        end
    end

    return res

end