struct HetBlock <: Block

    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    T::Int

    # Perform each step of the backwards iteration
    # in the fake news algorithm
    # iterate_block!(v, Y, d, dₜ, xₜ, tmps)
    # Given v_(t+1)=v, dₜ and xₜ:
    #   - Fills d with Λ(v_(t+1), xₜ)'dₜ (eq 11)
    #   - Fills y with y(v_(t+1), xₜ) (as in eq 12)
    #   - Finally iterates v=v(v_(t+1), xₜ) (eq 10)
    iterate_block!::Function
    # Preallocate temporary cache used by iterate_block!
    makecache::Function

    xss::Array{Float64, 1} # Steady state inputs
    vss::Array{Float64, 2} # Steady state value function
    dss::Array{Float64, 1} # Distribution (order is that of vec(vss))
    Λss::SparseMatrixCSC{Float64, Int} # Transition matrix (used only to make curlyEs)
    yss::Array{Float64, 2} # y from eq12, each column is a different output=>size(2) = length(outputs)

    # A function which can be used to update steady-state values
    # takes two arguments, the block itself (for vss, dss, Λss, yss), and a new input vector xss
    updatesteadystate!::Function

end

inputs(ha::HetBlock) = ha.inputs
outputs(ha::HetBlock) = ha.outputs
getT(ha::HetBlock) = ha.T
updatesteadystate!(ha::HetBlock, new_xss) = ha.updatesteadystate!(ha, new_xss)

# ===== Uses the fake news algorithm to compute the Jacobian =====

function jacobian(ha::HetBlock)

    xss, vss, dss, Λss, yss = ha.xss, ha.vss, ha.dss, ha.Λss, ha.yss
    derivs = get_derivs(xss, vss, dss, ha)

    dindices = axes(dss, 1)
    yindices = (length(dss)+1):size(derivs, 1)

    Es = zeros(size(dss, 1), ha.T-1) #[zeros(axes(dss)) for t in 1:ha.T]
    F  = zeros(ha.T, ha.T)

    allJ = Dict(
        (input, output) => zeros(ha.T, ha.T)
        for output in ha.outputs, input in ha.inputs
    )

    for outputindex in eachindex(ha.outputs)
        updatecurlyE!(Es, ha.T, outputindex, Λss, yss)
        for inputindex in eachindex(ha.inputs)
            @views updateF!(
                F, ha.T, Es, 
                derivs[dindices, :, inputindex],
                derivs[yindices[outputindex], :, inputindex]
            )
            updateJ!(
                allJ[ha.inputs[inputindex], ha.outputs[outputindex]],
                F
            )
        end
    end

    return allJ
end

# ===== Step 1: get 𝒟 and 𝒴 =====
function get_diffs!(diffs, dx, inputindex, xss, vss, dss, ha)

    # ===== Set up =====
    S = eltype(dx)
    tmps = ha.makecache(S)
    dindices = axes(dss, 1)
    yindices = (length(dss)+1):size(diffs, 1) # add y vals to the bottom

    Y = zeros(S, size(dss, 1), length(yindices))
    v = convert(Array{S}, vss)
    x = convert(Array{S}, xss)
    x[inputindex] += dx

    # ===== Computation =====
    # shock at s=0
    ha.iterate_block!(v, Y, view(diffs, dindices, 1), dss, x, tmps)
    mul!(view(diffs, yindices, 1), transpose(Y), dss) # (implement eq 12)
    # back to steady state inputs for rest (but v has changed)
    for s in 2:ha.T
        ha.iterate_block!(v, Y, view(diffs, dindices, s), dss, xss, tmps)
        mul!(view(diffs, yindices, s), transpose(Y), dss)
    end
    return diffs
end

function get_derivs(xss, vss, dss, ha)

    diffs = zeros(size(vss, 1) * size(vss, 2) + length(ha.outputs), ha.T)
    res   = zeros((size(vss, 1) * size(vss, 2) + length(ha.outputs)) * ha.T, length(xss))
    Threads.@threads for inputindex in axes(xss, 1)
        ForwardDiff.derivative!(
            view(res, :, inputindex),
            (y, dx) -> get_diffs!(y, dx, inputindex, xss, vss, dss, ha), 
            diffs, 0.0
        )
    end
    # Reshapes the Jacobian to 3 dimensions:
    #  (1) values of 𝒟 and 𝒴 (last #outputs elements)
    #  (2) s (0:T-1)
    #  (3) input
    reshape(res, length(ha.outputs) + length(dss), ha.T, length(xss))
end

# ===== Step 2: get ℰ =====
function updatecurlyE!(Es, T, outputindex, Λss, yss)
    Es[:, 1] .= view(yss, :, outputindex)
    for i in 2:T-1
        @views mul!(Es[:, i], Λss, Es[:, i-1])
    end
end

# ===== Step 3: make the fake news matrix =====
function updateF!(F, T, Es, Ds, Ys)
    F[1, :] .= Ys
    mul!(view(F, 2:T, :), transpose(Es), Ds)
    #=
    for s in 1:T
        F[1, s] = Ys[s]
        for t in 2:T
            # F[t, s] = @views dot(Es[:, t-1], Ds[:, s])
            # F[t, s] = transpose(Es[t-1]) * Ds[:, s]
        end
    end
    =#
    return F
end

# ===== Step 4: make the jacobian =====
function updateJ!(J, F)

    # Copy first row and column from the fake news matrix
    ax = axes(J)
    J[:, firstindex(ax[2])] .= F[:, 1]
    J[firstindex(ax[1]), :] .= F[1, :]
    # then build recursively
    for s in Iterators.drop(eachindex(ax[2]), 1) # all cols but first
        for t in Iterators.drop(eachindex(ax[1]), 1) # all rows but first
            J[t, s] = J[t-1, s-1] + F[t, s]
        end
    end
    return J
end