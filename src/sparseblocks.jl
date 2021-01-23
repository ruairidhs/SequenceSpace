struct SparseBlock <: Block

    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    T::Int

    f!::Function # the function which updates a T×#out matrix given T×#in input matrix
    f!_reshaped::Function # version of f! which acts on vectors

    jac::SparseMatrixCSC{Float64, Int64} # encodes sparsity pattern
    colours::Vector{Int64} # colouring of jac

end

# Returns same function as f! but operates on vecs instead of matrices
# Needed for sparse diff
function vec2vec(f!, ax_out, ax_in)
    let ax_out = ax_out, ax_in = ax_in
        (ovec, ivec) -> f!(reshape(ovec, ax_out), reshape(ivec, ax_in))
    end
end

# Constructor which automatically creates f!_reshaped and the sparsity pattern
# Does not actually compute the jacobian, just initializes
function SparseBlock(inputs, outputs, f!, T)
    f!_reshaped = vec2vec(f!, (Base.OneTo(T), axes(outputs, 1)), (Base.OneTo(T), axes(inputs, 1)))
    sparsity_pattern = jacobian_sparsity(f!, zeros(T, length(outputs)), zeros(T, length(inputs)), verbose = false)
    jac = Float64.(sparse(sparsity_pattern))
    colours = matrix_colors(jac)

    return SparseBlock(
        inputs, outputs, T,
        f!, f!_reshaped, jac, colours
    )
end

inputs(sb::SparseBlock)  = sb.inputs
outputs(sb::SparseBlock) = sb.outputs
getT(sb::SparseBlock) = sb.T

function jacobian(sb::SparseBlock, steady_state)

    input_vals, = steady_state

    # Updates the full Jacobian (all inputs and outputs)
    # Then splits it into a separate matrix for each input, output pair

    forwarddiff_color_jacobian!(
        sb.jac, sb.f!_reshaped,
        reshape(repeat(input_vals, inner = sb.T), sb.T, :), # repeat input in each period
        colorvec = sb.colours
    )

    row_indices = [s:s+sb.T-1 for s in 1:sb.T:length(sb.outputs)*sb.T]
    col_indices = [s:s+sb.T-1 for s in 1:sb.T:length(sb.inputs)*sb.T]
    return Dict(
        (sb.inputs[i], sb.outputs[o]) => sb.jac[row_indices[o], col_indices[i]]
        for o in eachindex(sb.outputs), i in eachindex(sb.inputs)
    )

end