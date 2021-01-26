struct SparseBlock <: Block

    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    T::Int

    # f!(outputs, inputs, xss)
    #   - outputs: T×#out matrix
    #   - inputs: T×#in matrix
    #   - xss: Inputs at steady-state, in case the function depends on the steady-state not just inputs
    #   - i.e., to evaluate f(zₜ, k_(t-1)) at t=0, need steady-state k
    f!::Function 
    f!_reshaped::Function # version of f! which acts on vec(outputs), vec(inputs)

    xss::Array{Float64, 1} # Steady-state inputs

    jac::SparseMatrixCSC{Float64, Int64} # encodes sparsity pattern
    colours::Vector{Int64} # colouring of jac

end

# Returns same function as f! but operates on vecs instead of matrices
# Needed for sparse diff
function vec2vec(f!, ax_out, ax_in)
    let ax_out = ax_out, ax_in = ax_in
        (ovec, ivec, xss) -> f!(reshape(ovec, ax_out), reshape(ivec, ax_in), xss)
    end
end

# Constructor which automatically creates f!_reshaped and the sparsity pattern
# Does not actually compute the jacobian, just initializes
function SparseBlock(inputs, outputs, xss, f!, T)

    f!_reshaped = vec2vec(f!, (Base.OneTo(T), axes(outputs, 1)), (Base.OneTo(T), axes(inputs, 1)))
    sparsity_pattern = jacobian_sparsity(
        (o, i) -> f!(o, i, xss),
        zeros(T, length(outputs)),
        zeros(T, length(inputs)),
        verbose = false
    )
    jac = Float64.(sparse(sparsity_pattern))
    colours = matrix_colors(jac)

    return SparseBlock(
        inputs, outputs, T,
        f!, f!_reshaped, xss, jac, colours
    )
end

inputs(sb::SparseBlock)  = sb.inputs
outputs(sb::SparseBlock) = sb.outputs
getT(sb::SparseBlock) = sb.T
updatesteadystate!(sb::SparseBlock, new_xss) = (sb.xss .= new_xss)
 
function jacobian(sb::SparseBlock)

    # Updates the full Jacobian (all inputs and outputs)
    # Then splits it into a separate matrix for each input, output pair

    forwarddiff_color_jacobian!(
        sb.jac, (o, i) -> sb.f!_reshaped(o, i, sb.xss),
        reshape(repeat(sb.xss, inner = sb.T), sb.T, :), # repeat input in each period
        colorvec = sb.colours
    )

    row_indices = [s:s+sb.T-1 for s in 1:sb.T:length(sb.outputs)*sb.T]
    col_indices = [s:s+sb.T-1 for s in 1:sb.T:length(sb.inputs)*sb.T]
    return Dict(
        (sb.inputs[i], sb.outputs[o]) => sb.jac[row_indices[o], col_indices[i]]
        for o in eachindex(sb.outputs), i in eachindex(sb.inputs)
    )

end