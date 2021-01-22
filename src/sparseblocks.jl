using Random,
      ForwardDiff,
      OffsetArrays,
      SparseDiffTools,
      SparseArrays,
      SparsityDetection,
      Test

# ===== Make the structure =====
abstract type Block end

# This function maps a Mat->Mat function into a Vec->Vec function which
# performs the same operation. Needed as sparse_differentiation only works for 
# vec -> vec.
function vec2vec(f, output, input)
    ax_out, ax_in = axes(output), axes(input)
    let ax_out = ax_out, ax_in = ax_in
        (ovec, ivec) -> f(reshape(ovec, ax_out), reshape(ivec, ax_in))
    end
end

struct SparseBlock <: Block

    f!::Function 
    # Maps a time series of inputs to outputs, arguments: output, input
    #   - output: T × #Outputs matrix
    #   - input: T × #Inputs matrix
    f!_reshaped::Function 
    # Same function as f! but rewritten to operate on vec(output), vec(input)
    jac::SparseMatrixCSC{Float64, Int64} # Jacobian incorporating the scarsity pattern
    colours::Vector{Int64} # Vector incorporating the colour pattern of the jacobian

    # Output and input are examples that could be passed to f!
    SparseBlock(f!, output, input) = begin
        f!_reshaped = vec2vec(f!, output, input)
        sp = jacobian_sparsity(f!, output, input)
        jac = Float64.(sparse(sp))
        colours = matrix_colors(jac)
        return new(
            f!, f!_reshaped, jac, colours
        )
    end

end

# Calling the block just runs the function
(sb::SparseBlock)(output, input) = sb.f!(output, input)

# Computing the Jacobian incorporates sparsity
function jacobian!(sb::SparseBlock, inputs)
    forwarddiff_color_jacobian!(sb.jac, sb.f!_reshaped, inputs, colorvec=sb.colours)
end

# ===== Tests =====

const δ = 0.025
const α = 0.11
const L = 0.89

const kmean, zmean = 3.14, 1.0
const kσ, zσ = 0.01, 0.01
function make_inputs(T)
    ks = kmean .* exp.(kσ .* randn(T))
    zs = zmean .* exp.(zσ .* randn(T))
    return OffsetArray([ks zs], -1, 0)
end

function ks_firms!(output, input, k0, T)
    # output is a matrix t, output_vars:(r,w))
    # input is a matrix t, input_vars:(k, z)
    # first period depends on k0, not the input matrix
    
    output[0, 1] = α * input[0, 2] * (k0 / L)^(α - 1.0) - δ 
    output[0, 2] = (1.0-α) * input[0, 2] * (k0 / L)^α
    for t in 1:T-1
        k, z = input[t-1, 1], input[t, 2]
        output[t, 1] = α * z * (k / L)^(α - 1.0) - δ # r
        output[t, 2] = (1.0-α) * z * (k / L)^α # w
    end
    return output
end

const T = 300
inputs = make_inputs(T)
inputss = OffsetArray([repeat([kmean], T) repeat([zmean], T)], -1, 0)

# Evaluate ks_firms! and jacobian directly and using the sparse block
outputs_naive = similar(inputs)
jacob_naive_base = similar(inputs)
ks_firms!(outputs_naive, inputs, kmean, T)
jacob_naive  = ForwardDiff.jacobian((o, i) -> ks_firms!(o, i, kmean, T), jacob_naive_base, inputss)

outputs_sparse = similar(inputs)
sparse_block = SparseBlock(
    (output, input) -> ks_firms!(output, input, kmean, T),
    outputs_sparse, inputs
)
sparse_block(outputs_sparse, inputs)
jacobian!(sparse_block, inputss)
jacob_sparse = Matrix(sparse_block.jac)

@testset "Sparse Blocks" begin
    @test outputs_naive ≈ outputs_sparse
    @test jacob_naive ≈ jacob_sparse
end