# Code for replicating the paper's Krusell-Smith example

const T = 10

# ===== Firms block =====
const δ = 0.025
const α = 0.11
const kss = 3.14
const zss = 1.0

r(k, z) = α * z * k^(α-1.0) - δ
w(k, z) = (1.0-α) * z * k^α

function firms!(output, input, k0)

    # inputs: k, z
    # outputs: r, w

    T = size(input, 1)

    output[1, 1] = r(k0, input[1, 2])
    output[1, 2] = w(k0, input[1, 2])

    for t in 2:T
        k, z = input[t-1, 1], input[t, 2]
        output[t, 1] = r(k, z)
        output[t, 2] = w(k, z)
    end

    return output
end

# ===== Testing =====
#=
function firms!_vec(output, input)
    T = length(input) ÷ 2
    # r[1]
    output[1] = r(kss, input[T+1])
    for t in 2:T
        k, z = input[t-1], input[t+T]
        output[t] = r(k, z)
        output[t+T] = w(k, z)
    end
    return output
end

input_mat = reshape(repeat([kss, zss], inner = T), T, :)

firms_block = SparseBlock([:k, :z], [:r, :w], (o, i) -> firms!(o, i, kss), T)
inputs(firms_block)
outputs(firms_block)
j = jacobian(firms_block, [3.14, 1.0])
j[(:z, :w)]
=#