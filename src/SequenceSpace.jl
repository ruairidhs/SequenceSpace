module SequenceSpace

using ForwardDiff
using LinearAlgebra

# Used in sparseblocks.jl
using SparseArrays
using SparseDiffTools
using SparsityDetection

export Block,
       inputs,
       outputs,
       jacobian,
       SparseBlock

# maybe some consts?

include("fastinterp.jl")

# Each subtype must implement the following methods:
#   - inputs
#   - outputs
#   - jacobian
abstract type Block end

include("sparseblocks.jl")

end # module
