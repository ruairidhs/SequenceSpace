module SequenceSpace


using ForwardDiff
using LinearAlgebra
using OffsetArrays

# Used in sparseblocks.jl
using SparseArrays
using SparseDiffTools
using SparsityDetection

using LightGraphs
using GraphPlot

export Block,
       inputs,
       outputs,
       getT,
       jacobian,
       SparseBlock,
       HetBlock,
       ModelGraph,
       plotgraph,
       makeG,
       generaleqJacobians,
       fastinterp!

include("fastinterp.jl")

# Each subtype must implement the following methods:
#   - inputs
#   - outputs
#   - jacobian
#   - getT
abstract type Block end

include("sparseblocks.jl")
include("hetblocks.jl")
include("graphs.jl")

end # module
