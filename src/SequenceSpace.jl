module SequenceSpace


using ForwardDiff
using FFTW
using LinearAlgebra
using OffsetArrays

# Used in sparseblocks.jl
using SparseArrays
using StaticArrays
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
       updatesteadystate!,
       ModelGraph,
       updatepartialJacobians!,
       plotgraph,
       makeG,
       generaleqJacobians,
       fastinterp!,
       fastcov,
       makefftcache,
       makeinput,
       updateMAcoefficients!,
       makeV,
       _likelihood
       
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

include("likelihood.jl")

end # module
