module SequenceSpace

using ForwardDiff
using FFTW
using LinearAlgebra
using LoopVectorization
using SparseArrays
using LightGraphs
using GraphPlot
using Colors

export Block,
       inputs,
       outputs,
       jacobian,
       SimpleBlock,
       @simpleblock,
       HetBlock,
       updatesteadystate!,
       ModelGraph,
       plotgraph,
       updatepartialJacobians!,
       geneqjacobians,
       geneqjacobians!,
       fastinterp!,
       make_likelihood,
       getcorrelations

# Each subtype must implement the following methods:
#   - inputs
#   - outputs
#   - jacobian
abstract type Block end

include("fastinterp.jl")
include("hetblocks.jl")
include("shiftmatrices.jl")
include("simpleblocks.jl")
include("graphs.jl")
include("likelihood.jl")

end # module
