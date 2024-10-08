# SequenceSpace

This module contains functions to implement the algorithms presented in [Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models](http://web.stanford.edu/~aauclert/sequence_space_jacobian.pdf), by Auclert, Bardóczy, Rognlie and Straub (2020).

WORK IN PROGRESS.

## Running the code
The module and all core dependencies can be installed by entering the package manager in the Julia REPL with `]`, then `add https://github.com/ruairidhs/SequenceSpace`.
The code is only tested for Julia 1.5.0.

**Important:**
This package relies on `ForwardDiff.jl` for derivatives.
It requires enabling ForwardDiff's NaN-safe mode by setting the `NANSAFE_MODE_ENABLED` constant to true in ForwardDiff's source. The constant is located in `src\prelude.jl` within ForwardDiff.
This issue is explained [here](http://www.juliadiff.org/ForwardDiff.jl/stable/user/advanced/#Fixing-NaN/Inf-Issues).

## Examples
The subdirectory `examples` contains well-commented code that replicates the Krusell-Smith and one-asset HANK models from the paper, illustrating how to use the package.
These files require some additional packages which are not dependencies of the package, e.g. for plotting and Bayesian estimation.
The required additional packages are listed at the top of each file.

## src
The subdirectory `src` defines the functions and types required for computing the jacobians and likelihoods and can be loaded with `using SequenceSpace`.
