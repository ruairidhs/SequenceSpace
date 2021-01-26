struct ModelGraph

    # A directed graph where each vertex and edge is associated with a T×T matrix
    # Matrices on edges are interpreted as partial Jacobians
    # Matrices on vertices are either total or general equilibrium Jacobians (as defined in the paper)
    # Not mutable! Can only update values, not the matrices themselves

    vars::Vector{Symbol}       # vertex labels
    varDict::Dict{Symbol, Int} # mapping between var and vertex index

    blocks::Vector{Block} # list of blocks in the model

    unknowns::Vector{Symbol} # list of symbols representing unknowns
    exog::Vector{Symbol}     # list of symbols representing exogenous variables
    eqvars::Vector{Symbol}   # list of symbols representing equilibrium conditions == 0

    graph::SimpleDiGraph{Int64}       # graph representing the input-output structure
    vJ::Dict{Symbol, AbstractMatrix{Float64}} # vertex matrices
    eJ::Dict{Tuple{Symbol, Symbol}, AbstractMatrix{Float64}} # edge matrices
    
    T::Int64 # Matrix dimension

end

# Given a list of variables and blocks, creates a directed graph
function makegraph(vars, blocks)

    d = Dict(vars[i] => i for i in eachindex(vars))
    g = DiGraph(length(vars))

    # add all the edges
    for block in blocks
        for input in inputs(block), output in outputs(block)
            add_edge!(g, d[input], d[output])
        end
    end

    return g
end

function ModelGraph(blocks, unknowns, exog, eqvars)

    # does not compute the jacobians, just initializes them

    T = getT(blocks[1])
    vars = union([vcat(inputs(block), outputs(block)) for block in blocks]...)
    g = makegraph(vars, blocks)

    @assert length(unknowns) == length(eqvars) # needed for invertibility
    @assert all([getT(block) == T for block in blocks]) # check all Ts are the same

    eJ = Dict{Tuple{Symbol, Symbol}, AbstractMatrix{Float64}}()
    for block in blocks
        for inp in inputs(block), outp in outputs(block)
            if typeof(block) == HetBlock
                # hetblock -> dense matrix
                push!(eJ, (inp, outp) => zeros(T, T))
            elseif typeof(block) == SparseBlock
                push!(eJ, (inp, outp) => spzeros(T, T))
            else
                error("Unsupported block type")
            end
        end
    end

    ModelGraph(
        vars,
        Dict(vars[i] => i for i in eachindex(vars)),
        blocks,
        unknowns, exog, eqvars,
        g,
        Dict(v => zeros(T, T) for v in vars),
        eJ,
        T
    )

end

# ===== Methods for ModelGraphs =====
plotgraph(mg::ModelGraph) = gplot(mg.graph, nodelabel = mg.vars)

function updatesteadystate!(mg::ModelGraph, new_steadystate)
    # new_steadystate is a vector containing new steady state values
    # in the same order as the blocks in mg.block
    @assert length(mg.blocks) == length(new_steadystate)
    for (block, xss) in zip(mg.blocks, new_steadystate)
        updatesteadystate!(block, xss)
    end

end

function updatepartialJacobians!(mg::ModelGraph)
    for block in mg.blocks
        j = jacobian(block)
        for inp in inputs(block), outp in outputs(block)
            mg.eJ[(inp, outp)] .= j[(inp, outp)]
        end
    end
end

function resetnodematrices!(mg::ModelGraph)
    # Resets all node matrices to zero
    for v in mg.vars
        fill!(mg.vJ[v], 0)
    end
end

function forward_accumulate!(start, mg::ModelGraph)

    # set starting node to identity
    mg.vJ[start] .= I(mg.T)

    # gets the subgraph containing all descendants of start 
    subgraph, indices = induced_subgraph(
        mg.graph, neighborhood(mg.graph, mg.varDict[start], length(mg.vars))
    )
    # go through vertices in topological order and build the jacobian
    for src in topological_sort_by_dfs(subgraph)
        srcvar = mg.vars[indices[src]]
        for dst in outneighbors(subgraph, src)
            dstvar = mg.vars[indices[dst]]
            mul!(mg.vJ[dstvar], mg.eJ[srcvar, dstvar], mg.vJ[srcvar], 1.0, 1.0)
        end
    end
    
end

function getJ!(Hu, start, targets, mg)

    # Computes total Jacobians for start => targets
    # Stores them in a stacked matrix
    # H = Array(T * length(targets), T)

    resetnodematrices!(mg)
    forward_accumulate!(start, mg)

    loc = 1
    for ti in eachindex(targets)
        Hu[loc:loc+mg.T-1, :] .= mg.vJ[targets[ti]]
        loc += mg.T
    end

end

function makeH(starts, targets, mg)
    T, ns, nt = mg.T, length(starts), length(targets)
    H = zeros(T * nt, T * ns)
    loc = 1
    for si in eachindex(starts)
        getJ!(view(H, :, loc:loc+T-1), starts[si], targets, mg)
    end
    return H
end

function makeG(mg)
    - makeH(mg.unknowns, mg.eqvars, mg) \ makeH(mg.exog, mg.eqvars, mg)
end

function generaleqJacobians(fullG, mg)
    T  = mg.T
    Gs = Dict( # initialize
        var => zeros(T, T * length(mg.exog)) for var in mg.vars
    )
    # for unknowns take the relevant rows from bigG
    loc = 1
    for ui in eachindex(mg.unknowns)
        Gs[mg.unknowns[ui]] .= fullG[loc:loc+T-1, :]
        loc += T
    end
    # for exogenous vars set the relevant cols to identity
    loc = 1
    for ei in eachindex(mg.exog)
        Gs[mg.exog[ei]][:, loc:loc+T-1] .= I(T)
        loc += T
    end

    # and then forward accumulate across the whole graph for the rest
    for src in topological_sort_by_dfs(mg.graph)
        srcvar = mg.vars[src]
        for dst in outneighbors(mg.graph, src)
            dstvar = mg.vars[dst]
            mul!(Gs[dstvar], mg.eJ[srcvar, dstvar], Gs[srcvar], 1.0, 1.0)
        end
    end

    return Gs
end