struct ModelGraph

    # A directed graph where each vertex and edge is associated with a T×T matrix
    # Matrices on edges are interpreted as partial Jacobians
    # Matrices on vertices are either total or general equilibrium Jacobians (as defined in the paper)
    # Not mutable! Can only update values, not the matrices themselves

    graph::SimpleDiGraph{Int64} 
    vars::Vector{Symbol} # list of all included variables
    varDict::Dict{Symbol, Int64} # mapping between vars and vertex indices
    vJ::Dict{Symbol, Matrix{Float64}} # vertex matrices
    eJ::Dict{Tuple{Symbol, Symbol}, Matrix{Float64}} # edge matrices
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

function ModelGraph(vars, blocks, steady_states)
    # Calculates the partial jacobians!
    T = getT(blocks[1]) # should check all blocks have the same T
    ModelGraph(
        makegraph(vars, blocks),
        vars,
        Dict(vars[i] => i for i in eachindex(vars)),
        Dict(v => zeros(T, T) for v in vars),
        merge((jacobian(blocks[i], steady_states[i]) for i in eachindex(blocks))...),
        T
    )
end

# ===== Methods for ModelGraphs =====
plotgraph(mg::ModelGraph) = gplot(mg.graph, nodelabel = mg.vars)

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

function getH!(H, start, target, mg)
    # ensure that all vertex matrices are
    # zero before starting accumulation
    resetnodematrices!(mg)
    forward_accumulate!(start, mg)
    H .= mg.vJ[target]
    return H
end

function getH(start, target, mg)
    H = zeros(mg.T, mg.T)
    getH!(H, start, target, mg)
    return H
end