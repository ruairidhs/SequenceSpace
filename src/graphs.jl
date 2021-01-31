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

    graph::SimpleDiGraph{Int64} # graph representing the input-output structure
    vJ::Dict{Symbol, Union{Matrix{Float64}, ShiftMatrix}} # vertex matrices
    eJ::Dict{Tuple{Symbol, Symbol}, Union{Matrix{Float64}, ShiftMatrix}} # edge matrices
    
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

function ModelGraph(blocks, unknowns, exog, eqvars, simplenodes)

    # does not compute the jacobians, just initializes them
    # have to manually specify which nodes need to be simple

    T = getT(blocks[1])
    vars = union([vcat(inputs(block), outputs(block)) for block in blocks]...)
    g = makegraph(vars, blocks)

    @assert !is_cyclic(g) "Graph is cyclic"
    @assert length(unknowns) == length(eqvars) "Number of unknowns does not equal number of targets"

    eJ = Dict{Tuple{Symbol, Symbol}, Union{Matrix{Float64}, ShiftMatrix}}()
    for block in blocks
        for inp in inputs(block), outp in outputs(block)
            if typeof(block) == HetBlock
                # hetblock -> dense matrix
                push!(eJ, (inp, outp) => zeros(T, T))
            elseif typeof(block) == SimpleBlock
                push!(eJ, (inp, outp) => shiftzero())
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
        Dict(v => (v ∈ simplenodes ? shiftzero() : zeros(T, T)) for v in vars),
        eJ,
        T
    )

end

# ===== Methods for ModelGraphs =====
function plotgraph(mg::ModelGraph)
    nodecolours = [
        colorant"SeaGreen", # unknowns
        colorant"Salmon",   # exogenous
        colorant"Thistle",     # targets
        colorant"SkyBlue",  # base
    ]
    membership = [
        v ∈ mg.unknowns ? 1 :
        v ∈ mg.exog     ? 2 :
        v ∈ mg.eqvars   ? 3 : 4
        for v in mg.vars
    ]
    sorted_graph = 
    gplot(mg.graph, nodelabel = mg.vars, nodefillc=nodecolours[membership], layout=circular_layout)
end

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
            if typeof(block) == SimpleBlock
                mg.eJ[(inp, outp)] = j[(inp, outp)] # simple blocks overwrite
            else
                mg.eJ[(inp, outp)] .= j[(inp, outp)]
            end
        end
    end
end

function resetnodematrices!(mg::ModelGraph)
    # Resets all node matrices to zero
    for (vi, mat) in pairs(mg.vJ)
        if typeof(mat) == ShiftMatrix
            mg.vJ[vi] = shiftzero()
        else
            fill!(mat, 0)
        end
    end
end

function resetnodematrices!(mg::ModelGraph, simplenodes)
    # this version can be used to change which nodes are simple
    # useful for swapping between forward and backwards accumulation
    for var in mg.vars
        if var ∈ simplenodes
            mg.vJ[var] = shiftzero()
        else
            mg.vJ[var] = zeros(mg.T, mg.T)
        end
    end
end

function accumulategraph!(start, mg::ModelGraph, ::Val{:forward})
    # FORWARD
    # set starting node to identity
    if typeof(mg.vJ[start]) == ShiftMatrix
        mg.vJ[start] = shiftidentity()
    else
        mg.vJ[start] .= I(mg.T)
    end

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

function accumulategraph!(start, mg::ModelGraph, ::Val{:backward})
    # BACKWARD
    if typeof(mg.vJ[start]) == ShiftMatrix
        mg.vJ[start] = shiftidentity()
    else
        mg.vJ[start] .= I(mg.T)
    end

    subgraph, indices = induced_subgraph( # changed dir -> :in
        mg.graph, neighborhood(mg.graph, mg.varDict[start], length(mg.vars), dir=:in)
    )
    for src in Iterators.reverse(topological_sort_by_dfs(subgraph))
        srcvar = mg.vars[indices[src]]
        for dst in inneighbors(subgraph, src) # changed outneighbors to inneighbors
            dstvar = mg.vars[indices[dst]]
            # change multiplication order
            mul!(mg.vJ[dstvar], mg.vJ[srcvar], mg.eJ[dstvar, srcvar], 1.0, 1.0)
        end
    end
end

# ===== Build total Jacobians and Gᵘᶻ based on forward accumulation =====
function forwardgetJ!(Hu, start, targets, mg)

    # Computes total Jacobians for start => targets
    # Stores them in a stacked matrix
    # H = Array(T * length(targets), T)

    resetnodematrices!(mg)
    accumulategraph!(start, mg, Val(:forward))

    loc = 1
    for ti in eachindex(targets)
        Hu[loc:loc+mg.T-1, :] .= mg.vJ[targets[ti]]
        loc += mg.T
    end

end

function forwardfillH!(H, starts, targets, mg)
    T, ns, nt = mg.T, length(starts), length(targets)
    loc = 1
    for si in eachindex(starts)
        forwardgetJ!(view(H, :, loc:loc+T-1), starts[si], targets, mg)
        loc += T
    end
    return H
end

function fillG!(G, Hu, Hx, mg, ::Val{:forward})
    forwardfillH!(Hu, mg.unknowns, mg.eqvars, mg)
    forwardfillH!(Hx, mg.exog, mg.eqvars, mg)
    ldiv!(G, lu!(Hu), -Hx)
end

# ===== Build total Jacobians and Gᵘᶻ based on backward accumulation =====

function backwardfillH!(Hu, Hx, mg)
    T = mg.T
    nt, nu, nx = length(mg.eqvars), length(mg.unknowns), length(mg.exog)

    loc = 1
    for ti in eachindex(mg.eqvars)
        resetnodematrices!(mg)
        accumulategraph!(mg.eqvars[ti], mg, Val(:backward)) # only one backwards accumulation per target
        cloc = 1
        for ui in 1:nu
            view(Hu, loc:loc+T-1, cloc:cloc+T-1) .= mg.vJ[mg.unknowns[ui]]
            cloc += T
        end
        cloc = 1
        for xi in 1:nx
            view(Hx, loc:loc+T-1, cloc:cloc+T-1) .= mg.vJ[mg.exog[xi]]
            cloc += T
        end
        loc += T
    end
end

function fillG!(G, Hu, Hx, mg, ::Val{:backward})
    backwardfillH!(Hu, Hx, mg)
    ldiv!(G, lu!(Hu), -Hx)
end

# ===== Create the general equilibrium jacobians for other variables =====

function geneqjacobians!(Gs, G, Hu, Hx, mg, direction)
    # version with preallocated arrays for performance
    T = mg.T

    fillG!(G, Hu, Hx, mg, direction)

    # for unknowns take the relevant rows from bigG
    loc = 1
    for ui in eachindex(mg.unknowns)
        Gs[mg.unknowns[ui]] .= G[loc:loc+T-1, :]
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

function geneqjacobians(mg, direction)

    # simple version which also allocates the memory
    T  = mg.T
    nt, nu, nx = length(mg.eqvars), length(mg.unknowns), length(mg.exog)
    Hu = zeros(T * nt, T * nu)
    Hx = zeros(T * nt, T * nx)
    G  = zeros(T * nu, T * nx)
    Gs = Dict( # initialize
        var => zeros(T, T * length(mg.exog)) for var in mg.vars
    )

    return geneqjacobians!(Gs, G, Hu, Hx, mg, direction)
end