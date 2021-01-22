using LightGraphs
using GraphPlot
using MetaGraphs
using SimpleWeightedGraphs
using LinearAlgebra

struct JacobianGraph
    # Not mutable! Can only update values, not the matrices themselves
    graph::SimpleDiGraph{Int64} # Directed graph
    totalJ::Vector{Matrix{Float64}} # Matrix (or I) for each vertex
    partialJ::Dict{Tuple{Int64, Int64}, Matrix{Float64}} # Matrix (or I) for each edge
    T::Int64  # Dimension of jacobians is T×T
end

function JacobianGraph(g::SimpleDiGraph, T)
    # Constructor which initializes a JacobianGraph
    totalJ = [(1.0*I)(T) for vertex in vertices(g)]
    partialJ = Dict(
        (src(edge), dst(edge)) => (1.0*I)(T) for edge in edges(g)
    )
    JacobianGraph(g, totalJ, partialJ, T)
end 

function forward_accumulate!(g::JacobianGraph)
    
    # Go through vertices in topological order
    for src in topological_sort_by_dfs(g.graph)
        # For each vertex, get its forward-neighbours
        for dst in outneighbors(g.graph, src)
            # Replace totalJ[dst] with partialJ[(src, dst)] * totalJ[src] + totalJ[dst]
            mul!(g.totalJ[dst], g.partialJ[(src, dst)], g.totalJ[src], 1, 1)
        end
    end
end



# ===== Testing ====


simple_graph = DiGraph(4)
add_edge!(simple_graph, 3, 1)
add_edge!(simple_graph, 3, 2)
add_edge!(simple_graph, 1, 4)
add_edge!(simple_graph, 2, 4)
gplot(simple_graph, nodelabel = 1:4)

jac = JacobianGraph(simple_graph, 10)

forward_accumulate!(jac)

# ===== Krusell-Smith =====

G = DiGraph(7)

# add the edges
add_edge!(G, 1, 2)
add_edge!(G, 1, 3)

add_edge!(G, 2, 4)
add_edge!(G, 2, 5)

add_edge!(G, 3, 4)
add_edge!(G, 3, 5)

add_edge!(G, 4, 6)
add_edge!(G, 5, 6)

add_edge!(G, 2, 7)
add_edge!(G, 6, 7)

labels = [
    "⊥",
    "K",
    "Z",
    "r", 
    "w",
    "𝒦",
    "H"
]

gplot(G, nodelabel = labels)

simple_graph = DiGraph(4)
add_edge!(simple_graph, 3, 1)
add_edge!(simple_graph, 3, 2)
add_edge!(simple_graph, 1, 4)
add_edge!(simple_graph, 2, 4)
gplot(simple_graph, nodelabel = 1:4)



res = bfs_tree(simple_graph, 3)
gplot(res)

Js = [
    [1.0, 1.0],
    [2.0],
    [3.0],
    []
]

Vs = [
    1.0,
    1.0,
    1.0,
    1.0
]

function forward_accumulate!(G, Vs, Js)
    
end

# dictionaries

d = Dict((1, 2) => 4.0, (2, 1) => 2.0)

# ==== making the graph automagically =====
vars = [:z, :k, :r, :w, :𝓀, :h]
# block :: Pair(inputs, outputs)
firms_block = ([:z, :k], [:r, :w])
ha_block    = ([:r, :w], [:𝓀])
eq_block    = ([:k, :𝓀], [:h])
blocks = [firms_block, ha_block, eq_block]

ks_graph = DiGraph(length(vars))
for vi in eachindex(vars)
    set_prop!(ks_graph, vi, :name, vars[vi])
end
set_indexing_prop!(ks_graph, :name)
for block in blocks
    for input in block[1], output in block[2]
        add_edge!(ks_graph, ks_graph[input, :name], ks_graph[output, :name])
    end
end
gplot(ks_graph, nodelabel = vars)

function makegraph(vars, blocks)
    # Map between vars and indices
    d = Dict(vars[i] => i for i in eachindex(vars))
    g = DiGraph(length(vars)) # make the base graph

    # Now add all the edges
    for block in blocks
        for input in block[1], output in block[2]
            add_edge!(g, d[input], d[output])
        end
    end

    return g
    
end