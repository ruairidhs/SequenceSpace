# Fast methods for linear interpolation when the input vector is increasing

function _innerinterp(x, n1, n2, v1, v2)
    muladd((x - n1) / (n2 - n1), v2 - v1, v1)
end

function fi2(x, start, nodes, vals)

    if start == length(nodes)
        return _innerinterp(x, nodes[end-1], nodes[end], vals[end-1], vals[end]), length(nodes)
    elseif start == 1
        if x <= nodes[1]
            return _innerinterp(x, nodes[1], nodes[2], vals[1], vals[2]), 1
        else
            @inbounds for i in 2:length(nodes)-1
                if x <= nodes[i]
                    return _innerinterp(x, nodes[i-1], nodes[i], vals[i-1], vals[i]), i
                end
            end
        end
    else
        @inbounds for i in start:length(nodes)-1
            if x <= nodes[i]
                return _innerinterp(x, nodes[i-1], nodes[i], vals[i-1], vals[i]), i
            end
        end
    end
    return _innerinterp(x, nodes[end-1], nodes[end], vals[end-1], vals[end]), length(nodes)
end

function fastinterp!(res, xs, nodes, vals)
    # @assert issorted(xs) "fastinterp! xs not sorted"
    # @assert issorted(nodes) "fastinterp! nodes not sorted"
    start = 1
    @inbounds for i in eachindex(res)
        res[i], start = fi2(xs[i], start, nodes, vals)
    end
end