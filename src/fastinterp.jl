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
    @assert issorted(xs)
    @assert issorted(nodes)
    start = 1
    for i in eachindex(res)
        res[i], start = fi2(xs[i], start, nodes, vals)
    end
end


#=
# ===== generate test data =====
function gen_data(T)
    nodes = sort(rand(T) * 10)
    vals  = log.(nodes)
    xs    = sort(rand(T*10) * 10)
    return nodes, vals, xs
end

# ===== test for correctness =====

@testset "fastinterp" begin
    nodes, vals, xs = gen_data(100)
    itp  = interpolate((nodes,), vals, Gridded(Linear()))
    etpl = extrapolate(itp, Line())

    res  = similar(xs)

    # Test fastinterp!
    fastinterp!(res, xs, nodes, vals)
    @test etpl.(xs) ≈ res

    # Test fastinterp2!
    res   = similar(xs)
    coefs = similar(vals)
    fastinterp2!(res, coefs, xs, nodes, vals)
    @test etpl.(xs) ≈ res
end

# ===== Test speed =====

function time_itp(res, nodes, vals, xs)
    itp = extrapolate(
        interpolate((nodes,), vals, Gridded(Linear())), Line()
    )
    res .= itp.(xs)
end

function run_benchmark(T)

    nodes, vals, xs = gen_data(T)
    res = similar(xs)
    coefs = similar(vals)
    fill!(res, 0)
    fill!(coefs, 0)

    io = IOBuffer()

    #=
    println("=====Interpolations.jl, $T=====")
    b = @benchmark time_itp($res, $nodes, $vals, $xs)
    show(io, "text/plain", b)
    println(String(take!(io)))
    =#
    fill!(res, 0)
    println("=====fastinterp!, $T=====")
    b = @benchmark fastinterp!($res, $xs, $nodes, $vals)
    show(io, "text/plain", b)
    println(String(take!(io)))

    #=
    fill!(res, 0)
    coefs = similar(vals)
    fill!(coefs, 0)
    println("=====fastinterp2!, $T=====")
    b = @benchmark fastinterp2!($res, $coefs, $xs, $nodes, $vals)
    show(io, "text/plain", b)
    println(String(take!(io)))
    =#

end
=#