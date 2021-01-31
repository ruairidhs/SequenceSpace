struct SimpleBlock <: Block

    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    xss::Vector{Float64}
    f::Function # shifted
    minmaxs::Vector{Tuple{Int, Int}}
    arglengths::Vector{Int}

end

SimpleBlock(inputs, outputs, xss, f, minmaxs) = SimpleBlock(
    inputs, outputs, xss, f, minmaxs,
    [p[2] - p[1] + 1 for p in minmaxs]
)

# Implement the @simpleblock macro to automatically shift function arguments
function searchrefs(f, exp::Expr, sym)
    # Finds all reference expressions and applies f to them
    # if sym = :x, then this finds e.g. x[3]
    if exp.head == :ref && exp.args[1] == sym 
        f(exp)
    else
        for subobject in exp.args
            searchrefs(f, subobject, sym)
        end
    end
end
function searchrefs(f, obj, sym)
    return nothing
end

function find_indices(obj, sym)
    pile = Set{Int}()
    searchrefs(ref -> push!(pile, ref.args[2]), obj, sym)
    return pile
end

function shiftindices(exp::Expr, sym, shift)
    searchrefs(ref -> ref.args[2] += shift, exp, sym)
    return exp
end

function getfunctionargs(exp::Expr)
    @assert exp.head == :-> "Must provide anonymous function!"
    if typeof(exp.args[1]) == Symbol
        return [exp.args[1]] # single argument function
    else 
        return exp.args[1].args # multi-argument function
    end
end

macro simpleblock(inputs, outputs, xss, funcdef)
    # funcdef must be anonymous function!
    # need to escape the function definition to capture any variables
    # in definition scope
    args = getfunctionargs(esc(funcdef).args[1])
    minmaxs = Tuple{Int, Int}[]
    for arg in args
        inds = find_indices(esc(funcdef), arg)
        @assert !isempty(inds) "Function must depend on arguments!"
        extremes = extrema(inds)
        push!(minmaxs, extremes)
        # shift indices so they start at 1
        shiftindices(esc(funcdef), arg, 1 - extremes[1])
    end
    return :(SimpleBlock(
        $(esc(inputs)), $(esc(outputs)), $(esc(xss)), $(esc(funcdef)), $minmaxs
    ))
end

inputs(sb::SimpleBlock)  = sb.inputs
outputs(sb::SimpleBlock) = sb.outputs
updatesteadystate!(sb::SimpleBlock, new_xss) = (sb.xss .= new_xss)

# now get the jacobian

function _jacobian(sb::SimpleBlock, input_index)
    
    raw_jac = ForwardDiff.jacobian(
        x -> (args=[
            i == input_index ? x : repeat([sb.xss[i]], sb.arglengths[i]) for i in eachindex(sb.xss)
        ]; sb.f(args...)), repeat([sb.xss[input_index]], sb.arglengths[input_index])
    )

    return Dict(
        (sb.inputs[input_index], sb.outputs[oi]) => toshiftmat(raw_jac[oi, :], sb.minmaxs[input_index][1])
        for oi in eachindex(sb.outputs)
    )

end

function toshiftmat(v, min_index)
    # return deletezeros!(ShiftMatrix(Dict(
    #     z[1] => z[2] for z in zip(Iterators.countfrom(min_index), v)
    #)))
    return ShiftMatrix(Dict(
        z[1] => z[2] for z in zip(Iterators.countfrom(min_index), v)
    ))
end

function jacobian(sb::SimpleBlock)
    merge((_jacobian(sb, i) for i in eachindex(sb.inputs))...)
end

#endregion