
import LinearAlgebra: mul!
import Base: *, +
import SparseArrays: sparse

#region Define ShiftMatrix =====

# Dictionary based implementation of shift matrices
struct ShiftMatrix
    shifts::Dict{Int, Float64} # keys are shifts, values are scales
end

# Define addition and multiplication of shift matrices
+(A::ShiftMatrix, B::ShiftMatrix) = ShiftMatrix(mergewith(+, A.shifts, B.shifts))
shiftprod(a::Pair{Int, Float64}, b::Pair{Int, Float64}) = (a[1] + b[1], a[2] * b[2])
*(a::Pair{Int, Float64}, B::ShiftMatrix) = ShiftMatrix(Dict(shiftprod(a, b) for b in pairs(B.shifts)))
*(A::ShiftMatrix, B::ShiftMatrix) = mapreduce(a -> a * B, +, pairs(A.shifts))

function shiftidentity()
    return ShiftMatrix(Dict(0 => 1.0))
end
function shiftzero()
    return ShiftMatrix(Dict(0 => 0.0))
end

function sparse(A::ShiftMatrix, T)
    # creates a square sparse matrix of dimensions T * T
    spdiagm(
        (shift => repeat([scale], T - abs(shift)) for (shift, scale) in pairs(A.shifts))...
    )
end

function deletezeros!(A::ShiftMatrix)
    # delete any i => elements in the shift matrix
    # warning: can result in an empty matrix!
    for p in pairs(A.shifts)
        p[2] == 0 && delete!(A.shifts, p[1])
    end
    return A
end

# Scalar multiplication
*(α::Float64, B::ShiftMatrix) = ShiftMatrix(Dict(s[1] => α * s[2] for s in pairs(B.shifts)))
*(B::ShiftMatrix, α::Float64) = α * B

# Now define interaction with dense matrices

# Define addition of shift + dense = (shift * I) + dense
# currently only implmented for square matrices

# mark 1
# function +(A::Matrix, S::ShiftMatrix)
#    @assert size(A, 1) == size(A, 2)
#    A + sparse(S, size(A, 1))
#end
#+(S::ShiftMatrix, A::Matrix) = A + S

# mark 2
function add!(C::Matrix, shifter::Pair{Int, Float64})
    shift, scale = shifter
    if shift >= 0
        @inbounds for i in shift+1:size(C, 2)
            C[i-shift, i] += scale
        end
    else
        @inbounds for i in 1:size(C,2)+shift
            C[i-shift, i] += scale
        end
    end
    return C
end

function add!(C::Matrix, S::ShiftMatrix)
    for shifter in pairs(S.shifts)
        add!(C, shifter)
    end
end

# Main application in accumulation is either:
#    C .= C + A * S or C + S * A
# implemented by adding method for 5 argument mul!
# last two arguments are scalars for multiplication, but i don't implement them yet
# Shift in each direction:
function addrightshift!(C, A, shifter)
    shift, scale = shifter
    if shift >= size(A, 2)
        return C
    else
        @avx view(C, :, shift+1:size(C, 2)) .+= scale .* view(A, :, 1:size(A, 2)-shift)
        return C
    end
end
function addleftshift!(C, A, shifter)
    shift, scale = shifter
    if shift >= size(A, 2)
        return C
    else
        @avx view(C, :, 1:size(C, 2)-shift) .+= scale * view(A, :, shift+1:size(A, 2))
        return C
    end
end
function addhshift!(C, A, shifter)
    if shifter[1] >= 0
        addrightshift!(C, A, shifter)
    else
        addleftshift!(C, A, (abs(shifter[1]) => shifter[2]))
    end
end
function addupshift!(C, A, shifter)
    shift, scale = shifter
    if shift >= size(A, 1)
        return C
    else
        @avx view(C, 1:size(C,1)-shift, :) .+= scale .* view(A, shift+1:size(A, 1), :)
        return C
    end
end
function adddownshift!(C, A, shifter)
    shift, scale = shifter
    if shift >= size(A, 1)
        return C
    else
        @avx view(C, shift+1:size(C,1), :) .+= scale .* view(A, 1:size(A,1)-shift, :)
        return C
    end
end
function addvshift!(C, A, shifter)
    if shifter[1] >= 0
        addupshift!(C, A, shifter)
    else
        adddownshift!(C, A, (abs(shifter[1]) => shifter[2]))
    end
end

function leftmuladd!(C, S, A)
    # ie C .= C + S * A
    for shift in pairs(S.shifts)
        addvshift!(C, A, shift)
    end
    return C
end
function rightmuladd!(C, A, S)
    # ie C .= C + A * S
    for shift in pairs(S.shifts)
        addhshift!(C, A, shift)
    end
    return C
end

# mul!(C, A, B, α, β) -> C = A * B * α + C * β
# C::Matrix
mul!(C::Matrix, A::ShiftMatrix, B::Matrix, α::Float64, β::Float64) = leftmuladd!(C, A, B)
mul!(C::Matrix, A::Matrix, B::ShiftMatrix, α::Float64, β::Float64) = rightmuladd!(C, A, B)
mul!(C::Matrix, A::ShiftMatrix, B::ShiftMatrix, α::Float64, β::Float64) = add!(C, A * B)
# C::ShiftMatrix
mul!(C::ShiftMatrix, A::ShiftMatrix, B::ShiftMatrix, α::Float64, β::Float64) = mergewith!(+, C.shifts, (A * B).shifts)
# mutating to dense is very expensive as you need to build the whole dense matrix, so i don't use it
mul!(C::ShiftMatrix, A, B, α::Float64, β::Float64) = error("Can't mutate shift matrix to dense matrix")

#region Functions for getting shift matrix from simple block =====

function get_indices!(refs, obj::Expr, sym)
    if obj.head == :ref
        if obj.args[1] == sym
            push!(refs, obj.args[2])
        end
    elseif (obj.head == :call) || (obj.head == :block) || (obj.head == :(=))
        for subobj in obj.args
            get_indices!(refs, subobj, sym)
        end
    end
end

function get_indices!(refs, obj, sym)
    nothing
end

function shift_indices!(obj::Expr, sym, shift)
    # Shifts all indexes into sym by shift
    if obj.head == :ref
        if obj.args[1] == sym
            obj.args[2] += shift
        end
    elseif (obj.head == :call) || (obj.head == :block) || (obj.head == :(=))
        for subobj in obj.args
            shift_indices!(subobj, sym, shift)
        end
    end
end

function shift_indices!(obj, sym, shift)
    nothing
end

function shift_to_one!(ex, sym)
    # resets indexes into sym so they start at one
    # returns the original min index
    refs = Set{Int}([])
    get_indices!(refs, ex, sym)
    if isempty(refs)
        error("No references to $sym in $ex")
    else
        minmax = (minimum(refs), maximum(refs))
    end
    shift_indices!(ex, sym, -minmax[1] + 1)
    return minmax
end

function shiftfunc(syms, funcexpr)
    # shifts the indices of all vars in syms
    rhs = funcexpr.args[1].args[2]
    minmaxs = Tuple{Int, Int}[]
    for i in eachindex(syms)
        push!(minmaxs, shift_to_one!(rhs, syms[i]))
    end
    return funcexpr, minmaxs
end

struct SimpleBlock <: Block

    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    f::Function # shifted
    xss::Vector{Float64}

    minmaxs::Vector{Tuple{Int, Int}}
    arglengths::Vector{Int}

end

SimpleBlock(inputs, outputs, f, xss, minmaxs) = SimpleBlock(
    inputs, outputs, f, xss, minmaxs,
    [p[2] - p[1] + 1 for p in minmaxs]
)

macro simpleblock(inputs, outputs, xss, f)
    # create a simple block with correctly shifter indices
    shifted_func_expr, minmaxs = shiftfunc(eval(inputs), esc(f))
    return :(SimpleBlock(
        $inputs,
        $(esc(outputs)),
        $shifted_func_expr,
        $(esc(xss)),
        $minmaxs
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
    return deletezeros!(ShiftMatrix(Dict(
        z[1] => z[2] for z in zip(Iterators.countfrom(min_index), v)
    )))
end

function jacobian(sb::SimpleBlock)
    merge((_jacobian(sb, i) for i in eachindex(sb.inputs))...)
end

#endregion


