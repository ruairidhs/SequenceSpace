
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

# Now define interaction with normal matrices
# Define vertical and horizontal shifts
function vshift!(C, M, shifter::Pair{Int, Float64})
    n, α = shifter # shift and scale
    if n == 0
        C .+= α .* M
    elseif n > 0 # -> shift up
        for row in Iterators.drop(axes(M, 1), n)
            C[row - n, :] .+= α .* view(M, row, :)
        end
    else # n < 0 -> shift down
        for row in Iterators.take(axes(M, 1), size(M, 1) + n)
            C[row - n, :] .+= α .*  view(M, row, :)
        end
    end
    return C
end

function hshift!(C, M, shifter::Pair{Int, Float64})
    n, α = shifter # shift and scale
    if n == 0
        C .+= α .* M
    elseif n > 0 # -> shift right
        for col in Iterators.take(axes(M, 1), size(M, 1) - n)
            C[:, col + n] .+= α .* view(M, :, col)
        end
    else # n < 0 -> shift left
        for col in Iterators.drop(axes(M, 1), abs(n))
            C[:, col + n] .+= α .* view(M, :, col)
        end
    end
    return C
end

# Define multiplication for shift * dense (vertical shift)
function _noresetvshift!(C, S, A)
    # C .= C + S * A
    for shift in pairs(S.shifts)
        vshift!(C, A, shift)
    end
    return C
end
function _noresethshift!(C, S, A)
    # C .= C + A * S
    for shift in pairs(S.shifts)
        hshift!(C, A, shift)
    end
    return C
end

function mul!(C::Matrix, S::ShiftMatrix, A::Matrix)
    # C .= S * A
    fill!(C, 0)
    _noresetvshift!(C, S, A)
end

function *(S::ShiftMatrix, A::Matrix)
    C = zeros(axes(A))
    _noresetvshift!(C, S, A)
end

# Define multiplication for dense * shift (horizontal shift)
function mul!(C::Matrix, A::Matrix, S::ShiftMatrix)
    # C .= A * S
    fill!(C, 0)
    _noresethshift!(C, S, A)
end

function *(A::Matrix, S::ShiftMatrix)
    C = zeros(axes(A))
    _noresethshift!(C, S, A)
end

# Define addition of shift + dense = (shift * I) + dense
# Works by converting shift -> sparse
# currently only implmented for square matrices
function +(A::Matrix, S::ShiftMatrix)
    @assert size(A, 1) == size(A, 2)
    A + sparse(S, size(A, 1))
end
+(S::ShiftMatrix, A::Matrix) = A + S

# 5 element mul! (used in accumulation)
# mul!(C, A, B, α, β) -> C = A * B * α + C * β
# C::Matrix
function mul!(C::Matrix, A::ShiftMatrix, B::Matrix, α::Float64, β::Float64)
    _noresetvshift!(C, α * A, B)
    C .*= β
end
function mul!(C::Matrix, A::Matrix, B::ShiftMatrix, α::Float64, β::Float64)
    _noresethshift!(C, α * B, A)
    C .*= β
end
function mul!(C::Matrix, A::ShiftMatrix, B::ShiftMatrix, α::Float64, β::Float64)
    C .= β .* C + (α * A * B)
end

# C::ShiftMatrix
function mul!(C::ShiftMatrix, A::ShiftMatrix, B::ShiftMatrix, α::Float64, β::Float64)
    mergewith!(+, β * C, α * A * B)
end
function mul!(C::ShiftMatrix, A, B, α::Float64, β::Float64)
    error("Can't mutate shift matrix to dense matrix")
end

#endregion

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


