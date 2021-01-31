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
#endregion