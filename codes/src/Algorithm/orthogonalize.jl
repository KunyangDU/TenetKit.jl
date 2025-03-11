function orthogonalize!(env::Environment{3},B::Union{DenseMPOTensor{4},MPSTensor{3}},EnvR::SparseRightEnvironmentTensor,osite::Int64)
    w,w2 = env.layer[2].D[osite]
    EnvRorth = Vector(undef,w)
    EnvRorth .= nothing

    for i in 1:w, j in 1:w2
        Hij = env.layer[2].ts[osite].m[i,j]
        isnothing(Hij) && continue
        tmp = contract(B,Hij,EnvR.A[j])
        if isnothing(EnvRorth[i])
            EnvRorth[i] = tmp - contract(tmp,B)
        else
            EnvRorth[i] += tmp - contract(tmp,B)
        end
    end

    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
end

function orthogonalize!(env::Environment{3},A::Union{DenseMPOTensor{4},MPSTensor{3}},EnvL::SparseLeftEnvironmentTensor,osite::Int64)
    w1,w = env.layer[2].D[osite]
    EnvLorth = Vector(undef,w)
    EnvLorth .= nothing

    for i in 1:w1, j in 1:w
        Hij = env.layer[2].ts[osite].m[i,j]
        isnothing(Hij) && continue
        tmp = contract(EnvL.A[i],A,Hij)
        if isnothing(EnvLorth[j])
            EnvLorth[j] = tmp - contract(tmp,A)
        else
            EnvLorth[j] += tmp - contract(tmp,A)
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
end

# function orthogonalize!(Q::MPSTensor{3},A::MPSTensor{3},direction::Symbol;tol::Number=1e-8)
#     if norm(_cbeinner(Q,A,direction)) > tol
#         if direction == :right 
#             @tensor tmp[-1,-2;-3] ≔ Q.A[-1,2,1] * A'.A[1,3,2] * A.A[3,-2,-3]
#             Q.A -= tmp
#         elseif direction == :left
#             @tensor tmp[-1,-2;-3] ≔ Q.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3]
#             Q.A -= tmp
#         end
#     end
#     @assert norm(_cbeinner(Q,A,direction)) < tol norm(_cbeinner(Q,A,direction))
#     return Q
# end

function orthogonalize!(Q::T,A::T,direction::Symbol;tol::Number=1e-4) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    ϵ = norm(_cbeinner(Q,A,direction))
    if ϵ > tol 
        for i in 1:2
            ϵ = _cbeorth!(Q,A,direction)
            ϵ < tol && break
            i == 2 && @error "cbe not orthogonal with ϵ=$(ϵ)"
        end
    end
    # @assert ϵ < tol ϵ
    return Q
end

function _cbeorth!(Q::T,A::T,direction::Symbol) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    Q.A -= _cbeproj(Q,A,direction)
    return norm(_cbeinner(Q,A,direction))
end


