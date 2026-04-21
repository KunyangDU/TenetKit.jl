function orthogonalize!(env::Environment{3},B::Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}},EnvR::SparseRightEnvironmentTensor,osite::Int64)
    w,w2 = env.layer[2].D[osite]
    EnvRorth = Vector(undef,w)
    EnvRorth .= nothing
    @show "here"

    for i in 1:w, j in 1:w2
        Hij = env.layer[2].ts[osite].m[i,j]
        isnothing(Hij) && continue
        tmp = contract(B,Hij,EnvR.A[j])
        EnvRorth[i] = axpy!(1, tmp - contract(tmp,B) ,EnvRorth[i])
        # if isnothing(EnvRorth[i])
        #     EnvRorth[i] = tmp - contract(tmp,B)
        # else
        #     EnvRorth[i] += tmp - contract(tmp,B)
        # end
    end

    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
end

function orthogonalize!(env::Environment{3},A::Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}},EnvL::SparseLeftEnvironmentTensor,osite::Int64)
    w1,w = env.layer[2].D[osite]
    EnvLorth = Vector(undef,w)
    EnvLorth .= nothing

    for i in 1:w1, j in 1:w
        Hij = env.layer[2].ts[osite].m[i,j]
        isnothing(Hij) && continue
        tmp = contract(EnvL.A[i],A,Hij)
        EnvLorth[j] = axpy!(1, tmp - contract(tmp,A) ,EnvLorth[j])
        # if isnothing(EnvLorth[j])
        #     EnvLorth[j] = tmp - contract(tmp,A)
        # else
        #     EnvLorth[j] += tmp - contract(tmp,A)
        # end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
end

# function orthogonalize!(H::SparseMPOTensor,B::Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}},EnvR::SparseRightEnvironmentTensor)
#     w,w2 = size(H)
#     EnvRorth = Vector(undef,w)
#     EnvRorth .= nothing

#     for i in 1:w, j in 1:w2
#         Hij = H.m[i,j]
#         isnothing(Hij) && continue
#         tmp = contract(B,Hij,EnvR.A[j])
#         if isnothing(EnvRorth[i])
#             EnvRorth[i] = tmp - contract(tmp,B)
#         else
#             EnvRorth[i] += tmp - contract(tmp,B)
#         end
#     end

#     return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
# end

# function orthogonalize!(H::SparseMPOTensor,A::Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}},EnvL::SparseLeftEnvironmentTensor)
#     w1,w = size(H.m)
#     EnvLorth = Vector(undef,w)
#     EnvLorth .= nothing

#     for i in 1:w1, j in 1:w
#         Hij = H.m[i,j]
#         isnothing(Hij) && continue
#         tmp = contract(EnvL.A[i],A,Hij)
#         if isnothing(EnvLorth[j])
#             EnvLorth[j] = tmp - contract(tmp,A)
#         else
#             EnvLorth[j] += tmp - contract(tmp,A)
#         end
#     end

#     return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
# end

function orthogonalize!(H::SparseMPOTensor,B::T,B′::T,EnvR::SparseRightEnvironmentTensor) where T <: Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}}
    w,w2 = size(H)
    EnvRorth = Vector(undef,w)
    EnvRorth .= nothing
    validinds = filter(x -> !isnothing(H.m[x...]), [(i,j) for i in 1:w, j in 1:w2][:])
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validinds) && break
                i,j = validinds[ct]
                C = contract(B,H.m[i,j],EnvR.A[j]) |> x -> x - contract(x,B′)
                lock(Lock)
                try
                    EnvRorth[i] = axpy!(1, C, EnvRorth[i])
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for (i,j) in validinds
            EnvRorth[i] = axpy!(1, contract(B,H.m[i,j],EnvR.A[j]) |> x -> x - contract(x,B′), EnvRorth[i])
        end
    end

    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
end

function orthogonalize!(H::SparseMPOTensor,A::T,A′::T,EnvL::SparseLeftEnvironmentTensor) where T <: Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}}
    w1,w = size(H.m)
    EnvLorth = Vector(undef,w)
    EnvLorth .= nothing
    validinds = filter(x -> !isnothing(H.m[x...]), [(i,j) for i in 1:w1, j in 1:w][:])
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validinds) && break
                i,j = validinds[ct]
                C = contract(EnvL.A[i],A,H.m[i,j]) |> x -> x - contract(x,A′)
                lock(Lock)
                try
                    EnvLorth[j] = axpy!(1, C, EnvLorth[j])
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for (i,j) in validinds
            EnvLorth[j] = axpy!(1, contract(EnvL.A[i],A,H.m[i,j]) |> x -> x - contract(x,A′), EnvLorth[j])
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
end

# function _orthogonalize!(Hij::AbstractLocalOperator,A::T,A′::T,EnvL::LeftEnvironmentTensor) where T <: Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}}
#     x = contract(EnvL,A,Hij) 
#     return x - contract(x,A′)
# end
# function _orthogonalize!(Hij::AbstractLocalOperator,B::T,B′::T,EnvR::RightEnvironmentTensor) where T <: Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}}
#     x = contract(B,Hij,EnvR)
#     return x - contract(x,B′)
# end


function orthogonalize!(A::Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}},A′::Union{DenseMPOTensor{<:Number, 4},MPSTensor{<:Number, 3}},Env::Union{DenseLeftEnvironmentTensor,DenseRightEnvironmentTensor})
    tmp = contract(Env.A,A)
    Envorth = tmp - contract(tmp,A′)
    return Envorth
end

function orthogonalize!(Q::T,A::T,direction::AbstractDirection;tol::Number=1e-4) where T <: Union{MPSTensor{<:Number, 3},DenseMPOTensor{<:Number, 4},AdjointMPOTensor{<:Number, 4}}
    ϵ = norm(_cbeinner(Q,A,direction))
    ϵ > tol && (ϵ = _cbeorth!(Q,A,direction))
    @assert ϵ < tol ϵ
    return Q
end




