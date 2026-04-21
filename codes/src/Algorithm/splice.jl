function splice(Envorth::SparseLeftEnvironmentTensor,Λ::Union{DenseMPOTensor{<:Number, 2},MPSTensor{<:Number, 2}})
    tmp = Vector{LeftCompositeEnvironmentTensor}(undef,Envorth.D[1])
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > Envorth.D[1] && break
                C = contract(Envorth.A[ct],Λ)
                lock(Lock)
                try
                    tmp[ct] = C
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for i in 1:Envorth.D[1]
            tmp[i] = contract(Envorth.A[i],Λ)
        end
    end
    return SparseLeftEnvironmentTensor(tmp)
end

function splice(Envorth::SparseRightEnvironmentTensor,Λ::Union{DenseMPOTensor{<:Number, 2},MPSTensor{<:Number, 2}})
    tmp = Vector{RightCompositeEnvironmentTensor}(undef,Envorth.D[1])
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > Envorth.D[1] && break
                C = contract(Envorth.A[ct],Λ)
                lock(Lock)
                try
                    tmp[ct] = C
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for i in 1:Envorth.D[1]
            tmp[i] = contract(Envorth.A[i],Λ)
        end
    end
    return SparseRightEnvironmentTensor(tmp)
end

function splice(Envorth::SparseLeftEnvironmentTensor,A::Union{AdjointMPOTensor{<:Number, 4},AdjointMPSTensor{<:Number, 3}})
    tmp = Vector{LeftEnvironmentTensor}(undef,Envorth.D[1])
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > Envorth.D[1] && break
                C = contract(Envorth.A[ct],A)
                lock(Lock)
                try
                    tmp[ct] = C
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for i in 1:Envorth.D[1]
            tmp[i] = contract(Envorth.A[i],A)
        end
    end
    return SparseLeftEnvironmentTensor(tmp)
end

function splice(Envorth::SparseRightEnvironmentTensor,A::Union{AdjointMPOTensor{<:Number, 4},AdjointMPSTensor{<:Number, 3}})
    tmp = Vector{RightEnvironmentTensor}(undef,Envorth.D[1])
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > Envorth.D[1] && break
                C = contract(Envorth.A[ct],A)
                lock(Lock)
                try
                    tmp[ct] = C
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for i in 1:Envorth.D[1]
            tmp[i] = contract(Envorth.A[i],A)
        end
    end
    return SparseRightEnvironmentTensor(tmp)
end

function splice!(Envorth::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor},
    Λ::Union{MPSTensor{<:Number, 2},AdjointMPSTensor{<:Number, 3},DenseMPOTensor{<:Number, 2},AdjointMPOTensor{<:Number, 4}})
    Envorth.A = splice(Envorth,Λ).A
end

#######################

# function splice!(obj::DenseMPO{L}, A::DenseMPOTensor{<:Number, 4}, csite::Int64) where L
#     site = obj.center[1]
#     if csite == site + 1
#         @tensor tmp[-1,-2;-3,-4] ≔ obj.ts[site].A[-1,-2,4,-4] * A.A[2,4,1,3] * obj.ts[csite]'.A[1,3,2,-3]
#         obj.ts[site] = DenseMPOTensor(tmp)
#     elseif csite == site - 1
#         @tensor tmp[-1,-2;-3,-4] ≔ obj.ts[site].A[-1,4,-3,-4] * A.A[2,1,4,3] * obj.ts[csite]'.A[-2,3,2,1]
#         obj.ts[site] = DenseMPOTensor(tmp)
#     else
#         @error "index out of range"
#     end
# end

function splice!(obj::DenseMPO{L}, A::DenseMPOTensor{<:Number, 4}, csite::Int64) where L
    site = obj.center[1]
    obj.ts[site] = splice(obj,A,csite)
end

function splice(obj::DenseMPO{L}, A::DenseMPOTensor{<:Number, 4}, csite::Int64) where L
    site = obj.center[1]
    if csite == site + 1
        @tensor tmp[-1,-2;-3,-4] ≔ obj.ts[site].A[-1,-2,4,-4] * A.A[2,4,1,3] * obj.ts[csite]'.A[1,3,2,-3]
        return DenseMPOTensor(tmp)
    elseif csite == site - 1
        @tensor tmp[-1,-2;-3,-4] ≔ obj.ts[site].A[-1,4,-3,-4] * A.A[2,1,4,3] * obj.ts[csite]'.A[-2,3,2,1]
        return DenseMPOTensor(tmp)
    else
        @error "index out of range"
    end
end

function splice(tl::DenseMPOTensor{<:Number, 4}, tr::DenseMPOTensor{<:Number, 4}, A::DenseMPOTensor{<:Number, 4}, ::L2R)
    @tensor tmp[-1,-2;-3,-4] ≔ tl.A[-1,-2,4,-4] * tr.A[2,4,1,3] * A'.A[1,3,2,-3]
    return DenseMPOTensor(tmp)
end

function splice(tl::DenseMPOTensor{<:Number, 4}, tr::DenseMPOTensor{<:Number, 4}, A::DenseMPOTensor{<:Number, 4}, ::R2L)
    @tensor tmp[-1,-2;-3,-4] ≔ tr.A[-1,4,-3,-4] * tl.A[2,1,4,3] * A'.A[-2,3,2,1]
    return DenseMPOTensor(tmp)
end

#######################

# function splice!(obj::DenseMPS{L}, A::MPSTensor{<:Number, 3}, csite::Int64) where L
#     site = obj.center[1]
#     if csite == site + 1
#         @tensor tmp[-1,-2;-3] ≔ obj.ts[site].A[-1,-2,1] * A.A[1,3,2] * obj.ts[csite]'.A[2,-3,3]
#         obj.ts[site] = MPSTensor(tmp)
#     elseif csite == site - 1
#         @tensor tmp[-1,-2;-3] ≔ obj.ts[site].A[1,-2,-3] * A.A[2,3,1] * obj.ts[csite]'.A[-1,2,3]
#         obj.ts[site] = MPSTensor(tmp)
#     else
#         @error "index out of range"
#     end
# end


function splice(obj::DenseMPS{L}, A::MPSTensor{<:Number, 3}, csite::Int64) where L
    site = obj.center[1]
    if csite == site + 1
        @tensor tmp[-1,-2;-3] ≔ obj.ts[site].A[-1,-2,1] * A.A[1,3,2] * obj.ts[csite]'.A[2,-3,3]
        return MPSTensor(tmp)
    elseif csite == site - 1
        @tensor tmp[-1,-2;-3] ≔ obj.ts[site].A[1,-2,-3] * A.A[2,3,1] * obj.ts[csite]'.A[-1,2,3]
        return MPSTensor(tmp)
    else
        @error "index out of range"
    end
end

function splice!(obj::DenseMPS{L}, A::MPSTensor{<:Number, 3}, csite::Int64) where L
    site = obj.center[1]
    obj.ts[site] = splice(obj,A,csite)
end

function splice(tl::MPSTensor{<:Number, 3}, tr::MPSTensor{<:Number, 3}, A::MPSTensor{<:Number, 3}, ::L2R)
    @tensor tmp[-1,-2;-3] ≔ tl.A[-1,-2,1] * tr.A[1,3,2] * A'.A[2,-3,3]
    return MPSTensor(tmp)
end

function splice(tl::MPSTensor{<:Number, 3}, tr::MPSTensor{<:Number, 3}, A::MPSTensor{<:Number, 3}, ::R2L)
    @tensor tmp[-1,-2;-3] ≔ tr.A[1,-2,-3] * tl.A[2,3,1] * A'.A[-1,2,3]
    return MPSTensor(tmp)
end

splice(tl::T, tr::T, A::T, direction::AbstractDirection) where T <: Union{AdjointMPOTensor{<:Number, 4},AdjointMPSTensor{<:Number, 3}}= splice(tl',tr',A',direction)'

splice(Envorth::T,A::Union{DenseMPOTensor{<:Number, 2},MPSTensor{<:Number, 2}}) where T <: Union{LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor} = contract(Envorth,A)

splice(Envorth::T,A::Union{AdjointMPOTensor{<:Number, 4},AdjointMPSTensor{<:Number, 3}}) where T <: Union{LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor} = contract(Envorth,A)

function splice!(Envorth::Union{LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor},
    Λ::Union{MPSTensor{<:Number, 2},AdjointMPSTensor{<:Number, 3},DenseMPOTensor{<:Number, 2},AdjointMPOTensor{<:Number, 4}})
    Envorth = splice(Envorth,Λ)
end

