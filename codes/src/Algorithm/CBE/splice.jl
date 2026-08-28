function splice(Envorth::SparseLeftEnvironmentTensor,Λ::Union{DenseMPOTensor{2},MPSTensor{2}})
    tmp = Vector{LeftCompositeEnvironmentTensor}(undef,Envorth.D[1])
    threaded_foreach(eachindex(Envorth.A)) do i
        tmp[i] = contract(Envorth.A[i], Λ)
    end
    return SparseLeftEnvironmentTensor(tmp)
end

function splice(Envorth::SparseRightEnvironmentTensor,Λ::Union{DenseMPOTensor{2},MPSTensor{2}})
    tmp = Vector{RightCompositeEnvironmentTensor}(undef,Envorth.D[1])
    threaded_foreach(eachindex(Envorth.A)) do i
        tmp[i] = contract(Envorth.A[i], Λ)
    end
    return SparseRightEnvironmentTensor(tmp)
end

function splice(Envorth::SparseLeftEnvironmentTensor,A::Union{AdjointMPOTensor{4},AdjointMPSTensor{3}})
    tmp = Vector{LeftEnvironmentTensor}(undef,Envorth.D[1])
    threaded_foreach(eachindex(Envorth.A)) do i
        tmp[i] = contract(Envorth.A[i], A)
    end
    return SparseLeftEnvironmentTensor(tmp)
end

function splice(Envorth::SparseRightEnvironmentTensor,A::Union{AdjointMPOTensor{4},AdjointMPSTensor{3}})
    tmp = Vector{RightEnvironmentTensor}(undef,Envorth.D[1])
    threaded_foreach(eachindex(Envorth.A)) do i
        tmp[i] = contract(Envorth.A[i], A)
    end
    return SparseRightEnvironmentTensor(tmp)
end

# 原地 splice（bond 张量，元素类型不变）：@tensor = 覆盖（β=0），复用 Envorth.A[i] 的容器与 wrapper。
# 仅当输入 Envorth 之后不再被引用时安全（svd.jl 的 splice Λ）；splice Ω 因输入 env 仍被复用，走分配版 splice。
function splice!(Envorth::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor}, Λ::Union{DenseMPOTensor{2},MPSTensor{2}})
    threaded_foreach(eachindex(Envorth.A)) do i
        _splice_inplace!(Envorth.A[i], Λ)
    end
    return Envorth
end
# 类型改变（Adjoint，Composite→plain）无法原地，保留分配 + 换 .A
function splice!(Envorth::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor}, A::Union{AdjointMPOTensor{4},AdjointMPSTensor{3}})
    Envorth.A = splice(Envorth, A).A
    return Envorth
end

# 原地写各元素（模式与 contract/splice.jl 一一对应，≔→=、tmp→x.A）
function _splice_inplace!(x::LeftCompositeEnvironmentTensor{2,3}, Λ::MPSTensor{2})
    x.A = x.A * Λ.A   # 直接 mul 无法别名（mul! 拒绝 C≡A），分配新数据、复用 wrapper
    return x
end
function _splice_inplace!(x::RightCompositeEnvironmentTensor{1,3}, Λ::MPSTensor{2})
    x.A = @tensor tmp[-1,-2;-3] ≔ Λ.A[-1,1] * x.A[1,-2,-3]   # 直接 mul 无法别名，分配新数据、复用 wrapper
    return x
end
function _splice_inplace!(x::LeftCompositeEnvironmentTensor{2,4}, Λ::MPSTensor{2})
    x.A = @tensor tmp[-1,-2;-3,-4] ≔ x.A[-1,-2,-3,1] * Λ.A[1,-4]   # 收缩末位 domain 腿，直接 mul 无法别名
    return x
end
function _splice_inplace!(x::RightCompositeEnvironmentTensor{1,4}, Λ::MPSTensor{2})
    x.A = @tensor tmp[-1,-2,-3;-4] ≔ Λ.A[-1,1] * x.A[1,-2,-3,-4]   # 收缩首位 codomain 腿，直接 mul 无法别名
    return x
end
function _splice_inplace!(x::LeftCompositeEnvironmentTensor{2,4}, Λ::DenseMPOTensor{2})
    x.A = @tensor tmp[-1,-2;-3,-4] ≔ x.A[-1,-2,1,-4] * Λ.A[1,-3]
    return x
end
function _splice_inplace!(x::RightCompositeEnvironmentTensor{2,4}, Λ::DenseMPOTensor{2})
    x.A = @tensor tmp[-1,-2;-3,-4] ≔ Λ.A[-1,1] * x.A[1,-2,-3,-4]
    return x
end
function _splice_inplace!(x::LeftCompositeEnvironmentTensor{2,5}, Λ::DenseMPOTensor{2})
    x.A = @tensor tmp[-1,-2;-3,-4,-5] ≔ x.A[-1,-2,-3,1,-5] * Λ.A[1,-4]
    return x
end
function _splice_inplace!(x::RightCompositeEnvironmentTensor{2,5}, Λ::DenseMPOTensor{2})
    x.A = @tensor tmp[-1,-2,-3;-4,-5] ≔ Λ.A[-1,1] * x.A[1,-2,-3,-4,-5]   # 收缩首位 codomain 腿，直接 mul 无法别名
    return x
end

#######################

# function splice!(obj::DenseMPO{L}, A::DenseMPOTensor{4}, csite::Int64) where L
#     site = obj.center[1]
#     if csite == site + 1
#         @tensor tmp[-1,-2;-3,-4] ≔ obj[site].A[-1,-2,4,-4] * A.A[2,4,1,3] * obj[csite]'.A[1,3,2,-3]
#         obj[site] = DenseMPOTensor(tmp)
#     elseif csite == site - 1
#         @tensor tmp[-1,-2;-3,-4] ≔ obj[site].A[-1,4,-3,-4] * A.A[2,1,4,3] * obj[csite]'.A[-2,3,2,1]
#         obj[site] = DenseMPOTensor(tmp)
#     else
#         @error "index out of range"
#     end
# end

function splice!(obj::DenseMPO{L}, A::DenseMPOTensor{4}, csite::Int64) where L
    site = obj.center[1]
    obj[site] = splice(obj,A,csite)
end

function splice(obj::DenseMPO{L}, A::DenseMPOTensor{4}, csite::Int64) where L
    site = obj.center[1]
    if csite == site + 1
        @tensor tmp[-1,-2;-3,-4] ≔ obj[site].A[-1,-2,4,-4] * A.A[2,4,1,3] * obj[csite]'.A[1,3,2,-3]
        return DenseMPOTensor(tmp)
    elseif csite == site - 1
        @tensor tmp[-1,-2;-3,-4] ≔ obj[site].A[-1,4,-3,-4] * A.A[2,1,4,3] * obj[csite]'.A[-2,3,2,1]
        return DenseMPOTensor(tmp)
    else
        @error "index out of range"
    end
end

function splice(tl::DenseMPOTensor{4}, tr::DenseMPOTensor{4}, A::DenseMPOTensor{4}, ::L2R)
    @tensor tmp[-1,-2;-3,-4] ≔ tl.A[-1,-2,4,-4] * tr.A[2,4,1,3] * A'.A[1,3,2,-3]
    return DenseMPOTensor(tmp)
end

function splice(tl::DenseMPOTensor{4}, tr::DenseMPOTensor{4}, A::DenseMPOTensor{4}, ::R2L)
    @tensor tmp[-1,-2;-3,-4] ≔ tr.A[-1,4,-3,-4] * tl.A[2,1,4,3] * A'.A[-2,3,2,1]
    return DenseMPOTensor(tmp)
end

#######################

# function splice!(obj::DenseMPS{L}, A::MPSTensor{3}, csite::Int64) where L
#     site = obj.center[1]
#     if csite == site + 1
#         @tensor tmp[-1,-2;-3] ≔ obj[site].A[-1,-2,1] * A.A[1,3,2] * obj[csite]'.A[2,-3,3]
#         obj[site] = MPSTensor(tmp)
#     elseif csite == site - 1
#         @tensor tmp[-1,-2;-3] ≔ obj[site].A[1,-2,-3] * A.A[2,3,1] * obj[csite]'.A[-1,2,3]
#         obj[site] = MPSTensor(tmp)
#     else
#         @error "index out of range"
#     end
# end


function splice(obj::DenseMPS{L}, A::MPSTensor{3}, csite::Int64) where L
    site = obj.center[1]
    if csite == site + 1
        @tensor tmp[-1,-2;-3] ≔ obj[site].A[-1,-2,1] * A.A[1,3,2] * obj[csite]'.A[2,-3,3]
        return MPSTensor(tmp)
    elseif csite == site - 1
        @tensor tmp[-1,-2;-3] ≔ obj[site].A[1,-2,-3] * A.A[2,3,1] * obj[csite]'.A[-1,2,3]
        return MPSTensor(tmp)
    else
        @error "index out of range"
    end
end

function splice!(obj::DenseMPS{L}, A::MPSTensor{3}, csite::Int64) where L
    site = obj.center[1]
    obj[site] = splice(obj,A,csite)
end

function splice(tl::MPSTensor{3}, tr::MPSTensor{3}, A::MPSTensor{3}, ::L2R)
    @tensor tmp[-1,-2;-3] ≔ tl.A[-1,-2,1] * tr.A[1,3,2] * A'.A[2,-3,3]
    return MPSTensor(tmp)
end

function splice(tl::MPSTensor{3}, tr::MPSTensor{3}, A::MPSTensor{3}, ::R2L)
    @tensor tmp[-1,-2;-3] ≔ tr.A[1,-2,-3] * tl.A[2,3,1] * A'.A[-1,2,3]
    return MPSTensor(tmp)
end

splice(tl::T, tr::T, A::T, direction::AbstractDirection) where T <: Union{AdjointMPOTensor{4},AdjointMPSTensor{3}}= splice(tl',tr',A',direction)'

splice(Envorth::T,A::Union{DenseMPOTensor{2},MPSTensor{2}}) where T <: Union{LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor} = contract(Envorth,A)

splice(Envorth::T,A::Union{AdjointMPOTensor{4},AdjointMPSTensor{3}}) where T <: Union{LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor} = contract(Envorth,A)

function splice!(Envorth::Union{LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor},
    Λ::Union{MPSTensor{2},AdjointMPSTensor{3},DenseMPOTensor{2},AdjointMPOTensor{4}})
    Envorth = splice(Envorth,Λ)
end

