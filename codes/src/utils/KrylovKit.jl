

# function MPLanczos(O::SparseProjectiveHamiltonian{N}, q1::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
#     LanczosLevel::Int64;kwargs...) where N
#     Q = Vector(undef, LanczosLevel)
#     α = zeros(LanczosLevel)
#     β = zeros(LanczosLevel-1)

#     Q[1] = q1

#     for j = 1:LanczosLevel
#         if j == 1
#             w = action(O, Q[j])
#         else
#             w = action(O, Q[j]) - β[j-1] * Q[j-1]
#         end

#         α[j] = ApproxReal((w*adjoint(Q[j]))[1])
#         w -= α[j] * Q[j]
        
#         if j < LanczosLevel
#             β[j] = norm(w)
#             if β[j] ≈ 0
#                 @error "flow interrupted"
#             else
#                 Q[j+1] = w / β[j]
#             end
#         end
        
#     end
    
#     T = diagm(0 => α) + diagm(-1 => β) + diagm(1 => β)
#     return T, Q, LanczosLevel
# end

# function MPLanczos(O::SparseProjectiveHamiltonian{N}, 
#     q1::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
#     tol::Float64
#     ;kwargs...) where N
#     maxlevel = get(kwargs,:maxlevel,50)
#     Q = []
#     α = []
#     β = []
#     α,β = map(x -> convert(Vector{Float64},x),[α,β])
    
#     E0 = 0

#     push!(Q,q1)

#     for j = 1:maxlevel
#         if j == 1
#             w = action(O, Q[j])
#         else
#             w = action(O, Q[j]) - β[j-1] * Q[j-1]
#         end

#         push!(α,ApproxReal(contract(w',Q[j])))
#         w -= α[j] * Q[j]

#         if j != 1 
#             tmpT = diagm(0 => α) +diagm(-1 => β) + diagm(1 => β)
#             tmpE0 = eigen(tmpT).values[1]
#             if (tmpE0-E0)/tmpE0 < tol
#                 return tmpT, Q, j
#             end
#             E0 = tmpE0
#         end
        
#         if j < maxlevel
#             push!(β,norm(w))
#             if β[j] ≈ 0
#                 @error "flow interrupted"
#             else
#                 push!(Q, w / β[j])
#             end
#         end    
        
#     end
    
#     T = diagm(0 => α) +diagm(-1 => β) + diagm(1 => β)
#     return T, Q, maxlevel
# end
# 
# function evolve!(
#     obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
#     O::SparseProjectiveHamiltonian{N}, τ::Number, LanczosInfo::Number=1e-6) where N
#     tmp = normalize!(obj)
#     T, Q, K = MPLanczos(O,obj,LanczosInfo)
#     obj.A = sum(tmp * exp(-1im*τ*T)[:,1] .* map(x->x.A, Q))
#     return obj, K
# end

# function groundEig(O::SparseProjectiveHamiltonian{N},alg::KrylovKit.KrylovAlgorithm = DMRGDefaultLanczos.Alg) where N
#     Eg,Ev,info = eigsolve(x -> action(O,x), _initialMPS(O), 1, :SR,alg)
#     return isapproxreal(Eg[1]), normalize(Ev[1]), Lanczosinfo(info)
# end

# function evolve!(
#     obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
#     O::SparseProjectiveHamiltonian{N}, τ::Number,
#     alg::KrylovKit.KrylovAlgorithm = TDVPDefaultLanczos.Alg) where N
#     nm = normalize!(obj)
#     tmp,info = exponentiate(x -> action(O,x),-1im * τ,obj,alg)
#     rmul!(tmp,nm)
#     obj.A = tmp.A
#     @assert info.residual ≈ 0
#     return obj, Lanczosinfo(info)
# end

function _initialMPS(O::SparseProjectiveHamiltonian{1})
    codom = ⊗(map(x -> collect(domain(x))[end],[O.EnvL.A[1].A, O.H[1].m[1,1].A])...)
    dom = collect(codomain(O.EnvR.A[1].A))[1]
    tmp = MPSTensor(randn,codom,dom)
    normalize!(tmp)
    return tmp
end

function _initialMPS(O::SparseProjectiveHamiltonian{2})
    codom = ⊗(map(x -> collect(domain(x))[end],[O.EnvL.A[1].A, [O.H[i].m[1,1].A for i in 1:2]...])...)
    dom = collect(codomain(O.EnvR.A[1].A))[1]
    tmp = CompositeMPSTensor(randn,codom,dom)
    normalize!(tmp)
    return tmp
end

function groundEig(O::Union{SparseProjectiveHamiltonian{N},DenseProjectiveHamiltonian{3,N}},alg::Krylovalgo = DMRGDefaultLanczos;x₀ = _initialMPS(O)) where N
    reset_action_timer!()
    Eg,Ev,info = eigsolve(x -> action(O,x), x₀, 1, :SR, alg.Alg)
    return isapproxreal(Eg[1]), normalize(Ev[1]), Lanczosinfo(info)
end

function evolve!(
    obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
    O::Union{SparseProjectiveHamiltonian{N},DenseProjectiveHamiltonian{3,N}}, τ::Number,
    alg::Krylovalgo = TDVPDefaultLanczos) where N
    nm = normalize!(obj)
    reset_action_timer!()
    tmp,info = exponentiate(x -> action(O,x), -τ, obj, alg.Alg)
    rmul!(tmp,nm)
    obj.A = tmp.A
    @assert info.residual ≈ 0
    return obj, Lanczosinfo(info)
end

# function evolve!(
#     obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
#     O::Union{SparseProjectiveHamiltonian{N},DenseProjectiveHamiltonian{3,N}}, 
#     τ::Number,
#     alg::Krylovalgo = TDVPDefaultGMRES) where N
#     nm = normalize!(obj)
#     reset_action_timer!()
#     dt_half = τ / 2.0
#     # b = obj - dt_half * action(O, obj)
#     # M = x -> x + dt_half * action(O, x)
#     # tmp, info = linsolve(M, b, obj, alg.Alg, 1.0, 0.0)
#     # τ 是实数步长 Δβ
#     b = obj  # 右侧不再有 (I - H) 项，直接就是初态
#     M = x -> x + τ * action(O, x) # 左侧算子： (I + τH)

#     # 调用 GMRES (同样需要传入 AI 初猜 obj)
#     tmp, info = linsolve(M, b, obj, alg.Alg, 1.0, 0.0)
#     rmul!(tmp, nm)
#     obj.A = tmp.A
#     # 示例：根据你的 info 结构调整返回
#     # info.converged 在 GMRES 中通常是迭代步数
#     # info.residual 是最后的残差向量
#     res_norm = norm(info.residual)
#     return obj, Lanczosinfo(info.converged, info.numops)
# end