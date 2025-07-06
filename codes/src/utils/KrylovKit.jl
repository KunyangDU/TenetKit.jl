
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
    codom = ⊗(map(x -> collect(domain(x))[end],[O.EnvL.A[1].A, O.H.ts[1].m[1,1].A])...)
    dom = collect(codomain(O.EnvR.A[1].A))[1]
    tmp = MPSTensor(randn,codom,dom)
    normalize!(tmp)
    return tmp
end

function _initialMPS(O::SparseProjectiveHamiltonian{2})
    codom = ⊗(map(x -> collect(domain(x))[end],[O.EnvL.A[1].A, [O.H.ts[i].m[1,1].A for i in 1:2]...])...)
    dom = collect(codomain(O.EnvR.A[1].A))[1]
    tmp = CompositeMPSTensor(randn,codom,dom)
    normalize!(tmp)
    return tmp
end

function groundEig(O::SparseProjectiveHamiltonian{N},alg::Krylovalgo = DMRGDefaultLanczos;x₀ = _initialMPS(O)) where N
    reset_timer!(get_timer("action"))
    Eg,Ev,info = eigsolve(x -> action(O,x), x₀, 1, :SR, alg.Alg)
    return isapproxreal(Eg[1]), normalize(Ev[1]), Lanczosinfo(info)
end

function evolve!(
    obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
    O::SparseProjectiveHamiltonian{N}, τ::Number,
    alg::Krylovalgo = TDVPDefaultLanczos) where N
    nm = normalize!(obj)
    tmp,info = exponentiate(x -> action(O,x), -τ, obj, alg.Alg)
    rmul!(tmp,nm)
    obj.A = tmp.A
    @assert info.residual ≈ 0
    return obj, Lanczosinfo(info)
end

