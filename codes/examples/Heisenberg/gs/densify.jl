using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")
# dataname = "examples/Heisenberg/data/U1"

# D = 2^7
# params = (Jz = 1,Jxy = 0.5)

# @save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

# ψ = let 
#     AuxSpace = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt):1//2:size(Latt) ),], size(Latt)-1))
#     randMPS(U₁Spin.PhySpace ,AuxSpace)
# end
# TensorKit.space(A::AbstractTensorWrapper) = space(A.A)
# TensorKit.space(A::AbstractLocalOperator) = space(A.A)


# function pushleft(A::SparseMPO, B::AdjointMPO, EnvR::SparseRightEnvironmentTensor, site::Int64)
#     @assert A.D[site][2] == EnvR.D
#     tmpEnvR = Vector{Any}(nothing,A.D[site][1])
#     for i in eachindex(tmpEnvR), j in 1:EnvR.D
#         isnothing(A.ts[site].m[i,j]) && continue
#         tmpEnvR[i] = axpy!(1, contract(A.ts[site].m[i,j], B.ts[site], EnvR.A[j]), tmpEnvR[i])
#     end
#     return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor},tmpEnvR))
# end

# function pushright(A::SparseMPO, B::AdjointMPO, EnvL::SparseLeftEnvironmentTensor, site::Int64)
#     @assert A.D[site][1] == EnvL.D
#     tmpEnvL = Vector{Any}(nothing,A.D[site][2])
#     for i in eachindex(tmpEnvL), j in 1:EnvL.D
#         isnothing(A.ts[site].m[j,i]) && continue
#         tmpEnvL[i] = axpy!(1, contract(A.ts[site].m[j,i], B.ts[site],EnvL.A[j]),tmpEnvL[i])
#     end
#     return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor},tmpEnvL))
# end

# function _isometry(sps::GradedSpace...;T::Type = ComplexF64)
#     sp = reduce(⊗, sps)
#     tmp = TensorMap(zeros,T,sp,sp)
#     return rightorth(tmp)[2]
# end

# function contract(::IdentityOperator{1}, obj::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ obj.A[2,1,1,-2] * EnvR.A[-1,2]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{1, 2}, obj::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     iso = _isometry(space(A)[2],space(EnvR)[1])
#     @tensor tmp[-1;-2] ≔ iso[-1,4,5] * A.A[2,4,1] * obj.A[3,1,2,-2] * EnvR.A[5,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{2, 1}, obj::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     iso = _isometry(space(A)[2],space(EnvR)[1])
#     @tensor tmp[-1;-2] ≔ iso[-1,4,5] * A.A[2,4,1] * obj.A[3,1,2,-2] * EnvR.A[5,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{1, 1}, obj::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[2,1] * obj.A[3,1,2,-2] * EnvR.A[-1,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{1, 2}, obj::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     iso = _isometry(space(EnvL)[2]',space(A)[2]')'
#     @tensor tmp[-1;-2] ≔ EnvL.A[3,4] * A.A[2,5,1] * obj.A[-1,1,2,3] * iso[4,5,-2]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{2, 1}, obj::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     iso = _isometry(space(EnvL)[2]',space(A)[2]')'
#     @tensor tmp[-1;-2] ≔ EnvL.A[3,4] * A.A[2,5,1] * obj.A[-1,1,2,3] * iso[4,5,-2]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{1, 1}, obj::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ EnvL.A[3,-2] * A.A[2,1] * obj.A[-1,1,2,3]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(::IdentityOperator{1}, obj::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ EnvL.A[2,-2] * obj.A[-1,1,1,2]
#     return LeftEnvironmentTensor(tmp)
# end

# function proj2(env::Environment{2},site1::Int64,site2::Int64;E₀::Number = 0.0)
#     !issparse(env.layer[1]) && return nothing
#     return site1 < site2 ? projright2(env,site1,E₀) : projleft2(env,site2,E₀)
# end
# projright2(env::Environment{2},site::Int64,E₀::Number = 0.0) = issparse(env.layer[1]) ? SparseProjectiveHamiltonian(env.envs[[site,site+2]]...,SparseMPO(env.layer[1].ts[site:site+1]),E₀) : nothing
# projleft2(env::Environment{2},site::Int64,E₀::Number = 0.0) = issparse(env.layer[1]) ? SparseProjectiveHamiltonian(env.envs[[site-1,site+1]]...,SparseMPO(env.layer[1].ts[site-1:site]),E₀) : nothing

# function action(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2})
#     x = nothing
#     to = get_timer("action")
#     timer_acc = TimerOutput()
#     Nthr = get_num_threads_julia()
#     for ind in O.validinds
#         C,localto = _action2(O,obj,ind)
#         x = axpy!(1,C,x)
#         merge!(timer_acc, localto)
#     end
#     return x
# end

# function _action2(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2}, ind::Tuple)
#     i,j,k = ind
#     localto = TimerOutput()
#     @timeit localto "_action2_EL1=El_H1" EL1 = contract(O.EnvL.A[i], obj.ts[1].m[i,j])
#     @timeit localto "_action2_EL2=EL1_H2" EL2 = contract(EL1, obj.ts[2].m[j,k])
#     @timeit localto "_action2_C=EL2_Er" C = contract(EL2, O.EnvR.A[k])
#     return C, localto
# end

# function contract(El::LeftEnvironmentTensor{2}, A::LocalOperator{1, 2})
#     iso = _isometry(space(El)[2]',space(A)[2]')'
#     @tensor tmp[-1,-2;-3,-4] ≔ El.A[-1,1] * A.A[-2,2,-4] * iso[1,2,-3]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftEnvironmentTensor{2}, A::LocalOperator{2, 1})
#     iso = _isometry(space(El)[2]',space(A)[2]')'
#     @tensor tmp[-1,-2;-3,-4] ≔ El.A[-1,1] * A.A[-2,2,-4] * iso[1,2,-3]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftEnvironmentTensor{2}, A::LocalOperator{1, 1})
#     @tensor tmp[-1,-2;-3,-4] ≔ El.A[-1,-3] * A.A[-2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 3}, A::IdentityOperator{1})
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-1,-2,-4,-6] * A.A[-3,-5]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftEnvironmentTensor{2}, A::IdentityOperator{1})
#     @tensor tmp[-1,-2;-3,-4] ≔ El.A[-1,-3] * A.A[-2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 3}, A::LocalOperator{1, 2})
#     iso = _isometry(space(El)[3]',space(A)[2]')'
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-1,-2,1,-6] * A.A[-3,2,-5] * iso[1,2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 3}, A::LocalOperator{2, 1})
#     iso = _isometry(space(El)[3]',space(A)[2]')'
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-1,-2,1,-6] * A.A[-3,2,-5] * iso[1,2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 3}, A::LocalOperator{1, 1})
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-1,-2,-4,-6] * A.A[-3,-5]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function densify!(α::Number, x::SparseMPO, y::DenseMPO;kwargs...)
#     trunc = get(kwargs,:trunc,notrunc())
#     N  = get(kwargs,:N,3)
#     tol = get(kwargs,:tol,1e-8)
#     algo = Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,N,tol)
#     return densify!(α,x,y,algo;kwargs...)
# end

# function densify!(α::Number, x::SparseMPO{L}, y::DenseMPO{L}, Alg::Algebraalgo;kwargs...) where L
#     y′ = y'
    
#     to = TimerOutput()
#     @timeit to "initialize XY Env" begin
#         Env = Environment([deepcopy(x),y′])
#         initialize!(Env;
#         left_default_space = ntuple(x -> space(DeS₀.ts[1])[2],2),
#         right_default_space = ntuple(x -> space(DeS₀.ts[end])[3]',2))
#     end

#     info = Algebrainfo()
#     while info.n ≤ Alg.N
#         localto = TimerOutput()

#         l2rinfo = Algebrasweepinfo(L2R())
#         mto = densify!(α,Env,Alg,l2rinfo)
#         merge!(localto,mto)
#         merge!(info,l2rinfo)

#         r2linfo = Algebrasweepinfo(R2L())
#         mto = densify!(α,Env,Alg,r2linfo)
#         merge!(localto,mto)
#         merge!(info,r2linfo)

#         show(localto;title = "densify!")
#         print("\n")
#         show(info)
#         merge!(to,localto)

#         info.err < Alg.tol && break
#     end

#     return xp!(y′',y),to,info
# end

# function densify!(α::Number, Env::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R})
#     localto = TimerOutput()
#     L = length(Env.layer[2])
#     for site in 1:L-1
#         siteinfo = Algebrasiteinfo()
#         @timeit localto "composite x₀" x₀ = deepcopy(composite(Env.layer[2].ts[site:site+1]...))
#         @assert (x2 = norm(x₀)^2) ≠ 0
#         @timeit localto "projection" projH = proj2(Env,site,site+1)
#         @timeit localto "SVD" tl, tc, tr, ~ = tsvd(rmul!(action(projH, projH.H),α); direction=:center,trunc = Alg.trunc)
#         siteinfo.bond = BondInfo(tc)
#         @timeit localto "contract" tr = contract(tc,tr)
#         Env.layer[2].ts[site:site+1] = adjoint.([tl, tr])
#         canonicalize!(Env.layer[2],site+1)
#         @timeit localto "push right" pushright!(Env)
#         @timeit localto "composite x" x = composite(Env.layer[2].ts[site:site+1]...)
#         siteinfo.err = norm(x-x₀)^2/x2
#         merge!(sweepinfo,siteinfo)
#     end

#     return localto
# end

# function densify!(α::Number, Env::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L})
#     localto = TimerOutput()
#     L = length(Env.layer[2])
#     for site in L:-1:2
#         siteinfo = Algebrasiteinfo()
#         @timeit localto "composite x₀" x₀ = deepcopy(composite(Env.layer[2].ts[site-1:site]...))
#         @assert (x2 = norm(x₀)^2) ≠ 0
#         @timeit localto "projection" projH = proj2(Env,site-1,site)
#         @timeit localto "SVD" tl, tc, tr, ~ = tsvd(rmul!(action(projH, projH.H),α); direction=:center,trunc = Alg.trunc)
#         siteinfo.bond = BondInfo(tc)
#         @timeit localto "contract" tl = contract(tl,tc) 
#         Env.layer[2].ts[site-1:site] = adjoint.([tl, tr])
#         canonicalize!(Env.layer[2],site-1)
#         @timeit localto "push left" pushleft!(Env)
#         @timeit localto "composite x" x = composite(Env.layer[2].ts[site-1:site]...)
#         siteinfo.err = norm(x-x₀)^2/x2
#         merge!(sweepinfo,siteinfo)
#     end

#     return localto
# end

# SU2 bug
# SpaceMismatch("ProductSpace(Rep[SU₂](0=>1, 1=>1, 2=>1)) ≠ ProductSpace(Rep[SU₂](0=>1))")

Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)

LocalSpace = U₁Spin
k = [0,0]
SpS = let 
    S₊ = LocalSpace.S₊S₋[1]
    @show space(S₊)
    Root = InteractionTreeNode()
    for i in 1:size(Latt)
        addIntr!(Root,S₊, i,"S+",exp(1im*dot(k,coordinate(Latt,i))) / sqrt(size(Latt)),nothing)
    end
    AutomataSparseMPO(InteractionTree(Root),size(Latt))
end

DeS₀ = let 
    # AuxSpaces = vcat(Rep[SU₂](1 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:size(Latt) ),], size(Latt)-1), Rep[SU₂](0 => 1))
    AuxSpaces = vcat(Rep[U₁](-1 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt):1//2:size(Latt) ),], size(Latt)-1), Rep[U₁](0 => 1))
    _funcDenseMPO(randn, repeat([LocalSpace.PhySpace,],size(Latt)), AuxSpaces)
end

canonicalize!(DeS₀,1)
densify!(1,SpS,DeS₀;trunc = truncdim(4) & truncbelow(1e-6))

1


