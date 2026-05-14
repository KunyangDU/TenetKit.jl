using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/U1"

function TensorKit.leftorth(A::MPSTensor{4}, B::MPSTensor{4})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3 -4] ≔ Rm.A[-1,1]*B.A[1,-2,-3,-4]
    return Q,MPSTensor(tmp)
end

function TensorKit.rightorth(A::MPSTensor{4}, B::MPSTensor{4})
    Lm,Q = rightorth(B)
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[-1,-2,1,-4] * Lm.A[1,-3]
    return MPSTensor(tmp),Q
end

function randMPS(PhySpaces::Vector,AuxSpaces::Vector,SuppSpaces::Vector;
    type::Type = Float64,tailSpace::ElementarySpace = trivial(PhySpaces[1]))
    @assert (L = length(PhySpaces)) == length(AuxSpaces) == length(SuppSpaces)
    push!(AuxSpaces, tailSpace)
    tmp = Vector{MPSTensor}(undef,L)
    for i in 1:L
        tmp[i] = MPSTensor(randn,AuxSpaces[i] ⊗ PhySpaces[i], AuxSpaces[i+1] ⊗ SuppSpaces[i])
    end

    obj = DenseMPS{L,type}(tmp)

    canonicalize!(obj, L)
    canonicalize!(obj, 1)
    normalize!(obj)

    return obj
end

# 写rank 3 MPO和MPS缩并
# 写rank 4 MPS的函数（能够完成mul、TDVP）
# 规范：两点算符的左算符作为作用项
D = 2^6
Lx = 10
Ly = 1
params = (Jz = 1,Jxy = 0.5, H= 1)
LocalSpace = U₁Spin

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

Sz = zeros(size(Latt),size(Latt))
for i in 1:size(Latt), j in i+1:size(Latt)
    Sz[i,j] = gsdata["SzSz"][(i,j)]
end
Sz = 2*sum(Sz) + size(Latt)/4

# HS₋ = let LocalSpace = U₁Spin
#     Root = InteractionTreeNode()

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.S₋S₊[1],i,"S-",1,nothing)
#     end

#     AutomataSparseMPO(InteractionTree(Root),size(Latt))  
# end

HSz = let 
    Root = InteractionTreeNode()

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",1,nothing)
    end

    AutomataSparseMPO(InteractionTree(Root),size(Latt))  
end

HS₊ = let 
    Root = InteractionTreeNode()

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.S₊S₋[1],i,"S+",1,nothing)
    end
    AutomataSparseMPO(InteractionTree(Root),size(Latt))  
end

ψ′ = let 
    suppspace = domain(LocalSpace.S₊S₋[1])[1]
    SuppSpaces = repeat([suppspace,],size(Latt))
    AuxSpace = vcat(Rep[U₁](size(Latt)//2 => 1),repeat([Rep[U₁](i => 1 for i in -2size(Latt):1//2:2size(Latt) ),], Lx*Ly-1))
    randMPS(repeat([U₁Spin.PhySpace,],size(Latt)) ,AuxSpace)
end
@show ψ′[1],ψ′[end]
# ψ′ = let 
#     AuxSpace = vcat(Rep[U₁](size(Latt)//2 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt)//2 :1//2:size(Latt)//2 ),], Lx*Ly-1))
#     randMPS(U₁Spin.PhySpace ,AuxSpace)
# end


hat = LocalSpace.S₊S₋[2]
@show hat
# for i in 1:size(Latt)
#     @tensor tmp[-1,-2,-3;-4,-5] ≔ ψ′.ts[i].A[-2,-3,-4,1] * hat[-1,1,-5]
#     pL = isometry(fuse(codomain(tmp)[1],codomain(tmp)[2]),codomain(tmp)[1] ⊗ codomain(tmp)[2])
#     pR = isometry(domain(tmp)[1] ⊗ domain(tmp)[2],fuse(domain(tmp)[1],domain(tmp)[2]))
#     @tensor tmp1[-1,-2;-3] ≔ pL[-1,1,2] * tmp[1,2,-2,3,4] * pR[3,4,-3]
#     ψ′.ts[i] = MPSTensor(tmp1)
# end
@show ψ′[1],ψ′[end]
# canonicalize!(ψ',size(Latt))
# canonicalize!(ψ',1)

# # # typeof(ψ′.ts[1])

Obs = Observable()
LocalSpace = U₁Spin

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
end

calObs!(Obs, ψ′)
gsdata = Obs.values

sum([gsdata["Sz"][(i,)] for i in 1:size(Latt)])

# ψ′
# ψSz = mul!(deepcopy(ψ),ψ,HSz,1, truncdim(D) & truncbelow(-Inf) )[1]
# ψS₊ = mul!(deepcopy(ψ),ψ,HS₊,1, truncdim(D) & truncbelow(-Inf) )[1]

# ψSz = mul!(deepcopy(ψ),ψ,HSz,1, Algebraalgo(DoubleSite(),NoAlgorithm(),truncdim(D) & truncbelow(-Inf),3,-Inf))[1]
# ψS₊ = mul!(deepcopy(ψ),ψ,HS₊,1, Algebraalgo(DoubleSite(),NoAlgorithm(),truncdim(D) & truncbelow(-Inf),3,-Inf))[1]
# LeftCompositeEnvironmentTensor
# inner(ψSz,ψSz'),Sz

# @tensor tmp[-1;-2] ≔ hat[1,-1,2] * hat'[2,1,-2]

