using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

# mul!( SparseMPO, MPS) without quantum number
# add quantum number SparseMPO
# LocalOperator defination error:
# LO{1,2} + ER{3} -> contract or iso-contract ? check space?

# densify SU2 error: SpaceMismatch
# isometry space error ? check it

function contract(A::MPSTensor{3},mpot::LocalOperator{1,2},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    # @tensor tmp[-1 -2;-3] ≔ A.A[-1,4,1] * mpot.A[3,-2,4] * B.A[2,-3,3] * EnvR.A[1,2]
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,2,1] * mpot.A[4,-2,2] * B.A[3,-3,4] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightCompositeEnvironmentTensor{1, 3})
    @tensor tmp[-1 -2 -3;-4] ≔ EnvL.A[-1,-2,1] * EnvR.A[1,-3,-4]
    return CompositeMPSTensor(tmp)
end

Lx = 8
Ly = 1
D = 2^6

Latt = YCSqua(Lx,Ly)

# H = U1Hamiltonian(Latt)

ψ = let 
    AuxSpace = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt):1//2:size(Latt) ),], size(Latt)-1))
    randMPS(U₁Spin.PhySpace ,AuxSpace)
end

ψ′ = let 
    AuxSpace = vcat(Rep[U₁](-1 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt):1//2:size(Latt) ),], size(Latt)-1))
    randMPS(U₁Spin.PhySpace ,AuxSpace)
end

# obs = let 
#     Obs = Observable()
#     LocalSpace = U₁Spin

#     for i in 1:size(Latt)
#         addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
#     end

#     calObs!(Obs, ψ)
# end

# sum([obs["Sz"][(i,)] for i in 1:size(Latt)])

# DeH = let LocalSpace = U₁Spin
#     AuxSpaces = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt):size(Latt) ),], size(Latt)-1), Rep[U₁](0 => 1))
#     _funcDenseMPO(randn, repeat([LocalSpace.PhySpace,],size(Latt)), AuxSpaces)
# end
LocalSpace = U₁Spin
SpS = let 
    S₊ = LocalSpace.S₊S₋[1]
    @show space(S₊)
    Root = InteractionTreeNode()
    for i in 1:size(Latt)
        addIntr!(Root,S₊, i,"S+",exp(1im*dot(k,coordinate(Latt,i))) / sqrt(size(Latt)),nothing)
    end
    AutomataSparseMPO(InteractionTree(Root),size(Latt))
end

# ψ1,_,_ = mul!(deepcopy(ψ),ψ,SpS,1,truncdim(D)&truncbelow(1e-12))
# E21 = inner(ψ1,ψ1')
# E2≈E21
@show (space(ψ′'.ts[1])[2]',space(ψ′'.ts[1])[2]' ⊗ space(ψ.ts[1])[1])
@show (space(ψ.ts[end])[3]' ⊗ trivial(space(ψ′.ts[end])[3]),space(ψ′'.ts[end])[1])
Env = Environment([ψ,SpS,ψ′'])
initialize!(Env;
left_default_space = (space(ψ′'.ts[1])[2]',space(ψ′'.ts[1])[2]' ⊗ space(ψ.ts[1])[1]),
right_default_space = (space(ψ.ts[end])[3]' ⊗ trivial(space(ψ′.ts[end])[3]),space(ψ′'.ts[end])[1]),
)
Env.envs[end]
# densify!(1,H,DeH;trunc = truncdim(D)&truncbelow(1e-12))
# ψ,_,_ = mul!(ψ,ψ,DeH,1,truncdim(D)&truncbelow(1e-12))

# E2 = inner(ψ,ψ')




