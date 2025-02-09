using TensorKit,CairoMakie
include("../../src/iMPS.jl")
include("model.jl")

function IdDenseMPO(PhySpace::ElementarySpace, AuxSpaces::AbstractVector)
    tmp = [DenseMPOTensor(isometry(PhySpace ⊗ AuxSpaces[i], AuxSpaces[i+1] ⊗ PhySpace)) for i in eachindex(AuxSpaces)[1:end-1]]
    return DenseMPO(tmp)
    #return _funcDenseMPO(ones, repeat([PhySpace,],length(AuxSpaces)), AuxSpaces)
end
#= 
function IdDenseMPO(PhySpace::ElementarySpace, AuxSpaces::AbstractVector)
    return _funcDenseMPO(ones, repeat([PhySpace,],length(AuxSpaces)), AuxSpaces)
end
=#

function contract(EnvL::DenseLeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::DenseMPOTensor{4}, D::DenseMPOTensor{4}, EnvR::DenseRightEnvironmentTensor{3})
    @tensor tmp1[-1 -2;-3 -4 -5] ≔ EnvL.A.A[-1,1,2] * A.A[-2,1,-3,3] * C.A[3,2,-4,-5]
    @tensor tmp2[-1 -2 -3;-4 -5] ≔ B.A[3,-1,1,-5] * D.A[-3,-2,2,3] * EnvR.A.A[1,2,-4]
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ tmp1[-3,-2,2,1,-6] * tmp2[1,2,-1,-4,-5]
    return CompositeMPOTensor(tmp)
end

function ParticleNumber(Latt::AbstractLattice)
    N = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinlessFermion
    
        for i in [1,2]
            addIntr!(Root,LocalSpace.n,i,"n",1,nothing)
        end
    
        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end

    return N
end

Lx = 8
Ly = 1

N = Lx*Ly

Latt = YCSqua(Lx,Ly)

params = (μ = 0,)
H = Hamiltonian(Latt;params...)
Nop = ParticleNumber(Latt)
D = 2^8

ρ = let 
    AuxSpaces = repeat([ℂ^1,], Lx*Ly+1)
    ρ = IdDenseMPO(TrivialSpinlessFermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end
lsβ = vcat(2. .^ (-15:1:-1),1:10)
ρ = SETTN!(lsβ[1],H,deepcopy(ρ);D = D)
lsρ = tanTRG2!(ρ, H, lsβ, D;LanczosLevel = 15)
#lsρ = SETTN!(lsβ,H,ρ;max_order = 30,F_tol = 1e-16)
@save "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "examples/TrivialSpinlessFermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ




