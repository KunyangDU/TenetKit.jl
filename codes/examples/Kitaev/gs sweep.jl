using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Kitaev/data"

D = 2^8
Lx = 4
Ly = 4
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
Np = 25
@load "$(dataname)/params_$(Np).jld2" params
# params = [[0.5,0.5,Jz] for Jz in 0:0.1:2]
for (i,(Jx,Jy,Jz)) in enumerate(params)
    @show i,length(params),(Jx,Jy,Jz)
    ψ = let 
        AuxSpace = repeat([ℂ^1,], size(Latt))
        randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
    end
    tmpparams = (Jx=Jx,Jy=Jy,Jz=Jz)

    H = TrivialHamiltonian(Latt;tmpparams...)

    lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 6)
    showQuantSweep(lsEg)
    @save "$(dataname)/sweep/lsEg_$(Lx)x$(Ly)_$(D)_$(tmpparams).jld2" lsEg
    @save "$(dataname)/sweep/lsinfo_$(Lx)x$(Ly)_$(D)_$(tmpparams).jld2" lsinfo
    @save "$(dataname)/sweep/ψ_$(Lx)x$(Ly)_$(D)_$(tmpparams).jld2" ψ
end


