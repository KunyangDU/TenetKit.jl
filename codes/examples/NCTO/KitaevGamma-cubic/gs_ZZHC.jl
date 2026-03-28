using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/KitaevGamma-cubic/data/ZZHC"

D = 100
Lx = 3
Ly = 3

Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

θ = 0.0
ϕ = 0.0

K = 1.0
J = 0.0
Γ = 0.0
Γ′ = 0.0
# for Γ′ in 0.01:0.01:0.09
params1_Kitaev = (J = J, K = K, Γ = Γ, Γ′ = Γ′)

# for Hf in 0.02:0.04:0.8
Hf = 0.01
Hx = Hf*sin(θ)*cos(ϕ)
Hy = Hf*sin(θ)*sin(ϕ)
Hz = Hf*cos(θ)

Hcx,Hcy,Hcz = Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)
params_H_name = (Hx = round.(Hcx;digits = 3), Hy = round.(Hcy;digits = 3), Hz = round.(Hcz;digits = 3))

params_Kitaev = merge(params1_Kitaev,params_H)
params_Kitaev_name = merge(params1_Kitaev,params_H_name)

println("$(Lx)x$(Ly), D = $(D), \nparams_Kitaev = $(params_Kitaev)")

H = TrivialHamiltonian(Latt;params_Kitaev...,
shift = [0,1], direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]])

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 10)
showQuantSweep(lsEg)

@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" ψ
# end





