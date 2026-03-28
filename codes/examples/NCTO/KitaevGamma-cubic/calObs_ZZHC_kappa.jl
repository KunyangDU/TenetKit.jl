using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/KitaevGamma-cubic/data/ZZHC"

D = 128
Lx = 3
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt


θ = 0.0 * pi
ϕ = 0.5 * pi

K = -1.0
J = -0.1
Γ = 0.3
Γ′ = -0.02

params1_Kitaev = (J = J, K = K, Γ = Γ, Γ′ = Γ′)


flux_Latt = YCTria(2Lx-1,Ly)
direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]
fluxsites,fluxdirections,_ = getPBCflux(Latt,flux_Latt,direction;d = 1/sqrt(3),edge_shift = [0,1],flux_shift = [2*sqrt(3)/3,0])

lsHf = 0:0.02:0.8
lsκ = zeros(length(lsHf) - 1)


for (i,Hf) in enumerate(lsHf[1:end-1])
@show i / (length(lsHf)-1)

Hx = Hf*sin(θ)*cos(ϕ)
Hy = Hf*sin(θ)*sin(ϕ)
Hz = Hf*cos(θ)

Hcx,Hcy,Hcz = Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)
params_H_name = (Hx = round.(Hcx;digits = 3), Hy = round.(Hcy;digits = 3), Hz = round.(Hcz;digits = 3))

params_Kitaev = merge(params1_Kitaev,params_H)
params_Kitaev_name = merge(params1_Kitaev,params_H_name)

@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" ψ
ψ₀ = ψ

ψ₁ = let Hf = lsHf[i+1]
    Hx = Hf*sin(θ)*cos(ϕ)
    Hy = Hf*sin(θ)*sin(ϕ)
    Hz = Hf*cos(θ)

    Hcx,Hcy,Hcz = Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3)

    params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)
    params_H_name = (Hx = round.(Hcx;digits = 3), Hy = round.(Hcy;digits = 3), Hz = round.(Hcz;digits = 3))

    params_Kitaev = merge(params1_Kitaev,params_H)
    params_Kitaev_name = merge(params1_Kitaev,params_H_name)
    @load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" ψ
    ψ
end

lsκ[i] = (1 - abs(inner(ψ₀,ψ₁')))/(lsHf[i+1] - Hf)

end

@save "$(dataname)/lsHf_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).jld2" lsHf
@save "$(dataname)/lsκ_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).jld2" lsκ

lsκ
# map(x -> showdomain(x.A),ψ.ts)