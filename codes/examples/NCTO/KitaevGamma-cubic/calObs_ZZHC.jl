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

@time "make tree" begin
    Obs = Observable()
    LocalSpace = TrivialSpinOneHalf
    opnames = ("Sx","Sy","Sz")
    Ops = (LocalSpace.Sx,LocalSpace.Sy,LocalSpace.Sz)
    fluxnames = map(y -> map(x -> opnames[x],y),fluxdirections)
    fluxops = map(y -> map(x -> Ops[x],y),fluxdirections)

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",false,nothing)
        addObs!(Obs,LocalSpace.Sy,i,"Sy",false,nothing)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",false,nothing)
    end

    for i in 1:size(Latt),j in i+1:size(Latt)
        addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),(false,false),nothing)
        addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),(false,false),nothing)
        addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),(false,false),nothing)
    end

    for i in eachindex(fluxsites)
        addObs!(Obs,fluxops[i],fluxsites[i],fluxnames[i],Tuple([false for _ in 1:6]),nothing)
    end
end

for Hf in 0.02:0.04:0.8
# Hf = 0.0

Hx = Hf*sin(θ)*cos(ϕ)
Hy = Hf*sin(θ)*sin(ϕ)
Hz = Hf*cos(θ)

Hcx,Hcy,Hcz = Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)
params_H_name = (Hx = round.(Hcx;digits = 3), Hy = round.(Hcy;digits = 3), Hz = round.(Hcz;digits = 3))

params_Kitaev = merge(params1_Kitaev,params_H)
params_Kitaev_name = merge(params1_Kitaev,params_H_name)

println("$(Lx)x$(Ly), D = $(D), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" lsEg
@load "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" lsinfo
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" ψ


@time "calculate observables" begin
    calObs!(Obs, ψ;destroy = false)
end

obsdata = Obs.values

gsdata = Dict(
    "obs" => obsdata,
    "SE" => lsinfo[end].S
)

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" gsdata

GC.gc()
end

# map(x -> showdomain(x.A),ψ.ts)