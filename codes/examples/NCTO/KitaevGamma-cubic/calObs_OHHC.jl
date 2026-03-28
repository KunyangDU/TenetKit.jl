using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/KitaevGamma-cubic/data/OHHC"

D = 128
L= 3

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)

@load "$(dataname)/Latt_$(L).jld2" Latt

# Hf = 0.

θ = 0.0
ϕ = pi / 2

K = 1
Γ = 0.0

params1_Kitaev = (K = K, Γ = Γ)

for Hf in 0:0.04:0.8
# Hf = 0.0

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

println("$(size(Latt))), D = $(D), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/lsEg_$(L)_$(D)_$(params_Kitaev).jld2" lsEg
@load "$(dataname)/lsinfo_$(L)_$(D)_$(params_Kitaev).jld2" lsinfo
@load "$(dataname)/ψ_$(L)_$(D)_$(params_Kitaev).jld2" ψ

flux_Latt = _OHTria(L,((1.0, 0.0),(1/2, sqrt(3)/2));scale = sqrt(3))
direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]]
fluxsites,fluxdirections,_ = getFlux(Latt,flux_Latt,direction)

@time "calculate observables" begin
    Obs = Observable()
    LocalSpace = TrivialSpinOneHalf
    names = ("Sx","Sy","Sz")
    Ops = (LocalSpace.Sx,LocalSpace.Sy,LocalSpace.Sz)
    fluxnames = map(y -> map(x -> names[x],y),fluxdirections)
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

    calObs!(Obs, ψ)
end

obsdata = Obs.values


area = Dict(
    2 => Dict(
        "A" => [2,3,4,10,11,18],
        "B" => [1,5,6,7,14,15],
        "C" => [19,20,21,22,23,24],
        "D" => [8,9,12,13,16,17]
    ),
    3 => Dict(
        "A" => [2,3,4,11,12,13,14,23,24,25,36,37],
        "B" => [1,5,6,7,8,17,18,19,20,30,31,42],
        "C" => [43,44,45,46,47,48,49,50,51,52,53,54],
        "D" => [9,10,15,16,21,22,26,27,28,29,32,33,34,35,38,39,40,41]
    )
)

# @time "calculate TEE" begin
#     @time "SA" SA = vonNeumann(ψ,area[L]["A"])
#     @time "SB" SB = vonNeumann(ψ,area[L]["B"])
#     @time "SC" SC = vonNeumann(ψ,area[L]["C"])
#     @time "SAB" SAB = vonNeumann(ψ,vcat(area[L]["A"],area[L]["B"]))
#     @time "SBC" SBC = vonNeumann(ψ,vcat(area[L]["B"],area[L]["C"]))
#     @time "SAC" SAC = vonNeumann(ψ,vcat(area[L]["A"],area[L]["C"]))
#     @time "SABC" SABC = vonNeumann(ψ,vcat(area[L]["A"],area[L]["B"],area[L]["C"]))

#     TEE = (SA + SB + SC - SAB - SBC - SAC + SABC) / log(2)
# end

entropy_data = Dict(
    "area" => area,
    "SA" => SA,
    "SB" => SB,
    "SC" => SC,
    "SAB" => SAB,
    "SBC" => SBC,
    "SAC" => SAC,
    "SABC" => SABC,
    # "TEE" => TEE
)

gsdata = Dict(
    "obs" => obsdata,
    "TEE" => entropy_data,
    "SE" => lsinfo[end].S
)

@save "$(dataname)/gsdata_$(L)_$(D)_$(params_Kitaev).jld2" gsdata

# values.(values(obsdata))
end

