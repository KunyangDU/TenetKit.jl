using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
dataname = "examples/BCAO/arxivJKGGp/data/pin"

D = 2^6
Lx = 6
Ly = 4

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=0.0,J3xy=0.0,J3z=0.0)
# params = (J1=-0.1,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.3,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinOneHalf

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
        addObs!(Obs,LocalSpace.Sy,i,"Sy",nothing)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    for i in 1:size(Latt),j in i+1:size(Latt)
        addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),nothing)
        addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),nothing)
        addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
    end

    
end
# for J1 in vcat(0.45:0.01:0.49,0.51:0.01:0.55)
#     @show J1
# params = (J1=-0.46,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.008)
# params = (J1=-0.50035,K1=-0.85893,Γ1=0.4538,Γ1′=0.09311,J2= -0.0321,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.28,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.036,J3xy=0.26,J3z=0.0078)
# for J1 in -0.513:-0.002:-0.515
# params = (J1=J1,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.0078)
params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.093,J2=-0.032,J3xy=0.26,J3z=0.0078)

@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
xbonds,ybonds,zbonds = getxyzbonds(Latt)


calObs!(Obs, ψ;destroy = false)
gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

# end