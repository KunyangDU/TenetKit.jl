
using TensorKit,CairoMakie
include("../../src/iMPS.jl")
include("model.jl")

Lx = 11
Ly = 1
D = 50

Latt = YCSqua(Lx,Ly)

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    PhySpace = TrivialSpinOneHalf.PhySpace 
    randMPS(PhySpace,AuxSpace)
end

params = (J=0,h=0,hz=1)

H,r = Hamiltonian(Latt;params...)
lsE = DMRG1!(ψ,H,D,1e-6;cbe=true,Nsweep=3)

begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinOneHalf
    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
    end
    calObs!(Obs, ψ)
end
#@show Szs = [Obs.values["Sz"][(i,)] for i in 1:size(Latt)]
#@show Sxs = [Obs.values["Sx"][(i,)] for i in 1:size(Latt)]


params = (J=1,h=1.,hz=0)

H,r = Hamiltonian(Latt;params...)
T = 6/params.J
Nt = 20

lsψ, lst = TDVP1!(ψ, H, T, Nt, D)
Szm = zeros(length(lst),size(Latt))
for ind in eachindex(lsψ)
    begin
        Obs = MPSObservable()
        LocalSpace = TrivialSpinOneHalf
        for i in 1:size(Latt)
            addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
        end
        calObs!(Obs, lsψ[ind])
    end
    Szs = [Obs.values["Sz"][(i,)] for i in 1:size(Latt)]
    Szm[ind,:] = Szs
end

fig = Figure()
ax = Axis(fig[1,1])

heatmap!(ax,lst,1:size(Latt),Szm)

resize_to_layout!(fig)
display(fig)

