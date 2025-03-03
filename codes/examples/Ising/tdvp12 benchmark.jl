
using TensorKit,CairoMakie,LaTeXStrings
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

params = (J=1,h=1.,hz=0)

H,r = Hamiltonian(Latt;params...)
T = 6/params.J
Nt = 20

lsψ, lst = TDVP1!(deepcopy(ψ), H, T, Nt, D)
Szm = zeros(length(lst),size(Latt),2)
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
    Szm[ind,:,1] = Szs
end

lsψ, lst = TDVP2!(deepcopy(ψ), H, T, Nt, D)
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
    Szm[ind,:,2] = Szs
end
cind = div(size(Latt),2) + 1
error12 = [sum(abs.(Szm[i,:,1] - Szm[i,:,2])) for i in eachindex(lst)] / size(Latt)
errorlr1 = [sum(abs.(Szm[i,1:cind-1,1] - Szm[i,end:-1:cind+1,1])) for i in eachindex(lst)]
errorlr2 = [sum(abs.(Szm[i,1:cind-1,2] - Szm[i,end:-1:cind+1,2])) for i in eachindex(lst)]

figsize = (width=300,height =200)

fig = Figure()
ax1 = Axis(fig[1,1];figsize...,ylabel="site",yticks = 1:size(Latt))
ax2 = Axis(fig[2,1];figsize...,xlabel=L"tJ",ylabel="site",yticks = 1:size(Latt))
ax3 = Axis(fig[1,2];figsize...,yscale=log10,ylabel="asymmetric error")
ax4 = Axis(fig[2,2];figsize...,xlabel=L"tJ",ylabel="TDVP1 - TDVP2")

heatmap!(ax1,lst,1:size(Latt),Szm[:,:,1])
heatmap!(ax2,lst,1:size(Latt),Szm[:,:,2])
scatterlines!(ax3,lst,errorlr1,label="TDVP1")
scatterlines!(ax3,lst,errorlr2,label="TDVP2")
scatterlines!(ax4,lst,error12)

hidexdecorations!(ax1;ticks=false)
hidexdecorations!(ax3;ticks=false,grid=false)

axislegend(ax3;position=:rb)


resize_to_layout!(fig)
display(fig)

save("examples/ising/figures/ising_tdvp12_benchmark.pdf",fig)
save("examples/ising/figures/ising_tdvp12_benchmark.png",fig)

