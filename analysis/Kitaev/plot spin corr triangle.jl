using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data"

D = 2^7
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

Np = 25
@load "$(dataname)/params_$(Np).jld2" params
@load "$(dataname)/coparams_$(Np).jld2" coparams

lskx = 2pi/sqrt(3)*range(-1,1,21)
lsky = 4pi/3*range(-1,1,21)
lsk = filter(x -> isinside(x,FBZpoint),[[kx,ky] for kx in lskx,ky in lsky][:])
theossz = map(x -> getKitaevSS(lsk,x),params)
@load "$(dataname)/ssz_$(Lx)x$(Ly)_$(D)_$(Np).jld2" ssz
trssz = ssz

figsize = (width = 450,height = 200)
figsizez = (height = 150,)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
title = "Kitaev model, ⟨SzSz⟩, $(2Lx)x$(2Ly), D = $(D)")


x = map(coparams) do p
    p[1]
end
y = map(coparams) do p
    p[2]
end

theoshift = -[sqrt(3)/2 .+ 0.4,0]
trshift = [sqrt(3)/2 .+ 0.4,0]

hm = heatmap!(ax,x .+ theoshift[1],y,-theossz,colormap = :Reds,colorrange = (0.,0.25))
heatmap!(ax,x .+ trshift[1],y,-trssz,colormap = :Reds,colorrange = (0.,0.25))

Colorbar(fig[1,2],hm,label = L"\langle S_zS_z\rangle")
boundary!(ax,Triapoint;basis = BASIS2,linewidth = 3,shift = theoshift)
boundary!(ax,KitaevBDpoiont;basis = BASIS2,linestyle = :dash,color = :white,shift = theoshift)

boundary!(ax,Triapoint;basis = BASIS2,linewidth = 3,shift = trshift)
boundary!(ax,KitaevBDpoiont;basis = BASIS2,linestyle = :dash,color = :white,shift = trshift)

# scatter!(ax,x[13] .+ theoshift[1],y[13],color = :red)

for (i,txt) in enumerate([L"J_x",L"J_y",L"J_z"])
    text!(ax,(coordinate(JBASIS2[i]*1.3;basis = BASIS2) .+ theoshift)...;text = txt,align = (:center,:center),fontsize = 20)
    text!(ax,(coordinate(JBASIS2[i]*1.3;basis = BASIS2) .+ trshift)...;text = txt,align = (:center,:center),fontsize = 20)
end
text!(ax,2theoshift[1] + 0.2,1;text = L"\mathrm{Theory}",align = (:left,:center),fontsize = 20)
text!(ax,0.2,1;text = L"\mathrm{DMRG}",align = (:left,:center),fontsize = 20)

ylims!(ax,-0.85,1.45)
# hidedecorations!(ax)

axz = Axis(fig[2,:];figsizez...,
xlabel = L"J_z",ylabel = L"\langle S_zS_z\rangle",
yticks = 0:0.05:0.25)

lsJz = 0:0.1:2
@load "$(dataname)/ssz_$(Lx)x$(Ly)_$(2^9)_$(length(lsJz)).jld2" ssz
jzssz = ssz
lines!(axz,lsJz,0.25*ones(length(lsJz)),color = :grey,linestyle = :dash)
lines!(axz,[1,1],[0,0.25],color = :grey,linestyle = :dash)

scatter!(axz,lsJz,-map(x -> getKitaevSS(lsk,[0.5,0.5,x]),lsJz),strokewidth = 2,label = L"\mathrm{Theory}",markersize = 12,color = :white,strokecolor = :red)
scatterlines!(axz,lsJz,-jzssz,label = L"\mathrm{DMRG}")

text!(axz,0.75,0.02;text = L"\mathrm{Gapless}",align = (:center,:center),fontsize = 20)
text!(axz,1.25,0.02;text = L"\mathrm{Gapped}",align = (:center,:center),fontsize = 20)

axislegend(axz,position = :rb)


resize_to_layout!(fig)
display(fig)
save("Kitaev/figures/spin corr compare_$(Lx)x$(Ly)_$(D).png",fig)
save("Kitaev/figures/spin corr compare_$(Lx)x$(Ly)_$(D).pdf",fig)

