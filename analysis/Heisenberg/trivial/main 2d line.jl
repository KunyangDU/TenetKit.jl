using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/Heisenberg/trivial/data"

D = 128
Lx = 8
Ly = 4
params = (J=1,Δ=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
E0 = lsEg[end]

# lsky = [0,]
# lsk = [[kx,ky] for kx in lskx, ky in lsky][:]
lskxc = pi * range(0,2,101)
lskx = pi * (0:0.1:2)

lsk = [[pi/4,pi/4],[pi/2,pi/2],[3pi/4,3pi/4],[pi,pi]]
lsω = range(0,6,100)

t = 10
Nt = 30

lst = range(0,t,Nt)

dySS = zeros(length(lsk),length(lsω),3)

for (i,k) in enumerate(lsk)
    @load "$(dataname)/lsSS_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(t)_$(Nt).jld2" lsSS
    map(1:3) do x
        dySS[i,:,x] = [(-2/pi)*sum(imag.(lsSS[x]) .* sin.(lst * ω) .* window.(lst/maximum(lst))) for ω in lsω]
    end
end

dySS = dySS / size(Latt)

figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...)
for (ik,k) in enumerate(lsk)
lines!(ax,lsω,sum([dySS[ik,:,i] for i in 1:3]))
end
# hm = heatmap!(ax,lskx,lsω,sum([dySS[:,:,i] for i in 1:3]),colormap = :Blues)

# # lines!(ax,lskxc,pi*sin.(lskxc/2),color = :black)
# # lines!(ax,lskxc,pi*abs.(sin.(lskxc))/2,color = :black)

# Colorbar(fig[1,2],hm)

resize_to_layout!(fig)
display(fig)
dySS
# save("Heisenberg/trivial/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
# save("Heisenberg/trivial/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


