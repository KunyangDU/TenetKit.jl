using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/Heisenberg/trivial/data"

D = 128
Lx = 20
Ly = 4
params = (J=1,Δ=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
# @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# E0 = lsEg[end]


lsk = vcat()
lsω = range(0,4,100)

t = 10
Nt = 20

lst = range(0,t,Nt)

dySS = zeros(length(lsk),length(lsω),3)

lsvs = [[1,0],[1,1]]
# lslabel = [L"(0,0)",L"(\pi/2,\pi/2)",L"(\pi,0)",L"(\pi/2,\pi/2)",L"(0,0)",L"(\pi,0)"]
lslabel = [L"(0,\pi)",L"(\pi,\pi)"]
lsk = let lsk = []
    for i in eachindex(lsvs[1:end-1])
    v₀ = lsvs[i]
    v₁ = lsvs[i+1]
    lsc = 0.125:0.125:1
    lsk = vcat(lsk,[pi*(v₀ + c*(v₁-v₀)) for c in lsc])
    end
    @show lsk
    vcat([lsvs[1]*pi,],lsk)
end
lskr = vcat(0,cumsum(norm.(diff(lsk))))
lsknode = vcat(0,cumsum(norm.(diff(pi .* lsvs))))

lst = range(0,t,Nt)

dySS = zeros(length(lsk),length(lsω),3)

for (i,k) in enumerate(lsk)
    @load "$(dataname)/lsSS_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(t)_$(Nt).jld2" lsSS
    map(1:3) do x
        dySS[i,:,x] = [-2/pi*sum(imag(lsSS[x]) .* sin.(lst * ω) .* window.(lst/maximum(lst))) for ω in lsω]
    end
end

dySS = dySS / size(Latt)

vpath,rpath,rnode = vrange(lsvs .* pi,100)
function ωk(k::Vector;J = 1,S = 1/2,a = 1)
    return 4*J*S*sqrt(1-sum(cos.((k .* a[1,1])))^2/4)
end


figsize = (width = 300,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xticks = (lsknode,lslabel),
title = "AFMH Squa$(Ly)x$(Lx), D=$(D), ϵ=$(round(8/t;digits = 3))",
# xlabel = L"\mathbf{k}",
ylabel = L"\hbar\omega/J",
# xticks = lsknode
)
hm = heatmap!(ax,lskr,lsω,sum([dySS[:,:,i] for i in 1:3]),colormap = :Blues,colorrange = (0,5))
lines!(ax,rpath,1.1727*ωk.(vpath),color = :red,)

# # lines!(ax,lskxc,pi*sin.(lskxc/2),color = :black)
# # lines!(ax,lskxc,pi*abs.(sin.(lskxc))/2,color = :black)

Colorbar(fig[1,2],hm;label = L"A(\mathbf{k},\omega)")

resize_to_layout!(fig)
display(fig)
# dySS
save("Heisenberg/trivial/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/trivial/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


