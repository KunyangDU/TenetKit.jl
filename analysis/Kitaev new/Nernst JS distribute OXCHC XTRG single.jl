using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit,Integrals
using Polynomials
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/XTRG/OXCHC"

FermionJS(x,p) =[FermionJS1(t,p) for t in x]
FermionJS1(x,p) = p[1]*solve(IntegralProblem((y,T) -> y^2/sqrt(y^2 +p[2]^2)*tanh(sqrt(y^2 + p[2]^2)/T/p[3]), (0.0, pi), x), QuadGKJL()).u


Lx = 8
Ly = 2
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 512
DS = 2^4
lsν = [0.0,0.5]
N = 20


params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.12)
Hc = params.Hc
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction = [[1,0],[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2]]
cornerpoints = vcat(1:6Ly,size(Latt)-6Ly+1:size(Latt))
xedgepoints = filter(x -> length(neighbor(Latt,x))==2 && x ∈ cornerpoints,1:size(Latt))
yedgepoints = filter(x -> length(neighbor(Latt,x))==2 && x ∉ cornerpoints,1:size(Latt))
yupedgepoints = filter(x -> Latt[x][2][2] == Ly, yedgepoints)
ydownedgepoints = filter(x -> Latt[x][2][2] == 1, yedgepoints)


Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")
proj = -[1,0]

lsJupedge = zeros(length(lsν),N,length(yupedgepoints))
lsJdownedge = zeros(length(lsν),N,length(ydownedgepoints))
lsβeff = zeros(length(lsν),N)

upbonds = map(y -> filter(x -> !isempty(intersect(x,yupedgepoints)), y),bonds)
downbonds = map(y -> filter(x -> !isempty(intersect(x,ydownedgepoints)), y),bonds)

for (iν,ν) in enumerate(lsν)
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_$(ν).jld2" lsβ
for i in 1:N
@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(ν)_$(i).jld2" data
obs = data["obs"]
for k in 1:3
    cinds = currentindex2(Js[k],h)
    for (ibond,(j,l)) in enumerate(upbonds[k])
        ans = 0.0
        for (jeff,(α,β)) in cinds
            ans += (real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff)
        end
        @show iν,i,ibond,j,l,ans,dot(relaVec(Latt,j,l),proj)
        lsJupedge[iν,i,ibond] += ans * dot(relaVec(Latt,j,l),proj)
    end
    for (ibond,(j,l)) in enumerate(downbonds[k])
        ans = 0.0
        for (jeff,(α,β)) in cinds
            ans += (real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff)
        end
        lsJdownedge[iν,i,ibond] += ans * dot(relaVec(Latt,j,l),-proj)
    end
end
lsβeff[iν,i] = lsβ[i]
end 
end


# iν = 1
# lsJupedge = lsJupedge[iν,:,:]
# lsJdownedge = lsJdownedge[iν,:,:]
# lsJedge = (lsJupedge .+ lsJdownedge) / 2

# figsize = (width = 400, height = 200)

# fig = Figure()
# ax = Axis(fig[1,1];
# yscale = log10,
# # xscale = log10,
# xminorticksvisible = true, 
# xminorticks = IntervalsBetween(10),
# yminorticksvisible = true,
# # xticks = [0.001,0.005,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
# # yticks = -500:50:500,
# yminorticks = IntervalsBetween(10),
# xgridvisible = false, ygridvisible = false,
# xminorgridvisible = false, yminorgridvisible = false,
# title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D)\n$(params)", 
# xlabel = L"T",
# ylabel = L"J_S",
# figsize...)

# for i in 10:N
#     scatterlines!(ax,1:length(yupedgepoints),abs.(lsJedge[i,:]),color = (sum(lsJedge[i,:]) > 0 ? :red : :blue,2(i-10)/N))
# end

# resize_to_layout!(fig)
# display(fig)
