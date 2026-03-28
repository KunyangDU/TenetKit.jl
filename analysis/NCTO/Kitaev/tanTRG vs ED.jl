using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/NCTO/Kitaev/data"
figurename = "NCTO/Kitaev/figures"
tailname = ""
edname = "NCTO/Kitaev/ED/data"

D = 256
Lx = 2
Ly = 2

inputHxy = 0.0
Hx = 0.0
Hy = 0.0
Hz = inputHxy

params1_Kitaev = (J1 = 0.0, K1 = -1.0, Γ1 = 0.0, Γ1′ = 0.0)
params23 = (J2xy = 0.0, J3xy = 0.0, J3z = 0.0)
paramsH = (Hx = Hx, Hy = Hy, Hz = Hz)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params23,paramsH)
params_Kitaev = merge(params1_Kitaev,params23,paramsH)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/lsβ_$(Lx)x$(Ly).jld2" lsβ
mind = 114


lsβ = vcat(1.5 .^ (-15:1:-1),1:0.5:100)
# lsβ = vcat(2. .^ (-15:1:-1),1:100)
lsβ2 = lsβ[2:end]*2
lsβ2eff = lsβ2[1:mind-1]

lsE = zeros(mind-1)
lsF = zeros(mind-1)
for i in 2:mind
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_$(i).jld2" data
    lsE[i-1] = data["E"]
    lsF[i-1] = data["F"]
end

lsC = - lsβ2eff[2:end] .^ 2 .* diff(lsE) ./ diff(lsβ2eff)
lsS = (lsE - lsF) .* lsβ2eff

figsize = (height = 150,width = 400)

fig = Figure()
# axF = Axis(fig[1,1])

axE = Axis(fig[1,1];figsize...,ylabel = L"E/N",
xlabel = L"T",
xscale = log10)
axS = Axis(fig[2,1];figsize...,ylabel = L"S/N",yticks = 0:0.2:1,
xlabel = L"T",
xscale = log10)
axC = Axis(fig[3,1];figsize...,ylabel = L"C/N",
xlabel = L"T",
xscale = log10)


@load "$(edname)/eddata_diff_$(Lx)x$(Ly)_K=$(K).jld2" eddata
lsβedd = eddata["β"]
lsEedd = eddata["E"]
lsSedd = eddata["S"]
lsCedd = eddata["C"]

scatter!(axE, 1 ./ lsβedd, lsEedd / size(Latt),color = :white,strokewidth = 1.5,markersize = 14,strokecolor = :red,label = L"\mathrm{ED}")
scatter!(axS, 1 ./ lsβedd, lsSedd / size(Latt) / log(2),color = :white,strokewidth = 1.5,markersize = 14,strokecolor = :red)
scatter!(axC, 1 ./ lsβedd[2:end], lsCedd / size(Latt),color = :white,strokewidth = 1.5,markersize = 14,strokecolor = :red)


scatter!(axE,1 ./ lsβ2eff,lsE / size(Latt),label = L"\mathrm{tanTRG}")
lines!(axE,[10. ^ (-2.2), 10 .^ (0.2)],-1.5*ones(2) / size(Latt),linestyle = :dash,color = :grey,label = L"\mathrm{DMRG}")
scatter!(axC,1 ./ lsβ2eff[2:end],lsC / size(Latt))
scatter!(axS,1 ./ lsβ2eff,lsS / size(Latt) / log(2))

@load "$(edname)/eddata_$(Lx)x$(Ly)_K=$(K).jld2" eddata
lsβed = eddata["β"]
lsEed = eddata["E"]
lsSed = eddata["S"]
lsCed = eddata["C"]

lines!(axE,1 ./ lsβed,lsEed / size(Latt),color = :red,label = L"\mathrm{ED}")
lines!(axS,1 ./ lsβed,lsSed / size(Latt) / log(2),color = :red)
lines!(axC,1 ./ lsβed[2:end],lsCed / size(Latt),color = :red)



for ax in [axC,axS,axE,axCerr,axCerr,axSerr]
xlims!(ax,10. ^ (-2.2), 10 .^ (0.2))
end

ylims!(axS,0,1)


axEerr = Axis(fig[1,2];figsize...,ylabel = L"\mathrm{Err}\ E",
xlabel = L"T",
xscale = log10)
axSerr = Axis(fig[2,2];figsize...,ylabel = L"\mathrm{Err}\ S",
xlabel = L"T",
xscale = log10)
axCerr = Axis(fig[3,2];figsize...,ylabel = L"\mathrm{Err}\ C",
xlabel = L"T",
xscale = log10)

scatterlines!(axEerr, 1 ./ lsβ2eff, -(lsE .- lsEedd) ./ lsEedd)
scatterlines!(axSerr, 1 ./ lsβ2eff, -(lsS .- lsSedd) ./ lsSedd)
scatterlines!(axCerr, 1 ./ lsβ2eff[2:end], -(lsC .- lsCedd) ./ lsCedd)

hidexdecorations!(axE,ticks = false,grid = false)
hidexdecorations!(axS,ticks = false,grid = false)
hidexdecorations!(axEerr,ticks = false,grid = false)
hidexdecorations!(axSerr,ticks = false,grid = false)

Label(fig[1,:1:2][1, 1, Top()],"K = $(params1_Kitaev.K1), $(Ly)x$(Lx)x2 ZZ-HC-CY, D = $(D), $(paramsH)",
fontsize = 20,
font = :bold,
padding = (0, 0, 10, 0),
halign = :center
)
axislegend(axE,position = :lt)


resize_to_layout!(fig)
display(fig)



save("$(figurename)/tanTRG vs ED_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).png",fig)
save("$(figurename)/tanTRG vs ED_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).pdf",fig)

# lsβ[35]
lsS / size(Latt)
lsβ2eff[end]
lsE .- lsEedd