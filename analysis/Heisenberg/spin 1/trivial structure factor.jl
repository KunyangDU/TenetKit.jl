using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("model.jl")

trivialname = "../codes/examples/Heisenberg/spin 1/data/trivial"
su2name = "../codes/examples/Heisenberg/spin 1/data/SU2"
u1name = "../codes/examples/Heisenberg/spin 1/data/U1"
edname = "Heisenberg/spin 1/data"


Lx = 100
Ly = 1
D = 3^4
su2params = (J=1,)
u1params = (Jz=1, Jxy = 1/2)
trivialparams = (J=1,)

@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(trivialname)/lsEg_$(Lx)x$(Ly)_$(D)_$(trivialparams).jld2" lsEg
@load "$(trivialname)/gsdata_$(Lx)x$(Ly)_$(D)_$(trivialparams).jld2" gsdata

lskx = pi*range(-1,1,201)
lsky = [0,]
lsk = filter(x -> isinside(x,MFBZpoint;isboundary = true),[[kx,ky] for kx in lskx,ky in lsky][:])
lstk = map(x -> Tuple(x),lsk)
x = map(lsk) do k
    k[1]
end
y = map(lsk) do k
    k[2]
end

FSxSx,FSySy,FSzSz = TrivialSSFT(Latt,gsdata,lstk,Ly+1:size(Latt)-Ly;SSonsite = 2/3)

Sm = (maximum(FSzSz .+ FSxSx .+ FSySy))


figsize = (width = 400,height = 220)
# insetsize = (width = 400,height = 130)

fig = Figure()

ax = Axis(fig[1,1];figsize...,
# title = "S = 1 Heisenberg chain, D = $(D), ΔE = $(round((fit.param[2] - Einf)*1e7;digits = 3))e-7",
# ylabel = L"E_0 / L",
# xlabel = L"1/L",
# xticks = (vcat(0,1 ./ vcat(lsLxed,[20,40,80])), [L"\frac{1}{\infty}",L"\frac{1}{4}",L"\frac{1}{6}",L"\frac{1}{8}",L"\frac{1}{10}",L"\frac{1}{20}",L"\frac{1}{40}",L"\frac{1}{80}"])
# xticks = vcat(lsLxed,[20,40,100]) |> y -> (vcat(0,1 ./ y) , vcat(L"1/\infty",map(x -> "1/$(x)",y))),
)


lines!(ax,x,(FSxSx .+ FSySy .+ FSzSz))

# scatter!(ax, 1 ./ lsLxed, Eed ./ lsLxed, color = :white, strokecolor = :green, marker = :diamond, markersize = 18, strokewidth = 2,label = L"\mathrm{ED}")
# scatter!(ax, 1 ./ lsLx[1:end-3], lsEgu1[1:end-3] ./ lsLx[1:end-3], color = :white, strokecolor = :red, marker = :circle, markersize = 14, strokewidth = 2,label = L"\mathrm{U(1)}")
# scatterlines!(ax,1 ./ lsLx, Esu2 ./ lsLx, color = :blue,  markersize = 12, label = L"\mathrm{SU(2)}")
# scatter!(ax, 1 ./ lsLx, lsEgtr ./ lsLx, color = :gold, marker = :star4, markersize = 10,label = L"\mathrm{NonSym.}")

# scatter!(ax, 0, Einf,color = :red, markersize = 12, marker = :star5, label = L"\mathrm{CBA}")
# # text!(ax,0.15,-1.35,text = "ΔE = $(round((fit.param[2] - Einf)*1e5;digits = 3))e-5")

# xlims!(ax,-0.01,0.26)

# inset_ax = Axis(fig[2,1];insetsize...,
# backgroundcolor = :white,
# ylabel = L"\mathrm{Error}\ E_0",
# xlabel = L"1/L",
# yscale = log10,
# xticks = (vcat(0,1 ./ vcat(lsLxed,[20,40,80])), [L"\frac{1}{\infty}",L"\frac{1}{4}",L"\frac{1}{6}",L"\frac{1}{8}",L"\frac{1}{10}",L"\frac{1}{20}",L"\frac{1}{40}",L"\frac{1}{80}"])
# # xticks = vcat(lsLxed,[20,40,100]) |> y -> (vcat(0,1 ./ y) , vcat(L"1/\infty",map(x -> "1/$(x)",y))),
# )

# scatterlines!(inset_ax,color = :blue, 1 ./ lsLxed, abs.((Esu2[eachindex(lsLxed)] .- Eed ) ./ Eed), label = L"E_\mathrm{SU(2)}-E_{\mathrm{ED}}")
# scatterlines!(inset_ax,color = :red, 1 ./ lsLxed, abs.((lsEgu1[eachindex(lsLxed)] .- Eed ) ./ Eed), label = L"E_\mathrm{U(1)}-E_{\mathrm{ED}}")
# scatterlines!(inset_ax,color = :purple, 1 ./ lsLxed, abs.((lsEgtr[eachindex(lsLxed)] .- Eed ) ./ Eed), label = L"E_\mathrm{NonSym.}-E_{\mathrm{ED}}")
# scatterlines!(inset_ax,color = :green, 1 ./ lsLx, abs.((lsEgtr .- Esu2 ) ./ Esu2), label = L"E_{\mathrm{SU(2)}}-E_{\mathrm{NonSym.}}")
# # scatterlines!(inset_ax,color = :blue, 1 ./ lsLx, abs.((lsEgtr .- Esu2 ) ./ Esu2))

# xlims!(inset_ax,-0.01,0.26)
# # ylims!(inset_ax,0,1/3)
# hidexdecorations!(ax,grid = false,ticks = false)
resize_to_layout!(fig)

# axislegend(ax,position = :lt)

display(fig)

save("Heisenberg/spin 1/figures/spin structure.pdf",fig)
save("Heisenberg/spin 1/figures/spin structure.png",fig)
