using CairoMakie,Integrals,LaTeXStrings

FermionJS(x,p) =[FermionJS1(t,p) for t in x]
FermionJS1(x,p) = p[1]*solve(IntegralProblem((y,T) -> y^2/sqrt(y^2 +p[2]^2)*tanh(sqrt(y^2 + p[2]^2)/2T), (0.0, 5pi), x), QuadGKJL()).u



model(x,a,T) = x^2/sqrt(x^2 +a^2)*tanh(sqrt(x^2 + a^2)/2T)

figsize = (width = 300,height=200)

fig = Figure()
ax0= Axis(fig[1,1];
figsize...,
# xscale= log10,
xlabel =  L"k_B T",ylabel = L"J_S"
# title = L"I\left(\frac{m}{\hbar v},T\right)="
# xscale = log10,
# xticks = [0.001,0.005,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
# yscale = log10
)

ax= Axis(fig[2,1];
figsize...,
xlabel =  L"k_B T",ylabel = L"\kappa/T"
# title = L"I\left(\frac{m}{\hbar v},T\right)="
# xscale = log10,
# xticks = [0.001,0.005,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
# yscale = log10
)

for a in [0.1]
    tmpf(x,T) = model(x,a,T)

    lsT  = range(0.0001,2,1000)
    
    # lsJ = zeros(length(lsT))
    # for (i,T) in enumerate(lsT)
    #     prob = IntegralProblem(tmpf, (0.0, sqrt(3)*pi),T)
    #     lsJ[i] = solve(prob, QuadGKJL()).u
    # end
    lsJ = FermionJS(lsT,[- 0.1,a])


    lsκ = diff(lsJ) ./ diff(lsT) ./ lsT[2:end]

    lines!(ax0,lsT,lsJ,label="m = $(a)",linewidth = 2)
    lines!(ax,lsT[1:end-1],lsκ,label="m = $(a)",linewidth = 2)
end

resize_to_layout!(fig)

axislegend(ax0,position = :lt)
# xlims!(ax0,1/4096,10 ^ (-1.5))

xlims!(ax0,0,2)
xlims!(ax,0,2)
# ylims!(ax,0,4.0)
display(fig)


save("Kitaev new/figures/1d_fermion_kappa.pdf",fig)
save("Kitaev new/figures/1d_fermion_kappa.png",fig)
