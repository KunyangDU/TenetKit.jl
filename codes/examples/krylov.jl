using KrylovKit,LinearAlgebra,CairoMakie
#include("../../src/iMPS.jl")

#= A= randn(4000,4000) |> x -> x .+ x'
@time fk = eigsolve(A,1,:SR)
@time fl = eigen(A)
minimum(fl.values),minimum(fk[1]) =#

model(x) = @. -2 * sin(pi*x) * sqrt(abs(x))

x = range(-5,5,200)
figsize = (height=150,width=300)
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax,x,model(x))
lines!(ax,x,x)
resize_to_layout!(fig)
display(fig)

f = eigsolve(model,10.,2,:LR)
minimum(real.(f[1]))

