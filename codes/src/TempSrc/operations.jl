


"""
mul!(C, A, B, α, β) -> C
Combined inplace matrix-matrix or matrix-vector multiply-add \$A B α + C β\$.
The result is stored in C by overwriting it. Note that C must not be aliased with either A or B.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of C, A, B.
Nsweep: times of variational calculation (sweep). Default is 2.
"""
function mul!(C::DenseMPO, A::Union{DenseMPO,SparseMPO}, B::Union{DenseMPO,SparseMPO}, α::Number, β::Number; kwargs...)
    #D_MPO = get(kwargs, :D_MPO, maximum(vcat(map(size, filter(x -> typeof(x) <: DenseMPO.[A,B])[1].ts)...)))
    D = get(kwargs, :D, maximum(vcat(collect.(map(size, filter(x -> typeof(x) <: DenseMPO, [A,B])[1].ts))...)))
    Nsweep = get(kwargs, :Nsweep, 2)

    @assert length(A) == length(B)
    L = length(A)

    tmp = deepcopy(C)'
    EnvAB = Environment([deepcopy(A),deepcopy(B),tmp])
    EnvC = Environment([deepcopy(C),tmp])
    initialize!(EnvAB)
    initialize!(EnvC)

    for i in 1:Nsweep
        totaltruncerror = 0
        for site in 1:L-1
            tl, tr, temptruncerr = tsvd(let 
                ts = map(z -> contract(z.envs[site], vcat(map(u -> z.layer[u].ts[site:site+1],1:length(z.layer)-1)...)..., z.envs[site+2]),[EnvAB,EnvC])
                axpby!(α, β, ts...)
            end; direction=:right,trunc = truncdim(D))
            map(z -> pushright!(z, tl, tr),[EnvAB,EnvC])
            totaltruncerror = max(totaltruncerror,temptruncerr)
        end
        for site in L:-1:2
            tl, tr, temptruncerr = tsvd(let 
                axpby!(α, β,map(z -> contract(z.envs[site-1], vcat(map(u -> z.layer[u].ts[site-1:site],1:length(z.layer)-1)...)..., z.envs[site+1]),[EnvAB,EnvC])...)
            end; direction=:left,trunc = truncdim(D))
            map(z -> pushleft!(z, tl, tr),[EnvAB,EnvC])
            totaltruncerror = max(totaltruncerror,temptruncerr)
        end
    end

    @assert EnvAB.layer[end] == EnvC.layer[end]
    return xpy!(EnvAB.layer[end]',C)
end
"""
axpy!(α, x, y) -> y
Overwrite y with x * α + y and return y. If x and y have the same axes, it's equivalent with y .+= x .* a.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of x, y.
Nsweep: times of variational calculation (sweep). Default is 2.

"""
function axpy!(α::Number, x::DenseMPO{L}, y::DenseMPO{L};kwargs...) where L
    D = get(kwargs, :D, max(map(y -> maximum(vcat(collect.(map(size, y.ts))...)),[x,y])...))
    Nsweep = get(kwargs, :Nsweep, 3)

    tmp = deepcopy(y)'
#=     @time "Initialize x,y Environment" begin
        Envx = Environment([deepcopy(x),tmp])
        Envy = Environment([deepcopy(y),tmp])
        initialize!(Envx)
        initialize!(Envy)
    end =#
    Envx = Environment([deepcopy(x),tmp])
    Envy = Environment([deepcopy(y),tmp])
    initialize!(Envx)
    initialize!(Envy)

    for i in 1:Nsweep
        totaltruncerror = 0
        for site in 1:L-1
            tl, tr, temptruncerr = tsvd(let 
                axpy!(α,map(z -> contract(z.envs[site], z.layer[1].ts[site:site+1]..., z.envs[site+2]),[Envx,Envy])...)
            end; direction=:right,trunc = truncdim(D))
            map(z -> pushright!(z, tl, tr),[Envx,Envy])
            totaltruncerror = max(totaltruncerror,temptruncerr)
        end
        for site in L:-1:2
            tl, tr, temptruncerr = tsvd(let 
                axpy!(α,map(z -> contract(z.envs[site-1], z.layer[1].ts[site-1:site]..., z.envs[site+1]),[Envx,Envy])...)
            end; direction=:left,trunc = truncdim(D))
            map(z -> pushleft!(z, tl, tr),[Envx,Envy])
            totaltruncerror = max(totaltruncerror,temptruncerr)
        end
        #= @time "sweep $i finished, max truncation error = $(totaltruncerror)" begin
            #println(">>>>>> Right >>>>>>")
            for site in 1:L-1
                tl, tr, temptruncerr = tsvd(let 
                    axpy!(α,map(z -> contract(z.envs[site], z.layer[1].ts[site:site+1]..., z.envs[site+2]),[Envx,Envy])...)
                end; direction=:right,trunc = truncdim(D_MPO))
                map(z -> pushright!(z,map(DenseMPOTensor, [tl, tr])...),[Envx,Envy])
                totaltruncerror = max(totaltruncerror,temptruncerr)
            end
            #println("<<<<<< Left <<<<<<")
            for site in L:-1:2
                tl, tr, temptruncerr = tsvd(let 
                    axpy!(α,map(z -> contract(z.envs[site-1], z.layer[1].ts[site-1:site]..., z.envs[site+1]),[Envx,Envy])...)
                end; direction=:left,trunc = truncdim(D_MPO))
                map(z -> pushleft!(z,map(DenseMPOTensor, [tl, tr])...),[Envx,Envy])
                totaltruncerror = max(totaltruncerror,temptruncerr)
            end
        end =#
    end

    @assert Envx.layer[2] == Envy.layer[2]
    return xpy!(Envx.layer[2]',y)
end

function axpy!(α::Number, x::CompositeMPOTensor{N₁,R₁}, y::CompositeMPOTensor{N₂,R₂}) where {N₁,R₁,N₂,R₂}
    return axpby!(α,1,x,y)
end

function axpby!(α::Number, β::Number, x::CompositeMPOTensor{N₁,R₁}, y::CompositeMPOTensor{N₂,R₂}) where {N₁,R₁,N₂,R₂}
    @assert N₁ == N₂ && R₁ == R₂
    y.A = x.A * α + y.A * β
    return y
end

function axpby!(::Number, β::Number, ::Nothing, y::CompositeMPOTensor)
    y.A = y.A * β
    return y
end

function xpy!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO}
    y.ts[:] = x.ts[:]
    return y
end

function tr(ρ::DenseMPO)
    return tr(ρ,ρ')
end

function tr(ρ1::DenseMPO,ρ2::AdjointMPO)
    Env = Environment([deepcopy(ρ1),deepcopy(ρ2)])
    initialize!(Env)
    return _scalar(Env)
end

function tr(ρ::DenseMPO, Opr::SparseMPO)
    Env = Environment([deepcopy(ρ), Opr, ρ'])
    initialize!(Env)
    return _scalar(Env)
end


function _scalar(Env::Environment{N}) where N
    @assert (site = Env.center[1]) == Env.center[2]
    t1 = map(x -> Env.layer[x].ts[site], 1:length(Env.layer))
    tmp = contract(Env.envs[site],t1...,Env.envs[site+1])
    return tmp
end
