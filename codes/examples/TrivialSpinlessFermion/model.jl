# Spinless Fermion 哈密顿量 — 基于 InteractionGraph + SparseMPO
# 新 API 版本，与 Heisenberg model.jl 模式一致

function YCRect(L::Int64, W::Int64, (a,b)::NTuple{2,Float64} = (1.0,1.0), θ::Real = 0.0)
    e = ((a, 0.0), (0.0, b))
    sites = [(x, y) for x in 1:L for y in 1:W]
    if iszero(θ)
         BC = PeriodicBoundaryCondition((0, W))
    else
         BC = TwistBoundaryCondition((0, W), θ)
    end
    return SquareLattice(e, sites, BC)
end

function ϵ(k)
    return -2sum(cos.(k))
end

function getk(L::Int;condition = :obc)
    if condition == :obc
        return @. pi * (1:L) / (L+1)
    elseif condition == :pbc
        return @. 2pi * (1:L) / L
    end
end

function getk(Lx::Int,Ly::Int)
    if Ly == 1
        lsk = getk(Lx;condition = :obc)
    else
        lskx = getk(Lx;condition = :obc)
        lsky = getk(Ly;condition = :pbc)
        lsk = [[kx,ky] for kx in lskx,ky in lsky][:]
    end
    return lsk
end

function ue(β::Number,Lx::Int,Ly::Int)
    lsk = getk(Lx,Ly)
    lsum = @.  ϵ(lsk) / (1 + exp( β * ϵ(lsk)))
    return sum(lsum) / Lx / Ly
end

function fe(β::Number,Lx::Int,Ly::Int)
    lsk = getk(Lx,Ly)
    return - sum(@. log(1+exp(-β*(ϵ(lsk))))) / β / Lx / Ly
end

function ce(β::Number,Lx::Int,Ly::Int)
    lsk = getk(Lx,Ly)
    return β^2/2 * sum(@. ϵ(lsk)^2/(1 + cosh(β * ϵ(lsk)))) / Lx / Ly
end

function ParticleNumber(Latt::AbstractLattice)
    L = size(Latt)
    LocalSpace = TrivialSpinlessFermion
    ig = InteractionGraph(L)

    for i in 1:L
        addIntr!(ig, LocalSpace.n, i, "n", false, 1, nothing)
    end

    return AutomataSparseMPO(ig)
end

function Hamiltonian(Latt::AbstractLattice; t::Number=1, μ::Number=0)
    L = size(Latt)
    LocalSpace = TrivialSpinlessFermion
    ig = InteractionGraph(L)

    for i in 1:L
        addIntr!(ig, LocalSpace.n, i, "n", false, μ, nothing)
    end

    for pair in neighbor(Latt)
        addIntr!(ig, LocalSpace.F⁺F, pair, ("F⁺","F"), (true,true), -t, LocalSpace.Z)
        addIntr!(ig, LocalSpace.FF⁺, pair, ("F","F⁺"), (true,true),  -t, LocalSpace.Z)
    end

    return AutomataSparseMPO(ig)
end
