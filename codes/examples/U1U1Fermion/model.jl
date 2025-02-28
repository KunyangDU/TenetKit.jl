function Hamiltonian(Latt::AbstractLattice; t::Number = 1, U::Number = 8, μ::Number = 0)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = U₁U₁Fermion
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.nd,i,"nd",U,nothing)
            addIntr!(Root,LocalSpace.n,i,"n",-μ,nothing)
        end
        
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.F₊⁺F₊,pair,("F⁺","Fup"),-t,LocalSpace.Z)
            addIntr!(Root,LocalSpace.F₋⁺F₋,pair,("F⁺","Fdn"),-t,LocalSpace.Z)
            addIntr!(Root,LocalSpace.F₊F₊⁺,pair,("F","F⁺up"),t,LocalSpace.Z)
            addIntr!(Root,LocalSpace.F₋F₋⁺,pair,("F","F⁺up"),t,LocalSpace.Z)
        end
    
        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end
    return H
end

function ϵ(k)
    return -2*sum(cos.(k))
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
        lsk = getk(Lx)
    else
        lskx = getk(Lx)
        lsky = getk(Ly;condition = :pbc)
        lsk = [[kx,ky] for kx in lskx,ky in lsky][:]
    end
    return lsk
end

function ue(β::Number,Lx::Int,Ly::Int)
    lsk = getk(Lx,Ly)
    lsum = @.  ϵ(lsk) / (1 + exp( β * ϵ(lsk)))
    return 2sum(lsum) / Lx / Ly
end

function fe(β::Number,Lx::Int,Ly::Int)
    lsk = getk(Lx,Ly)
    return - 2sum(@. log(1+exp(-β*(ϵ(lsk))))) / β / Lx / Ly
end

function ce(β::Number,Lx::Int,Ly::Int)
    lsk = getk(Lx,Ly)
    return 2β^2/2 * sum(@. ϵ(lsk)^2/(1 + cosh(β * ϵ(lsk)))) / Lx / Ly
end

