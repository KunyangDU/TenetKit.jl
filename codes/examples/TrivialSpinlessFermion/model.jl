
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
    N = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinlessFermion
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.n,i,"n",false,1,nothing)
        end
    
        AutomataSparseMPO(Root,size(Latt))
    end

    return N
end

function Hamiltonian(Latt::AbstractLattice;t::Number=1,μ::Number=0)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinlessFermion
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.n,i,"n",false,μ,nothing)
        end
        
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.F⁺F,pair,("F⁺","F"),(true,true),-t,LocalSpace.Z)
            addIntr!(Root,LocalSpace.FF⁺,pair,("F","F⁺"),(true,true),t,LocalSpace.Z)
        end
        
        AutomataSparseMPO(Root,size(Latt))
    end

    return H
end
