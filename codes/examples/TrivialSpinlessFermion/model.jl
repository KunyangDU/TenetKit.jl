
function ϵ(k)
    return -2cos(k)
end

function fe(β,L)
    lsk = @. (1:L) / (L+1) * pi
    return - sum(@. log(1+exp(-β*(ϵ(lsk))))) / β / L
end

function ce(β,L)
    lsk = @. (1:L) / (L+1) * pi
    return β^2/2 * sum(@. ϵ(lsk)^2/(1 + cosh(β * ϵ(lsk)))) / L
end

function ue(β,L)
    lsk = @. pi * (1:L) / (L+1)
    lsum = @.  ϵ(lsk) / (1 + exp( β * ϵ(lsk)))
    return sum(lsum) / L
end

function ParticleNumber(Latt::AbstractLattice)
    N = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinlessFermion
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.n,i,"n",1,nothing)
        end
    
        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end

    return N
end

function Hamiltonian(Latt::AbstractLattice;t::Number=1,μ::Number=0)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinlessFermion
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.n,i,"n",μ,nothing)
        end
        
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.F⁺F,pair,("F⁺","F"),-t,LocalSpace.Z)
            addIntr!(Root,LocalSpace.FF⁺,pair,("F","F⁺"),t,LocalSpace.Z)
        end
    
        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end

    return H
end
