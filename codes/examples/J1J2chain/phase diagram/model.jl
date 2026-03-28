function Hamiltonian(Latt::AbstractLattice;J1::Number=1, J2::Number = 0.5)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = SU₂Spin
    
        for pair in neighbor(Latt;level = 1)
            addIntr!(Root,LocalSpace.SS,pair,("S","S"),J1,nothing)
        end

        for pair in neighbor(Latt;level = 2)
            addIntr!(Root,LocalSpace.SS,pair,("S","S"),J2,nothing)
        end

        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end
    
    return H
end