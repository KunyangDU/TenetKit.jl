
function Hamiltonian(Latt::AbstractLattice;
    J::Number=1,hx::Number=0,hy::Number=0,hz::Number = 0,
    returntree::Bool=false)

    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()
    
    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J,nothing)
    end

    if returntree
        return InteractionTree(Root)
    else
        return AutomataSparseMPO(InteractionTree(Root),size(Latt))  
    end
    
end

