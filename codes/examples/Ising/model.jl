

function Hamiltonian(Latt::AbstractLattice;J::Number=1,h::Number=0.2,hz::Number=0)
    r,H = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinOneHalf
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.Sx,i,"Sx",h,nothing)
        end

        if hz != 0
            for i in 1:size(Latt)
                if i == div(size(Latt),2)+1
                    addIntr!(Root,LocalSpace.Sz,i,"Sz",hz,nothing)
                else
                    addIntr!(Root,LocalSpace.Sz,i,"Sz",-hz,nothing)
                end
            end
        end
        
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J,nothing)
        end
        tree = InteractionTree(Root)
        tree,AutomataSparseMPO(tree,size(Latt))
    end

    return H,r
end
