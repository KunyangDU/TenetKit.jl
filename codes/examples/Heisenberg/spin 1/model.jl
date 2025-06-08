
function TrivialHamiltonian(Latt::AbstractLattice;
    J::Number=1,H::Number = 0,
    D::Number = 0,
    pinh::Vector = repeat([zeros(3),],2*get_cellsize(Latt)[2]),
    pinsites::Vector = vcat(1:get_cellsize(Latt)[2], size(Latt)-get_cellsize(Latt)[2]+1:size(Latt))
    )

    LocalSpace = TrivialSpinOne
  
    Root = InteractionTreeNode()
    
    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",-H,nothing)
    end

    if sum(abs.(vcat(pinh...))) != 0
        @assert length(pinh) == length(pinsites) "pin field not compatible"
        for (i,site) in enumerate(pinsites)
            @show pinh
            addIntr!(Root,LocalSpace.Sx,site,"Sx",pinh[i][1],nothing)
            addIntr!(Root,LocalSpace.Sy,site,"Sy",pinh[i][2],nothing)
            addIntr!(Root,LocalSpace.Sz,site,"Sz",pinh[i][3],nothing)
        end
    end

    for site in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sz2,site,"Sz2",D,nothing)
    end

    return AutomataSparseMPO(InteractionTree(Root),size(Latt))  
    
end


function U1Hamiltonian(Latt::AbstractLattice;Jz::Number=1, Jxy::Number=1/2, h::Number = 0,H::Number = 0, D::Number = 0)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = U₁Spin1
    
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),Jz,nothing)
            addIntr!(Root,LocalSpace.S₊S₋,pair,("S₊","S₋"),Jxy,nothing)
            addIntr!(Root,LocalSpace.S₋S₊,pair,("S₋","S₊"),Jxy,nothing)
        end

        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.Sz,i,"Sz",-H,nothing)
        end

        addIntr!(Root,LocalSpace.Sz,div(size(Latt),2),"Sz",h,nothing)

        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.Sz2,i,"Sz2",D,nothing)
        end

        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end
    return H
end


function SU2Hamiltonian(Latt::AbstractLattice;J::Number=1)
    LocalSpace = SU₂Spin1

    Root = InteractionTreeNode()

    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.SS,pair,("S","S"),J,nothing)
    end
    
    return AutomataSparseMPO(InteractionTree(Root),size(Latt))
end


