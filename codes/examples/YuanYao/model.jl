

function MGHamiltonian(Latt::AbstractLattice;
    J₁::Number=1, J₂::Number = 0, PBC::Bool = false
    )

    LocalSpace = TrivialSpinOneHalf
  
    Root = InteractionTreeNode()
    
    for pair in neighbor(Latt;level = 1)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J₁,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J₁,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J₁,nothing)
    end

    for pair in neighbor(Latt;level = 2)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J₂,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J₂,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J₂,nothing)
    end

    if PBC
        for pair in [(1,size(Latt)),]
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J₁,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J₁,nothing)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J₁,nothing)
        end

        for pair in [(1,size(Latt)-1),(2,size(Latt))]
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J₂,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J₂,nothing)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J₂,nothing)
        end
    end

    

    return AutomataSparseMPO(Root,size(Latt))  
end


function MGHamiltonian1(Latt::AbstractLattice;
    J₁::Number=1, J₂::Number = 0, PBC::Bool = false
    )

    LocalSpace = TrivialSpinOne
  
    Root = InteractionTreeNode()
    
    for pair in neighbor(Latt;level = 1)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J₁,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J₁,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J₁,nothing)
    end

    for pair in neighbor(Latt;level = 2)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J₂,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J₂,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J₂,nothing)
    end

    if PBC
        for pair in [(1,size(Latt)),]
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J₁,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J₁,nothing)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J₁,nothing)
        end

        for pair in [(1,size(Latt)-1),(2,size(Latt))]
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J₂,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J₂,nothing)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J₂,nothing)
        end
    end

    

    return AutomataSparseMPO(Root,size(Latt))  
end
