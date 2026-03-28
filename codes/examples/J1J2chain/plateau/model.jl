function Hamiltonian(Latt::AbstractLattice;
    J1::Number=1, J2::Number = 0.5, J1xy::Number = 0.0, 
    Hx::Number = 0.0, Hy::Number = 0.0, Hz::Number = 0.0,
    h::Number = 1e-2)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinOneHalf
    
        for pair in neighbor(Latt;level = 1)
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J1,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J1,nothing)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J1,nothing)

            addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),(false,false),J1xy,nothing)
            addIntr!(Root,LocalSpace.SySx,pair,("Sy","Sx"),(false,false),J1xy,nothing)
        end

        for pair in neighbor(Latt;level = 2)
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J2,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J2,nothing)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J2,nothing)
        end

        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.Sx,i,"Sx",false,-Hx,nothing)
            addIntr!(Root,LocalSpace.Sy,i,"Sy",false,-Hy,nothing)
            addIntr!(Root,LocalSpace.Sz,i,"Sz",false,-Hz,nothing)
        end

        addIntr!(Root,LocalSpace.Sz,1,"Sz",false,h,nothing)

        AutomataSparseMPO(Root,size(Latt))
    end
    
    return H
end