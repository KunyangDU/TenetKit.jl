
function TrivialHamiltonian(Latt::AbstractLattice;
    J1xy,J1z,Jpm,Jzpm,
    J3,D,H,kwargs...
    )

    LocalSpace = TrivialSpinOneHalf
  
    Root = InteractionTreeNode()
    cossin = [(1,0),(-1/2,sqrt(3)/2),(-1/2,-sqrt(3)/2)]
    direction = [[0,1],[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2]]
    bonds = getxyzbonds(Latt;direction=direction)
    for i in 1:3
        cs = cossin[i]
        paramsM = [
            J1xy+2cs[1]*Jpm -2cs[2]*Jpm cs[2]*Jzpm;
            -2cs[2]*Jpm J1xy-2cs[1]*Jpm -cs[1]*Jzpm
            cs[2]*Jzpm -cs[1]*Jzpm J1z
        ]

        params12 = _One2OneHalf1(paramsM,D)
        Jxx = params12.Jxx
        Jyy = params12.Jyy 
        Jxy = params12.Jxy 
        Jyx = params12.Jyx
        Jzz = params12.Jzz

        @show params12

        for pair in bonds[i]
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),Jxx,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),Jyy,nothing)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),Jzz,nothing)
            addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),Jxy,nothing)
            addIntr!(Root,LocalSpace.SySx,pair,("Sy","Sx"),Jyx,nothing)
        end
    end

    for pair in neighbor(Latt;level = 3)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),-J3^2/4D,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),-J3^2/4D,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J3 + J3^2/4D,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",-H,nothing)
    end

    Lx,Ly = get_cellsize(Latt)

    h = get(kwargs,:h, 0.)
    pinvec = get(kwargs,:pinvec, repeat([[0.,0.,1.],],4Ly))
    pinh = get(kwargs,:pinh, map(x -> x * h,pinvec) )
    pinsites = get(kwargs,:pinsites,vcat(1:2Ly,size(Latt)-2Ly-1:size(Latt)))

    for (i,hv) in enumerate(pinh)
        addIntr!(Root,LocalSpace.Sx,pinsites[i],"Sx",-hv[1],nothing)
        addIntr!(Root,LocalSpace.Sy,pinsites[i],"Sy",-hv[2],nothing)
        addIntr!(Root,LocalSpace.Sz,pinsites[i],"Sz",-hv[3],nothing)
    end

    return AutomataSparseMPO(InteractionTree(Root),size(Latt))  
    
end

