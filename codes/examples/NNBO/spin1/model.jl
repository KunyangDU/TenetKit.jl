
function TrivialHamiltonian(Latt::AbstractLattice;
    J1xy,J1z,Jpm,Jzpm,
    J3,D,H,
    # gx = 4.8, gy = 4.85, gz = 2.5, μB = 0.05788,
    # Hx = 0, Hy = 0, Hz = 0, 
    # hx = 0, hy = 0, hz = 0,
    kwargs...)

    LocalSpace = TrivialSpinOne

    Root = InteractionTreeNode()
    triavals = [(1,0),(-1/2,sqrt(3)/2),(-1/2,-sqrt(3)/2)]
    direction = [[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]
    bonds = getxyzbonds(Latt;direction=direction)
    for pair in neighbor(Latt;level = 1)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J1xy,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J1xy,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J1z,nothing)

        Jpm_xx = 0
        Jpm_xy = 0
        for j in 1:3
            if pair in bonds[j]
                Jpm_xx = 2 * Jpm * triavals[j][1]
                Jpm_xy = - 2 * Jpm * triavals[j][2]
                break
            end
        end
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),Jpm_xx,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),-Jpm_xx,nothing)
        addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),Jpm_xy,nothing)
        addIntr!(Root,LocalSpace.SySx,pair,("Sy","Sx"),Jpm_xy,nothing)

        Jzpm_xz = 0
        Jzpm_yz = 0
        for j in 1:3
            if pair in bonds[j]
                Jzpm_xz = Jzpm * triavals[j][2]
                Jzpm_yz = - Jzpm * triavals[j][1]
                break
            end
        end
        addIntr!(Root,LocalSpace.SxSz,pair,("Sx","Sz"),Jzpm_xz,nothing)
        addIntr!(Root,LocalSpace.SzSx,pair,("Sz","Sx"),Jzpm_xz,nothing)
        addIntr!(Root,LocalSpace.SySz,pair,("Sy","Sz"),Jzpm_yz,nothing)
        addIntr!(Root,LocalSpace.SzSy,pair,("Sz","Sy"),Jzpm_yz,nothing)
    end

    for pair in neighbor(Latt;level = 3)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J3,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J3,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J3,nothing)
    end

    for i in 1:size(Latt)
        # addIntr!(Root,LocalSpace.Sx,i,"Sx",-μB*gx*Hx,nothing)
        # addIntr!(Root,LocalSpace.Sy,i,"Sy",-μB*gy*Hy,nothing)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",-H,nothing)
        addIntr!(Root,LocalSpace.Sz2,i,"Sz2",D,nothing)
    end

    # cind = div(size(Latt),2)
    # addIntr!(Root,LocalSpace.Sx,cind,"Sx",-hx,nothing)
    # addIntr!(Root,LocalSpace.Sy,cind,"Sy",-hy,nothing)
    # addIntr!(Root,LocalSpace.Sz,cind,"Sz",-hz,nothing)

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





