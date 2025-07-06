

# function SU2M2(Latt::AbstractLattice)
#     LocalSpace = SU₂Spin

#     Root = InteractionTreeNode()

#     for i in 1:size(Latt),j in i+1:size(Latt)
#         addIntr!(Root,LocalSpace.SS,(i,j),("S","S"),2,nothing)
#     end

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.S2,i,"S2",1,nothing)
#     end
    
#     return AutomataSparseMPO(InteractionTree(Root),size(Latt))
# end

# function U1M2(Latt::AbstractLattice)
#     LocalSpace = U₁Spin

#     Root = InteractionTreeNode()

#     for i in 1:size(Latt),j in i+1:size(Latt)
#         addIntr!(Root,LocalSpace.SzSz,(i,j),("Sz","Sz"),2,nothing)
#         addIntr!(Root,LocalSpace.S₊S₋,(i,j),("S₊","S₋"),1,nothing)
#         addIntr!(Root,LocalSpace.S₋S₊,(i,j),("S₋","S₊"),1,nothing)
#     end

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.S2,i,"S2",1,nothing)
#     end
    
#     return AutomataSparseMPO(InteractionTree(Root),size(Latt))
# end

# function U1Mz(Latt::AbstractLattice)
#     LocalSpace = U₁Spin

#     Root = InteractionTreeNode()

#     # for i in 1:size(Latt),j in i+1:size(Latt)
#     #     addIntr!(Root,LocalSpace.SzSz,(i,j),("Sz","Sz"),2,nothing)
#     #     addIntr!(Root,LocalSpace.S₊S₋,(i,j),("S₊","S₋"),1,nothing)
#     #     addIntr!(Root,LocalSpace.S₋S₊,(i,j),("S₋","S₊"),1,nothing)
#     # end

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.Sz,i,"Sz",1,nothing)
#     end
    
#     return AutomataSparseMPO(InteractionTree(Root),size(Latt))
# end

# function U1Mz2(Latt::AbstractLattice)
#     LocalSpace = U₁Spin

#     Root = InteractionTreeNode()

#     for i in 1:size(Latt),j in i+1:size(Latt)
#         addIntr!(Root,LocalSpace.SzSz,(i,j),("Sz","Sz"),2,nothing)
#     end

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.Sz2,i,"Sz2",1,nothing)
#     end
    
#     return AutomataSparseMPO(InteractionTree(Root),size(Latt))
# end

# function TrivialM2(Latt::AbstractLattice)
#     LocalSpace = TrivialSpinOneHalf

#     Root = InteractionTreeNode()

#     for i in 1:size(Latt),j in i+1:size(Latt)
#         addIntr!(Root,LocalSpace.SxSx,(i,j),("Sx","Sx"),2,nothing)
#         addIntr!(Root,LocalSpace.SySy,(i,j),("Sy","Sy"),2,nothing)
#         addIntr!(Root,LocalSpace.SzSz,(i,j),("Sz","Sz"),2,nothing)    
#     end

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.S2,i,"S2",1,nothing)
#     end
    
#     return AutomataSparseMPO(InteractionTree(Root),size(Latt))
# end

# function TrivialMz(Latt::AbstractLattice)
#     LocalSpace = TrivialSpinOneHalf

#     Root = InteractionTreeNode()

#     # for i in 1:size(Latt),j in i+1:size(Latt)
#     #     addIntr!(Root,LocalSpace.SxSx,(i,j),("Sx","Sx"),2,nothing)
#     #     addIntr!(Root,LocalSpace.SySy,(i,j),("Sy","Sy"),2,nothing)
#     #     addIntr!(Root,LocalSpace.SzSz,(i,j),("Sz","Sz"),2,nothing)    
#     # end

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.Sz,i,"Sz",1,nothing)
#     end
    
#     return AutomataSparseMPO(InteractionTree(Root),size(Latt))
# end

# function TrivialMz2(Latt::AbstractLattice)
#     LocalSpace = TrivialSpinOneHalf

#     Root = InteractionTreeNode()

#     for i in 1:size(Latt),j in i+1:size(Latt)
#         addIntr!(Root,LocalSpace.SzSz,(i,j),("Sz","Sz"),2,nothing)    
#     end

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.Sz2,i,"Sz2",1,nothing)
#     end
    
#     return AutomataSparseMPO(InteractionTree(Root),size(Latt))
# end


function PINVEC120(Latt,h)
    Lx,Ly = get_cellsize(Latt) 
    A0 = [0.,1.,0.] * h
    R = [-1/2 -sqrt(3)/2 0.;sqrt(3)/2 -1/2 0.;0. 0. 1.]
    return vcat(repeat([A0,R^2*A0],div(Ly,2)),repeat([R^(mod(Lx-1,3))*A0,R^(mod(Lx+1,3))*A0],div(Ly,2)))
end

function TrivialHamiltonian(Latt::AbstractLattice;
    J::Number=1,H::Number = 0,
    pinh::Vector = repeat([zeros(3),],2*get_cellsize(Latt)[2]),
    pinsites::Vector = vcat(1:get_cellsize(Latt)[2], size(Latt)-get_cellsize(Latt)[2]+1:size(Latt))
    )

    LocalSpace = TrivialSpinOneHalf
  
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

    return AutomataSparseMPO(InteractionTree(Root),size(Latt))  
    
end

function U1Hamiltonian(Latt::AbstractLattice;Jz::Number=1, Jxy::Number=1/2, h::Number = 0,H::Number = 0)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = U₁Spin
    
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),Jz,nothing)
            addIntr!(Root,LocalSpace.S₊S₋,pair,("S₊","S₋"),Jxy,nothing)
            addIntr!(Root,LocalSpace.S₋S₊,pair,("S₋","S₊"),Jxy,nothing)
        end

        # if H != 0
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.Sz,i,"Sz",-H,nothing)
        end
        # end

        # addIntr!(Root,LocalSpace.Sz,div(size(Latt),2),"Sz",h,nothing)

        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end
    return H
end

function SU2Hamiltonian(Latt::AbstractLattice;J::Number=1)
    LocalSpace = SU₂Spin

    Root = InteractionTreeNode()

    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.SS,pair,("S","S"),J,nothing)
    end
    
    return AutomataSparseMPO(InteractionTree(Root),size(Latt))
end
