# Heisenberg 哈密顿量 — 基于 InteractionGraph + SparseMPO
# 与旧 Tree 版接口一致，内部用图结构构建

function PINVEC120(Latt,h)
    Lx,Ly = get_cellsize(Latt)
    A0 = [0.,1.,0.] * h
    R = [-1/2 -sqrt(3)/2 0.;sqrt(3)/2 -1/2 0.;0. 0. 1.]
    return vcat(repeat([A0,R^2*A0],div(Ly,2)),repeat([R^(mod(Lx-1,3))*A0,R^(mod(Lx+1,3))*A0],div(Ly,2)))
end

function YCRect(L::Int64, W::Int64, (a,b)::NTuple{2,Float64} = (1.0,1.0),θ::Real = 0.0)
    e = ((a, 0.0), (0.0, b))
    sites = [(x, y) for x in 1:L for y in 1:W]
    if iszero(θ)
         BC = PeriodicBoundaryCondition((0, W))
    else
         BC = TwistBoundaryCondition((0, W), θ)
    end
    return SquareLattice(e, sites, BC)
end

function TrivialHamiltonian(Latt::AbstractLattice;
    J::Number=1, Hx::Number=0, Hy::Number=0, Hz::Number=0, Δ::Number=1, hxd::Number=0.0,
    hx::Number=0,
    pinh::Vector=repeat([zeros(3),],2*get_cellsize(Latt)[2]),
    pinsites::Vector=vcat(1:get_cellsize(Latt)[2], size(Latt)-get_cellsize(Latt)[2]+1:size(Latt)),
    returnnode::Bool=false,
    )

    L = size(Latt)
    LocalSpace = TrivialSpinOneHalf
    ig = InteractionGraph(L)

    for pair in neighbor(Latt)
        addIntr!(ig, LocalSpace.SxSx, pair, ("Sx","Sx"), (false,false), J, nothing)
        addIntr!(ig, LocalSpace.SySy, pair, ("Sy","Sy"), (false,false), J, nothing)
        addIntr!(ig, LocalSpace.SzSz, pair, ("Sz","Sz"), (false,false), J*Δ, nothing)
    end

    for i in 1:L
        addIntr!(ig, LocalSpace.Sx, i, "Sx", false, -Hx, nothing)
        addIntr!(ig, LocalSpace.Sy, i, "Sy", false, -Hy, nothing)
        addIntr!(ig, LocalSpace.Sz, i, "Sz", false, -Hz, nothing)
    end

    addIntr!(ig, LocalSpace.Sx, 1, "Sx", false, hx, nothing)
    addIntr!(ig, LocalSpace.Sx^2, 1, "Sx2", false, hxd, nothing)

    if sum(abs.(vcat(pinh...))) != 0
        @assert length(pinh) == length(pinsites) "pin field not compatible"
        for (i, site) in enumerate(pinsites)
            addIntr!(ig, LocalSpace.Sx, site, "Sx", false, pinh[i][1], nothing)
            addIntr!(ig, LocalSpace.Sy, site, "Sy", false, pinh[i][2], nothing)
            addIntr!(ig, LocalSpace.Sz, site, "Sz", false, pinh[i][3], nothing)
        end
    end

    if returnnode
        return ig
    else
        return build_sparse_mpo(ig)
    end
end

function U1Hamiltonian(Latt::AbstractLattice; Jz::Number=1, Jxy::Number=1/2, h::Number=0, H::Number=0)
    H = let
        L = size(Latt)
        LocalSpace = U₁Spin
        ig = InteractionGraph(L)

        for pair in neighbor(Latt)
            addIntr!(ig, LocalSpace.SzSz,  pair, ("Sz","Sz"),   (false,false), Jz,  nothing)
            addIntr!(ig, LocalSpace.S₊S₋, pair, ("S₊","S₋"),   (false,false), Jxy, nothing)
            addIntr!(ig, LocalSpace.S₋S₊, pair, ("S₋","S₊"),   (false,false), Jxy, nothing)
        end

        for i in 1:L
            addIntr!(ig, LocalSpace.Sz, i, "Sz", false, -H, nothing)
        end

        build_sparse_mpo(ig)
    end
    return H
end

function SU2Hamiltonian(Latt::AbstractLattice; J::Number=1)
    L = size(Latt)
    LocalSpace = SU₂Spin
    ig = InteractionGraph(L)

    for pair in neighbor(Latt)
        addIntr!(ig, LocalSpace.SS, pair, ("S","S"), (false,false), J, nothing)
    end

    return build_sparse_mpo(ig)
end
