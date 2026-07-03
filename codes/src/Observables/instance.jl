function SSE1(Latt::AbstractLattice, H::InteractionGraph, S₊::AbstractTensorMap, S₋::AbstractTensorMap)
    L = size(Latt)
    ig = InteractionGraph(L,CompositeLocalOperator{2},ObservableWeight)
    for i in 1:L
        SSEup = ObservableGraph(L)
        addObs!(SSEup,S₊,i,"S₊",false,nothing,1.0)
        SSEdown = ObservableGraph(L)
        S₋t = InteractionTunnel(S₋,i,"S₋",false,1.0,nothing,L,LocalOperator)
        addIntr!(SSEdown,commutate(H,S₋t).tunnel)
        addIntr!(ig,composite(SSEup,SSEdown).tunnel)
    end
    initialize!(ig)
    return ig
end

function SSE2(Latt::AbstractLattice, H::InteractionGraph, S₊::AbstractTensorMap, S₋::AbstractTensorMap)
    L = size(Latt)
    ig = InteractionGraph(L,CompositeLocalOperator{2},ObservableWeight)
    SSEup = ObservableGraph(L)
    SSEdown = ObservableGraph(L)
    for i in 1:L
        addObs!(SSEup,S₊,i,"S₊",false,nothing,1.0)
        S₋t = InteractionTunnel(S₋,i,"S₋",false,1.0,nothing,L,LocalOperator)
        addIntr!(SSEdown,commutate(H,S₋t).tunnel)
    end
    addIntr!(ig,composite(SSEup,SSEdown).tunnel)
    initialize!(ig)
    return ig
end
