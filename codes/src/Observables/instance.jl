function SSE1(Latt::AbstractLattice, H::InteractionGraph, S₊::AbstractTensorMap, S₋::AbstractTensorMap)
    L = size(Latt)
    ig = InteractionGraph(L,CompositeObservableOperator{2})
    for i in 1:L
        SSEup = ObservableGraph(L)
        addObs!(SSEup,S₊,i,"S₊",false,nothing,1.0)
        SSEdown = InteractionGraph(L)
        S₋t = InteractionTunnel(S₋,i,"S₋",false,1.0,nothing,L,LocalOperator)
        addIntr!(SSEdown,commutate(H,S₋t).tunnel)
        SSEdown = observe(SSEdown)
        addIntr!(ig,composite(SSEup,SSEdown).tunnel)
    end
    initialize!(ig)
    return ig
end
