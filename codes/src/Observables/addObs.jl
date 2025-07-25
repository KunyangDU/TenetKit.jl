function addObs!(node::AbstractObservableTreeNode,
    Opri::Union{AbstractTensorMap,NTuple{2,AbstractTensorMap}},
    site::Union{Int64,NTuple{2,Int64}},
    name::Union{String,NTuple{2,String}},
    fermionic::Union{Bool,NTuple{2,Bool}},
    Z::Union{Nothing,AbstractTensorMap}
    )
    addIntr!(node, Opri, site, name,fermionic,1,Z)
    return node
end

function _addBranch!(Root::ObservableTreeNode,name::Union{String,Tuple})
    addchild!(Root,ObservableTreeNode(IdentityOperator(0,name)))
end

function addObs!(Obs::Observable,     
    Opri::Union{AbstractTensorMap,NTuple{2,AbstractTensorMap}},
    site::Union{Int64,NTuple{2,Int64}},
    name::Union{String,NTuple{2,String}},
    fermionic::Union{Bool,NTuple{2,Bool}},
    Z::Union{Nothing,AbstractTensorMap}
    )
    addObs!(Obs.node, Opri, site, name,fermionic, Z)
    update!(Obs)
end

function addObs!(Obs::Observable,     
    Opri::AbstractTensorMap,
    site::Int64,
    name::String,
    fermionic::Bool,
    Z::Union{Nothing,AbstractTensorMap}
    )
    addObs!(Obs.node, Opri, site, name,fermionic, Z)
    update!(Obs)
end

function addObs!(Obs::Observable,     
    Opri::NTuple{2,AbstractTensorMap},
    site::NTuple{2,Int64},
    name::NTuple{2,String},
    fermionic::NTuple{2,Bool},
    Z::Union{Nothing,AbstractTensorMap}
    )
    addObs!(Obs.node, Opri, site, name,fermionic, Z)
    update!(Obs)
end

