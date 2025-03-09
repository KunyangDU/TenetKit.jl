function addIntr!(Root::InteractionTreeNode,
    Opri::AbstractTensorMap,
    site::Int64,
    name::String,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength == 0 && return nothing
    addIntr1!(Root,Opri,site,name,strength,Z)
end

function addIntr!(Root::InteractionTreeNode,
    Opri::NTuple{1,AbstractTensorMap},
    site::NTuple{1,Int64},
    name::NTuple{1,String},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength == 0 && return nothing
    addIntr1!(Root,Opri[1],site[1],name[1],strength,Z)
end

function addIntr!(Root::InteractionTreeNode,
    Opri::NTuple{2,AbstractTensorMap},
    site::NTuple{2,Int64},
    name::NTuple{2,String},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength == 0 && return nothing
    addIntr2!(Root,Opri,site,name,strength,Z)
end

function addIntr!(Tree::InteractionTree,
    Opri::AbstractTensorMap,
    site::Int64,
    name::String,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength == 0 && return nothing
    addIntr1!(Tree.Root.children[1],Opri,site,name,strength,Z)
end

function addIntr!(Tree::InteractionTree,
    Opri::NTuple{1,AbstractTensorMap},
    site::NTuple{1,Int64},
    name::NTuple{1,String},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength == 0 && return nothing
    addIntr1!(Tree.Root.children[1],Opri[1],site[1],name[1],strength,Z)
end

function addIntr!(Tree::InteractionTree,
    Opri::NTuple{2,AbstractTensorMap},
    site::NTuple{2,Int64},
    name::NTuple{2,String},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength == 0 && return nothing
    addIntr2!(Tree.Root.children[1],Opri,site,name,strength,Z)
end

############# k ####################

function addIntr!(Root::InteractionTreeNode,
    Opri::AbstractTensorMap,
    Latt::AbstractLattice,k::Vector,
    name::String,
    strength::Number,
    string::Union{Nothing,AbstractTensorMap})
    strength == 0 && return nothing
    L = size(Latt)
    for site in 1:L
        addIntr1!(Root,Opri,site,name,strength*exp(-1im*dot(k,coordinate(Latt,site))) / sqrt(L),string)
    end
end

function addIntr!(Tree::InteractionTree,
    Opri::AbstractTensorMap,
    Latt::AbstractLattice,k::Vector,
    name::String,
    strength::Number,
    string::Union{Nothing,AbstractTensorMap})
    strength == 0 && return nothing
    addIntr!(Tree.Root.children[1],Opri,Latt,k,name,strength,string)
end

# c⁺ = exp(-1im)* ... 
function addIntr!(Root::InteractionTreeNode,
    Opri::Tuple,
    Latt::AbstractLattice,k::Vector,
    name::Tuple,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    L = size(Latt)
    for i in 1:L, j in i+1:L, ind in 1:2 
        addIntr2!(Root,Opri[ind],(i,j),name[ind],(-1)^(ind-1)*strength*exp((-1)^ind*1im*dot(k, coordinate(Latt,i) .- coordinate(Latt,j))) / L,Z)
    end
end
