function addIntr2(inds::NTuple{2,Int64},Ops::Tuple,L::Int64,strength::Number = 1)
    id = diagm(ones(size(Ops[1])[1]))
    idl,idr = inds
    tot = idl == 1 ? Ops[1] : id 
    for i in 2:L
        if i == idl
            tot = kron(tot,Ops[1])
        elseif i == idr
            tot = kron(tot,Ops[2])
        else
            tot = kron(tot,id)
        end
    end
    return tot * strength
end