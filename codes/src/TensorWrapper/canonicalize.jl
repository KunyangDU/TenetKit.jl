
function canonicalize!(obj::Union{DenseMPO{L},DenseMPS{L}},sl::Int64,sr::Int64) where {L}
    @assert 1 ≤ sl ≤ sr ≤ L 

    for sli in obj.center[1]:sl-1
        leftorth!(obj,sli)
        # @show space.(obj.ts[sli:sli+1])
        obj.center[1] += 1
        ( obj.center[1] > obj.center[2] ) && ( obj.center[2] += 1 )
    end
    for sri in obj.center[2]:-1:sr+1
        rightorth!(obj,sri)
        # @show space.(obj.ts[sri-1:sri])
        obj.center[2] -= 1
        ( obj.center[1] > obj.center[2] ) && ( obj.center[1] -= 1 )
    end
    return obj
end

function canonicalize!(::SparseMPO, ::Int64) end

function canonicalize!(obj::Union{DenseMPO{L},DenseMPS{L}},si::Int64) where {L}
    @assert 1 ≤ si ≤ L 
    return canonicalize!(obj,si,si)
end

function canonicalize!(obj::Union{AdjointMPO{L},AdjointMPS{L}},si::Int64) where {L}
    @assert 1 ≤ si ≤ L 
    return adjoint(canonicalize!(obj',si,si))
end

