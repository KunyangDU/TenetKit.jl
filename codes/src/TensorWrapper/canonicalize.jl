
function canonicalize!(obj::T,sl::Int64,sr::Int64) where T <: Union{DenseMPO{L},DenseMPS{L},AdjointMPO{L},AdjointMPS{L}} where L
    @assert 1 ≤ sl ≤ sr ≤ L 

    for sli in obj.center[1]:sl-1
        leftorth!(obj,sli)
        # @show space.(obj[sli:sli+1])
        obj.center[1] += 1
        ( obj.center[1] > obj.center[2] ) && ( obj.center[2] += 1 )
    end
    for sri in obj.center[2]:-1:sr+1
        rightorth!(obj,sri)
        # @show space.(obj[sri-1:sri])
        obj.center[2] -= 1
        ( obj.center[1] > obj.center[2] ) && ( obj.center[1] -= 1 )
    end
    return obj
end

function canonicalize!(obj::T,si::Int64) where T <: Union{DenseMPO{L},DenseMPS{L},AdjointMPO{L},AdjointMPS{L}} where L
    @assert 1 ≤ si ≤ L 
    return canonicalize!(obj,si,si)
end

canonicalize!(::SparseMPO, ::Int64) = nothing
canonicalize!(obj::RefMPO, i::Int64) = canonicalize!(obj.pointer,i)
