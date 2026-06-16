
function Base.:*(objl::InteractionTunnel{L,T},objr::InteractionTunnel{L,T}) where {L,T,W}
    @assert (Z = objl.Z) == objr.Z
    
    Al,Ar = objl.A, objr.A
    ferml,fermr = objl.fermionic, objr.fermionic

    A = T[]
    fermionic = Bool[]
    strength = objl.strength * objr.strength
    
    k = 0
    for j in eachindex(Ar)
        r,rf= Ar[j],fermr[j]
        for i in reverse(1:length(Al))
            l,lf = Al[i],ferml[i]
            if l.site > r.site
                (lf && rf) && (strength *= -1)
                if i == k+1
                    push!(A, r)
                    push!(fermionic, rf)
                end
                continue
            else
                if l.site == r.site
                    for s in k+1:i-1
                        push!(A, Al[s])
                        push!(fermionic, ferml[s])
                    end
                    push!(A, l * r)
                    push!(fermionic, lf ⊻ rf)
                else                
                    for s in k+1:i
                        push!(A, Al[s])
                        push!(fermionic, ferml[s])
                    end
                    push!(A, r)
                    push!(fermionic, rf)
                end
                k = i
            end
            break
        end
    end
    for s in k+1:length(Al)
        push!(A, Al[s])
        push!(fermionic, ferml[s])
    end
    return InteractionTunnel{L,T}(Tuple(A),Tuple(fermionic),Z,strength)
end

function Base.:+(objl::InteractionTunnel{L,T,N₁},objr::InteractionTunnel{L,T,N₂}) where {L,T,N₁,N₂}
    N₁ ≠ N₂ && return (objl,objr)
    _site(objl) ≠ _site(objr) && return (objl,objr)
    objl.fermionic ≠ objr.fermionic && return (objl,objr)
    isdiff = false

    @assert (Z = objl.Z) == objr.Z
    A = T[]
    fermionic = Bool[]

    for i in 1:N₁
        if isequal(objl.A[i], objr.A[i])
            push!(A, objl.A[i])
            push!(fermionic, objl.fermionic[i])
        else
            isdiff && return (objl,objr)
            A′ = objl.strength * objl.A[i] + objr.strength * objr.A[i]
            norm(A′) ≈ 0 && return InteractionTunnel{L,T,N₁}[]
            push!(A, A′)
            push!(fermionic, objl.fermionic[i])
            isdiff = true
        end
    end
    if !isdiff
        return objl.strength + objr.strength ≈ 0 ? InteractionTunnel{L,T,N₁}[] : InteractionTunnel{L,T}(Tuple(A),Tuple(fermionic),Z, objl.strength + objr.strength)
    else
        return InteractionTunnel{L,T}(Tuple(A),Tuple(fermionic),Z, 1.0)
    end
end

function Base.:-(objl::InteractionTunnel{L,T,N₁},objr::InteractionTunnel{L,T,N₂}) where {L,T,N₁,N₂}
    N₁ ≠ N₂ && return (objl,objr)
    _site(objl) ≠ _site(objr) && return (objl,objr)
    objl.fermionic ≠ objr.fermionic && return (objl,objr)
    isdiff = false

    @assert (Z = objl.Z) == objr.Z
    A = T[]
    fermionic = Bool[]

    for i in 1:N₁
        if isequal(objl.A[i], objr.A[i])
            push!(A, objl.A[i])
            push!(fermionic, objl.fermionic[i])
        else
            isdiff && return (objl,objr)
            A′ = objl.strength * objl.A[i] - objr.strength * objr.A[i]
            norm(A′) ≈ 0 && return InteractionTunnel{L,T,N₁}[]
            push!(A, A′)
            push!(fermionic, objl.fermionic[i])
            isdiff = true
        end
    end
    if !isdiff
        return objl.strength - objr.strength ≈ 0 ? InteractionTunnel{L,T,N₁}[] : InteractionTunnel{L,T}(Tuple(A),Tuple(fermionic),Z, objl.strength - objr.strength)
    else
        return InteractionTunnel{L,T}(Tuple(A),Tuple(fermionic),Z, 1.0)
    end
end

function Base.:+(A::InteractionGraph{L,T,W}, B::InteractionGraph{L,T,W}) where {L,T,W}
    skiplist = Int64[]
    tunnel = InteractionTunnel{L, T}[]
    for b in B.tunnel
        ispushed = false
        for (i,a) in enumerate(A.tunnel)
            i in skiplist && continue 
            (apb = a + b) isa Tuple && continue
            push!(skiplist,i)
            ispushed = true
            apb isa Vector && continue
            push!(tunnel, apb)
            break
        end
        ispushed && continue
        push!(tunnel, deepcopy(b))
    end
    map(x -> (x[1] ∉ skiplist && push!(tunnel,deepcopy(x[2]))), enumerate(A.tunnel))
    return InteractionGraph(tunnel)
end

function Base.:-(A::InteractionGraph{L,T,W}, B::InteractionGraph{L,T,W}) where {L,T,W}
    skiplist = Int64[]
    tunnel = InteractionTunnel{L, T}[]
    for b in B.tunnel
        ispushed = false
        for (i,a) in enumerate(A.tunnel)
            i in skiplist && continue 
            (apb = a - b) isa Tuple && continue
            push!(skiplist,i)
            ispushed = true
            apb isa Vector && continue
            push!(tunnel, apb)
            break
        end
        ispushed && continue
        push!(tunnel, -1 * b)
    end
    map(x -> (x[1] ∉ skiplist && push!(tunnel,deepcopy(x[2]))), enumerate(A.tunnel))
    return InteractionGraph(tunnel)
end

function Base.:*(A::InteractionGraph{L,T,W}, B::InteractionGraph{L,T,W}) where {L,T,W}
    tunnel = InteractionTunnel{L, T}[]
    for b in B.tunnel
        push!(tunnel, (A * b)...)
    end
    return InteractionGraph(tunnel)
end

function Base.:*(A::InteractionGraph{L,T,W}, b::InteractionTunnel{L,T}) where {L,T,W}
    tunnel = InteractionTunnel{L, T}[]
    for a in A.tunnel
        push!(tunnel, a * b)
    end
    return InteractionGraph(tunnel)
end

function Base.:*(b::InteractionTunnel{L,T}, A::InteractionGraph{L,T,W}) where {L,T,W}
    tunnel = InteractionTunnel{L, T}[]
    for a in A.tunnel
        push!(tunnel, b * a)
    end
    return InteractionGraph(tunnel)
end

commutate(H::InteractionGraph, O::InteractionTunnel) = InteractionGraph(filter(x -> !isempty(intersect(_site(x),_site(O))), H.tunnel)) |> x -> (x * O - O * x)
commutate(O::InteractionTunnel, H::InteractionGraph) = InteractionGraph(filter(x -> !isempty(intersect(_site(x),_site(O))), H.tunnel)) |> x -> (O * x - x * O)

