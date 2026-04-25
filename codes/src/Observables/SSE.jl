
function traceback(A::InteractionTreeNode)
    root = InteractionTreeNode()
    for l in Leaves(A)
        addIntr!(root,l.Leave)
    end
    return cutparent!(root.children[1])
end

function TimerOutputs.merge!(A::AbstractTreeNode, B::AbstractTreeNode, value::Function = nodevalue)
    ind = findfirst(x -> isequal(value(x), value(B)), A.children)
    if !isnothing(ind)
        for c in B.children
            merge!(A.children[ind], c)
        end
    else
        B.parent = nothing
        addchild!(A,B)
        return A
    end
end

function TimerOutputs.merge!(A::ObservableTreeLeave{N}, B::InteractionTreeLeave, i::Int64, N′::Int64) where N 
    @assert N == N′
    A.site[i] = B.site
    A.name[i] = B.name
    return A
end

TimerOutputs.merge!(::Nothing, B::InteractionTreeLeave, i::Int64, N′::Int64) = merge!(ObservableTreeLeave(B,i,N′),B,i,N′)
function TimerOutputs.merge!(A::ObservableTreeLeave{N}, ::Nothing, ::Int64, N′::Int64) where N
    @assert N == N′
    return A
end
TimerOutputs.merge!(::Nothing, ::Nothing, ::Int64, ::Int64) = nothing

function Tuple!(A::ObservableTreeLeave)
    A.site = Tuple(Tuple.(A.site))
    A.name = Tuple(Tuple.(A.name))
    return A
end


function relevent_node(root::InteractionTreeNode, site::Int64, dismiss::Function = (x -> false))
    A = []
    for l in Leaves(root)
        site ∉ l.Leave.site && continue
        dismiss(l.Leave.A[findfirst(x -> x == site, l.Leave.site)]) && continue
        push!(A,l.Leave)
    end
    relroot = InteractionTreeNode()

    for a in A
        addIntr!(relroot, a)
    end

    return relroot
end


function commutate_node(root::InteractionTreeNode, obj::LocalOperator, commute::Function = x -> x*obj.A - obj.A*x)
    A = []

    for l in Leaves(root)
        obj.site ∉ l.Leave.site && continue
        i = findfirst(x -> x == obj.site, l.Leave.site)
        cA = commute(l.Leave.A[i])
        norm(cA) ≈ 0 && continue
        tA = collect(l.Leave.A)
        tname = collect(l.Leave.name)
        tA[i] = cA
        tname[i] = "[$(l.Leave.name[i]),$(obj.name)]"
        tleave = deepcopy(l.Leave)
        tleave.A = Tuple(tA)
        tleave.name = Tuple(tname)
        push!(A,tleave)
    end
    relroot = InteractionTreeNode()

    for a in A
        addIntr!(relroot, a)
    end

    return relroot
end

function _onsite_replace!(root::AbstractTreeNode, site::Int64, node_replace!::Function)
    for p in PreOrderDFS(root)
        if typeof(p.A) <: LocalOperator && p.A.site == site
            node_replace!(p.A)
        end
    # for l in Leaves(root)
    #     site ∉ l.Leave.site && continue 
    #     node_replace!
    end

    return root
end

function commutate(O::AbstractTreeNode, obj::LocalOperator)
    # root = relevent_node(O, obj.site, x -> norm(x.A*obj.A - obj.A*x.A) ≈ 0)
    # root = relevent_node(O, obj.site, x -> norm(x*obj.A - obj.A*x) ≈ 0)
    # node_replace!(x) = let 
    #     x.A = x.A*obj.A - obj.A*x.A 
    #     x.name = "[$(x.name),$(obj.name)]"
    # end
    # return _onsite_replace!(root, obj.site, node_replace!)
    return commutate_node(O,obj)
end


trivial(x::T) where T<: AbstractTreeNode = T(trivial(nodevalue(x)))
function trivialchild(x::InteractionTreeNode)
    y = deepcopy(trivial(nodevalue(x)))
    y.site += 1
    return InteractionTreeNode(y)
end

function buildtree!(A::CompositeObservableTreeNode)
    if !reduce(&, map(x ->  isempty(x.children), A.A))
        for a in Iterators.product(map(x -> isempty(x.children) ? [trivialchild(x),] : x.children, A.A)...)
            x = CompositeObservableTreeNode(a)
            leaves = map(x -> x.Leave, a)
            Aleave = deepcopy(A.Leave)
            
            for (i,l) in enumerate(leaves)
                Aleave = merge!(Aleave,l,i,length(a))
            end
            x.Leave = Aleave
            
            addchild!(A, x)
        end
        A.Leave = nothing
    else
        Tuple!(A.Leave)
    end

    # A.A = map(x -> isnothing(x) ? nothing : nodevalue(x), A.A)
    A.A = map(x -> nodevalue(x), A.A)

    for c in A.children
        buildtree!(c)
    end

    return A
end

function pushright(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    # opt=true: greedy optimizer contracts small (d²) local operators and EnvL (D²) before
    # ever pairing objt with objb.  Without it the default left-to-right order reaches
    # objt×objb — which share only d-sized bonds — producing a D⁴ intermediate
    # (~1.6 GB at D=100).  opt=true keeps every intermediate at O(d²D²).
    # @tensor opt=true x[-1;-2] ≔ ht.A[2,6] * objt.A[1,3,-2,2] * hb.A[5,1] * objb.A[-1,6,5,4] * EnvL.A[4,3]
    @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[2,3,-2,1] * hb.A[6,2] * objb.A[-1,5,6,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function pushright(::IdentityOperator{1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    # @tensor opt=true x[-1;-2] ≔ objt.A[1,3,-2,2] * hb.A[5,1] * objb.A[-1,2,5,4] * EnvL.A[4,3]
    @tensor x[-1;-2] ≔ objt.A[2,3,-2,5] * hb.A[6,2] * objb.A[-1,5,6,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function pushright(::IdentityOperator{1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    # Even with only 3 tensors, the default order contracts objt×objb first (D⁴).
    # @tensor opt=true x[-1;-2] ≔ objt.A[1,3,-2,2] * objb.A[-1,2,1,4] * EnvL.A[4,3]
    @tensor x[-1;-2] ≔  objt.A[6,3,-2,5] * objb.A[-1,5,6,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function pushright(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    # @tensor opt=true x[-1;-2] ≔ ht.A[2,5] * objt.A[1,3,-2,2] * objb.A[-1,5,1,4] * EnvL.A[4,3]
    @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[6,3,-2,1] * objb.A[-1,5,6,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function SSE1(H::InteractionTreeNode)
    @time "SSE tree built spending" lsroot = let
        lsroot = CompositeObservableTreeNode[]
        
        node_replace!(x,obj) = let 
            x.A = x.A*obj.A - obj.A*x.A 
            x.name = "[$(x.name),$(obj.name)]"
        end

        for i in 1:size(Latt)
            rootup = InteractionTreeNode()
            addIntr!(rootup,TrivialSpinOneHalf.S₊,i,"S₊",false,1,nothing)
            S₋ = LocalOperator(TrivialSpinOneHalf.S₋,"S₋",i,1)
            rootdown = commutate(H,S₋)
            tmpr = CompositeObservableTreeNode((rootup,rootdown))
            buildtree!(tmpr)
            push!(lsroot,cutparent!(tmpr))
        end
        lsroot
    end 
    map(lsroot[2:end]) do root 
        merge!!(lsroot[1],(root))
    end
    return Observable(lsroot[1])
end
