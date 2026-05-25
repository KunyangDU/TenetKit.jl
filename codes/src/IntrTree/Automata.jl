
function AutomataSparseMPO(Root::InteractionTreeNode,L::Int64=treeheight(Root) - 1)
    MPO = let 
        tempMPO = Vector{SparseMPOTensor}(undef,L)

        # idtensor = getIdTensor(Root.children[1].children[1].A)
        # idtensor = getIdTensor(Root.children[1].A)
        idtensor = getIdTensor(first(Leaves(Root)).A)

        lastnode = Dict(
            "leaves" => [],
            "roots" => [],
            "inverse_root" => 0,
        )
        nextnode = Dict(
            "leaves" => [],
            "roots" => [],
            "inverse_root" => 0,
        )
        lastnode["roots"] = [Root,]
        
        for iL in 1:L

            nextnode["leaves"] = []
            nextnode["roots"] = []
            nextnode["leaves_inds"] = []
            nextnode["roots_inds"] = []

            if nextnode["inverse_root"] == 0 && !isempty(findall(x -> isempty(x.children),vcat([lastroot.children for lastroot in lastnode["roots"]]...)))
                nextnode["inverse_root"] = 1
            end

            for (lastind,last_root) in enumerate(lastnode["roots"])
                for next_subtree in last_root.children
                    if isempty(next_subtree.children)
                        push!(nextnode["leaves"],next_subtree)
                        push!(nextnode["leaves_inds"],((lastind + lastnode["inverse_root"], nextnode["inverse_root"]), length(nextnode["leaves"])))
                    else
                        push!(nextnode["roots"],next_subtree)
                        push!(nextnode["roots_inds"],((lastind + lastnode["inverse_root"], length(nextnode["roots"]) + nextnode["inverse_root"]), length(nextnode["roots"])))
                    end
                end
            end

            localMPOdims = length.((lastnode["roots"], nextnode["roots"])) .+ (lastnode["inverse_root"], nextnode["inverse_root"])
            localMPO = SparseMPOTensor(nothing,localMPOdims...)

            map([("leaves_inds","leaves"),("roots_inds","roots")]) do (x,y)
                for inds in nextnode[x]
                    A = nextnode[y][inds[2]].A
                    !isnan(A.strength) && (A.A *= A.strength)
                    localMPO.m[inds[1]...] = axpy!(1, A , localMPO.m[inds[1]...])
                    # localMPO.m[inds[1]...] += nextnode[y][inds[2]].A
                    # DenseMPOTensor(let 
                    #     localOpr = nextnode[y][inds[2]].A.A
                    #     strength = nextnode[y][inds[2]].A.strength
                    #     if isnan(strength)
                    #         localOpr
                    #     else
                    #         localOpr*strength
                    #     end
                    # end)
                end
            end

            if isnothing(localMPO.m[1,1]) && lastnode["inverse_root"] == 1
                # lastnode["inverse_root"] == 1
                # localMPO.m[1,1] = DenseMPOTensor(lastnode["inverse_root"]*idtensor)
                localMPO.m[1,1] = IdentityOperator(idtensor, iL)
            end
            
            lastnode["leaves"] = nextnode["leaves"]
            lastnode["roots"] = nextnode["roots"]
            lastnode["inverse_root"] = nextnode["inverse_root"]

            tempMPO[iL] = localMPO
        end

        SparseMPO(tempMPO)
    end

    return MPO
end

# function AutomataSparseMPO(Tree::InteractionTree,L::Int64 = treeheight(Tree.Root) - 1)
#     return AutomataSparseMPO(Tree.Root,L)
# end







