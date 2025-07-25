

# struct ObserableTree{N} <: AbstractObservableForest
#     Root::ObservableTreeNode
#     function ObserableTree{N}() where N
         
#          Root = ObservableTreeNode(nothing)
#          for i in 1:N
#               addchild!(Root, ObservableTreeNode(IdentityOperator(0)))
#          end
#          return new{N}(Root)
#     end

#     ObserableTree() = ObserableTree{0}()
# end

# struct ObserableForest{N} <: AbstractObservableForest
#     Roots::ObservableTreeNode
#     function ObserableForest{N}() where N
         
#          Root = ObservableTreeNode(nothing)
#          for i in 1:N
#               addchild!(Root, ObservableTreeNode(IdentityOperator(0)))
#          end
#          return new{N}(Root)
#     end

#     ObserableForest() = ObserableForest{0}()
# end

# mutable struct ObserableForest <: AbstractObservableForest
#      Roots::Dict{String,ObservableTreeNode}
#      width::Int64
#      function ObserableForest(A::Vector{Tuple{String,ObservableTreeNode}})
#           roots = Dict{String,ObservableTreeNode}()
#           for a in A 
#                roots[a] = A
#           end
#           return new(roots,length(A))
#      end
 
#      ObserableForest() = new(Dict{String,ObservableTreeNode}(),0)
# end

mutable struct Observable <: AbstractObservable
     node::Union{Nothing, AbstractObservableTreeNode}
     values::Union{Nothing, AbstractDict}
     L::Union{Nothing, Int64}
 
     function Observable(node::AbstractObservableTreeNode,
         L::Int64)
         return new(node, nothing, L) 
     end
 
     function Observable(node::AbstractObservableTreeNode)
         return new(node, nothing, treeheight(node) - 1) 
     end
 
     Observable() = new(ObservableTreeNode(), nothing, 0)
end


update!(obj::Observable) = obj.L = treeheight(obj.node) - 1

