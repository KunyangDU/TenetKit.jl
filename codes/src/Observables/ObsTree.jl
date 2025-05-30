

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

struct ObserableForest <: AbstractObservableForest
     Roots::Dict{String,ObservableTreeNode}
     width::Int64
     function ObserableForest(A::Vector{Tuple{String,ObservableTreeNode}})
          roots = Dict{String,ObservableTreeNode}()
          for a in A 
               roots[a] = A
          end
          return new(roots,length(A))
     end
 
     ObserableForest() = new(Dict{String,ObservableTreeNode}(),0)
 end

mutable struct Observable <: AbstractObservable
     forest::Union{Nothing, AbstractObservableForest}
     values::Union{Nothing, AbstractDict}
     L::Union{Nothing, Int64}
 
     function Observable(forest::AbstractObservableForest,
         L::Int64)
         return new(forest, L) 
     end
 
     function Observable(forest::AbstractObservableForest)
         return new(forest, treeheight(obj.forest.Roots) - 2) 
     end
 
     Observable() = new(ObserableForest(), nothing)
end


update!(obj::Observable) = obj.L = treeheight(obj.forest.Roots) - 2

