

struct ObserableTree{N} <: AbstractObservableForest
    Root::InteractionTreeNode
    function ObserableTree{N}() where N
         
         Root = InteractionTreeNode(nothing)
         for i in 1:N
              addchild!(Root, InteractionTreeNode(IdentityOperator(0)))
         end
         return new{N}(Root)
    end

    ObserableTree() = ObserableTree{0}()
end

struct ObserableForest{N} <: AbstractObservableForest
    Roots::InteractionTreeNode
    function ObserableForest{N}() where N
         
         Root = InteractionTreeNode(nothing)
         for i in 1:N
              addchild!(Root, InteractionTreeNode(IdentityOperator(0)))
         end
         return new{N}(Root)
    end

    ObserableForest() = ObserableForest{0}()
end

mutable struct MPSObservable <: AbstractObservable
     forest::Union{Nothing, AbstractObservableForest}
     values::Union{Nothing, AbstractDict}
     L::Union{Nothing, Int64}
 
     function MPSObservable(forest::AbstractObservableForest,
         L::Int64)
         return new(forest, L) 
     end
 
     function MPSObservable(forest::AbstractObservableForest)
         return new(forest, treeheight(obj.forest.Roots) - 2) 
     end
 
     MPSObservable() = new(ObserableForest(), nothing)
end


update!(obj::MPSObservable) = obj.L = treeheight(obj.forest.Roots) - 2

