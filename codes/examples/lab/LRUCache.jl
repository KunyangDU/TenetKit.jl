using LRUCache,TensorKit

A = TensorMap(randn,ℂ^4,ℂ^4)
Base.summarysize(A)
