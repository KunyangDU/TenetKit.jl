
# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe LIFO stack for DFS-ordered work scheduling.
#
# Why LIFO instead of the original Channel (FIFO)?
# A FIFO ch_swap causes BFS traversal: all siblings at depth k are scheduled
# before any node at depth k+1. This keeps O(L) distinct parent-Env tensors
# alive simultaneously (one per tree level), totalling O(N·L·D²) memory with
# N workers. LIFO (DFS) processes the deepest available sibling first, so a
# parent Env is freed as soon as its subtree is finished — dramatically
# reducing the peak number of live Env tensors.
# ─────────────────────────────────────────────────────────────────────────────
mutable struct LIFOStack{T}
    data::Vector{T}
    lock::ReentrantLock
    LIFOStack{T}() where T = new{T}(Vector{T}(), ReentrantLock())
end

function Base.push!(s::LIFOStack{T}, x::T) where T
    lock(s.lock) do
        push!(s.data, x)
    end
end

# LIFO pop — caller must handle the case where the stack is empty.
function _take_lifo!(s::LIFOStack)
    lock(s.lock) do
        pop!(s.data)
    end
end

Base.isready(s::LIFOStack) = !isempty(s.data)   # approximate (no lock), for scheduling hints only
