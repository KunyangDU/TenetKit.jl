using LRUCache, Serialization

# ───────────────────────────────────────────────────────────────
# CachedVector{T} — LRU-backed lazy vector that spills cold entries to disk.
# ───────────────────────────────────────────────────────────────
# Memory-aware eviction via LRUCache's `by` parameter: the cache tracks the
# total memory (bytes) of hot entries. When total exceeds `memory_limit`,
# cold entries are evicted and serialized to disk.
#
# Access patterns supported:
#   cv[i]      — single-element get (cache hit → return; cold → deserialize)
#   cv[i] = v  — single-element set (cache store; cold entry cleaned up)
#   cv[r]      — UnitRange get (returns plain Vector, used for [site:site+1])
#   cv[r] = vs — UnitRange set (iterates, calls setindex! per element)
#   cv[:]      — Colon get/set (rare, only Algebra/axpby.jl)
#   length(cv), iterate(cv), firstindex(cv), lastindex(cv)
# ───────────────────────────────────────────────────────────────

# Memory estimator for LRU `by` parameter — returns allocated bytes of value.
_bytes(x) = Base.summarysize(x)

# Callable struct replaces anonymous closure as LRU finalizer.
# Named struct avoids JLD2 serialization warning for anonymous functions.
struct LRUEvictHandler
    cache_ref::Ref{Any}
    diskdir::String
    cold::Dict{Int, String}
end
function (h::LRUEvictHandler)(k, v)
    haskey(h.cache_ref[], k) && return  # overwrite — skip
    path = joinpath(h.diskdir, "tensor_$(k).bin")
    serialize(path, v)
    h.cold[k] = path
end

mutable struct CachedVector{T}
    cache::LRU{Int, T}       # hot entries: index → tensor wrapper (memory-limited)
    len::Int                 # logical length
    diskdir::String          # temp directory for serialized cold entries
    cold::Dict{Int, String}  # evicted entries: index → filepath
    evict::LRUEvictHandler   # named callable (not closure) for JLD2 compatibility

    function CachedVector{T}(len::Int, memory_limit::Int) where T
        diskdir = mktempdir()
        cold = Dict{Int, String}()
        cache_ref = Ref{Any}()
        evict = LRUEvictHandler(cache_ref, diskdir, cold)
        cache = LRU{Int, T}(maxsize = memory_limit, by = _bytes, finalizer = evict)
        cache_ref[] = cache
        cv = new(cache, len, diskdir, cold, evict)
        # Auto-cleanup temp directory when CachedVector is garbage collected.
        finalizer(cv) do x
            isdir(x.diskdir) && rm(x.diskdir, recursive = true, force = true)
        end
        return cv
    end
end

function CachedVector{T}(len::Int) where T
    return CachedVector{T}(len, _cache_memory_limit(T))
end

function CachedVector{T}(data::Vector{<:T}, memory_limit::Int) where T
    cv = CachedVector{T}(length(data), memory_limit)
    for (i, v) in enumerate(data)
        cv[i] = v
    end
    return cv
end

function CachedVector{T}(data::Vector{<:T}) where T
    return CachedVector{T}(data, _cache_memory_limit(T))
end

# ───────────────────────────────────────────────────────────────
# Access interface
# ───────────────────────────────────────────────────────────────

function Base.getindex(cv::CachedVector{T}, i::Int) where T
    @boundscheck 1 <= i <= cv.len || throw(BoundsError(cv, i))
    if haskey(cv.cache, i)
        return cv.cache[i]
    elseif haskey(cv.cold, i)
        path = cv.cold[i]
        val = deserialize(path)
        rm(path)
        delete!(cv.cold, i)
        cv.cache[i] = val          # may evict another entry via finalizer
        return val
    else
        error("CachedVector[$i] not initialized")
    end
end

function Base.setindex!(cv::CachedVector{T}, val, i::Int) where T
    @boundscheck 1 <= i <= cv.len || throw(BoundsError(cv, i))
    if haskey(cv.cold, i)
        rm(cv.cold[i])
        delete!(cv.cold, i)
    end
    cv.cache[i] = val              # may evict another entry via finalizer
    return val
end

# UnitRange — returns plain Vector (used exclusively for [site:site+1] / [site-1:site])
function Base.getindex(cv::CachedVector{T}, r::UnitRange) where T
    return [cv[i] for i in r]
end

function Base.setindex!(cv::CachedVector{T}, vals, r::UnitRange) where T
    @boundscheck length(vals) == length(r) || throw(DimensionMismatch("..."))
    for (i, v) in zip(r, vals)
        cv[i] = v
    end
    return vals
end

# Colon — rare (only Algebra/axpby.jl:228: y.ts[:] = x.ts[:])
function Base.getindex(cv::CachedVector{T}, ::Colon) where T
    return [cv[i] for i in 1:cv.len]
end

function Base.setindex!(cv::CachedVector{T}, vals, ::Colon) where T
    @boundscheck length(vals) == cv.len || throw(DimensionMismatch("..."))
    for (i, v) in enumerate(vals)
        cv[i] = v
    end
    return vals
end

# ───────────────────────────────────────────────────────────────
# Container interface
# ───────────────────────────────────────────────────────────────

Base.length(cv::CachedVector) = cv.len
Base.firstindex(cv::CachedVector) = 1
Base.lastindex(cv::CachedVector) = cv.len

function Base.iterate(cv::CachedVector{T}, state::Int = 1) where T
    state > cv.len && return nothing
    return (cv[state], state + 1)
end

function Base.show(io::IO, cv::CachedVector{T}) where T
    hot = length(cv.cache)
    col = length(cv.cold)
    cached_bytes = isempty(cv.cache) ? 0 : mapreduce(_bytes, +, values(cv.cache))
    limit_str = Base.format_bytes(cv.cache.maxsize)
    if col > 0
        print(io, "CachedVector{$T}($(cv.len) sites, $(hot)⊕$(col) hot/cold, $(Base.format_bytes(cached_bytes)) in RAM, limit $limit_str)")
    else
        print(io, "CachedVector{$T}($(cv.len) sites, all $hot in RAM ($(Base.format_bytes(cached_bytes))), limit $limit_str)")
    end
end

# ───────────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────────

hot_entries(cv::CachedVector) = collect(keys(cv.cache))
cold_entries(cv::CachedVector) = collect(keys(cv.cold))
cache_capacity(cv::CachedVector) = cv.cache.maxsize

# deepcopy: one-at-a-time iteration, never materializes all entries at once.
# deepcopy_internal is called by Julia's recursive deepcopy (e.g. deepcopy(::DenseMPS)).
# Only copies initialized indices (hot or cold); skips indices never set.
function Base.deepcopy_internal(cv::CachedVector{T}, idict::IdDict) where T
    result = CachedVector{T}(cv.len, cache_capacity(cv))
    idict[cv] = result
    for i in 1:cv.len
        if haskey(cv.cache, i) || haskey(cv.cold, i)
            result[i] = Base.deepcopy_internal(cv[i], idict)
        end
    end
    return result
end

# ───────────────────────────────────────────────────────────────
# JLD2 serialization — materialize to plain Vector to avoid
# serializing LRU internals (cache / cold dict / disk paths).
# JLD2 calls writeas(instance) and rconvert(Type, data), both
# must be defined. Do NOT add a type-level writeas — that would
# change how struct fields are deserialized and break @load.
# ───────────────────────────────────────────────────────────────
function JLD2.writeas(cv::CachedVector{T}) where T
    return [cv[i] for i in 1:cv.len]
end

function JLD2.rconvert(::Type{<:CachedVector{T}}, data::Vector{T}) where T
    return CachedVector{T}(data)
end
