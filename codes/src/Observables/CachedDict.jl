# ───────────────────────────────────────────────────────────────
# CachedDict{K,V} — LRU-backed dictionary that spills cold entries to disk.
# ───────────────────────────────────────────────────────────────
# Same LRU eviction pattern as CachedVector: when total memory (bytes)
# exceeds `memory_limit`, cold entries are evicted via the LRU finalizer
# and serialized to disk.
#
# Interface:
#   d[key] = val  — store (may trigger eviction of cold entries to disk)
#   take!(d, key) — retrieve and delete (hot → pop, cold → deserialize+rm)
# ───────────────────────────────────────────────────────────────

_bytes(x) = Base.summarysize(x)

struct EnvEvictHandler
    cache_ref::Ref{Any}
    cold::Dict{UInt64, String}
    diskdir::String
end
function (h::EnvEvictHandler)(k, v)
    haskey(h.cache_ref[], k) && return
    path = joinpath(h.diskdir, "env_$(k).bin")
    serialize(path, v)
    h.cold[k] = path
end

mutable struct CachedDict{K,V}
    cache::LRU{K, V}
    cold::Dict{K, String}
    diskdir::String
    evict::EnvEvictHandler
    lock::ReentrantLock

    function CachedDict{K,V}(memory_limit::Int) where {K,V}
        diskdir = mktempdir()
        cold = Dict{K, String}()
        cache_ref = Ref{Any}()
        evict = EnvEvictHandler(cache_ref, cold, diskdir)
        cache = LRU{K, V}(maxsize = memory_limit, by = _bytes, finalizer = evict)
        cache_ref[] = cache
        dict = new(cache, cold, diskdir, evict, ReentrantLock())
        finalizer(dict) do x
            isdir(x.diskdir) && rm(x.diskdir, recursive = true, force = true)
        end
        return dict
    end
end

function Base.setindex!(d::CachedDict, val, key)
    lock(d.lock) do
        d.cache[key] = val
    end
end

function Base.take!(d::CachedDict, key)
    lock(d.lock) do
        if haskey(d.cache, key)
            return pop!(d.cache, key)
        elseif haskey(d.cold, key)
            path = d.cold[key]
            val = deserialize(path)
            rm(path)
            delete!(d.cold, key)
            return val
        else
            return nothing
        end
    end
end
