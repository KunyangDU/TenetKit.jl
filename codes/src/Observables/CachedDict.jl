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
    dict_ref::Ref{Any}           # CachedDict reference (for lock access)
end
function (h::EnvEvictHandler)(k, v)
    d = h.dict_ref[]             # CachedDict
    lock(d.lock) do               # atomic: haskey + serialize
        haskey(d.cache, k) && return
        path = joinpath(d.diskdir, "env_$(k).bin")
        @timeit get_timer("io") "serialize" serialize(path, v)
        d.cold[k] = path
    end
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
        dict_ref = Ref{Any}()
        evict = EnvEvictHandler(dict_ref)
        cache = LRU{K, V}(maxsize = memory_limit, by = _bytes, finalizer = evict)
        dict = new(cache, cold, diskdir, evict, ReentrantLock())
        dict_ref[] = dict
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
            # LRU.pop! calls finalizer (eviction handler) on the popped entry,
            # which would serialize a value that is about to be consumed and
            # never read back.  Temporarily disable the finalizer during pop!.
            old_fin = d.cache.finalizer
            d.cache.finalizer = nothing
            val = pop!(d.cache, key)
            d.cache.finalizer = old_fin
            return val
        elseif haskey(d.cold, key)
            path = d.cold[key]
            val = @timeit get_timer("io") "deserialize" deserialize(path)
            rm(path)
            delete!(d.cold, key)
            return val
        else
            return nothing
        end
    end
end
