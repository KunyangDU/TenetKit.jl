
TensorKit.usebraidcache_abelian[] = false
TensorKit.usebraidcache_nonabelian[] = false

# Detect memory limit for CachedVector:
#   1. --heap-size-hint flag (servers often pass this)
#   2. Fallback: 30% of total physical memory
#   3. Last resort: 2GB
function _detect_memory_limit()
    # 1. Try --heap-size-hint (Julia's GC heap hint, passed via CLI)
    hint = try
        Int(Base.JLOptions().heap_size_hint)
    catch
        0
    end
    if hint > 0
        return hint * 2 ÷ 5  # 40% of heap hint for cache
    end
    # 2. Fallback: total physical memory
    total_mem = try
        Int(Sys.total_memory())
    catch
        0
    end
    if total_mem > 0
        return total_mem * 3 ÷ 10  # 30% of total RAM
    end
    # 3. Last resort
    return 2_000_000_000
end

const CACHE_MEMORY_LIMIT = Ref(_detect_memory_limit())

# Per-type cache memory ratios — multiplied by CACHE_MEMORY_LIMIT.
# MPS  O(D²×d)   ~32MB/D=1000;  MPO  O(D²×d²) ~64MB/D=1000
# Env  O(D²) ~16MB or O(D³) ~16GB when composite.
const MPS_CACHE_RATIO  = Ref(0.3)
const MPO_CACHE_RATIO  = Ref(0.3)
const ENV_CACHE_RATIO  = Ref(0.4)
const OBS_ENV_CACHE_RATIO = Ref(0.15)
const DEFAULT_CACHE_RATIO = Ref(1.0)

# Resolve memory limit for a CachedVector{T}. Type-specific methods are
# added in AbstractTensor.jl / AbstractEnvironment.jl after types are defined.
function _cache_memory_limit(::Type)
    return round(Int, CACHE_MEMORY_LIMIT[] * DEFAULT_CACHE_RATIO[])
end

get_num_cpus() = Sys.CPU_THREADS
get_num_threads_julia() = Threads.nthreads()
