
TensorKit.usebraidcache_abelian[] = false
TensorKit.usebraidcache_nonabelian[] = false

get_num_cpus() = Sys.CPU_THREADS
get_num_threads_julia() = Threads.nthreads()

function _disk(A::AbstractArray; kwargs...)
    return SerializedElementArrays.disk(A; path=DISK_BASEDIR[], kwargs...)
end

const DISK_BASEDIR = Ref{String}(tempdir())
diskdir!(path = mktempdir(pwd())) = (DISK_BASEDIR[] = path)
const _io_timers = TimerOutput[]

