
# TensorKit.usebraidcache_abelian[] = false
# TensorKit.usebraidcache_nonabelian[] = false

get_num_cpus() = GLOBAL_THREADS[]["TOTAL_CPUS"]
get_num_threads_julia() = GLOBAL_THREADS[]["JULIA_THREADS"]
get_nworker() = GLOBAL_THREADS[]["NWORKER"]

function _disk(A::AbstractArray; kwargs...)
    return SerializedElementArrays.disk(A; path=DISK_BASEDIR[], kwargs...)
end

const DISK_BASEDIR = Ref{String}(tempdir())
const IS_DISK = Ref{Bool}(false)
diskdir!(path = mktempdir(pwd())) = (DISK_BASEDIR[] = path)
# diskdir!(path = joinpath(pwd(), "disk_data")) = (DISK_BASEDIR[] = path)

const _io_timers = TimerOutput[]

const GLOBAL_THREADS = Ref{Dict{String,Int}}()

function BLAS_MIN!()
    d = GLOBAL_THREADS[]
    BLAS.set_num_threads(1)
    d["BLAS_THREADS"] = 1
    d["NWORKER"] = d["TOTAL_CPUS"]
end

function BLAS_MAX!()
    d = GLOBAL_THREADS[]
    n = d["BLAS_THREADS_DEFAULT"]
    BLAS.set_num_threads(n)
    d["BLAS_THREADS"] = n
    d["NWORKER"] = div(d["TOTAL_CPUS"], d["BLAS_THREADS"])
end
