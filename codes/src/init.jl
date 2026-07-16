
function __init_threads__()
    d = Dict{String,Int}()
    d["JULIA_THREADS"] = Threads.nthreads()
    d["TOTAL_CPUS"] = parse(Int, get(ENV, "SLURM_CPUS_ON_NODE", string(d["JULIA_THREADS"])))
    d["BLAS_THREADS_DEFAULT"] = BLAS.get_num_threads()
    d["BLAS_THREADS"] = d["BLAS_THREADS_DEFAULT"]
    d["NWORKER"] = div(d["TOTAL_CPUS"], d["BLAS_THREADS_DEFAULT"])
    GLOBAL_THREADS[] = d
    return d
end

function __init__()
    println("Julia Version $(VERSION)")
    __init_threads__()
    __multithreading_init__()
    flush(stdout)
end


function __multithreading_init__()
    # if MKL is not used, set num_threads of BLAS to 1 to avoid the confliction to the high-level multi-threading implementations
    if !any(x -> startswith(x, "libmkl_rt"),
        basename(lib.libname) for lib in BLAS.get_config().loaded_libs)
        BLAS.set_num_threads(1)
        GLOBAL_THREADS[]["BLAS_THREADS"] = BLAS.get_num_threads()
        GLOBAL_THREADS[]["BLAS_THREADS_DEFAULT"] = GLOBAL_THREADS[]["BLAS_THREADS"]
        GLOBAL_THREADS[]["NWORKER"] = div(GLOBAL_THREADS[]["TOTAL_CPUS"], GLOBAL_THREADS[]["BLAS_THREADS_DEFAULT"])
    else
        # check if n_mkl * n_threads ≤ n_cpus
        if GLOBAL_THREADS[]["BLAS_THREADS_DEFAULT"] * GLOBAL_THREADS[]["NWORKER"] > GLOBAL_THREADS[]["TOTAL_CPUS"]
            @warn "n_threads * n_mkl > n_cpus, may lead to bad performance!" 
        end
    end
    println("* Multi-threading infomation:")
    println("  - CPU: $(GLOBAL_THREADS[]["TOTAL_CPUS"])")
    println("  - Julia: $(GLOBAL_THREADS[]["JULIA_THREADS"])")
    println("  - nworker: $(GLOBAL_THREADS[]["NWORKER"])")
    println("  - BLAS: $(GLOBAL_THREADS[]["BLAS_THREADS_DEFAULT"])")
    println("* BLAS infomation: $(BLAS.get_config())")
end


function __init_io__()
    empty!(_io_timers)
    for _ in 1:Threads.nthreads()
        push!(_io_timers, TimerOutput())
    end
end

_local_io_timer() = _io_timers[Threads.threadid()]

function _merge_io!(to::TimerOutput)
    for t in _io_timers
        merge!(to, t)
        reset_timer!(t)
    end
end

__init__()
__init_io__()

