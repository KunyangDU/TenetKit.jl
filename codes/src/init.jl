
import LinearAlgebra: BLAS

function __init__()
    println("Julia Version $(VERSION)")
    __multithreading_init__()
    flush(stdout)
end


function __multithreading_init__()
    # if MKL is not used, set num_threads of BLAS to 1 to avoid the confliction to the high-level multi-threading implementations
    if !any(x -> startswith(x, "libmkl_rt"),
        basename(lib.libname) for lib in BLAS.get_config().loaded_libs)
        BLAS.set_num_threads(1)
    else
        # check if n_mkl * n_threads ≤ n_cpus
        if BLAS.get_num_threads() * Threads.nthreads() > get_num_cpus() 
        @warn "n_threads * n_mkl > n_cpus, may lead to bad performance!" 
        end
    end
    println("* Multi-threading infomation:")
    println("  - Julia: $(get_num_threads_julia())")
    println("  - BLAS: $(BLAS.get_num_threads())")
    println("* BLAS infomation: $(BLAS.get_config())")
end

__init__()

