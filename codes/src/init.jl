

function __init__()
    println("Julia Version $(VERSION)")
    __multithreading_init__()
    flush(stdout)
end

function __multithreading_init__()
    println("* Multi-threading infomation:")
    println("  - Julia: $(Threads.nthreads())")
    println("  - BLAS: $(MKL.BLAS.get_num_threads())")
    println("* BLAS infomation: $(MKL.BLAS.get_config())")
end

__init__()

