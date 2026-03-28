using JLD2,Serialization

struct Test
    a::Int64
end

NODE_DATA_PATH = mktempdir(".")
A = Test(100)
(filepath, io) = mktemp(NODE_DATA_PATH)
serialize(io, A)
close(io)
A = nothing
GC.gc()

A = deserialize(filepath)
NODE_DATA_PATH

# rm(a)
# test = randn(10,10)
# @save "hello/test.jld2" test

# rm("hello";force = true,recursive=true)
# getpid()