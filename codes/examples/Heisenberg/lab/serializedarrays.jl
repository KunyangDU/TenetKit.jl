using SerializedElementArrays: disk, pathname

a = reshape(1:6, 2, 3)
d = disk(a)
@show d isa SerializedElementArrays.SerializedElementArray
@show a[1, 2]
@show d[1, 2]
@show readdir(pathname(d))
d[2, 2] = 3