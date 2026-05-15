using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 32
Lx = 8
Ly = 1
params = (J=1, Δ = 1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

begin
    Obs = Observable()
    LocalSpace = TrivialSpinOneHalf

    for i in 1:size(Latt), j in i+1:size(Latt)
        pair = (i,j)
        addObs!(Obs,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),nothing)
        addObs!(Obs,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),nothing)
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),nothing)
    end

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",false,nothing)
        addObs!(Obs,LocalSpace.Sy,i,"Sy",false,nothing)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",false,nothing)
    end
    # @show Obs.node
    # 在 calObs! 之前加几行
    using Base: summarysize
    # trivial calObs.jl 里，calObs! 之前加
    # import TenetKit.TimerOutputs: get_timer
    reset_timer!(get_timer("io"))   # 确保从零开始
    GC.gc()                          # 排除 finalizer 延迟触发

    # 同时在 CachedVector.jl 的 LRUEvictHandler 和 CachedDict.jl 的 EnvEvictHandler 里临时加：
    # (h::LRUEvictHandler)(k, v) 函数内第一行加：
    #     
    # (h::EnvEvictHandler)(k, v) 函数内第一行加：
    #     
    # calObs!(Obs, ψ)



    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata



# gsdata[ (("Sz", "Sz"),)][((6,9),)]

gsdata
