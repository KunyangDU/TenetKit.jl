# 终端柱状图（水平，Unicode 1/8 精度）
# 单条 bar：直接打印 ratio ∈ [0,1]
function bar(ratio::Float64; width::Int=20)
    eighths = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    nfill = round(Int, ratio * width * 8)
    nfill = clamp(nfill, 0, width * 8)
    fill_chars = repeat("█", nfill ÷ 8) * eighths[nfill % 8 + 1]
    empty_chars = repeat(" ", width - length(fill_chars))
    # 刻度：每 0.1（10% 宽度）一个 ·，只在空白区
    step = div(width, 10)
    tickline = IOBuffer()
    for (i, c) in enumerate(empty_chars)
        pos = i + length(fill_chars)   # 整体位置
        if step > 0 && pos % step == 0 && pos < width
            write(tickline, '·')
        else
            write(tickline, c)
        end
    end
    print("|", fill_chars, String(take!(tickline)), "|")
end

# 向量版：自动归一化
function bar(data::Vector; labels=nothing, width=20, title="")
    isempty(data) && return
    m = maximum(data)
    m == 0 && (m = 1)
    w = max(maximum(length(string(l)) for l in (labels === nothing ? string.(1:length(data)) : labels)), 4)

    isempty(title) || println(title)
    for (i, v) in enumerate(data)
        lbl = labels === nothing ? "$i" : labels[i]
        print(rpad(lbl, w), " ")
        bar(v / m; width=width)
        println(" ", v)
    end
end


function vbshow(site::Int64, t₀::Number, localinfo::AbstractInformation, Alg::AbstractAlgorithm)
    Dr = localinfo.bond.D / _getdim(Alg.trunc)
    SEr = localinfo.bond.S / log(_getdim(Alg.trunc))
    print("site ",lpad("$(site)",3),", time ",rpad("$(round(time() - t₀;digits = 3)) s",10),"-> D ")
    bar(Dr)
    print(", S ")
    bar(SEr)
    print(" -> ",localinfo.bond," \n")
end