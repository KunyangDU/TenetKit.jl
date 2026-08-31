function contract(EnvL::SparseLeftEnvironmentTensor{2}, Hup::SparseMPOTensor, t::DenseMPOTensor, Hdown::SparseMPOTensor, t′::AdjointMPOTensor, EnvR::SparseRightEnvironmentTensor{2})
    vind_up = _validind(Hup)
    vind_down = _validind(Hdown)
    pairs = [(a,b) for a in eachindex(vind_up) for b in eachindex(vind_down)]
    accs = zeros(ComplexF64, get_nworker())   # 标量归约，scalartype = ComplexF64
    threaded_reduce!(pairs, accs; combine! = +) do pair, acc, w
        a, b = pair
        l_up, op_up, r_up, wl_up, wr_up = vind_up[a]
        l_down, op_down, r_down, wl_down, wr_down = vind_down[b]
        for (pi, i) in enumerate(l_up), (pj, j) in enumerate(l_down)
            for (pk, k) in enumerate(r_up), (pl, l) in enumerate(r_down)
                ww = wl_up[pi] * wl_down[pj] * wr_up[pk] * wr_down[pl]
                acc += ww * contract(EnvL.A[i,j], Hup[op_up], t, Hdown[op_down], t′, EnvR.A[k,l])
            end
        end
        acc
    end
end

function contract(EnvL::LeftEnvironmentTensor{2},ht::LocalOperator{1,1},t::DenseMPOTensor{4},hb::LocalOperator{1,1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[4,3] * ht.A[1,5] * t.A[2,3,7,1] * hb.A[6,2] * t′.A[8,5,6,4] * EnvR.A[7,8]
end

function contract(EnvL::LeftEnvironmentTensor{2},::IdentityOperator{1},t::DenseMPOTensor{4},hb::LocalOperator{1,1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[3,2] * t.A[1,2,6,4] * hb.A[5,1] * t′.A[7,4,5,3] * EnvR.A[6,7]
end

function contract(EnvL::LeftEnvironmentTensor{2},ht::LocalOperator{1,1},t::DenseMPOTensor{4},::IdentityOperator{1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[3,2] * ht.A[1,4] * t.A[5,2,6,1] * t′.A[7,4,5,3] * EnvR.A[6,7]
end

function contract(EnvL::LeftEnvironmentTensor{2},::IdentityOperator{1},t::DenseMPOTensor{4},::IdentityOperator{1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[2,1] * t.A[3,1,5,4] * t′.A[6,4,3,2] * EnvR.A[5,6]
end