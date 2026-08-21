splice(tr::MPSTensor{2},obj::MPSTensor{3}) = MPSTensor(@tensor tmp[-1,-2;-3] ≔ tr.A[-1,1] * obj.A[1,-2,-3])
splice(obj::MPSTensor{3},tl::MPSTensor{2}) = MPSTensor(@tensor tmp[-1,-2;-3] ≔ obj.A[-1,-2,1] * tl.A[1,-3])
splice(tl::MPSTensor{2},tr::MPSTensor{2}) = MPSTensor(@tensor tmp[-1,;-2] ≔ tl.A[-1,1] * tr.A[1,-2])

splice(tr::DenseMPOTensor{2},obj::DenseMPOTensor{4}) = DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ tr.A[-2,1] * obj.A[-1,1,-3,-4])
splice(obj::DenseMPOTensor{4},tl::DenseMPOTensor{2}) = DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ obj.A[-1,-2,1,-4] * tl.A[1,-3])

