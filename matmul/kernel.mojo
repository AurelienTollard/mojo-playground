from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim

fn naive_matmul_gpu_kernel[
    dtype: DType,
    lhs_layout: Layout,
    rhs_layout: Layout,
    out_layout: Layout,
](
    lhs: LayoutTensor[dtype, lhs_layout, MutAnyOrigin],
    rhs: LayoutTensor[dtype, rhs_layout, MutAnyOrigin],
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
):
    M = lhs.shape[0]()
    N = rhs.shape[1]()
    K = rhs.shape[0]()

    row = Int(block_dim.x * block_idx.x + thread_idx.x)
    col = Int(block_dim.y * block_idx.y + thread_idx.y)

    acc: output.element_type = 0
    if row < M and col < N:
        for k_idx in range(K):
            acc += lhs[row, k_idx] * rhs[k_idx, col]

    output[row, col] = acc
