from layout import Layout, LayoutTensor


fn matmul_naive_cpu[
    dtype: DType,
    lhs_layout: Layout,
    rhs_layout: Layout,
    out_layout: Layout,
](
    lhs: LayoutTensor[dtype, lhs_layout, MutAnyOrigin],
    rhs: LayoutTensor[dtype, rhs_layout, MutAnyOrigin],
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
):
    """Naive host-side matmul baseline for benchmarking/reference.

    This baseline is intentionally simple and is not used by kernel-only tests.
    """
    comptime m = Int(lhs_layout.shape[0])
    comptime k = Int(lhs_layout.shape[1])
    comptime n = Int(rhs_layout.shape[1])

    for i in range(m):
        for j in range(n):
            var acc: output.element_type = 0
            for kk in range(k):
                acc += lhs[i, kk] * rhs[kk, j]
            output[i, j] = acc
