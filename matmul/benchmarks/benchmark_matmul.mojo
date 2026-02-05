from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_almost_equal
from time import global_perf_counter_ns


from matmul.kernel import naive_matmul_gpu_kernel
from matmul.cpu import matmul_naive_cpu

comptime dtype = DType.float32

comptime M = 128
comptime K = 128
comptime N = 128
comptime BLOCK_X = 16
comptime BLOCK_Y = 16
comptime WARMUP_ITERS = 10
comptime BENCH_ITERS = 100


fn main() raises:
    with DeviceContext() as ctx:
        lhs_buf = ctx.enqueue_create_buffer[dtype](M * K)
        lhs_buf.enqueue_fill(0)

        rhs_buf = ctx.enqueue_create_buffer[dtype](K * N)
        rhs_buf.enqueue_fill(0)

        gpu_out_buf = ctx.enqueue_create_buffer[dtype](M * N)
        gpu_out_buf.enqueue_fill(0)

        cpu_out_host = ctx.enqueue_create_host_buffer[dtype](M * N)
        cpu_out_host.enqueue_fill(0)

        with lhs_buf.map_to_host() as lhs_host, rhs_buf.map_to_host() as rhs_host:
            for i in range(M):
                for j in range(K):
                    lhs_host[i * K + j] = i + j

            for i in range(K):
                for j in range(N):
                    rhs_host[i * N + j] = Scalar[DType.float32]((i + j) / 2)

        comptime lhs_layout = Layout.row_major(M, K)
        comptime rhs_layout = Layout.row_major(K, N)
        comptime out_layout = Layout.row_major(M, N)
        var cpu_total_ns: UInt64 = 0
        var cpu_avg_ns: UInt64 = 0

        with lhs_buf.map_to_host() as lhs_host, rhs_buf.map_to_host() as rhs_host:
            lhs_cpu = LayoutTensor[dtype, lhs_layout, MutAnyOrigin](
                lhs_host.unsafe_ptr()
            )
            rhs_cpu = LayoutTensor[dtype, rhs_layout, MutAnyOrigin](
                rhs_host.unsafe_ptr()
            )
            out_cpu = LayoutTensor[dtype, out_layout, MutAnyOrigin](
                cpu_out_host.unsafe_ptr()
            )
            for _ in range(WARMUP_ITERS):
                matmul_naive_cpu[dtype, lhs_layout, rhs_layout, out_layout](
                    lhs_cpu,
                    rhs_cpu,
                    out_cpu,
                )

            var cpu_start_ns: UInt64 = global_perf_counter_ns()
            for _ in range(BENCH_ITERS):
                matmul_naive_cpu[dtype, lhs_layout, rhs_layout, out_layout](
                    lhs_cpu,
                    rhs_cpu,
                    out_cpu,
                )
            var cpu_end_ns: UInt64 = global_perf_counter_ns()
            cpu_total_ns = cpu_end_ns - cpu_start_ns
            cpu_avg_ns = cpu_total_ns // UInt64(BENCH_ITERS)

        print("CPU baseline loop complete")
        print("CPU total ns:", cpu_total_ns)
        print("CPU avg ns/iter:", cpu_avg_ns)

        lhs_tensor = LayoutTensor[dtype, lhs_layout, MutAnyOrigin](
            lhs_buf.unsafe_ptr()
        )
        rhs_tensor = LayoutTensor[dtype, rhs_layout, MutAnyOrigin](
            rhs_buf.unsafe_ptr()
        )
        out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](
            gpu_out_buf.unsafe_ptr()
        )

        comptime blocks_y = (M + BLOCK_Y - 1) // BLOCK_Y
        comptime blocks_x = (N + BLOCK_X - 1) // BLOCK_X
        comptime kernel = naive_matmul_gpu_kernel[
            dtype, lhs_layout, rhs_layout, out_layout
        ]

        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[kernel, kernel](
                lhs_tensor,
                rhs_tensor,
                out_tensor,
                grid_dim=(blocks_x, blocks_y),
                block_dim=(BLOCK_X, BLOCK_Y),
            )
        ctx.synchronize()

        var gpu_start_ns: UInt64 = global_perf_counter_ns()
        for _ in range(BENCH_ITERS):
            ctx.enqueue_function[kernel, kernel](
                lhs_tensor,
                rhs_tensor,
                out_tensor,
                grid_dim=(blocks_x, blocks_y),
                block_dim=(BLOCK_X, BLOCK_Y),
            )
        ctx.synchronize()
        var gpu_end_ns: UInt64 = global_perf_counter_ns()
        var gpu_total_ns: UInt64 = gpu_end_ns - gpu_start_ns
        var gpu_avg_ns: UInt64 = gpu_total_ns // UInt64(BENCH_ITERS)

        with gpu_out_buf.map_to_host() as gpu_host:
            for i in range(M):
                for j in range(N):
                    idx = i * N + j
                    assert_almost_equal(
                        gpu_host[idx],
                        cpu_out_host[idx],
                        atol=1e-3,
                        rtol=1e-2,
                    )

        print("✓ GPU kernel output matches CPU baseline")
        print("GPU benchmark loop complete")
        print("GPU total ns:", gpu_total_ns)
        print("GPU avg ns/iter:", gpu_avg_ns)
