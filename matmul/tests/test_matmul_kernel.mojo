from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_almost_equal

from matmul.kernel import naive_matmul_gpu_kernel
from matmul.cpu import matmul_naive_cpu

comptime dtype = DType.float32

comptime M = 32
comptime K = 32
comptime N = 32


fn run_kernel_integration_test() raises:
    with DeviceContext() as ctx:
        lhs_buf = ctx.enqueue_create_buffer[dtype](M * K)
        lhs_buf.enqueue_fill(0)

        rhs_buf = ctx.enqueue_create_buffer[dtype](K * N)
        rhs_buf.enqueue_fill(0)

        gpu_out_buf = ctx.enqueue_create_buffer[dtype](M * N)
        gpu_out_buf.enqueue_fill(0)

        cpu_out_buf = ctx.enqueue_create_buffer[dtype](M * N)
        cpu_out_buf.enqueue_fill(0)

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

        with lhs_buf.map_to_host() as lhs_host, rhs_buf.map_to_host() as rhs_host, cpu_out_buf.map_to_host() as cpu_out_host:
            lhs_cpu = LayoutTensor[dtype, lhs_layout, MutAnyOrigin](
                lhs_host.unsafe_ptr()
            )
            rhs_cpu = LayoutTensor[dtype, rhs_layout, MutAnyOrigin](
                rhs_host.unsafe_ptr()
            )
            out_cpu = LayoutTensor[dtype, out_layout, MutAnyOrigin](
                cpu_out_host.unsafe_ptr()
            )
            matmul_naive_cpu[dtype, lhs_layout, rhs_layout, out_layout](
                lhs_cpu,
                rhs_cpu,
                out_cpu,
            )

        lhs_tensor = LayoutTensor[dtype, lhs_layout, MutAnyOrigin](
            lhs_buf.unsafe_ptr()
        )
        rhs_tensor = LayoutTensor[dtype, rhs_layout, MutAnyOrigin](
            rhs_buf.unsafe_ptr()
        )
        out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](
            gpu_out_buf.unsafe_ptr()
        )

        comptime kernel = naive_matmul_gpu_kernel[
            dtype, lhs_layout, rhs_layout, out_layout
        ]

        ctx.enqueue_function[kernel, kernel](
            lhs_tensor,
            rhs_tensor,
            out_tensor,
            grid_dim=(1, 1),
            block_dim=(M, N),
        )
        ctx.synchronize()

        with gpu_out_buf.map_to_host() as out_host, cpu_out_buf.map_to_host() as cpu_out_host:
            for i in range(M):
                for j in range(N):
                    assert_almost_equal(
                        out_host[i * N + j],
                        cpu_out_host[i * N + j],
                        atol=1e-5,
                        rtol=1e-5,
                    )

        print("✓ matmul kernel integration test passed")


fn main() raises:
    run_kernel_integration_test()
