import unittest

import torch
from sgl_kernel import fp8_scaled_mm


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    o = o.to(torch.float32)
    temp1 = o * scale_a.view(-1, 1)
    temp2 = temp1 * scale_b.view(1, -1)
    final = temp2.to(out_dtype)
    if bias is not None:
        final = final + bias.view(1, -1)

    return final


class TestFp8Gemm(unittest.TestCase):
    def _test_accuracy_once(self, M, N, K, with_bias, out_dtype, device):
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        a_fp32 = (
            (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
        )
        a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        b_fp32 = (
            (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
        )
        b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        scale_a = torch.randn((M,), device=device, dtype=torch.float32) * 0.001
        scale_b = torch.randn((N,), device=device, dtype=torch.float32) * 0.001
        if with_bias:
            bias = torch.randn((N,), device=device, dtype=out_dtype)
        else:
            bias = None
        o1 = torch.empty((M, N), device=device, dtype=torch.bfloat16)
        b_fp8 = b_fp8.t()
        o = torch_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
        o1 = fp8_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
        rtol = 0.02
        atol = 1
        torch.testing.assert_close(o, o1, rtol=rtol, atol=atol)
        print(f"M={M}, N={N}, K={K}, with_bias={with_bias}, out_dtype={out_dtype}: OK")

    def test_accuracy(self):
        Ms = [1, 128, 512, 1024, 4096]
        Ns = [16, 128, 512, 1024, 4096]
        Ks = [512, 1024, 4096, 8192, 16384]
        bias_opts = [True, False]
        out_dtypes = [torch.bfloat16, torch.float16]
        for M in Ms:
            for N in Ns:
                for K in Ks:
                    for with_bias in bias_opts:
                        for out_dtype in out_dtypes:
                            self._test_accuracy_once(
                                M, N, K, with_bias, out_dtype, "cuda"
                            )


if __name__ == "__main__":
    unittest.main()
