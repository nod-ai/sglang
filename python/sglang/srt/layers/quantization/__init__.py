# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py

from typing import Dict, Type

from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.gptq import (
    GPTQConfig,
    GPTQMarlinConfig,
    GPTQMarlinMoEMethod,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
)
from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
from sglang.srt.layers.quantization.qoq import QoQConfig
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

# VLLM-dependent quantization methods
VLLM_QUANTIZATION_METHODS = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "tpu_int8": Int8TpuConfig,
    "fbgemm_fp8": FBGEMMFp8Config,
    "marlin": MarlinConfig,
    "modelopt": ModelOptFp8Config,
    "gguf": GGUFConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "qqq": QQQConfig,
    "experts_int8": ExpertsInt8Config,
    "w8a8_int8": W8A8Int8Config,
}

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS, **VLLM_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    if quantization in VLLM_QUANTIZATION_METHODS and not VLLM_AVAILABLE:
        raise ValueError(
            f"{quantization} quantization requires some operators from vllm. "
            "Please install vllm by `pip install vllm==0.8.4`"
        )

    return QUANTIZATION_METHODS[quantization]


def gptq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
        GPTQMarlinMoEMethod,
    )

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if isinstance(layer, LinearBase):
        return GPTQMarlinLinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return GPTQMarlinMoEMethod(self)

    if isinstance(self, GPTQConfig):
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )
    elif isinstance(self, GPTQMarlinConfig):
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQMarlinLinearMethod
        )
    return None


def awq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinLinearMethod,
        AWQMoEMethod,
    )

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    def patched_isinstance(obj, classinfo):
        if classinfo is LinearBase:
            return original_isinstance(obj, PatchedLinearBase)
        if classinfo is FusedMoE:
            return original_isinstance(obj, PatchedFusedMoE)
        if classinfo is VocabParallelEmbedding:
            return original_isinstance(obj, PatchedVocabParallelEmbedding)
        return original_isinstance(obj, classinfo)

    builtins.isinstance = patched_isinstance


def patch_vllm_linear_base_isinstance():
    import builtins

    from vllm.model_executor.layers.linear import LinearBase

    from sglang.srt.layers.linear import LinearBase as PatchedLinearBase

    original_isinstance = builtins.isinstance

    def patched_isinstance(obj, classinfo):
        if classinfo is LinearBase:
            return original_isinstance(obj, PatchedLinearBase)
        return original_isinstance(obj, classinfo)

    builtins.isinstance = patched_isinstance


def apply_monkey_patches():
    """Apply all monkey patches in one place."""
    setattr(GPTQMarlinConfig, "get_quant_method", gptq_get_quant_method)
    setattr(GPTQConfig, "get_quant_method", gptq_get_quant_method)

    monkey_patch_moe_apply(AWQMoEMethod)
    monkey_patch_moe_apply(GPTQMarlinMoEMethod)
    monkey_patch_moe_apply(CompressedTensorsW8A8Fp8MoEMethod)
    monkey_patch_moe_apply(CompressedTensorsWNA16MoEMethod)


patch_vllm_linear_base_isinstance()
# Apply patches when module is imported
apply_monkey_patches()


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
