import argparse
import glob
from dataclasses import dataclass

from sglang.test.test_utils import run_unittest_files


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


suites = {
    "per-commit": [
        "models/test_embedding_models.py",
        "models/test_generation_models.py",
        "models/test_lora.py",
        "models/test_qwen_models.py",
        "models/test_reward_models.py",
        "sampling/penaltylib",
        "test_abort.py",
        "test_chunked_prefill.py",
        "test_custom_allreduce.py",
        "test_double_sparsity.py",
        "test_eagle_infer.py",
        "test_embedding_openai_server.py",
        "test_eval_accuracy_mini.py",
        "test_gguf.py",
        "test_input_embeddings.py",
        "test_json_constrained.py",
        "test_large_max_new_tokens.py",
        "test_metrics.py",
        "test_mla.py",
        "test_mla_fp8.py",
        "test_no_chunked_prefill.py",
        "test_no_overlap_scheduler.py",
        "test_openai_server.py",
        "test_pytorch_sampling_backend.py",
        "test_radix_attention.py",
        "test_regex_constrained.py",
        "test_release_memory_occupation.py",
        "test_request_length_validation.py",
        "test_retract_decode.py",
        "test_server_args.py",
        "test_session_control.py",
        "test_skip_tokenizer_init.py",
        "test_srt_engine.py",
        "test_srt_endpoint.py",
        "test_torch_compile.py",
        "test_torch_compile_moe.py",
        "test_torch_native_attention_backend.py",
        "test_torchao.py",
        "test_triton_attention_kernels.py",
        "test_triton_attention_backend.py",
        "test_update_weights_from_disk.py",
        "test_update_weights_from_tensor.py",
        "test_vision_chunked_prefill.py",
        "test_vision_llm.py",
        "test_vision_openai_server.py",
        "test_w8a8_quantization.py",
        "test_fp8_kvcache.py",
        "test_fp8_kernel.py",
    ],
    "nightly": [
        "test_nightly_gsm8k_eval.py",
        # Disable temporarily
        # "test_nightly_math_eval.py",
    ],
}

# Expand suite
for target_suite_name, target_tests in suites.items():
    for suite_name, tests in suites.items():
        if suite_name == target_suite_name:
            continue
        if target_suite_name in tests:
            tests.remove(target_suite_name)
            tests.extend(target_tests)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1800,
        help="The time limit for running one file in seconds.",
    )
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()) + ["all"],
        help="The suite to run",
    )
    arg_parser.add_argument(
        "--range-begin",
        type=int,
        default=0,
        help="The begin index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="The end index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    arg_parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)
    else:
        files = files[args.range_begin : args.range_end]

    print("The running tests are ", [f.name for f in files])

    exit_code = run_unittest_files(files, args.timeout_per_file)
    exit(exit_code)
