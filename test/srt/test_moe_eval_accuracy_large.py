"""
Usage:
python -m unittest test_moe_eval_accuracy_large.TestMoEEvalAccuracyLarge.test_mmlu
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestMoEEvalAccuracyLarge(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MOE_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--log-level-http",
                "warning",
                "--tp",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=5000,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.62)

        if is_in_ci():
            write_github_step_summary(f"### test_mmlu\n" f'{metrics["score"]=:.4f}\n')

    def test_human_eval(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="humaneval",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.40)

        if is_in_ci():
            write_github_step_summary(
                f"### test_human_eval\n" f'{metrics["score"]=:.4f}\n'
            )

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.61)


if __name__ == "__main__":
    unittest.main()
