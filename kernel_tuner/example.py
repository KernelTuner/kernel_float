import numpy as np
import os.path
from pprint import pprint

import kernel_tuner
from kernel_tuner.observers import BenchmarkObserver


class AccuracyObserver(BenchmarkObserver):
    def __init__(self, score_function, args, metric=None):
        """ AccuracyObserver
        :param score_function: User-defined function that returns a single floating-point
            value to score the GPU result. The function is called with a single
            argument, a list of numpy arrays for each non-None value in the args list.
        :type score_function: callable
        :param args: List of arguments to the kernel, corresponding to the number and types
            of arguments the kernel takes. Values that are not None are reset before
            the kernel starts. Use None for arguments that are not results nor need
            to be reset to obtain meaningful results to avoid needless data movement.
        :type args: list
        :param metric: string name for the metric, defaults to "error".
        :type metric: string
        """
        self.metric = metric or "error"
        self.args = args
        self.func = score_function
        self.scores = []

    def before_start(self):
        for i, arg in enumerate(self.args):
            if not arg is None:
                self.dev.memcpy_htod(self.dev.allocations[i], arg)

    def after_finish(self):
        gpu_result = []
        for i, arg in enumerate(self.args):
            if not arg is None:
                res = np.zeros_like(arg)
                self.dev.memcpy_dtoh(res, self.dev.allocations[i])
                gpu_result.append(res)
        score = self.func(gpu_result)
        self.scores.append(score)

    def get_results(self):
        result = {self.metric: np.average(self.scores)}
        self.scores = []
        return result


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"


if __name__ == "__main__":
    kernel_name = "vector_add<items_per_thread>"
    kernel_source = "example.cu"
    problem_size = int(10 ** 8)
    flags = [f"-I{ROOT_DIR}/include", "-std=c++17"]

    N = np.int32(problem_size)
    A = np.random.randn(N).astype(np.float64)
    B = np.random.randn(N).astype(np.float64)
    C = np.zeros(N).astype(np.float64)
    args = [N, A, B, C]

    expected_C = A + B
    answer = [None, None, expected_C]

    tune_params = dict()
    tune_params["block_size_x"] = [256, 512, 1024]
    tune_params["items_per_thread"] = [1, 2, 4, 8]
    tune_params["TYPE"] = ["float", "double", "__half", "__nv_bfloat16"]

    def compute_error(gpu_results):
        return np.average(np.abs(expected_C - gpu_results[0]))

    observers = [AccuracyObserver(compute_error, answer, "error")]

    results, env = kernel_tuner.tune_kernel(
            kernel_name,
            kernel_source,
            problem_size,
            args,
            tune_params,
            # answer=answer,
            grid_div_x=["items_per_thread * block_size_x"],
            lang="cupy",
            observers=observers,
            compiler_options=flags
    )

    pprint(results)
