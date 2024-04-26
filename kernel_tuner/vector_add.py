#!/usr/bin/env python
import os

import numpy
from kernel_tuner import tune_kernel
from kernel_tuner.accuracy import TunablePrecision, AccuracyObserver

# Specify the compiler flags Kernel Tuner should use to compile our kernel
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
flags = [f"-I{ROOT_DIR}/include", "-std=c++17"]

def tune():

    # Prepare input data
    size = 100000000
    n = numpy.int32(size)
    a = numpy.random.randn(size).astype(numpy.float64)
    b = numpy.random.randn(size).astype(numpy.float64)
    c = numpy.zeros_like(b)

    # Prepare the argument list of the kernel
    args = [
        TunablePrecision("float_type", c),
        TunablePrecision("float_type", a),
        TunablePrecision("float_type", b),
        n,
    ]

    # Define the reference answer to compute the kernel output against
    answer = [a+b, None, None, None]

    # Define the tunable parameters, in this case thread block size
    # and the type to use for the input and output data of our kernel
    tune_params = dict()
    tune_params["block_size_x"] = [64, 128, 256, 512]
    tune_params["float_type"] = ["half", "float", "double"]
    tune_params["elements_per_thread"] = [1, 2, 4, 8]

    # Observers will measure the error using either RMSE or MRE as error metric
    observers = [
        AccuracyObserver("RMSE", "error_rmse"),
        AccuracyObserver("MRE", "error_relative"),
    ]

    # The metrics here are only to ensure Kernel Tuner prints them to the console
    metrics = dict(RMSE=lambda p: p["error_rmse"], MRE=lambda p: p["error_relative"])

    results, env = tune_kernel(
        "vector_add",
        "vector_add.cu",
        size,
        args,
        tune_params,
        answer=answer,
        observers=observers,
        metrics=metrics,
        lang="cupy",
        compiler_options=flags
    )


if __name__ == "__main__":
    tune()
