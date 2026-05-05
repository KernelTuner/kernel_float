# Kernel Float

![Kernel Float logo](https://raw.githubusercontent.com/KernelTuner/kernel_float/main/docs/logo.png)

[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/KernelTuner/kernel_float/)
![GitHub branch checks state](https://img.shields.io/github/actions/workflow/status/KernelTuner/kernel_float/docs.yml)
![GitHub](https://img.shields.io/github/license/KernelTuner/kernel_float)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/KernelTuner/kernel_float)
![GitHub Repo stars](https://img.shields.io/github/stars/KernelTuner/kernel_float?style=social)


_Kernel Float_ is a header-only library for CUDA/HIP that makes working with reduced-precision floating-point types and vector arithmetic simple and expressive, with zero performance overhead.

## Summary

CUDA/HIP natively offers several reduced precision floating-point types (`__half`, `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2`)
and vector types (e.g., `__half2`, `__nv_fp8x4_e4m3`, `float3`).
However, working with these types is cumbersome:
mathematical operations require intrinsics (e.g., `__hadd2` performs addition for `__half2`),
type conversion is awkward (e.g., `__nv_cvt_halfraw2_to_fp8x2` converts float16 to float8),
and some functionality is missing (e.g., one cannot convert a `__half` to `__nv_bfloat16`).

_Kernel Float_ resolves this by offering a single unified vector type `kernel_float::vec<T, N>` that stores `N` elements of type `T`.
Internally, the data is stored using the optimal data layout for the given type.
Operator overloading (like `+`, `*`, `&&`) has been implemented such that the most optimal intrinsic for the available types is selected automatically.
Many mathematical functions (like `log`, `exp`, `sin`) and common operations (such as `sum`, `range`, `for_each`) are also available.

The generated assembly is identical to hand-written intrinsics code, meaning you get clean and maintainable source code without sacrificing performance.


## Features

In a nutshell, _Kernel Float_ offers the following features:

* Single type `vec<T, N>` that unifies all vector types.
* Operator overloading to simplify programming.
* Support for half (16 bit) floating-point arithmetic, with a fallback to single precision for unsupported operations.
* Support for quarter (8 bit) floating-point types.
* Easy integration as a single header file.
* Written for C++17.
* Compatible with CUDA: `nvcc` (NVIDIA Compiler) and `nvrtc` (NVIDIA Runtime Compilation).
* Compatible with HIP: `hipcc` (AMD HIP Compiler)


## Quick Example

Below shows a simple example kernel that multiplies an `input` array by a `constant` and accumulates into an `output` array.
Each thread processes two elements.


```cpp
#include "kernel_float.h"
namespace kf = kernel_float;

__global__ void kernel(kf::vec_ptr<const half, 2> input, int constant, kf::vec_ptr<float, 2> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] += input[i] * constant;
}
```

Notice how easy it would be to change the precision (for example, `double` to `half`) or the vector size (for example, 4 instead of 2 items per thread).
Check out the [examples](https://github.com/KernelTuner/kernel_float/tree/main/examples) directory for some examples.

Here is how the same kernel would look for CUDA without Kernel Float.

```cpp
#include <cuda_fp16.h>

__global__ void kernel(const half* input, int constant, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __half in0 = input[2 * i + 0];
    __half in1 = input[2 * i + 1];
    __half2 a = __halves2half2(in0, in1);
    __half b = __int2half_rn(constant);
    __half2 c = __half2half2(b);
    __half2 d = __hmul2(a, c);
    __half e = __low2half(d);
    __half f = __high2half(d);
    float out0 = __half2float(e);
    float out1 = __half2float(f);
    output[2 * i + 0] += out0;
    output[2 * i + 1] += out1;
}
```

Even though the second kernel looks a lot more complex, both generate nearly identical PTX code.


## Installation

This is a header-only library. Copy the file `single_include/kernel_float.h` to your project and include it:

```cpp
#include "kernel_float.h"
```

Use the provided Makefile to generate this single-include header file if it is outdated:

```
make
```


## Links

- [Documentation](https://kerneltuner.github.io/kernel_float/)
- [API reference](https://kerneltuner.github.io/kernel_float/api.html)
- [Examples](https://github.com/KernelTuner/kernel_float/tree/main/examples)


## Citation

If you use Kernel Float in scholarly work, please cite the following paper:
```
@article{heldens2025kernel,
  author = {Heldens, Stijn and van Werkhoven, Ben},
  title = {Kernel Float: Unlocking Mixed-Precision GPU Programming},
  publisher = {ACM},
  journal = {ACM Trans. Math. Softw.},
  year = {2025},
  doi = {10.1145/3779120},
}
```

## License

Licensed under Apache 2.0. See [LICENSE](https://github.com/KernelTuner/kernel_float/blob/main/LICENSE).


## Related Work

* [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner)
* [Kernel Launcher](https://github.com/KernelTuner/kernel_launcher)

