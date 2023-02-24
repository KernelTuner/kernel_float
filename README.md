# Kernel Float


[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/KernelTuner/kernel_float/)
![GitHub branch checks state](https://img.shields.io/github/actions/workflow/status/KernelTuner/kernel_float/docs.yml)
![GitHub](https://img.shields.io/github/license/KernelTuner/kernel_float)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/KernelTuner/kernel_float)
![GitHub Repo stars](https://img.shields.io/github/stars/KernelTuner/kernel_float?style=social)


_Kernel Float_ is a header-only library for CUDA that makes it easy to work with vector types and reduced precision.

The CUDA standard library offers several reduce precision floating-point types (`__half`, `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2`)
and vector types (e.g., `__half2`, `__nv_fp8x4_e4m3`, `float3`).
However, working with these types is cumbersome since
there is not operator overloading (e.g., `__hadd2(x, y)` is required for addition on `__half2`),
type conversion is awkward (e.g., `__nv_cvt_halfraw2_to_fp8x2` converts 16 bit to 8 bit numbers),
and some functionality is missing (e.g., one cannot convert a `__half` to `__nv_bfloat16`).

_Kernel Float_ attempt to resolve this issue by offering a single data type `kernel_float::vec<T, N>`
that stores `N` elements of type `T`.
Internally, the data is stored using the most optimal type available, for example, `vec<half, 2>` uses `__half2` to store its items and `vec<fp8_e5m2, 4>` uses a `__nv_fp8_e5m2` internally.
Operator overloading (like `+`, `*`, `&&`) has been implemented in such a way that the most optimal intrinsic for the available types is selected automatically.
Many mathmetical functions (like `log`, `exp`, `sin`) and common operations (such as `sum`, `product`, `for_each`) are also available.




## Example

See the documentation for [examples](https://kerneltuner.github.io/kernel_float/example.html) or check out the [examples](https://github.com/KernelTuner/kernel_float/tree/master/examples) directory.


Below shows a CUDA kernel that adds a `constant` to the `input` array and writes the results to the `output` array.
Each thread processes two elements.
Notice how easy it would be change the precision (for example, `double` to `half`) or the vector size (for example, 4 instead of 2 items per thread).


```cpp
#include "kernel_float.h"
namespace kf = kernel_float;

__global__ void kernel(const kf::vec<half, 2>* input, double constant, kf::vec<float, 2>* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = input[i] + cast<half>(constant);
}

```

Here is how the same kernel would like without Kernel Float.

```cpp
__global__ void kernel(const __half* input, double constant, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __half in0 = input[2 * i + 0];
    __half in1 = input[2 * 1 + 1];
    __half2 a = __halves2half2(in0, int1);
    float b = float(constant);
    __half c = __float2half(b);
    __half2 d = __half2half2(c);
    __half2 e = __hadd2(a, d);
    __half f = __low2half(e);
    __half g = __high2half(e);
    float out0 = __half2float(f);
    float out1 = __half2float(g);
    output[2 * i + 0] = out0;
    output[2 * i + 1] = out1;
}

```

The ptx code generate by these two kernels is nearly identical.


## Installation

TODO


## Documentation

TODO


## License

Licensed under Apache 2.0. See [LICENSE](https://github.com/KernelTuner/kernel_float/blob/master/LICENSE).


## Related Work

* [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner)
* [Kernel Launcher](https://github.com/KernelTuner/kernel_launcher)

