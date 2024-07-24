Accuracy level
===

Many of the functions in Kernel Float take an additional `Accuracy` option as a template parameter.
This option can be used to increase the performance of certain operations, at the cost of lower accuracy.

There are four possible values for this parameter:

* `accurate_policy`: Use the most accurate version of the function available.
* `fast_policy`: Use the "fast math" version (for example, `__sinf` for sin on CUDA devices). Falls back to `accurate_policy` if such a version is not available.
* `approx_policy<N>`: Rough approximation using a polynomial of degree `N`. Falls back to `fast_policy` if no such polynomial exists.
* `default_policy`: Use a global default policy (see the next section).


For example, consider this code:

```C++

#include "kernel_float.h"
namespace kf = kernel_float;


int main() {
  kf::vec<float, 2> input = {1.0f, 2.0f};

  // Use the default policy
  kf::vec<float, 2> A = kf::cos(input);

  // Use the most accuracy policy
  kf::vec<float, 2> B = kf::cos<kf::accurate_policy>(input);

  // Use the fastest policy
  kf::vec<float, 2> C = kf::cos<kf::fast_policy>(input);

  printf("A = %f, %f", A[0], A[1]);
  printf("B = %f, %f", B[0], B[1]);
  printf("C = %f, %f", C[0], C[1]);

  return EXIT_SUCCESS;
}

```


Setting `default_policy`
---
By default, the value for `default_policy` is `accurate_policy`.

Set the preprocessor option `KERNEL_FLOAT_FAST_MATH=1` to change the default policy to `fast_policy`.
