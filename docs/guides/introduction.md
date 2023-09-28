Getting started
===============

Kernel Float is a header-only library that makes it easy to work with vector types and low-precision floating-point types, mainly focusing on CUDA kernel code.

Installation
------------

The easiest way to use the library is get the single header file from github:

```bash
wget https://raw.githubusercontent.com/KernelTuner/kernel_float/main/single_include/kernel_float.h
```

Next, include this file into your program.
It is conventient to define a namespace alias `kf` to shorten the full name `kernel_float`.


```C++
#include "kernel_float.h"
namespace kf = kernel_float;
```


Example C++ code
----------------

Kernel Float essentially offers a single data-type `kernel_float::vec<T, N>` that stores `N` elements of type `T`.
This type can be initialized normally using list-initialization (e.g., `{a, b, c}`) and elements can be accessed using the `[]` operator.
Operation overload is available to perform binary operations (such as `+`, `*`, and `&`), where the optimal intrinsic for the available types is selected automatically.

Many mathetical functions (like `log`, `sin`, `cos`) are also available, see the [API reference](../api) for the full list of functions.
In some cases, certain operations might not be natively supported by the platform for the some floating-point type.
In these cases, Kernel Float falls back to performing the operations in 32 bit precision.

The code below shows a very simple example of how to use Kernel Float:

```C++
#include "kernel_float.h"
namespace kf = kernel_float;

int main() {
  using Type = float;
  const int N = 8;

  kf::vec<int, N> i = kf::range<int, N>();
  kf::vec<Type, N> x = kf::cast<Type>(i);
  kf::vec<Type, N> y = x * kf::sin(x);
  Type result = kf::sum(y);
  printf("result=%f", double(result));

  return EXIT_SUCCESS;
}
```

Notice how easy it would be to change the floating-point type `Type` or the vector length `N` without affecting the rest of the code.
