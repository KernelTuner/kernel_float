# Getting started

**Kernel Float** is a header-only library that makes it easy to work with vector types and low-precision floating-point types, mainly focusing on CUDA kernel code.

## Installation

The easiest way to use the library is to get the single header file from GitHub:

```bash
wget https://raw.githubusercontent.com/KernelTuner/kernel_float/main/single_include/kernel_float.h
```

Next, include this file in your program. It is convenient to define a namespace alias `kf` to shorten the full name `kernel_float`:

```cpp
#include "kernel_float.h"
namespace kf = kernel_float;
```

## Vector types

Kernel Float essentially offers a single data type `kernel_float::vec<T, N>` that stores `N` elements of type `T`. The simplest way to initialize a vector is using list-initialization:

```cpp
kf::vec<float, 4> my_vector = {1.0f, 2.0f, 3.0f, 4.0f};
```

It is also possible to automatically derive the type using `make_vec`:

```cpp
// The type will be vec<double, 3>
auto a = kf::make_vec(1.0, 2.0, 3.0);

// The type will be vec<int, 2>
auto b = kf::make_vec(7, 7);

// The type will be vec<bool, 4>
auto c = kf::make_vec(true, true, false, true);

// This does not compile!
auto d = kf::make_vec();
```

There are also many helper methods available to generate vectors; see the [API reference](../api). Some examples are `range`, `fill`, `ones`, and `zeros`.

```cpp
// Generates [0, 1, 2, 3]
kf::vec<int, 4> a = kf::range<int, 4>();

// Generates [42.0, 42.0, 42.0, 42.0]
kf::vec<double, 4> b = kf::fill<4>(42.0);

// Generates [0, 0, 0, 0]
kf::vec<int, 4> c = kf::zeros<int, 4>();

// Generates [true, true, true, true]
kf::vec<bool, 4> d = kf::ones<bool, 4>();
```

You can also use the `*_like` functions to generate a vector based on another vector:

```cpp
// Generates [1.0, 2.0, 3.0, 4.0]
kf::vec<float, 4> a = {1.0f, 2.0f, 3.0f, 4.0f};

// Generates [0.0, 0.0, 0.0, 0.0]
kf::vec<float, 4> b = kf::zeros_like(a);

// Generates [1.0, 1.0, 1.0, 1.0]
kf::vec<float, 4> c = kf::ones_like(a);
```

## Accessing elements

Accessing elements can be done using the regular `[]` operator.

```cpp
// Generate [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
kf::vec<float, 6> a = kf::range<float, 6>();

// Returns 2.0
float x = a[2];

// Set element at 2 to 42.0
a[2] = 42.0;

// Returns 42.0
float y = a[2];
```

You can get a pointer to the vector buffer by calling `data`:

```cpp
// Generate vector
kf::vec<float, 4> v = {1.0f, 2.0f, 3.0f, 4.0f};

float* address = v.data();

// Set element at 0 to element at 1
address[0] = address[1];
```

Iteration can be done by using a regular for-loop:

```cpp
kf::vec<float, 4> vector = {1.0f, 2.0f, 3.0f, 4.0f};

for (float x : vector) {
  printf("x=%f\n", x);
}
```

## Operator overloading

The **arithmetic** operators `+`, `-`, `*`, `/`, and `%` are overloaded to perform element-wise operations.

```cpp
// Generate [1.0f, 2.0f, 3.0f]
kf::vec<float, 3> a = {1.0f, 2.0f, 3.0f};

// Generate [1.0f, 1.0f, 1.0f]
kf::vec<float, 3> b = kf::ones<float, 3>();

// Add them together to create [2.0f, 3.0f, 4.0f]
kf::vec<float, 3> c = a + b;
```

The **comparison** operators `<`, `>`, `==`, `!=`, `<=`, `>=` are overloaded to perform element-wise operations. Note that the returned value is a vector containing 0s (`false`) and 1s (`true`). The element type and vector length will match the inputs.

```cpp
// Generate doubles
kf::vec<double, 5> a = {4.0, -100.0, 0.0, 0.5, -3.0};

// Generate zeros
kf::vec<double, 5> zeros = kf::zeros_like(a);

// Generates [false, true, false, false, true]
kf::vec<bool, 5> result = a < zeros;
```

The **logical** operators `&&` and `||` are NOT overloaded. This is because there is no method to simulate the short-circuiting behavior. Instead, the operators `!` (not), `&` (and), `|` (or), and `^` (xor) are overloaded to behave as logical operators.

```cpp
// Generate doubles
kf::vec<double, 5> a = {4.0, -100.0, 0.0, 0.5, -3.0};

// Generate zeros and ones
kf::vec<double, 5> zeros = kf::zeros_like(a);
kf::vec<double, 5> ones = kf::ones_like(a);

// Generates [false, false, true, true, false]
kf::vec<bool, 5> result = (a >= zeros) & (a <= ones);

// Using `&&` instead of `&` results in a compilation error!
// kf::vec<bool, 5> fail = (a >= zeros) && (a <= ones);
```

If the two inputs of a binary operator do not match (either element type and/or vector length), Kernel Float will automatically perform **type promotion** (described on the page [Type Promotion](promotion)). This allows our example to be simplified to just:

```cpp
// Generate doubles
kf::vec<double, 5> a = {4.0, -100.0, 0.0, 0.5, -3.0};

// Generates [false, false, true, true, false]
kf::vec<bool, 5> result = (a >= 0.0) & (a <= 1.0);
```

## Mathematical functions

Many mathematical functions (like `log`, `sin`, `cos`) are also available; see the [API reference](../api) for the full list of functions. These always work element-wise:

```cpp
// Input vector
kf::vec<float, 4> x = {0.0f, 1.0f, 2.0f, 3.0f};

// Gives [0.0, 0.84147098, 0.9092974, 0.14112001]
kf::vec<float, 4> a = kf::sin(x);

// Gives [1.0, 0.54030231, -0.41614684, -0.9899925]
kf::vec<float, 4> b = kf::cos(x);

// Gives [0.0, 1.0, 1.4142135, 1.7320508]
kf::vec<float, 4> c = kf::sqrt(x);

// Gives [1.0, 2.7182818, 7.3890561, 20.085537]
kf::vec<float, 4> d = kf::exp(x);

// Gives [0, 0, 0, 0]
kf::vec<bool, 4> e = kf::isnan(x);
```

In some cases, certain operations might not be natively supported by the platform for some floating-point types. In these cases, Kernel Float falls back to performing the operations in 32-bit precision.
