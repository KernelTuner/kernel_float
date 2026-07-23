# Accuracy Level

For certain operations, there might be alternative versions available that provide better performance at the cost of lower accuracy.
In other words, they are faster but also have a small error.

## Fast Math

For several operations in single precision (float32), there are "fast math" versions. These functions are faster since they are hardware accelerated, but are not IEEE compliant for all inputs.

To use this functionality, use the `fast_*` functions from Kernel Float.

```cpp
kf::vec<float, 4> x = {1.0f, 2.0f, 3.0f, 4.0f};

// Sine
kf::vec<float, 4> a = kf::fast_sin(x);

// Square root
kf::vec<float, 4> b = kf::fast_sqrt(x);

// Reciprocal `1/x`
kf::vec<float, 4> c = kf::fast_rcp(x);

// Division `a/b`
kf::vec<float, 4> d = kf::fast_div(a, b);
```

These functions are only functional for 32-bit and 16-bit floats.
For other input types, the operation falls back to the regular version.

## Approximate Math

Approximate math goes a step further than fast math: instead of relying on a hardware-accelerated instruction, it evaluates a low-degree polynomial that approximates the function. This trades even more accuracy for speed, and is only available for 16-bit floats.

To use this functionality, use the `approx_*` functions from Kernel Float. For other input types, the operation falls back to the `fast_*` variant.

```cpp
kf::vec<half, 4> x = {1.0, 2.0, 3.0, 4.0};

// Sine
kf::vec<half, 4> a = kf::approx_sin(x);

// Square root
kf::vec<half, 4> b = kf::approx_sqrt(x);

// Reciprocal `1/x`
kf::vec<half, 4> c = kf::approx_rcp(x);

// Division `a/b`
kf::vec<half, 4> d = kf::approx_div(a, b);
```

You can adjust the degree of approximation by supplying an integer template parameter:

```cpp
// Sine approximation with polynomial of degree 1
kf::vec<half, 4> a = kf::approx_sin<1>(x);

// Polynomial of degree 2
kf::vec<half, 4> a = kf::approx_sin<2>(x);

// Polynomial of degree 3
kf::vec<half, 4> a = kf::approx_sin<3>(x);
```

## Tuning Accuracy Level

Calling `fast_sin`/`approx_sin` instead of `sin` works, but it means picking a fixed accuracy level up front and hard-coding it into every call site. Many functions in Kernel Float instead accept an additional `Accuracy` option as a template parameter on the regular function name, so the accuracy level can be tuned (or overridden per call) without renaming anything.

There are five possible values for this parameter:

- `kf::accurate_policy`: Use the most accurate version of the function available.
- `kf::fast_policy`: Use the "fast math" version.
- `kf::approx_level_policy<N>`: Use the approximate version with accuracy level `N` (higher is more accurate).
- `kf::approx_policy`: Use the approximate version with a default accuracy level.
- `kf::default_policy`: Use a global default policy (see the next section).

For example, consider this code:

```cpp
kf::vec<float, 2> input = {1.0f, 2.0f};

// Use the default policy
kf::vec<float, 2> a = kf::cos(input);

// Use the default policy
kf::vec<float, 2> b = kf::cos<kf::default_policy>(input);

// Use the most accurate policy
kf::vec<float, 2> c = kf::cos<kf::accurate_policy>(input);

// Use the fastest policy
kf::vec<float, 2> d = kf::cos<kf::fast_policy>(input);

// Use the approximate policy
kf::vec<float, 2> e = kf::cos<kf::approx_policy>(input);

// Use the approximate policy with degree 3 polynomial.
kf::vec<float, 2> f = kf::cos<kf::approx_level_policy<3>>(input);

// You can use aliases to define your own policy
using my_own_policy = kf::fast_policy;
kf::vec<float, 2> g = kf::cos<my_own_policy>(input);
```

## Setting `default_policy`

If no policy is explicitly set, any function uses the `kf::default_policy`.
By default, `kf::default_policy` is set to `kf::accurate_policy`.

Set the preprocessor option `KERNEL_FLOAT_FAST_MATH=1` to change the default policy to `kf::fast_policy`.
This will use fast math for all functions and data types that support it.
