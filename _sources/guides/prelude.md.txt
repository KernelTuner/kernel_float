Using `kernel_float::prelude`
===

When working with Kernel Float, you'll find that you need to prefix every function and type with the `kernel_float::...` prefix. 
This can be a bit cumbersome. 
It's strongly discouraged not to dump the entire `kernel_float` namespace into the global namespace (with `using namespace kernel_float`) since
many symbols in Kernel Float may clash with global symbols, causing conflicts and issues.

To work around this, the library provides a handy `kernel_float::prelude` namespace. This namespace contains a variety of useful type and function aliases that won't conflict with global symbols.

To make use of it, use the following code:


```C++
#include "kernel_float.h"
using namespace kernel_float::prelude;

// You can now use aliases like `kf`, `kvec`, `kint`, etc.
```

The prelude defines many aliases, include the following:

| Prelude name | Full name |
|---|---|
| `kf` | `kernel_float` |
| `kvec<T, N>`  | `kernel_float::vec<T, N>`  |
| `into_kvec(v)`  | `kernel_float::into_vec(v)`  |
| `make_kvec(a, b, ...)`  | `kernel_float::make_vec(a, b, ...)`  |
| `kvec2<T>`, `kvec3<T>`, ...  | `kernel_float::vec<T, 2>`, `kernel_float::vec<T, 3>`, ...  |
| `kint<N>` | `kernel_float::vec<int, N>` |
| `kint2`, `kint3`, ...  | `kernel_float::vec<int, 2>`, `kernel_float::vec<int, 3>`, ...  |
| `klong<N>` | `kernel_float::vec<long, N>` |
| `klong2`, `klong3`, ...  | `kernel_float::vec<long, 2>`, `kernel_float::vec<long, 3>`, ...  |
| `kbfloat16x<N>` | `kernel_float::vec<bfloat16, N>` |
| `kbfloat16x2`, `kbfloat16x3`, ...  | `kernel_float::vec<bfloat16, 2>`, `kernel_float::vec<bfloat16, 3>`, ...  |
| `khalf<N>` | `kernel_half::vec<half, N>` |
| `khalf2`, `khalf3`, ...  | `kernel_half::vec<half, 2>`, `kernel_half::vec<half, 3>`, ...  |
| `kfloat<N>` | `kernel_float::vec<float, N>` |
| `kfloat2`, `kfloat3`, ...  | `kernel_float::vec<float, 2>`, `kernel_float::vec<float, 3>`, ...  |
| `kdouble<N>` | `kernel_float::vec<double, N>` |
| `kdouble2`, `kdouble3`, ...  | `kernel_float::vec<double, 2>`, `kernel_float::vec<double, 3>`, ...  |
| ... | ... |
