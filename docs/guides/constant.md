Using `kernel_float::constant`
===

When working with mixed precision types, you will find that working with constants presents a bit a challenge.

For example, a simple expression such as `3.14 * x` where `x` is of type `vec<float, 2>` will NOT be performed
in `float` precision as you might expect, but instead in `double` precision.
This happens since the left-hand side of this expression
(a constant) is a `double` and thus `kernel_float` will also cast the right-hand side to `double`.

To solve this problem, `kernel_float` offers a type called `constant<T>` that can be used to represents
constants. Any binary operations between a value of type `U` and a `constant<T>` will result in both
operands being cast to type `U` and the operation is performed in the precision of type `U`. This makes
`constant<T>` useful for representing constant in your code.


For example, consider the following code:

```
#include "kernel_float.h"
namespace kf = kernel_float;

int main() {
  using Type = float;
  const int N = 8;
  static constexpr auto PI = kf::make_constant(3.14);

  kf::vec<int, N> i = kf::range<int, N>();
  kf::vec<Type, N> x = kf::cast<Type>(i) * PI;
  kf::vec<Type, N> y = x * kf::sin(x);
  Type result = kf::sum(y);
  printf("result=%f", double(result));

  return EXIT_SUCCESS;
}
```

This code example uses the ``make_constant`` utility function to create `constant<T>`.
