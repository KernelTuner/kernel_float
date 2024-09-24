# Casting and Type Promotion

This section is on data type casting and implicit type promotion.

## Type Casting

You can use the function `cast` to explicitly change the element type of a vector:

```cpp
// a = [0, 1, 2, 3]
kf::vec<int, 4> a = kf::range<int, 4>();

// b = [0, 1, 2, 3]
kf::vec<short, 4> b = kf::cast<short>(a);

// c = [0.0f, 1.0f, 2.0f, 3.0f]
kf::vec<float, 4> c = kf::cast<float>(a);

// d = [false, true, true, true]
kf::vec<bool, 4> d = kf::cast<bool>(a);
```

Kernel Float allows **implicit** type conversions between vector types **only** for **widening** casts (i.e., converting a smaller type into a larger type).

```cpp
// a = [0, 1, 2, 3]
kf::vec<int, 4> a = kf::range<int, 4>();

// This is allowed: int -> long
kf::vec<long, 4> b = a;

// This is also allowed: int -> float
kf::vec<float, 4> c = a;
```

Kernel Float does **not** perform implicit conversion for narrowing casts; this requires an explicit cast:

```cpp
// a = [0, 1, 2, 3]
kf::vec<double, 4> a = kf::range<double, 4>();

// This is NOT allowed, use `kf::cast<float>(a)`
kf::vec<float, 4> b = a; // ERROR

// This is NOT allowed, use `kf::cast<int>(a)`
kf::vec<int, 4> c = a; // ERROR

// This is NOT allowed, use `kf::cast<bool>(a)`
kf::vec<bool, 4> d = a; // ERROR
```

Alternatively, it is possible to use `kf::cast_to` on the left side of the assignment to perform a cast:

```cpp
// a = [0, 1, 2, 3]
kf::vec<double, 4> a = kf::range<double, 4>();

// Define b
kf::vec<float, 4> b;

// This is equivalent to `b = kf::cast<float>(a)`
kf::cast_to(b) = a;
```

## Type Promotion

When performing operations between vectors of different types or sizes, Kernel Float automatically promotes types to ensure compatibility. This process is known as **type promotion**.



Consider the following example. What should the type of `c` be in this case?

```cpp
kf::vec<float, 4> a = {1.0f, 2.0f, 3.0f, 4.0f};
kf::vec<double, 4> b = {10.0, 20.0, 30.0, 40.0};

kf::vec<???, 4> c = a + b;
```

Another example is the following snippet. Again, what should the type of `c` be?

```cpp
kf::vec<float, 4> x = {1.0f, 2.0f, 3.0f, 4.0f};
int factor = 2;

kf::vec<???, ???> c = x * factor;
```

### How Type Promotion Works
Type promotion in Kernel Float unifies arguments for binary (and ternary) operations through the following steps:

* **Vectorization**: Each non-vector argument is converted into a vector using the `into_vec` function.

* **Length Unification**: All arguments must have the same length N or length 1. Vectors of length 1 are broadcasted to match length N.

* **Type Unification**: The element types are promoted to a common type based on the following promotion rules.

### Promotion Rules
The rules for element type promotion in Kernel Float are slightly different from standard C++. Here's a summary:

* **Boolean Types**: If one of the types is bool, the result type is the other type.

* **Floating-Point and Integer**: If one type is floating-point and the other is an integer (signed or unsigned), the result is the floating-point type.

* **Floating-Point Types**: If both are floating-point types, the larger (wider) type is chosen.
Exception: Combining half and bfloat16 results in float.

* **Integer Types**: If both are integers of the same signedness (both signed or both unsigned), the larger type is chosen.
Combining a signed integer and an unsigned integer is not allowed.

### Overview

The following table summarizes the type promotion rules. The labels used are:

- `b`: boolean
- `iN`: signed integer of `N` bits (e.g., `int`, `long`)
- `uN`: unsigned integer of `N` bits (e.g., `unsigned int`, `size_t`)
- `fN`: floating-point type of `N` bits (e.g., `float`, `double`)
- `bf16`: bfloat16 floating-point format.

```{csv-table} Type Promotion Rules.
:file: promotion_table.csv
```

.. csv-table:: Type Promotion Rules.
   :file: promotion_table.csv

