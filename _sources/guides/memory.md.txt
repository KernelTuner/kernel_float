# Memory Operations

A common problem in vector programming is that a "simple" pointer (such as `float*`) is provided, but you would like to read a vector of elements. This page describes solutions to that problem.

## Using `read`/`write`

The simplest solution is to use `read` and `write` to access `N` consecutive elements.

```cpp
std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};

float* pointer = buffer.data();

// Read 2 elements: buffer[0], buffer[1]
kf::vec<float, 2> a = kf::read<2>(pointer);

// Read 3 elements: buffer[1], buffer[2], buffer[3]
kf::vec<float, 3> b = kf::read<3>(pointer + 1);

// Write 2 elements: buffer[0], buffer[1]
kf::write(pointer, kf::vec<float, 2>(100.0f, 200.0f));

// Write 3 elements: buffer[1], buffer[2], buffer[3]
kf::write(pointer + 1, kf::vec<float, 3>(0.0f, 0.0f, 0.0f));
```

## Using `read_aligned`/`write_aligned`

For small data types, it can be highly beneficial to use **aligned** memory operations. 
These allow the compiler to read/write the elements more efficiently but require that the accessed pointer is **aligned** to a certain vector size.

```cpp
// Read 2 elements: buffer[0], buffer[1]
kf::vec<float, 2> a = kf::read_aligned<2>(pointer);

// Read 2 elements: buffer[2], buffer[3]
kf::vec<float, 2> b = kf::read_aligned<2>(pointer + 2);

// This is not allowed! `pointer+1` is not aligned to 2 elements!
// kf::vec<float, 2> b = kf::read_aligned<2>(pointer + 1);

// Write 2 elements: buffer[0], buffer[1]
kf::write_aligned<2>(pointer, kf::vec<float, 2>(100.0f, 200.0f));

// Again, this is not allowed! `pointer+1` is not aligned to 3 elements!
// kf::write_aligned<3>(pointer + 1, kf::vec<float, 3>(0.0f, 0.0f, 0.0f));
```

Note that Kernel Float does **not** check the alignment, not at compile-time and not at runtime. 
Using an unaligned address results in **undefined behavior**: either a runtime crash, miscompilation, or invalid results.

## Using `vec_ptr<T, N>`

`kf::vec_ptr<T, N>` is a data type that wraps a regular `T*` (or `const T*`) pointer and allows easy accessing of elements as a vector using aligned memory operations.

For example, given `vec_ptr<float, 4> x`, reading `x[10]` returns a vector containing `{buffer[40], buffer[41], buffer[42], buffer[43]}`.
Each index advances by `N` elements because you're working with vectors of size `N`.

The code below shows an example:

```cpp
std::vector<float> buffer = {/* some data */};
float* pointer = buffer.data();

kf::vec_ptr<float, 2> x = kf::assert_aligned<2>(pointer);

// Get the elements {buffer[20], buffer[21]}
kf::vec<float, 2> a = x[10];

// Set the elements at buffer[20] and buffer[21].
x(10) = kf::make_vec(42.0f, 42.0f);
```

In this example:

* `kf::vec_ptr<float, 2>` wraps the `float*` pointer, allowing vectorized access to the data.
* `x[10]` accesses the elements at positions `20` and `21` (`10 * 2` and `10 * 2 + 1`).
* `x(10)` provides a way to write to the data at the same positions.

### Handling Different Storage Types

A notable feature of `vec_ptr` is its ability to interact with data stored in a different underlying type than the one you operate with. 
This is achieved by specifying a third template argument `U` in `vec_ptr<T, N, U>`.

The `vec_ptr` class automatically handles the necessary type casting between `U` and `T`.
A cast from `U` to `T` is automatically inserted after each read and from `T` to `U` before each write.

For example, suppose your data is stored in half precision, but you want to read and write it using double precision. 


```cpp
std::vector<half> buffer = {/* some data */};
half* pointer = buffer.data();

// Create vec_ptr
kf::vec_ptr<half, 2> x = kf::assert_aligned<2>(pointer);

// Interact with the data in double precision
kf::vec_ptr<double, 2, half> y = x;

// Get the elements {buffer[20], buffer[21]} in double precision
kf::vec<double, 2> a = y[10];

// Set the elements at buffer[20] and buffer[21].
y(10) = kf::make_vec(42.0, 42.0);
```

In this example:

* `x` is a `vec_ptr<half, 2>` pointing to the data stored in half precision.
* `y` is a `vec_ptr<double, 2, half>` that allows you to interact with the data as double precision, even though it is stored as half.

