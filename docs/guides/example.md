# Full CUDA example

This page explains a CUDA program that estimates the value of pi using Kernel Float.


## Overview

The program calculates Pi by generating random points within a unit square and counting how many fall inside the unit circle inscribed within that square. The ratio of points inside the circle to the total number of points approximates Pi/4.

The kernel is shown below:


```c++
namespace kf = kernel_float;

using float_type = float;
static constexpr int VECTOR_SIZE = 4;

__global__ void calculate_pi_kernel(int nx, int ny, int* global_count) {
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

    kf::vec<int, VECTOR_SIZE> xi = thread_x * VECTOR_SIZE + kf::range<int, VECTOR_SIZE>();
    kf::vec<int, VECTOR_SIZE> yi = thread_y;

    kf::vec<float_type, VECTOR_SIZE> xf = kf::cast<float_type>(xi) / float_type(nx);
    kf::vec<float_type, VECTOR_SIZE> yf = kf::cast<float_type>(yi) / float_type(ny);

    kf::vec<float_type, VECTOR_SIZE> dist_squared = xf * xf + yf * yf;
    kf::vec<float_type, VECTOR_SIZE> dist = kf::sqrt(dist_squared);

    int n = kf::count(dist <= float_type(1));

    if (n > 0) atomicAdd(global_count, n);
}
```


## Code Explanation

Let's go through the code step by step.

```cpp
// Alias `kernel_float` as `kf`
namespace kf = kernel_float;
```

This creates an alias for `kernel_float`.

```cpp
// Define the float type and vector size
using float_type = float;
static constexpr int VECTOR_SIZE = 4;
```

Define `float_type` as an alias for `float` to make it easy to change precision if needed.
The vector size is set to 4, meaning each thread will process 4 data points.

```cpp
__global__ void calculate_pi_kernel(int nx, int ny, int* global_count) {
```

The CUDA kernel. There are `nx` points along the x axis and `ny` points along the y axis.

```cpp
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
```

Compute the global x- and y-index of this thread.

```cpp
    kf::vec<int, VECTOR_SIZE> xi = thread_x * VECTOR_SIZE + kf::range<int, VECTOR_SIZE>();
    kf::vec<int, VECTOR_SIZE> yi = thread_y;
```

Compute the points that this thread will process.
The x coordinates start at `thread_x * VECTOR_SIZE` and then the vector `[0, 1, 2, ..., VECTOR_SIZE-1]`.
The y coordinates are all `thread_y`.

```cpp
    kf::vec<float_type, VECTOR_SIZE> xf = kf::cast<float_type>(xi) / float_type(nx);
    kf::vec<float_type, VECTOR_SIZE> yf = kf::cast<float_type>(yi) / float_type(ny);
```

Divide `xi` and `yi` by `nx` and `ny` to normalize them to `[0, 1]` range.

```cpp
    kf::vec<float_type, VECTOR_SIZE> dist_squared = xf * xf + yf * yf;
```

Compute the squared distance from the origin (0, 0) to each point from `xf`,`yf`.

```cpp
    kf::vec<float_type, VECTOR_SIZE> dist = kf::sqrt(dist_squared);
```

Take the element-wise square root.

```cpp
    int n = kf::count(dist <= float_type(1));
```

Count the number of points in the unit circle (i.e., for which the distance is less than 4).
The expression `dist <= 1` returns a vector of booleans and `kf::count` counts the number of `true` values.

```cpp
atomicAdd(global_count, n);
```

Add `n` to the `global_count` variable.
This must be done using an atomic operation since multiple thread will write this variable simultaneously.
